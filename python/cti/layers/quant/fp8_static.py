"""Static per-tensor FP8, ModelOpt's FP8 format: an E4M3 weight with one FP32 scale per
checkpoint module and a calibrated FP32 input scale (``numerics.py``)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import torch
import torch.nn.functional as F

from cti.utils import init_logger
from ctikernels.linear import gated_input

from ..linear_method import GemmRoutes, LinearMethod
from ..placement import WeightShard
from . import numerics
from ._utils import provider_row

if TYPE_CHECKING:
    from ..linear_method import GemmCallSite, LinearHolder

logger = init_logger(__name__)


class _Provider(Protocol):
    """One implementation of ``x @ W_hat.T`` for a loaded weight."""

    def warm_up(self, max_rows: int, *, out_row_stride: int) -> GemmRoutes:
        """Compile every call of up to ``max_rows`` rows, into outputs of
        ``out_row_stride`` too."""
        ...

    def workspace_bytes(self) -> int:
        """The scratch the largest warmed call needs from the model's shared workspace."""
        ...

    def __call__(
        self,
        x: torch.Tensor,
        out: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class QuantizedMethod(LinearMethod):
    """A frozen quantized weight served by its device type's row of ``providers``, built by
    ``bind``; the method is its own call site. It serves, and serves LoRA adapters, but does
    not train: its checkpoints' recipe refuses training before the model is built."""

    providers: ClassVar[Mapping[str, Callable[..., _Provider]]]

    def __init__(self, layer: LinearHolder) -> None:
        super().__init__(layer)
        self.services = layer.services
        # The dtype the layer computes in: the one in effect when it was built.
        self._dtype = torch.get_default_dtype()
        self._provider: _Provider | None = None

    def operands(self) -> tuple[Any, ...]:
        """What the providers take, after ``bind``'s conversions."""
        raise NotImplementedError

    def bind(self) -> None:
        device = self._layer.weight.device
        operands = self.operands()
        self._provider = self.providers[device.type](*operands, dtype=self._dtype)

    def call_sites(self) -> Iterator[GemmCallSite]:
        yield self

    @classmethod
    def computed_by(cls, device: torch.device) -> str:
        return provider_row(cls.providers, device)

    def warm_up(
        self, max_rows: int, device: torch.device, *, renew: Callable[[str], None]
    ) -> GemmRoutes:
        # The kernel warms every row bucket in one call: nothing to renew between them.
        if self._layer.weight.device.type != device.type:
            raise ValueError(f"a weight on {self._layer.weight.device} warms up on {device}")
        if self._provider is None:
            self.bind()
        return _bound(self._provider).warm_up(
            max_rows, out_row_stride=self._layer.output_row_stride
        )

    def workspace_bytes(self, device: torch.device) -> int:
        return _bound(self._provider).workspace_bytes()

    def apply(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gate is not None:
            x = gated_input(x, gate)
        y = _bound(self._provider)(x, out, self.services.gemm_workspace)
        return y if bias is None else y.add_(bias)


def _bound(provider: _Provider | None) -> Any:
    if provider is None:
        raise RuntimeError("a quantized method binds its weight before its use")
    return provider


def merged_input_scale(input_scales: torch.Tensor) -> torch.Tensor:
    """The one input scale (FP32 ``[1]``) every merge slot's activation quantizes by: the
    largest of theirs, logged where they differ (ModelOpt calibrates siblings alike)."""

    input_scale = input_scales.max().reshape(1).float()
    if (input_scales != input_scale).any():
        logger.info_rank0(
            "Merged input scales differ (%s); quantizing by the largest", input_scales.tolist()
        )
    return input_scale


class Fp8StaticKernel:
    """cti-kernels' ``StaticFP8Linear``: the activation quantized by the input scale, one GEMM
    accumulating in FP32 and each merge slot's rows taking its own weight scale before one
    rounding (the kernel's contract)."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        slot_rows: tuple[int, ...],
        *,
        dtype: torch.dtype,
    ) -> None:
        from ctikernels.linear.fp8_static import StaticFP8Linear

        self._operation = StaticFP8Linear(weight, weight_scale, input_scale, slot_rows=slot_rows)
        self._dtype = dtype
        self._max_rows = 0

    def warm_up(self, max_rows: int, *, out_row_stride: int) -> GemmRoutes:
        operation = self._operation
        strided = out_row_stride if out_row_stride != operation.out_features else None
        operation.warm_up(max_rows, dtype=self._dtype, out_row_stride=strided)
        self._max_rows = max_rows
        return Counter({((1, max_rows), f"fp8-static {operation.provider}"): 1})

    def workspace_bytes(self) -> int:
        return self._operation.workspace_bytes(self._max_rows) if self._max_rows else 0

    def __call__(
        self,
        x: torch.Tensor,
        out: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The product, its scratch from ``workspace`` (the model's shared one) as the GEMMs'
        is, so that no captured graph takes a workspace of its own."""

        y = self._operation(
            x.reshape(-1, x.shape[-1]), out_dtype=self._dtype, out=out, workspace=workspace
        )
        return y if out is not None else y.view(*x.shape[:-1], -1)


# Each device type's provider of the static-FP8 product.
class Fp8StaticReference:
    """The contract computed by torch, for the CPU and as the tests' oracle: the activation
    quantized and dequantized, the weight dequantized by its slot's scale, both in the
    activation dtype, multiplied with FP32 accumulation."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        slot_rows: tuple[int, ...],
        *,
        dtype: torch.dtype,
    ) -> None:
        self._input_scale = input_scale
        alpha = weight_scale.repeat_interleave(torch.tensor(slot_rows, device=weight.device))
        self._weight = numerics.dequantize_static(weight, alpha.float().unsqueeze(-1))

    def warm_up(self, max_rows: int, *, out_row_stride: int) -> GemmRoutes:
        return Counter()

    def workspace_bytes(self) -> int:
        return 0

    def __call__(
        self,
        x: torch.Tensor,
        out: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = numerics.quantize_static_activation(x, self._input_scale)
        x_hat = numerics.dequantize_static(q, self._input_scale).to(x.dtype)
        y = F.linear(x_hat, self._weight.to(x.dtype))
        return y if out is None else out.copy_(y)


PROVIDERS: Mapping[str, type[_Provider]] = MappingProxyType(
    {"cpu": Fp8StaticReference, "cuda": Fp8StaticKernel}
)


class Fp8StaticLinear(QuantizedMethod):
    """A ModelOpt FP8 weight: the E4M3 rows of every merged module, with each module's weight
    and input scale by merge slot. Every slot's activation quantizes by the largest input
    scale; each slot's output columns take its own weight scale."""

    providers = PROVIDERS

    def declare(self) -> dict[str, torch.Tensor]:
        layer = self._layer
        slots = len(layer.output_slices)
        return {
            "weight": torch.empty(
                layer.local_output_size, layer.local_input_size, dtype=numerics.E4M3
            ),
            "weight_scale": torch.empty(slots, dtype=torch.float32),
            "input_scale": torch.empty(slots, dtype=torch.float32),
        }

    def windows(self, leaf: str, windows: tuple[WeightShard, ...]) -> tuple[WeightShard, ...]:
        return windows if leaf == "weight" else ()

    def bind(self) -> None:
        """Merge the input scales into one ``[1]``; the weight scales stay one per merge slot."""

        self._layer.input_scale = merged_input_scale(self._layer.input_scale)
        super().bind()

    def operands(self) -> tuple[Any, ...]:
        """The E4M3 weight, its FP32 scale per merge slot, the input scale and each slot's
        rows."""

        layer = self._layer
        slot_rows = tuple(piece.size for piece in layer.output_slices)
        return layer.weight, layer.weight_scale, layer.input_scale, slot_rows
