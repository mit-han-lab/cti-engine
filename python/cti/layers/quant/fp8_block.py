"""Block-scaled FP8, the format of Qwen's official FP8 checkpoints.

Weights are E4M3 with one scale per 128x128 block (``weight_scale_inv``, a multiplier);
activations quantize at run time per token and 128 columns (``numerics.py``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

import torch
import torch.nn.functional as F

from cti.utils import init_logger
from ctikernels.linear import gated_input

from ..experts_method import ExpertsMethod
from ..linear_method import (
    GemmCallSite,
    GemmRoutes,
    LinearHolder,
    LinearMethod,
)
from ..moe import Experts
from ..placement import WeightShard
from . import numerics
from ._utils import block_windows, provider_row

if TYPE_CHECKING:
    from ctikernels.linear.training import FrozenProduct

    from . import CheckpointIndex, CheckpointPart, Documents

# The values this format fixes, and why another value is refused. Absent keys take them,
# except the two the claim requires.
_FIXED = MappingProxyType(
    {
        "fmt": ("e4m3", "weights are E4M3"),
        "weight_block_size": ([128, 128], "scales cover 128x128 weight blocks"),
        "activation_scheme": (
            "dynamic",
            "activations quantize dynamically, per token and 128 columns",
        ),
        "weight_per_tensor": (False, "weights carry 128x128 block scales"),
        "act_per_tensor": (
            False,
            "per-tensor dynamic activation scales are refused; activations quantize per "
            "token and 128 columns",
        ),
    }
)
_REQUIRED = ("weight_block_size", "activation_scheme")
_KEYS = frozenset({"quant_method", "modules_to_not_convert", *_FIXED})
_WEIGHT_DTYPE = "F8_E4M3"
_SCALE_DTYPES = frozenset({"BF16", "F32"})
# The dequantized weight rows the reference's shared scratch holds at once, in bytes; a larger
# weight is multiplied in fixed row chunks.
_SCRATCH_BYTES = 256 * 2**20
# The FP32 weight rows one dequantization step holds, in bytes, which bounds its temporaries.
_DEQUANTIZE_BYTES = 32 * 2**20
logger = init_logger(__name__)


@dataclass(frozen=True)
class Fp8BlockRecipe:
    """HF ``quant_method=fp8`` with 128x128 weight blocks and dynamic activations."""

    NAME: ClassVar[str] = "fp8_block"
    TRAINS: ClassVar[bool] = True
    # ``modules_to_not_convert``: cross-checked against the headers, never trusted alone.
    unconverted: tuple[str, ...]

    @classmethod
    def parse(cls, documents: Documents) -> Self | None:
        declared = documents["config.json"].get("quantization_config")
        if not isinstance(declared, Mapping) or declared.get("quant_method") != "fp8":
            return None
        if unknown := sorted(set(declared) - _KEYS):
            raise ValueError(f"fp8 quantization_config has unknown keys {unknown}")
        if missing := [key for key in _REQUIRED if key not in declared]:
            raise ValueError(f"fp8 quantization_config lacks {missing}")
        for key, (value, reason) in _FIXED.items():
            if key in declared and declared[key] != value:
                raise ValueError(f"fp8 quantization_config {key}={declared[key]!r}: {reason}")
        unconverted = declared.get("modules_to_not_convert", [])
        if not isinstance(unconverted, list) or not all(isinstance(m, str) for m in unconverted):
            raise ValueError("fp8 modules_to_not_convert must be a list of module names")
        return cls(tuple(unconverted))

    def verify(self, index: CheckpointIndex) -> None:
        """Refuse an FP8 weight inside a module listed as unconverted; record stale entries."""

        listed = set(self.unconverted)
        for name, header in index.tensors.items():
            if header.dtype == _WEIGHT_DTYPE and (owners := listed & _prefixes(name)):
                raise ValueError(
                    f"{name} is {header.dtype}, but modules_to_not_convert lists {min(owners)!r}"
                )
        if stale := self.stale(index):
            logger.info_rank0(
                "%d modules_to_not_convert entries name no checkpoint tensor, e.g. %r",
                len(stale),
                stale[0],
            )

    def method_for(self, part: CheckpointPart) -> type[LinearMethod]:
        """``Fp8Block128Linear`` for an E4M3 weight with its 128x128 scale grid."""

        self._claim(part, ("weight", "weight_scale_inv", "bias"))
        return Fp8Block128Linear

    def experts_method_for(self, part: CheckpointPart) -> type[ExpertsMethod]:
        """``Fp8Block128Experts`` for an expert's E4M3 weight with its 128x128 scale grid."""

        self._claim(part, ("weight", "weight_scale_inv"))
        return Fp8Block128Experts

    def _claim(self, part: CheckpointPart, leaves: tuple[str, ...]) -> None:
        """Refuse a module unless it holds an E4M3 weight with its scale grid, and ``leaves``
        alone, and is not listed as unconverted."""

        weight, scale = part.tensors.get("weight"), part.tensors.get("weight_scale_inv")
        found = {leaf: f"{h.dtype}{list(h.shape)}" for leaf, h in part.tensors.items()}
        extra = set(part.tensors) - set(leaves)
        if extra or weight is None or weight.dtype != _WEIGHT_DTYPE or scale is None:
            raise ValueError(f"{part.module}: fp8_block claims no {found}")
        grid = tuple(-(-extent // numerics.BLOCK) for extent in weight.shape)
        if scale.shape != grid or scale.dtype not in _SCALE_DTYPES:
            raise ValueError(f"{part.module}: fp8_block expects a {list(grid)} scale grid: {found}")
        if listed := set(self.unconverted) & _prefixes(part.module):
            raise ValueError(
                f"{part.module} is FP8, but modules_to_not_convert lists {min(listed)!r}"
            )

    def stale(self, index: CheckpointIndex) -> tuple[str, ...]:
        """``modules_to_not_convert`` entries that name no checkpoint tensor; recorded only."""

        named = {prefix for name in index.tensors for prefix in _prefixes(name)}
        return tuple(module for module in self.unconverted if module not in named)


def _prefixes(name: str) -> set[str]:
    """``name`` and each dotted prefix of it: the modules that contain it."""

    parts = name.split(".")
    return {".".join(parts[:end]) for end in range(1, len(parts) + 1)}


class Fp8Block128Linear(LinearMethod):
    """An Fp8Block128 weight; its product runs on the provider ``PROVIDERS`` names for the
    device type."""

    # The leaf of the weight's FP32 scale grid, as its checkpoint names it.
    SCALE: ClassVar[str] = "weight_scale_inv"

    def __init__(self, layer: LinearHolder) -> None:
        super().__init__(layer)
        self._gemm = _Fp8BlockGemm(layer, self.SCALE)

    @classmethod
    def computed_by(cls, device: torch.device) -> str:
        return provider_row(PROVIDERS, device)

    def declare(self) -> dict[str, torch.Tensor]:
        rows, cols = self._layer.local_output_size, self._layer.local_input_size
        grid = (-(-rows // numerics.BLOCK), -(-cols // numerics.BLOCK))
        return {
            "weight": torch.empty(rows, cols, dtype=numerics.E4M3),
            self.SCALE: torch.empty(grid, dtype=torch.float32),
        }

    def windows(self, leaf: str, windows: tuple[WeightShard, ...]) -> tuple[WeightShard, ...]:
        if leaf == "weight":
            return windows
        return block_windows(windows, {0: numerics.BLOCK, 1: numerics.BLOCK})

    def call_sites(self) -> Iterator[GemmCallSite]:
        yield self._gemm

    def apply(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gate is not None:
            x = gated_input(x, gate)
        y = self._gemm(x, out)
        return y if bias is None else y.add_(bias)

    def training(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = _FrozenTraining.apply(x, self.frozen_product())
        return y if bias is None else y + bias

    def frozen_product(self) -> FrozenProduct:
        """The frozen weight as training multiplies by it: what serving computes forward,
        ``dY @ W_hat`` back with the activation's quantizer passed straight through."""

        return self._gemm.frozen_product()


class _Provider(Protocol):
    """One implementation of the product ``x @ W_hat.T``, bound to a loaded weight."""

    def warm_up(self, max_rows: int, *, out_row_stride: int) -> GemmRoutes:
        """Compile every call of up to ``max_rows`` rows, into outputs of
        ``out_row_stride`` too."""
        ...

    def workspace_bytes(self) -> int: ...

    def __call__(
        self, x: torch.Tensor, workspace: torch.Tensor | None, out: torch.Tensor | None = None
    ) -> torch.Tensor: ...

    def frozen_product(self) -> FrozenProduct: ...


class _Fp8BlockGemm:
    """One layer's ``x @ W_hat.T``. Its warm-up binds the loaded weight to the provider of the
    device type, which every later call runs in its model's ``services.gemm_workspace``."""

    def __init__(self, layer: LinearHolder, scale: str) -> None:
        self._layer, self._scale = layer, scale
        self.services = layer.services
        # The dtype the layer computes in: the one in effect when it was built.
        self._dtype = torch.get_default_dtype()
        self._provider: _Provider | None = None

    def warm_up(
        self, max_rows: int, device: torch.device, *, renew: Callable[[str], None]
    ) -> GemmRoutes:
        # The kernel warms every row bucket in one call: nothing to renew between them.
        weight = self._layer.weight
        if weight.device.type != device.type:
            raise ValueError(f"a block-FP8 weight on {weight.device} warms up on {device}")
        return self._bind().warm_up(max_rows, out_row_stride=self._layer.output_row_stride)

    def frozen_product(self) -> FrozenProduct:
        """The bound provider's frozen product; training without a warm-up binds it."""

        return (self._provider or self._bind()).frozen_product()

    def _bind(self) -> _Provider:
        layer = self._layer
        self._provider = PROVIDERS[layer.weight.device.type](
            # A grid stored narrower than FP32 (UE8M0's powers of two) widens exactly; an FP32
            # grid is used as stored, so an in-place reload of it is seen.
            layer.weight,
            getattr(layer, self._scale).float(),
            dtype=self._dtype,
        )
        return self._provider

    def workspace_bytes(self, device: torch.device) -> int:
        return self._bound().workspace_bytes()

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return self._bound()(x, self.services.gemm_workspace, out)

    def _bound(self) -> _Provider:
        if self._provider is None:
            raise RuntimeError(
                "prepare_inference binds a block-FP8 weight's provider before its use"
            )
        return self._provider


class Fp8BlockKernel:
    """cti-kernels' ``BlockFP8Linear``: the activation quantizes per token and 128 columns in
    the kernel and the product accumulates in FP32; it picks its implementation per device,
    weight shape and row count. Its scratch and staging come from the model's shared GEMM
    workspace, sized for every row count it warmed, rather than held by each layer."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype) -> None:
        from ctikernels.linear.block_fp8 import BlockFP8Linear

        self._operation = BlockFP8Linear(weight, scale)
        self._dtype = dtype
        self._max_rows = 0

    def warm_up(self, max_rows: int, *, out_row_stride: int) -> GemmRoutes:
        """Warm the operation for every batch of up to ``max_rows`` packed rows, the only
        layout ``__call__`` hands it (graph capture accepts only what it warmed), into
        outputs of ``out_row_stride`` too; return its provider runs."""

        strided = out_row_stride if out_row_stride != self._operation.out_features else None
        self._operation.warm_up(
            max_rows, dtype=self._dtype, input_layouts="packed", out_row_stride=strided
        )
        self._max_rows = max_rows
        _, runs = warm_rows(self._operation.provider_for, max_rows)
        return Counter({((first, last), provider): 1 for first, last, provider in runs})

    def workspace_bytes(self) -> int:
        return self._operation.workspace_bytes(self._max_rows)

    def __call__(
        self, x: torch.Tensor, workspace: torch.Tensor | None, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = packed(x)
        if workspace is None and out is None:
            return self._operation(x)
        # With the caller's workspace and output, the operation stages nothing of its own.
        rows = x.reshape(-1, x.shape[-1])
        given = out is not None
        if out is None:
            out = rows.new_empty(rows.shape[0], self._operation.out_features)
        self._operation(rows, out=out, workspace=workspace)
        return out if given else out.view(*x.shape[:-1], -1)

    def frozen_product(self) -> FrozenProduct:
        from ctikernels.linear.training import BlockFP8Product

        return BlockFP8Product(self._operation)


def packed(x: torch.Tensor) -> torch.Tensor:
    """``x`` as contiguous rows on a 16-byte-aligned base: itself, or a copy of a strided or
    offset view."""

    if x.is_contiguous() and x.data_ptr() % 16 == 0:
        return x
    return x.clone(memory_format=torch.contiguous_format)


# The providers that split K per row count; the others allocate only their output and
# quantized activation per call.
_SPLIT_K = frozenset({"triton-fp8"})


def warm_rows(
    provider_for: Callable[[int], str], max_rows: int
) -> tuple[list[int], list[tuple[int, int, str]]]:
    """The row counts to call an operation at, eagerly, so that every route it takes up to
    ``max_rows`` rows compiles; and its provider runs ``(first, last, provider)``.

    It names a batch's provider but not its buckets, so: every row count a split-K provider runs
    (each may split differently), the last of each other provider's run, and each power of two,
    where the CuTe configuration tables change; each also with the row count before it in its
    run, as Triton compiles a kernel apart for integer arguments not divisible by 16.
    """

    providers = [provider_for(count) for count in range(1, max_rows + 1)]
    runs: list[tuple[int, int, str]] = []
    for count, provider in enumerate(providers, 1):
        if runs and runs[-1][2] == provider:
            runs[-1] = (runs[-1][0], count, provider)
        else:
            runs.append((count, count, provider))
    rows = {1 << i for i in range(max_rows.bit_length())} | {last for _, last, _ in runs}
    rows |= {count for count, provider in enumerate(providers, 1) if provider in _SPLIT_K}
    rows |= {
        count - 1 for count in rows if count > 1 and providers[count - 2] == providers[count - 1]
    }
    return sorted(rows), runs


class Fp8BlockReference:
    """The contract computed by torch, for the CPU and as the tests' oracle: each product
    dequantizes the weight into the shared scratch in whole 128-row chunks, quantizes and
    dequantizes the activation as the contract does, and multiplies in the activation dtype."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype) -> None:
        self._weight, self._scale, self._dtype = weight, scale, dtype

    def warm_up(self, max_rows: int, *, out_row_stride: int) -> GemmRoutes:
        return Counter()

    def workspace_bytes(self) -> int:
        rows, cols = self._weight.shape
        row_bytes = cols * self._dtype.itemsize
        return _rows(rows, row_bytes, _SCRATCH_BYTES) * row_bytes

    def __call__(
        self, x: torch.Tensor, workspace: torch.Tensor | None, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if out is not None:
            return out.copy_(self(x, workspace))
        rows, cols = self._weight.shape
        scratch = workspace.view(x.dtype)
        x_q = _contract_input(x)
        step = _rows(rows, cols * x.dtype.itemsize, scratch.numel() * x.dtype.itemsize)
        outputs = []
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            dequantized = scratch[: (stop - start) * cols].view(-1, cols)
            for first, last, decoded in _dequantized_rows(self._weight, self._scale, start, stop):
                dequantized[first - start : last - start].copy_(decoded)
            outputs.append(F.linear(x_q, dequantized))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, -1)

    def frozen_product(self) -> FrozenProduct:
        return Fp8BlockReferenceProduct(self._weight, self._scale)


class Fp8BlockReferenceProduct:
    """cti-kernels' ``FrozenProduct`` by the contract, for the CPU and as the tests' oracle:
    the forward is the serving product (the contract's activation and weight); ``grad_input``
    is ``dY @ W_hat`` with the activation's quantizer passed straight through, the frozen
    weight dequantized again in 128-row chunks rather than kept. It reduces over no group:
    a caller's ``exchange`` is refused, the layer's exit reduces."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self._weight, self._scale = weight, scale
        self.out_features, self.in_features = weight.shape

    def forward(
        self,
        x: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        exchange: Any = None,
        epilogue: Callable[[slice, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        _refuse(exchange)
        x_q = _contract_input(x)
        rows = _dequantized_rows(self._weight, self._scale, 0, self.out_features)
        y = torch.cat([F.linear(x_q, decoded.to(x.dtype)) for _, _, decoded in rows], -1)
        if epilogue is not None:
            epilogue(slice(0, y.shape[0]), y)
        return y if out is None else out.copy_(y)

    def grad_input(
        self,
        dy: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        exchange: Any = None,
    ) -> torch.Tensor:
        _refuse(exchange)
        x_gradient = dy.new_zeros(dy.shape[0], self.in_features, dtype=torch.float32)
        if accumulate:
            x_gradient += _bound_out(out)
        rows = _dequantized_rows(self._weight, self._scale, 0, self.out_features)
        for first, last, decoded in rows:
            x_gradient.add_(dy[:, first:last].float() @ decoded)
        x_gradient = x_gradient.to(dy.dtype)
        return x_gradient if out is None else out.copy_(x_gradient)


def _refuse(exchange: Any) -> None:
    if exchange is not None:
        raise ValueError("the reference product reduces over no group; pass exchange=None")


def _bound_out(out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        raise ValueError("accumulate=True adds into out, which it needs")
    return out


# Each device type's provider of the block-FP8 product.
PROVIDERS: Mapping[str, type[_Provider]] = MappingProxyType(
    {"cpu": Fp8BlockReference, "cuda": Fp8BlockKernel}
)


class _FrozenExperts(torch.autograd.Function):
    """Frozen routed experts for training through cti-kernels' ``BlockFP8MoE``: its forward
    (the serving numerics), and ``dx`` back, which gives the input's gradient (the activation
    quantizers passed straight through) and the routing weights'."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, weights: torch.Tensor, selected: torch.Tensor, operation: Any
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weights, selected)
        ctx.operation = operation
        # The kernel takes plain tensors; this function records the graph.
        return operation(x.detach(), weights.detach(), selected)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x, weights, selected = ctx.saved_tensors
        x_gradient, weight_gradient = ctx.operation.dx(
            x.detach(), weights.detach(), selected, gradient.contiguous()
        )
        return x_gradient, weight_gradient.to(weights.dtype), None, None


class _FrozenTraining(torch.autograd.Function):
    """A frozen product for training: its forward, and ``grad_input`` back; nothing kept but
    the product."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, product: FrozenProduct) -> torch.Tensor:
        ctx.product, ctx.shape = product, x.shape
        y = product.forward(x.detach().reshape(-1, x.shape[-1]))
        return y.view(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        dy = gradient.reshape(-1, gradient.shape[-1]).contiguous()
        return ctx.product.grad_input(dy).view(ctx.shape), None


def _contract_input(x: torch.Tensor) -> torch.Tensor:
    """``x`` quantized and dequantized per token and 128 columns, in its dtype."""

    return numerics.dequantize_activation(*numerics.quantize_activation(x)).to(x.dtype)


def _dequantized_rows(
    weight: torch.Tensor, scale: torch.Tensor, start: int, stop: int
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """``(first, last, W_hat[first:last])`` in FP32 over rows ``[start, stop)``, in whole
    128-row blocks within ``_DEQUANTIZE_BYTES``."""

    cols = weight.shape[1]
    padded = -(-cols // numerics.BLOCK) * numerics.BLOCK
    piece = _rows(stop - start, padded * 4, _DEQUANTIZE_BYTES)
    for first in range(start, stop, piece):
        last = min(first + piece, stop)
        blocks = slice(first // numerics.BLOCK, -(-last // numerics.BLOCK))
        yield first, last, numerics.dequantize_block_weight(weight[first:last], scale[blocks])


def _dequantized_stack(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """``W_hat`` of every expert of an E4M3 stack ``[E, R, C]`` and its grids, in ``dtype``."""

    output = torch.empty(weight.shape, dtype=dtype, device=weight.device)
    for expert in range(weight.shape[0]):
        output[expert] = numerics.dequantize_block_weight(weight[expert], scale[expert])
    return output


def _rows(rows: int, row_bytes: int, budget: int) -> int:
    """The most whole 128-row blocks of ``row_bytes``-byte rows within ``budget`` bytes (one
    block at least), and no more than ``rows``."""

    return min(rows, max(1, budget // (numerics.BLOCK * row_bytes)) * numerics.BLOCK)


class Fp8Block128Experts(ExpertsMethod):
    """Fp8Block128 routed experts: E4M3 stacks with their scale grids, stacked by expert; they
    route and run on the provider ``EXPERTS_PROVIDERS`` names for the device type."""

    def __init__(self, experts: Experts) -> None:
        super().__init__(experts)
        self._call = _Fp8ExpertsCall(experts)

    @classmethod
    def computed_by(cls, device: torch.device) -> str:
        return provider_row(EXPERTS_PROVIDERS, device)

    def declare(self) -> dict[str, torch.Tensor]:
        experts = self._experts
        count, width = len(experts.placement.experts), len(experts.placement.intermediate)
        hidden = experts.hidden_size
        if width % numerics.BLOCK:
            raise ValueError(
                f"an expert window of {width} intermediate columns cuts its "
                f"{numerics.BLOCK}-element scale blocks"
            )
        blocks = (-(-hidden // numerics.BLOCK), width // numerics.BLOCK)
        return {
            "gate_up_proj": torch.empty(count, 2 * width, hidden, dtype=numerics.E4M3),
            "down_proj": torch.empty(count, hidden, width, dtype=numerics.E4M3),
            # Gate then up rows, each in whole blocks.
            "gate_up_proj_weight_scale_inv": torch.empty(
                count, 2 * blocks[1], blocks[0], dtype=torch.float32
            ),
            "down_proj_weight_scale_inv": torch.empty(
                count, blocks[0], blocks[1], dtype=torch.float32
            ),
        }

    def windows(self, leaf: str, windows: tuple[WeightShard, ...]) -> tuple[WeightShard, ...]:
        if not leaf.endswith("_weight_scale_inv"):
            return windows
        # The stacks' expert axis leads; the expert window is not cut into blocks.
        return block_windows(windows, {0: numerics.BLOCK, 1: numerics.BLOCK})

    def call_sites(self) -> Iterator[GemmCallSite]:
        yield self._call

    def joins_shared(self) -> bool:
        """An adapted call on CUDA: cti-kernels' fused_experts_lora joins the shared expert in
        its combine (the frozen BlockFP8MoE takes no shared input)."""

        experts = self._experts
        return experts.lora is not None and experts.gate_up_proj.device.type == "cuda"

    def apply(
        self,
        hidden_states: torch.Tensor,
        router: torch.Tensor,
        shared: torch.Tensor | None = None,
        shared_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        experts = self._experts
        if experts.lora is None:
            return self._call(hidden_states, router)
        return experts.moe_backend.forward(
            hidden_states=hidden_states,
            router=router,
            topk=experts.top_k,
            renormalize=experts.renormalize,
            expert_map=experts.expert_map,
            lora=experts.lora,
            shared=shared,
            shared_gate=shared_gate,
            **self._adapted_stacks(hidden_states),
        )

    def training(self, hidden_states: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        experts = self._experts
        operation = None if experts.lora is not None else self._call.frozen_experts()
        if operation is not None:
            weights, selected = experts.moe_backend.training_route(
                hidden_states, router, experts.top_k, experts.renormalize
            )
            return _FrozenExperts.apply(hidden_states, weights, selected.int(), operation)
        return experts.moe_backend.forward_training(
            hidden_states=hidden_states,
            router=router,
            topk=experts.top_k,
            renormalize=experts.renormalize,
            expert_map=experts.expert_map,
            lora=None if experts.lora is None else experts.lora.training(),
            **self._adapted_stacks(hidden_states),
        )

    def training_bytes(self, tokens: int, *, rank: int) -> tuple[int, int]:
        """Adapted on CUDA, cti-kernels' straight-through autograd saves the rows (by
        reference), the FP32 router weights and the local ids, and its backward recomputes
        the forward's state: per route the gate|up, activated and down rows (cache 1-3, BF16),
        the 1x128 activation scales and the low-rank inputs, beside its FP32 dX rows.
        Elsewhere ``ExpertsMethod.training_bytes``."""

        experts = self._experts
        if experts.lora is None or experts.gate_up_proj.device.type != "cuda":
            return super().training_bytes(tokens, rank=rank)
        hidden, width = experts.hidden_size, len(experts.placement.intermediate)
        routes = tokens * experts.top_k
        saved = routes * (4 + 4)
        caches = 2 * (max(2 * width, hidden) + width + width // 2)
        backward = routes * (caches + 4 * -(-width // 128) + 3 * 2 * rank + 4 * hidden)
        return saved, backward

    def _adapted_stacks(self, hidden_states: torch.Tensor) -> dict[str, Any]:
        """The stacks an adapted or kernel-less call passes the backend: on CUDA the E4M3
        stacks as stored, with their block multipliers (cti-kernels runs the LoRA beside
        the FP8 product, its rows quantized 1x128 per token); elsewhere fresh dequantized
        stacks, which a backward reads as its own and no later layer overwrites."""

        experts = self._experts
        if hidden_states.is_cuda and experts.lora is not None:
            return {
                "w1": experts.gate_up_proj,
                "w2": experts.down_proj,
                "scales": (
                    experts.gate_up_proj_weight_scale_inv,
                    experts.down_proj_weight_scale_inv,
                ),
            }
        gate_up, down = self._dequantized(hidden_states.dtype)
        return {"w1": gate_up, "w2": down}

    def _dequantized(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        experts = self._experts
        return tuple(
            _dequantized_stack(weight, scale, dtype)
            for weight, scale in (
                (experts.gate_up_proj, experts.gate_up_proj_weight_scale_inv),
                (experts.down_proj, experts.down_proj_weight_scale_inv),
            )
        )


class _ExpertsProvider(Protocol):
    """One implementation of a layer's routed block-FP8 experts, bound to its loaded stacks."""

    def warm_up(self, max_rows: int) -> GemmRoutes: ...

    def __call__(self, hidden_states: torch.Tensor, router: torch.Tensor) -> torch.Tensor: ...

    def frozen_experts(self) -> Any:
        """The kernel operation whose forward and ``dx`` train the frozen experts, or None
        where training dequantizes the stacks."""
        ...


class _Fp8ExpertsCall:
    """One layer's routed experts. Their warm-up binds the loaded stacks to the provider of the
    device type, which every later call runs."""

    def __init__(self, experts: Experts) -> None:
        self._experts = experts
        self.services = experts.services
        # The dtype the layer computes in: the one in effect when it was built.
        self._dtype = torch.get_default_dtype()
        self._provider: _ExpertsProvider | None = None

    def warm_up(
        self, max_rows: int, device: torch.device, *, renew: Callable[[str], None]
    ) -> GemmRoutes:
        # The kernel warms every row bucket in one call: nothing to renew between them.
        experts = self._experts
        if experts.gate_up_proj.device.type != device.type:
            raise ValueError(
                f"block-FP8 experts on {experts.gate_up_proj.device} warm up on {device}"
            )
        return self._bind().warm_up(max_rows)

    def frozen_experts(self) -> Any:
        """The bound provider's frozen experts; training without a warm-up binds it."""

        return (self._provider or self._bind()).frozen_experts()

    def _bind(self) -> _ExpertsProvider:
        experts = self._experts
        self._provider = EXPERTS_PROVIDERS[experts.gate_up_proj.device.type](
            experts.gate_up_proj,
            experts.gate_up_proj_weight_scale_inv,
            experts.down_proj,
            experts.down_proj_weight_scale_inv,
            top_k=experts.top_k,
            renormalize=experts.renormalize,
            expert_map=experts.expert_map,
            dtype=self._dtype,
        )
        return self._provider

    def workspace_bytes(self, device: torch.device) -> int:
        return 0

    def __call__(self, hidden_states: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        if self._provider is None:
            raise RuntimeError(
                "prepare_inference binds block-FP8 experts' provider before their use"
            )
        return self._provider(hidden_states, router)


# The most routes a batch lays out one block per route for (the Triton chain).
_ROUTE_BLOCKS = 64


class Fp8ExpertsKernel:
    """cti-kernels' ``BlockFP8MoE``, routed by the row-count-invariant router every MoE path
    shares (``ctikernels.moe.route``); it picks its implementation per device and row count.
    With ``expert_map`` the stacks hold the experts it maps to an index; a route to another
    expert adds 0."""

    def __init__(
        self,
        gate_up: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down: torch.Tensor,
        down_scale: torch.Tensor,
        *,
        top_k: int,
        renormalize: bool,
        expert_map: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> None:
        from ctikernels.moe_block_fp8 import BlockFP8MoE

        self._operation = BlockFP8MoE(
            gate_up, gate_up_scale, down, down_scale, top_k=top_k, expert_map=expert_map
        )
        self._top_k, self._renormalize, self._dtype = top_k, renormalize, dtype
        self._num_experts = gate_up.shape[0] if expert_map is None else expert_map.numel()

    def warm_up(self, max_rows: int) -> GemmRoutes:
        """Run, eagerly, every route a batch of up to ``max_rows`` rows takes (``warm_rows``)
        and every row count whose route layout the operation keeps per route count."""

        rows, runs = warm_rows(self._route_for, max_rows)
        laid_out = [
            count for count in range(1, max_rows + 1) if count * self._top_k <= _ROUTE_BLOCKS
        ]
        device, h = self._operation.device, self._operation.h
        router = torch.zeros(self._num_experts, h, dtype=self._dtype, device=device)
        for count in sorted({*rows, *laid_out}):
            self(torch.zeros(count, h, dtype=self._dtype, device=device), router)
        return Counter(
            {((first, last), f"experts {route.split()[0]}"): 1 for first, last, route in runs}
        )

    def _route_for(self, rows: int) -> str:
        """The implementation and configuration a batch of ``rows`` rows runs."""

        config = self._operation.config_for(rows)
        if isinstance(config, tuple):
            config = tuple(tuple(sorted(stage.items())) for stage in config)
        return f"{self._operation.provider_for(rows)} {config}"

    def __call__(self, hidden_states: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        from ctikernels.moe import route

        weights, ids = route(hidden_states, router, self._top_k, renormalize=self._renormalize)
        return self._operation(hidden_states, weights, ids)

    def frozen_experts(self) -> Any:
        return self._operation


class Fp8ExpertsReference:
    """The contract computed by torch, for the CPU and as the tests' oracle: rows route by the
    router every MoE path shares; each route dequantizes its expert's stacks, quantizes and
    dequantizes both projections' inputs as the contract does and multiplies in the
    activation dtype; the weighted routes sum in FP32 and round once."""

    def __init__(
        self,
        gate_up: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down: torch.Tensor,
        down_scale: torch.Tensor,
        *,
        top_k: int,
        renormalize: bool,
        expert_map: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> None:
        self._stacks = ((gate_up, gate_up_scale), (down, down_scale))
        self._top_k, self._renormalize, self._expert_map = top_k, renormalize, expert_map

    def warm_up(self, max_rows: int) -> GemmRoutes:
        return Counter()

    def frozen_experts(self) -> None:
        return None

    def __call__(self, hidden_states: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        weights, ids = _route(hidden_states, router, self._top_k, renormalize=self._renormalize)
        if self._expert_map is not None:
            ids = self._expert_map[ids].long()
        output = hidden_states.new_zeros(hidden_states.shape, dtype=torch.float32)
        for expert in ids[ids >= 0].unique().tolist():
            rows, slots = (ids == expert).nonzero(as_tuple=True)
            x = hidden_states[rows]
            for index, (weight, scale) in enumerate(self._stacks):
                w_hat = numerics.dequantize_block_weight(weight[expert], scale[expert])
                product = F.linear(_contract_input(x), w_hat.to(x.dtype))
                if index == 0:
                    gate, up = product.chunk(2, dim=-1)
                    x = F.silu(gate) * up
            output.index_add_(0, rows, product.float() * weights[rows, slots, None].float())
        return output.to(hidden_states.dtype)


def _route(
    x: torch.Tensor, router: torch.Tensor, top_k: int, *, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """``ctikernels.moe.route`` in torch: FP32 logits of ``x`` under ``router``, the ``top_k``
    largest (ties to the lower expert) weighted by a softmax over them with ``renormalize``,
    else over all experts."""

    logits = F.linear(x, router).float()
    ids = logits.sort(dim=-1, descending=True, stable=True).indices[:, :top_k]
    if renormalize:
        return torch.softmax(logits.gather(-1, ids), dim=-1), ids
    return torch.softmax(logits, dim=-1).gather(-1, ids), ids


# Each device type's provider of block-FP8 routed experts.
EXPERTS_PROVIDERS: Mapping[str, type[_ExpertsProvider]] = MappingProxyType(
    {
        "cpu": Fp8ExpertsReference,
        "cuda": Fp8ExpertsKernel,
    }
)
