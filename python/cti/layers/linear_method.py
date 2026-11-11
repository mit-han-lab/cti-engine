"""How a linear layer stores its weight and multiplies by it.

A ``LinearBase`` owns its parallel role, bias, exit and LoRA composition; its method owns the
tensors it declares on the layer and the product. Every layer starts with ``Unquantized``;
the format methods in ``layers/quant`` replace it by path once the model is built on the meta
device, before LoRA attaches and weights load. This module names no format.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import torch
import torch.nn.functional as F

from ctikernels.linear import Linear

from .placement import WeightShard

if TYPE_CHECKING:
    from .lora import LoRAProjection
    from .services import LayerServices

# The routes one call site's warm-up compiled: ``((first_row, last_row), route)`` -> count.
GemmRoutes = Counter[tuple[tuple[int, int], str]]


class GemmCallSite(Protocol):
    """One inference call site ``prepare_inference`` compiles after the weights load and before
    any forward or graph capture; its calls run in its model's ``services.gemm_workspace``."""

    services: LayerServices

    def warm_up(
        self, max_rows: int, device: torch.device, *, renew: Callable[[str], None]
    ) -> GemmRoutes:
        """Compile the routes of up to ``max_rows`` rows; ``renew(bucket)`` hears each row
        bucket it compiles one by one."""
        ...

    def workspace_bytes(self, device: torch.device) -> int: ...


class AdaptedGemm:
    """The inference call site of a weight an adapter's fused product reads as stored: its
    LoRA inference plan compiles the delta's routes and the base product's."""

    def __init__(
        self, lora: LoRAProjection, weight: Callable[[], torch.Tensor], services: LayerServices
    ) -> None:
        self.lora = lora
        self.weight = weight
        self.services = services

    def workspace_bytes(self, device: torch.device) -> int:
        return 0

    def warm_up(
        self, max_rows: int, device: torch.device, *, renew: Callable[[str], None]
    ) -> GemmRoutes:
        routes: GemmRoutes = Counter()
        for route in self.lora.warm_up(max_rows, self.weight()):
            rows = route.get("m") or route["bucket"]
            routes[tuple(rows), f"lora {route['op']} {route.get('provider', 'cute')}"] += 1
        return routes


class FrozenBase(Protocol):
    """A frozen base ``W`` (``[out_features, in_features]``) as cti-kernels' training ops
    multiply it (their ``FrozenProduct``): ``forward`` is ``x @ W^T`` as serving computes it,
    into ``out`` when given, ``epilogue(rows, y)`` adding the caller's terms; ``grad_input`` is
    ``dy @ W``, added to ``out`` with ``accumulate``. ``exchange`` sums the result over a
    group. Neither call is differentiable nor mutates its input."""

    out_features: int
    in_features: int

    def forward(
        self,
        x: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        exchange: object = None,
        epilogue: Callable[[slice, torch.Tensor], None] | None = None,
    ) -> torch.Tensor: ...

    def grad_input(
        self,
        dy: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        exchange: object = None,
    ) -> torch.Tensor: ...


class AdaptedDelta:
    """The inference call site of an adapter's delta that a product adds after its base's
    (``LinearMethod.lora_product``). cti-kernels' ``lora_delta_factors_`` prepares a
    projection's factor tables and each row count's metadata on first use, which a capture
    may not do; every graph group runs its forward eagerly at each row count it captures just
    before capturing it, which prepares both. The warm-up compiles the kernels with one row."""

    def __init__(self, lora: LoRAProjection, services: LayerServices) -> None:
        self.lora = lora
        self.services = services

    def workspace_bytes(self, device: torch.device) -> int:
        return 0

    def warm_up(
        self, max_rows: int, device: torch.device, *, renew: Callable[[str], None]
    ) -> GemmRoutes:
        lora = self.lora
        if device.type != "cuda" or not lora.is_materialized:
            return Counter()
        lora.warm_up_delta(device)
        return Counter({((1, max_rows), "lora delta"): 1})


class InferenceGemm:
    """One inference call site's GEMM ``x @ weight().T`` of a frozen, biasless weight; a call
    site with an output gate of row stride ``gate_stride`` projects ``x * sigmoid(gate)``.

    cti-kernels' ``Linear`` picks the kernel per device, weight shape and M bucket, and fuses
    the gate into the GEMM where it accepts the call; elsewhere it gates first. It takes a
    gate with BF16 inputs only; other dtypes (the CPU references') gate before the call. Its
    split-K kernels run in its model's ``services.gemm_workspace``.
    """

    def __init__(
        self,
        weight: Callable[[], torch.Tensor],
        services: LayerServices,
        *,
        gate_stride: int = 0,
        out_row_stride: Callable[[], int] | None = None,
    ) -> None:
        """``out_row_stride``: the row stride of the output a call writes, where it is not
        the weight's row count (a split projection's part)."""

        self.weight = weight
        self.services = services
        self.gate_stride = gate_stride
        self._out_row_stride = out_row_stride
        self._operation = Linear()

    def __call__(
        self,
        x: torch.Tensor,
        gate: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gate is not None and x.dtype != torch.bfloat16:
            x, gate = self.services.gated_input()(x, gate), None
        return self._operation(
            x, self.weight(), out, gate=gate, workspace=self.services.gemm_workspace
        )

    def workspace_bytes(self, device: torch.device) -> int:
        """Split-K scratch bytes the largest call on ``device`` can need."""

        return self._operation.workspace_bytes(
            *self.weight().shape, device=device, gate_stride=self.gate_stride
        )

    def warm_up(
        self, max_rows: int, device: torch.device, *, renew: Callable[[str], None]
    ) -> Counter[tuple[tuple[int, int], str]]:
        """Compile every route a call of at most ``max_rows`` rows can take on ``device``, the
        fused gated one too at a gated call site, with one zero-row call per M bucket that
        runs a CuTe kernel (a bucket's row counts share one plan); return how many buckets
        each ``(bucket, route)`` covers. A gated route is ``fused`` where the kernel fuses the
        gate and ``unfused`` where the gate runs first and the plain route follows."""

        weight = self.weight()
        n, k = weight.shape
        routes: Counter[tuple[tuple[int, int], str]] = Counter()
        for gate_stride in sorted({0, self.gate_stride}):
            m = 1
            while m <= max_rows:
                decision = self._operation.decision(n, k, m, gate_stride=gate_stride, device=device)
                bucket = decision["bucket"]
                if bucket is None:  # past Linear's buckets: cuBLAS, nothing to compile
                    break
                rows = min(bucket[1], max_rows)
                route = decision["provider"]
                if gate_stride:
                    route = "fused" if route == "cute" else "unfused"
                routes[(bucket[0], rows), route] += 1
                if decision["provider"] == "cute":
                    renew(f"{n}x{k} rows {bucket[0]}-{rows}")
                    x = weight.new_zeros(rows, k)
                    gate = weight.new_zeros(rows, gate_stride)[:, :k] if gate_stride else None
                    self(x, gate)
                    stride = n if self._out_row_stride is None else self._out_row_stride()
                    if stride != n:
                        # A strided output compiles its own CuTe kernel.
                        self(x, gate, out=weight.new_empty(rows, stride)[:, :n])
                m = bucket[1] + 1
        return routes


def require_meta(tensors: Iterable[torch.Tensor]) -> None:
    """Refuse to change an op's method once any of its ``tensors`` left the meta device."""

    if any(not tensor.is_meta for tensor in tensors):
        raise RuntimeError("an op changes its method on the meta device, before weights load")


def declare_method(
    op: object, method: type[Any], declared: tuple[str, ...]
) -> tuple[Any, tuple[str, ...]]:
    """Build ``method`` for ``op`` and store the tensors it declares on ``op`` in place of the
    ``declared`` ones; return the method and the names it declared."""

    for name in declared:
        delattr(op, name)
    instance = method(op)
    tensors = instance.declare()
    for name, tensor in tensors.items():
        setattr(op, name, tensor)
    return instance, tuple(tensors)


class LinearHolder(Protocol):
    """The layer a method stores its tensors on: its local ``[out, in]`` extent, checkpoint
    windows over those elements, and the row stride of an output gate its GEMM may fuse."""

    local_output_size: int
    local_input_size: int
    gate_stride: int
    # The row stride of the output the product writes (a split's part writes its columns).
    output_row_stride: int
    services: LayerServices
    bias: torch.Tensor | None

    @property
    def windows(self) -> tuple[WeightShard, ...]: ...


class FrozenProjection(Protocol):
    """A frozen weight ``W [N, K]`` as a chunked training loss reads it: FP32 logits of a
    chunk of rows and the input gradient of their FP32 logit gradients; nothing it computes
    persists between calls."""

    def logits(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor: ...

    def input_gradient(self, gradient: torch.Tensor) -> torch.Tensor: ...


class LinearMethod:
    """One layer's weight storage and product ``x @ W.T (+ bias)``.

    ``declare`` returns the tensors the layer holds, on the device and in the dtype in effect
    (meta while the model is built); ``windows`` maps the layer's element windows to each
    stored tensor's. After the weights load, ``bind`` turns them into what ``apply`` reads, in
    place; ``call_sites`` are what ``prepare_inference`` compiles. Neither product reduces across
    ranks: the layer's exit does, and it passes ``bias`` where the product adds it.
    """

    # Whether ``weight`` holds the model dtype's matrix as multiplied, which ops that read
    # the weight themselves (a MoE layer's whole-block op) need.
    dense: ClassVar[bool] = False

    def __init__(self, layer: LinearHolder) -> None:
        self._layer = layer
        # The delta ``lora_product`` adds after this method's product, once an adapter attaches.
        self._delta: AdaptedDelta | None = None

    @classmethod
    def computed_by(cls, device: torch.device) -> str:
        """The row of the method's provider table that computes it on ``device``, by name;
        a device without one raises ``ValueError``."""
        raise NotImplementedError

    def declare(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def windows(self, leaf: str, windows: tuple[WeightShard, ...]) -> tuple[WeightShard, ...]:
        """The checkpoint windows of stored tensor ``leaf`` for the layer's element windows;
        the same by default. A window the tensor cannot take raises ``ValueError``."""

        return windows

    def bind(self) -> None:
        """Prepare the loaded tensors for ``apply``, replacing them on the layer."""

    def call_sites(self) -> Iterator[GemmCallSite]:
        return iter(())

    def apply(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The inference product of ``x * sigmoid(gate)`` (``x`` without a gate): fresh
        storage, or ``out``, a ``[rows, N]`` view of unit column stride and the layer's
        ``output_row_stride``, which it returns (a split projection's part writes its
        columns)."""
        raise NotImplementedError

    def training(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """The product for training, differentiable in ``x``, the weight frozen."""
        raise NotImplementedError

    def frozen_product(self) -> FrozenBase:
        """The frozen base as cti-kernels' training ops multiply it."""
        raise NotImplementedError

    def lora_product(self, lora: LoRAProjection) -> Callable[..., torch.Tensor]:
        """The inference product with ``lora``'s delta: this method's product, the delta of the
        unquantized ``x`` added in place."""

        self._delta = AdaptedDelta(lora, self._layer.services)

        def product(
            x: torch.Tensor, bias: torch.Tensor | None = None, gate: torch.Tensor | None = None
        ) -> torch.Tensor:
            if gate is not None:
                x = self._layer.services.gated_input()(x, gate)
            return lora.add_delta_(self.apply(x, bias), x)

        return product

    def lora_call_sites(self) -> Iterator[GemmCallSite]:
        """The call site of the delta ``lora_product`` adds, once an adapter attaches."""

        if self._delta is not None:
            yield self._delta

    def lora_training_product(self, lora: LoRAProjection) -> Callable[..., torch.Tensor]:
        """The training product with ``lora``'s delta: one kernel op over the frozen base."""

        def product(x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            return lora.training_product(x, self.frozen_product(), bias)

        return product


class Unquantized(LinearMethod):
    """A weight of the model dtype, multiplied as stored: cti-kernels' ``Linear`` for a
    biasless layer (it may fuse an output gate), ``F.linear`` with a bias."""

    dense = True

    def __init__(self, layer: LinearHolder) -> None:
        super().__init__(layer)
        self._gemm = InferenceGemm(
            lambda: layer.weight,
            layer.services,
            gate_stride=layer.gate_stride,
            out_row_stride=lambda: layer.output_row_stride,
        )
        # The adapter whose fused product, which reads the weight itself, replaced the GEMM.
        self._lora: LoRAProjection | None = None

    def declare(self) -> dict[str, torch.Tensor]:
        layer = self._layer
        return {"weight": torch.empty(layer.local_output_size, layer.local_input_size)}

    def call_sites(self) -> Iterator[GemmCallSite]:
        layer = self._layer
        if self._lora is not None:
            yield AdaptedGemm(self._lora, lambda: layer.weight, layer.services)
        # A biased layer (the vision tower's) runs F.linear.
        elif layer.bias is None:
            yield self._gemm

    def apply(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None:
            return self._gemm(x, gate, out)
        if gate is not None:
            x = self._layer.services.gated_input()(x, gate)
        y = F.linear(x, self._layer.weight, bias)
        return y if out is None else out.copy_(y)

    def training(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return F.linear(x, self._layer.weight, bias)

    def frozen_product(self) -> FrozenBase:
        return _DenseBase(self._layer.weight)

    # The fused LoRA kernels read the weight as stored.
    def lora_product(self, lora: LoRAProjection) -> Callable[..., torch.Tensor]:
        self._lora = lora

        def product(
            x: torch.Tensor, bias: torch.Tensor | None = None, gate: torch.Tensor | None = None
        ) -> torch.Tensor:
            if gate is not None:
                x = self._layer.services.gated_input()(x, gate)
            return lora.product(x, self._layer.weight, bias)

        return product


class _DenseBase:
    """An unquantized weight multiplied as stored, by cuBLAS. It reduces over no group: the
    layer's exit reduces, so a caller's ``exchange`` is refused."""

    def __init__(self, weight: torch.Tensor) -> None:
        self._weight = weight
        self.out_features, self.in_features = weight.shape

    def forward(
        self,
        x: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        exchange: object = None,
        epilogue: Callable[[slice, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        _no_exchange(exchange)
        y = torch.mm(x, self._weight.t(), out=out)
        if epilogue is not None:
            epilogue(slice(0, y.shape[0]), y)
        return y

    def grad_input(
        self,
        dy: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        exchange: object = None,
    ) -> torch.Tensor:
        _no_exchange(exchange)
        if accumulate:
            if out is None:
                raise ValueError("accumulate=True adds into out, which it needs")
            return out.addmm_(dy, self._weight)
        return torch.mm(dy, self._weight, out=out)


def _no_exchange(exchange: object) -> None:
    if exchange is not None:
        raise ValueError("a dense base reduces over no group; pass exchange=None")


class Split(LinearMethod):
    """A projection whose runs of slices take methods of their own (``LinearBase.split``):
    its parts hold the weights and their products concatenate."""

    def declare(self) -> dict[str, torch.Tensor]:
        return {}

    def apply(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Each part writes its columns of one output."""

        if gate is not None:
            x = self._layer.services.gated_input()(x, gate)
        rows = x.reshape(-1, x.shape[-1])
        if out is None:
            out = rows.new_empty(rows.shape[0], self._layer.local_output_size)
        first = 0
        for part in self._layer.parts:
            width = part.local_output_size
            part.method.apply(rows, out=out[:, first : first + width])
            first += width
        y = out.view(*x.shape[:-1], -1)
        return y if bias is None else y.add_(bias)

    def training(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = torch.cat([part.method.training(x) for part in self._layer.parts], -1)
        return y if bias is None else y + bias

    def frozen_product(self) -> FrozenBase:
        return _SplitBase(tuple(part.method.frozen_product() for part in self._layer.parts))


class _SplitBase:
    """A split projection's parts' frozen bases side by side: ``forward`` fills each part's
    columns, ``grad_input`` sums each part's product of its columns of ``dy``. It reduces
    over no group, as its parts do."""

    def __init__(self, parts: tuple[FrozenBase, ...]) -> None:
        self._parts = parts
        self.out_features = sum(part.out_features for part in parts)
        self.in_features = parts[0].in_features

    def _columns(self) -> Iterator[tuple[FrozenBase, slice]]:
        first = 0
        for part in self._parts:
            yield part, slice(first, first + part.out_features)
            first += part.out_features

    def forward(
        self,
        x: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        exchange: object = None,
        epilogue: Callable[[slice, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        _no_exchange(exchange)
        y = x.new_empty(x.shape[0], self.out_features) if out is None else out
        for part, columns in self._columns():
            y[:, columns] = part.forward(x)
        if epilogue is not None:
            epilogue(slice(0, y.shape[0]), y)
        return y

    def grad_input(
        self,
        dy: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        exchange: object = None,
    ) -> torch.Tensor:
        _no_exchange(exchange)
        (part, columns), *rest = self._columns()
        x_gradient = part.grad_input(dy[:, columns].contiguous(), out=out, accumulate=accumulate)
        for part, columns in rest:
            part.grad_input(dy[:, columns].contiguous(), out=x_gradient, accumulate=True)
        return x_gradient


__all__ = [
    "declare_method",
    "FrozenProjection",
    "require_meta",
    "GemmCallSite",
    "InferenceGemm",
    "LinearHolder",
    "LinearMethod",
    "Split",
    "Unquantized",
]
