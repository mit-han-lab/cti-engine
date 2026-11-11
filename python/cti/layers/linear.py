# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/layers/linear.py`, `python/minisgl/utils/misc.py`);
# see THIRD_PARTY_NOTICES.

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from cti.attention.requirement import kv_head_window, qo_head_window

from .base import BaseOP
from .linear_method import (
    GemmCallSite,
    InferenceGemm,
    LinearMethod,
    Split,
    Unquantized,
    declare_method,
    require_meta,
)
from .lora import LoRAProjection
from .placement import Param, WeightShard

if TYPE_CHECKING:
    from cti.config import LoRAConfig
    from cti.distributed import CollectiveGroup

    from .method import FrozenBase
    from .site import Site


class _Copy(torch.autograd.Function):
    """f: the identity forward; the ranks' input gradients summed backward, through the
    exit's reduction."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, sum_exit: AllReduceExit) -> torch.Tensor:
        ctx.sum_exit = sum_exit
        return x

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return ctx.sum_exit.reduce(gradient.contiguous().clone()), None


class _Reduce(torch.autograd.Function):
    """g: the ranks' partial outputs summed forward, through the exit's reduction; the
    identity backward."""

    @staticmethod
    def forward(ctx: Any, y: torch.Tensor, sum_exit: AllReduceExit) -> torch.Tensor:
        return sum_exit.reduce(y.contiguous().clone())

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient, None


class LocalEntry:
    """Input every rank of one holds whole."""

    @staticmethod
    def enter(x: torch.Tensor) -> torch.Tensor:
        return x


class ParallelEntry:
    """Input the ranks hold alike, each computing its own part from it: training sums
    the parts' input gradients over the ranks (f)."""

    def __init__(self, group: CollectiveGroup) -> None:
        self._exit = AllReduceExit(group)

    def enter(self, x: torch.Tensor) -> torch.Tensor:
        return _Copy.apply(x, self._exit)


class LocalExit:
    """One rank's product, bias included, is the layer's output."""

    @staticmethod
    def apply(
        product: Callable[..., torch.Tensor],
        x: torch.Tensor,
        bias: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return product(x, bias, *args, **kwargs)

    apply_training = apply

    @staticmethod
    def reduce(y: torch.Tensor) -> torch.Tensor:
        return y

    reduce_inference = reduce_training = reduce


class PartialExit(LocalExit):
    """Each rank's biasless product is its share of an output a later exit sums."""


class AllReduceExit:
    """Several ranks' products without the bias, summed over the ranks, then the bias once;
    training sums through g, whose backward hands each rank the whole output gradient."""

    def __init__(self, group: CollectiveGroup) -> None:
        self._group = group

    def apply(
        self,
        product: Callable[..., torch.Tensor],
        x: torch.Tensor,
        bias: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        y = self.reduce(product(x, None, *args, **kwargs))
        return y if bias is None else y + bias

    def apply_training(
        self,
        product: Callable[..., torch.Tensor],
        x: torch.Tensor,
        bias: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        y = self.reduce_training(product(x, None, *args, **kwargs))
        return y if bias is None else y + bias

    def reduce(self, y: torch.Tensor) -> torch.Tensor:
        return self._group.all_reduce(y)

    def reduce_inference(self, y: torch.Tensor) -> torch.Tensor:
        """A block's partial sums as inference's output."""

        return self.reduce(y)

    def reduce_training(self, y: torch.Tensor) -> torch.Tensor:
        return _Reduce.apply(y, self)


class DeferredExit(AllReduceExit):
    """An exit a residual norm follows: inference leaves each rank's share to that norm,
    which sums it with the residual add, rank 0's share carrying the bias; training sums
    here, as ``AllReduceExit``."""

    def apply(
        self,
        product: Callable[..., torch.Tensor],
        x: torch.Tensor,
        bias: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return product(x, bias if self._group.rank == 0 else None, *args, **kwargs)

    @staticmethod
    def reduce_inference(y: torch.Tensor) -> torch.Tensor:
        return y


@dataclass(frozen=True)
class ProjectionSlice:
    name: str
    offset: int
    size: int


class LinearBase(BaseOP):
    """A linear layer as one rank holds it.

    ``windows`` holds this rank's checkpoint window for each merged output slice, or the
    input-column window of a row-sharded weight; it is empty when every rank holds the
    whole weight. ``lora`` names the adapter target that may attach to it.

    Its ``method`` declares the weight's storage and computes the products,
    ``_product(x, bias, gate)`` for inference and ``_training_product(x, bias)``; an attached
    adapter replaces both with the method's products plus its own.
    """

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        *,
        has_bias: bool,
        site: Site,
        windows: tuple[WeightShard, ...] = (),
        gate_stride: int = 0,
        lora: str | None = None,
        lora_shared_a: Mapping[str, tuple[str, ...]] | None = None,
    ):
        """``gate_stride``: the row stride of an output gate the GEMM may fuse (0: none);
        ``lora_shared_a`` as in ``LoRAProjection``."""
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.gate_stride = gate_stride
        self.lora_target = lora
        self._lora_shared_a = lora_shared_a
        self._windows = windows
        self._lora: LoRAProjection | None = None
        # Column-parallel layers on several ranks enter through ParallelEntry.
        self._entry: LocalEntry | ParallelEntry = LocalEntry()
        self.services = site.services
        self._site = site
        self._declared: tuple[str, ...] = ()
        # A split projection's parts (``split``); none otherwise.
        self.parts: tuple[LinearBase, ...] = ()
        self._take(Unquantized)
        self.bias = torch.empty(local_osize) if has_bias else None

    @property
    def site(self) -> Site:
        """Where the projection runs, which the layers it feeds share."""
        return self._site

    @property
    def output_row_stride(self) -> int:
        """The row stride of the output its product writes: its own width, or a split
        projection's for a part, which writes its columns of the whole's."""
        return self.local_output_size

    def use_method(self, method: type[LinearMethod]) -> None:
        """Store and multiply the weight as ``method`` does: its tensors replace the ones
        declared before, on the device and in the dtype in effect, before weights load and
        before an adapter attaches."""

        if self._lora is not None:
            raise RuntimeError("a layer takes its method before a LoRA adapter attaches")
        require_meta(tensor for _, tensor in self._tensors())
        self._take(method)

    def _take(self, method: type[LinearMethod]) -> None:
        self.method, self._declared = declare_method(self, method, self._declared)
        self._product: Callable[..., torch.Tensor] = self.method.apply
        self._training_product: Callable[..., torch.Tensor] = self.method.training

    @property
    def windows(self) -> tuple[WeightShard, ...]:
        return self._windows

    @property
    def lora(self) -> LoRAProjection | None:
        """The attached adapter's factors, if any."""

        return self._lora

    @property
    def ranks(self) -> int:
        """How many ranks split this layer."""

        return self._site.group.size

    def frozen_product(self) -> FrozenBase | None:
        """The frozen base as a training op multiplies it, or None for a biased layer."""

        return None if self.bias is not None else self.method.frozen_product()

    def split(self, methods: Sequence[type[LinearMethod]]) -> None:
        """Compute each run of output slices that ``methods`` (one per slice, in order) give
        one method in a part of its own, named for its slices, holding the run's checkpoint
        parts; the product concatenates the parts'. Before weights load and before an
        adapter attaches."""

        require_meta(tensor for _, tensor in self._tensors())
        slices = self.output_slices
        if len(methods) != len(slices):
            raise ValueError(f"{len(methods)} methods for {len(slices)} output slices")
        runs: list[tuple[type[LinearMethod], list[int]]] = []
        for index, method in enumerate(methods):
            if runs and runs[-1][0] is method:
                runs[-1][1].append(index)
            elif any(run[0] is method for run in runs):
                raise ValueError(f"the slices {method.__name__} computes are not adjacent")
            else:
                runs.append((method, [index]))
        parts = []
        for method, indices in runs:
            part = _LinearPart(self, tuple(indices))
            part._take(method)
            setattr(self, part.name, part)
            parts.append(part)
        self.parts = tuple(parts)
        self._take(Split)

    def inference_gemms(self) -> Iterator[GemmCallSite]:
        yield from self.method.call_sites()
        yield from self.method.lora_call_sites()
        yield from super().inference_gemms()

    def _param(self, path: str, name: str, tensor: torch.Tensor) -> Param:
        # A bias follows column-sharded output rows; a row-sharded layer adds its whole bias
        # after the reduction.
        windows = self._windows
        if name == "bias":
            windows = windows if windows and windows[0].axis == 0 else ()
        elif windows:
            try:
                windows = self.method.windows(name, windows)
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from None
        return Param(path, tuple(tensor.shape), tensor.dtype, windows)

    def attach_lora(self, config: LoRAConfig) -> LoRAProjection:
        """Hold ``config``'s LoRA factors sized to this rank's input and output slices; their
        products replace the layer's from now on."""

        if self._lora is not None:
            raise RuntimeError("a LoRA adapter is already attached")
        self._lora = lora = LoRAProjection(
            input_size=self.local_input_size,
            output_slices=self.output_slices,
            rank=config.rank,
            scaling=config.alpha / config.rank,
            shared_a=self._lora_shared_a,
            windows=self._windows,
            logical_input_size=self.full_input_size,
            owns_whole=self._site.group.rank == 0,
        )
        self._product = self.method.lora_product(lora)
        self._training_product = self.method.lora_training_product(lora)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._product(x, self.bias)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        return self._training_product(self._entry.enter(x), self.bias)

    @property
    def output_slices(self) -> tuple[ProjectionSlice, ...]:
        return (
            ProjectionSlice(
                name="full",
                offset=0,
                size=self.local_output_size,
            ),
        )


class _LinearPart(LinearBase):
    """The adjacent output slices ``slots`` of a split projection, which hold its merged
    checkpoint tensors' parts of those slots."""

    def __init__(self, whole: LinearBase, slots: tuple[int, ...]) -> None:
        slices = [whole.output_slices[slot] for slot in slots]
        size = sum(output_slice.size for output_slice in slices)
        windows = tuple(whole.windows[slot] for slot in slots) if whole.windows else ()
        super().__init__(
            whole.full_input_size,
            size,
            whole.local_input_size,
            size,
            has_bias=False,
            site=whole.site,
            windows=windows,
        )
        self.name = "".join(output_slice.name for output_slice in slices)
        self._slots = slots
        self._whole_width = whole.local_output_size
        offsets = [0, *itertools.accumulate(output_slice.size for output_slice in slices)]
        self._slices = tuple(
            ProjectionSlice(output_slice.name, offset, output_slice.size)
            for output_slice, offset in zip(slices, offsets, strict=False)
        )

    @property
    def output_slices(self) -> tuple[ProjectionSlice, ...]:
        return self._slices

    @property
    def output_row_stride(self) -> int:
        return self._whole_width

    def _param(self, path: str, name: str, tensor: torch.Tensor) -> Param:
        param = super()._param(path, name, tensor)
        whole, _, _ = path.rpartition(f".{self.name}.")
        return replace(param, merge=(f"{whole}.{name}" if whole else name, self._slots))


class LinearReplicated(LinearBase):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        has_bias: bool,
        site: Site,
        lora: str | None = None,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
            site=site,
            lora=lora,
        )


class LinearColParallelMerged(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        *,
        has_bias: bool,
        site: Site,
        output_names: tuple[str, ...] | None = None,
        lora: str | None = None,
        lora_shared_a: Mapping[str, tuple[str, ...]] | None = None,
    ):
        rank, size = site.group.rank, site.group.size
        local_output_sizes = [_div_even(output_size, size) for output_size in output_sizes]
        windows = (
            tuple(WeightShard(0, rank * local, (rank + 1) * local) for local in local_output_sizes)
            if size > 1
            else ()
        )
        super().__init__(
            input_size,
            sum(output_sizes),
            input_size,
            sum(local_output_sizes),
            has_bias=has_bias,
            site=site,
            windows=windows,
            lora=lora,
            lora_shared_a=lora_shared_a,
        )
        if size > 1:
            self._entry = ParallelEntry(site.group)
        self._output_names = (
            tuple(f"slice_{index}" for index in range(len(output_sizes)))
            if output_names is None
            else output_names
        )
        if len(self._output_names) != len(output_sizes):
            raise ValueError("Merged-linear output names must match output sizes")
        if len(set(self._output_names)) != len(self._output_names) or any(
            not name for name in self._output_names
        ):
            raise ValueError("Merged-linear output names must be non-empty and unique")

        self._local_output_sizes = tuple(local_output_sizes)

    @property
    def output_slices(self) -> tuple[ProjectionSlice, ...]:
        result = []
        offset = 0
        for name, size in zip(self._output_names, self._local_output_sizes, strict=True):
            result.append(ProjectionSlice(name=name, offset=offset, size=size))
            offset += size
        return tuple(result)


class LinearQKVMerged(LinearBase):
    """Merged Q, K and V projection, optionally followed by a per-query-head output gate.

    With ``with_output_gate`` the output is ``q | k | v | gate``; the gate has one
    ``head_dim`` slice per local query head, so tensor parallelism splits it with Q. With
    fewer KV heads than ranks, consecutive ranks hold one head; its windows name its
    replicas.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        *,
        has_bias: bool,
        site: Site,
        with_output_gate: bool = False,
        projects_kv: bool = False,
        lora: str | None = None,
        lora_shared_a: Mapping[str, tuple[str, ...]] | None = None,
    ):
        """``projects_kv`` adds ``project_kv``, a GEMM of the K and V rows alone."""
        rank, size = site.group.rank, site.group.size
        first_qo, local_num_qo = qo_head_window(num_qo_heads, rank, size)
        first_kv, local_num_kv = kv_head_window(num_kv_heads, rank, size)
        slices_per_query_head = 2 if with_output_gate else 1
        full_osize = (slices_per_query_head * num_qo_heads + 2 * num_kv_heads) * head_dim
        local_osize = (slices_per_query_head * local_num_qo + 2 * local_num_kv) * head_dim
        windows: tuple[WeightShard, ...] = ()
        if size > 1:
            q_size = local_num_qo * head_dim
            q = WeightShard(0, first_qo * head_dim, first_qo * head_dim + q_size)
            replicas = max(1, size // num_kv_heads)
            kv = WeightShard(
                0,
                first_kv * head_dim,
                (first_kv + local_num_kv) * head_dim,
                replicas=replicas,
                replica=rank % replicas,
            )
            # The output gate belongs to the query heads, so it shares their window.
            windows = (q, kv, kv, q) if with_output_gate else (q, kv, kv)
        super().__init__(
            hidden_size,
            full_osize,
            hidden_size,
            local_osize,
            has_bias=has_bias,
            site=site,
            windows=windows,
            lora=lora,
            lora_shared_a=lora_shared_a,
        )

        if size > 1:
            self._entry = ParallelEntry(site.group)
        # This rank's heads, which the attention layer it feeds reads.
        self.head_dim = head_dim
        self.num_qo_heads = local_num_qo
        self.num_kv_heads = local_num_kv
        self._with_output_gate = with_output_gate
        self._kv_gemm: InferenceGemm | None = None
        if projects_kv:
            key, value = self.output_slices[1:3]
            end = value.offset + value.size
            self._kv_gemm = InferenceGemm(lambda: self.weight[key.offset : end], site.services)

    def project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """K and V of ``x``, Q not computed; for rows that only fill the KV cache. It reads
        the stored rows of a dense weight."""

        if self._kv_gemm is None:
            raise RuntimeError("this projection was built without projects_kv")
        if not self.method.dense:
            raise NotImplementedError(
                f"project_kv reads the stored weight, which {type(self.method).__name__} packs"
            )
        size = self.num_kv_heads * self.head_dim
        k, v = self._kv_gemm(x).split([size, size], dim=-1)
        return k, v

    def inference_gemms(self) -> Iterator[GemmCallSite]:
        if self._kv_gemm is not None and self.method.dense:
            yield self._kv_gemm
        yield from super().inference_gemms()

    @property
    def output_slices(self) -> tuple[ProjectionSlice, ...]:
        q_size = self.num_qo_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        slices = (
            ProjectionSlice(name="q", offset=0, size=q_size),
            ProjectionSlice(name="k", offset=q_size, size=kv_size),
            ProjectionSlice(name="v", offset=q_size + kv_size, size=kv_size),
        )
        if not self._with_output_gate:
            return slices
        return (*slices, ProjectionSlice(name="gate", offset=q_size + 2 * kv_size, size=q_size))


class LinearRowParallel(LinearBase):
    """Each rank multiplies its window of input columns; the exit the rank count picks at
    construction turns the products into the output."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        has_bias: bool,
        site: Site,
        gate_of: LinearQKVMerged | None = None,
        lora: str | None = None,
        reduce: bool = True,
        deferred: bool = False,
    ):
        """``gate_of`` is the projection whose output gate ``forward_gated`` may fuse; without
        ``reduce``, each rank's biasless product is its share of an output a later exit sums;
        ``deferred``, the residual norm after it sums inference's shares (``DeferredExit``)."""
        if has_bias and not reduce:
            raise ValueError("a row-parallel share without its reduction cannot add a bias")
        rank, size = site.group.rank, site.group.size
        local_input_size = _div_even(input_size, size)
        windows = (
            (WeightShard(1, rank * local_input_size, (rank + 1) * local_input_size),)
            if size > 1
            else ()
        )
        super().__init__(
            input_size,
            output_size,
            local_input_size,
            output_size,
            has_bias=has_bias,
            site=site,
            windows=windows,
            gate_stride=0 if gate_of is None else gate_of.local_output_size,
            lora=lora,
        )
        self._exit: LocalExit | AllReduceExit
        if size == 1:
            self._exit = LocalExit()
        elif not reduce:
            self._exit = PartialExit()
        else:
            self._exit = DeferredExit(site.group) if deferred else AllReduceExit(site.group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._exit.apply(self._product, x, self.bias)

    def forward_gated(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``forward(x * sigmoid(gate))`` for inference, the gate fused into the GEMM where
        cti-kernels accepts the call."""

        return self._exit.apply(self._product, x, self.bias, gate)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        return self._exit.apply_training(self._training_product, x, self.bias)


def _div_even(a: int, b: int) -> int:
    """Divides two integers that divide evenly."""
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b
