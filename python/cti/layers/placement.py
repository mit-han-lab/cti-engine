"""The tensors a rank stores and the checkpoint windows they load, and how routed experts
place on the ranks of a group."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch


class GroupShare(Protocol):
    """A rank of a group (``cti.distributed.Topology`` is one): its index and the
    group's size."""

    @property
    def rank(self) -> int: ...

    @property
    def size(self) -> int: ...


@dataclass(frozen=True, slots=True)
class WeightShard:
    """One rank's contiguous window of a checkpoint tensor, optionally zero-padded.
    ``replicas`` ranks hold the same window; this rank is holder ``replica`` of them."""

    axis: int
    start: int
    stop: int
    padded_size: int | None = None
    replicas: int = 1
    replica: int = 0

    def __post_init__(self) -> None:
        if self.axis < 0 or not 0 <= self.start <= self.stop:
            raise ValueError("invalid checkpoint shard coordinates")
        if self.padded_size is not None and self.padded_size < self.stop - self.start:
            raise ValueError("checkpoint shard padding cannot truncate its window")
        if not 0 <= self.replica < self.replicas:
            raise ValueError("a window's replica index must be below its replica count")

    @property
    def local_size(self) -> int:
        """The window's extent on this rank, padding included."""

        return self.stop - self.start if self.padded_size is None else self.padded_size

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return this window of a CPU checkpoint tensor.

        The window is a view of the checkpoint mapping. Only a vocabulary window shorter
        than its padded size is copied, to append zero rows.
        """

        if self.axis >= tensor.ndim or self.stop > tensor.shape[self.axis]:
            raise ValueError(
                f"checkpoint tensor of shape {tuple(tensor.shape)} does not contain "
                f"its window [{self.start}, {self.stop}) on axis {self.axis}"
            )
        window = tensor.narrow(self.axis, self.start, self.stop - self.start)
        if self.local_size == window.shape[self.axis]:
            return window
        padded_shape = list(window.shape)
        padded_shape[self.axis] = self.local_size
        padded = window.new_zeros(padded_shape)
        padded.narrow(self.axis, 0, window.shape[self.axis]).copy_(window)
        return padded


@dataclass(frozen=True, slots=True)
class Param:
    """One tensor a rank stores: its state-dict path, local shape and dtype, and the
    checkpoint windows it loads. ``windows`` holds one window per checkpoint part merged into
    the tensor (empty: every rank loads it whole); a stacked expert tensor names the global
    ``experts`` it holds. ``partial``: the tensor's gradient is summed over the ranks.
    ``merge`` names the merged tensor whose parts ``(path, slots)`` this one holds instead,
    where a projection computes groups of its parts apart."""

    path: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    windows: tuple[WeightShard, ...] = ()
    experts: range | None = None
    partial: bool = False
    merge: tuple[str, tuple[int, ...]] | None = None

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True, slots=True)
class Placement:
    """This rank's share of ``num_experts`` routed experts of width ``intermediate_size``: the
    global experts ``experts``, each cut to its intermediate window ``intermediate``."""

    num_experts: int
    intermediate_size: int
    experts: range
    intermediate: range

    def __post_init__(self) -> None:
        for name, window, size in (
            ("expert", self.experts, self.num_experts),
            ("intermediate", self.intermediate, self.intermediate_size),
        ):
            if window.step != 1 or not 0 <= window.start < window.stop <= size:
                raise ValueError(
                    f"{name} range [{window.start}, {window.stop}) is not inside {size}"
                )

    @property
    def holds_every_expert(self) -> bool:
        return len(self.experts) == self.num_experts

    @property
    def is_whole(self) -> bool:
        return self.holds_every_expert and len(self.intermediate) == self.intermediate_size

    def expert_map(self) -> torch.Tensor:
        """CPU int32 ``[num_experts]``: each held expert's local index, -1 for the others."""

        cpu = torch.device("cpu")
        expert_map = torch.full((self.num_experts,), -1, dtype=torch.int32, device=cpu)
        held = self.experts
        expert_map[held.start : held.stop] = torch.arange(len(held), dtype=torch.int32, device=cpu)
        return expert_map


class ExpertPlacement(Protocol):
    """How routed experts split over the ranks of a group; the MoE block reduces the ranks'
    partial outputs once."""

    def shard(
        self, num_experts: int, intermediate_size: int, topology: GroupShare
    ) -> Placement: ...


def _share(count: int, topology: GroupShare, what: str) -> range:
    """Rank ``topology.rank``'s equal, contiguous share of ``count`` items."""

    if count % topology.size:
        raise ValueError(f"{count} {what} do not divide evenly over {topology.size} ranks")
    size = count // topology.size
    return range(topology.rank * size, (topology.rank + 1) * size)


@dataclass(frozen=True, slots=True)
class ExpertParallel:
    """Each rank holds an equal range of whole experts."""

    def shard(self, num_experts: int, intermediate_size: int, topology: GroupShare) -> Placement:
        experts = _share(num_experts, topology, "experts")
        return Placement(num_experts, intermediate_size, experts, range(intermediate_size))


@dataclass(frozen=True, slots=True)
class ExpertTensorParallel:
    """Each rank holds every expert cut to an equal window of its intermediate width: its gate
    and up rows and the down columns that read them, as ``SharedExpert`` is cut."""

    def shard(self, num_experts: int, intermediate_size: int, topology: GroupShare) -> Placement:
        rows = _share(intermediate_size, topology, "expert intermediate rows")
        return Placement(num_experts, intermediate_size, range(num_experts), rows)


__all__ = [
    "ExpertParallel",
    "ExpertPlacement",
    "ExpertTensorParallel",
    "Param",
    "Placement",
    "WeightShard",
]
