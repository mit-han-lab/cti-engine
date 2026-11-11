"""What the engine needs of the ranks' group: one rank's place in it, and the collectives
its layers run."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, slots=True)
class Topology:
    """Rank ``rank`` of ``size`` ranks; tensor and expert parallelism span all of them."""

    rank: int
    size: int

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.size:
            raise ValueError(f"rank {self.rank} is not in a group of {self.size}")


class ResidualReduction(Protocol):
    """The ranks' sum of their shares of ``x``, added to ``residual`` and normalized by
    ``weight``, in place: ``(x, residual)`` become the norm and the stream
    (``ctikernels.collective.inference.AllReduceAddRMSNorm``)."""

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def prepare(self, token_counts: Iterable[int]) -> None:
        """Compile for these row counts, without communicating."""
        ...


class CollectiveGroup(Protocol):
    """The collectives a layer runs; a group of one rank returns its input."""

    @property
    def rank(self) -> int: ...

    @property
    def size(self) -> int: ...

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def residual_reduction(
        self, *, hidden_size: int, eps: float, zero_centered: bool
    ) -> ResidualReduction:
        """A new reduction of the ranks' shares into a residual norm of this geometry,
        built collectively; the group closes it."""
        ...


__all__ = ["CollectiveGroup", "ResidualReduction", "Topology"]
