"""Where this rank's selected parameters sit in their logical (single-rank) tensors: how
their gradients become logical gradients at a commit, and how checkpoints see them.

A layout, in parameter order: ``reduce`` makes each FP32 gradient its logical gradient in
place and returns the global L2 norm on the device; ``abandon`` joins that reduction without
valid gradients; ``logical``/``local`` map a value or per-element optimizer state to its
logical tensor and back; ``logical_shape`` names the former's shape; ``primary`` ranks
publish what every rank gathers. One rank's is ``LocalLayout``; tensor- and
expert-parallel ranks' is ``TensorParallelLayout``, over the group the worker injects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


class GradientLayout(Protocol):
    @property
    def primary(self) -> bool: ...

    def reduce(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor: ...

    def abandon(self, parameters: Sequence[torch.Tensor]) -> None: ...

    def logical(self, index: int, tensor: torch.Tensor) -> torch.Tensor: ...

    def local(self, index: int, logical: torch.Tensor) -> torch.Tensor: ...

    def logical_shape(self, index: int, tensor: torch.Tensor) -> tuple[int, ...]: ...


class LocalLayout:
    """One rank: its parameters are the logical ones."""

    primary = True

    @staticmethod
    def reduce(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.nn.utils.get_total_norm(gradients, foreach=gradients[0].is_cuda)

    @staticmethod
    def abandon(parameters: Sequence[torch.Tensor]) -> None:
        pass

    @staticmethod
    def logical(index: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def local(index: int, logical: torch.Tensor) -> torch.Tensor:
        return logical

    @staticmethod
    def logical_shape(index: int, tensor: torch.Tensor) -> tuple[int, ...]:
        return tuple(tensor.shape)


class Ranks(Protocol):
    """The tensor-parallel ranks a layout reduces over; the worker injects its group."""

    @property
    def rank(self) -> int: ...

    @property
    def size(self) -> int: ...

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor: ...


class Window(Protocol):
    """A contiguous range of one axis of a logical tensor, and which copy of it this is."""

    @property
    def axis(self) -> int: ...

    @property
    def start(self) -> int: ...

    @property
    def stop(self) -> int: ...

    @property
    def replica(self) -> int: ...


class FactorPlace(Protocol):
    """Where a rank holds one tensor, as its owner decided: at most one ``window`` (None: the
    whole tensor); ``reduce``, whether the ranks' gradients are partial sums of it;
    ``owner``, whether this rank counts it once."""

    @property
    def window(self) -> Window | None: ...

    @property
    def reduce(self) -> bool: ...

    @property
    def owner(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class PlacedTensor:
    """One selected tensor: its name, this rank's shape and its placement."""

    path: str
    shape: tuple[int, ...]
    location: FactorPlace

    @property
    def window(self) -> Window | None:
        return self.location.window

    @property
    def partial(self) -> bool:
        return self.location.reduce


class TensorParallelLayout:
    """Tensor- and expert-parallel ranks, each selected tensor placed as its owner placed
    it: at most one window of its logical tensor (a stacked expert tensor's experts are a
    window of dimension 0), counted once by its owner; partial when the ranks' local
    gradients sum to the logical one, over the group the worker injects.

    A commit sums every partial parameter, zero-padded to its logical shape, in one FP32
    all-reduce that also carries the owners' squared norm of the complete ones; a rank that
    abandons the commit sends NaN there, so every rank rejects it alike. A logical tensor is
    the all-reduce of each window's owner's values. Implementation: the ranks'
    torch.distributed all-reduce until cti-kernels' FP32 PyNCCL replaces it.
    """

    def __init__(
        self, group: Ranks, params: Sequence[PlacedTensor], *, device: torch.device
    ) -> None:
        """Construction gathers each window's stop once: the largest stop is the logical
        extent. Complete windows of dimension 0 (experts) must tile it."""

        self._group = group
        self._params = tuple(params)
        self._windows = tuple(param.window for param in params)
        ranges = [
            (0, 0) if window is None else (window.start, window.stop) for window in self._windows
        ]
        gathered = group.all_gather(torch.tensor(ranges, dtype=torch.int64, device=device))
        gathered = gathered.view(group.size, len(ranges), 2).tolist()
        stops = [max(rank[index][1] for rank in gathered) for index in range(len(ranges))]
        for index, (param, window) in enumerate(zip(params, self._windows, strict=True)):
            if window is not None and window.axis == 0 and not param.partial:
                spans = [rank[index] for rank in gathered]
                if sum(stop - start for start, stop in spans) != stops[index]:
                    raise ValueError(f"{param.path}: ranks' windows {spans} must tile it")
        self._shapes = tuple(
            tuple(
                extent if window is not None and axis == window.axis else size
                for axis, size in enumerate(param.shape)
            )
            for param, window, extent in zip(params, self._windows, stops, strict=True)
        )
        self._owners = tuple(param.location.owner for param in params)

    @property
    def primary(self) -> bool:
        """Whether this rank publishes what every rank gathers."""
        return self._group.rank == 0

    def reduce(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        buffer = self._sum(self._packed(gradients, poisoned=False))
        squares = buffer[-1].clone()
        for index, logical in self._partials(buffer):
            squares += logical.square().sum()
            gradients[index].copy_(self.local(index, logical))
        return squares.sqrt()

    def abandon(self, parameters: Sequence[torch.Tensor]) -> None:
        self._sum(self._packed(parameters, poisoned=True))

    def logical(self, index: int, tensor: torch.Tensor) -> torch.Tensor:
        logical = tensor.new_zeros(self._shapes[index])
        if self._owners[index]:
            self.local(index, logical).copy_(tensor)
        return self._sum(logical)

    def local(self, index: int, logical: torch.Tensor) -> torch.Tensor:
        return _window(logical, self._windows[index])

    def logical_shape(self, index: int, tensor: torch.Tensor) -> tuple[int, ...]:
        return self._shapes[index]

    def _sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._group.all_reduce(tensor)

    def _partials(self, buffer: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        """Each partial parameter's logical slot in ``buffer``."""

        slots, offset = [], 0
        for index, (param, shape) in enumerate(zip(self._params, self._shapes, strict=True)):
            if param.partial:
                size = math.prod(shape)
                slots.append((index, buffer.narrow(0, offset, size).view(shape)))
                offset += size
        return slots

    def _packed(self, tensors: Sequence[torch.Tensor], *, poisoned: bool) -> torch.Tensor:
        """The partial parameters' logical slots, then the owners' squared norm of the rest."""

        size = sum(
            math.prod(s) for p, s in zip(self._params, self._shapes, strict=True) if p.partial
        )
        buffer = torch.zeros(size + 1, dtype=torch.float32, device=tensors[0].device)
        if poisoned:
            buffer[-1] = float("nan")
            return buffer
        for index, logical in self._partials(buffer):
            self.local(index, logical).copy_(tensors[index])
        for tensor, param, owner in zip(tensors, self._params, self._owners, strict=True):
            if owner and not param.partial:
                buffer[-1] += tensor.float().square().sum()
        return buffer


class PlacedParameters(Protocol):
    """Parameters that place each of them, as adapters do."""

    def named_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]: ...

    def placements(self) -> Sequence[FactorPlace]: ...


def tensor_parallel_layout(
    group: Ranks,
    placed: PlacedParameters | None,
    names: Sequence[str],
    *,
    device: torch.device,
) -> TensorParallelLayout:
    """The layout of the tensors ``names``, each where ``placed`` puts it."""

    places = {}
    if placed is not None:
        for (name, tensor), location in zip(
            placed.named_parameters(), placed.placements(), strict=True
        ):
            places[name] = PlacedTensor(name, tuple(tensor.shape), location)
    if unplaced := [name for name in names if name not in places]:
        raise ValueError(f"Tensor-parallel training has no placement for {unplaced}")
    return TensorParallelLayout(group, [places[name] for name in names], device=device)


def _window(logical: torch.Tensor, window: Window | None) -> torch.Tensor:
    if window is None:
        return logical
    return logical.narrow(window.axis, window.start, window.stop - window.start)


__all__ = [
    "FactorPlace",
    "GradientLayout",
    "LocalLayout",
    "PlacedParameters",
    "PlacedTensor",
    "Ranks",
    "TensorParallelLayout",
    "Window",
    "tensor_parallel_layout",
]
