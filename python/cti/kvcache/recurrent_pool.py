"""Physical request and prefix-snapshot state of recurrent layers.

``InferenceState.allocate`` constructs the pool and ``InferenceRunner`` owns it; each
recurrent layer binds it. It is not a KV pool: rows hold recurrent state, never token
pages.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import torch


def next_snapshot_position(position, alignment: int):
    """The first grid point past ``position``, an int or an int64 tensor of positions."""

    return (position // alignment + 1) * alignment


@dataclass(frozen=True, slots=True)
class SlotStateSpec:
    """One request's slot-row state of one layer: named ``(name, shape, dtype)`` tensors,
    stored in this order in each row; a prompt forward of the layer exports the state after
    the most whole multiples of ``export_granularity`` tokens it runs."""

    tensors: tuple[tuple[str, tuple[int, ...], torch.dtype], ...]
    export_granularity: int = 1

    @property
    def tensor_bytes(self) -> tuple[int, ...]:
        """Bytes of each tensor in one row."""
        return tuple(prod(shape) * dtype.itemsize for _, shape, dtype in self.tensors)

    @property
    def bytes_per_slot(self) -> int:
        return sum(self.tensor_bytes)


class RecurrentStatePool:
    """The tensors of each slot-state layer's ``SlotStateSpec``, one row per slot.

    Row ``s`` belongs to request slot ``s`` (``RequestSlot.index``); there is no sink
    row, because the operations treat slot -1 as padding. Snapshot rows follow the
    request slots and share their allocations and residency regions. Rows are never reset: a
    sequence without an initial state starts from zero and does not read its row.
    Each layer's tensors are views of one buffer (``allocate_rows``), which a caller may
    allocate in a region that releases its memory and restores it at the same addresses; the
    pool, which builds the row copy's table of their addresses, is built outside it.
    """

    @staticmethod
    def allocate_rows(
        specs: Sequence[SlotStateSpec], num_rows: int, *, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        """Each layer's uint8 buffer of ``num_rows`` rows."""

        return tuple(
            torch.empty(num_rows * spec.bytes_per_slot, dtype=torch.uint8, device=device)
            for spec in specs
        )

    def __init__(
        self,
        specs: Sequence[SlotStateSpec],
        *,
        num_slots: int,
        num_snapshot_rows: int = 0,
        device: torch.device,
        rows: Sequence[torch.Tensor] | None = None,
    ) -> None:
        """The pool over ``rows``, the layers' buffers from ``allocate_rows``, or new ones."""

        specs = tuple(specs)
        if not specs:
            raise ValueError("recurrent state must contain at least one layer")
        if type(num_slots) is not int or num_slots < 1:
            raise ValueError(f"num_slots must be a positive integer, got {num_slots!r}")

        if type(num_snapshot_rows) is not int or num_snapshot_rows < 0:
            raise ValueError("num_snapshot_rows must be a non-negative integer")
        num_rows = num_slots + num_snapshot_rows
        if rows is None:
            rows = self.allocate_rows(specs, num_rows, device=device)
        layer_views = []
        for spec, buffer in zip(specs, rows, strict=True):
            views, offset = [], 0
            for (_, shape, dtype), nbytes in zip(spec.tensors, spec.tensor_bytes, strict=True):
                end = offset + num_rows * nbytes
                views.append(buffer[offset:end].view(dtype).view(num_rows, *shape))
                offset = end
            layer_views.append(tuple(views))
        self._layer_views = tuple(layer_views)
        # One launch copies rows of every layer's tensors, through a device table of their
        # addresses: metadata no region may release.
        from ctikernels.kvcache.inference import RowCopy

        self._row_copy = RowCopy([view for views in layer_views for view in views])
        self._row_copy.warm_up()
        self.specs = specs
        self.num_layers = len(specs)
        self.num_slots = num_slots
        self.device = device
        self.snapshot_rows = range(num_slots, num_rows)

    def copy_rows(self, indices: torch.Tensor) -> None:
        """Copy int64 ``[2, copies]`` source/target rows across layers on the caller's stream,
        in one launch and without scratch.

        Source/target domains are disjoint; snapshot admission protects source rows.
        """
        sources, targets = indices
        self._row_copy(sources, targets)

    def layer_views(self, layer_index: int) -> tuple[torch.Tensor, ...]:
        """The stable views of one layer's tensors, in its spec's order."""

        return self._layer_views[layer_index]

    def layer_state(self, layer_index: int) -> dict[str, torch.Tensor]:
        """The stable views of one layer's tensors, by their spec's names."""

        names = (name for name, _, _ in self.specs[layer_index].tensors)
        return dict(zip(names, self._layer_views[layer_index], strict=True))


__all__ = [
    "RecurrentStatePool",
    "SlotStateSpec",
    "next_snapshot_position",
]
