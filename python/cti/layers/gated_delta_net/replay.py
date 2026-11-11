"""Verify and commit fixed token blocks through the Gated DeltaNet layers with
``ctikernels.gdn.replay.GatedDeltaNetReplay`` (speculative rounds' K2): each layer's
``LayerReplay`` and the family's ``RoundTransaction`` they share."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cti.forward import RoundShape
    from cti.memory import GraphPool
    from cti.recurrent import RecurrentGeometry, RecurrentVerify

    from .layer import GatedDeltaNet

# Replay records hold no ordinary decode window here; 4 is the smallest the kernel accepts.
_WINDOW = 4


class GatedDeltaNetLayerReplay:
    """One Gated DeltaNet layer's ``LayerReplay``: layer ``index`` of its family's K2, which
    reads the layer's live weights."""

    def __init__(self, transaction: GatedDeltaNetK2, index: int) -> None:
        self._transaction, self._index = transaction, index

    @property
    def transaction(self) -> GatedDeltaNetK2:
        return self._transaction

    def verify(self, projected: torch.Tensor, v: RecurrentVerify) -> torch.Tensor:
        """``y [B * Q, values]``; live state stays unchanged."""

        return self._transaction.verify_layer(self._index, v, projected)

    def handoff(self, slot: int, rows: Sequence[int], frontiers: torch.Tensor) -> None:
        """Mirror ``rows``' canonical state at their ``frontiers`` (int64, by slot) in place."""

        self._transaction.handoff_layer(self._index, slot, rows, frontiers)

    def commit(
        self,
        v: RecurrentVerify,
        lengths: torch.Tensor,
        snapshot_rows: torch.Tensor | None = None,
        snapshot_offsets: torch.Tensor | None = None,
    ) -> None:
        """Commit each active row's first ``lengths`` block tokens (int32 ``[B]``); with
        ``snapshot_rows`` (int32 ``[B]``, -1: none), also store the state after its first
        ``snapshot_offsets`` (int64 ``[B]``) tokens in that pool row when it commits them."""

        snapshot = None if snapshot_rows is None else (snapshot_rows, snapshot_offsets)
        self._transaction.commit_layer(self._index, v, lengths, snapshot)


@dataclass(frozen=True, slots=True)
class BlockStatus:
    """A closed round (``RoundStatus``): copy each ``(device, host)`` pair
    once it completed, then call ``settle`` in invocation order."""

    copies: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    settle: Callable[[], None]


def _operation(layer: GatedDeltaNet, block_size: int, device: torch.device):
    """The K2 replay operation of ``layer``'s head geometry for blocks of ``block_size``."""

    from ctikernels.gdn.replay import GatedDeltaNetReplay

    return GatedDeltaNetReplay(
        num_k_heads=layer.num_key_heads,
        num_v_heads=layer.num_value_heads,
        device=device,
        window=_WINDOW,
        verify_length=block_size,
        checks="trusted",
    )


class GatedDeltaNetK2:
    """The Gated DeltaNet family's ``RoundTransaction``: K2 storage, the round's begin and
    end, and the status of every joined layer; the canonical pool stays authoritative.
    Replay metadata mirrors each slot's committed position (commit keeps it, a handoff
    restores it). Layers join in ``create_replay`` order; the first use allocates the
    storage and closes the family. Trusted (CuTe) phases report no status, so settlement
    covers handoffs."""

    @classmethod
    def device_bytes(
        cls,
        layers: Sequence[GatedDeltaNet],
        *,
        num_slots: int,
        shape: RoundShape,
        num_snapshot_rows: int,
    ) -> int:
        op = _operation(layers[0], shape.width, layers[0].device)
        return cls._layout(op, len(layers), num_slots, shape, num_snapshot_rows)[1]

    @staticmethod
    def _layout(
        op, num_layers: int, num_slots: int, shape: RoundShape, num_snapshot_rows: int
    ) -> tuple[dict[object, tuple[int, tuple[int, ...], torch.dtype]], int]:
        """The offset, shape and dtype of each device tensor in the blocks' one allocation
        (256-byte aligned, as K2 regions are), and its bytes."""

        max_batch_size, depth = shape.max_rows, shape.depth
        state = op.state_spec(num_slots=num_slots, num_snapshot_rows=num_snapshot_rows)
        controls = op.graph_buffer_spec(
            batch_size=max_batch_size, depth=depth, num_layers=num_layers
        ).values()
        work = op.workspace_spec("verify", batch_size=max_batch_size)
        raw = work["pending.raw"]
        entries: list[tuple[object, tuple[int, ...], torch.dtype]] = [
            ((name, index), state[name].shape, state[name].dtype)
            for index in range(num_layers)
            for name in ("ring", "metadata", "pool_epoch", "snapshot_metadata")
        ]
        entries += [
            ("controls", (max(r.offset + r.aligned_bytes for r in controls),), torch.uint8),
            ("out", work["out"].shape, work["out"].dtype),
            # Each layer's pending records, per status slot (the K2 graph data layout).
            ("pending_raw", (num_layers, depth, *raw.shape), raw.dtype),
            ("next_positions", (num_slots,), torch.int64),
            ("handoff_slots", (depth, num_slots), torch.int32),
            ("snapshot_valid", (depth, max_batch_size), torch.bool),
        ]
        table = op.layer_table_spec(num_layers=num_layers)["layer_table"]
        entries.append(("layer_table", table.shape, table.dtype))
        if op.checks == "trusted":
            # CuTe's trusted commits borrow arrival counters; the Triton replay's spec has none.
            spec = op.graph_data_spec(batch_size=max_batch_size, depth=depth)
            if "phase_arrivals" in spec:
                entries.append(("arrivals", spec["phase_arrivals"].shape, torch.int32))
        layout, offset = {}, 0
        for name, dims, dtype in entries:
            layout[name] = (offset, tuple(dims), dtype)
            offset += -(-prod(dims) * dtype.itemsize // 256) * 256
        return layout, offset

    def __init__(self, geometry: RecurrentGeometry, *, shape: RoundShape) -> None:
        self._geometry, self._shape = geometry, shape
        self._layers: list[GatedDeltaNet] = []
        # Each joined layer's pool-row state and convolution history, all rows.
        self._rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._allocated = False
        # The (status slot, round) whose handoff rows are staged on the device.
        self._staged: tuple[int, int] | None = None
        self._round = 0

    def join(
        self, layer: GatedDeltaNet, row: tuple[torch.Tensor, torch.Tensor]
    ) -> GatedDeltaNetLayerReplay:
        """``layer``'s replay, the family's next layer over its bound pool ``row``."""

        if self._allocated:
            raise RuntimeError("the Gated DeltaNet K2 storage is already allocated")
        self._layers.append(layer)
        self._rows.append(row)
        return GatedDeltaNetLayerReplay(self, len(self._layers) - 1)

    def _allocate(self) -> None:
        if self._allocated:
            return
        self._allocated = True
        layers, geometry, shape = self._layers, self._geometry, self._shape
        max_batch_size, self.depth = shape.max_rows, shape.depth
        self._num_slots = num_slots = geometry.num_slots
        self._device = device = geometry.device
        self._op = op = _operation(layers[0], shape.width, device)
        layout, size = self._layout(op, len(layers), num_slots, shape, geometry.num_snapshot_rows)
        # Zeroed once: each commit's last arriving block resets the arrival counters it used,
        # and verify's zeroed output rows of inactive slots stay defined.
        backing = torch.zeros(size, dtype=torch.uint8, device=device)

        def view(name: object) -> torch.Tensor:
            offset, dims, dtype = layout[name]
            return backing[offset : offset + prod(dims) * dtype.itemsize].view(dtype).view(dims)

        controls = view("controls")
        buffers = {}
        for name, region in op.graph_buffer_spec(
            batch_size=max_batch_size, depth=self.depth, num_layers=len(layers)
        ).items():
            region_view = controls[region.offset : region.offset + region.payload_bytes]
            buffers[name] = region_view.view(region.dtype).view(region.shape)
            if region.initial_value is not None:
                buffers[name].fill_(region.initial_value)
        self._controls = op.bind_graph_buffers(buffers=buffers)
        self._sequences = buffers["sequence"]
        self._arrivals = view("arrivals") if "arrivals" in layout else None
        self._out = view("out")
        self._pending_raw = view("pending_raw")
        self._next_positions = view("next_positions")
        self._snapshot_valid = view("snapshot_valid")
        # One host and one device handoff slot vector per status slot: a status slot is
        # reused only after its settlement, so neither is rewritten while a copy reads it.
        # Pinning serves only the asynchronous copies of a CUDA pool.
        pin = device.type == "cuda"
        self._handoff_host = torch.full(
            (self.depth, num_slots), -1, dtype=torch.int32, pin_memory=pin
        )
        self._handoff_slots = view("handoff_slots")
        # Per status slot, every layer's handoff as one graph, once captured.
        self._handoff_graphs: tuple[torch.cuda.CUDAGraph, ...] = ()
        self._records_host = torch.zeros(
            (self.depth, len(layers), 16), dtype=torch.uint8, pin_memory=pin
        )
        self._validity_host = torch.zeros(
            (self.depth, len(layers), max_batch_size), dtype=torch.bool, pin_memory=pin
        )
        # Each layer's bound pool row supplies its checkpoint and history.
        states = []
        for index, (checkpoint, conv_history) in enumerate(self._rows):
            own = {
                name: view((name, index))
                for name in ("ring", "metadata", "pool_epoch", "snapshot_metadata")
            }
            buffers = {"checkpoint": checkpoint, "conv_history": conv_history, **own}
            states.append(op.create_state(buffers=buffers))
        self._states = tuple(states)
        # Every layer's handoff in one launch: the slot rows are the canonical inputs.
        self._layer_table = op.bind_layers(
            states=self._states,
            table=view("layer_table"),
            canonical_states=tuple(state.checkpoint[:num_slots] for state in self._states),
            histories=tuple(state.conv_history[:num_slots] for state in self._states),
        )

    def begin(self, slot: int, sequence: int) -> None:
        """Open round ``sequence`` in status slot ``slot`` on the current stream."""

        self._allocate()
        self._round = sequence
        self._sequences[slot].fill_(sequence)
        self._out.zero_()
        status = self._status(slot, 0)
        self._op.begin(
            status=status,
            sequence=status.sequence,
            layer_weights=tuple(layer.weights for layer in self._layers),
        )

    def handoff_layer(
        self, index: int, slot: int, rows: Sequence[int], frontiers: torch.Tensor
    ) -> None:
        """Layer ``index``'s handoff of ``rows``. One launch hands off every layer, so the
        family's first layer in a round runs it and the others find it done."""

        if self._staged == (slot, self._round):
            return
        host = self._handoff_host[slot]
        host.fill_(-1)
        host[list(rows)] = torch.tensor(rows, dtype=torch.int32)
        self._handoff_slots[slot].copy_(host, non_blocking=True)
        self._next_positions.copy_(frontiers[: self._num_slots])
        self._staged = (slot, self._round)
        if self._handoff_graphs:
            self._handoff_graphs[slot].replay()
        else:
            self._run_handoff(slot)

    def capture_handoffs(self, pool: GraphPool, stream: torch.cuda.Stream) -> None:
        """Capture each status slot's handoff through ``pool``; later handoffs replay it.
        Each slot's handoff must have run eagerly before, compiling its kernels."""

        self._allocate()
        graphs = [torch.cuda.CUDAGraph() for _ in range(self.depth)]
        with pool.workspace(self._num_slots):
            for slot, graph in enumerate(graphs):
                with pool.capture(graph, stream):
                    self._run_handoff(slot)
        self._handoff_graphs = tuple(graphs)

    def _run_handoff(self, slot: int) -> None:
        # Handoff row ``b`` reads slot ``b``: the canonical inputs are the slot rows.
        self._op.handoff_layers(
            self._layer_table,
            next_positions=self._next_positions,
            slot_indices=self._handoff_slots[slot],
            status=self._status(slot, 0),
        )

    def commit_layer(
        self,
        index: int,
        v: RecurrentVerify,
        lengths: torch.Tensor,
        snapshot: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        """Layer ``index``'s commit; ``snapshot`` is its rows and offsets, if any."""

        size = v.slot_ids.numel()
        workspace = None
        if self._arrivals is not None:
            workspace = {"arrivals": self._arrivals[v.slot, :size]}
        captures = {}
        if snapshot is not None:
            captures = {
                "snapshot_rows": snapshot[0],
                "snapshot_offsets": snapshot[1],
                # Settlement derives captures from the committed lengths: write-only.
                "snapshot_valid": self._snapshot_valid[v.slot, :size],
            }
        self._op.commit(
            *self._layers[index].weights,
            slot_indices=v.slot_ids,
            accepted_lengths=lengths,
            state=self._states[index],
            pending=self._pending(v, index),
            status=self._phase_status(v, index),
            workspace=workspace,
            **captures,
        )

    def end(self, slot: int, size: int, sequence: int) -> BlockStatus:
        """Close round ``sequence``. Its status reaches the host through the returned copies;
        settlement then checks every layer's record."""

        status = self._status(slot, 0, size)
        self._op.end(status=status)
        records, validity = self._records_host[slot], self._validity_host[slot]
        stream = torch.cuda.current_stream(self._device)

        def settle() -> None:
            result = self._op.read_status_copy(
                records, validity[:, :size], expected_sequence=sequence
            )
            if not result.ok:
                raise RuntimeError(
                    f"Gated DeltaNet round {sequence} failed: codes {result.codes}, "
                    f"completions {result.completions}"
                )
            with torch.cuda.stream(stream):
                self._op.settle(status=status, result=result)

        return BlockStatus(((status.records, records), (status.validity, validity)), settle)

    def verify_layer(self, index: int, v: RecurrentVerify, projected: torch.Tensor) -> torch.Tensor:
        """Layer ``index``'s verify of ``projected``; live state stays unchanged."""

        self._allocate()
        size, block_size = v.positions.shape
        out = self._out[:size]
        self._op.verify(
            projected.view(size, block_size, -1),
            *self._layers[index].weights,
            slot_indices=v.slot_ids,
            positions=v.positions,
            state=self._states[index],
            pending=self._pending(v, index),
            out=out,
            status=self._phase_status(v, index),
        )
        return out.view(size * block_size, -1)

    def _status(self, slot: int, index: int, size: int | None = None):
        return self._controls.status(slot, index, batch_size=size)

    def _phase_status(self, v: RecurrentVerify, index: int):
        """Verify's and commit's status: none when the kernel runs them trusted."""

        if self._arrivals is not None:
            return None
        return self._status(v.slot, index, v.slot_ids.numel())

    def _pending(self, v: RecurrentVerify, index: int):
        size = v.slot_ids.numel()
        return self._controls.pending(
            self._pending_raw[index, v.slot, :size], slot=v.slot, layer=index
        )


__all__ = ["BlockStatus", "GatedDeltaNetK2", "GatedDeltaNetLayerReplay"]
