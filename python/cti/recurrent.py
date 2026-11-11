"""Engine-side addressing of layers with fixed per-request state rows.

Per-batch addressing, snapshot restores and exports, and decode-graph slot buffers. The pool
and its spec live in ``cti.kvcache``; each layer family owns its operations. Engine
types are imported for annotations only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from cti.forward import ForwardInputs, ForwardRows, RoundShape
    from cti.kvcache import BoundSlot, RecurrentStatePool, SlotStateSpec, StateSlot
    from cti.memory import GraphPool


class RecurrentLayer(Protocol):
    """What the runtime sees of one recurrent layer."""

    state_spec: SlotStateSpec  # this rank's state in one pool row

    def state_slots(self) -> tuple[StateSlot, ...]:
        """Its ``kvcache.StateLayer`` slots: one slot-row state, ``state_spec``."""
        ...

    def bind_state(self, bound: Mapping[str, BoundSlot]) -> None:
        """Read and write its pool row's views, ``bound`` by slot name, from now on."""
        ...

    def reset_scratch(self) -> None:
        """Forget device scratch tied to its pool row, after that memory came back without
        its contents."""
        ...

    def warm_up(self, max_tokens: int, *, max_sequences: int) -> None:
        """Compile what an extend forward of at most ``max_tokens`` rows and ``max_sequences``
        sequences runs, captures or not, so that none compiles."""
        ...

    def warm_up_training(self, max_tokens: int) -> None:
        """Compile and autotune what a training forward and backward of at most
        ``max_tokens`` tokens runs, so that no training command first launches it."""
        ...

    # Whether the layer's replay takes part in a speculative round's K2 transaction.
    transactional: bool

    def create_replay(
        self, *, max_batch: int, rounds: RoundShape, geometry: RecurrentGeometry
    ) -> LayerReplay:
        """The layer's replay for rounds of ``rounds``' shape over at most ``max_batch``
        rows of a pool of ``geometry``, made once on its bound row; the layer keeps it for
        its verify forwards."""
        ...


@dataclass(frozen=True, slots=True)
class RecurrentGeometry:
    """What a replay needs of the recurrent pool besides its bound row: the request slots,
    the snapshot rows after them, and the device."""

    num_slots: int
    num_snapshot_rows: int
    device: torch.device


class RecurrentPhase(Enum):
    """What a batch's rows after its extend rows do: step their state or verify a block."""

    STEP = "step"
    VERIFY = "verify"


@dataclass(frozen=True, slots=True, eq=False)
class RecurrentVerify:
    """The rows of a verify forward: ``[B]`` int32 slots (-1: an inactive row), ``[B, Q]``
    int64 positions, and the round's status slot."""

    slot: int
    slot_ids: torch.Tensor
    positions: torch.Tensor


class RoundStatus(Protocol):
    """A closed round: copy each ``(device, host)`` pair once it completed, then call
    ``settle`` in round order; it raises on any layer's failure."""

    @property
    def copies(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]: ...

    @property
    def settle(self) -> Callable[[], None]: ...


class RoundTransaction(Protocol):
    """A layer family's part of a round that spans all its layers at once; every replay of
    the family holds the same one."""

    @classmethod
    def device_bytes(
        cls, layers: Sequence[Any], *, num_slots: int, shape: RoundShape, num_snapshot_rows: int
    ) -> int:
        """Bytes of the family's one device allocation over a pool of ``num_slots`` slots and
        ``num_snapshot_rows`` snapshot rows."""
        ...

    def begin(self, slot: int, sequence: int) -> None:
        """Open round ``sequence`` (1, 2, ...) in status slot ``slot`` on the current stream."""
        ...

    def end(self, slot: int, size: int, sequence: int) -> RoundStatus:
        """Close round ``sequence`` of ``size`` rows; return its status copies and settlement."""
        ...

    def capture_handoffs(self, pool: GraphPool, stream: torch.cuda.Stream) -> None:
        """Capture each status slot's handoff of every layer as a CUDA graph that later
        handoffs replay; each slot's handoff must have run eagerly before."""
        ...


class LayerReplay(Protocol):
    """One layer's verify, handoff and commit for fixed blocks; live state stays the pool's."""

    @property
    def transaction(self) -> RoundTransaction: ...

    def verify(self, projected: torch.Tensor, v: RecurrentVerify) -> torch.Tensor:
        """``y [B * Q, values]`` of the verified blocks; live state stays unchanged."""
        ...

    def handoff(self, slot: int, rows: Sequence[int], frontiers: torch.Tensor) -> None:
        """Re-read ``rows`` from the pool at their int64 ``frontiers`` (indexed by slot)."""
        ...

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
        ...


class RecurrentRounds:
    """The K2 transaction of speculative rounds over every transactional layer's replay.

    The pool stays the authoritative state; a row an ordinary forward advanced is stale and
    is handed off before its next round. Round ``n`` (``begin`` .. ``end``, on the execution
    stream) uses status slot ``n % depth``; its settlement runs, in order, before ``n + depth``.
    """

    def __init__(self, replays: Sequence[LayerReplay], *, shape: RoundShape) -> None:
        self.depth = shape.depth
        self._replays = tuple(replays)
        transactions: dict[int, RoundTransaction] = {}
        for replay in self._replays:
            transactions.setdefault(id(replay.transaction), replay.transaction)
        self._transactions = tuple(transactions.values())
        self._rounds = 0
        self._unsettled: set[int] = set()
        self._stale: set[int] = set()
        # Per (status slot, rows), the verify rows ``prepare_verify`` fixed.
        self._verifies: dict[tuple[int, int], RecurrentVerify] = {}

    def prepare_verify(
        self, slot: int, slot_ids: torch.Tensor, positions: torch.Tensor
    ) -> RecurrentMetadata:
        """The metadata of a verify forward in status slot ``slot`` over ``[B]`` slots and
        ``[B, Q]`` positions, at fixed addresses; commits of ``B`` rows in ``slot`` read them."""

        verify = RecurrentVerify(slot, slot_ids, positions)
        self._verifies[slot, len(slot_ids)] = verify
        return RecurrentMetadata(
            slot_ids=slot_ids,
            num_extend_requests=0,
            num_extend_tokens=0,
            phase=RecurrentPhase.VERIFY,
            verify=verify,
        )

    def verify_after(self, extension: RecurrentMetadata, slot: int, size: int) -> RecurrentMetadata:
        """``extension``'s metadata with the ``size`` verify rows of ``slot`` after its rows."""

        return replace(extension, phase=RecurrentPhase.VERIFY, verify=self._verifies[slot, size])

    def mark_stale(self, rows: Iterable[int]) -> None:
        """Rows whose state an ordinary forward advanced."""

        self._stale.update(rows)

    def clear(self) -> None:
        """Forget stale rows, as before the first prompt."""

        self._stale.clear()

    def begin(self, rows: Sequence[int], frontiers: torch.Tensor) -> int:
        """Open the next round over ``rows`` on the current stream, hand off its stale rows
        at their int64 ``frontiers`` (indexed by slot), and return its status slot."""

        slot = self._rounds % self.depth
        if slot in self._unsettled:
            raise RuntimeError(f"recurrent status slot {slot} is not settled")
        self._rounds += 1
        for transaction in self._transactions:
            transaction.begin(slot, self._rounds)
        self._unsettled.add(slot)
        if stale := [row for row in rows if row in self._stale]:
            for replay in self._replays:
                replay.handoff(slot, stale, frontiers)
            self._stale.difference_update(stale)
        return slot

    def commit(
        self,
        slot: int,
        lengths: torch.Tensor,
        snapshot_rows: torch.Tensor | None = None,
        snapshot_offsets: torch.Tensor | None = None,
    ) -> None:
        """Commit each of ``slot``'s verify rows' first ``lengths`` block tokens (int32
        ``[B]``) in every layer; ``snapshot_rows`` as ``LayerReplay.commit``."""

        verify = self._verifies[slot, len(lengths)]
        for replay in self._replays:
            replay.commit(verify, lengths, snapshot_rows, snapshot_offsets)

    def end(self, slot: int, size: int) -> RoundStatus:
        """Close the round of ``size`` rows; its settlement frees ``slot``."""

        statuses = [transaction.end(slot, size, self._rounds) for transaction in self._transactions]

        def settle() -> None:
            for status in statuses:
                status.settle()
            self._unsettled.discard(slot)

        copies = tuple(pair for status in statuses for pair in status.copies)
        return _RoundStatus(copies, settle)

    def capture_handoffs(self, pool: GraphPool, stream: torch.cuda.Stream) -> None:
        """Capture every family's handoffs; later handoffs replay them."""

        for transaction in self._transactions:
            transaction.capture_handoffs(pool, stream)


@dataclass(frozen=True, slots=True)
class _RoundStatus:
    copies: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    settle: Callable[[], None]


@dataclass(frozen=True, slots=True)
class RecurrentMetadata:
    """Recurrent-state addressing of one execution batch, shared by every layer.

    Extend entries precede decode entries, so the first ``num_extend_tokens`` rows of
    the batch belong to the first ``num_extend_requests`` entries. In ``VERIFY`` phase the
    batch holds extend entries only and the rows after theirs verify ``verify``'s blocks.
    """

    slot_ids: torch.Tensor  # int32 [padded_size]; -1 for graph padding
    num_extend_requests: int
    num_extend_tokens: int
    restore: torch.Tensor | None = None  # int64 [2, copies]: sources, targets
    # Per extend request, the snapshot row its capture goes to (-1: none): the state after
    # the most whole multiples of the layers' export granularity among its extend tokens.
    capture_rows: torch.Tensor | None = None  # int32 [num_extend_requests]
    cu_seqlens: torch.Tensor | None = None  # int32 [num_extend_requests + 1]
    has_initial_state: torch.Tensor | None = None  # bool [num_extend_requests]
    # Decode entries' (slot, snapshot row) pairs: the slot's row is copied there after the step.
    exports: tuple[tuple[int, int], ...] = ()
    phase: RecurrentPhase = RecurrentPhase.STEP
    verify: RecurrentVerify | None = None


class RecurrentBackend:
    """Build ``RecurrentMetadata`` per batch and own the slot buffer decode graphs read."""

    def __init__(self, pool: RecurrentStatePool, *, device: torch.device) -> None:
        self._device = device
        self._pool = pool
        self._capture_slot_ids: torch.Tensor | None = None
        self._capture_batch_sizes: frozenset[int] = frozenset()

    def prepare_metadata(self, rows: ForwardRows) -> RecurrentMetadata:
        """Build the batch's state addressing on the preparation stream."""

        size, num_extend = rows.size, rows.num_extend
        slot_ids = [slot if index < size else -1 for index, slot in enumerate(rows.slots)]
        slot_ids_gpu = _to_device(slot_ids, torch.int32, self._device)
        exports = tuple(
            (rows.slots[index], capture.row)
            for index in range(num_extend, size)
            for capture in rows.captures[index]
        )
        if not num_extend:
            return RecurrentMetadata(
                slot_ids=slot_ids_gpu,
                num_extend_requests=0,
                num_extend_tokens=0,
                exports=exports,
            )

        extends = range(num_extend)
        extend_lengths = [rows.extend_len(index) for index in extends]
        return RecurrentMetadata(
            slot_ids=slot_ids_gpu,
            restore=self._row_copy(list(rows.restores)),
            capture_rows=self._capture_rows(rows),
            num_extend_requests=num_extend,
            num_extend_tokens=sum(extend_lengths),
            cu_seqlens=_to_device([0, *accumulate(extend_lengths)], torch.int32, self._device),
            has_initial_state=_to_device(
                [rows.kv_prefix_lens[index] > 0 for index in extends],
                torch.bool,
                self._device,
            ),
            exports=exports,
        )

    def prepare_inputs(self, rows: ForwardRows) -> dict[str, RecurrentMetadata]:
        """``rows``' metadata, prepared on the preparation stream, as the forward input it
        fills."""

        return {"recurrent_metadata": self.prepare_metadata(rows)}

    def _row_copy(self, pairs: list[tuple[int, int]]) -> torch.Tensor | None:
        if not pairs:
            return None
        sources, targets = zip(*pairs, strict=True)
        return _to_device([list(sources), list(targets)], torch.int64, self._device)

    def _capture_rows(self, rows: ForwardRows) -> torch.Tensor | None:
        """Each extension's capture row, -1 without one; None when none captures."""
        captures = rows.captures[: rows.num_extend]
        if not any(captures):
            return None
        capture_rows = [extension[0].row if extension else -1 for extension in captures]
        return _to_device(capture_rows, torch.int32, self._device)

    def restore(self, batch: ForwardInputs) -> None:
        """Restore snapshot rows before model execution on the caller's stream."""
        indices = batch.recurrent_metadata.restore
        if indices is not None:
            self._pool.copy_rows(indices)

    def export(self, batch: ForwardInputs) -> None:
        """Copy each exporting decode entry's advanced row to its snapshot row after model
        execution (eager or replayed), on the caller's stream, in one launch."""
        exports = batch.recurrent_metadata.exports
        if exports:
            sources = [slot for slot, _ in exports]
            targets = [row for _, row in exports]
            self._pool.copy_rows(_to_device([sources, targets], torch.int64, self._device))

    # ============================ Decode CUDA graphs ============================

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        """Allocate the static slot buffer that captured decode graphs read."""

        _ = max_seq_len
        if self._capture_slot_ids is not None:
            raise RuntimeError("recurrent graph capture is already initialized")
        self._capture_batch_sizes = frozenset(bs_list)
        self._capture_slot_ids = torch.full(
            (max(bs_list),),
            -1,
            dtype=torch.int32,
            device=self._device,
        )

    def capture_metadata(self, rows: ForwardRows) -> RecurrentMetadata:
        """Capture with every row padding, so no pool row changes."""

        capture_slot_ids = self._require_capture_batch_size(rows.size)
        capture_slot_ids.fill_(-1)
        return RecurrentMetadata(
            slot_ids=capture_slot_ids[: rows.size],
            num_extend_requests=0,
            num_extend_tokens=0,
        )

    def prepare_for_capture(self, rows: ForwardRows, inputs: ForwardInputs) -> ForwardInputs:
        return replace(inputs, recurrent_metadata=self.capture_metadata(rows))

    def prepare_for_replay(self, batch: ForwardInputs, bs: int) -> None:
        """Copy the batch's ``bs`` slots, padding included, into the captured buffer."""

        capture_slot_ids = self._require_capture_batch_size(bs)
        metadata = batch.recurrent_metadata
        if metadata.num_extend_requests:
            raise RuntimeError("decode CUDA graphs replay only decode batches")
        capture_slot_ids[:bs].copy_(metadata.slot_ids)

    def _require_capture_batch_size(self, batch_size: int) -> torch.Tensor:
        capture_slot_ids = self._capture_slot_ids
        if capture_slot_ids is None or batch_size not in self._capture_batch_sizes:
            raise RuntimeError(f"no recurrent decode graph for batch size {batch_size}")
        return capture_slot_ids


def _to_device(
    values: list[int] | list[bool] | list[list[int]],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # Pinning serves only the asynchronous copy to a CUDA pool; CPU pools copy nothing.
    host = torch.tensor(values, dtype=dtype, pin_memory=device.type == "cuda")
    return host.to(device, non_blocking=True)


__all__ = [
    "LayerReplay",
    "RecurrentBackend",
    "RecurrentGeometry",
    "RecurrentLayer",
    "RecurrentMetadata",
    "RecurrentPhase",
    "RecurrentRounds",
    "RecurrentVerify",
    "RoundStatus",
    "RoundTransaction",
]
