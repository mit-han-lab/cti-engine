"""Recurrent-state snapshots on one grid: where a forward exports, which rows it writes, and
when the rows pass to the radix."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cti.inference._step import QueryKind, Row, StateCapture
from cti.kvcache import (
    BaseCacheHandle,
    RestorePlan,
    StateRowIndex,
    next_snapshot_position,
)

if TYPE_CHECKING:
    from ._cache import CacheUse
    from ._scheduling.table import RequestSlot


def export_points(hit_limit: int, branch: int, alignment: int, prompt_len: int) -> frozenset[int]:
    """The grid points a prompt's forwards export at: the deepest one a later hit can restore
    (at most ``hit_limit``, ``BasePrefixCache.hit_limit``), ``branch``, its KV match on the
    grid, where the prompt leaves a cached one, and its end when that is on the grid, which
    its final forward captures: no repeat restores there, but extensions do, and the KV it
    publishes shows later prompts their branch. Chunk boundaries add none."""

    end = {prompt_len} if prompt_len % alignment == 0 else set()
    return frozenset({branch, hit_limit // alignment * alignment, *end} - {0})


def plan_exports(
    start: int, end: int, points: Collection[int], granularity: int
) -> tuple[int, ...]:
    """Where a forward over ``[start, end)`` exports: the one point it captures, the most
    whole multiples of ``granularity`` tokens past ``start``, if that is one of ``points``."""

    point = start + (end - start) // granularity * granularity
    return (point,) if point > start and point in points else ()


@dataclass
class _Slot:
    points: frozenset[int]  # ``export_points``: the prompt forward capturing one exports it
    branch: int  # the KV match on the grid: a prompt forward crossing it stops there
    restore: BaseCacheHandle | None  # its row locked until the forward restoring it settles
    restore_row: int | None
    # The decode snapshot row (decode steps and speculative rounds write it), the position its
    # last write holds and whether that committed.
    row: int | None = None
    position: int = 0
    committed: bool = False


class StateSnapshots:
    """Plan each forward's exports and where prompt forwards end, lease the rows without
    waiting, and hand the rows to the radix once they hold settled state.

    A prompt forward exports at most once, after the most whole multiples of
    ``granularity`` tokens it runs (``plan_exports``), and only at an ``export_points``
    position. So that those land on the grid, every prompt forward starts on it: one short
    of its prompt's end ends on it (``extension_end``), and one crossing the branch point
    without capturing it stops there (``export_stop``). A final forward whose capture misses
    the deepest point leaves it unexported, as SGLang does; one ending on the grid exports
    its end, as SGLang keeps its final state.
    A row comes from the free list, else from the least recently used unlocked snapshot;
    without one the export is skipped. A prompt forward's row passes to the radix when it
    settles. Decode steps, or speculative rounds, overwrite one row leased for the request,
    which passes to the radix when the slot closes if the write it holds last committed:
    one stream orders the writes, and nothing reads the row before.
    """

    def __init__(self, rows: StateRowIndex | None, granularity: int) -> None:
        self._rows = rows
        self._granularity = granularity
        self._slots: dict[RequestSlot, _Slot] = {}

    def open(
        self,
        slot: RequestSlot,
        plan: RestorePlan,
        handle: BaseCacheHandle,
        hit_limit: int,
        prompt_len: int,
    ) -> None:
        """Track an admission of a ``prompt_len``-token prompt whose hit ``plan`` was acquired
        as ``handle``, which holds its restore row locked; ``hit_limit`` bounds its hits."""

        if self._rows is None:
            return
        restore = handle if plan.row is not None else None
        grid = self._rows.grid
        branch = plan.kv_match_len // grid * grid
        points = export_points(hit_limit, branch, grid, prompt_len)
        self._slots[slot] = _Slot(points, branch, restore, plan.row)

    @property
    def grid(self) -> int:
        """The grid every prompt forward starts on: the snapshot grid, 1 without rows."""

        return 1 if self._rows is None else self._rows.grid

    def positions(self, slot: RequestSlot, start: int, end: int) -> tuple[int, ...]:
        """Where a forward of ``slot`` over ``[start, end)`` exports when it gets a row."""
        state = self._slots.get(slot)
        if state is None:
            return ()
        return plan_exports(start, end, state.points, self._granularity)

    def extension_end(self, slot: RequestSlot, end: int, prompt_end: int) -> int:
        """Where a prompt forward of ``slot`` toward ``end`` ends: there, or short of
        ``prompt_end`` on the grid point at or before it, its start when nothing is left."""

        if slot not in self._slots or end == prompt_end:
            return end
        return end - end % self._rows.grid

    def export_stop(self, slot: RequestSlot, start: int, end: int) -> int | None:
        """The branch point, when it lies inside a prompt forward of ``slot`` over
        ``[start, end)`` that would not capture it: ending there, it captures it."""

        state = self._slots.get(slot)
        if state is None or not start < state.branch < end:
            return None
        return None if state.branch in self.positions(slot, start, end) else state.branch

    def plan(self, slot: RequestSlot, entry: Row) -> tuple[int | None, tuple[StateCapture, ...]]:
        """The row the slot's next forward restores first and the exports it leases."""

        state = self._slots.get(slot)
        if state is None:
            return None, ()
        if entry.query_kind is QueryKind.DECODE:
            return None, self._decode_exports(state, entry.sequence_len)
        restore = state.restore
        # Only the admission's first forward starts at the restored position.
        first = restore is not None and entry.kv_prefix_len == restore.cached_len
        positions = self.positions(slot, entry.kv_prefix_len, entry.sequence_len)
        leases = [(position, self._rows.lease()) for position in positions]
        exports = tuple(StateCapture(p, row) for p, row in leases if row is not None)
        return (state.restore_row if first else None), exports

    def _decode_exports(self, state: _Slot, end: int) -> tuple[StateCapture, ...]:
        if end % self._rows.grid:
            return ()
        if state.row is None:
            state.row = self._rows.lease()
            if state.row is None:
                return ()
        state.position, state.committed = end, False
        return (StateCapture(end, state.row),)

    def round_row(self, slot: RequestSlot) -> int | None:
        """The row a speculative round of ``slot`` captures into, at the first grid point past
        its frontier that it commits: the slot's decode row, leased on first use."""

        state = self._slots.get(slot)
        if state is None:
            return None
        if state.row is None:
            state.row = self._rows.lease()
        return state.row

    def settle_round(
        self, slot: RequestSlot, frontier: int, device_count: int, *, committed: bool
    ) -> None:
        """Settle a round that captured into the slot's row from ``frontier``, where the
        device committed ``device_count`` positions; ``committed`` when the host keeps them.
        A discarded round's capture voids the row."""

        state = self._slots[slot]
        position = next_snapshot_position(frontier, self._rows.grid)
        if position <= frontier + device_count:
            state.position, state.committed = position, committed

    def settle(
        self, use: CacheUse, *, committed: bool, publish: bool = False
    ) -> tuple[StateCapture, ...]:
        """Unlock a restore source the forward read and record whether a decode export
        committed. Return a committed prompt forward's exports when ``publish``, the caller
        handing their rows to the radix; otherwise free them."""

        state = self._slots.get(use.slot)
        if use.restore_row is not None:
            self._rows.unlock(state.restore)
            state.restore = None
        exports = use.exports
        if state is not None and exports and exports[0].row == state.row:
            if exports[0].position == state.position:  # the row's last write
                state.committed = committed
            return ()
        if committed and publish:
            return exports
        for export in exports:
            self._rows.free(export.row)
        return ()

    def close(self, slot: RequestSlot) -> StateCapture | None:
        """Retire the slot; return its decode snapshot when the last write committed, the
        caller handing the row to the radix, else free the row."""

        state = self._slots.pop(slot, None)
        if state is None:
            return None
        if state.restore is not None:
            self._rows.unlock(state.restore)
        if state.row is not None and not state.committed:
            self._rows.free(state.row)
        return StateCapture(state.position, state.row) if state.committed else None

    def check_integrity(self) -> None:
        if self._slots:
            raise RuntimeError("snapshot rows are not quiescent")
