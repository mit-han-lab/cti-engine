# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/scheduler/cache.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch

from cti.inference._step import PageMap, QueryKind, Row, StateCapture
from cti.kvcache import BaseCacheHandle, create_prefix_cache
from cti.utils import div_ceil

from ._prefix_state import StateSnapshots, plan_exports

if TYPE_CHECKING:
    from cti.kvcache import StateCapacity

    from ._scheduling.table import RequestSlot


@dataclass(frozen=True, slots=True, eq=False)
class CacheUse:
    """One submitted use of a slot's cache below the absolute ``end``. A forward's use
    commits from ``start``, restoring ``restore_row`` first and exporting ``exports``; a
    DFlash round's (``start`` ``None``) learns its frontier when it settles and captures
    into ``snapshot_row`` (``StateSnapshots.round_row``), if any."""

    slot: RequestSlot
    end: int
    start: int | None = None
    restore_row: int | None = None
    exports: tuple[StateCapture, ...] = ()
    snapshot_row: int | None = None


@dataclass(frozen=True, slots=True)
class CacheOccupancy:
    """KV pages and snapshot rows: used by requests, cached for reuse (evictable), or free."""

    num_pages: int
    num_used_pages: int
    num_cached_pages: int
    num_snapshot_rows: int
    num_used_snapshot_rows: int
    num_cached_snapshot_rows: int

    def __str__(self) -> str:
        text = (
            f"KV pages {self.num_used_pages} used, {self.num_cached_pages} cached, "
            f"{self.num_pages} total"
        )
        if self.num_snapshot_rows:
            text += (
                f"; snapshot rows {self.num_used_snapshot_rows} used, "
                f"{self.num_cached_snapshot_rows} cached, {self.num_snapshot_rows} total"
            )
        return text


@dataclass(slots=True)
class _DeferredPageRelease:
    pages: np.ndarray
    # The slot's uses in flight when the pages were rewired, whose forwards may read them.
    blocking_uses: set[CacheUse]


@dataclass(slots=True)
class _SlotCacheState:
    prefix_handle: BaseCacheHandle
    valid_len: int
    reserved_len: int
    deferred_page_releases: list[_DeferredPageRelease] = field(default_factory=list)
    pending_cache_mapping_ranges: list[tuple[int, int]] = field(default_factory=list)
    # Uses in submission order; they settle first in, first out.
    uses: list[CacheUse] = field(default_factory=list)


class _FreePages:
    """The free page IDs, oldest first, in a ring over one preallocated int32 buffer:
    taking pages copies the oldest out and returning pages writes them after the newest,
    so a call costs the pages it moves, never the whole list."""

    def __init__(self, num_pages: int) -> None:
        self._ring = np.arange(num_pages, dtype=np.int32)
        self._head = 0  # the oldest free page's index in the ring
        self._count = num_pages

    def __len__(self) -> int:
        return self._count

    def take(self, count: int) -> np.ndarray:
        """The ``count`` oldest free page IDs, removed."""

        if not 0 <= count <= self._count:
            raise RuntimeError(f"cannot take {count} of {self._count} free pages")
        taken = self._span(count)
        self._head = self._wrap(self._head + count)
        self._count -= count
        return taken

    def put(self, pages: np.ndarray) -> None:
        """Return ``pages`` after the newest free page."""

        count, capacity = len(pages), len(self._ring)
        if self._count + count > capacity:
            raise RuntimeError(
                f"returning {count} pages to {self._count} free exceeds the {capacity} pages"
            )
        tail = self._wrap(self._head + self._count)
        first = min(count, capacity - tail)
        self._ring[tail : tail + first] = pages[:first]
        self._ring[: count - first] = pages[first:]
        self._count += count

    def reset(self) -> None:
        """Every page free again, in its initial order."""

        self._ring[:] = np.arange(len(self._ring), dtype=np.int32)
        self._head, self._count = 0, len(self._ring)

    def ids(self) -> np.ndarray:
        """A copy of the free page IDs, oldest first."""

        return self._span(self._count)

    def _span(self, count: int) -> np.ndarray:
        """A copy of the ``count`` oldest free page IDs."""

        start, end = self._head, self._head + count
        if end <= len(self._ring):
            return self._ring[start:end].copy()
        return np.concatenate((self._ring[start:], self._ring[: end - len(self._ring)]))

    def _wrap(self, index: int) -> int:
        return index - len(self._ring) if index >= len(self._ring) else index


class CacheManager:
    """Own per-slot cache state, prefix visibility, and page allocation.

    ``page_table[request_index, page_index]`` stores a page ID, -1 where unmapped; the
    per-step upkeep writes it through a numpy view (``_table``), free of torch dispatch.
    ``_free_pages`` holds the ID of each available page.
    A slot must be opened before its uses and closed before its table lifetime ends.
    A slot holds forward uses, then up to ``MAX_ROUNDS`` round uses; a round may follow
    the last forward before it settles. Uses settle in submission order.
    Publishing rewires a slot's duplicate prefix pages to the cached ones; the duplicates
    return once every use in flight at publication has settled. A slot with rounds takes
    no forward use, so its next round applies the rewiring on the execution stream,
    behind the rounds that still read the duplicates.
    A failure part-way through a transition leaves the manager inconsistent: it ends the
    rank's process (``EngineCore``), so nothing here survives it.
    """

    MAX_ROUNDS = 2

    def __init__(
        self,
        capacity: StateCapacity,
        *,
        prefix: Literal["naive", "radix"],
        checks: bool,
    ) -> None:
        """``capacity``'s usable pages and request slots; the ``prefix`` cache, a radix one
        reusing prefixes and restoring recurrent state from its snapshot rows; ``checks``
        adds the expensive scans and the use-order invariants."""
        num_pages, page_size = capacity.num_pages, capacity.page_size
        self._checks = checks
        self.page_table = torch.full(
            (capacity.num_slots, capacity.table_width), -1, dtype=torch.int32
        )
        self._table = self.page_table.numpy()
        self._free_pages = _FreePages(num_pages)
        self.prefix_cache = create_prefix_cache(
            capacity, kind=prefix, device=self.page_table.device
        )
        self._export_granularity = capacity.export_granularity
        self._snapshots = StateSnapshots(self.prefix_cache.rows, self._export_granularity)
        self.num_pages = num_pages
        self.page_size = page_size
        self._slot_states: dict[RequestSlot, _SlotCacheState] = {}

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self._free_pages) * self.page_size

    @property
    def occupancy(self) -> CacheOccupancy:
        cached_pages = self.prefix_cache.size_info.evictable_size // self.page_size
        num_rows = cached_rows = free_rows = 0
        if (rows := self.prefix_cache.rows) is not None:
            info = rows.info
            num_rows, cached_rows, free_rows = len(rows), info.evictable, info.free
        return CacheOccupancy(
            num_pages=self.num_pages,
            num_used_pages=self.num_pages - len(self._free_pages) - cached_pages,
            num_cached_pages=cached_pages,
            num_snapshot_rows=num_rows,
            num_used_snapshot_rows=num_rows - free_rows - cached_rows,
            num_cached_snapshot_rows=cached_rows,
        )

    def try_open_slot(
        self,
        slot: RequestSlot,
        prompt_token_ids: torch.Tensor,
        max_output_tokens: int,
        *,
        reserved_size: int,
        lookahead_tokens: int = 0,
        prefix: bool = True,
    ) -> int | None:
        """Open cache state for an allocated slot and return its cached prompt length, or
        ``None`` if it cannot fit.

        ``lookahead_tokens`` is the KV the request may write past its last output;
        without ``prefix`` the prompt restores nothing from the prefix cache.
        """
        plan = self.prefix_cache.plan_restore(prompt_token_ids if prefix else prompt_token_ids[:0])
        required_size = self._required_size(
            len(prompt_token_ids) - plan.cached_len + max_output_tokens + lookahead_tokens,
            reserved_size,
        )
        # Acquiring the hit removes its unprotected pages from evictable capacity.
        if required_size > self.available_size - plan.unprotected_len:
            return None
        prefix_handle = self.prefix_cache.acquire(plan)
        hit_limit = self.prefix_cache.hit_limit(len(prompt_token_ids))
        self._snapshots.open(slot, plan, prefix_handle, hit_limit, len(prompt_token_ids))
        self._initialize_slot_state(slot, prefix_handle)
        return plan.cached_len

    def growth_size(self, slot: RequestSlot, end: int) -> int:
        """The cache ``slot`` takes to reserve below ``end``."""

        reserved = self._require_slot_state(slot).reserved_len
        pages = div_ceil(end, self.page_size) - div_ceil(reserved, self.page_size)
        return max(pages, 0) * self.page_size

    def cached_prefix_len(self, prompt_token_ids: torch.Tensor) -> int:
        """The prompt tokens a hit would reuse if the prompt were admitted now. Like the
        hit, the query refreshes the cache's LRU (``BasePrefixCache.plan_restore``)."""
        return self.prefix_cache.plan_restore(prompt_token_ids).cached_len

    def restorable_positions(self, slot: RequestSlot, start: int, end: int) -> tuple[int, ...]:
        """The prefix lengths a forward of ``slot`` over prompt positions ``[start, end)``
        makes restorable once it settles (``BasePrefixCache.restorable``)."""

        return self.prefix_cache.restorable(end, self.export_positions(slot, start, end))

    def published_positions(self, use: CacheUse) -> tuple[int, ...]:
        """The prefix lengths ``use``'s forward makes restorable once it settles."""

        exports = tuple(export.position for export in use.exports)
        return self.prefix_cache.restorable(use.end, exports)

    def export_positions(self, slot: RequestSlot, start: int, end: int) -> tuple[int, ...]:
        """Where a forward of ``slot`` over prompt positions ``[start, end)`` exports if
        reserved now: nowhere while no row is free or reclaimable."""
        rows = self.prefix_cache.rows
        if rows is None or not rows.can_lease:
            return ()
        return self._snapshots.positions(slot, start, end)

    @property
    def prompt_grid(self) -> int:
        """The grid prompt forwards start on, but after a cut before an image: the snapshot
        grid with snapshot rows, else 1 (``StateSnapshots``)."""

        return self._snapshots.grid

    def extension_end(self, slot: RequestSlot, end: int, prompt_end: int) -> int:
        """Where a prompt forward of ``slot`` toward ``end`` ends, its prompt ending at
        ``prompt_end`` (``StateSnapshots.extension_end``)."""

        return self._snapshots.extension_end(slot, end, prompt_end)

    def export_stop(self, slot: RequestSlot, start: int, end: int) -> int | None:
        """Where a prompt forward of ``slot`` over ``[start, end)`` stops short to export
        (``StateSnapshots.export_stop``), if anywhere: nowhere while no row is free or
        reclaimable."""

        rows = self.prefix_cache.rows
        if rows is None or not rows.can_lease:
            return None
        return self._snapshots.export_stop(slot, start, end)

    def reserve(
        self, slots: Sequence[RequestSlot], entries: Sequence[Row]
    ) -> tuple[tuple[CacheUse, ...], PageMap | None]:
        """Reserve each forward's exact frontiers, with the snapshot rows it restores and
        exports, and return the uses and their mapping snapshot."""
        if self._checks:
            for slot, entry in zip(slots, entries, strict=True):
                state = self._require_slot_state(slot)
                if entry.kv_prefix_len != state.reserved_len or any(
                    use.start is None for use in state.uses
                ):
                    raise RuntimeError(f"forward of {slot} does not continue its reserved cache")
        uses = tuple(
            CacheUse(
                slot,
                entry.sequence_len,
                entry.kv_prefix_len,
                *self._snapshots.plan(slot, entry),
            )
            for slot, entry in zip(slots, entries, strict=True)
        )
        if self._checks:
            # A prefill captures only after its most whole multiples of the granularity.
            for use, entry in zip(uses, entries, strict=True):
                positions = tuple(export.position for export in use.exports)
                if entry.query_kind is QueryKind.EXTEND and positions != plan_exports(
                    use.start, use.end, positions, self._export_granularity
                ):
                    raise RuntimeError(
                        f"prompt forward of {use.slot} exports off its capture point"
                    )
        return uses, self._grow(uses)

    def reserve_rounds(
        self, slots: Sequence[RequestSlot], ends: Sequence[int]
    ) -> tuple[tuple[CacheUse, ...], PageMap | None]:
        """Reserve pages below each absolute round ``end`` and return the uses and their
        mapping snapshot. Valid KV does not move: a slot keeps its frontier until the round
        settles."""
        if self._checks:
            for slot in slots:
                rounds = sum(use.start is None for use in self._require_slot_state(slot).uses)
                if rounds >= self.MAX_ROUNDS:
                    raise RuntimeError(f"{slot} already holds {self.MAX_ROUNDS} rounds")
        uses = tuple(
            CacheUse(slot, end, snapshot_row=self._snapshots.round_row(slot))
            for slot, end in zip(slots, ends, strict=True)
        )
        return uses, self._grow(uses)

    def _grow(self, uses: Sequence[CacheUse]) -> PageMap | None:
        """Map pages below each use's end, record the uses, and return the mappings."""
        states = [self._require_slot_state(use.slot) for use in uses]
        page_count_ranges: list[tuple[RequestSlot, int, int]] = []
        num_needed_pages = 0
        for use, state in zip(uses, states, strict=True):
            source_pages = div_ceil(state.reserved_len, self.page_size)
            target_pages = div_ceil(use.end, self.page_size)
            if target_pages > source_pages:
                num_needed_pages += target_pages - source_pages
                page_count_ranges.append((use.slot, source_pages, target_pages))
        if num_needed_pages:
            self._write_pages(self._allocate_pages(num_needed_pages), page_count_ranges)
        for use, state in zip(uses, states, strict=True):
            state.reserved_len = max(state.reserved_len, use.end)
            state.uses.append(use)
        return self._consume_cache_mapping_updates([use.slot for use in uses], states)

    def _consume_cache_mapping_updates(
        self,
        slots: Sequence[RequestSlot],
        slot_states: Sequence[_SlotCacheState],
    ) -> PageMap | None:
        """Consume the slots' pending mappings after an accepted transition."""
        spans = [
            (slot.index, start_page, end_page)
            for slot, state in zip(slots, slot_states, strict=True)
            for start_page, end_page in state.pending_cache_mapping_ranges
        ]
        if not spans:
            return None
        for state in slot_states:
            state.pending_cache_mapping_ranges.clear()
        return page_map_of(self._table, spans)

    def settle(
        self,
        use: CacheUse,
        committed: int,
        *,
        prefix_ids: torch.Tensor | None = None,
        record: tuple[int, int] | None = None,
    ) -> None:
        """Retire a fenced use, making ``committed`` more positions valid (0 for a discarded
        or finished use).

        A forward's ``prefix_ids`` (the CPU tokens up to its end) publish its KV and the
        snapshots it exported; without them the exports are freed. A round with a snapshot
        row passes ``record``: the frontier and positions its device record committed.
        Pages past both the new frontier and every use still pending return, as do
        duplicate prefix pages whose last use in flight at publication settles; an older
        use never shrinks a successor's capacity.
        """
        state = self._require_slot_state(use.slot)
        if self._checks and (
            state.uses[0] is not use or committed and use.start not in (None, state.valid_len)
        ):
            raise RuntimeError(f"use of {use.slot} settles out of order")
        if prefix_ids is not None:
            self._defer_duplicate_pages(state, self._publish_prefix(state, use.slot, prefix_ids))
        exports: tuple[StateCapture, ...] = ()
        if use.snapshot_row is not None:
            frontier, device_count = record
            self._snapshots.settle_round(
                use.slot, frontier, device_count, committed=committed == device_count
            )
        elif use.start is not None:
            publish = prefix_ids is not None
            exports = self._snapshots.settle(use, committed=committed > 0, publish=publish)
        for export in exports:
            duplicates = self._publish_prefix(
                state, use.slot, prefix_ids[: export.position], export.row
            )
            self._defer_duplicate_pages(state, duplicates)
        state.valid_len += committed
        state.uses.remove(use)
        staged_pages = self._retire_use(state, use)
        retained = max([state.valid_len, *(pending.end for pending in state.uses)])
        first_page = div_ceil(retained, self.page_size)
        last_page = div_ceil(state.reserved_len, self.page_size)
        if last_page > first_page:
            staged_pages.append(self._table[use.slot.index, first_page:last_page])
        state.reserved_len = retained
        self._release_pages(*staged_pages)

    def close_slot(self, slot: RequestSlot, *, prefix_ids: torch.Tensor) -> None:
        """Terminally publish a settled slot's valid prefix and decode snapshot, and release
        its private tail."""
        state = self._require_slot_state(slot)
        if state.valid_len > state.prefix_handle.cached_len:
            duplicate_pages = self._publish_prefix(
                state,
                slot,
                prefix_ids[: state.valid_len],
            )
            self._release_pages(duplicate_pages)
        if (export := self._snapshots.close(slot)) is not None:
            ids = prefix_ids[: export.position]
            self._release_pages(self._publish_prefix(state, slot, ids, export.row))
        private_tail_pages = self._table[
            slot.index,
            state.prefix_handle.cached_len // self.page_size : div_ceil(
                state.reserved_len, self.page_size
            ),
        ]
        self.prefix_cache.unlock(state.prefix_handle)
        self._release_pages(private_tail_pages)
        del self._slot_states[slot]

    def in_use(self, slot: RequestSlot) -> bool:
        """Whether a live slot still has submitted cache use."""
        return bool(self._require_slot_state(slot).uses)

    def invalidate_all(self) -> None:
        """Invalidate quiescent logical KV state without scrubbing stale table rows.

        Every cache slot must already be closed. Future uses overwrite their complete live
        page ranges before those mappings become visible.
        """
        self.audit()
        self.prefix_cache.reset()
        self._free_pages.reset()
        self.audit()

    def warm_up(self) -> None:
        """Load what prefix matching needs before the first request."""
        self.prefix_cache.warm_up()

    def audit(self) -> None:
        """``check_integrity`` when the manager runs its checks; nothing on the serving path."""

        if self._checks:
            self.check_integrity()

    def check_integrity(self) -> None:
        """Audit a quiescent manager: raise on provider or allocation corruption."""
        if self._slot_states:
            raise RuntimeError("integrity check requires all cache slots to be closed")
        self._snapshots.check_integrity()
        if self._checks:
            self.prefix_cache.check_integrity()
        size_info = self.prefix_cache.size_info
        if (
            size_info.evictable_size < 0
            or size_info.protected_size != 0
            or size_info.evictable_size % self.page_size != 0
        ):
            raise RuntimeError("prefix cache size is incompatible with quiescence")
        cache_pages = size_info.evictable_size // self.page_size
        if len(self._free_pages) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self._free_pages)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self._checks and len(self._free_pages):
            free = self._free_pages.ids()
            if int(free.min()) < 0 or int(free.max()) >= self.num_pages:
                raise RuntimeError("CacheManager integrity check failed: free page out of range")

    # ========================== Prefix and slot state ==========================

    def _required_size(self, tokens: int, reserved_size: int) -> int:
        """Cache an opening slot needs for ``tokens`` more positions, past the
        ``reserved_size`` admissions planned before it."""

        return div_ceil(tokens, self.page_size) * self.page_size + reserved_size

    def _initialize_slot_state(
        self,
        slot: RequestSlot,
        prefix_handle: BaseCacheHandle,
    ) -> None:
        cached_len = prefix_handle.cached_len
        if cached_len > 0:
            self._table[slot.index, : cached_len // self.page_size] = (
                prefix_handle.get_matched_indices().numpy()
            )
        state = _SlotCacheState(
            prefix_handle=prefix_handle,
            valid_len=cached_len,
            reserved_len=cached_len,
        )
        self._slot_states[slot] = state
        self._record_cache_mapping_range(state, 0, cached_len // self.page_size)

    # ============================ Use settlement =============================

    def _publish_prefix(
        self,
        state: _SlotCacheState,
        slot: RequestSlot,
        prefix_ids: torch.Tensor,
        row: int | None = None,
    ) -> np.ndarray:
        """Publish the slot's KV of ``prefix_ids`` and, with ``row``, the snapshot row of the
        state after them; return the slot's pages the cache already held."""

        prefix_len = len(prefix_ids)
        pages = self._table[slot.index, : prefix_len // self.page_size]
        existing_prefix_len, new_prefix_handle = self.prefix_cache.publish(
            prefix_ids, torch.from_numpy(pages), row
        )
        if row is not None and prefix_len <= state.prefix_handle.cached_len:
            return pages[:0]
        return self._adopt_prefix_handle(state, slot, existing_prefix_len, new_prefix_handle)

    def _adopt_prefix_handle(
        self,
        state: _SlotCacheState,
        slot: RequestSlot,
        existing_prefix_len: int,
        new_prefix_handle: BaseCacheHandle,
    ) -> np.ndarray:
        old_handle_len = state.prefix_handle.cached_len
        self.prefix_cache.lock(new_prefix_handle)
        # Both lengths are whole pages.
        first, end = old_handle_len // self.page_size, existing_prefix_len // self.page_size
        if end > first:
            canonical_pages = new_prefix_handle.get_matched_indices().numpy()
            duplicate_pages = self._table[slot.index, first:end].copy()
            self._table[slot.index, first:end] = canonical_pages[first:end]
            self._record_cache_mapping_range(state, first, end)
        else:
            duplicate_pages = self._table[slot.index, :0]
        self.prefix_cache.unlock(state.prefix_handle)
        state.prefix_handle = new_prefix_handle
        return duplicate_pages

    @staticmethod
    def _defer_duplicate_pages(state: _SlotCacheState, pages: np.ndarray) -> None:
        if len(pages):
            state.deferred_page_releases.append(_DeferredPageRelease(pages, set(state.uses)))

    @staticmethod
    def _retire_use(state: _SlotCacheState, use: CacheUse) -> list[np.ndarray]:
        """The deferred pages ``use`` was the last in-flight reader of."""
        staged: list[np.ndarray] = []
        remaining: list[_DeferredPageRelease] = []
        for release in state.deferred_page_releases:
            release.blocking_uses.discard(use)
            if release.blocking_uses:
                remaining.append(release)
            else:
                staged.append(release.pages)
        state.deferred_page_releases = remaining
        return staged

    # =================== Page allocation and table mapping ====================

    def _allocate_pages(self, num_pages: int) -> np.ndarray:
        if num_pages > (num_free_pages := len(self._free_pages)):
            evicted = self.prefix_cache.evict((num_pages - num_free_pages) * self.page_size)
            self._release_pages(evicted.numpy())
            if len(self._free_pages) < num_pages:
                raise RuntimeError("prefix cache eviction did not free enough pages")
        return self._free_pages.take(num_pages)

    def _release_pages(self, *page_groups: np.ndarray) -> None:
        for group in page_groups:
            self._free_pages.put(group)

    def _write_pages(
        self,
        pages: np.ndarray,
        page_count_ranges: list[tuple[RequestSlot, int, int]],
    ) -> None:
        offset = 0
        for slot, source_page_count, target_page_count in page_count_ranges:
            next_offset = offset + target_page_count - source_page_count
            self._table[slot.index, source_page_count:target_page_count] = pages[offset:next_offset]
            self._record_cache_mapping_range(
                self._require_slot_state(slot),
                source_page_count,
                target_page_count,
            )
            offset = next_offset

    @staticmethod
    def _record_cache_mapping_range(
        state: _SlotCacheState,
        start: int,
        end: int,
    ) -> None:
        if start == end:
            return
        if not state.pending_cache_mapping_ranges:
            state.pending_cache_mapping_ranges.append((start, end))
            return
        pending = sorted((*state.pending_cache_mapping_ranges, (start, end)))
        merged: list[tuple[int, int]] = []
        for range_start, range_end in pending:
            if merged and range_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], range_end))
            else:
                merged.append((range_start, range_end))
        state.pending_cache_mapping_ranges = merged

    # ================================ State guards ================================

    def _require_slot_state(self, slot: RequestSlot) -> _SlotCacheState:
        try:
            return self._slot_states[slot]
        except KeyError as error:
            raise RuntimeError(f"cache state does not exist for {slot}") from error


def page_map_of(table: np.ndarray, spans: Sequence[tuple[int, int, int]]) -> PageMap:
    """The page table changes of ``(slot, first page index, end page index)`` spans of the
    host ``table``."""

    triples = np.empty((sum(end - first for _, first, end in spans), 3), np.int32)
    offset = 0
    for slot, first, end in spans:
        rows = triples[offset : offset + end - first]
        rows[:, 0] = slot
        rows[:, 1] = np.arange(first, end)
        rows[:, 2] = table[slot, first:end]
        offset += end - first
    return PageMap(triples.tobytes())
