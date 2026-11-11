from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pytest
import torch

from cti.inference._control._cache import CacheManager, CacheUse
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.prefill import PrefillQueue
from cti.inference._control._scheduling.table import RequestSlot, SlotTable
from cti.inference._step import QueryKind, Row
from cti.kvcache import BaseCacheHandle, RestorePlan
from cti.protocol import SamplingParams, UserMsg
from tests.support.inference.cache import (
    RequestInput,
    commit_reservation,
    new_cache_manager,
    prefix_cache,
)
from tests.support.inference.prefix_cache import reference_radix_comparison  # noqa: F401
from tests.support.inference.scheduling import page_triples


class _SizeInfo(NamedTuple):
    """``BasePrefixCache.size_info``'s fields."""

    evictable_size: int
    protected_size: int


@dataclass(frozen=True)
class _Handle(BaseCacheHandle):
    indices: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        return self.indices


class _UnlockedPrefixCache:
    """Admits a slot on its ``opening`` hit, holding no pages, and records unlocks."""

    rows = None

    def __init__(self) -> None:
        self.opening: BaseCacheHandle = _Handle(0, torch.empty(0, dtype=torch.int32))
        self.unlocked: list[BaseCacheHandle] = []

    @property
    def size_info(self) -> _SizeInfo:
        return _SizeInfo(evictable_size=0, protected_size=0)

    def hit_limit(self, _prompt_len: int) -> int:
        return 0

    def plan_restore(self, _input_ids: torch.Tensor) -> RestorePlan:
        return RestorePlan(self.opening.cached_len)

    def acquire(self, _plan: RestorePlan) -> BaseCacheHandle:
        return self.opening

    def lock(self, _handle: BaseCacheHandle) -> None:
        pass

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.unlocked.append(handle)


class _RecordingPrefixCache(_UnlockedPrefixCache):
    def __init__(self) -> None:
        super().__init__()
        self.inserted: list[torch.Tensor] = []

    def publish(
        self, input_ids: torch.Tensor, indices: torch.Tensor, row: int | None = None
    ) -> tuple[int, BaseCacheHandle]:
        self.inserted.append(input_ids.clone())
        return (0, _Handle(len(input_ids), indices.clone()))


class _CanonicalizingPrefixCache(_UnlockedPrefixCache):
    def __init__(self) -> None:
        super().__init__()
        self._tokens: torch.Tensor | None = None
        self._indices: torch.Tensor | None = None

    def publish(
        self, input_ids: torch.Tensor, indices: torch.Tensor, row: int | None = None
    ) -> tuple[int, BaseCacheHandle]:
        if self._tokens is None:
            self._tokens = input_ids.clone()
            self._indices = indices.clone()
            return (0, _Handle(len(input_ids), self._indices))

        assert self._indices is not None
        common_len = min(len(input_ids), len(self._tokens))
        assert torch.equal(input_ids[:common_len], self._tokens[:common_len])
        if len(input_ids) > len(self._tokens):
            cached_len = len(self._tokens)
            self._tokens = input_ids.clone()
            self._indices = indices.clone()
        else:
            cached_len = len(input_ids)
        return (
            cached_len,
            _Handle(len(input_ids), self._indices[: len(input_ids)]),
        )


class _MatchedPrefixCache:
    rows = None

    def __init__(self, prefix_handle: BaseCacheHandle) -> None:
        self.prefix_handle = prefix_handle
        self.protection_calls: list[str] = []

    @property
    def size_info(self) -> _SizeInfo:
        return _SizeInfo(evictable_size=0, protected_size=0)

    def hit_limit(self, _prompt_len: int) -> int:
        return 0

    def plan_restore(self, _input_ids: torch.Tensor) -> RestorePlan:
        return RestorePlan(self.prefix_handle.cached_len)

    def acquire(self, _plan: RestorePlan) -> BaseCacheHandle:
        self.protection_calls.append("protect")
        return self.prefix_handle

    def lock(self, _handle: BaseCacheHandle) -> None:
        self.protection_calls.append("protect")

    def unlock(self, _handle: BaseCacheHandle) -> None:
        self.protection_calls.append("unprotect")


class _EvictableHitPrefixCache(_MatchedPrefixCache):
    """A hit no request protects: acquiring it would take its pages from capacity."""

    @property
    def size_info(self) -> _SizeInfo:
        return _SizeInfo(evictable_size=self.prefix_handle.cached_len, protected_size=0)

    def plan_restore(self, _input_ids: torch.Tensor) -> RestorePlan:
        cached_len = self.prefix_handle.cached_len
        return RestorePlan(cached_len, cached_len, unprotected_len=cached_len)


class _FailingAdmissionPrefixCache(_MatchedPrefixCache):
    def __init__(self, stage: str) -> None:
        super().__init__(_Handle(0, torch.empty(0, dtype=torch.int32)))
        self._stage = stage

    @property
    def size_info(self) -> _SizeInfo:
        if self._stage == "capacity":
            raise RuntimeError("injected initial-capacity failure")
        return super().size_info

    def plan_restore(self, input_ids: torch.Tensor) -> RestorePlan:
        if self._stage == "match":
            raise RuntimeError("injected prefix-match failure")
        return super().plan_restore(input_ids)


class _UnderEvictingPrefixCache:
    def __init__(self, evicted_indices: torch.Tensor) -> None:
        self._evicted_indices = evicted_indices

    @property
    def size_info(self) -> _SizeInfo:
        return _SizeInfo(evictable_size=len(self._evicted_indices), protected_size=0)

    def evict(self, _size: int) -> torch.Tensor:
        return self._evicted_indices


class _IntegrityPrefixCache:
    def __init__(
        self,
        size_info: _SizeInfo,
        error: BaseException | None = None,
    ) -> None:
        self._size_info = size_info
        self._error = error

    @property
    def size_info(self) -> _SizeInfo:
        return self._size_info

    def check_integrity(self) -> None:
        if self._error is not None:
            raise self._error


def _try_open_slot(
    cache: CacheManager,
    slot: RequestSlot,
    request: RequestInput,
    *,
    reserved_size: int,
) -> int | None:
    return cache.try_open_slot(
        slot,
        request.input_ids,
        request.output_len,
        reserved_size=reserved_size,
    )


def _open(cache: CacheManager, slot: RequestSlot, prompt_len: int = 1) -> None:
    """Open ``slot`` on a ``prompt_len``-token prompt with no output."""
    prompt = torch.zeros(prompt_len, dtype=torch.int32)
    assert cache.try_open_slot(slot, prompt, 0, reserved_size=0) is not None


def _free_pages(cache: CacheManager, probe: RequestSlot) -> set[int]:
    """The pages the cache hands out before evicting: reserved by ``probe``, then returned.
    Their count matches the free count, so the free pages hold no duplicate."""
    occupancy = cache.occupancy
    free = occupancy.num_pages - occupancy.num_used_pages - occupancy.num_cached_pages
    prompt = torch.zeros(1, dtype=torch.int32)
    assert cache.try_open_slot(probe, prompt, 0, reserved_size=0, prefix=False) == 0
    use = _reserve_slot(cache, probe, 0, free * cache.page_size)
    pages = set(cache.page_table[probe.index, :free].tolist())
    cache.settle(use, 0)
    cache.close_slot(probe, prefix_ids=prompt[:0])
    assert len(pages) == free
    return pages


def _make_execution_entry(
    request_index: int,
    source_len: int,
    target_len: int,
) -> Row:
    return Row(
        request_index=request_index,
        query_kind=QueryKind.EXTEND,
        kv_prefix_len=source_len,
        sequence_len=target_len,
        write_position=target_len,
    )


def _reserve_slot(
    cache: CacheManager,
    slot: RequestSlot,
    source_len: int,
    target_len: int,
) -> CacheUse:
    entry = _make_execution_entry(slot.index, source_len, target_len)
    (use,), _ = cache.reserve((slot,), (entry,))
    return use


def _release_reservation(cache: CacheManager, use: CacheUse) -> None:
    cache.settle(use, 0)


def test_in_use_tracks_each_open_slot_exactly() -> None:
    page_table = torch.full((2, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, page_table, reuse=False, checks=True)
    active_slot = RequestSlot(index=0, generation=0)
    idle_slot = RequestSlot(index=1, generation=0)
    for slot in (active_slot, idle_slot):
        _open(cache, slot)

    assert not cache.in_use(active_slot)
    assert not cache.in_use(idle_slot)

    predecessor = _reserve_slot(cache, active_slot, 0, 2)
    successor = _reserve_slot(cache, active_slot, 2, 3)
    assert cache.in_use(active_slot)
    assert not cache.in_use(idle_slot)

    _release_reservation(cache, predecessor)
    assert cache.in_use(active_slot)

    _release_reservation(cache, successor)
    assert not cache.in_use(active_slot)

    cache.close_slot(active_slot, prefix_ids=torch.empty(0, dtype=torch.int32))
    cache.close_slot(idle_slot, prefix_ids=torch.empty(0, dtype=torch.int32))
    cache.check_integrity()


def test_batch_reservation_returns_its_mapping_snapshot_atomically() -> None:
    cache_index_table = torch.full((1, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, cache_index_table, reuse=False, checks=True)
    slot = RequestSlot(index=0, generation=0)
    _open(cache, slot)

    reservations, updates = cache.reserve(
        (slot,),
        (_make_execution_entry(slot.index, 0, 2),),
    )

    assert len(reservations) == 1
    assert updates is not None
    assert page_triples(updates)[:, :2].T.tolist() == [[slot.index, slot.index], [0, 1]]
    assert torch.equal(page_triples(updates)[:, 2], cache.page_table[slot.index, :2])


def test_cache_mapping_updates_are_page_granular_owned_snapshots() -> None:
    page_table = torch.full((2, 6), -1, dtype=torch.int32)
    cache = new_cache_manager(6, 2, page_table, reuse=False, checks=True)
    slots = (
        RequestSlot(index=1, generation=0),
        RequestSlot(index=0, generation=0),
    )
    for slot in slots:
        _open(cache, slot)

    reservations, updates = cache.reserve(
        slots,
        (
            _make_execution_entry(slots[0].index, 0, 3),
            _make_execution_entry(slots[1].index, 0, 5),
        ),
    )
    expected_pages = torch.cat(
        (cache.page_table[slots[0].index, :2], cache.page_table[slots[1].index, :3])
    ).clone()

    assert updates is not None
    assert page_triples(updates)[:, :2].T.tolist() == [
        [slots[0].index, slots[0].index, slots[1].index, slots[1].index, slots[1].index],
        [0, 1, 0, 1, 2],
    ]
    assert torch.equal(page_triples(updates)[:, 2], expected_pages)

    successor_reservations, repeated_updates = cache.reserve(
        slots,
        (
            _make_execution_entry(slots[0].index, 3, 4),
            _make_execution_entry(slots[1].index, 5, 6),
        ),
    )
    assert repeated_updates is None

    logical_snapshot = cache.page_table.clone()
    cache.page_table.fill_(-1)
    assert torch.equal(page_triples(updates)[:, 2], expected_pages)
    cache.page_table.copy_(logical_snapshot)

    for reservation in (*reservations, *successor_reservations):
        _release_reservation(cache, reservation)
    for slot in slots:
        cache.close_slot(slot, prefix_ids=torch.empty(0, dtype=torch.int32))
    cache.check_integrity()


def test_mapping_snapshot_failure_never_returns_a_partial_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_index_table = torch.full((1, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, cache_index_table, reuse=False, checks=True)
    slot = RequestSlot(index=0, generation=0)
    _open(cache, slot)
    failure = RuntimeError("injected mapping snapshot failure")

    def fail_snapshot(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("cti.inference._control._cache.page_map_of", fail_snapshot)

    with pytest.raises(RuntimeError) as caught:
        cache.reserve((slot,), (_make_execution_entry(slot.index, 0, 2),))

    assert caught.value is failure


def test_reservation_preflight_failure_preserves_pending_mapping_updates() -> None:
    cache_index_table = torch.full((2, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, cache_index_table, reuse=False, checks=True)
    prefix_cache = _RecordingPrefixCache()
    cache.prefix_cache = prefix_cache
    live_slot = RequestSlot(index=0, generation=0)
    unknown_slot = RequestSlot(index=1, generation=0)
    prefix_indices = torch.tensor([6, 7], dtype=torch.int32)
    prefix_cache.opening = _Handle(2, prefix_indices)
    _open(cache, live_slot, 3)

    with pytest.raises(RuntimeError, match="cache state does not exist"):
        cache.reserve(
            (live_slot, unknown_slot),
            (
                _make_execution_entry(live_slot.index, 2, 3),
                _make_execution_entry(unknown_slot.index, 0, 1),
            ),
        )

    reservations, updates = cache.reserve(
        (live_slot,),
        (_make_execution_entry(live_slot.index, 2, 3),),
    )
    assert updates is not None
    assert page_triples(updates)[:, :2].T.tolist() == [
        [live_slot.index, live_slot.index, live_slot.index],
        [0, 1, 2],
    ]
    assert page_triples(updates)[:, 2][:2].tolist() == prefix_indices.tolist()

    _release_reservation(cache, reservations[0])
    cache.close_slot(live_slot, prefix_ids=torch.tensor([10, 11], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="cache state does not exist"):
        cache.in_use(live_slot)


def test_a_released_use_returns_its_pages_and_a_committed_one_keeps_them() -> None:
    page_table = torch.full((3, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, page_table, reuse=False, checks=True)
    committed_slot = RequestSlot(index=0, generation=0)
    released_slot = RequestSlot(index=1, generation=0)
    probe = RequestSlot(index=2, generation=0)
    for slot in (committed_slot, released_slot):
        _open(cache, slot)
    committed = _reserve_slot(cache, committed_slot, 0, 2)
    released = _reserve_slot(cache, released_slot, 0, 2)
    committed_prefix = torch.tensor([11, 12], dtype=torch.int32)
    free_page_bases = _free_pages(cache, probe)
    committed_page_bases = set(cache.page_table[committed_slot.index, :2].tolist())
    released_page_bases = set(cache.page_table[released_slot.index, :2].tolist())

    cache.settle(committed, 2, prefix_ids=committed_prefix)
    cache.settle(released, 0)

    assert not cache.in_use(committed_slot) and not cache.in_use(released_slot)
    assert _free_pages(cache, probe) == free_page_bases | released_page_bases
    cache.close_slot(released_slot, prefix_ids=torch.empty(0, dtype=torch.int32))
    cache.close_slot(committed_slot, prefix_ids=committed_prefix)
    assert _free_pages(cache, probe) == (
        free_page_bases | released_page_bases | committed_page_bases
    )
    cache.check_integrity()


def test_a_successor_settling_before_its_predecessor_is_rejected() -> None:
    page_table = torch.full((2, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, page_table, reuse=False, checks=True)
    slot = RequestSlot(index=0, generation=0)
    probe = RequestSlot(index=1, generation=0)
    _open(cache, slot)
    predecessor = _reserve_slot(cache, slot, 0, 2)
    successor = _reserve_slot(cache, slot, 2, 3)
    occupancy = cache.occupancy
    free_page_bases = _free_pages(cache, probe)
    cache_index_table = cache.page_table[slot.index].clone()

    with pytest.raises(RuntimeError, match="out of order"):
        commit_reservation(cache, successor)

    assert cache.occupancy == occupancy
    assert _free_pages(cache, probe) == free_page_bases
    assert torch.equal(cache.page_table[slot.index], cache_index_table)
    commit_reservation(cache, predecessor)
    commit_reservation(cache, successor)
    cache.close_slot(slot, prefix_ids=torch.tensor([11, 12, 13], dtype=torch.int32))
    cache.check_integrity()


def test_commit_uses_the_reserved_frontier_after_a_successor_is_reserved() -> None:
    page_table = torch.zeros((1, 8), dtype=torch.int32)
    cache = new_cache_manager(8, 1, page_table, reuse=False, checks=True)
    prefix_cache = _RecordingPrefixCache()
    cache.prefix_cache = prefix_cache
    slot = RequestSlot(index=0, generation=0)
    tokens = torch.tensor([11, 12, 13, 14], dtype=torch.int32)

    _open(cache, slot)
    first = _reserve_slot(cache, slot, 0, 3)
    successor = _reserve_slot(cache, slot, 3, 4)

    commit_reservation(cache, first, prefix_ids=tokens[:3])

    assert len(prefix_cache.inserted) == 1
    assert torch.equal(prefix_cache.inserted[0], tokens[:3])

    _release_reservation(cache, successor)
    cache.close_slot(slot, prefix_ids=tokens)


def test_terminal_close_publishes_valid_prefix_and_releases_speculative_tail() -> None:
    page_table = torch.full((2, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, page_table, reuse=False, checks=True)
    prefix_cache = _RecordingPrefixCache()
    cache.prefix_cache = prefix_cache
    slot = RequestSlot(index=0, generation=0)
    handle = prefix_cache.opening
    tokens = torch.tensor([11, 12, 13], dtype=torch.int32)
    _open(cache, slot)
    valid = _reserve_slot(cache, slot, 0, 2)
    valid_pages = set(cache.page_table[slot.index, :2].tolist())
    speculative = _reserve_slot(cache, slot, 2, 3)

    commit_reservation(cache, valid)
    assert prefix_cache.inserted == [] and prefix_cache.unlocked == []
    _release_reservation(cache, speculative)
    cache.close_slot(slot, prefix_ids=tokens)

    assert len(prefix_cache.inserted) == 1
    assert torch.equal(prefix_cache.inserted[0], tokens[:2])
    # The slot kept its opening handle until the close published past it.
    assert prefix_cache.unlocked[0] is handle
    # The published pages stay with the prefix cache; the speculative tail returned.
    assert _free_pages(cache, RequestSlot(index=1, generation=0)) == set(range(8)) - valid_pages


def test_cache_manager_owns_prefix_match_protection_and_row_initialization() -> None:
    page_table = torch.full((1, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, page_table, reuse=False, checks=True)
    matched = _MatchedPrefixCache(_Handle(2, torch.tensor([6, 7], dtype=torch.int32)))
    cache.prefix_cache = matched
    slot = RequestSlot(index=0, generation=0)
    pending = RequestInput(
        uid=7,
        input_ids=torch.tensor([11, 12, 13], dtype=torch.int32),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=2),
    )

    cached_len = _try_open_slot(cache, slot, pending, reserved_size=0)

    assert cached_len == 2
    assert matched.protection_calls == ["protect"]
    assert torch.equal(cache.page_table[slot.index, :2], torch.tensor([6, 7]))

    cache.close_slot(slot, prefix_ids=pending.input_ids)
    assert matched.protection_calls == ["protect", "unprotect"]


def test_slot_admission_counts_the_pages_its_hit_takes_from_capacity() -> None:
    page_table = torch.full((2, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(1, 2, page_table, reuse=False, checks=True)
    # Another slot holds the only page: capacity is the hit's evictable pages alone.
    holder = RequestSlot(index=1, generation=0)
    _open(cache, holder)
    _reserve_slot(cache, holder, 0, 2)
    prefix_cache = _EvictableHitPrefixCache(_Handle(2, torch.tensor([0, 1], dtype=torch.int32)))
    cache.prefix_cache = prefix_cache
    request = RequestInput(
        uid=7,
        input_ids=torch.tensor([11, 12, 13], dtype=torch.int32),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )

    # The two evictable pages fit the request only while the hit leaves them evictable.
    assert cache.available_size == 2
    assert (
        _try_open_slot(
            cache,
            RequestSlot(index=0, generation=0),
            request,
            reserved_size=0,
        )
        is None
    )

    assert prefix_cache.protection_calls == []
    assert cache.available_size == 2


@pytest.mark.parametrize(
    ("stage", "error_match"),
    [
        pytest.param("match", "injected prefix-match failure", id="match"),
        pytest.param("capacity", "injected initial-capacity failure", id="capacity"),
    ],
)
def test_admission_provider_failure_raises_before_protection(
    stage: str,
    error_match: str,
) -> None:
    page_table = torch.full((1, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(1, 1, page_table, reuse=False, checks=True)
    prefix_cache = _FailingAdmissionPrefixCache(stage)
    cache.prefix_cache = prefix_cache
    request = RequestInput(
        uid=7,
        input_ids=torch.tensor([11], dtype=torch.int32),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )

    with pytest.raises(RuntimeError, match=error_match):
        _try_open_slot(
            cache,
            RequestSlot(index=0, generation=0),
            request,
            reserved_size=0,
        )

    assert prefix_cache.protection_calls == []


def test_batch_reservation_uses_frozen_execution_frontiers() -> None:
    page_table = torch.full((2, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, page_table, reuse=False, checks=True)
    slots = (RequestSlot(index=0, generation=0), RequestSlot(index=1, generation=0))
    for slot in slots:
        _open(cache, slot)
    entries = (_make_execution_entry(0, 0, 2), _make_execution_entry(1, 0, 3))

    reservations, _ = cache.reserve(slots, entries)

    assert [(item.slot, item.start, item.end) for item in reservations] == [
        (slots[0], 0, 2),
        (slots[1], 0, 3),
    ]
    pages = [*cache.page_table[0, :2].tolist(), *cache.page_table[1, :3].tolist()]
    assert len(set(pages)) == 5 and all(0 <= page < 8 for page in pages)
    assert cache.occupancy.num_used_pages == 5

    for reservation in reservations:
        _release_reservation(cache, reservation)
    for slot in slots:
        cache.close_slot(slot, prefix_ids=torch.empty(0, dtype=torch.int32))


def test_prefill_admission_accounts_for_per_request_page_rounding() -> None:
    page_table = torch.full((3, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(1, 4, page_table, reuse=False, checks=True)
    table = SlotTable(2)
    prefill = PrefillQueue(cache, table, encoder_cache=EncoderCache(0))
    for uid in (11, 12):
        prefill.enqueue(
            UserMsg(
                uid=uid,
                input_ids=torch.tensor([uid], dtype=torch.int32),
                sampling=SamplingParams(temperature=0, max_new_tokens=1).snapshot(),
            )
        )

    requests = prefill.schedule_extensions(8, reserved_size=0)

    assert [req.uid for req in requests] == [11]
    assert prefill.has_pending
    assert table.available_size == 1
    reservation = _reserve_slot(cache, requests[0].slot, 0, 1)
    _release_reservation(cache, reservation)
    cache.close_slot(requests[0].slot, prefix_ids=requests[0].token_ids)
    table.release(requests[0].slot)
    waiting = prefill.abort(12)
    assert waiting is not None
    assert waiting.uid == 12
    cache.check_integrity()


def test_canonical_rewire_waits_only_for_pre_rewire_successors() -> None:
    page_table = torch.full((3, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, page_table, reuse=False, checks=True)
    cache.prefix_cache = _CanonicalizingPrefixCache()
    canonical_slot = RequestSlot(index=0, generation=0)
    duplicate_slot = RequestSlot(index=1, generation=0)
    probe = RequestSlot(index=2, generation=0)
    tokens = torch.tensor([11, 12, 13], dtype=torch.int32)

    _open(cache, canonical_slot)
    canonical = _reserve_slot(cache, canonical_slot, 0, 2)
    commit_reservation(cache, canonical, prefix_ids=tokens[:2])
    canonical_pages = cache.page_table[canonical_slot.index, :2].tolist()

    _open(cache, duplicate_slot)
    current = _reserve_slot(cache, duplicate_slot, 0, 2)
    pre_rewire_successor = _reserve_slot(cache, duplicate_slot, 2, 3)
    *duplicate_pages, tail = cache.page_table[duplicate_slot.index, :3].tolist()
    assert len({*canonical_pages, *duplicate_pages, tail}) == 5

    commit_reservation(cache, current, prefix_ids=tokens[:2])

    # Rewired, but the duplicates stay held while the pre-rewire successor may read them.
    assert cache.page_table[duplicate_slot.index, :3].tolist() == [*canonical_pages, tail]
    held = {*canonical_pages, *duplicate_pages, tail}
    assert _free_pages(cache, probe) == set(range(8)) - held

    post_rewire_successor = _reserve_slot(cache, duplicate_slot, 3, 4)
    *rewired, fresh = cache.page_table[duplicate_slot.index, :4].tolist()
    assert rewired == [*canonical_pages, tail] and fresh not in held

    commit_reservation(cache, pre_rewire_successor)

    assert _free_pages(cache, probe) == set(range(8)) - {*canonical_pages, tail, fresh}

    _release_reservation(cache, post_rewire_successor)
    cache.close_slot(duplicate_slot, prefix_ids=tokens)
    cache.close_slot(canonical_slot, prefix_ids=tokens[:2])


def test_deferred_release_returns_whole_pages() -> None:
    page_table = torch.full((3, 6), -1, dtype=torch.int32)
    cache = new_cache_manager(6, 2, page_table, reuse=False, checks=True)
    cache.prefix_cache = _CanonicalizingPrefixCache()
    canonical_slot = RequestSlot(index=0, generation=0)
    duplicate_slot = RequestSlot(index=1, generation=0)
    probe = RequestSlot(index=2, generation=0)
    tokens = torch.tensor([11, 12, 13, 14], dtype=torch.int32)

    _open(cache, canonical_slot)
    canonical = _reserve_slot(cache, canonical_slot, 0, 4)
    commit_reservation(cache, canonical, prefix_ids=tokens)
    canonical_pages = cache.page_table[canonical_slot.index, :2].tolist()

    _open(cache, duplicate_slot)
    current = _reserve_slot(cache, duplicate_slot, 0, 4)
    successor = _reserve_slot(cache, duplicate_slot, 4, 5)
    *duplicate_pages, tail = cache.page_table[duplicate_slot.index, :3].tolist()
    held = {*canonical_pages, *duplicate_pages, tail}
    assert len(held) == 5

    commit_reservation(cache, current, prefix_ids=tokens)
    assert cache.page_table[duplicate_slot.index, :3].tolist() == [*canonical_pages, tail]
    assert _free_pages(cache, probe) == set(range(6)) - held

    _release_reservation(cache, successor)
    assert _free_pages(cache, probe) == set(range(6)) - set(canonical_pages)


def test_naive_prefix_cache_reuses_nothing() -> None:
    first, second = prefix_cache(kind="naive"), prefix_cache(kind="naive")
    tokens = torch.tensor([11, 12], dtype=torch.int32)

    plan = first.plan_restore(tokens)
    handle = first.acquire(plan)

    assert plan == RestorePlan(0) and handle.cached_len == 0
    assert handle.get_matched_indices().dtype == torch.int32
    assert handle.get_matched_indices().numel() == 0
    # Each cache's handles hold its own empty indices.
    assert handle.get_matched_indices() is first.acquire(plan).get_matched_indices()
    assert handle.get_matched_indices() is not second.acquire(plan).get_matched_indices()
    assert first.publish(tokens, torch.tensor([0, 1], dtype=torch.int32))[0] == 0
    assert first.plan_restore(tokens) == RestorePlan(0)
    assert first.size_info == (0, 0) and first.evict(0).numel() == 0


def test_radix_double_unlock_fails_before_mutating_refcounts() -> None:
    cache = prefix_cache()
    tokens = torch.tensor([11, 12], dtype=torch.int32)
    cache.publish(tokens[:1], torch.tensor([0], dtype=torch.int32))
    leaf = cache.publish(tokens, torch.tensor([0, 1], dtype=torch.int32)).handle
    parent = cache.acquire(cache.plan_restore(tokens))  # locks the one-token prefix

    # The leaf is not locked; unlocking it must not release the locked parent either.
    with pytest.raises(RuntimeError, match="not locked"):
        cache.unlock(leaf)

    assert cache.size_info == (1, 1)
    cache.unlock(parent)
    assert cache.size_info == (2, 0)


def test_short_radix_prefix_keeps_the_root_handle_safe() -> None:
    page_table = torch.full((1, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(2, 4, page_table, reuse=True, checks=True)
    slot = RequestSlot(index=0, generation=0)
    tokens = torch.tensor([11, 12], dtype=torch.int32)
    assert cache.try_open_slot(slot, tokens, 0, reserved_size=0) == 0
    reservation = _reserve_slot(cache, slot, 0, 2)

    commit_reservation(cache, reservation, prefix_ids=tokens)

    cache.close_slot(slot, prefix_ids=tokens)


def test_under_eviction_raises() -> None:
    page_table = torch.full((1, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(1, 1, page_table, reuse=False, checks=True)
    slot = RequestSlot(index=0, generation=0)
    _open(cache, slot)
    cache.prefix_cache = _UnderEvictingPrefixCache(torch.empty(0, dtype=torch.int32))

    target_len = 2
    with pytest.raises(RuntimeError, match="eviction"):
        _reserve_slot(cache, slot, 0, target_len)


def test_integrity_check_requires_quiescence() -> None:
    page_table = torch.full((1, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(8, 1, page_table, reuse=False, checks=True)
    slot = RequestSlot(index=0, generation=0)
    request = RequestInput(
        uid=7,
        input_ids=torch.tensor([11], dtype=torch.int32),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    assert _try_open_slot(cache, slot, request, reserved_size=0) == 0

    with pytest.raises(RuntimeError, match="requires all cache slots to be closed"):
        cache.check_integrity()

    cache.close_slot(slot, prefix_ids=request.input_ids)
    cache.check_integrity()


def test_quiescent_integrity_failure_raises_on_a_lost_page() -> None:
    page_table = torch.full((1, 4), -1, dtype=torch.int32)
    cache = new_cache_manager(2, 2, page_table, reuse=True, checks=True)
    slot = RequestSlot(index=0, generation=0)
    tokens = torch.tensor([11, 12], dtype=torch.int32)
    assert cache.try_open_slot(slot, tokens, 0, reserved_size=0) == 0
    commit_reservation(cache, _reserve_slot(cache, slot, 0, 2), prefix_ids=tokens)
    cache.close_slot(slot, prefix_ids=tokens)
    cache.check_integrity()
    # Evicting outside the manager drops the cached page from every account.
    assert len(cache.prefix_cache.evict(2)) == 1

    with pytest.raises(RuntimeError, match="integrity check failed"):
        cache.check_integrity()


def test_quiescent_integrity_failure_raises_on_a_page_out_of_range() -> None:
    page_table = torch.full((1, 4), -1, dtype=torch.int32)
    cache = new_cache_manager(2, 2, page_table, reuse=False, checks=True)
    slot = RequestSlot(index=0, generation=0)
    _open(cache, slot)
    use = _reserve_slot(cache, slot, 0, 2)
    cache.page_table[slot.index, 0] = cache.num_pages  # a corrupt mapping, released as is
    _release_reservation(cache, use)
    cache.close_slot(slot, prefix_ids=torch.empty(0, dtype=torch.int32))

    with pytest.raises(RuntimeError, match="free page out of range"):
        cache.check_integrity()


@pytest.mark.parametrize(
    "size_info",
    [
        pytest.param(_SizeInfo(evictable_size=1, protected_size=0), id="partial-page"),
        pytest.param(_SizeInfo(evictable_size=0, protected_size=2), id="protected-page"),
    ],
)
def test_invalid_quiescent_prefix_cache_size_raises(
    size_info: _SizeInfo,
) -> None:
    page_table = torch.full((1, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(2, 2, page_table, reuse=False, checks=True)
    cache.prefix_cache = _IntegrityPrefixCache(size_info)

    with pytest.raises(RuntimeError, match="prefix cache size"):
        cache.check_integrity()


def test_prefix_cache_integrity_failure_raises() -> None:
    page_table = torch.full((1, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(2, 2, page_table, reuse=False, checks=True)
    cache.prefix_cache = _IntegrityPrefixCache(
        _SizeInfo(evictable_size=0, protected_size=0),
        RuntimeError("injected prefix integrity failure"),
    )

    with pytest.raises(RuntimeError, match="injected prefix integrity failure"):
        cache.check_integrity()
