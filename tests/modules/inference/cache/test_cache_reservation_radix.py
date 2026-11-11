"""Real-Radix integration contracts for the cache-reservation lifecycle."""

from __future__ import annotations

import torch

from cti.inference._control._cache import CacheManager, CacheUse
from cti.inference._control._scheduling.table import RequestSlot
from cti.inference._step import QueryKind, Row
from cti.kvcache import RestorePlan
from cti.protocol import SamplingParams
from tests.support.inference.cache import RequestInput, commit_reservation, new_cache_manager
from tests.support.inference.prefix_cache import reference_radix_comparison  # noqa: F401

PAGE_SIZE = 2


def _make_cache(*, num_pages: int, num_rows: int) -> CacheManager:
    page_table = torch.full((num_rows, 16), -1, dtype=torch.int32)
    return new_cache_manager(num_pages, PAGE_SIZE, page_table, reuse=True, checks=True)


def _free_page_count(cache: CacheManager) -> int:
    occupancy = cache.occupancy
    return occupancy.num_pages - occupancy.num_used_pages - occupancy.num_cached_pages


def _hit(cache: CacheManager, token_ids: torch.Tensor) -> RestorePlan:
    """The hit on a prompt one token past ``token_ids``, which may restore all of them."""

    return cache.prefix_cache.plan_restore(torch.cat((token_ids, token_ids[-1:])))


def _pending_request(uid: int, token_ids: torch.Tensor, *, max_tokens: int) -> RequestInput:
    return RequestInput(
        uid=uid,
        input_ids=token_ids,
        sampling_params=SamplingParams(temperature=0, max_new_tokens=max_tokens),
    )


def _open_slot(
    cache: CacheManager,
    slot: RequestSlot,
    token_ids: torch.Tensor,
    *,
    expected_prefix_len: int,
    max_tokens: int = 1,
) -> None:
    assert (
        cache.try_open_slot(
            slot,
            token_ids,
            max_tokens,
            reserved_size=0,
        )
        == expected_prefix_len
    )


def _reserve_slot(
    cache: CacheManager,
    slot: RequestSlot,
    source_len: int,
    target_len: int,
) -> CacheUse:
    entry = Row(
        request_index=slot.index,
        query_kind=QueryKind.EXTEND,
        kv_prefix_len=source_len,
        sequence_len=target_len,
        write_position=target_len,
    )
    reservations, _ = cache.reserve((slot,), (entry,))
    return reservations[0]


def _publish_closed_prefix(
    cache: CacheManager,
    slot: RequestSlot,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    _open_slot(cache, slot, token_ids, expected_prefix_len=0)
    reservation = _reserve_slot(cache, slot, 0, len(token_ids))
    commit_reservation(cache, reservation, prefix_ids=token_ids)
    cache_indices = cache.page_table[slot.index, : len(token_ids) // PAGE_SIZE].clone()
    cache.close_slot(slot, prefix_ids=token_ids)
    return cache_indices


def test_single_node_handle_returns_its_stored_indices() -> None:
    cache = _make_cache(num_pages=2, num_rows=1)
    token_ids = torch.tensor([11, 12], dtype=torch.int32)
    cache_indices = _publish_closed_prefix(cache, RequestSlot(index=0, generation=0), token_ids)

    handle = cache.prefix_cache.acquire(_hit(cache, token_ids))

    try:
        assert torch.equal(handle.get_matched_indices(), cache_indices)
    finally:
        cache.prefix_cache.unlock(handle)


def test_active_radix_handle_protects_its_pages_during_lifecycle_eviction() -> None:
    cache = _make_cache(num_pages=3, num_rows=3)
    seed_a = RequestSlot(index=0, generation=0)
    seed_b = RequestSlot(index=1, generation=0)
    active_a = RequestSlot(index=0, generation=1)
    consumer = RequestSlot(index=2, generation=0)
    token_ids_a = torch.tensor([11, 12], dtype=torch.int32)
    token_ids_b = torch.tensor([21, 22], dtype=torch.int32)

    cache_indices_a = _publish_closed_prefix(cache, seed_a, token_ids_a)
    cache_indices_b = _publish_closed_prefix(cache, seed_b, token_ids_b)
    assert cache.prefix_cache.size_info == (4, 0)

    _open_slot(
        cache,
        active_a,
        torch.tensor([11, 12, 13], dtype=torch.int32),
        expected_prefix_len=2,
    )
    assert cache.prefix_cache.size_info == (2, 2)

    # One page is free and only B is evictable, so C's two-page reservation must
    # retain A and evict B.
    token_ids_c = torch.tensor([31, 32, 33], dtype=torch.int32)
    _open_slot(cache, consumer, token_ids_c, expected_prefix_len=0)
    reservation_c = _reserve_slot(cache, consumer, 0, len(token_ids_c))
    cache_indices_c = cache.page_table[consumer.index, :2].clone()

    assert set(cache_indices_a.tolist()).isdisjoint(cache_indices_c.tolist())
    assert set(cache_indices_b.tolist()).issubset(cache_indices_c.tolist())
    assert _hit(cache, token_ids_a).cached_len == 2
    assert _hit(cache, token_ids_b).cached_len == 0
    assert cache.prefix_cache.size_info == (0, 2)

    cache.settle(reservation_c, 0)
    cache.close_slot(consumer, prefix_ids=torch.empty(0, dtype=torch.int32))
    cache.close_slot(active_a, prefix_ids=token_ids_a)
    assert cache.prefix_cache.size_info == (2, 0)
    cache.check_integrity()


def test_partial_radix_rewire_defers_then_safely_reuses_duplicate_page() -> None:
    cache = _make_cache(num_pages=5, num_rows=2)
    slot_a = RequestSlot(index=0, generation=0)
    slot_b = RequestSlot(index=1, generation=0)
    token_ids_a = torch.tensor([11, 12, 21, 22], dtype=torch.int32)
    token_ids_b = torch.tensor([11, 12, 31, 32, 41, 42, 43], dtype=torch.int32)

    # Both requests match the root before either publishes, creating a genuine cold
    # partial-prefix collision in the real Radix tree.
    _open_slot(cache, slot_a, token_ids_a, expected_prefix_len=0)
    _open_slot(cache, slot_b, token_ids_b[:4], expected_prefix_len=0, max_tokens=3)
    reservation_a = _reserve_slot(cache, slot_a, 0, 4)
    reservation_b = _reserve_slot(cache, slot_b, 0, 4)
    private_a = cache.page_table[slot_a.index, :2].clone()
    private_b = cache.page_table[slot_b.index, :2].clone()
    successor_b = _reserve_slot(cache, slot_b, 4, 5)
    assert _free_page_count(cache) == 0

    commit_reservation(cache, reservation_a, prefix_ids=token_ids_a)
    commit_reservation(cache, reservation_b, prefix_ids=token_ids_b[:4])

    duplicate_page_base = int(private_b[0])
    expected_b = torch.cat((private_a[:1], private_b[1:]))
    assert torch.equal(cache.page_table[slot_b.index, :2], expected_b)
    # The duplicate waits for the successor that may read it: still no page is free.
    assert _free_page_count(cache) == 0
    assert cache.prefix_cache.size_info == (0, 3 * PAGE_SIZE)

    commit_reservation(cache, successor_b)
    assert _free_page_count(cache) == 1

    fill_existing_page = _reserve_slot(cache, slot_b, 5, 6)
    commit_reservation(cache, fill_existing_page)
    assert _free_page_count(cache) == 1

    # The returned duplicate is now the only free page and nothing is evictable, so B's
    # next page is it. Reusing it is safe only if B's prefix row was already rewired.
    reuse_duplicate_page = _reserve_slot(cache, slot_b, 6, 7)
    assert _free_page_count(cache) == 0
    assert int(cache.page_table[slot_b.index, 3]) == duplicate_page_base
    assert duplicate_page_base not in cache.page_table[slot_b.index, :2]
    commit_reservation(cache, reuse_duplicate_page)

    expected_b_prefix_indices = cache.page_table[slot_b.index, :3].clone()
    cache.close_slot(slot_b, prefix_ids=token_ids_b)
    cache.close_slot(slot_a, prefix_ids=token_ids_a)
    match_a = cache.prefix_cache.acquire(_hit(cache, token_ids_a))
    match_b = cache.prefix_cache.acquire(_hit(cache, token_ids_b))
    assert match_a.cached_len == 4
    assert torch.equal(match_a.get_matched_indices(), private_a)
    assert match_b.cached_len == 6
    assert torch.equal(
        match_b.get_matched_indices(),
        expected_b_prefix_indices,
    )
    cache.prefix_cache.unlock(match_a)
    cache.prefix_cache.unlock(match_b)
    assert cache.prefix_cache.size_info == (4 * PAGE_SIZE, 0)
    cache.check_integrity()
