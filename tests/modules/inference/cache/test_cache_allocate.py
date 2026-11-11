"""Page allocation and eviction contracts for ``CacheManager``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from cti.inference._control._cache import CacheManager
from cti.inference._control._scheduling.table import RequestSlot
from cti.inference._step import QueryKind, Row
from cti.protocol import SamplingParams
from tests.support.inference.cache import new_cache_manager


@dataclass(frozen=True)
class _RequestInput:
    input_ids: torch.Tensor
    sampling_params: SamplingParams

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_new_tokens


def _make_cache_manager(num_pages: int, page_size: int) -> CacheManager:
    page_table = torch.full((3, num_pages), -1, dtype=torch.int32)
    return new_cache_manager(num_pages, page_size, page_table, reuse=True, checks=True)


def _free_page_count(cache: CacheManager) -> int:
    occupancy = cache.occupancy
    return occupancy.num_pages - occupancy.num_used_pages - occupancy.num_cached_pages


def _cache_closed_prefix(cache: CacheManager, slot: RequestSlot, num_pages: int) -> list[int]:
    """Publish a ``num_pages``-page prefix through ``slot`` and close it: its pages turn
    evictable. Return them."""
    token_ids = torch.arange(num_pages * cache.page_size, dtype=torch.int32)
    pages = _reserve(cache, slot, token_ids)
    cache.close_slot(slot, prefix_ids=token_ids)
    assert (cache.occupancy.num_used_pages, cache.occupancy.num_cached_pages) == (0, num_pages)
    return pages


def _reserve(
    cache: CacheManager, slot: RequestSlot, token_ids: torch.Tensor, *, commit: bool = True
) -> list[int]:
    """Open ``slot`` on ``token_ids``, reserve them all and settle; return the slot's pages."""
    assert cache.try_open_slot(slot, token_ids, 0, reserved_size=0) == 0
    entry = Row(slot.index, QueryKind.EXTEND, 0, len(token_ids), len(token_ids))
    (use,), _ = cache.reserve((slot,), (entry,))
    if commit:
        cache.settle(use, len(token_ids), prefix_ids=token_ids)
    else:
        cache.settle(use, 0)
    return cache.page_table[slot.index, : len(token_ids) // cache.page_size].tolist()


def _assert_valid_pages(cache: CacheManager, ids: list[int]) -> None:
    assert len(ids) == len(set(ids)), f"duplicate page IDs: {ids}"
    assert all(0 <= page < cache.num_pages for page in ids), f"page IDs out of range: {ids}"


def _prompt(num_pages: int, page_size: int, first: int) -> torch.Tensor:
    """A prompt of ``num_pages`` whole pages that shares no prefix with another ``first``."""
    return torch.arange(first, first + num_pages * page_size, dtype=torch.int32)


def test_page_aligned_table_width_accepts_partial_final_page() -> None:
    page_size = 64
    page_table = torch.full((1, 2), -1, dtype=torch.int32)
    cache = new_cache_manager(2, page_size, page_table, reuse=False, checks=True)
    slot = RequestSlot(index=0, generation=0)
    request = _RequestInput(
        input_ids=torch.tensor([11], dtype=torch.int32),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=64),
    )
    assert cache.try_open_slot(slot, request.input_ids, request.output_len, reserved_size=0) == 0
    entry = Row(
        request_index=slot.index,
        query_kind=QueryKind.EXTEND,
        kv_prefix_len=0,
        sequence_len=65,
        write_position=65,
    )

    (reservation,), _ = cache.reserve((slot,), (entry,))

    assert cache.page_table[0].tolist() == [0, 1]
    cache.settle(reservation, 0)
    cache.close_slot(slot, prefix_ids=torch.empty(0, dtype=torch.int32))
    cache.check_integrity()


class TestPageAllocation:
    def test_radix_eviction_returns_cached_page_ids(self) -> None:
        page_size, num_pages = 4, 4
        cache = _make_cache_manager(num_pages, page_size)
        cached = _cache_closed_prefix(cache, RequestSlot(0, 0), 2)

        evicted = cache.prefix_cache.evict(page_size).tolist()

        assert evicted
        _assert_valid_pages(cache, evicted)
        assert set(evicted) <= set(cached)

    def test_reserve_evicts_cached_pages_once_none_is_free(self) -> None:
        page_size, num_pages = 4, 4
        cache = _make_cache_manager(num_pages, page_size)
        cached = _cache_closed_prefix(cache, RequestSlot(0, 0), 2)
        assert _free_page_count(cache) == 2

        # Three pages: both free ones, then eviction.
        slot = RequestSlot(1, 0)
        allocated = _reserve(cache, slot, _prompt(3, page_size, 100), commit=False)
        assert cache.occupancy.num_used_pages == 0
        cache.close_slot(slot, prefix_ids=torch.empty(0, dtype=torch.int32))

        _assert_valid_pages(cache, allocated)
        assert set(allocated) & set(cached)
        assert cache.occupancy.num_cached_pages < 2
        cache.check_integrity()

    def test_consecutive_reservations_after_evict_never_overlap(self) -> None:
        page_size, num_pages = 4, 4
        cache = _make_cache_manager(num_pages, page_size)
        _cache_closed_prefix(cache, RequestSlot(0, 0), 2)
        first, second = RequestSlot(1, 0), RequestSlot(2, 0)
        for slot, base in ((first, 100), (second, 200)):
            token_ids = _prompt(2, page_size, base)
            assert cache.try_open_slot(slot, token_ids, 0, reserved_size=0) == 0
            entry = Row(slot.index, QueryKind.EXTEND, 0, len(token_ids), len(token_ids))
            cache.reserve((slot,), (entry,))

        live = cache.page_table[[first.index, second.index], :2].flatten().tolist()

        _assert_valid_pages(cache, live)
        assert cache.occupancy.num_used_pages == num_pages
        assert cache.available_size == 0

    def test_reserve_takes_exactly_the_pages_it_needs_from_eviction(self) -> None:
        page_size, num_pages = 4, 4
        cache = _make_cache_manager(num_pages, page_size)
        _cache_closed_prefix(cache, RequestSlot(0, 0), 3)
        slot = RequestSlot(1, 0)
        token_ids = _prompt(2, page_size, 100)
        assert cache.try_open_slot(slot, token_ids, 0, reserved_size=0) == 0

        entry = Row(slot.index, QueryKind.EXTEND, 0, len(token_ids), len(token_ids))
        (use,), _ = cache.reserve((slot,), (entry,))

        _assert_valid_pages(cache, cache.page_table[slot.index, :2].tolist())
        assert cache.occupancy.num_used_pages == 2
        cache.settle(use, 0)
        cache.close_slot(slot, prefix_ids=torch.empty(0, dtype=torch.int32))
        cache.check_integrity()

    def test_check_integrity_passes_after_evict_cycle(self) -> None:
        page_size, num_pages = 4, 8
        cache = _make_cache_manager(num_pages, page_size)
        # A finished request's two pages, cached: free=6, cache=2, total=8.
        _cache_closed_prefix(cache, RequestSlot(0, 0), 2)
        cache.check_integrity()

        holder, evicting = RequestSlot(1, 0), RequestSlot(2, 0)
        holder_ids, evicting_ids = _prompt(6, page_size, 100), _prompt(1, page_size, 200)
        assert cache.try_open_slot(holder, holder_ids, 0, reserved_size=0) == 0
        entry = Row(holder.index, QueryKind.EXTEND, 0, len(holder_ids), len(holder_ids))
        (holder_use,), _ = cache.reserve((holder,), (entry,))
        assert _free_page_count(cache) == 0
        # One more page must evict from the cache.
        assert cache.try_open_slot(evicting, evicting_ids, 0, reserved_size=0) == 0
        entry = Row(evicting.index, QueryKind.EXTEND, 0, len(evicting_ids), len(evicting_ids))
        (evicting_use,), _ = cache.reserve((evicting,), (entry,))
        live = [
            *cache.page_table[holder.index, :6].tolist(),
            *cache.page_table[evicting.index, :1].tolist(),
        ]
        _assert_valid_pages(cache, live)

        for slot, use in ((holder, holder_use), (evicting, evicting_use)):
            cache.settle(use, 0)
            cache.close_slot(slot, prefix_ids=torch.empty(0, dtype=torch.int32))
        cache.check_integrity()
        assert _free_page_count(cache) + cache.occupancy.num_cached_pages == num_pages


if __name__ == "__main__":
    pytest.main([__file__])
