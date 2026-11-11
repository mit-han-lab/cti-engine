"""The free pages through ``CacheManager``'s public API: slots take and return pages
across many cycles, never sharing one, until the pool is exhausted or reset."""

from __future__ import annotations

import pytest
import torch

from cti.inference._control._cache import CacheManager
from cti.inference._control._scheduling.table import RequestSlot
from cti.inference._step import QueryKind, Row
from cti.utils import div_ceil
from tests.support.inference.cache import new_cache_manager

_SLOTS = 3
_WIDTH = 32


def _cache(num_pages: int, page_size: int) -> CacheManager:
    table = torch.full((_SLOTS, _WIDTH), -1, dtype=torch.int32)
    return new_cache_manager(num_pages, page_size, table, reuse=False, checks=True)


def _open(cache: CacheManager, slot: RequestSlot, length: int) -> bool:
    """Open ``slot`` for a ``length``-token prompt and one output, mapping the prompt."""

    prompt = torch.arange(length, dtype=torch.int32)
    if cache.try_open_slot(slot, prompt, 1, reserved_size=0) is None:
        return False
    row = Row(slot.index, QueryKind.EXTEND, kv_prefix_len=0, sequence_len=length)
    (use,), _ = cache.reserve((slot,), (row,))
    cache.settle(use, length)
    return True


def _close(cache: CacheManager, slot: RequestSlot, length: int) -> None:
    cache.close_slot(slot, prefix_ids=torch.arange(length, dtype=torch.int32))


def _pages(cache: CacheManager, slot: RequestSlot, length: int) -> list[int]:
    return cache.page_table[slot.index, : div_ceil(length, cache.page_size)].tolist()


@pytest.mark.parametrize("page_size", [1, 4])
def test_slots_never_share_a_page_across_cycles(page_size: int) -> None:
    num_pages = 10  # three live slots of up to three pages, and an output page
    cache = _cache(num_pages, page_size)
    live: dict[RequestSlot, int] = {}
    # Lengths that leave the pool at different fills, so returns wrap past its end.
    for cycle, length in enumerate([3, 1, 2, 3, 2, 1, 3, 3, 1, 2] * 3):
        slot = RequestSlot(index=cycle % _SLOTS, generation=cycle)
        for old in [s for s in live if s.index == slot.index]:
            _close(cache, old, live.pop(old))
        assert _open(cache, slot, length * page_size)
        live[slot] = length * page_size
        pages = [p for s, n in live.items() for p in _pages(cache, s, n)]
        assert len(pages) == len(set(pages))
        assert all(0 <= p < num_pages for p in pages)
        assert cache.available_size == (num_pages - len(pages)) * page_size
    for slot, length in live.items():
        _close(cache, slot, length)
    cache.check_integrity()
    assert cache.available_size == num_pages * page_size


def test_the_last_page_is_taken_and_none_past_it() -> None:
    cache = _cache(4, 2)
    first, second = RequestSlot(0, 0), RequestSlot(1, 0)
    assert _open(cache, first, 7)  # 7 tokens and the output: all four pages
    assert cache.available_size == 0
    assert not _open(cache, second, 1)
    _close(cache, first, 7)
    assert _open(cache, second, 7)
    assert sorted(_pages(cache, second, 7)) == [0, 1, 2, 3]


def test_invalidation_restores_every_page_in_its_first_order() -> None:
    cache = _cache(5, 1)
    for generation, length in enumerate([2, 3, 1]):
        slot = RequestSlot(0, generation)
        assert _open(cache, slot, length)
        _close(cache, slot, length)
    cache.invalidate_all()
    slot = RequestSlot(0, 9)
    assert _open(cache, slot, 4)
    assert _pages(cache, slot, 4) == [0, 1, 2, 3]
