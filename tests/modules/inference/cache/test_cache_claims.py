"""DFlash round uses: bounded lookahead pages, FIFO settlement after forward uses.

The cases use absolute per-round ends and committed lengths.
"""

from __future__ import annotations

import pytest
import torch

from cti.inference._control._cache import CacheManager
from cti.inference._control._scheduling.table import RequestSlot
from cti.inference._step import QueryKind, Row
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.prefix_cache import reference_radix_comparison  # noqa: F401
from tests.support.inference.scheduling import page_triples


def _cache(page_size: int) -> CacheManager:
    return new_cache_manager(
        256, page_size, torch.full((3, 256), -1, dtype=torch.int32), reuse=False, checks=True
    )


def _entry(slot: RequestSlot, start: int, end: int) -> Row:
    return Row(slot.index, QueryKind.EXTEND, start, end, end)


def _prefill(cache: CacheManager, slot: RequestSlot, length: int = 17) -> None:
    prompt = torch.arange(length, dtype=torch.int32)
    assert cache.try_open_slot(slot, prompt, 100, reserved_size=0) == 0
    (use,), _ = cache.reserve([slot], [_entry(slot, 0, length)])
    cache.settle(use, length)


def _used_pages(cache: CacheManager, length: int) -> int:
    """The pages one slot holding cache below ``length`` uses."""
    return -(-length // cache.page_size)


def _assert_frontier(cache: CacheManager, slot: RequestSlot, length: int) -> None:
    """A forward continues ``slot`` from ``length`` exactly: a quiescent slot's frontier."""
    with pytest.raises(RuntimeError, match="does not continue"):
        cache.reserve([slot], [_entry(slot, length + 1, length + 1)])
    (probe,), _ = cache.reserve([slot], [_entry(slot, length, length)])
    cache.settle(probe, 0)


def _free_pages(cache: CacheManager, probe: RequestSlot) -> set[int]:
    """The pages the cache hands out before evicting: reserved by ``probe``, then returned."""
    occupancy = cache.occupancy
    free = occupancy.num_pages - occupancy.num_used_pages - occupancy.num_cached_pages
    prompt = torch.zeros(1, dtype=torch.int32)
    assert cache.try_open_slot(probe, prompt, 0, reserved_size=0, prefix=False) == 0
    end = free * cache.page_size
    (use,), _ = cache.reserve([probe], [_entry(probe, 0, end)])
    pages = set(cache.page_table[probe.index, :free].tolist())
    cache.settle(use, 0)
    cache.close_slot(probe, prefix_ids=prompt[:0])
    assert len(pages) == free
    return pages


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("q", [8, 16])
def test_settling_the_first_claim_keeps_the_pages_of_a_third(page_size: int, q: int) -> None:
    cache, slot = _cache(page_size), RequestSlot(0, 0)
    _prefill(cache, slot)
    available = cache.available_size

    (first,), _ = cache.reserve_rounds([slot], [17 + q])
    (second,), _ = cache.reserve_rounds([slot], [17 + 2 * q])
    with pytest.raises(RuntimeError, match="already holds 2"):
        cache.reserve_rounds([slot], [17 + 3 * q])

    cache.settle(first, q)
    # The second round keeps its capacity; the third extends from the new frontier.
    assert cache.occupancy.num_used_pages == _used_pages(cache, 17 + 2 * q)
    (third,), _ = cache.reserve_rounds([slot], [17 + 3 * q])
    third_pages = cache.page_table[slot.index, : _used_pages(cache, 17 + 3 * q)].clone()
    cache.settle(second, 1)

    assert cache.occupancy.num_used_pages == _used_pages(cache, 17 + 3 * q)
    assert torch.equal(cache.page_table[slot.index, : len(third_pages)], third_pages)
    cache.settle(third, 0)
    assert cache.occupancy.num_used_pages == _used_pages(cache, 18 + q)
    _assert_frontier(cache, slot, 18 + q)
    reserved_pages = -(-(18 + q) // page_size) - -(-17 // page_size)
    assert cache.available_size == available - reserved_pages * page_size
    cache.close_slot(slot, prefix_ids=torch.arange(18 + q, dtype=torch.int32))
    cache.check_integrity()


@pytest.mark.parametrize("page_size", [1, 16])
def test_rounds_follow_a_forward_and_settle_after_it(page_size: int) -> None:
    cache, slot = _cache(page_size), RequestSlot(0, 0)
    prompt = torch.arange(17, dtype=torch.int32)
    cache.try_open_slot(slot, prompt, 100, reserved_size=0)
    (forward,), _ = cache.reserve([slot], [_entry(slot, 0, 17)])
    # A prompt's first round may follow its forward before the forward settles.
    (round_use,), _ = cache.reserve_rounds([slot], [33])
    with pytest.raises(RuntimeError, match="out of order"):
        cache.settle(round_use, 1)
    cache.settle(forward, 17)

    assert cache.in_use(slot)
    with pytest.raises(RuntimeError, match="does not continue"):
        cache.reserve([slot], [_entry(slot, 33, 34)])
    cache.settle(round_use, 0)
    assert not cache.in_use(slot)
    cache.close_slot(slot, prefix_ids=prompt)
    cache.check_integrity()


@pytest.mark.parametrize("in_flight", [1, 2])
def test_with_prefix_reuse_a_round_waits_for_the_forward(in_flight: int) -> None:
    # Rounds may follow an unsettled forward and settle after it. The duplicate page its
    # publication rewires returns once the rounds then in flight settle; a later round
    # carries the rewiring instead of holding the page.
    cache = new_cache_manager(
        64, 2, torch.full((3, 64), -1, dtype=torch.int32), reuse=True, checks=True
    )
    slots = RequestSlot(0, 0), RequestSlot(1, 0)
    probe = RequestSlot(2, 0)
    ids = torch.tensor([[1, 2, 3, 4, 9, 9], [1, 2, 5, 6, 7, 8]], dtype=torch.int32)
    for slot in slots:
        assert cache.try_open_slot(slot, ids[slot.index, :4], 8, reserved_size=0) == 0
    forwards, _ = cache.reserve(slots, [_entry(slot, 0, 4) for slot in slots])
    canonical, duplicate = cache.page_table[:2, 0].tolist()
    rounds = [cache.reserve_rounds([slots[1]], [8 + 4 * n])[0][0] for n in range(in_flight)]

    for forward in forwards:
        cache.settle(forward, 4, prefix_ids=ids[forward.slot.index, :4])
    assert cache.page_table[1, 0] == canonical
    *earlier, last = rounds
    for round_use in earlier:
        cache.settle(round_use, 1)
    (later,), updates = cache.reserve_rounds([slots[1]], [16])
    triples = page_triples(updates)
    assert triples[triples[:, 1] == 0, 2].tolist() == [canonical]
    assert duplicate not in _free_pages(cache, probe)
    cache.settle(last, 1)
    assert duplicate in _free_pages(cache, probe)
    assert canonical not in _free_pages(cache, probe)

    cache.settle(later, 0)
    for slot in slots:
        cache.close_slot(slot, prefix_ids=ids[slot.index])
    cache.check_integrity()


def test_admission_counts_the_round_lookahead() -> None:
    cache = new_cache_manager(
        10, 1, torch.full((2, 32), -1, dtype=torch.int32), reuse=False, checks=True
    )
    prompt = torch.arange(4, dtype=torch.int32)

    assert cache.try_open_slot(RequestSlot(0, 0), prompt, 2, reserved_size=0) == 0
    assert (
        cache.try_open_slot(RequestSlot(1, 0), prompt, 2, reserved_size=0, lookahead_tokens=5)
        is None
    )


def test_a_failure_after_acceptance_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cache, slot = _cache(1), RequestSlot(0, 0)
    _prefill(cache, slot)
    (round_use,), _ = cache.reserve_rounds([slot], [33])

    def fail(*_args: object) -> None:
        raise RuntimeError("injected page-release failure")

    # No public input makes a page release fail; inject it at the private seam.
    monkeypatch.setattr(cache, "_release_pages", fail)
    with pytest.raises(RuntimeError, match="injected"):
        cache.settle(round_use, 1)
