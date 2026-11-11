from __future__ import annotations

from types import SimpleNamespace

import pytest

from cti.inference._control._scheduling.encoder_cache import EncoderCache


def test_the_least_recently_used_image_makes_room_deterministically() -> None:
    cache = EncoderCache(8)

    assert cache.plan([(b"a", 3), (b"b", 3), (b"a", 3)]) == ({}, {b"a": [0, 1, 2], b"b": [3, 4, 5]})
    assert cache.plan([(b"a", 3)]) == ({b"a": [0, 1, 2]}, {})
    # b is now the least recently used: its rows follow the 2 free ones.
    assert cache.plan([(b"c", 4)]) == ({}, {b"c": [6, 7, 3, 4]})
    # a and c are hits; b, evicted, is not.
    assert set(cache.plan([(b"a", 3), (b"c", 4)])[0]) == {b"a", b"c"}


def test_a_steps_hits_are_touched_before_its_stores_evict() -> None:
    cache = EncoderCache(8)
    cache.plan([(b"a", 3), (b"b", 3)])

    # a, the least recently used, is a hit of the step that stores c ahead of it: b goes.
    assert cache.plan([(b"c", 4), (b"a", 3)]) == ({b"a": [0, 1, 2]}, {b"c": [6, 7, 3, 4]})
    assert b"b" not in cache.plan([(b"b", 3)])[0]


def test_an_image_larger_than_the_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="an image of 5 tokens exceeds 4 encoder rows"):
        EncoderCache(4).plan([(b"a", 5)])


def test_a_reset_forgets_every_image() -> None:
    cache = EncoderCache(4)
    cache.plan([(b"a", 4)])

    cache.reset()

    # Every row is free again: a is gone.
    assert cache.plan([(b"b", 4)]) == ({}, {b"b": [0, 1, 2, 3]})


@pytest.mark.parametrize(
    ("store_rows", "src_rows", "cached"),
    [([0, 0], [0, 0], 0), ([4], [4], 0), ([1], [2], 0)],
    ids=["twice", "outside", "load-not-stored"],
)
def test_checks_refuse_a_step_whose_stores_do_not_serve_its_loads(
    store_rows: list[int], src_rows: list[int], cached: int
) -> None:
    EncoderCache(4).check(store_rows, src_rows, cached=cached)
    with pytest.raises(RuntimeError, match="stores repeat, leave the rows or miss a load"):
        EncoderCache(4, checks=True).check(store_rows, src_rows, cached=cached)


@pytest.mark.parametrize(
    ("store_rows", "src_rows", "cached"),
    [([1], [1, 1], 1), ([0], [0], 0), ([3], [2], 1)],
    ids=["cached-load", "pinned", "not-held"],
)
def test_checks_refuse_a_store_on_pinned_or_cached_rows_and_a_load_of_free_ones(
    store_rows: list[int], src_rows: list[int], cached: int
) -> None:
    cache = EncoderCache(4, checks=True)
    cache.plan([(b"a", 1), (b"b", 1)])
    cache.pin(7, b"a")

    with pytest.raises(RuntimeError, match="stores repeat, leave the rows or miss a load"):
        cache.check(store_rows, src_rows, cached=cached)


def test_pinned_and_touched_images_are_never_evicted() -> None:
    cache = EncoderCache(8)
    cache.plan([(b"a", 3), (b"b", 3), (b"c", 2)])
    cache.pin(1, b"a")

    # a is pinned and b touched: c, then nothing else, makes room for d.
    assert cache.plan([(b"b", 3), (b"d", 2)]) == ({b"b": [3, 4, 5]}, {b"d": [6, 7]})
    cache.pin(1, None)
    assert cache.plan([(b"e", 3)]) == ({}, {b"e": [0, 1, 2]})


def test_a_steps_budget_takes_each_image_once_beside_the_pinned_rows() -> None:
    cache = EncoderCache(8)
    cache.plan([(b"a", 3)])
    cache.pin(1, b"a")
    budget = cache.budget()

    # a's 3 rows are pinned; of the 5 left, b takes 4 once, and c's 2 do not fit.
    assert [budget.take(key, rows) for key, rows in [(b"a", 3), (b"b", 4), (b"b", 4)]] == [True] * 3
    assert not budget.take(b"c", 2)
    assert budget.take(b"d", 1)


def test_checks_hold_the_pins_to_the_requests_split_images() -> None:
    cache = EncoderCache(4, checks=True)
    cache.plan([(b"a", 2)])
    cache.pin(1, b"a")
    split = SimpleNamespace(uid=1, split_image=b"a")
    done = SimpleNamespace(uid=1, split_image=None)

    cache.check_pins([split])
    with pytest.raises(RuntimeError, match="pins are not the split images'"):
        cache.check_pins([done])
    with pytest.raises(RuntimeError, match="still pins images"):
        cache.reset()
