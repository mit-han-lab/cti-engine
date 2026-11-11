"""Vision graph packing: images sorted by tokens, packed greedily into the smallest budget
that holds them, padding one sequence of its own, and images past the largest budget eager."""

from __future__ import annotations

import pytest

from cti.inference._execution._vision_graph import VisionBudgets, VisionPack

BUDGETS = VisionBudgets((64, 128, 256, 1000))


@pytest.mark.parametrize(
    ("tokens", "images", "expected"),
    [
        ((1,), 1, 64),
        ((64,), 1, 64),
        ((65,), 1, 128),
        ((256,), 1, 256),
        ((257,), 1, 1000),
        ((1000,), 1, 1000),
        ((1001,), 1, None),
        # At most budget // 64 images, whatever their tokens.
        ((10,), 2, 128),
        ((10,), 4, 256),
        ((10,), 5, 1000),
        ((10,), 16, None),
    ],
)
def test_select_takes_the_smallest_budget_holding_the_tokens_and_images(
    tokens: tuple[int], images: int, expected: int | None
) -> None:
    assert BUDGETS.select(tokens[0], images) == expected


def test_a_budget_runs_as_many_images_as_it_holds_of_the_smallest() -> None:
    assert [BUDGETS.max_images(budget) for budget in BUDGETS.sizes] == [1, 2, 4, 15]


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        # One image: the smallest budget that holds it.
        ((100,), (VisionPack(128, (0,)),)),
        # Sorted ascending, all in one pack.
        ((100, 64, 80), (VisionPack(256, (1, 2, 0)),)),
        # The next image would overflow the largest budget: it opens a pack.
        ((600, 300, 200), (VisionPack(1000, (2, 1)), VisionPack(1000, (0,)))),
        # The largest budget holds 15 images: the 16th opens a pack.
        ((64,) * 16, (VisionPack(1000, tuple(range(15))), VisionPack(64, (15,)))),
        # Images past the largest budget run eagerly alone, last.
        (
            (2000, 64, 1500, 64),
            (VisionPack(128, (1, 3)), VisionPack(None, (2,)), VisionPack(None, (0,))),
        ),
        ((), ()),
    ],
)
def test_pack_sorts_and_packs_greedily_with_overflow_eager(
    tokens: tuple[int, ...], expected: tuple[VisionPack, ...]
) -> None:
    assert BUDGETS.pack(tokens) == expected


def test_every_image_lands_in_exactly_one_pack_that_holds_it() -> None:
    tokens = (64, 1200, 77, 300, 64, 999, 5, 256, 64, 64, 128)
    packs = BUDGETS.pack(tokens)
    assert sorted(image for pack in packs for image in pack.images) == list(range(len(tokens)))
    for pack in packs:
        total = sum(tokens[image] for image in pack.images)
        if pack.budget is None:
            assert pack.images == (pack.images[0],) and total > BUDGETS.sizes[-1]
        else:
            assert pack.budget == BUDGETS.select(total, len(pack.images))


def test_boundaries_pad_as_one_sequence_then_empty_ones() -> None:
    # Budget 256 of 4 patches a token: 1024 patches, 4 images at most, 6 bounds.
    assert BUDGETS.boundaries(256, [64, 320], 4) == [0, 64, 384, 1024, 1024, 1024]
    # An exact fit leaves the padding sequence empty.
    assert BUDGETS.boundaries(256, [256] * 4, 4) == [0, 256, 512, 768, 1024, 1024]
    assert BUDGETS.boundaries(64, [100], 4) == [0, 100, 256]
