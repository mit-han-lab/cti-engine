from __future__ import annotations

import pytest

from cti.attention.base import PageConstraint
from cti.kvcache import choose_page_size

_ANY = PageConstraint()
_TRTLLM = PageConstraint(frozenset({16, 32, 64}))


@pytest.mark.parametrize(
    ("constraints", "ratios", "expected"),
    [
        ((_ANY,), (1,), 1),
        ((_TRTLLM,), (1,), 16),
        ((_ANY, _TRTLLM), (1,), 16),
        ((PageConstraint(multiple=8),), (1,), 8),
        ((_TRTLLM,), (32,), 32),
        ((PageConstraint(multiple=4),), (6,), 12),
        ((), (), 1),
    ],
)
def test_unset_page_size_is_the_smallest_every_constraint_and_ratio_admits(
    constraints, ratios, expected
):
    assert choose_page_size(None, constraints, ratios) == expected


def test_an_explicit_page_size_wins_when_every_constraint_admits_it():
    assert choose_page_size(64, (_ANY, _TRTLLM), (1,)) == 64
    assert choose_page_size(3, (_ANY,), (1,)) == 3


@pytest.mark.parametrize(
    ("requested", "constraints", "ratios"),
    [(1, (_TRTLLM,), (1,)), (16, (_TRTLLM,), (64,)), (6, (_ANY,), (4,)), (0, (_ANY,), (1,))],
)
def test_an_explicit_page_size_outside_the_intersection_is_refused(requested, constraints, ratios):
    with pytest.raises(ValueError, match=f"page size {requested} does not suit"):
        choose_page_size(requested, constraints, ratios)


def test_constraints_without_a_common_page_size_are_refused():
    with pytest.raises(ValueError, match="no page size suits"):
        choose_page_size(None, (_TRTLLM, PageConstraint(frozenset({8}))), (1,))
