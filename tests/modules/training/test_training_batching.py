from __future__ import annotations

import pytest

from cti.config import (
    BatchLayout,
    DynamicBatching,
    StaticBatching,
    TrainingBatching,
)
from cti.training._batching import plan_microbatches


@pytest.mark.parametrize(
    ("batching", "expected"),
    [
        (
            TrainingBatching(
                BatchLayout.PADDED,
                StaticBatching(microbatch_size=2, max_tokens_per_sample=8),
            ),
            ((0, 1), (2, 3), (4,)),
        ),
        (
            TrainingBatching(
                BatchLayout.PACKED,
                StaticBatching(microbatch_size=3, max_tokens_per_sample=8),
            ),
            ((0, 1, 2), (3, 4)),
        ),
        (
            TrainingBatching(
                BatchLayout.PACKED,
                DynamicBatching(max_tokens_per_microbatch=7),
            ),
            ((0, 1), (2, 3), (4,)),
        ),
    ],
)
def test_plans_each_valid_layout_policy_combination(
    batching: TrainingBatching,
    expected: tuple[tuple[int, ...], ...],
) -> None:
    assert plan_microbatches(batching.policy, (3, 4, 5, 2, 3)) == expected


def test_dynamic_batching_preserves_input_order() -> None:
    policy = DynamicBatching(max_tokens_per_microbatch=8)

    assert plan_microbatches(policy, (6, 3, 5, 2)) == ((0,), (1, 2), (3,))


@pytest.mark.parametrize(
    ("batching", "field"),
    [
        (
            TrainingBatching(
                BatchLayout.PACKED,
                StaticBatching(microbatch_size=2, max_tokens_per_sample=4),
            ),
            "max_tokens_per_sample",
        ),
        (
            TrainingBatching(
                BatchLayout.PACKED,
                DynamicBatching(max_tokens_per_microbatch=4),
            ),
            "max_tokens_per_microbatch",
        ),
    ],
)
def test_rejects_an_overbudget_sample(
    batching: TrainingBatching,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        plan_microbatches(batching.policy, (3, 5))


@pytest.mark.parametrize("value", (0, -1))
def test_batching_parameters_must_be_positive(value: int) -> None:
    with pytest.raises(ValueError, match="max_tokens_per_microbatch"):
        DynamicBatching(value)


def test_static_batching_bounds_a_microbatch_by_every_sample_at_its_limit() -> None:
    policy = StaticBatching(microbatch_size=3, max_tokens_per_sample=32)

    assert policy.max_tokens_per_microbatch == 96


def test_padded_dynamic_batching_is_invalid() -> None:
    with pytest.raises(ValueError, match="requires packed layout"):
        TrainingBatching(
            BatchLayout.PADDED,
            DynamicBatching(max_tokens_per_microbatch=8),
        )


def test_empty_logical_batch_has_no_microbatches() -> None:
    policy = DynamicBatching(max_tokens_per_microbatch=8)

    assert plan_microbatches(policy, ()) == ()
