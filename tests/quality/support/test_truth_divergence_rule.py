"""Two greedy runs may first differ only where both tokens are a truth near tie with its top-1; a
free run is scored by the decided positions where it leaves the truth's top-1."""

from __future__ import annotations

import pytest
import torch

from tests.support.hf_judge import NEAR_TIE_NATS
from tests.support.reference_truth import free_run_flips, judge_divergence


def _truth(*rows: tuple[tuple[int, ...], tuple[float, ...]]) -> dict:
    return {
        "topk_ids": torch.tensor([ids for ids, _ in rows]),
        "topk_logits": torch.tensor([logits for _, logits in rows]),
    }


@pytest.mark.parametrize(
    ("row", "tokens", "allowed"),
    [
        (((7, 9, 4, 3), (0.0, -0.1, -5.0, -6.0)), (9, 7), True),
        (((7, 9, 4, 3), (0.0, -NEAR_TIE_NATS, -5.0, -6.0)), (7, 9), False),  # the bound is decided
        (((7, 9, 4, 3), (0.0, -0.02, -0.1, -6.0)), (7, 4), True),  # a three-way tie
        (((7, 9, 4, 3), (0.0, -0.1, -0.3, -6.0)), (7, 4), False),  # a third token past the bound
        (((7, 9, 4, 3), (0.0, -0.1, -0.2, -0.2)), (7, 8), False),  # outside the truth's top four
        (((7, 9, 4, 3), (0.0, -2.0, -5.0, -6.0)), (7, 9), False),
    ],
    ids=[
        "near-tie",
        "at-the-bound",
        "three-way-tie",
        "third-past-bound",
        "outside-top-four",
        "decided",
    ],
)
def test_a_divergence_is_allowed_only_at_a_truth_near_tie_of_its_tokens(row, tokens, allowed):
    truth = _truth(((5, 6, 1, 2), (3.0, 0.0, -1.0, -2.0)), row)

    ruling = judge_divergence(truth, 1, tokens)

    assert ruling["allowed"] is allowed
    assert ruling["truth_top"] == list(row[0])


def test_a_free_run_counts_only_decided_positions_off_the_truths_top_one() -> None:
    sequence = ((1, 2), (5, 7, 8, 4))
    entry = {
        "greedy_ids": torch.tensor([5, 7, 3, 4]),
        "greedy_margins": torch.tensor([3.0, 0.1, 1.0, 2.0]),
        "near_tie_nats": torch.full((4,), NEAR_TIE_NATS),
    }

    # Position 1 leaves a near tie, position 2 a decided top-1.
    assert free_run_flips({sequence: entry}, [sequence]) == {
        "decided": 3,
        "decided_disagreements": 1,
    }
    # The same request twice is one sample; another output shares only its common prefix.
    other = ((1, 2), (5, 9, 8, 6))
    truth = {sequence: entry, other: entry}
    assert free_run_flips(truth, [sequence, sequence, other]) == {
        "decided": 5,
        "decided_disagreements": 3,
    }
    with pytest.raises(ValueError, match="tokens for"):
        free_run_flips({((1,), (5,)): entry}, [((1,), (5,))])
