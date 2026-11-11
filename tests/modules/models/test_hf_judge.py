"""The parity judge's greedy report and near-tie rule: a greedy divergence is reported, never
gated, with whether the truth reached a near tie first; a margin below the step's bound (two
BF16 steps at the top logit, at least ``NEAR_TIE_NATS``) is a near tie and one at the bound is
decided; the receipt records every bound."""

from __future__ import annotations

import json

import pytest
import torch

from tests.support.hf_judge import (
    BOUNDS,
    NEAR_TIE_NATS,
    bf16_spacing,
    compare_greedy,
    is_near_tie,
    near_tie_nats,
)

_STEPS = 12


def _truth(near_tie: int | None) -> dict[str, torch.Tensor]:
    margins = torch.full((_STEPS,), 4 * NEAR_TIE_NATS)
    if near_tie is not None:
        margins[near_tie] = NEAR_TIE_NATS / 2
    return {
        "greedy_ids": torch.arange(_STEPS),
        "greedy_margins": margins,
        "near_tie_nats": torch.full((_STEPS,), NEAR_TIE_NATS),
    }


@pytest.mark.parametrize(
    ("near_tie", "divergence", "agrees"),
    [(None, 5, False), (3, 5, True), (7, 5, False), (5, 5, True)],
    ids=["no-near-tie", "near-tie-first", "near-tie-after", "near-tie-at-divergence"],
)
def test_a_greedy_divergence_is_reported_with_the_truths_first_near_tie(
    near_tie: int | None, divergence: int, agrees: bool
) -> None:
    truth = _truth(near_tie)
    candidate = {"greedy_ids": truth["greedy_ids"].clone()}
    candidate["greedy_ids"][divergence:] += 100

    report = compare_greedy(truth, candidate)

    assert report["agreed"] == divergence
    assert report["divergence"]["step"] == divergence
    assert report["truth_first_near_tie"] == near_tie
    assert report["agrees_until_truth_near_tie"] is agrees
    assert json.loads(json.dumps(report)) == report


def test_identical_continuations_report_no_divergence() -> None:
    truth = _truth(None)

    report = compare_greedy(truth, {"greedy_ids": truth["greedy_ids"].clone()})

    assert (report["agreed"], report["divergence"]) == (_STEPS, None)
    assert report["agrees_until_truth_near_tie"]
    with pytest.raises(ValueError, match="greedy continuations"):
        compare_greedy(truth, {"greedy_ids": truth["greedy_ids"][:-1]})


def test_near_ties_are_strictly_below_the_threshold() -> None:
    margins = torch.tensor([0.0, NEAR_TIE_NATS - 1e-3, NEAR_TIE_NATS, 1.0])

    assert is_near_tie(margins, NEAR_TIE_NATS).tolist() == [True, True, False, False]
    assert is_near_tie(0.3, 0.5)
    assert set(BOUNDS) >= {"calibration_factor", "max_error_floor", "near_tie_bf16_spacings"}


def test_bf16_spacing_is_the_step_between_adjacent_bf16_numbers() -> None:
    values = torch.tensor([1.0, 1.5, 15.5, 16.0, 36.75, -36.75, 64.0, 100.0])

    spacing = bf16_spacing(values)

    assert spacing.tolist() == [2**-7, 2**-7, 2**-4, 2**-3, 2**-2, 2**-2, 2**-1, 2**-1]
    # Adding the spacing reaches the next BF16 number; half of it rounds back.
    assert torch.equal((values + spacing).bfloat16().float() - values, spacing)
    assert torch.equal((values + spacing / 4).bfloat16().float(), values.bfloat16().float())


def test_the_near_tie_bound_is_two_bf16_steps_at_the_top_logit_and_at_least_the_floor() -> None:
    # Qwen3-8B, H100: the truth's top two at 36.94 and 36.67 (0.27 nats) are a near tie.
    bounds = near_tie_nats(torch.tensor([3.0, 20.0, 36.94, 70.0]))

    assert bounds.tolist() == [NEAR_TIE_NATS, NEAR_TIE_NATS, 0.5, 1.0]
    assert is_near_tie(36.94 - 36.67, bounds[2])
    assert near_tie_nats(torch.tensor([70.0]), floor=1.5).tolist() == [1.5]
