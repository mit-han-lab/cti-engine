"""The DFlash MoE gate's teacher-forced rule (expert TP, the radix cache) judges each
configuration's own top-1s on the plain engine's tokens: decided positions only, each distinct
position once."""

from __future__ import annotations

import torch

from tests.support.dflash_moe_gate import TEACHER_FORCED, _teacher_forced_flips
from tests.support.hf_judge import NEAR_TIE_NATS


def test_teacher_forced_flips_count_decided_positions_once() -> None:
    sequence = ((1, 2), (5, 6, 7))
    other = ((1, 2), (5, 8, 9))
    entry = {
        "greedy_ids": torch.tensor([5, 6, 7]),
        "greedy_margins": torch.tensor([2.0, NEAR_TIE_NATS / 2, 3.0]),
        "near_tie_nats": torch.full((3,), NEAR_TIE_NATS),
    }
    truth = {sequence: entry, other: {**entry, "greedy_ids": torch.tensor([5, 8, 9])}}

    def scored(key, greedy_ids):
        return {"prompt": key[0], "output": key[1], "greedy_ids": greedy_ids}

    # Step 1 is a near tie; step 2 leaves the truth's top-1.
    assert _teacher_forced_flips(truth, [scored(sequence, [5, 8, 4])]) == {
        "decided": 2,
        "decided_disagreements": 1,
    }
    # The same sequence twice is one sample; another one shares only its common prefix.
    assert _teacher_forced_flips(
        truth,
        [scored(sequence, [5, 6, 4]), scored(sequence, [5, 6, 4]), scored(other, [5, 8, 9])],
    ) == {"decided": 3, "decided_disagreements": 1}


def test_expert_tp_and_the_radix_cache_are_judged_on_identical_tokens() -> None:
    assert {"expert-tp", "radix-prefix"} <= TEACHER_FORCED
