"""The parity judge's bounds: an engine against Transformers in FP32 (the truth), each bound
calibrated per prompt by BF16 Transformers' distance from the same truth
(``tests/support/hf_parity.py`` applies them to teacher-forced top-1 rows), and the greedy
report beside them.

BF16 Transformers is no oracle on long prompts: on Qwen3.5-9B's 756-token ``lighthouse-log`` it
flipped 5 of 680 decided positions, its top-1 logprob off by up to 3.97 nats, and at two of them
FP32 sided with the engine. ``NEAR_TIE_NATS`` bounds BF16-vs-BF16 drift, not BF16-vs-FP32: on
Qwen3.8-27B's ``system-noble-gases`` both left the truth at a 0.34-nat margin.

A near tie is a top-two margin the engine's BF16 logits cannot resolve: below
``NEAR_TIE_BF16_SPACINGS`` steps of the BF16 grid at the top logit, and never below
``NEAR_TIE_NATS``. The grid coarsens with the logit: on Qwen3-8B (H100, DFlash prefix gate) the
truth's top two were 36.94 and 36.67, a 0.27-nat margin where BF16 steps by 0.25, and the
engine's own decode, DFlash or not, picked either one."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

# Near-tie reporting and teacher-forced decided-position cutoff. BF16 can overstate FP32
# margins (two-images step 6: 0.375 vs 0.226), so greedy identity is report-only.
NEAR_TIE_NATS = 0.25
# Two steps of the BF16 grid at the top logit: a flip of one step each way (accumulation order).
NEAR_TIE_BF16_SPACINGS = 2
# Engine bound: this times BF16 Transformers' distance from the truth, plus a slack for prompts
# where that distance is about zero (a bare ratio of two tiny numbers fails on noise).
CALIBRATION_FACTOR = 2.0
# Decided flips (truth margin >= NEAR_TIE_NATS) beyond the calibrated count: a flip just past
# the margin is BF16 rounding (BF16 Transformers: 5 of 680 on lighthouse-log).
DISAGREEMENT_SLACK = 2
# Truth top-1 logprob error, nats. Qwen3.5-9B short prompts: the engine within 0.014 (mean) and
# 0.1 (p99) of BF16 Transformers.
MEAN_ERROR_SLACK = 0.02
P99_ERROR_SLACK = 0.1
# One position on a rounding boundary reaches 4 nats even in BF16 Transformers (lighthouse-log),
# so the maximum bounds gross faults only: twice BF16's own, never below this floor.
MAX_ERROR_FLOOR = 2.0
# Above these the BF16 calibration is broken and would loosen every bound. Qwen3.5-9B, RTX PRO
# 6000: BF16 at most 7/922 decided flips, mean error 0.032.
BF16_MAX_DISAGREEMENT_SHARE = 0.02
BF16_MAX_MEAN_ERROR = 0.1
# Decode through CUDA graphs against eager decode, logprob nats. Not bitwise: eager decode runs
# the real batch size, its graph the padded one, so GEMM tiles and attention plans may differ.
# RTX PRO 6000 receipts: Qwen3.8-27B 2.1e-5 (128 slots), Qwen3.5-9B up to 0.096, but two kernel
# development builds reached 0.108 (256 slots) and 0.10066 (16).
GRAPH_LOGPROB_TOLERANCE = 0.1
# Recorded in the parity receipt, beside NEAR_TIE_NATS.
BOUNDS: Mapping[str, float] = MappingProxyType(
    {
        "calibration_factor": CALIBRATION_FACTOR,
        "disagreement_slack": DISAGREEMENT_SLACK,
        "mean_error_slack": MEAN_ERROR_SLACK,
        "p99_error_slack": P99_ERROR_SLACK,
        "max_error_floor": MAX_ERROR_FLOOR,
        "bf16_max_disagreement_share": BF16_MAX_DISAGREEMENT_SHARE,
        "bf16_max_mean_error": BF16_MAX_MEAN_ERROR,
        "near_tie_bf16_spacings": NEAR_TIE_BF16_SPACINGS,
    }
)


def bf16_spacing(values) -> torch.Tensor:
    """The step between adjacent BF16 numbers at each of ``values``: BF16 keeps 8 significant
    bits, so ``|x|`` in ``[2**(e - 1), 2**e)`` steps by ``2**(e - 8)``."""

    _, exponent = torch.frexp(torch.as_tensor(values, dtype=torch.float32).abs())
    return torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent - 8)


def near_tie_nats(top_logits, floor: float = NEAR_TIE_NATS) -> torch.Tensor:
    """The near-tie bound at each step whose top logit is ``top_logits``: two BF16 steps there,
    never below ``floor``."""

    return (NEAR_TIE_BF16_SPACINGS * bf16_spacing(top_logits)).clamp(min=floor)


def compare_greedy(
    truth: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Report greedy agreement and near ties (below the truth's ``near_tie_nats`` per step);
    token identity does not gate parity."""

    truth_ids, margins, bounds = (
        truth["greedy_ids"],
        truth["greedy_margins"],
        truth["near_tie_nats"],
    )
    ids = candidate["greedy_ids"]
    if ids.shape != truth_ids.shape:
        raise ValueError(f"greedy continuations of {tuple(ids.shape)} and {tuple(truth_ids.shape)}")
    mismatches = (ids != truth_ids).nonzero().flatten().tolist()
    near_ties = is_near_tie(margins, bounds).nonzero().flatten().tolist()
    divergence = None
    if mismatches:
        step = mismatches[0]
        divergence = {
            "step": step,
            "truth_margin": float(margins[step]),
            "near_tie_nats": float(bounds[step]),
            "truth_token": int(truth_ids[step]),
            "token": int(ids[step]),
        }
    return {
        "tokens": len(truth_ids),
        "agreed": mismatches[0] if mismatches else len(truth_ids),
        "divergence": divergence,
        "truth_near_tie_steps": len(near_ties),
        "truth_first_near_tie": near_ties[0] if near_ties else None,
        "agrees_until_truth_near_tie": not mismatches
        or bool(near_ties and near_ties[0] <= mismatches[0]),
    }


def is_near_tie(margin, nats):
    """Whether a top-two margin in nats (a number or a tensor of them) is a near tie: strictly
    below ``nats`` (``near_tie_nats``' bound there); a margin of ``nats`` or more is decided."""

    return torch.as_tensor(margin) < torch.as_tensor(nats)
