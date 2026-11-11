"""Compare recorded greedy decode logits of two serving paths, request by request."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def compare_greedy(reference: torch.Tensor, candidate: torch.Tensor) -> list[dict[str, object]]:
    """Per row of ``[steps, rows, vocab]`` greedy logits: where the paths part, each token's
    lead there, the largest greedy-token logprob gap before it and logit gap up to it."""

    reference_logprobs = reference.log_softmax(-1).amax(-1)
    candidate_logprobs = candidate.log_softmax(-1).amax(-1)
    result = []
    for row in range(reference.shape[1]):
        expected = reference[:, row]
        actual = candidate[:, row]
        parted_steps = (expected.argmax(-1) != actual.argmax(-1)).nonzero()
        parted = None if parted_steps.numel() == 0 else int(parted_steps[0])
        last = expected.shape[0] if parted is None else parted + 1
        compared = last if parted is None else parted
        leads = None
        if parted is not None:
            ours, theirs = int(expected[parted].argmax()), int(actual[parted].argmax())
            leads = [
                float(expected[parted, ours] - expected[parted, theirs]),
                float(actual[parted, theirs] - actual[parted, ours]),
            ]
        logprob_delta = (
            reference_logprobs[:compared, row] - candidate_logprobs[:compared, row]
        ).abs()
        result.append(
            {
                "parted_at": parted,
                "leads_where_parted": leads,
                "max_logprob_delta": float(logprob_delta.max()) if compared else 0.0,
                "max_abs_logit_delta": float((expected[:last] - actual[:last]).abs().max()),
            }
        )
    return result


def assert_greedy_follows(
    comparison: list[dict[str, object]],
    *,
    logprob_tolerance: float,
    max_tokens: int,
    min_compared_fraction: float,
) -> None:
    """Each request's greedy-token logprobs stay within ``logprob_tolerance`` until the paths
    part, and together the requests compare at least ``min_compared_fraction`` of their
    ``max_tokens`` steps."""

    for request in comparison:
        assert request["max_logprob_delta"] <= logprob_tolerance, request
    compared = sum(
        max_tokens if request["parted_at"] is None else request["parted_at"]
        for request in comparison
    )
    assert compared >= min_compared_fraction * max_tokens * len(comparison), comparison


__all__ = ["assert_greedy_follows", "compare_greedy"]
