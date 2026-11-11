"""The near-tie rule for tensor-parallel against single-GPU greedy continuations: reordered
reductions may flip two nearly tied tokens, never a confident pick."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Basis: Qwen3-8B/32B TP2/TP4 on RTX PRO 6000; worst flip 0.24 nats, picks p <= 0.68,
# prefix mean 0.012.
TIE_NATS = 0.25
MAX_TIED_PICK_LOGPROB = math.log(0.75)
MEAN_PREFIX_NATS = 0.02


@dataclass(frozen=True, slots=True)
class Continuation:
    """One request's sampled tokens and each token's log-probability in its own run."""

    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ParityReport:
    identical: int
    # (request, first divergent position, |reference - candidate| log-probability there)
    divergences: tuple[tuple[int, int, float], ...]
    mean_prefix_nats: float


def first_divergences(
    reference: Sequence[Continuation],
    candidate: Sequence[Continuation],
    *,
    mean_prefix_nats: float = MEAN_PREFIX_NATS,
) -> tuple[tuple[int | None, ...], float]:
    """Each request's first position where ``candidate`` leaves ``reference`` (``None`` if
    nowhere), after checking both answered every request alike in length with finite scores;
    and the mean log-probability difference over the common prefixes, checked to be at most
    ``mean_prefix_nats``."""

    assert len(candidate) == len(reference), "the runs answered different request counts"
    positions = []
    prefix_deltas = []
    for index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        for run in (expected, actual):
            assert len(run.logprobs) == len(run.token_ids), f"request {index} is misaligned"
            assert all(math.isfinite(value) for value in run.logprobs), (
                f"request {index} has a non-finite log-probability"
            )
        assert len(actual.token_ids) == len(expected.token_ids), (
            f"request {index} generated {len(actual.token_ids)} tokens, "
            f"expected {len(expected.token_ids)}"
        )
        position = next(
            (
                position
                for position, (left, right) in enumerate(
                    zip(expected.token_ids, actual.token_ids, strict=True)
                )
                if left != right
            ),
            None,
        )
        prefix_deltas.extend(
            abs(left - right)
            for left, right in zip(
                expected.logprobs[:position], actual.logprobs[:position], strict=True
            )
        )
        positions.append(position)
    mean = sum(prefix_deltas) / len(prefix_deltas) if prefix_deltas else 0.0
    assert mean <= mean_prefix_nats, (
        f"mean common-prefix log-probability difference {mean:.4f} exceeds {mean_prefix_nats}"
    )
    return tuple(positions), mean


def assert_near_tie_parity(
    reference: Sequence[Continuation],
    candidate: Sequence[Continuation],
    *,
    tie_nats: float = TIE_NATS,
    mean_prefix_nats: float = MEAN_PREFIX_NATS,
) -> ParityReport:
    """Check ``candidate`` against ``reference`` request by request: each equals it or first
    diverges where both picks are within ``tie_nats`` and at most ``MAX_TIED_PICK_LOGPROB``;
    the common prefix's log-probabilities differ by at most ``mean_prefix_nats`` on average."""

    positions, mean = first_divergences(reference, candidate, mean_prefix_nats=mean_prefix_nats)
    divergences = []
    for index, position in enumerate(positions):
        if position is None:
            continue
        picks = (reference[index].logprobs[position], candidate[index].logprobs[position])
        delta = abs(picks[0] - picks[1])
        assert delta <= tie_nats, (
            f"request {index} flips a confident token at position {position}: its picks "
            f"differ by {delta:.3f} nats, more than {tie_nats}"
        )
        # Two runs that each pick their own token with high probability disagree on
        # the distribution itself, however close their scores are.
        assert max(picks) <= MAX_TIED_PICK_LOGPROB, (
            f"request {index} flips a confident token at position {position}: a pick "
            f"has log-probability {max(picks):.3f}, above {MAX_TIED_PICK_LOGPROB:.3f}"
        )
        divergences.append((index, position, delta))
    return ParityReport(
        identical=len(reference) - len(divergences),
        divergences=tuple(divergences),
        mean_prefix_nats=mean,
    )


__all__ = ["Continuation", "ParityReport", "assert_near_tie_parity", "first_divergences"]
