from __future__ import annotations

import math

import pytest

from tests.support.tp_parity import Continuation, assert_near_tie_parity


def _run(*steps: tuple[int, float]) -> Continuation:
    return Continuation(tuple(token for token, _ in steps), tuple(score for _, score in steps))


_REFERENCE = (
    _run((5, -0.1), (8, -0.6), (2, -0.3)),
    _run((7, -0.2), (9, -0.05), (4, -1.2)),
)


def test_identical_continuations_pass() -> None:
    report = assert_near_tie_parity(_REFERENCE, _REFERENCE)

    assert report.identical == 2
    assert report.divergences == ()
    assert report.mean_prefix_nats == 0.0


def test_a_near_tie_flip_passes_and_is_reported() -> None:
    # Two tokens near p = 0.5 swap at position 1; later tokens may differ freely.
    candidate = (_run((5, -0.11), (3, -0.75), (6, -3.0)), _REFERENCE[1])

    report = assert_near_tie_parity(_REFERENCE, candidate)

    assert report.identical == 1
    assert report.divergences == (pytest.approx((0, 1, 0.15)),)
    assert report.mean_prefix_nats == pytest.approx(0.01 / 4)


def test_a_confident_flip_fails() -> None:
    candidate = (_REFERENCE[0], _run((7, -0.2), (1, -2.5), (4, -1.2)))

    with pytest.raises(AssertionError, match="request 1 flips a confident token at position 1"):
        assert_near_tie_parity(_REFERENCE, candidate)


def test_two_confident_picks_fail_even_with_equal_scores() -> None:
    reference = (_run((1, -0.01), (2, -0.02), (3, -0.01)),)
    candidate = (_run((1, -0.01), (9, -0.02), (9, -0.01)),)

    with pytest.raises(AssertionError, match="request 0 flips a confident token at position 1"):
        assert_near_tie_parity(reference, candidate)


def test_a_drifting_common_prefix_fails() -> None:
    candidate = tuple(
        Continuation(run.token_ids, tuple(score - 0.05 for score in run.logprobs))
        for run in _REFERENCE
    )

    with pytest.raises(AssertionError, match="mean common-prefix"):
        assert_near_tie_parity(_REFERENCE, candidate)


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ((_REFERENCE[0],), "different request counts"),
        ((_REFERENCE[0], _run((7, -0.2), (9, -0.05))), "generated 2 tokens, expected 3"),
        ((_REFERENCE[0], Continuation((7, 9, 4), (-0.2, -0.05))), "request 1 is misaligned"),
        ((_REFERENCE[0], _run((7, -0.2), (9, math.nan), (4, -1.2))), "non-finite"),
    ],
    ids=["request-count", "length", "scores", "nan"],
)
def test_misaligned_or_non_finite_runs_fail(
    candidate: tuple[Continuation, ...], message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        assert_near_tie_parity(_REFERENCE, candidate)
