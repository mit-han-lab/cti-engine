from __future__ import annotations

import torch

from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.strategy import SchedulingLimits, TokenSteps
from cti.inference._step import RequestSpec
from cti.protocol import RejectionKind, SamplingParams, UserMsg
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.scheduling import request_ids


def _scheduler(*, max_seq_len: int) -> InferenceScheduler:
    return InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(
            max_running_requests=1,
            max_batch_tokens=16,
            max_sequence_length=max_seq_len,
            max_prompt_tokens=16,
        ),
        cache_manager=new_cache_manager(
            16, 1, torch.full((1, 16), -1, dtype=torch.int32), reuse=False, checks=True
        ),
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
    )


def _user_message(
    uid: int, *, input_len: int, max_tokens: int | None, min_tokens: int = 0
) -> UserMsg:
    return UserMsg(
        uid=uid,
        input_ids=torch.arange(input_len, dtype=torch.int32),
        sampling=SamplingParams(
            temperature=0, max_new_tokens=max_tokens, min_tokens=min_tokens
        ).snapshot(),
    )


def _admitted(scheduler: InferenceScheduler) -> dict[int, RequestSpec]:
    """The requests the next plan admits, by uid; none when nothing was enqueued."""

    plan = scheduler.plan_next()
    if plan is None:
        return {}
    return dict(zip(request_ids(plan), plan.admit, strict=True))


def test_admission_rejects_a_prompt_and_max_tokens_beyond_the_context_length() -> None:
    request_scheduler = _scheduler(max_seq_len=6)
    fits = _user_message(1, input_len=4, max_tokens=2)

    assert request_scheduler.handle_message(fits) == ()
    (rejection,) = request_scheduler.handle_message(_user_message(2, input_len=4, max_tokens=3))
    (prompt_too_long,) = request_scheduler.handle_message(
        _user_message(3, input_len=6, max_tokens=1)
    )

    admitted = _admitted(request_scheduler)
    assert list(admitted) == [1]
    assert (admitted[1].prompt_len, admitted[1].output_end) == (4, 6)
    assert not request_scheduler.has_runnable_work
    assert rejection.uid == 2
    assert rejection.kind is RejectionKind.CONTEXT_LENGTH_EXCEEDED
    assert rejection.reason == (
        "This model's maximum context length is 6 tokens. However, you requested 7 tokens "
        "(4 in the prompt, 3 in the completion). Please reduce the length of the prompt or "
        "completion."
    )
    assert prompt_too_long.kind is RejectionKind.CONTEXT_LENGTH_EXCEEDED


def test_an_unbounded_request_runs_to_the_end_of_the_context() -> None:
    request_scheduler = _scheduler(max_seq_len=6)

    assert request_scheduler.handle_message(_user_message(1, input_len=4, max_tokens=None)) == ()
    (full,) = request_scheduler.handle_message(_user_message(2, input_len=6, max_tokens=None))
    # The resolved budget meets the policy's rules: min_tokens beyond the room is refused.
    (short,) = request_scheduler.handle_message(
        _user_message(3, input_len=4, max_tokens=None, min_tokens=3)
    )

    admitted = _admitted(request_scheduler)
    assert list(admitted) == [1]
    # Up to the end of the context: 2 tokens past the 4-token prompt.
    assert (admitted[1].prompt_len, admitted[1].output_end) == (4, 6)
    assert full.kind is RejectionKind.CONTEXT_LENGTH_EXCEEDED
    assert short.kind is RejectionKind.INVALID_REQUEST
    assert "min_tokens" in short.reason


def test_a_logit_bias_outside_the_vocabulary_is_rejected_at_admission() -> None:
    # vLLM's: an out-of-vocabulary bias is an error, not a silent drop.
    request_scheduler = _scheduler(max_seq_len=16)
    sampling = SamplingParams(max_new_tokens=4, temperature=0, logit_bias={3: 1.0, 1 << 20: 2.0})
    message = UserMsg(
        uid=9, input_ids=torch.ones(2, dtype=torch.int32), sampling=sampling.snapshot()
    )

    (rejection,) = request_scheduler.handle_message(message)

    assert rejection.kind is RejectionKind.INVALID_REQUEST
    assert f"[{1 << 20}] are outside the vocabulary" in rejection.reason
    assert _admitted(request_scheduler) == {}


def test_empty_prompt_is_rejected_at_admission() -> None:
    """An empty prompt (an empty string tokenizes to no tokens) is a rejection: admitted,
    constructing its request raises inside the inference loop."""
    request_scheduler = _scheduler(max_seq_len=16)
    message = UserMsg(
        uid=8,
        input_ids=torch.empty(0, dtype=torch.int32),
        sampling=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
    )

    (rejection,) = request_scheduler.handle_message(message)

    assert rejection.uid == 8
    assert rejection.kind is RejectionKind.INVALID_REQUEST
    assert "at least one token" in rejection.reason
    assert not request_scheduler.has_runnable_work
    assert _admitted(request_scheduler) == {}
