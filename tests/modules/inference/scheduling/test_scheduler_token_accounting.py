"""Settlement: which sampled tokens a request keeps, what it publishes, and its usage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cti.inference._control._cache import CacheManager
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.strategy import RequestTotals
from cti.inference._step import QueryKind, Row
from cti.protocol import (
    AbortBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    DetokenizeMsg,
    RequestAbortedMsg,
    RequestEnd,
    TokenUsage,
    UserReply,
)
from cti.resolution import ResolvedModel
from tests.support.inference.inference_steps import (
    cache_manager,
    engine,
    scheduler,
    settle,
    submit,
    user_msg,
)
from tests.support.inference.scheduling import request_ids, reservations, sampled

_STOP_TOKEN_IDS = (2, 5)
_PROMPT = [10, 11]


def _ints(data: bytes) -> list[int]:
    return list(memoryview(data).cast("i"))


def _scheduler(cache: CacheManager | None = None, **limits: int) -> InferenceScheduler:
    return scheduler(stop_token_ids=_STOP_TOKEN_IDS, cache=cache, **limits)


def _serving(
    max_tokens: int, *, ignore_eos: bool = True, cache: CacheManager | None = None
) -> InferenceScheduler:
    sched = _scheduler(cache)
    sched.handle_message(user_msg(7, _PROMPT, max_tokens=max_tokens, ignore_eos=ignore_eos))
    return sched


def _spy_settle(cache: CacheManager, check) -> list[dict]:
    """Run ``check(**call)`` before each settlement of the scheduler's ``cache``; each call's
    use, committed count and keywords."""

    settle, calls = cache.settle, []

    def spy(use, committed, **kwargs):
        call = {"use": use, "committed": committed, **kwargs}
        calls.append(call)
        check(**call)
        return settle(use, committed, **kwargs)

    cache.settle = spy
    return calls


def _assert_reclaimed(sched: InferenceScheduler) -> None:
    assert not sched.has_runnable_work
    sched.require_quiescent()


def test_cache_settles_before_the_request_takes_its_token() -> None:
    cache = cache_manager()
    sched = _serving(2, cache=cache)
    step = submit(sched)
    request = step.rows[0].request

    def unchanged(**_kwargs) -> None:
        assert request.token_ids.tolist() == _PROMPT

    _spy_settle(cache, unchanged)

    assert [reply.token_ids for reply in settle(sched, step, 20)] == [(20,)]
    assert request.token_ids.tolist() == [*_PROMPT, 20]


def test_cache_settlement_failure_changes_no_request_and_publishes_nothing() -> None:
    cache = cache_manager()
    sched = _scheduler(cache)
    sched.handle_message(user_msg(7, _PROMPT))
    sched.handle_message(user_msg(8, _PROMPT))
    step = submit(sched)
    failure = RuntimeError("cache settlement failed")

    def fail(**_kwargs):
        raise failure

    _spy_settle(cache, fail)
    with pytest.raises(RuntimeError) as raised:
        sched.settle_step(step, sampled([20, 21]))

    assert raised.value is failure
    for entry in step.rows:
        assert entry.request.token_ids.tolist() == _PROMPT
        assert not entry.request.is_finished


def test_overlapped_step_settles_the_prefix_its_predecessor_appended() -> None:
    cache = cache_manager()
    sched = _serving(2, cache=cache)
    first = submit(sched)
    successor = submit(sched)
    request = first.rows[0].request
    reservation = reservations(successor)[0]
    assert (reservation.start, reservation.end) == (2, 3)
    settle(sched, first, 20)
    calls = _spy_settle(cache, lambda **_: None)

    settle(sched, successor, 21)

    [call] = calls
    assert call["use"] is reservation
    assert call["prefix_ids"].tolist() == [*_PROMPT, 20]
    assert request.token_ids.tolist() == [*_PROMPT, 20, 21]


@pytest.mark.parametrize("mismatch", ["token count", "committed frontier"])
def test_a_step_that_does_not_match_its_claim_raises_with_nothing_changed(mismatch: str) -> None:
    cache = cache_manager()
    sched = _serving(2, cache=cache)
    step = submit(sched)
    request = step.rows[0].request
    tokens = [20]
    if mismatch == "token count":
        tokens, match = [], "sampled token ID count does not match"
    else:
        request.append_sampled_token(19)
        match = "host-committed token frontier .* expected"
    before = request.token_ids.clone()
    calls = _spy_settle(cache, lambda **_: None)

    with pytest.raises(RuntimeError, match=match):
        sched.settle_step(step, sampled(tokens))

    assert torch.equal(request.token_ids, before)
    assert not request.is_finished and calls == []


def test_plan_freezes_prompt_rows() -> None:
    sched = _scheduler(max_batch_tokens=6)
    sched.handle_message(user_msg(7, [10, 11, 12], max_tokens=3))
    sched.handle_message(user_msg(8, [20, 21, 22, 23, 24], max_tokens=3))

    plan = sched.plan_next()

    assert plan is not None
    regular, chunk = (entry.request for entry in plan.rows)
    assert plan.op.rows == (
        Row(regular.slot.index, QueryKind.EXTEND, 0, 3, write_position=3),
        Row(chunk.slot.index, QueryKind.EXTEND, 0, 3, write_position=None),
    )
    assert _ints(plan.op.extend_token_ids) == [10, 11, 12, 20, 21, 22]
    sched.mark_submitted(plan)


def test_intermediate_prompt_chunk_publishes_its_prefix_but_no_token() -> None:
    cache = cache_manager()
    sched = _scheduler(cache, max_batch_tokens=4)
    sched.handle_message(user_msg(7, [10, 11, 12, 13, 14], max_tokens=2))
    step = submit(sched)
    request = step.rows[0].request
    calls = _spy_settle(cache, lambda **_: None)

    assert settle(sched, step, 20) == []

    assert request.token_ids.tolist() == [10, 11, 12, 13, 14]
    [call] = calls
    assert call["use"] is reservations(step)[0]
    assert call["prefix_ids"].tolist() == [10, 11, 12, 13]
    assert call["committed"] == 4


def test_final_chunk_samples_after_its_predecessor_settles() -> None:
    sched = _scheduler(max_running=1, max_batch_tokens=2)
    sched.handle_message(user_msg(7, [10, 11, 12], max_tokens=2))
    first_chunk, final_chunk = submit(sched), submit(sched)

    assert settle(sched, first_chunk, 999) == []
    replies = settle(sched, final_chunk, 20)

    assert [(reply.uid, reply.token_ids) for reply in replies] == [(7, (20,))]
    assert final_chunk.rows[0].request.token_ids.tolist() == [10, 11, 12, 20]


def test_prompts_take_at_most_the_prompt_budget_beside_decodes() -> None:
    sched = _scheduler(max_running=3, max_batch_tokens=8, max_prompt_tokens=3)
    sched.handle_message(user_msg(8, [10, 11]))
    settle(sched, submit(sched), 41)
    sched.handle_message(user_msg(7, [10, 11]))
    sched.handle_message(user_msg(9, [20, 21, 22, 23]))

    step = submit(sched)

    # 8 decodes; 7 takes 2 of the 3 prompt tokens and 9 the last, though 7 of 8 are left.
    assert request_ids(step) == (7, 9, 8)
    entries = step.rows[:2]
    assert [(entry.use.start, entry.use.end) for entry in entries] == [
        (0, 2),
        (0, 1),
    ]


def test_mixed_step_settles_the_decode_row_beside_a_chunk() -> None:
    sched = _scheduler(max_running=2, max_batch_tokens=3)
    sched.handle_message(user_msg(8, [10, 11]))
    settle(sched, submit(sched), 41)
    sched.handle_message(user_msg(7, [10, 11, 12], max_tokens=2))

    step = submit(sched)
    assert request_ids(step) == (7, 8)
    replies = settle(sched, step, [999, 42])

    assert [(reply.uid, reply.token_ids) for reply in replies] == [(8, (42,))]
    chunk, decode = (entry.request for entry in step.rows)
    assert chunk.token_ids.tolist() == [10, 11, 12]
    assert decode.token_ids.tolist() == [10, 11, 41, 42]


@pytest.mark.parametrize("overlap", [False, True])
def test_engine_overlap_launches_the_next_step_before_settling_the_last(overlap: bool) -> None:
    harness = engine(overlap=overlap)
    harness.engine.handle([user_msg(7, _PROMPT, max_tokens=3)], admitting=True)

    harness.engine.step()
    harness.engine.step()

    if overlap:
        assert harness.executor.events == ["launch:0", "launch:1", "sync:0"]
    else:
        assert harness.executor.events == ["launch:0", "sync:0", "launch:1", "sync:1"]


@pytest.mark.parametrize("arrives", [False, True])
def test_with_nothing_to_launch_the_oldest_step_parks_until_the_next_call(arrives: bool) -> None:
    harness = engine()
    harness.engine.handle([user_msg(7, _PROMPT, max_tokens=2)], admitting=True)
    for _ in range(3):
        harness.engine.step()

    # The last step stays in flight: the loop waits for it together with new input.
    assert harness.executor.events == ["launch:0", "launch:1", "sync:0"]
    assert harness.engine.completion is not None
    if arrives:
        harness.engine.handle([user_msg(8, _PROMPT, max_tokens=1)], admitting=True)
    harness.engine.step()

    # A request that arrived meanwhile launches before the parked step settles.
    tail = ["launch:2", "sync:1"] if arrives else ["sync:1"]
    assert harness.executor.events == ["launch:0", "launch:1", "sync:0", *tail]
    assert harness.engine.completion is None


@pytest.mark.parametrize("overlap", [False, True])
def test_length_finish_counts_settled_tokens(overlap: bool) -> None:
    sched = _serving(2)
    first = submit(sched)
    successor = submit(sched) if overlap else None

    first_replies = settle(sched, first, 20)
    second_replies = settle(sched, successor or submit(sched), 21)

    assert first_replies == [DetokenizeMsg(uid=7, token_ids=(20,))]
    assert second_replies == [
        DetokenizeMsg(uid=7, token_ids=(21,), end=RequestEnd(TokenUsage(2, 2), 2))
    ]
    assert first.rows[0].request.token_ids.tolist() == [*_PROMPT, 20, 21]
    _assert_reclaimed(sched)


@pytest.mark.parametrize("stop_token_id", _STOP_TOKEN_IDS)
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_stop_tokens_finish_a_request_unless_it_ignores_eos(
    stop_token_id: int, ignore_eos: bool
) -> None:
    sched = _serving(3, ignore_eos=ignore_eos)

    [reply] = settle(sched, submit(sched), stop_token_id)

    if ignore_eos:
        assert reply == DetokenizeMsg(uid=7, token_ids=(stop_token_id,))
    else:
        usage = TokenUsage(prompt_tokens=2, completion_tokens=1)
        end = RequestEnd(usage, 3, is_stop_token=True)
        assert reply == DetokenizeMsg(uid=7, token_ids=(stop_token_id,), end=end)
        assert reply.end.finish_reason == "stop"
        _assert_reclaimed(sched)


def test_unlisted_token_does_not_stop_the_request() -> None:
    sched = _serving(3, ignore_eos=False)

    assert settle(sched, submit(sched), 3) == [DetokenizeMsg(uid=7, token_ids=(3,))]
    assert sched.has_runnable_work


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_ignore_eos_also_covers_the_tokenizer_eos_of_the_resolved_stop_set(
    tmp_path: Path,
    ignore_eos: bool,
) -> None:
    # The model config names <|endoftext|>; the tokenizer's eos_token is <|im_end|>.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "eos_token_id": 151643}
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {"151645": {"content": "<|im_end|>", "special": True}},
                "eos_token": "<|im_end|>",
            }
        ),
        encoding="utf-8",
    )
    model = ResolvedModel.load(str(tmp_path))
    assert model.facts.stop_token_ids == (151643, 151645)
    sched = scheduler(stop_token_ids=model.facts.stop_token_ids)
    sched.handle_message(user_msg(7, _PROMPT, max_tokens=3, ignore_eos=ignore_eos))

    [reply] = settle(sched, submit(sched), 151645)

    assert reply.finished is not ignore_eos
    assert (reply.end is not None and reply.end.is_stop_token) is not ignore_eos


def test_a_request_reports_the_revision_it_ended_at() -> None:
    revision = [3]
    sched = scheduler(stop_token_ids=_STOP_TOKEN_IDS, model_revision=lambda: revision[0])
    sched.handle_message(user_msg(7, _PROMPT, max_tokens=1, ignore_eos=True))

    [reply] = settle(sched, submit(sched), 20)
    revision[0] = 4  # an optimizer step after the request ended

    assert reply.end is not None and reply.end.model_revision == 3


def test_ignore_eos_emits_terminal_eos_as_a_regular_length_token() -> None:
    sched = _serving(1)

    assert settle(sched, submit(sched), _STOP_TOKEN_IDS[0]) == [
        DetokenizeMsg(
            uid=7,
            token_ids=(_STOP_TOKEN_IDS[0],),
            end=RequestEnd(TokenUsage(prompt_tokens=2, completion_tokens=1), max_new_tokens=1),
        )
    ]
    _assert_reclaimed(sched)


@pytest.mark.parametrize("finish", ["stop", "abort"])
def test_a_finished_request_reports_once_and_drops_its_in_flight_token(finish: str) -> None:
    sched = _serving(3, ignore_eos=finish == "abort")
    first = submit(sched)
    successor = submit(sched)
    request = first.rows[0].request
    if finish == "abort":
        assert sched.handle_message(AbortBackendMsg(uid=7)) == (RequestAbortedMsg(uid=7),)
        assert sched.handle_message(AbortBackendMsg(uid=7)) == ()
        assert settle(sched, first, 20) == []
        tokens, totals = _PROMPT, RequestTotals()  # aborts are not counted
    else:
        [reply] = settle(sched, first, _STOP_TOKEN_IDS[0])
        assert reply.end is not None and reply.end.usage == TokenUsage(2, 1)
        tokens, totals = [*_PROMPT, _STOP_TOKEN_IDS[0]], RequestTotals(1, 0, TokenUsage(2, 1))

    assert settle(sched, successor, 99) == []
    assert request.token_ids.tolist() == tokens
    assert sched.totals == totals
    _assert_reclaimed(sched)


def test_terminal_usage_survives_message_round_trips() -> None:
    message = DetokenizeMsg(
        uid=7,
        token_ids=(2,),
        end=RequestEnd(TokenUsage(prompt_tokens=4, completion_tokens=3), max_new_tokens=3),
    )

    decoded = BaseTokenizerMsg.family().decode(BaseTokenizerMsg.family().encode(message))

    assert decoded == message

    frontend_message = UserReply(
        uid=7,
        incremental_output="done",
        finished=True,
        usage=message.end.usage,
        finish_reason=message.end.finish_reason,
    )
    decoded_frontend = BaseFrontendMsg.family().decode(
        BaseFrontendMsg.family().encode(frontend_message)
    )

    assert decoded_frontend == frontend_message
