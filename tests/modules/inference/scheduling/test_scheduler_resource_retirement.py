"""A finished request keeps its slot and cache until every step it is in has settled."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from cti.inference._control._cache import CacheManager
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.request import Request, RequestBatch
from cti.inference._step import QueryKind
from cti.protocol import AbortBackendMsg, RequestAbortedMsg, SamplingParams
from tests.support.inference.inference_steps import (
    FakeExecutor,
    cache_manager,
    engine,
    scheduler,
    settle,
    submit,
    user_msg,
)
from tests.support.inference.scheduling import ScriptedDecode, request_ids, reservations, sampled

_STOP = 2
_PARAMS = SamplingParams(temperature=0, max_new_tokens=4, ignore_eos=True).snapshot()


def _held(scheduler: InferenceScheduler, cache: CacheManager, step) -> bool:
    """Whether the step's first request still holds its slot and its ``cache`` use."""

    slot = step.rows[0].request.slot
    return scheduler.slots.available_size == 0 and cache.in_use(slot)


def test_scheduler_combines_prompt_extension_and_decode_in_one_plan() -> None:
    sched = scheduler(max_running=2)
    sched.handle_message(user_msg(7, [10, 11, 12]))
    settle(sched, submit(sched))
    sched.handle_message(user_msg(8, [20, 21, 22]))

    plan = sched.plan_next()

    assert plan is not None
    assert request_ids(plan) == (8, 7)
    assert tuple(row.query_kind for row in plan.op.rows) == (QueryKind.EXTEND, QueryKind.DECODE)
    assert list(memoryview(plan.op.extend_token_ids).cast("i")) == [20, 21, 22]


@pytest.mark.parametrize("finish", ["stop", "abort"])
def test_finished_request_keeps_its_slot_until_its_steps_in_flight_settle(finish: str) -> None:
    cache = cache_manager(max_running=1)
    sched = scheduler(max_running=1, cache=cache)
    sched.handle_message(user_msg(7, [10, 11, 12], ignore_eos=False))
    first = submit(sched)
    successor = submit(sched)  # its decode, launched before the prompt step settles
    if finish == "abort":
        assert sched.handle_message(AbortBackendMsg(uid=7)) == (RequestAbortedMsg(uid=7),)
        assert settle(sched, first) == []
    else:
        [reply] = settle(sched, first, _STOP)
        assert reply.finished and reply.end.is_stop_token
    sched.handle_message(user_msg(8, [10, 11, 12]))

    assert _held(sched, cache, successor)
    assert sched.plan_next() is None  # the replacement waits for the slot

    assert settle(sched, successor) == []  # the in-flight token is dropped
    replacement = submit(sched)
    old_slot = first.rows[0].request.slot
    new_slot = replacement.rows[0].request.slot
    assert (new_slot.index, new_slot.generation) == (old_slot.index, old_slot.generation + 1)


@pytest.mark.parametrize("finish", ["length", "abort"])
def test_finished_request_without_steps_in_flight_frees_its_slot(finish: str) -> None:
    sched = scheduler(max_running=1)
    sched.handle_message(user_msg(7, [10, 11, 12], max_tokens=1 if finish == "length" else 4))
    replies = settle(sched, submit(sched))
    if finish == "abort":
        assert sched.handle_message(AbortBackendMsg(uid=7)) == (RequestAbortedMsg(uid=7),)
    else:
        assert replies[0].finished

    assert sched.slots.available_size == 1
    sched.require_quiescent()


def test_multi_request_step_frees_each_slot_after_its_last_step() -> None:
    sched = scheduler(max_running=2)
    sched.handle_message(user_msg(7, [10, 11], max_tokens=1))
    sched.handle_message(user_msg(8, [20, 21], ignore_eos=False))
    first = submit(sched)
    successor = submit(sched)  # 8's decode; 7 is done after its prompt
    assert request_ids(successor) == (8,)

    settle(sched, first, [5, _STOP])  # 7 ends by length, 8 on a stop token

    assert sched.slots.available_size == 1
    settle(sched, successor)
    assert sched.slots.available_size == 2
    sched.require_quiescent()


def test_chunked_prompt_keeps_its_slot_until_every_chunk_settles() -> None:
    cache = cache_manager(max_running=1)
    sched = scheduler(max_running=1, max_batch_tokens=3, cache=cache)
    sched.handle_message(user_msg(7, [10, 11, 12, 13]))
    first = submit(sched)
    final = submit(sched)
    first_reservation, final_reservation = reservations(first)[0], reservations(final)[0]
    assert first_reservation.slot is final_reservation.slot
    assert (first_reservation.start, first_reservation.end) == (0, 3)
    assert (final_reservation.start, final_reservation.end) == (3, 4)

    sched.handle_message(AbortBackendMsg(uid=7))
    settle(sched, first)
    assert _held(sched, cache, final)

    settle(sched, final)
    assert sched.slots.available_size == 1
    sched.require_quiescent()


def test_a_slot_pending_reclamation_cannot_be_planned() -> None:
    decode = ScriptedDecode()
    cache = cache_manager(max_running=1)
    sched = scheduler(max_running=1, decode=decode, cache=cache)
    request = Request(
        uid=7, prompt_token_ids=torch.tensor([10, 11], dtype=torch.int32), policy=_PARAMS
    )
    slot = sched.slots.allocate(request)
    cache.try_open_slot(slot, request.prompt_token_ids, 4, reserved_size=0)
    request.admit(slot, cached_len=0)
    request.plan_prompt_extension(2)
    decode.batch = RequestBatch(extend_requests=(request,))
    step = submit(sched)
    sched.handle_message(AbortBackendMsg(uid=7))  # retired, its step in flight
    decode.batch = RequestBatch(decode_requests=(request,))

    with pytest.raises(RuntimeError, match="pending reclamation"):
        sched.plan_next()

    decode.batch = None
    settle(sched, step)
    sched.require_quiescent()


@pytest.mark.parametrize("failure_at", ["launch", "completion"])
def test_a_failed_step_keeps_its_requests_resources(failure_at: str) -> None:
    failure = RuntimeError(f"{failure_at} failed")
    executor = FakeExecutor(**{f"{failure_at}_failure": failure})
    harness = engine(overlap=False, max_running=1, executor=executor)
    inference = harness.engine
    inference.handle([user_msg(7, [10, 11, 12])], admitting=True)

    with pytest.raises(RuntimeError) as caught:
        inference.step()

    assert caught.value is failure
    with pytest.raises(RuntimeError, match="not (quiescent|fully released)"):
        inference.release()


@pytest.mark.parametrize("in_flight", [False, True])
def test_cleanup_failure_keeps_the_finished_request_pending(in_flight: bool) -> None:
    cache = cache_manager(max_running=1)
    sched = scheduler(max_running=1, cache=cache)
    sched.handle_message(user_msg(7, [10, 11, 12]))
    step = submit(sched)
    if not in_flight:
        settle(sched, step)
    failure = RuntimeError("cache cleanup failed")
    cache.close_slot = Mock(side_effect=failure)

    if in_flight:
        sched.handle_message(AbortBackendMsg(uid=7))
        with pytest.raises(RuntimeError) as caught:
            sched.settle_step(step, sampled([7]))
    else:
        with pytest.raises(RuntimeError) as caught:
            sched.handle_message(AbortBackendMsg(uid=7))
    assert caught.value is failure

    cache.close_slot.assert_called_once()
    assert sched.slots.available_size == 0
    with pytest.raises(RuntimeError, match="retired requests await reclamation"):
        sched.require_quiescent()
