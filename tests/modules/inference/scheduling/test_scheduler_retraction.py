"""Admission counts a bounded output estimate; running requests that outgrow the cache are
retracted, resume from what the prefix cache published, and finish with their whole
output."""

from __future__ import annotations

import pytest

import ctikernels.radix
from cti.inference._control._scheduling import prefill
from cti.inference._step import QueryKind
from cti.protocol import AbortBackendMsg, DetokenizeMsg, RequestAbortedMsg
from tests.support.inference.inference_steps import engine, scheduler, settle, user_msg
from tests.support.inference.prefix_cache import reference_radix_comparison  # noqa: F401

_PROMPT = [10, 11, 12, 13]


@pytest.fixture(autouse=True)
def _small_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prefill, "ESTIMATED_OUTPUT_TOKENS", 4)
    monkeypatch.setattr(ctikernels.radix, "warm_up", lambda: None)


@pytest.mark.parametrize("prefix_cache", ["naive", "radix"])
def test_requests_beyond_the_estimate_run_together_and_resume_after_retraction(
    prefix_cache: str,
) -> None:
    # 24 pages hold one request's 5 + 15 tokens, not two: reserving whole budgets would
    # serialize them.
    harness = engine(num_pages=24, max_seq_len=24, prefix_cache=prefix_cache)
    inference = harness.engine
    inference.handle(
        [user_msg(uid, [*_PROMPT, uid], max_tokens=15) for uid in (1, 2)], admitting=True
    )
    while inference.busy:
        inference.step()

    tokens = [reply for reply in harness.replies if isinstance(reply, DetokenizeMsg)]
    finished = [reply.uid for reply in tokens if reply.finished]
    assert sorted(finished) == [1, 2]
    for uid in (1, 2):
        assert sum(reply.uid == uid for reply in tokens) == 15
    # Both decoded before either finished.
    first_finish = next(index for index, reply in enumerate(tokens) if reply.finished)
    assert {reply.uid for reply in tokens[:first_finish]} == {1, 2}
    # A retracted request resumed: an extension past its prompt.
    resumed = [
        row
        for forward in harness.executor.forwards
        for row in forward.rows
        if row.query_kind is QueryKind.EXTEND and row.sequence_len > len(_PROMPT) + 1
    ]
    assert resumed
    # Its last row samples the output position it would have decoded at.
    for row in resumed:
        if row.write_position is not None:
            assert row.sampling_position == row.sequence_len - len(_PROMPT) - 1
    occupancy = inference._cache.occupancy
    assert occupancy.num_used_pages == 0, occupancy


def test_requests_within_the_estimate_reserve_their_whole_budget() -> None:
    sched = scheduler(num_pages=12, max_seq_len=12)
    for uid in (1, 2):
        sched.handle_message(user_msg(uid, _PROMPT, max_tokens=4))

    step = sched.plan_next()

    # 4 + 4 tokens each: the second does not fit beside the first.
    assert step is not None and len(step.forward_rows) == 1


def test_an_aborted_request_retracting_behind_its_step_in_flight_is_not_requeued() -> None:
    sched = scheduler(num_pages=16, max_seq_len=12, max_batch_tokens=16)
    for uid in (1, 2):
        sched.handle_message(user_msg(uid, _PROMPT, max_tokens=7))
    prefilled = sched.plan_next()
    sched.mark_submitted(prefilled)
    settle(sched, prefilled)

    # Decode until the next step no longer fits: the plan retracts request 2 behind its
    # step in flight and waits for it.
    in_flight = sched.plan_next()
    while in_flight is not None:
        sched.mark_submitted(in_flight)
        nxt = sched.plan_next()
        if nxt is None:
            break
        settle(sched, in_flight)
        in_flight = nxt
    assert in_flight is not None and sched.slots.has_retired

    (aborted,) = sched.handle_message(AbortBackendMsg(uid=2))
    assert aborted == RequestAbortedMsg(uid=2)
    settle(sched, in_flight)

    assert 2 not in sched._prefill_queue.pending_ids and not sched.slots.has_retired
