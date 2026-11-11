"""Steps of requests that decode in rounds: the round of every row that can run, beside the
waiting prompts the rest of the token budget admits."""

from __future__ import annotations

import pytest
import torch

from cti.forward import RoundShape
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.rounds import RoundLedger
from cti.inference._control._scheduling.rows import RoundSteps
from cti.inference._control._scheduling.strategy import SchedulingLimits
from cti.inference._step import QueryKind
from cti.protocol import SamplingParams, UserMsg
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.scheduling import round_record, sampled
from tests.support.replies import per_token

_Q = 4
_EOS = 9


def _scheduler(
    *,
    max_running: int = 4,
    max_batch_tokens: int = 64,
    max_prompt_tokens: int | None = None,
    max_seq_len: int = 128,
    pages: int = 1024,
    reuse: bool = False,
    snapshot_rows: range = range(0),
) -> InferenceScheduler:
    limits = SchedulingLimits(
        max_running, max_batch_tokens, max_seq_len, max_prompt_tokens or max_batch_tokens
    )
    cache = new_cache_manager(
        pages,
        1,
        torch.full((max_running, max_seq_len), -1, dtype=torch.int32),
        reuse=reuse,
        checks=True,
        snapshot_rows=snapshot_rows,
    )
    ledger = RoundLedger(
        RoundShape(_Q, 2, min(4, limits.max_batch_tokens // _Q)),
        limits=limits,
        cache_manager=cache,
        stop_token_ids=frozenset({_EOS}),
    )
    return InferenceScheduler(
        checks=True,
        limits=limits,
        cache_manager=cache,
        stop_token_ids=(_EOS,),
        vocab_size=1 << 20,
        decode=RoundSteps(ledger),
        encoder_cache=EncoderCache(limits.max_batch_tokens, checks=True),
    )


def _message(
    uid: int, *, length: int = 5, max_tokens: int = 32, prompt: torch.Tensor | None = None
) -> UserMsg:
    start = 10 + 20 * uid
    return UserMsg(
        uid=uid,
        input_ids=torch.arange(start, start + length, dtype=torch.int32)
        if prompt is None
        else prompt,
        sampling=SamplingParams(
            temperature=0, max_new_tokens=max_tokens, seed=1000 + uid
        ).snapshot(),
    )


def _submit(scheduler: InferenceScheduler):
    plan = scheduler.plan_next()
    assert plan is not None
    scheduler.mark_submitted(plan)
    return plan


def _settle(scheduler: InferenceScheduler, plan) -> list:
    """Settle a submitted step: first tokens 100 + uid, rounds rejecting every draft."""

    output = None
    if claim := plan.forward_rows:
        output = sampled([100 + entry.request.uid for entry in claim])
    if claim := plan.round_rows:
        size = len(claim)
        output = round_record(
            [[1, 2, 3, 4]] * size,
            [0] * size,
            [1] * size,
            [_ends(entry.request) for entry in claim],
            [entry.request.cached_len for entry in claim],
            output,
        )
    return per_token(scheduler.settle_step(plan, output).replies)


def _ends(request) -> bool:
    """Whether a round committing one token at the request's frontier ends its output."""

    return request.cached_len + 1 >= request.max_device_len - 1


def _kinds(plan) -> tuple[bool, bool]:
    return bool(plan.forward_rows), bool(plan.round_rows)


def _uids(claim) -> list[int]:
    return [entry.request.uid for entry in claim]


def _after_a_prompt_step(scheduler: InferenceScheduler, *uids: int, **message: int) -> None:
    """Run and settle a prompt step for ``uids``, so the next step runs rows first."""

    for uid in uids:
        assert scheduler.handle_message(_message(uid, **message)) == ()
    plan = _submit(scheduler)
    assert _kinds(plan) == (True, False)
    _settle(scheduler, plan)


def test_a_prompt_waiting_beside_running_rows_runs_in_their_round() -> None:
    scheduler = _scheduler()
    _after_a_prompt_step(scheduler, 0)
    assert scheduler.handle_message(_message(1)) == ()

    plan = _submit(scheduler)

    assert _kinds(plan) == (True, True)
    assert all(row.query_kind is QueryKind.EXTEND for row in plan.op.extend.rows)
    assert _uids(plan.forward_rows) == [1] and _uids(plan.round_rows) == [0]
    # The mixed-in request joins the next round before the mixed step settles.
    successor = _submit(scheduler)
    assert _kinds(successor) == (False, True)
    assert sorted(_uids(successor.round_rows)) == [0, 1]
    replies = _settle(scheduler, plan)
    assert [(reply.uid, reply.next_token) for reply in replies] == [(1, 101), (0, 1)]
    _settle(scheduler, successor)


def test_every_step_runs_its_rows_while_prompts_keep_arriving() -> None:
    scheduler = _scheduler()
    _after_a_prompt_step(scheduler, 0)
    for uid in (1, 2, 3):
        assert scheduler.handle_message(_message(uid)) == ()
        plan = _submit(scheduler)
        assert _kinds(plan) == (True, True)
        assert _uids(plan.forward_rows) == [uid]
        _settle(scheduler, plan)

    # Nothing waits: the step is a round alone.
    assert _kinds(_submit(scheduler)) == (False, True)


def test_rows_take_the_token_budget_first_and_prompts_the_rest_in_chunks() -> None:
    # Two rows of Q=4 leave 8 of 16 tokens: a 20-token prompt runs in chunks of 8, 8, 4.
    scheduler = _scheduler(max_batch_tokens=16)
    _after_a_prompt_step(scheduler, 0, 1)
    assert scheduler.handle_message(_message(2, length=20)) == ()

    extents = []
    while len(extents) < 3:
        plan = _submit(scheduler)
        assert _kinds(plan) == (True, True) and sorted(_uids(plan.round_rows)) == [0, 1]
        (entry,) = plan.forward_rows
        extents.append((entry.use.start, entry.use.end))
        _settle(scheduler, plan)

    assert extents == [(0, 8), (8, 16), (16, 20)]


@pytest.mark.parametrize(("max_prompt_tokens", "chunk"), [(4, 4), (12, 8)])
def test_prompts_take_at_most_the_prompt_budget_of_what_rows_leave(
    max_prompt_tokens: int, chunk: int
) -> None:
    # Two rows of Q=4 leave 8 of 16 tokens; the prompt budget caps the chunk below that.
    scheduler = _scheduler(max_batch_tokens=16, max_prompt_tokens=max_prompt_tokens)
    _after_a_prompt_step(scheduler, 0, 1, length=2)
    assert scheduler.handle_message(_message(2, length=20)) == ()

    plan = _submit(scheduler)

    assert _kinds(plan) == (True, True)
    (entry,) = plan.forward_rows
    assert (entry.use.start, entry.use.end) == (0, chunk)
    _settle(scheduler, plan)


def test_a_mixed_step_settles_its_prompt_then_its_round() -> None:
    scheduler = _scheduler()
    _after_a_prompt_step(scheduler, 0)
    assert scheduler.handle_message(_message(1, max_tokens=1)) == ()
    plan = _submit(scheduler)
    assert _kinds(plan) == (True, True)

    replies = _settle(scheduler, plan)

    # Request 1 finishes with its first token; request 0 publishes its round.
    assert [(reply.uid, reply.finished) for reply in replies] == [(1, True), (0, False)]
    requests = {entry.request.uid: entry.request for entry in plan.rows}
    assert requests[1].is_finished and not requests[0].is_finished
    assert requests[0].num_output_tokens == 2 and scheduler.has_runnable_work


def test_with_no_row_to_run_a_prompt_runs_alone() -> None:
    scheduler = _scheduler()
    assert scheduler.handle_message(_message(0)) == ()

    plan = _submit(scheduler)

    assert _kinds(plan) == (True, False)
