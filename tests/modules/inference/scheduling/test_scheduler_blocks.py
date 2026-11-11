"""DFlash round planning and settlement in the request scheduler, against tests.support.dflash's
``count_oracle``.

Rounds execute on a CPU model of the runner's rows: the real capped commit advances each
row's frontier and done flag in submission order, and settlement checks its record.
"""

from __future__ import annotations

import weakref
from array import array
from dataclasses import dataclass, replace

import pytest
import torch

import ctikernels
from cti.forward import RoundShape
from cti.inference import Round, RoundCompletion
from cti.inference._control._cache import CacheManager
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.request import Request
from cti.inference._control._scheduling.rounds import RoundLedger, RoundTotals
from cti.inference._control._scheduling.rows import RoundSteps
from cti.inference._control._scheduling.strategy import SchedulingLimits
from cti.inference._step import RequestSpec
from cti.protocol import AbortBackendMsg, SamplingParams, UserMsg
from cti.sampling import TokenSampling
from cti.tokenizer.text import TextStream
from tests.support.dflash import count_oracle
from tests.support.dflash_frontier import BlockGeometry, commit_block
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.prefix_cache import compare_keys
from tests.support.inference.scheduling import (
    reservations,
    round_record,
    sampled,
    settle_forward,
)
from tests.support.replies import TokenReply, per_token

_Q = 4
_EOS = 9


class _WordDecoder:
    """Decodes token t to "w<t> "."""

    prefix_stable = True

    def stream(self, *, skip_special_tokens: bool) -> TextStream:
        return TextStream(self, skip_special_tokens=skip_special_tokens)

    def renders(self, token_id: int, *, skip_special_tokens: bool) -> bool:
        return True

    def batch_decode(self, sequences: list[list[int]], *, skip_special_tokens: bool) -> list[str]:
        return ["".join(f"w{t} " for t in ids) for ids in sequences]


def _scheduler_and_cache(
    *,
    max_seq_len: int = 64,
    max_running: int = 3,
    max_batch_tokens: int = 16,
    reuse: bool = False,
) -> tuple[InferenceScheduler, CacheManager]:
    """A round scheduler over a cache of single-token pages, which the test observes."""

    limits = SchedulingLimits(
        max_running_requests=max_running,
        max_batch_tokens=max_batch_tokens,
        max_sequence_length=max_seq_len,
        max_prompt_tokens=max_batch_tokens,
    )
    cache = new_cache_manager(
        256,
        1,
        torch.full((max_running, max_seq_len), -1, dtype=torch.int32),
        reuse=reuse,
        checks=True,
    )
    return _round_scheduler(limits, cache), cache


def _scheduler(**limits: int) -> InferenceScheduler:
    return _scheduler_and_cache(**limits)[0]


def _round_scheduler(limits: SchedulingLimits, cache: CacheManager) -> InferenceScheduler:
    rounds = RoundLedger(
        RoundShape(_Q, 2, min(4, limits.max_batch_tokens // _Q)),
        limits=limits,
        cache_manager=cache,
        stop_token_ids=frozenset({_EOS}),
        checks=True,
    )
    return InferenceScheduler(
        checks=True,
        limits=limits,
        cache_manager=cache,
        stop_token_ids=(_EOS,),
        vocab_size=1 << 20,
        decode=RoundSteps(rounds),
        encoder_cache=EncoderCache(0),
        text_decoder=_WordDecoder(),
    )


def _message(uid: int, *, max_tokens: int = 32, **sampling: object) -> UserMsg:
    # The primary rank's RequestSeeds stamps every DFlash request before scheduling.
    sampling.setdefault("seed", 1000 + uid)
    sampling.setdefault("temperature", 0)
    return UserMsg(
        uid=uid,
        input_ids=torch.arange(10, 15, dtype=torch.int32),
        sampling=SamplingParams(max_new_tokens=max_tokens, **sampling).snapshot(),
    )


@dataclass(frozen=True)
class _Prefilled:
    requests: tuple[Request, ...]
    replies: list[TokenReply]


def _prefill(scheduler: InferenceScheduler, *messages: UserMsg) -> _Prefilled:
    """Admit and prefill ``messages`` in one forward; their first tokens are 100 + uid."""

    for message in messages:
        assert scheduler.handle_message(message) == ()
    plan = scheduler.plan_next()
    assert not plan.round_rows
    scheduler.mark_submitted(plan)
    first_tokens = torch.tensor([100 + m.uid for m in messages], dtype=torch.int32)
    replies = per_token(settle_forward(scheduler, plan, first_tokens).replies)
    return _Prefilled(tuple(entry.request for entry in plan.rows), replies)


class _Device:
    """The runner's request rows: frontier and done flag, advanced by ``commit_block``."""

    def __init__(self, max_seq_len: int) -> None:
        self._geometry = BlockGeometry(_Q, max_seq_len, guard_start=0)
        self._rows: dict[int, tuple[int, bool]] = {}

    def run(self, plan, rows: dict[int, tuple[list[int], int]]) -> RoundCompletion:
        """Run a round, and the prompt extensions beside it, whose first tokens are 100 +
        uid."""

        requests = [entry.request for entry in plan.round_rows]
        forward = plan.forward_rows
        extended = None
        if forward:
            extended = sampled([100 + entry.request.uid for entry in forward])
        tokens = torch.tensor([rows[request.uid][0] for request in requests], dtype=torch.int32)
        accepted = torch.tensor([rows[request.uid][1] for request in requests], dtype=torch.int32)
        # A row's first round starts where its prompt's forward left it.
        state = [self._rows.get(r.uid, (r.num_prompt_tokens, False)) for r in requests]
        starts = torch.tensor([frontier for frontier, _ in state])
        frontiers, done = starts.clone(), torch.tensor([flag for _, flag in state])
        committed, finished = commit_block(
            self._geometry,
            rows=torch.arange(len(requests), dtype=torch.int32),
            frontiers=starts,
            done=done.clone(),
            accepted=accepted,
            tokens=tokens,
            output_ends=torch.tensor([r.max_device_len for r in requests]),
            ignore_eos=torch.tensor([r.sampling.ignore_eos for r in requests]),
            stop_token_ids=torch.tensor([_EOS], dtype=torch.int32),
            token_table=torch.zeros(
                (len(requests), self._geometry.max_seq_len + _Q), dtype=torch.int32
            ),
            frontier_table=frontiers,
            done_table=done,
        )
        for index, request in enumerate(requests):
            self._rows[request.uid] = (int(frontiers[index]), bool(done[index]))
        # A row done before the round verified nothing: its record says a = -1.
        inactive = torch.tensor([flag for _, flag in state])
        accepted = accepted.masked_fill(inactive, -1)
        # Each target token's log-probability is the token over -8, exact in FP32.
        return round_record(
            tokens.tolist(),
            accepted.tolist(),
            committed.tolist(),
            finished.tolist(),
            starts.tolist(),
            extended,
            logprobs=(tokens.float() / -8).tolist(),
        )


_DEVICES: weakref.WeakKeyDictionary[InferenceScheduler, _Device] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class _Submitted:
    plan: object
    completion: RoundCompletion


def _submit(
    scheduler: InferenceScheduler, rows: dict[int, tuple[list[int], int]] | None = None
) -> _Submitted | None:
    """Plan and submit a round, with any prompt extensions beside it, which the device runs at
    once; ``rows`` maps uid -> (target tokens [Q], accepted), by default the drafts 1..Q all
    rejected."""

    plan = scheduler.plan_next()
    if plan is None:
        return None
    scheduler.mark_submitted(plan)
    if rows is None:
        entries = plan.round_rows
        rows = {entry.request.uid: (list(range(1, _Q + 1)), 0) for entry in entries}
    device = _DEVICES.setdefault(scheduler, _Device(scheduler.max_seq_len))
    return _Submitted(plan, device.run(plan, rows))


def _ints(data: bytes, code: str = "i") -> list[int]:
    return list(memoryview(data).cast(code))


def _committed(submitted: _Submitted) -> list[int]:
    return _ints(submitted.completion.committed)


def _settle(scheduler: InferenceScheduler, submitted: _Submitted) -> list[TokenReply]:
    """The step's replies, one token at a time."""

    return per_token(scheduler.settle_step(submitted.plan, submitted.completion).replies)


def _round(
    scheduler: InferenceScheduler, rows: dict[int, tuple[list[int], int]]
) -> tuple[Round, list[TokenReply]]:
    """Plan, run and settle one round; ``rows`` maps uid -> (target tokens [Q], accepted)."""

    submitted = _submit(scheduler, rows)
    uids = [entry.request.uid for entry in submitted.plan.rows]
    assert sorted(uids) == sorted(rows)
    return submitted.plan.op, _settle(scheduler, submitted)


def test_a_round_replies_once_per_request_with_its_visible_tokens() -> None:
    scheduler = _scheduler()
    _prefill(scheduler, _message(0), _message(1))
    submitted = _submit(scheduler, {0: ([1, 2, 3, 4], 3), 1: ([5, 6, 7, 8], 2)})

    replies = scheduler.settle_step(submitted.plan, submitted.completion).replies

    assert [(reply.uid, reply.token_ids) for reply in replies] == [
        (0, (1, 2, 3, 4)),
        (1, (5, 6, 7)),
    ]


def test_a_round_returns_its_visible_tokens_logprobs() -> None:
    scheduler = _scheduler()
    message = _message(0, temperature=1.0, logprobs=True, logprobs_processed=True)
    assert scheduler.handle_message(message) == ()
    plan = scheduler.plan_next()
    scheduler.mark_submitted(plan)
    first = torch.tensor([100], dtype=torch.int32)
    settle_forward(scheduler, plan, first, torch.tensor([-0.5]))

    _, replies = _round(scheduler, {0: ([20, 21, 22, 23], 1)})

    assert [(r.next_token, r.logprobs.logprob) for r in replies] == [(20, -2.5), (21, -2.625)]


def test_a_slot_admits_its_request_policy_and_seed_once() -> None:
    scheduler = _scheduler()
    messages = (_message(0, temperature=0.6, top_k=20, top_p=0.95, seed=7), _message(1))
    for message in messages:
        assert scheduler.handle_message(message) == ()
    plan = scheduler.plan_next()
    scheduler.mark_submitted(plan)
    settle_forward(scheduler, plan, torch.tensor([100, 101], dtype=torch.int32))

    first, second = plan.admit
    assert first == RequestSpec(first.slot, 5, 37, TokenSampling(0.6, 20, 0.95), seed=7)
    assert second.sampling.is_greedy and second.seed == 1001
    # The prefill's sampled token is response position 0 of the ordinary namespace.
    assert [row.sampling_position for row in plan.op.rows] == [0, 0]
    # Rounds over the same requests admit nothing again.
    assert _submit(scheduler).plan.admit == ()


def test_greedy_requests_carry_no_filters() -> None:
    scheduler = _scheduler()
    (request,) = _prefill(scheduler, _message(0)).requests

    params = request.sampling
    assert (params.temperature, params.top_k, params.top_p) == (0.0, None, 1.0)


def test_an_unseeded_dflash_request_never_reaches_scheduling() -> None:
    message = replace(
        _message(0), sampling=SamplingParams(temperature=0, max_new_tokens=4).snapshot()
    )

    with pytest.raises(RuntimeError, match="unseeded"):
        _scheduler().handle_message(message)


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("eos_index", [None, 0, 2])
@pytest.mark.parametrize("accepted", range(_Q))
def test_a_round_publishes_the_oracle_prefix_and_commits_its_kv(
    accepted: int, eos_index: int | None, budget: int
) -> None:
    scheduler, cache = _scheduler_and_cache()
    # The prompt's settled first token is the round's anchor; `budget` tokens remain.
    (request,) = _prefill(scheduler, _message(0, max_tokens=budget + 1)).requests
    anchor, drafts = 100, [20, 21, 22]
    targets = ([*drafts[:accepted], 30 + accepted] + [40, 50, 60])[:_Q]
    if eos_index is not None:
        targets[eos_index] = _EOS
        # Drafts are never EOS: an EOS target rejects its draft.
        accepted = min(accepted, eos_index)
    expected = count_oracle(
        anchor, tuple(drafts), tuple(targets), output_budget=budget, eos_token_ids=(_EOS,)
    )
    assert expected.accepted_drafts == accepted

    batch, replies = _round(scheduler, {0: (targets, accepted)})

    assert len(batch.slots) == 1
    assert tuple(reply.next_token for reply in replies) == expected.visible_tokens
    finished = expected.visible_tokens[-1] == _EOS or len(expected.visible_tokens) == budget
    assert [reply.finished for reply in replies] == [False] * (len(replies) - 1) + [finished]
    if finished:
        scheduler.require_quiescent()
        return
    # The anchor and accepted drafts are cached; the correction or bonus anchors next.
    assert scheduler.has_runnable_work and not request.is_finished
    assert request.cached_len == 5 + len(expected.committed_inputs)
    assert request.token_ids.tolist()[5:] == [anchor, *expected.output_tokens]
    # Nothing in flight, one-token pages: the slot holds exactly its committed KV.
    assert cache.occupancy.num_used_pages == request.cached_len


def test_rounds_rotate_past_the_batch_limit_beside_a_prompt_in_chunks() -> None:
    scheduler = _scheduler(max_batch_tokens=10)
    _prefill(scheduler, _message(0), _message(1))
    assert scheduler.handle_message(_message(2)) == ()

    # Two rows of Q=4 leave 2 of 10 tokens: request 2's 5-token prompt runs beside the
    # rounds in chunks of 2, 2 and 1, then two of the three requests fit a round.
    rounds, extended = [], []
    for _ in range(5):
        submitted = _submit(scheduler)
        forward = submitted.plan.forward_rows
        if forward:
            extended += [(e.use.start, e.use.end) for e in forward]
        rounds.append(set(submitted.plan.op.slots))
        _settle(scheduler, submitted)
    assert extended == [(0, 2), (2, 4), (4, 5)]
    assert [len(rows) for rows in rounds] == [2, 2, 2, 2, 2]
    assert rounds[3] | rounds[4] == {0, 1, 2}


def test_mixed_budgets_past_the_batch_limit_finish_with_two_submissions_in_flight() -> None:
    # The engine's loop: submit the next plan, then settle the oldest of two in flight.
    scheduler = _scheduler(max_seq_len=128, max_running=8)
    budgets = [3, 40, 5, 17, 2, 33, 9, 60, 18, 4]
    for uid, budget in enumerate(budgets):
        assert scheduler.handle_message(_message(uid, max_tokens=budget)) == ()
    device = _DEVICES.setdefault(scheduler, _Device(scheduler.max_seq_len))
    in_flight, replies = [], []
    for step in range(400):
        plan = scheduler.plan_next()
        if plan is not None:
            scheduler.mark_submitted(plan)
            uids = [entry.request.uid for entry in plan.rows]
            if bool(plan.round_rows):
                # Every acceptance length occurs; tokens 1..Q never stop a request.
                rows = {uid: (list(range(1, _Q + 1)), (uid + step) % _Q) for uid in uids}
                in_flight.append(_Submitted(plan, device.run(plan, rows)))
            else:
                in_flight.append(plan)
        if in_flight and (plan is None or len(in_flight) == 2):
            oldest = in_flight.pop(0)
            if isinstance(oldest, _Submitted):
                replies += _settle(scheduler, oldest)
            else:
                first = torch.tensor([100 + e.request.uid for e in oldest.rows])
                replies += per_token(settle_forward(scheduler, oldest, first.int()).replies)
        if not in_flight and not scheduler.has_runnable_work:
            break
    else:
        raise AssertionError("requests did not finish")

    counts = {uid: sum(reply.uid == uid for reply in replies) for uid in range(len(budgets))}
    assert counts == dict(enumerate(budgets))
    assert sorted(reply.uid for reply in replies if reply.finished) == list(range(len(budgets)))
    scheduler.require_quiescent()


def test_a_request_filling_the_sequence_finishes_in_rounds() -> None:
    # Without room past its budget the lookahead is zero, not a stall: rounds run to the
    # capacity, where the runner's guard rows take the block's positions past it.
    scheduler = _round_scheduler(
        SchedulingLimits(
            max_running_requests=1,
            max_batch_tokens=16,
            max_sequence_length=16,
            max_prompt_tokens=16,
        ),
        new_cache_manager(
            16, 1, torch.full((1, 16), -1, dtype=torch.int32), reuse=False, checks=True
        ),
    )
    replies = _prefill(scheduler, _message(0, max_tokens=11)).replies
    rounds = 0
    while scheduler.has_runnable_work:
        submitted = _submit(scheduler, {0: ([1, 2, 3, 4], 3)})
        rounds += 1
        replies.extend(_settle(scheduler, submitted))

    # 10 tokens after the first: 4 + 4, then the last 2 at positions 14 and 15.
    assert rounds == 3
    assert len(replies) == 11 and replies[-1].finished
    assert scheduler.totals.num_requests == 1
    assert scheduler.totals.usage.completion_tokens == 11
    # Acceptance counts raw drafts, before the last round's budget clip.
    assert scheduler.totals.rounds == RoundTotals(3, 9)
    assert "3 request rounds, 9 drafts accepted, mean acceptance length 4.00" in str(
        scheduler.totals
    )
    scheduler.require_quiescent()


def test_an_aborted_request_releases_its_round_without_output() -> None:
    scheduler = _scheduler()
    _prefill(scheduler, _message(0))
    submitted = _submit(scheduler, {0: ([1, 2, 3, 4], 3)})

    (aborted,) = scheduler.handle_message(AbortBackendMsg(uid=0))

    assert aborted.uid == 0
    assert _settle(scheduler, submitted) == []
    scheduler.require_quiescent()


@pytest.mark.parametrize(
    "own_stop", [{"stop_token_ids": (2,)}, {"stop": ("w2 ",)}], ids=["token", "string"]
)
def test_a_requests_own_stop_ends_its_round_where_the_device_runs_on(
    own_stop: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A prefix cache shows which KV the slot commits: closing it publishes that prefix.
    monkeypatch.setattr(ctikernels, "fast_compare_key", compare_keys)
    scheduler, cache = _scheduler_and_cache(max_running=1, reuse=True)
    (request,) = _prefill(scheduler, _message(0, **own_stop)).requests
    slot = request.slot
    first = _submit(scheduler, {0: ([1, 2, 3, 4], 3)})
    successor = _submit(scheduler, {0: ([5, 6, 7, 8], 3)})
    # The device checks only the model's stop tokens: both rows commit their rounds.
    assert _committed(first) == _committed(successor) == [4]

    replies = _settle(scheduler, first)

    assert [reply.next_token for reply in replies] == [1, 2]
    stop_str = own_stop.get("stop", (None,))[0]
    assert (replies[-1].is_stop_token, replies[-1].stop_str) == (stop_str is None, stop_str)
    assert replies[-1].finish_reason == "stop" and replies[-1].usage.completion_tokens == 3
    # Its row's round is discarded like a finished request's successor: the successor's
    # claim holds the slot until it settles, and none of the round's KV is committed, so
    # closing the slot publishes the prompt alone.
    assert scheduler.slots.available_size == 0
    assert _settle(scheduler, successor) == []
    assert scheduler.totals.rounds == RoundTotals(1, 3)
    scheduler.require_quiescent()
    assert cache.cached_prefix_len(request.token_ids) == 5
    (reused,) = _prefill(scheduler, _message(1)).requests
    assert reused.slot.index == slot.index


def test_a_requests_own_stop_before_eos_in_one_round_ends_it_first() -> None:
    scheduler = _scheduler()
    _prefill(scheduler, _message(0, stop_token_ids=(2,)))

    # Three drafts accepted, then EOS as the bonus: the own stop at index 1 comes first.
    _, replies = _round(scheduler, {0: ([1, 2, 3, _EOS], 3)})

    assert [reply.next_token for reply in replies] == [1, 2]
    assert replies[-1].is_stop_token and replies[-1].finish_reason == "stop"
    # The totals count the drafts the round accepted, before any stop cut, as for EOS.
    assert scheduler.totals.rounds == RoundTotals(1, 3)
    scheduler.require_quiescent()


def test_a_requests_own_stop_leaves_the_other_rows_of_its_round_cached() -> None:
    scheduler, cache = _scheduler_and_cache()
    _, request = _prefill(
        scheduler, _message(0, stop_token_ids=(2,)), _message(1, stop_token_ids=(2,))
    ).requests

    _, replies = _round(scheduler, {0: ([1, 2, 3, 4], 3), 1: ([5, 6, 7, 8], 2)})

    assert [(reply.uid, reply.next_token) for reply in replies] == [
        (0, 1),
        (0, 2),
        (1, 5),
        (1, 6),
        (1, 7),
    ]
    assert request.uid == 1 and not request.is_finished
    assert request.token_ids.tolist()[5:] == [101, 5, 6, 7]
    # Request 0's slot closed; request 1's holds exactly its committed KV.
    assert cache.occupancy.num_used_pages == request.cached_len == 8


def test_admission_reserves_a_block_past_the_output_budget() -> None:
    # 18 tokens: prompt 5 + output 8 + lookahead Q - 1 fit; a second request does not.
    scheduler = _round_scheduler(
        SchedulingLimits(
            max_running_requests=2,
            max_batch_tokens=16,
            max_sequence_length=32,
            max_prompt_tokens=16,
        ),
        new_cache_manager(
            18, 1, torch.full((2, 32), -1, dtype=torch.int32), reuse=False, checks=True
        ),
    )
    scheduler.handle_message(_message(0, max_tokens=8))
    scheduler.handle_message(_message(1, max_tokens=8))

    plan = scheduler.plan_next()

    assert [entry.request.uid for entry in plan.rows] == [0]


# Two unsettled rounds per request.


def _claim_end(submitted: _Submitted) -> int:
    (use,) = reservations(submitted.plan)
    assert use.start is None  # a round's claim
    return use.end


def test_a_third_round_waits_for_the_oldest_and_keeps_its_successors_capacity() -> None:
    scheduler, cache = _scheduler_and_cache()
    (request,) = _prefill(scheduler, _message(0)).requests

    first = _submit(scheduler, {0: ([1, 2, 3, 4], 0)})
    second = _submit(scheduler, {0: ([5, 6, 7, 8], 3)})
    assert _submit(scheduler) is None
    # Each claim reaches Q past the furthest anchor its predecessors may leave.
    assert [_claim_end(first), _claim_end(second)] == [5 + _Q, 5 + 2 * _Q]
    assert cache.occupancy.num_used_pages == 5 + 2 * _Q

    _settle(scheduler, first)
    # Settling the first keeps the second's capacity.
    assert cache.occupancy.num_used_pages == 5 + 2 * _Q
    third = _submit(scheduler, {0: ([1, 2, 3, 4], 0)})
    # The third extends from H = 6.
    assert _claim_end(third) == 6 + 2 * _Q
    _settle(scheduler, second)
    # The third's claim still holds its pages past the second's committed frontier.
    assert request.cached_len == 10
    assert cache.occupancy.num_used_pages == 6 + 2 * _Q
    _settle(scheduler, third)
    assert request.cached_len == 11 and request.token_ids.tolist()[5:] == [100, 1, 5, 6, 7, 8, 1]
    # Nothing in flight: the slot holds exactly its committed KV.
    assert cache.occupancy.num_used_pages == 11


def test_a_round_that_may_finish_its_request_has_no_lone_successor() -> None:
    scheduler = _scheduler()
    # The anchor (first output) leaves 3 tokens, which one round of Q=4 may cover.
    _prefill(scheduler, _message(0, max_tokens=4))

    first = _submit(scheduler, {0: ([1, 2, 3, 4], 3)})
    assert _submit(scheduler) is None
    replies = _settle(scheduler, first)

    assert [reply.next_token for reply in replies] == [1, 2, 3] and replies[-1].finished
    scheduler.require_quiescent()


@pytest.mark.parametrize("finishes", [False, True])
def test_a_successor_its_predecessor_may_finish_rides_along_with_others(finishes: bool) -> None:
    scheduler = _scheduler()
    _prefill(scheduler, _message(0, max_tokens=4), _message(1))
    first = _submit(scheduler, {0: ([1, 2, 3, 4], 3 if finishes else 0), 1: ([1, 2, 3, 4], 0)})
    ride = _submit(scheduler, {0: ([5, 6, 7, 8], 3), 1: ([5, 6, 7, 8], 0)})
    assert [entry.request.uid for entry in ride.plan.rows] == [0, 1]

    replies = _settle(scheduler, first) + _settle(scheduler, ride)

    # A finished request's successor row is done on the device and publishes nothing.
    tokens = [1, 2, 3] if finishes else [1, 5, 6]
    assert [reply.next_token for reply in replies if reply.uid == 0] == tokens
    assert [reply.finished for reply in replies if reply.uid == 0][-1]
    assert [reply.next_token for reply in replies if reply.uid == 1] == [1, 5]


def test_a_round_that_certainly_finishes_its_request_has_no_successor() -> None:
    scheduler = _scheduler()
    # The anchor leaves 1 token: the first round always finishes the request.
    _prefill(scheduler, _message(0, max_tokens=2), _message(1))
    _submit(scheduler)

    second = _submit(scheduler)
    assert [entry.request.uid for entry in second.plan.rows] == [1]


@pytest.mark.parametrize("ending", ["eos", "abort"])
def test_a_successor_of_a_finished_request_publishes_nothing_and_frees_the_slot(
    ending: str,
) -> None:
    scheduler = _scheduler(max_running=1)
    (request,) = _prefill(scheduler, _message(0)).requests
    slot = request.slot
    first = _submit(scheduler, {0: ([1, _EOS, 3, 4], 3)})
    successor = _submit(scheduler, {0: ([5, 6, 7, 8], 3)})
    # The device already knows the first round finished the request: the successor's row
    # is done and commits nothing.
    assert _committed(successor) == [0]

    if ending == "abort":
        (aborted,) = scheduler.handle_message(AbortBackendMsg(uid=0))
        assert aborted.uid == 0
    replies = _settle(scheduler, first)
    assert [reply.finished for reply in replies] == ([] if ending == "abort" else [False, True])
    # The successor's claim still holds the slot: no new request can take it.
    assert scheduler.slots.available_size == 0
    assert _settle(scheduler, successor) == []
    # Only a live request's round counts: never a discarded successor or an aborted round.
    live = ending == "eos"
    assert scheduler.totals.rounds == (RoundTotals(1, 3) if live else RoundTotals())

    scheduler.require_quiescent()
    (reused,) = _prefill(scheduler, _message(1)).requests
    assert reused.slot.index == slot.index and reused.slot.generation == slot.generation + 1


@pytest.mark.parametrize("ending", ["eos", "abort"])
def test_a_request_ending_before_its_joined_round_settles_frees_the_slot(ending: str) -> None:
    scheduler = _scheduler(max_running=1)
    assert scheduler.handle_message(_message(0)) == ()
    prompt = scheduler.plan_next()
    scheduler.mark_submitted(prompt)
    # The device never checks the first token for a stop: the joined row runs on.
    joined = _submit(scheduler, {0: ([1, 2, 3, 4], 3)})
    assert _committed(joined) == [4]

    if ending == "abort":
        (aborted,) = scheduler.handle_message(AbortBackendMsg(uid=0))
        assert aborted.uid == 0
    outcome = settle_forward(scheduler, prompt, torch.tensor([_EOS], dtype=torch.int32))
    assert [reply.finished for reply in outcome.replies] == ([] if ending == "abort" else [True])
    # The joined round's claim still holds the slot; its settlement publishes nothing.
    assert scheduler.slots.available_size == 0
    assert _settle(scheduler, joined) == []
    assert scheduler.totals.rounds == RoundTotals()
    scheduler.require_quiescent()


def test_a_successor_may_emit_what_a_short_predecessor_leaves() -> None:
    scheduler = _scheduler()
    # 6 tokens remain after the anchor: the first round may take 4, the second then up to 5.
    _prefill(scheduler, _message(0, max_tokens=7))

    first = _submit(scheduler, {0: ([1, 2, 3, 4], 0)})
    second = _submit(scheduler, {0: ([5, 6, 7, 8], 3)})
    replies = _settle(scheduler, first) + _settle(scheduler, second)

    assert [reply.next_token for reply in replies] == [1, 5, 6, 7, 8]


def test_a_failed_claim_settlement_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, cache = _scheduler_and_cache()
    _prefill(scheduler, _message(0))
    submitted = _submit(scheduler, {0: ([1, 2, 3, 4], 3)})

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("page release failed")

    monkeypatch.setattr(cache, "settle", fail)
    with pytest.raises(RuntimeError, match="page release failed"):
        scheduler.settle_step(submitted.plan, submitted.completion)


@pytest.mark.parametrize(
    ("ending", "field"),
    [("eos", "frontier"), ("eos", "live"), ("abort", "frontier"), ("abort", "committed")],
)
def test_a_corrupt_discarded_round_fails_settlement(ending: str, field: str) -> None:
    scheduler = _scheduler()
    _prefill(scheduler, _message(0))
    first = _submit(scheduler, {0: ([1, _EOS if ending == "eos" else 2, 3, 4], 3)})
    successor = _submit(scheduler, {0: ([5, 6, 7, 8], 0)})
    if ending == "abort":
        scheduler.handle_message(AbortBackendMsg(uid=0))
    _settle(scheduler, first)
    record = successor.completion
    corrupt = {
        "frontier": {
            "frontiers": array("q", [f + 1 for f in _ints(record.frontiers, "q")]).tobytes()
        },
        "live": {"accepted": array("i", [2]).tobytes(), "done": bytes([False])},
        "committed": {"committed": array("i", [4]).tobytes()},
    }[field]

    with pytest.raises(RuntimeError, match="diverged from settlement"):
        scheduler.settle_step(successor.plan, replace(record, **corrupt))
