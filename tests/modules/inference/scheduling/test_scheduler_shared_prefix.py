"""Batch selection holds back a prompt whose shared prefix a planned forward is exporting."""

import gc
from array import array
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cti.inference import Sampled
from cti.inference._control._cache import CacheUse
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.prefill import PrefillQueue, SharedPrefixDeferral
from cti.inference._control._scheduling.request import Request
from cti.inference._control._scheduling.strategy import SchedulingLimits, TokenSteps
from cti.inference._control._scheduling.table import RequestSlot, SlotTable
from cti.inference._step import QueryKind, StateCapture
from cti.protocol import AbortBackendMsg, RequestAbortedMsg, SamplingParams, UserMsg
from cti.sampling import TokenLogprobs
from tests.support.inference.cache import new_cache_manager, prefix_cache
from tests.support.inference.prefix_cache import reference_radix_comparison  # noqa: F401
from tests.support.inference.scheduling import sampled, settle_forward

SYSTEM = torch.arange(1100, dtype=torch.int32) + 5000


def question(index, length=100):
    return torch.cat((SYSTEM, torch.arange(length, dtype=torch.int32) + 1000 * (index + 1)))


@pytest.fixture
def cache():
    """A recurrent radix cache the scheduler fixture runs on and a test observes."""

    return new_cache_manager(
        16384,
        1,
        torch.full((8, 4096), -1, dtype=torch.int32),
        reuse=True,
        snapshot_rows=range(8, 16),
        checks=True,
    )


@pytest.fixture
def scheduler(monkeypatch, cache):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    return InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(8, 4096, 4096, 4096),
        cache_manager=cache,
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
    )


def message(uid, tokens, max_tokens=2):
    return UserMsg(
        uid=uid,
        input_ids=tokens,
        sampling=SamplingParams(
            temperature=0, max_new_tokens=max_tokens, ignore_eos=True
        ).snapshot(),
    )


def submit(scheduler, uid, tokens, max_tokens=2):
    assert scheduler.handle_message(message(uid, tokens, max_tokens)) == ()


class Loop:
    """The engine's overlapped loop: submit a plan, then settle the one before it."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.pending = deque()
        self.cached = {}
        self.plans = []

    def step(self):
        plan = self.scheduler.plan_next()
        if plan is not None:
            self.plans.append(
                [
                    (e.request.uid, e.use.start, e.use.end)
                    for e in plan.rows
                    if e.query_kind is QueryKind.EXTEND
                ]
            )
            for entry in plan.rows:
                request = entry.request
                self.cached.setdefault(request.uid, request.num_cached_prompt_tokens)
            self.scheduler.mark_submitted(plan)
            self.pending.append(plan)
        if len(self.pending) > 1 or (plan is None and self.pending):
            claim = self.pending.popleft()
            settle_forward(self.scheduler, claim, torch.zeros(len(claim.rows), dtype=torch.int32))

    def drain(self):
        for _ in range(200):
            if not (self.scheduler.has_runnable_work or self.pending):
                self.scheduler.require_quiescent()
                return self
            self.step()
        raise AssertionError("no progress")


@pytest.mark.parametrize("siblings", [2, 3, 4])
def test_siblings_behind_a_running_prefill_restore_the_leaders_branch(scheduler, siblings):
    loop = Loop(scheduler)
    submit(scheduler, 0, question(0), max_tokens=4)
    loop.drain()  # the seed's tail snapshot (1152) lies past the shared system prompt
    submit(scheduler, 1, torch.arange(2000, dtype=torch.int32) + 20000, max_tokens=1)
    loop.step()  # the blocker runs
    for uid in range(2, 2 + siblings):
        submit(scheduler, uid, question(uid))
    loop.drain()
    # The leader stops at the branch 1088 to export it; the rest wait for that forward to
    # settle and restore it.
    assert [loop.cached[uid] for uid in range(2, 2 + siblings)] == [0] + [1088] * (siblings - 1)
    prompt_plans = [p for p in loop.plans if p and p[0][0] >= 2]
    assert prompt_plans[:2] == [[(2, 0, 1088)], [(2, 1088, 1200)]]
    assert prompt_plans[2] == [(uid, 1088, 1200) for uid in range(3, 2 + siblings)]


def test_a_short_shared_prefix_runs_its_siblings_together(scheduler):
    loop = Loop(scheduler)
    shared = torch.arange(200, dtype=torch.int32)
    prompts = [
        torch.cat((shared, torch.arange(100, dtype=torch.int32) + 1000 * i)) for i in range(3)
    ]
    submit(scheduler, 0, prompts[0])
    loop.drain()
    for uid in (1, 2):
        submit(scheduler, uid, prompts[uid])
    loop.drain()
    # Restoring the branch would save under 256 tokens: both recompute it at once, each
    # stopping there to export it.
    assert loop.cached == {0: 0, 1: 0, 2: 0}
    assert [p for p in loop.plans if p][1:3] == [
        [(1, 0, 192), (2, 0, 192)],
        [(1, 192, 300), (2, 192, 300)],
    ]


def test_siblings_never_wait_for_an_export_they_cannot_restore(scheduler):
    loop = Loop(scheduler)
    for uid in range(3):
        submit(scheduler, uid, question(uid))
    loop.drain()
    # Nothing is cached: the leader exports only its own tail, past the shared prefix.
    assert loop.cached == {0: 0, 1: 0, 2: 0}
    assert len([p for p in loop.plans if p][0]) == 3


def test_a_leader_without_a_row_holds_no_one_back(scheduler, cache):
    loop = Loop(scheduler)
    submit(scheduler, 0, question(0))
    loop.drain()
    rows = cache.prefix_cache.rows
    leasable = rows.info.free + rows.info.evictable
    leased = [rows.lease() for _ in range(leasable)]
    for uid in (1, 2):
        submit(scheduler, uid, question(uid))
    loop.drain()
    # Both compute the shared prefix together rather than wait for an export with no row.
    assert [p for p in loop.plans if p][1] == [(1, 0, 1200), (2, 0, 1200)]
    for row in leased:
        rows.free(row)
    cache.check_integrity()


def test_held_prompts_keep_their_places_and_others_pass(scheduler):
    loop = Loop(scheduler)
    submit(scheduler, 0, question(0))
    loop.drain()
    for uid in (1, 2):
        submit(scheduler, uid, question(uid))
    submit(scheduler, 3, torch.arange(50, dtype=torch.int32) + 30000)
    submit(scheduler, 4, question(4))
    loop.drain()
    # The leader stops at the branch, and the next prompt passes the held ones; they restore
    # the branch once it settles.
    plans = [p for p in loop.plans if p][1:4]
    assert [[uid for uid, _, _ in plan] for plan in plans] == [[1, 3], [1], [2, 4]]
    assert loop.cached[2] == loop.cached[4] == 1088


def test_aborting_a_held_prompt_forgets_it(scheduler):
    loop = Loop(scheduler)
    submit(scheduler, 0, question(0))
    loop.drain()
    for uid in (1, 2):
        submit(scheduler, uid, question(uid))
    loop.step()
    # The leader runs to the branch; its sibling waits for the export.
    assert loop.plans[-1] == [(1, 0, 1088)]
    assert scheduler.has_runnable_work

    (aborted,) = scheduler.handle_message(AbortBackendMsg(uid=2))

    assert aborted == RequestAbortedMsg(uid=2)
    # Nothing, the deferral's bookkeeping included, keeps the aborted request alive.
    gc.collect()
    assert not [o for o in gc.get_objects() if type(o) is Request and o.uid == 2]
    loop.drain()
    assert all(uid != 2 for plan in loop.plans for uid, _, _ in plan)


def _recurrent_cache(**methods):
    """A recurrent radix cache's deferral answers: its exports alone are restorable."""

    return Mock(
        prefix_cache=prefix_cache(snapshot_rows=range(8, 9)),
        page_size=1,
        published_positions=lambda reservation: tuple(e.position for e in reservation.exports),
        **methods,
    )


def _leader(prompt, positions):
    request = Request(
        uid=0,
        prompt_token_ids=prompt,
        policy=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
    )
    reservation = CacheUse(
        RequestSlot(0, 0),
        len(prompt),
        start=0,
        exports=tuple(StateCapture(p, 9) for p in positions),
    )
    return request, reservation


def test_deferral_is_bounded_and_ends_when_the_export_settles():
    cache = _recurrent_cache(cached_prefix_len=Mock(return_value=0))
    deferral = SharedPrefixDeferral(cache, threshold=512, max_deferrals=3)
    leader, reservation = _leader(question(0), (1088,))
    follower = Request(
        uid=1,
        prompt_token_ids=question(1),
        policy=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
    )
    deferral.track(leader, reservation)
    assert [deferral.defers(follower, ()) for _ in range(4)] == [True, True, True, False]
    other = Request(
        uid=2,
        prompt_token_ids=question(2),
        policy=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
    )
    assert deferral.defers(other, ())
    deferral.untrack(reservation)
    assert not deferral.defers(other, ())


@pytest.mark.parametrize(
    "positions, cached, held",
    [
        ((1088,), 0, True),
        ((1088,), 896, False),  # restoring there now already saves all but 192 tokens
        ((1152,), 0, False),  # past the shared prefix
        ((), 0, False),  # the leader got no row
    ],
)
def test_only_a_restorable_export_that_saves_the_threshold_holds(positions, cached, held):
    cache = _recurrent_cache(cached_prefix_len=Mock(return_value=cached))
    deferral = SharedPrefixDeferral(cache)
    leader, reservation = _leader(question(0), positions)
    deferral.track(leader, reservation)
    follower = Request(
        uid=1,
        prompt_token_ids=question(1),
        policy=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
    )
    assert deferral.defers(follower, ()) is held


def test_a_prompt_identical_to_its_leader_never_waits_for_its_final_position():
    cache = _recurrent_cache(cached_prefix_len=Mock(return_value=0))
    deferral = SharedPrefixDeferral(cache)
    prompt = torch.arange(1024, dtype=torch.int32)
    leader, reservation = _leader(prompt, (1024,))
    deferral.track(leader, reservation)
    follower = Request(
        uid=1,
        prompt_token_ids=prompt,
        policy=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
    )
    assert not deferral.defers(follower, ())


def _prefill_queue(cache):
    return PrefillQueue(cache, SlotTable(8), encoder_cache=EncoderCache(0))


def test_without_room_admission_consults_no_deferral_and_keeps_the_queue(cache):
    queue = _prefill_queue(cache)
    for uid in range(3):
        queue.enqueue(message(uid, question(uid)))
    queue.deferral = Mock(wraps=queue.deferral)

    assert queue.schedule_extensions(0, reserved_size=0) == ()
    queue.deferral.defers.assert_not_called()
    assert queue.pending_ids == [0, 1, 2]
    assert queue.continuations == ()


def test_admission_asks_each_admitted_prompt_once_where_it_exports(cache):
    queue = _prefill_queue(cache)
    for uid in range(6):
        queue.enqueue(message(uid, torch.arange(400, dtype=torch.int32) + 1000 * uid))
    cache.restorable_positions = Mock(wraps=cache.restorable_positions)

    admitted = queue.schedule_extensions(4096, reserved_size=0)

    assert len(admitted) == 6
    assert cache.restorable_positions.call_count == 6


def test_without_any_export_no_prompt_is_compared():
    cache = _recurrent_cache(restorable_positions=Mock(return_value=()), cached_prefix_len=Mock())
    deferral = SharedPrefixDeferral(cache)
    prompt = question(0)
    # An admitted leader: its slot and where its planned forward ends.
    leader = SimpleNamespace(slot=object(), device_len=len(prompt), prompt_token_ids=prompt)
    follower = Request(
        uid=1,
        prompt_token_ids=question(1),
        policy=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
    )

    assert deferral.planned_exports(leader, 0) is None
    cache.restorable_positions.assert_called_once_with(leader.slot, 0, len(prompt))
    assert not deferral.defers(follower, ())
    cache.cached_prefix_len.assert_not_called()


def test_a_forward_promises_only_the_export_it_captures(cache):
    deferral = SharedPrefixDeferral(cache)
    prompt = torch.arange(300, dtype=torch.int32)
    slot = RequestSlot(0, 0)
    cache.try_open_slot(slot, prompt, 1, reserved_size=0)
    # The first of its two forwards captures 256; the second, from there, captures nothing.
    forwards = [(0, 256), (256, 300)]
    planned = [
        deferral.planned_exports(
            SimpleNamespace(slot=slot, device_len=end, prefill_token_ids=prompt), start
        )
        for start, end in forwards
    ]
    assert [None if export is None else export[1] for export in planned] == [(256,), None]


def test_n_siblings_without_recurrent_state_wait_for_one_prefill():
    cache = new_cache_manager(
        16384, 1, torch.full((8, 4096), -1, dtype=torch.int32), reuse=True, checks=True
    )
    scheduler = InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(8, 4096, 4096, 4096),
        cache_manager=cache,
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
    )
    loop = Loop(scheduler)
    prompt = torch.arange(40, dtype=torch.int32)
    for uid in range(3):
        submit(scheduler, uid, prompt)
    loop.drain()
    # One prefill; its siblings restore all of the prompt but its final token.
    assert [loop.cached[uid] for uid in range(3)] == [0, 39, 39]


def test_n_siblings_wait_past_a_shorter_prefix_another_prompt_cached():
    cache = new_cache_manager(
        16384, 1, torch.full((8, 4096), -1, dtype=torch.int32), reuse=True, checks=True
    )
    scheduler = InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(8, 4096, 4096, 4096),
        cache_manager=cache,
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
    )
    loop = Loop(scheduler)
    prompt = torch.arange(40, dtype=torch.int32)
    submit(scheduler, 0, torch.cat((prompt[:3], torch.full((5,), 99, dtype=torch.int32))))
    loop.drain()
    for uid in (1, 2):
        submit(scheduler, uid, prompt)
    loop.drain()
    # The first sibling restores the 3 shared tokens; the second waits for its prefill.
    assert [loop.cached[uid] for uid in (1, 2)] == [3, 39]


def test_prompt_logprobs_gather_over_chunks_and_never_restore_a_prefix():
    cache = new_cache_manager(
        16384, 1, torch.full((8, 4096), -1, dtype=torch.int32), reuse=True, checks=True
    )
    scheduler = InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(4, 8, 4096, 8),
        cache_manager=cache,
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
    )
    prompt = torch.arange(20, dtype=torch.int32)
    replies = []
    for uid in range(2):
        scheduler.handle_message(
            UserMsg(
                uid=uid,
                input_ids=prompt,
                sampling=SamplingParams(
                    temperature=0, max_new_tokens=1, prompt_logprobs=1
                ).snapshot(),
            )
        )
        while scheduler.has_runnable_work:
            plan = scheduler.plan_next()
            targets = memoryview(plan.op.prompt_targets).cast("i")
            prompts, offset = [], 0
            for row in plan.op.rows:
                # Chunks of 8 tokens: each scores the prompt tokens its rows predict, the
                # next chunk's first one included. Score token t as -t, with 7 beside it.
                start = row.kv_prefix_len
                row_targets = [t for t in targets[offset : offset + row.extend_len] if t >= 0]
                offset += row.extend_len
                end = min(row.sequence_len, len(prompt) - 1)
                assert row_targets == prompt[start + 1 : end + 1].tolist()
                assert uid == 0 or start % 8 == 0
                scores = [-float(t) for t in row_targets]
                prompts.append(
                    (
                        start + 1,
                        Sampled(
                            array("i", row_targets).tobytes(),
                            array("f", scores).tobytes(),
                            1,
                            array("i", [7] * len(row_targets)).tobytes(),
                            array("f", [score - 1 for score in scores]).tobytes(),
                        ),
                    )
                )
            scheduler.mark_submitted(plan)
            output = replace(sampled([0] * len(prompts)), prompts=tuple(prompts))
            outcome = scheduler.settle_step(plan, output)
            replies += outcome.replies
    expected = (None, *(TokenLogprobs(p, -float(p), ((7, -float(p) - 1),)) for p in range(1, 20)))
    assert [reply.prompt_logprobs for reply in replies] == [expected, expected]
