"""Real scheduler/cache overlap: followers may plan before the leader settles."""

from collections import deque
from unittest.mock import Mock

import pytest
import torch

from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.strategy import SchedulingLimits, TokenSteps
from cti.inference._step import QueryKind
from cti.protocol import SamplingParams, UserMsg
from tests.support.inference.prefix_cache import (
    make_cache,
    reference_radix_comparison,  # noqa: F401
)
from tests.support.inference.scheduling import settle_forward


def scheduler_for(c, budget=512):
    return InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(
            max_running_requests=8,
            max_batch_tokens=budget,
            max_sequence_length=512,
            max_prompt_tokens=budget,
        ),
        cache_manager=c,
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
    )


def submit(scheduler, uid, tokens):
    scheduler.handle_message(
        UserMsg(
            uid=uid,
            input_ids=tokens,
            sampling=SamplingParams(temperature=0, max_new_tokens=1, ignore_eos=True).snapshot(),
        )
    )


def drain(scheduler, max_steps=150):
    """Run to quiescence with overlapped settlement; return openings and capture count."""
    pending = deque()
    openings = {}
    captures = 0
    steps = 0
    while scheduler.has_runnable_work or pending:
        steps += 1
        assert steps < max_steps, "admission made no progress"
        plan = scheduler.plan_next()
        if plan is not None:
            for entry in plan.rows:
                request = entry.request
                openings.setdefault(request.uid, request.num_cached_prompt_tokens)
            assert all(len(e.captures) <= 1 for e in plan.op.rows)
            captures += sum(len(e.captures) for e in plan.op.rows)
            scheduler.mark_submitted(plan)
            pending.append(plan)
        if plan is None:
            assert pending, "admission is stuck with nothing in flight"
        if len(pending) >= 2 or plan is None:
            claim = pending.popleft()
            settle_forward(scheduler, claim, torch.zeros(len(claim.rows), dtype=torch.int32))
    scheduler.require_quiescent()
    return openings, captures


@pytest.mark.parametrize("budget", [127, 512])
@pytest.mark.parametrize("interleaved", [False, True])
def test_grpo_with_two_rows_never_waits_for_a_row(monkeypatch, budget, interleaved):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    c = make_cache(2, pages=4096)
    scheduler = scheduler_for(c, budget)
    groups, siblings = 6, 4
    order = [(g, s) for g in range(groups) for s in range(siblings)]
    if interleaved:
        order.sort(key=lambda pair: pair[1])
    for g, s in order:
        submit(scheduler, g * siblings + s, torch.arange(193, dtype=torch.int32) + 1000 * g)
    openings, captures = drain(scheduler)
    assert len(openings) == groups * siblings and set(openings.values()) <= {0, 64, 128, 192}
    assert captures > 0
    c.check_integrity()


@pytest.mark.parametrize("budget", [127, 512])
def test_siblings_whose_only_row_is_their_restore_source_run_uncaptured(monkeypatch, budget):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    c = make_cache(1, pages=4096)
    scheduler = scheduler_for(c, budget)
    submit(scheduler, 0, torch.arange(129, dtype=torch.int32))
    assert drain(scheduler) == ({0: 0}, 1)
    for uid in (1, 2):
        submit(scheduler, uid, torch.arange(193, dtype=torch.int32))
    # Chunked or not, the seed exports only its deepest grid point.
    assert drain(scheduler) == ({1: 128, 2: 128}, 0)
    c.check_integrity()


def extends(scheduler):
    """Plan and settle one step at a time to quiescence; each step's extensions as
    ``(uid, start, end, capture positions)``."""
    steps = []
    while scheduler.has_runnable_work:
        plan = scheduler.plan_next()
        steps.append(
            [
                (e.request.uid, e.use.start, e.use.end, [c.position for c in e.use.exports])
                for e in plan.rows
                if e.query_kind is QueryKind.EXTEND
            ]
        )
        scheduler.mark_submitted(plan)
        settle_forward(scheduler, plan, torch.zeros(len(plan.rows), dtype=torch.int32))
    return [step for step in steps if step]


@pytest.mark.parametrize(
    ("budget", "chunks"),
    [
        (64, [(0, 64, [64]), (64, 65, [])]),
        (65, [(0, 65, [64])]),
        (127, [(0, 65, [64])]),
    ],
)
def test_a_capture_rides_the_chunk_that_captures_it(monkeypatch, budget, chunks):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    scheduler = scheduler_for(make_cache(), budget)
    submit(scheduler, 0, torch.arange(65, dtype=torch.int32))
    assert extends(scheduler) == [[(0, *chunk)] for chunk in chunks]


def test_a_chunk_short_of_its_prompt_ends_on_the_grid(monkeypatch):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    scheduler = scheduler_for(make_cache(), 100)
    submit(scheduler, 0, torch.arange(300, dtype=torch.int32))
    # Each budget of 100 rounds down to the 64-token grid; the last chunk captures inside.
    assert extends(scheduler) == [
        [(0, 0, 64, [])],
        [(0, 64, 128, [])],
        [(0, 128, 192, [])],
        [(0, 192, 256, [256])],
        [(0, 256, 300, [])],
    ]


def test_a_prompt_that_cannot_reach_the_grid_waits_a_step(monkeypatch):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    scheduler = scheduler_for(make_cache(), 128)
    submit(scheduler, 0, torch.arange(100, dtype=torch.int32) + 1000)
    submit(scheduler, 1, torch.arange(300, dtype=torch.int32))
    # The 28 tokens the first prompt leaves end off the grid: the second waits for a step.
    assert extends(scheduler)[:2] == [[(0, 0, 100, [64])], [(1, 0, 128, [])]]


def test_a_budget_below_the_grid_fails_the_checks():
    with pytest.raises(RuntimeError, match="does not reach the 64-token prompt grid"):
        scheduler_for(make_cache(), 63)


def test_an_extension_stops_at_the_branch_to_export_it(monkeypatch):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    scheduler = scheduler_for(make_cache())
    shared = torch.arange(256, dtype=torch.int32)
    planned = []
    for uid in range(3):
        submit(
            scheduler, uid, torch.cat((shared, torch.arange(70, dtype=torch.int32) + 1000 * uid))
        )
        planned.append(extends(scheduler))
    # The first exports its tail; the second, whose KV match holds no state, stops at the
    # branch to export it, then exports its tail; the third restores the branch.
    assert planned == [
        [[(0, 0, 326, [320])]],
        [[(1, 0, 256, [256])], [(1, 256, 326, [320])]],
        [[(2, 256, 326, [320])]],
    ]


def test_a_branch_within_a_grid_step_of_the_restore_adds_no_forward(monkeypatch):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    c = make_cache()
    scheduler = scheduler_for(c)
    tokens = torch.arange(400, dtype=torch.int32)
    branch = torch.cat((tokens[:200], torch.arange(100, dtype=torch.int32) + 1000))
    for uid, prompt in enumerate((tokens[:200], branch)):
        submit(scheduler, uid, prompt)
        extends(scheduler)
    # The KV matches 200 tokens, the snapshot 192: aligned down, the branch is the restore.
    assert c.prefix_cache.plan_restore(tokens[:260])[:2] == (192, 200)
    submit(scheduler, 2, tokens[:260])
    assert extends(scheduler) == [[(2, 192, 260, [256])]]


@pytest.mark.parametrize("length", [128, 256])
def test_a_prompt_on_the_grid_exports_its_end_and_its_repeats_hit_from_the_third(
    monkeypatch, length
):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    scheduler = scheduler_for(make_cache())
    prompt = torch.arange(length, dtype=torch.int32)
    deepest = length - 64
    # Its one forward captures its end, past what a hit restores but with its KV: the first
    # repeat finds the branch there, stops to export it, and the next restores it.
    submit(scheduler, 0, prompt)
    assert extends(scheduler) == [[(0, 0, length, [length])]]
    submit(scheduler, 1, prompt)
    assert extends(scheduler) == [[(1, 0, deepest, [deepest])], [(1, deepest, length, [length])]]
    submit(scheduler, 2, prompt)
    assert extends(scheduler) == [[(2, deepest, length, [length])]]


def test_a_stop_to_export_leaves_admission_going(monkeypatch):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    scheduler = scheduler_for(make_cache())
    shared = torch.arange(256, dtype=torch.int32)
    submit(scheduler, 0, torch.cat((shared, torch.arange(70, dtype=torch.int32) + 1000)))
    extends(scheduler)
    submit(scheduler, 1, torch.cat((shared, torch.arange(70, dtype=torch.int32) + 2000)))
    submit(scheduler, 2, torch.arange(100, dtype=torch.int32) + 5000)
    # The second stops at the branch to export it; the third still joins its step.
    assert extends(scheduler) == [
        [(1, 0, 256, [256]), (2, 0, 100, [64])],
        [(1, 256, 326, [320])],
    ]


def test_the_next_turn_restores_the_generated_history_from_the_decode_snapshot(monkeypatch):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())
    scheduler = scheduler_for(make_cache(3))
    turn = torch.arange(100, dtype=torch.int32) + 1000
    scheduler.handle_message(
        UserMsg(
            uid=0,
            input_ids=turn,
            sampling=SamplingParams(temperature=0, max_new_tokens=120, ignore_eos=True).snapshot(),
        )
    )
    assert drain(scheduler) == ({0: 0}, 3)  # the prompt's 64, then decode at 128 and 192
    # The settled outputs are zeros; the next turn carries them and a new question.
    follow_up = torch.cat((turn, torch.zeros(120, dtype=torch.int32), turn[:30]))
    submit(scheduler, 1, follow_up)
    assert drain(scheduler)[0] == {1: 192}
