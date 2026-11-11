"""Cached prompt tokens per request, their totals, and cache occupancy on a real scheduler."""

from unittest.mock import Mock

import pytest
import torch

import cti.inference._control._engine as engine_module
from cti.inference._control._cache import CacheOccupancy
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.strategy import (
    RequestTotals,
    SchedulingLimits,
    TokenSteps,
)
from cti.inference._control._scheduling.table import RequestSlot
from cti.protocol import SamplingParams, TokenUsage, UserMsg
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.inference_steps import engine, user_msg
from tests.support.inference.prefix_cache import (
    make_cache,
    reference_radix_comparison,  # noqa: F401
)
from tests.support.inference.scheduling import settle_forward

_TOKEN = 7


@pytest.fixture(autouse=True)
def _quiet_scheduler(monkeypatch):
    monkeypatch.setattr("cti.inference._control._scheduler.logger", Mock())


def _cache(reuse=True, pages=256):
    return new_cache_manager(
        pages, 1, torch.full((4, 256), -1, dtype=torch.int32), reuse=reuse, checks=True
    )


def _scheduler(cache=None):
    """A scheduler over ``cache``, by default a radix ``_cache()``."""

    return InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(4, 64, 256, 64),
        cache_manager=_cache() if cache is None else cache,
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
    )


def _serve(scheduler, prompts, max_tokens=4):
    """Run prompts to completion together; return each one's terminal usage."""
    first_uid = scheduler.totals.num_requests
    for uid, prompt in enumerate(prompts, first_uid):
        scheduler.handle_message(
            UserMsg(
                uid=uid,
                input_ids=torch.tensor(prompt, dtype=torch.int32),
                sampling=SamplingParams(
                    temperature=0, max_new_tokens=max_tokens, ignore_eos=True
                ).snapshot(),
            )
        )
    usages = {}
    while scheduler.has_runnable_work:
        plan = scheduler.plan_next()
        scheduler.mark_submitted(plan)
        tokens = torch.full((len(plan.rows),), _TOKEN, dtype=torch.int32)
        outcome = settle_forward(scheduler, plan, tokens)
        usages |= {reply.uid: reply.end.usage for reply in outcome.replies if reply.finished}
    scheduler.require_quiescent()
    return [usages[uid] for uid in range(first_uid, first_uid + len(prompts))]


def test_hit_and_miss_report_cached_prompt_tokens_and_totals():
    scheduler = _scheduler()
    prompt = list(range(100, 110))

    assert _serve(scheduler, [prompt]) == [TokenUsage(10, 4, 0)]
    assert _serve(scheduler, [prompt]) == [TokenUsage(10, 4, 9)]
    assert scheduler.totals == RequestTotals(2, 1, TokenUsage(20, 8, 9))
    assert str(scheduler.totals) == (
        "2 requests finished, 1 with prefix hits; prompt 20 tokens, 9 cached (45.0%); "
        "generated 8 tokens"
    )


def test_multi_turn_reuses_the_prompt_and_all_but_the_last_sampled_token():
    scheduler = _scheduler()
    turn = list(range(100, 108))
    _serve(scheduler, [turn])
    follow_up = turn + [_TOKEN] * 4 + [200, 201]

    assert _serve(scheduler, [follow_up]) == [TokenUsage(14, 4, 11)]
    assert scheduler.totals.num_prefix_hits == 1


def test_eviction_reports_only_the_prefix_that_survives():
    scheduler = _scheduler(_cache(pages=40))
    first = list(range(100, 120))
    _serve(scheduler, [first])
    _serve(scheduler, [list(range(300, 330))])

    [usage] = _serve(scheduler, [first])
    assert usage.cached_prompt_tokens == 0
    assert scheduler.totals == RequestTotals(3, 0, TokenUsage(70, 12, 0))


def test_naive_cache_reports_no_cached_tokens():
    cache = _cache(reuse=False)
    scheduler = _scheduler(cache)
    prompt = list(range(100, 110))

    assert _serve(scheduler, [prompt, prompt]) == [TokenUsage(10, 4, 0)] * 2
    _serve(scheduler, [prompt])
    assert scheduler.totals == RequestTotals(3, 0, TokenUsage(30, 12, 0))
    assert cache.occupancy == CacheOccupancy(256, 0, 0, 0, 0, 0)


def test_occupancy_splits_pages_and_snapshot_rows_into_used_cached_and_free():
    cache = make_cache(rows=3, pages=1024)
    scheduler = InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(8, 512, 512, 512),
        cache_manager=cache,
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
    )
    prompt = list(range(1000, 1193))

    assert [u.cached_prompt_tokens for u in _serve(scheduler, [prompt] * 2)] == [0, 192]
    assert [u.cached_prompt_tokens for u in _serve(scheduler, [prompt])] == [192]
    # Siblings in one batch freed the duplicate export; recurrent prefixes keep KV only up to
    # their snapshot at 192.
    assert cache.occupancy == CacheOccupancy(1024, 0, 192, 3, 0, 1)

    tokens = torch.tensor(prompt, dtype=torch.int32)
    cache.try_open_slot(RequestSlot(0, 0), tokens, 4, reserved_size=0)
    assert cache.occupancy == CacheOccupancy(1024, 192, 0, 3, 1, 0)
    assert str(cache.occupancy) == (
        "KV pages 192 used, 0 cached, 1024 total; snapshot rows 1 used, 0 cached, 3 total"
    )


@pytest.mark.parametrize("overlap", [True, False])
def test_engine_logs_totals_every_interval_while_serving_and_once_as_it_drains(
    monkeypatch, overlap
):
    harness = engine(overlap=overlap)
    inference = harness.engine
    now, lines = [0.0], []
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args: lines.append(args))
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: now[0])

    def step(seconds=0.0):
        now[0] += seconds
        before = len(lines)
        inference.step()
        return len(lines) - before

    assert step() == 0  # idle
    inference.handle([user_msg(0, list(range(100, 110)), max_tokens=8)], admitting=True)
    assert [step(), step(30), step(5), step(5)] == [0, 1, 0, 1]
    logged = 0
    while inference.busy:
        logged += step()
    assert logged == 1  # as it drains
    assert step(60) == 0
    assert lines[-1][0] == "Inference: %s; %s; %s"
    assert lines[-1][1].num_requests == 1
    assert lines[-1][3].eager == len(harness.executor.forwards)
