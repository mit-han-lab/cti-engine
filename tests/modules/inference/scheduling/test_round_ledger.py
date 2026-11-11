"""The round ledger alone: depth, discarded successors and requests ended on the host.

Each case drives the scheduler's side by hand: it settles prompts, appends published tokens
and finishes requests; device records are written out row by row.
"""

from __future__ import annotations

import pytest
import torch

from cti.forward import RoundShape
from cti.inference import RoundCompletion
from cti.inference._control._cache import CacheManager, CacheUse
from cti.inference._control._scheduling.request import Request, RowClaim
from cti.inference._control._scheduling.rounds import (
    DeviceRow,
    RoundLedger,
    RoundTotals,
)
from cti.inference._control._scheduling.strategy import SchedulingLimits
from cti.inference._control._scheduling.table import RequestSlot, Slot
from cti.inference._step import QueryKind, Row
from cti.protocol import SamplingParams
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.scheduling import round_record

_Q = 4
_EOS = 9
_PROMPT = 5


def _ledger() -> tuple[RoundLedger, CacheManager]:
    cache = new_cache_manager(
        256, 1, torch.full((2, 64), -1, dtype=torch.int32), reuse=False, checks=True
    )
    limits = SchedulingLimits(
        max_running_requests=2, max_batch_tokens=16, max_sequence_length=64, max_prompt_tokens=16
    )
    ledger = RoundLedger(
        RoundShape(_Q, 2, 4),
        limits=limits,
        cache_manager=cache,
        stop_token_ids=frozenset({_EOS}),
        checks=True,
    )
    return ledger, cache


def _submit_prompt(cache: CacheManager) -> tuple[Request, CacheUse]:
    """A request whose whole prompt is submitted: its anchor is the one uncached token."""

    prompt = torch.arange(10, 10 + _PROMPT, dtype=torch.int32)
    request = Request(
        uid=0,
        prompt_token_ids=prompt,
        policy=SamplingParams(temperature=0, max_new_tokens=32, seed=0).snapshot(),
    )
    slot = RequestSlot(0, 0)
    assert cache.try_open_slot(slot, prompt, 32, reserved_size=0, lookahead_tokens=_Q - 1) == 0
    request.admit(slot, cached_len=0)
    request.plan_prompt_extension(_PROMPT)
    entry = Row(0, QueryKind.EXTEND, 0, _PROMPT, _PROMPT)
    (use,), _ = cache.reserve([slot], [entry])
    request.mark_submitted()
    return request, use


def _running(cache: CacheManager) -> Request:
    request, use = _submit_prompt(cache)
    cache.settle(use, _PROMPT)
    request.append_sampled_token(100)
    return request


def _record(
    tokens: list[int], accepted: int, committed: int, done: bool, frontier: int
) -> RoundCompletion:
    return round_record([tokens], [accepted], [committed], [done], [frontier])


def _settle(
    ledger: RoundLedger,
    cache: CacheManager,
    slot: Slot,
    rows: tuple[RowClaim, ...],
    record: RoundCompletion,
) -> list[DeviceRow]:
    """The round's device rows, of ``slot``'s request, each use then committing its row's
    tokens, as the scheduler commits a round no stop string cuts."""

    settled = ledger.settle(rows, [slot], record)
    for row, device in zip(rows, settled, strict=True):
        cache.settle(row.use, len(device.tokens), record=device.record)
    return settled


def test_a_third_round_waits_for_the_oldest_to_settle() -> None:
    ledger, cache = _ledger()
    request = _running(cache)
    slot = Slot(request.slot, request)
    _, first = ledger.plan([slot])
    _, second = ledger.plan([slot])

    assert ledger.plan([slot]) is None
    # All drafts rejected: the round emits its correction and commits the anchor.
    assert _settle(ledger, cache, slot, first, _record([1, 2, 3, 4], 0, 1, False, _PROMPT)) == [
        DeviceRow([1], None, (_PROMPT, 1))
    ]
    request.append_block([1], cached_len=_PROMPT + 1)
    assert ledger.plan([slot]) is not None
    # The second round continues where the first left the device row.
    assert _settle(
        ledger, cache, slot, second, _record([5, 6, 7, 8], 3, 4, False, _PROMPT + 1)
    ) == [DeviceRow([5, 6, 7, 8], None, (_PROMPT + 1, 4))]
    assert ledger.totals == RoundTotals(2, 3)


def test_a_discarded_successor_shows_no_tokens_and_retires_its_use() -> None:
    ledger, cache = _ledger()
    request = _running(cache)
    slot = Slot(request.slot, request)
    _, first = ledger.plan([slot])
    _, successor = ledger.plan([slot])

    assert _settle(ledger, cache, slot, first, _record([1, _EOS, 3, 4], 1, 2, True, _PROMPT)) == [
        DeviceRow([1, _EOS], None, (_PROMPT, 2))
    ]
    request.append_block([1, _EOS], cached_len=_PROMPT + 2)
    request.mark_finished()
    # The device knows the first round finished the request: a live successor diverges.
    with pytest.raises(RuntimeError, match="diverged from settlement"):
        ledger.settle(successor, [slot], _record([5, 6, 7, 8], 0, 1, False, _PROMPT + 2))
    assert _settle(
        ledger, cache, slot, successor, _record([5, 6, 7, 8], -1, 0, True, _PROMPT + 2)
    ) == [DeviceRow([], None, (_PROMPT + 2, 0))]

    assert not cache.in_use(request.slot)
    # One-token pages, one slot, no use in flight: the slot holds exactly its frontier.
    assert cache.occupancy.num_used_pages == _PROMPT + 2
    assert ledger.totals == RoundTotals(1, 1)


@pytest.mark.parametrize("ending", ["eos", "abort"])
def test_a_request_ended_on_the_host_runs_its_rounds_on(ending: str) -> None:
    ledger, cache = _ledger()
    request, use = _submit_prompt(cache)
    slot = Slot(request.slot, request)
    # Rounds join before the prompt settles: the device holds the anchor.
    _, joined = ledger.plan([slot])
    _, successor = ledger.plan([slot])
    cache.settle(use, _PROMPT)
    if ending == "eos":
        # The device never checks the first token for a stop.
        request.append_sampled_token(_EOS)
    request.mark_finished()

    # Neither row is done on the device: each publishes nothing and continues the last.
    assert _settle(ledger, cache, slot, joined, _record([1, 2, 3, 4], 3, 4, False, _PROMPT)) == [
        DeviceRow([], None, (_PROMPT, 4))
    ]
    with pytest.raises(RuntimeError, match="diverged from settlement"):
        ledger.settle(successor, [slot], _record([5, 6, 7, 8], 0, 1, False, _PROMPT))
    assert _settle(
        ledger, cache, slot, successor, _record([5, 6, 7, 8], 0, 1, False, _PROMPT + 4)
    ) == [DeviceRow([], None, (_PROMPT + 4, 1))]

    assert not cache.in_use(request.slot)
    assert ledger.totals == RoundTotals()
