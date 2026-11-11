from __future__ import annotations

import pytest
import torch

from cti.inference._control._cache import CacheManager
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.prefill import PrefillQueue
from cti.inference._control._scheduling.request import Request, RequestBatch
from cti.inference._control._scheduling.strategy import SchedulingLimits
from cti.inference._control._scheduling.table import RequestSlot, SlotTable
from cti.inference._step import QueryKind
from cti.protocol import AdmissionPriority, SamplingParams, UserMsg
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.scheduling import ScriptedDecode, request_ids, reservations


class _OpeningCache:
    available_size = 32
    page_size = 1

    def __init__(self) -> None:
        self.open_slot_calls: list[tuple[RequestSlot, torch.Tensor, int, int]] = []

    def try_open_slot(
        self,
        slot: RequestSlot,
        prompt_token_ids: torch.Tensor,
        max_output_tokens: int,
        *,
        reserved_size: int,
        lookahead_tokens: int,
        prefix: bool,
    ) -> int:
        assert lookahead_tokens == 0 and prefix
        self.open_slot_calls.append((slot, prompt_token_ids, max_output_tokens, reserved_size))
        return 0

    def restorable_positions(self, slot: object, start: int, end: int) -> tuple[int, ...]:
        return ()

    def extension_end(self, slot: object, end: int, prompt_end: int) -> int:
        return end

    def export_stop(self, slot: object, start: int, end: int) -> None:
        return None


def _message(
    uid: int,
    *,
    priority: AdmissionPriority = AdmissionPriority.NORMAL,
    input_len: int = 2,
) -> UserMsg:
    return UserMsg(
        uid=uid,
        input_ids=torch.arange(input_len, dtype=torch.int32),
        sampling=SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=2).snapshot(),
        priority=priority,
    )


def _pending(uid: int) -> Request:
    return Request(
        uid=uid,
        prompt_token_ids=torch.tensor([10, 11], dtype=torch.int32),
        policy=SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=2).snapshot(),
    )


def _request(request: Request, slot: RequestSlot) -> Request:
    """``request`` admitted into ``slot`` with its whole prompt planned."""

    request.admit(slot, cached_len=0)
    request.plan_prompt_extension(request.num_prompt_tokens)
    return request


def _scheduler(
    max_running_reqs: int = 1,
) -> tuple[InferenceScheduler, ScriptedDecode, CacheManager]:
    page_table = torch.full((max_running_reqs + 1, 8), -1, dtype=torch.int32)
    cache = new_cache_manager(max_running_reqs * 8, 1, page_table, reuse=False, checks=True)
    decode = ScriptedDecode()  # each test scripts the batch the scheduler plans
    scheduler = InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(
            max_running_requests=max_running_reqs,
            max_batch_tokens=4,
            max_sequence_length=8,
            max_prompt_tokens=4,
        ),
        cache_manager=cache,
        stop_token_ids=(99,),
        vocab_size=1 << 20,
        decode=decode,
        encoder_cache=EncoderCache(0),
    )
    return scheduler, decode, cache


def _allocate_request(scheduler: InferenceScheduler, cache: CacheManager, uid: int) -> Request:
    """Admit a request into a slot of the scheduler's table and its ``cache``."""

    pending = _pending(uid)
    request = _request(pending, scheduler.slots.allocate(pending))
    opened = cache.try_open_slot(
        request.slot, request.prompt_token_ids, request.max_output_tokens, reserved_size=0
    )
    assert opened == 0
    return request


def test_plan_binds_its_cache_reservations() -> None:
    scheduler, decode, cache = _scheduler()
    request = _allocate_request(scheduler, cache, 7)
    decode.batch = RequestBatch(extend_requests=(request,))

    plan = scheduler.plan_next()

    assert plan is not None
    assert request_ids(plan) == (request.uid,)
    assert all(row.query_kind is QueryKind.EXTEND for row in plan.op.rows)
    assert plan.op.pages is not None
    assert reservations(plan)[0].slot is request.slot
    assert cache.in_use(request.slot)


def test_slot_reuse_advances_generation() -> None:
    table = SlotTable(1)

    first = table.allocate(_pending(1))
    table.release(first)
    second = table.allocate(_pending(2))

    assert second.index == first.index
    assert second.generation == first.generation + 1


def test_duplicate_slot_release_does_not_duplicate_capacity() -> None:
    table = SlotTable(1)
    slot = table.allocate(_pending(1))
    table.release(slot)

    with pytest.raises(RuntimeError, match="not live"):
        table.release(slot)

    assert table.available_size == 1


def test_stale_slot_release_cannot_release_current_slot() -> None:
    table = SlotTable(1)
    stale = table.allocate(_pending(1))
    table.release(stale)
    current = table.allocate(_pending(2))

    with pytest.raises(RuntimeError, match="stale"):
        table.release(stale)

    assert table.available_size == 0
    table.require_live(current)


def test_stale_slot_cannot_be_planned() -> None:
    scheduler, decode, cache = _scheduler()
    table = scheduler.slots
    stale = table.allocate(_pending(6))
    table.release(stale)
    current = table.allocate(_pending(8))
    decode.batch = RequestBatch(extend_requests=(_request(_pending(7), stale),))

    with pytest.raises(RuntimeError, match="stale"):
        scheduler.plan_next()

    cache.check_integrity()  # nothing reserved
    assert table.available_size == 0
    table.require_live(current)


def test_chunk_continuation_reuses_request_slot() -> None:
    table = SlotTable(1)
    cache = _OpeningCache()
    prefill = PrefillQueue(cache, table, encoder_cache=EncoderCache(0))
    message = UserMsg(
        uid=7,
        input_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        sampling=SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=2).snapshot(),
    )
    prefill.enqueue(message)

    first_batch = prefill.schedule_extensions(2, reserved_size=0)
    first = first_batch[0]
    first.mark_submitted()

    final_batch = prefill.schedule_extensions(2, reserved_size=0)
    final = final_batch[0]

    assert final is first
    assert final.slot is first.slot
    assert final.slot.generation == first.slot.generation
    assert table.available_size == 0
    assert len(cache.open_slot_calls) == 1
    slot, prompt_token_ids, max_output_tokens, reserved_size = cache.open_slot_calls[0]
    assert slot == first.slot
    assert torch.equal(prompt_token_ids, message.input_ids)
    assert max_output_tokens == message.sampling.max_new_tokens
    assert reserved_size == 0


def test_waiting_requests_are_admitted_by_priority_then_arrival() -> None:
    table = SlotTable(4)
    prefill = PrefillQueue(_OpeningCache(), table, encoder_cache=EncoderCache(0))
    prefill.enqueue(_message(1, priority=AdmissionPriority.LOW))
    prefill.enqueue(_message(2, priority=AdmissionPriority.HIGH))
    prefill.enqueue(_message(3, priority=AdmissionPriority.HIGH))
    prefill.enqueue(_message(4))

    batch = prefill.schedule_extensions(8, reserved_size=0)

    assert tuple(request.uid for request in batch) == (2, 3, 4, 1)


def test_waiting_priority_does_not_preempt_admitted_chunk_continuation() -> None:
    table = SlotTable(2)
    prefill = PrefillQueue(_OpeningCache(), table, encoder_cache=EncoderCache(0))
    prefill.enqueue(_message(1, priority=AdmissionPriority.LOW, input_len=4))

    first_batch = prefill.schedule_extensions(2, reserved_size=0)
    continuation = first_batch[0]
    continuation.mark_submitted()

    prefill.enqueue(_message(2, priority=AdmissionPriority.HIGH))
    second_batch = prefill.schedule_extensions(2, reserved_size=0)

    assert tuple(request.uid for request in second_batch) == (1,)
    assert second_batch[0] is continuation
    assert second_batch[0].slot is continuation.slot


def test_duplicate_slot_batch_is_rejected() -> None:
    scheduler, decode, cache = _scheduler()
    request = _allocate_request(scheduler, cache, 7)
    decode.batch = RequestBatch(extend_requests=(request, request))

    with pytest.raises(RuntimeError, match="more than once"):
        scheduler.plan_next()
