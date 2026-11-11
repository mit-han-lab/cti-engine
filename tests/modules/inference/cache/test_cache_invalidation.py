from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

import ctikernels.radix
from cti.inference._control._cache import CacheManager
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.request import Request
from cti.inference._control._scheduling.strategy import SchedulingLimits, TokenSteps
from cti.inference._control._scheduling.table import SlotTable
from cti.inference._step import QueryKind, Row
from cti.protocol import AbortBackendMsg, SamplingParams
from tests.support.inference.cache import new_cache_manager, prefix_cache
from tests.support.inference.inference_steps import engine, settle, submit, user_msg
from tests.support.inference.prefix_cache import reference_radix_comparison  # noqa: F401


def _pending_request() -> Request:
    return Request(
        uid=7,
        prompt_token_ids=torch.tensor([11, 12, 13], dtype=torch.int32),
        policy=SamplingParams(temperature=0, max_new_tokens=2, ignore_eos=True).snapshot(),
    )


def _scheduler(
    *, reuse: bool = False, page_size: int = 1
) -> tuple[InferenceScheduler, CacheManager]:
    cache_manager = new_cache_manager(
        num_pages=8,
        page_size=page_size,
        page_table=torch.full((1, 8), -1, dtype=torch.int32),
        reuse=reuse,
        checks=True,
    )
    scheduler = InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(
            max_running_requests=1, max_batch_tokens=8, max_sequence_length=8, max_prompt_tokens=8
        ),
        cache_manager=cache_manager,
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
    )
    return scheduler, cache_manager


def _open_request(scheduler: InferenceScheduler, cache_manager: CacheManager) -> Request:
    request = _pending_request()
    slot = scheduler.slots.allocate(request)
    cached_len = cache_manager.try_open_slot(
        slot,
        request.prompt_token_ids,
        request.max_output_tokens,
        reserved_size=0,
    )
    assert cached_len == 0
    request.admit(slot, cached_len=cached_len)
    request.plan_prompt_extension(request.num_prompt_tokens)
    return request


def _execution_entry(request: Request) -> Row:
    return Row(
        request_index=request.slot.index,
        query_kind=QueryKind.EXTEND,
        kv_prefix_len=0,
        sequence_len=request.num_prompt_tokens,
        write_position=None,
    )


def test_engine_release_waits_for_every_admitted_request() -> None:
    harness = engine(overlap=True)
    inference = harness.engine
    inference.handle([user_msg(7, [11, 12, 13], max_tokens=2)], admitting=True)
    inference.step()  # launched, not settled

    assert not inference.drained()
    with pytest.raises(RuntimeError, match="not quiescent"):
        inference.release()

    while not inference.drained():
        inference.step()
    inference.handle([user_msg(8, [11, 12, 13], max_tokens=1)], admitting=False)  # waits
    assert inference.wants_to_run() and not inference.busy
    inference.release()
    inference.release()
    inference.switched(True, None)
    while inference.busy:
        inference.step()

    assert [reply.uid for reply in harness.replies if reply.finished] == [7, 8]


@pytest.mark.parametrize(
    ("owned_state", "reason"),
    [
        ("prefill", "pending prefill"),
        ("decode", "running decode"),
        ("retired", "retired requests await reclamation"),
    ],
)
def test_scheduler_quiescence_rejects_owned_request_state(owned_state: str, reason: str) -> None:
    scheduler, _ = _scheduler()
    scheduler.handle_message(user_msg(7, [11, 12, 13], max_tokens=4))
    if owned_state != "prefill":
        step = submit(scheduler)
        if owned_state == "decode":
            settle(scheduler, step)
        else:  # aborted with its step in flight
            scheduler.handle_message(AbortBackendMsg(uid=7))

    with pytest.raises(RuntimeError, match=reason):
        scheduler.require_quiescent()


def test_scheduler_quiescence_audits_cache_integrity_without_invalidation() -> None:
    scheduler, cache_manager = _scheduler()
    check_integrity = Mock(wraps=cache_manager.check_integrity)
    invalidate_all = Mock(wraps=cache_manager.invalidate_all)
    cache_manager.check_integrity = check_integrity
    cache_manager.invalidate_all = invalidate_all

    scheduler.require_quiescent()

    check_integrity.assert_called_once_with()
    invalidate_all.assert_not_called()


@pytest.mark.parametrize("partial_owner", ["table", "cache"])
def test_quiescence_rejects_partial_admission(partial_owner: str) -> None:
    scheduler, cache_manager = _scheduler()
    pending = _pending_request()
    slot = scheduler.slots.allocate(pending)
    if partial_owner == "cache":
        cached_len = cache_manager.try_open_slot(
            slot,
            pending.prompt_token_ids,
            pending.max_output_tokens,
            reserved_size=0,
        )
        assert cached_len == 0
        # Simulate a failed admission after cache state opened but the table owner
        # incorrectly released its side. CacheManager must still fail closed.
        scheduler.slots.release(slot)

    with pytest.raises(RuntimeError):
        scheduler.require_quiescent()

    if partial_owner == "table":
        scheduler.slots.require_live(slot)
        scheduler.slots.release(slot)
    else:
        with pytest.raises(RuntimeError):
            cache_manager.check_integrity()
        cache_manager.close_slot(slot, prefix_ids=pending.prompt_token_ids)

    scheduler.require_quiescent()
    cache_manager.invalidate_all()
    assert cache_manager.available_size == cache_manager.num_pages


@pytest.mark.parametrize("reuse", [False, True], ids=["naive", "radix"])
def test_empty_cache_invalidation_is_repeatable(reuse: bool) -> None:
    _, cache_manager = _scheduler(reuse=reuse)

    cache_manager.invalidate_all()
    cache_manager.invalidate_all()

    assert cache_manager.prefix_cache.size_info == (0, 0)
    assert cache_manager.available_size == cache_manager.num_pages


def test_radix_reset_rejects_a_protected_handle_before_mutation() -> None:
    cache = prefix_cache()
    tokens = torch.tensor([11, 12], dtype=torch.int32)
    prompt = torch.tensor([11, 12, 13], dtype=torch.int32)
    handle = cache.publish(tokens, torch.arange(2, dtype=torch.int32)).handle
    cache.lock(handle)

    with pytest.raises(RuntimeError):
        cache.reset()

    assert cache.size_info == (0, 2)
    assert cache.plan_restore(prompt).cached_len == 2
    cache.unlock(handle)
    cache.reset()
    assert cache.size_info == (0, 0)
    assert cache.plan_restore(prompt).cached_len == 0


def test_release_empties_the_radix_the_snapshot_rows_and_the_encoder_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ctikernels.radix, "warm_up", lambda: None)
    harness = engine(
        max_batch_tokens=256,
        max_seq_len=256,
        num_pages=512,
        prefix_cache="radix",
        snapshot_rows=range(4, 8),
    )
    inference = harness.engine
    inference.handle([user_msg(7, range(100, 230), max_tokens=2)], admitting=True)
    while inference.busy:
        inference.step()
    inference._encoder_cache.plan([(b"image", 4)])
    occupancy = inference._cache.occupancy
    assert occupancy.num_cached_pages and occupancy.num_cached_snapshot_rows, occupancy
    assert b"image" in inference._encoder_cache.plan([(b"image", 4)])[0]

    inference.release()

    occupancy = inference._cache.occupancy
    assert (occupancy.num_used_pages, occupancy.num_cached_pages) == (0, 0), occupancy
    assert (occupancy.num_used_snapshot_rows, occupancy.num_cached_snapshot_rows) == (0, 0)
    assert inference._cache.prefix_cache.rows.info.free == 4
    assert b"image" not in inference._encoder_cache.plan([(b"image", 4)])[0]


@pytest.mark.parametrize(
    ("free_indices", "diagnostic"),
    [
        pytest.param([0, 1, 2, 2], "duplicates=[2]", id="duplicate"),
        pytest.param([0, 2], "missing=[1]", id="missing"),
        pytest.param([0, 1, 2, 7], "unexpected=[7]", id="unexpected"),
    ],
)
def test_table_quiescence_rejects_corrupt_free_accounting_without_mutation(
    free_indices: list[int],
    diagnostic: str,
) -> None:
    table = SlotTable(3)
    slot = table.allocate(_pending_request())
    table.release(slot)
    # Only corrupting SlotTable's own free list can produce the accounting it rejects.
    table._free = free_indices.copy()
    # Only its private state shows the rejecting check left it unmutated.
    generations = table._generations.copy()
    free_order = table._free.copy()

    with pytest.raises(RuntimeError) as caught:
        table.require_all_released()

    assert diagnostic in str(caught.value)
    assert table._generations == generations
    assert table._free == free_order


def test_cache_invalidation_raises_a_prefix_reset_failure() -> None:
    _, cache_manager = _scheduler(reuse=True)
    failure = RuntimeError("injected prefix reset failure")
    cache_manager.prefix_cache.reset = Mock(side_effect=failure)

    with pytest.raises(RuntimeError) as caught:
        cache_manager.invalidate_all()

    assert caught.value is failure


def test_paged_invalidation_preserves_stale_rows_until_live_range_is_rewritten() -> None:
    scheduler, cache_manager = _scheduler(reuse=True, page_size=2)
    cache_manager.page_table.fill_(12345)
    stale_table = cache_manager.page_table.clone()

    cache_manager.invalidate_all()

    assert torch.equal(cache_manager.page_table, stale_table)
    occupancy = cache_manager.occupancy
    assert (occupancy.num_used_pages, occupancy.num_cached_pages) == (0, 0)
    assert cache_manager.available_size == cache_manager.num_pages * cache_manager.page_size
    request = _open_request(scheduler, cache_manager)
    _, mapping_updates = cache_manager.reserve(
        (request.slot,),
        (_execution_entry(request),),
    )
    assert mapping_updates is not None
    torch.testing.assert_close(
        cache_manager.page_table[request.slot.index, :2],
        torch.arange(2, dtype=torch.int32),
    )
    assert torch.equal(cache_manager.page_table[request.slot.index, 2:], stale_table[0, 2:])


def test_quiescent_invalidation_discards_radix_visibility_without_scrubbing_table() -> None:
    scheduler, cache_manager = _scheduler(reuse=True)
    table = scheduler.slots
    request = _open_request(scheduler, cache_manager)
    first_slot = request.slot
    entry = _execution_entry(request)
    reservations, mapping_updates = cache_manager.reserve((first_slot,), (entry,))
    assert mapping_updates is not None
    reservation = reservations[0]
    cache_manager.settle(
        reservation, reservation.end - reservation.start, prefix_ids=request.token_ids
    )
    cache_manager.close_slot(first_slot, prefix_ids=request.token_ids)
    table.release(first_slot)
    assert cache_manager.prefix_cache.size_info.evictable_size == len(request.token_ids)

    cache_manager.page_table.fill_(12345)
    stale_table = cache_manager.page_table.clone()

    cache_manager.invalidate_all()
    cache_manager.invalidate_all()

    assert cache_manager.prefix_cache.size_info == (0, 0)
    assert cache_manager.available_size == cache_manager.num_pages
    assert torch.equal(cache_manager.page_table, stale_table)

    next_request = _pending_request()
    next_slot = table.allocate(next_request)
    assert next_slot.index == first_slot.index
    assert next_slot.generation == first_slot.generation + 1
    assert (
        cache_manager.try_open_slot(
            next_slot,
            next_request.token_ids,
            next_request.max_output_tokens,
            reserved_size=0,
        )
        == 0
    )
    _, next_mapping_updates = cache_manager.reserve((next_slot,), (entry,))
    assert next_mapping_updates is not None
    assert not torch.equal(cache_manager.page_table, stale_table)
