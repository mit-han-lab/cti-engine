from collections import deque

import pytest
import torch

from cti.inference._control._prefix_state import plan_exports
from cti.inference._control._scheduling.table import RequestSlot
from cti.inference._step import QueryKind, Row, StateCapture
from cti.kvcache import next_snapshot_position
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.prefix_cache import (
    make_cache,
    reference_radix_comparison,  # noqa: F401
)


def extend_entry(index, start, end):
    return Row(index, QueryKind.EXTEND, start, end)


def decode_entry(index, position):
    return Row(index, QueryKind.DECODE, position, position + 1, position + 1)


def prompt(n=193, offset=0):
    return torch.arange(n, dtype=torch.int32) + offset


def open_slot(c, index, tokens):
    slot = RequestSlot(index, 0)
    return slot, c.try_open_slot(slot, tokens, 1, reserved_size=0)


def reserve(c, slot, start, end):
    return c.reserve((slot,), (extend_entry(slot.index, start, end),))[0][0]


def commit(c, use, tokens):
    c.settle(use, use.end - use.start, prefix_ids=tokens[: use.end])


def serve(c, tokens, *, chunk=None, index=0):
    """Prefill ``tokens`` in forwards of at most ``chunk`` tokens, each ending where the
    scheduler ends it (``extension_end``, ``export_stop``), settling each; return the cached
    length and the exported positions."""
    slot, cached = open_slot(c, index, tokens)
    start, exports = cached, []
    while start < len(tokens):
        end = min(len(tokens), start + (chunk or len(tokens)))
        end = c.extension_end(slot, end, len(tokens))
        reservation = reserve(c, slot, start, c.export_stop(slot, start, end) or end)
        exports += [export.position for export in reservation.exports]
        commit(c, reservation, tokens)
        start = reservation.end
    c.close_slot(slot, prefix_ids=tokens)
    return cached, exports


@pytest.mark.parametrize(
    "length, chunk, exports",
    [
        # An end on the grid exports: no repeat restores it (a hit recomputes the final
        # token), but extensions do, and its KV shows later prompts their branch.
        (64, None, [64]),
        (65, None, [64]),
        (128, None, [128]),  # 64 is missed, as in SGLang
        (128, 100, [64, 128]),  # a chunk ending at 64 exports it
        (193, None, [192]),
        (193, 64, [192]),  # chunk boundaries export nothing
        (500, 200, [448]),  # chunks end on the grid: 192, 384, then 448 inside the last
    ],
)
def test_a_miss_exports_only_its_deepest_restorable_grid_point(length, chunk, exports):
    c = make_cache()
    assert serve(c, prompt(length), chunk=chunk) == (0, exports)
    assert c.prefix_cache.rows.info.cached == len(exports)
    c.check_integrity()


def test_a_full_prompt_hit_restores_the_deepest_snapshot():
    c = make_cache()
    assert serve(c, prompt(500), chunk=200) == (0, [448])
    assert serve(c, prompt(500), chunk=200) == (448, [])
    assert c.prefix_cache.rows.info.cached == 1
    c.check_integrity()


def test_a_partial_hit_past_every_snapshot_recomputes_and_exports_the_branch():
    c = make_cache()
    serve(c, prompt(500), chunk=200)
    # The KV matches 300 tokens but no snapshot lies at or below 300: recompute from 0,
    # exporting where the prompts part; the next prompt sharing those 300 restores there.
    shares = [torch.cat((prompt(300), prompt(150, 1000 * (i + 1)))) for i in range(2)]
    assert [serve(c, tokens, chunk=200) for tokens in shares] == [(0, [256, 448]), (256, [448])]
    c.check_integrity()


def test_each_turn_restores_the_previous_turns_deepest_snapshot():
    c = make_cache(4)
    trace = [serve(c, prompt(length)) for length in (100, 230, 300, 420, 470, 500)]
    assert trace == [(0, [64]), (64, [192]), (192, [256]), (256, [384]), (384, [448]), (448, [])]
    c.check_integrity()


def test_a_branch_waits_for_the_chunk_that_crosses_it():
    c = make_cache(6)
    serve(c, prompt(300))
    divergent = torch.cat((prompt(200), prompt(200, 1000)))
    assert serve(c, divergent, chunk=150) == (0, [192, 384])
    c.check_integrity()


def test_a_branch_export_serves_the_next_sibling():
    c = make_cache()
    shared = prompt(256)
    questions = [torch.cat((shared, prompt(70, 1000 * (i + 1)))) for i in range(3)]
    trace = [serve(c, tokens) for tokens in questions]
    # The first tail lies past the branch; the second exports it; the third restores there.
    assert trace == [(0, [320]), (0, [256, 320]), (256, [320])]
    c.check_integrity()


def test_restore_is_locked_until_the_first_reservation_settles():
    c = make_cache(2)
    serve(c, prompt(129))
    slot, cached = open_slot(c, 0, prompt(257))
    assert cached == 128 and c.prefix_cache.rows.info.locked == 1
    first = reserve(c, slot, 128, 192)
    assert first.restore_row is not None and first.exports == ()
    commit(c, first, prompt(257))
    assert c.prefix_cache.rows.info.locked == 0
    second = reserve(c, slot, 192, 257)
    assert second.restore_row is None and [e.position for e in second.exports] == [256]
    commit(c, second, prompt(257))
    c.close_slot(slot, prefix_ids=prompt(257))
    c.check_integrity()


def test_a_prompt_forward_off_the_grid_exports_nothing():
    c = make_cache()
    slot, _ = open_slot(c, 0, prompt(193))
    commit(c, reserve(c, slot, 0, 100), prompt(193))
    # After a cut before an image: its capture, 164, is off the grid, so 192 is missed.
    assert reserve(c, slot, 100, 193).exports == ()


@pytest.mark.parametrize(
    "start, end, points, exports",
    [
        (0, 300, {128, 256}, (256,)),  # one capture, after the whole 64-token multiples
        (0, 300, {128}, ()),  # not where the prefill captures
        (64, 100, {64, 128}, ()),  # under 64 tokens: no capture
    ],
)
def test_a_forward_exports_only_its_one_capture_point(start, end, points, exports):
    assert plan_exports(start, end, points, 64) == exports


def test_a_prompt_forward_exporting_off_its_capture_point_fails_the_checks(monkeypatch):
    c = make_cache()
    slot, _ = open_slot(c, 0, prompt(300))
    monkeypatch.setattr(c._snapshots, "plan", lambda *_: (None, (StateCapture(128, 8),)))
    with pytest.raises(RuntimeError, match="exports off its capture point"):
        reserve(c, slot, 0, 300)


@pytest.mark.parametrize("locked", [False, True])
def test_a_branch_stops_a_forward_only_with_a_row_to_export_into(locked):
    c = make_cache(1)
    first = torch.cat((prompt(256), prompt(70, 1000)))
    serve(c, first)  # its one row holds the tail at 320
    if locked:  # a hit restoring that row locks it: none is free or reclaimable
        open_slot(c, 1, first)
    slot, cached = open_slot(c, 0, torch.cat((prompt(256), prompt(70, 2000))))
    assert cached == 0
    assert c.export_stop(slot, 0, 326) == (None if locked else 256)


def page_cache():
    """A cache of 128-token pages: snapshots on a 128-token grid, captures on 64."""

    table = torch.full((8, 8), -1, dtype=torch.int32)
    return new_cache_manager(64, 128, table, reuse=True, snapshot_rows=range(8, 11), checks=True)


def test_128_token_pages_end_a_chunk_on_their_grid():
    c = page_cache()
    slot, _ = open_slot(c, 0, prompt(300))
    assert [c.extension_end(slot, end, 300) for end in (127, 200, 300)] == [0, 128, 300]


@pytest.mark.parametrize(
    "length, chunk, exports",
    [
        (200, None, []),  # its capture, 192, lies off the grid: 128 is missed
        (300, None, [256]),
        (300, 200, [256]),  # the chunk ends at 128, which is no export point
    ],
)
def test_128_token_pages_export_on_their_grid(length, chunk, exports):
    c = page_cache()
    assert serve(c, prompt(length), chunk=chunk) == (0, exports)
    c.check_integrity()


def test_128_token_pages_stop_at_the_branch():
    c = page_cache()
    questions = [torch.cat((prompt(256), prompt(150, 1000 * (i + 1)))) for i in range(3)]
    trace = [serve(c, tokens) for tokens in questions]
    assert trace == [(0, [384]), (0, [256, 384]), (256, [384])]
    c.check_integrity()


@pytest.mark.parametrize("rows", [1, 2])
def test_a_forward_without_a_free_row_skips_its_export(rows):
    c = make_cache(rows)
    serve(c, prompt(129))  # the readers' restore source
    readers = [open_slot(c, index, prompt(193)) for index in range(2)]
    assert [cached for _, cached in readers] == [128, 128]
    # Locked restore sources are never reclaimed; with one row the export is skipped.
    reservations = [reserve(c, slot, 128, 193) for slot, _ in readers]
    exported = [len(r.exports) for r in reservations]
    assert exported == ([0, 0] if rows == 1 else [1, 0])
    for reservation in reservations:
        commit(c, reservation, prompt(193))
    for slot, _ in readers:
        c.close_slot(slot, prefix_ids=prompt(193))
    assert c.prefix_cache.rows.info.cached == rows
    c.check_integrity()


def test_an_export_evicts_the_least_recently_used_unlocked_snapshot():
    c = make_cache(2)
    serve(c, prompt(65))
    serve(c, prompt(65, 1000))
    c.cached_prefix_len(prompt(65))  # touch the first
    assert serve(c, prompt(65, 2000)) == (0, [64])
    assert serve(c, prompt(65)) == (64, [])
    assert serve(c, prompt(65, 1000)) == (0, [64])
    c.check_integrity()


def test_duplicate_exports_in_one_batch_keep_the_first_and_free_the_rest():
    c = make_cache(3)
    tokens = prompt(193)
    slots = [open_slot(c, index, tokens)[0] for index in range(2)]
    reservations, _ = c.reserve(slots, [extend_entry(s.index, 0, 193) for s in slots])
    assert [[e.position for e in r.exports] for r in reservations] == [[192], [192]]
    assert c.prefix_cache.rows.info.leased == 2
    for reservation in reservations:
        commit(c, reservation, tokens)
    assert c.prefix_cache.rows.info == (2, 0, 1, 0, 1)
    for slot in slots:
        c.close_slot(slot, prefix_ids=tokens)
    c.check_integrity()


def test_kv_eviction_frees_a_snapshot_row_and_falls_back_to_a_shallower_one():
    c = make_cache(3, pages=400)
    serve(c, prompt(100))
    assert serve(c, prompt(150)) == (64, [128])
    assert c.prefix_cache.rows.info.cached == 2
    c._allocate_pages(len(c._free_pages) + 50)  # evict the path past 100
    assert c.prefix_cache.rows.info.cached == 1
    assert c.cached_prefix_len(prompt(150)) == 64


@pytest.mark.parametrize("outcome", ["released", "unnamed"])
def test_an_unpublishable_export_frees_its_row(outcome):
    c = make_cache()
    tokens = prompt()
    slot, _ = open_slot(c, 0, tokens)
    reservation = reserve(c, slot, 0, 193)
    if outcome == "released":
        c.settle(reservation, 0)
    else:  # committed without prefix ids: the radix cannot name the state
        c.settle(reservation, 193)
    c.close_slot(slot, prefix_ids=tokens)
    assert c.prefix_cache.rows.info == (3, 0, 0, 0, 0)
    c.check_integrity()


def test_closing_before_submission_unlocks_the_restore():
    c = make_cache()
    serve(c, prompt(129))
    slot, cached = open_slot(c, 0, prompt(193))
    assert cached == 128 and c.prefix_cache.rows.info.locked == 1
    c.close_slot(slot, prefix_ids=prompt(193))
    assert c.prefix_cache.rows.info.locked == 0
    c.check_integrity()


@pytest.mark.parametrize("rows", [0, 2])
def test_off_grid_reservations_own_no_snapshot_rows(rows):
    c = make_cache(rows)
    tokens = prompt(2)
    slot, _ = open_slot(c, 0, tokens)
    for entry in (extend_entry(slot.index, 0, 2), decode_entry(slot.index, 2)):
        reservation = c.reserve((slot,), (entry,))[0][0]
        assert reservation.restore_row is None and reservation.exports == ()
        c.settle(reservation, reservation.end - reservation.start)
    c.close_slot(slot, prefix_ids=torch.arange(3, dtype=torch.int32))
    c.check_integrity()


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("rows", [0, 2])
def test_only_a_reusing_cache_with_snapshot_rows_indexes_them(reuse, rows):

    cache = new_cache_manager(
        32,
        1,
        torch.full((2, 16), -1, dtype=torch.int32),
        reuse=reuse,
        snapshot_rows=range(2, 2 + rows),
        checks=True,
    )
    indexed = cache.prefix_cache.rows
    assert (0 if indexed is None else len(indexed)) == (rows if reuse else 0)


def decode(c, slot, start, end, *, abort_at=None):
    """Decode positions ``[start, end)`` two steps in flight, releasing the steps from
    ``abort_at`` on; return each step's exports."""
    pending, exports = deque(), []
    for position in range(start, end):
        reservation = c.reserve((slot,), (decode_entry(slot.index, position),))[0][0]
        exports += [(e.position, e.row) for e in reservation.exports]
        pending.append(reservation)
        while len(pending) > (1 if position < end - 1 else 0):
            settled = pending.popleft()
            aborted = abort_at is not None and settled.start >= abort_at
            c.settle(settled, 0 if aborted else settled.end - settled.start)
    return exports


def generate(c, tokens, steps, *, index=0, abort_at=None):
    slot, cached = open_slot(c, index, tokens[: len(tokens) - steps])
    commit(c, reserve(c, slot, cached, len(tokens) - steps), tokens)
    exports = decode(c, slot, len(tokens) - steps, len(tokens) - 1, abort_at=abort_at)
    c.close_slot(slot, prefix_ids=tokens)
    return cached, exports


def test_decode_overwrites_one_leased_row_and_publishes_the_last_write_at_close():
    c = make_cache(3)
    history = prompt(300)
    cached, exports = generate(c, history, 260)
    # Prompt 40, then decode crossings at 64, 128, 192 and 256 rewrite one row.
    assert cached == 0 and [p for p, _ in exports] == [64, 128, 192, 256]
    assert len({row for _, row in exports}) == 1
    assert c.prefix_cache.rows.info == (2, 0, 1, 0, 1)
    # The next turn restores the generated history's deepest grid point.
    assert serve(c, torch.cat((history, prompt(40, 1000)))) == (256, [320])
    c.check_integrity()


@pytest.mark.parametrize("abort_at, published", [(None, 192), (192, 192), (191, 0), (150, 0)])
def test_an_aborted_last_decode_write_discards_the_private_row(abort_at, published):
    # Steps from ``abort_at`` on are released. When the last write to the row was one of
    # them, its bytes cannot be published, and the write before it is gone.
    c = make_cache(2)
    history = prompt(200)
    _, exports = generate(c, history, 140, abort_at=abort_at)
    assert [p for p, _ in exports] == [64, 128, 192]
    assert c.cached_prefix_len(history) == published
    assert c.prefix_cache.rows.info.leased == 0
    c.check_integrity()


def test_a_decode_export_without_a_row_is_skipped_and_the_step_runs():
    c = make_cache(1)
    slots = [open_slot(c, index, prompt(60, 1000 * index))[0] for index in range(2)]
    for slot in slots:
        commit(c, reserve(c, slot, 0, 60), prompt(60, 1000 * slot.index))
    # The first crossing leases the only row; the other request's step runs without one.
    assert [len(decode(c, slot, 60, 70)) for slot in slots] == [1, 0]
    for slot in slots:
        c.close_slot(slot, prefix_ids=prompt(71, 1000 * slot.index))
    assert c.prefix_cache.rows.info.cached == 1
    c.check_integrity()


def test_a_duplicate_decode_snapshot_keeps_the_published_row():
    c = make_cache(3)
    history = prompt(140)
    generate(c, history, 100)
    _, exports = generate(c, history, 100, index=1)
    assert [p for p, _ in exports] == [64, 128]
    assert c.prefix_cache.rows.info == (2, 0, 1, 0, 1)
    c.check_integrity()


def run_rounds(c, tokens, rounds):
    """Prefill ``tokens[:250]``, then settle DFlash rounds in order, each ``(device_count,
    count)`` from the frontier its predecessors left; return each claim's row and the
    position the published snapshot restores for ``tokens``."""
    slot, _ = open_slot(c, 0, tokens[:250])
    commit(c, reserve(c, slot, 0, 250), tokens)
    frontier, device_frontier, rows = 250, 250, []
    for device_count, count in rounds:
        (claim,), _ = c.reserve_rounds([slot], [frontier + 16])
        rows.append(claim.snapshot_row)
        c.settle(claim, count, record=(device_frontier, device_count))
        frontier += count
        device_frontier += device_count
    c.close_slot(slot, prefix_ids=tokens)
    return rows, c.cached_prefix_len(tokens)


@pytest.mark.parametrize(
    "rounds, restored",
    [
        ([(10, 10)], 256),  # 250 + 10 crosses 256
        ([(5, 5)], 192),  # 255 does not: the prefill's 192 stays the deepest
        ([(10, 10), (16, 16), (16, 16)], 256),  # 260..292 cross nothing
        ([(10, 10), (16, 16), (16, 16), (16, 16), (16, 16)], 320),  # 308 + 16 crosses 320
        # A discarded round's capture (at 320) overwrote the row: it publishes nothing.
        ([(10, 10), (16, 16), (16, 16), (16, 16), (16, 0)], 192),
        ([(10, 10), (16, 16), (16, 16), (16, 0)], 256),  # a discarded round that wrote nothing
    ],
)
def test_rounds_capture_the_grid_point_they_commit_and_publish_it_at_close(rounds, restored):
    c = make_cache(3)
    tokens = prompt(400)
    rows, cached = run_rounds(c, tokens, rounds)
    # Every round captures into the request's one leased row.
    assert len(set(rows)) == 1 and rows[0] is not None
    assert cached == restored
    c.check_integrity()


def test_rounds_of_a_cache_without_snapshot_rows_capture_nothing():

    c = new_cache_manager(
        2048, 1, torch.full((8, 512), -1, dtype=torch.int32), reuse=True, checks=True
    )
    slot, _ = open_slot(c, 0, prompt(250))
    commit(c, reserve(c, slot, 0, 250), prompt(250))
    (claim,), _ = c.reserve_rounds([slot], [266])
    assert claim.snapshot_row is None
    c.settle(claim, 10)


@pytest.mark.parametrize("frontier, position", [(250, 256), (255, 256), (256, 320), (319, 320)])
def test_a_round_captures_at_the_first_grid_point_past_its_frontier(frontier, position):
    # Host positions and device offsets (position - frontier, in [1, 64]) share the rule.
    assert next_snapshot_position(frontier, 64) == position
    frontiers = torch.tensor([frontier, 0])
    assert torch.equal(next_snapshot_position(frontiers, 64), torch.tensor([position, 64]))


def test_two_rounds_in_flight_share_the_row_and_a_discarded_write_voids_it():
    c = make_cache(3)
    tokens = prompt(400)
    slot, _ = open_slot(c, 0, tokens[:300])
    commit(c, reserve(c, slot, 0, 300), tokens)
    (first,), _ = c.reserve_rounds([slot], [316])
    (second,), _ = c.reserve_rounds([slot], [332])
    assert first.snapshot_row == second.snapshot_row is not None
    # The first round stops at 316 by EOS; its successor crossed 320 on the device.
    c.settle(first, 16, record=(300, 16))
    c.settle(second, 0, record=(316, 16))
    c.close_slot(slot, prefix_ids=tokens[:316])
    assert c.cached_prefix_len(tokens) == 256  # the prefill's
    c.check_integrity()


def test_a_round_submitted_beside_its_prompt_forward_is_voided_by_a_prompt_eos():
    c = make_cache(3)
    tokens = prompt(400)
    slot, _ = open_slot(c, 0, tokens[:250])
    reservation = reserve(c, slot, 0, 250)
    (claim,), _ = c.reserve_rounds([slot], [266])
    commit(c, reservation, tokens)
    # The prompt's first token stops it; the round committed 10 on the device.
    c.settle(claim, 0, record=(250, 10))
    c.close_slot(slot, prefix_ids=tokens[:250])
    assert c.cached_prefix_len(tokens) == 192
    c.check_integrity()


def test_a_round_without_a_row_captures_nothing_while_its_successor_leases_one():
    c = make_cache(1)
    other = prompt(70, 1000)
    held, _ = open_slot(c, 1, other[:60])
    commit(c, reserve(c, held, 0, 60), other)
    decode(c, held, 60, 64)  # its decode row takes the only row
    tokens = prompt(400)
    slot, _ = open_slot(c, 0, tokens[:250])
    commit(c, reserve(c, slot, 0, 250), tokens)  # no row: the prefill exports nothing
    (first,), _ = c.reserve_rounds([slot], [266])
    c.close_slot(held, prefix_ids=other)  # its row passes to the radix, unlocked
    (second,), _ = c.reserve_rounds([slot], [276])
    assert first.snapshot_row is None and second.snapshot_row is not None
    # Only the rowless round crossed 256; its successor's row holds no capture.
    c.settle(first, 10, record=(250, 10))
    c.settle(second, 5, record=(260, 5))
    c.close_slot(slot, prefix_ids=tokens[:265])
    assert c.cached_prefix_len(tokens) == 0
    c.check_integrity()
