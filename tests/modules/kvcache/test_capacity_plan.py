"""Snapshot budgets, native allocation rounding, state-kind allocations and capacity."""

import pytest
import torch

from cti.attention.paged import KVPair
from cti.config import MemoryOptions
from cti.distributed import Topology
from cti.kvcache import (
    CapacityError,
    GroupLayout,
    GroupRole,
    InferenceState,
    KVLayout,
    RecurrentRows,
    SnapshotPolicy,
    Spec,
    StateBudget,
    StateCapacity,
    TableGeometry,
    plan_capacity,
)
from cti.kvcache.state import PAYLOAD
from cti.layers.gated_delta_net import GatedDeltaNet
from cti.layers.gated_delta_net.layer import gated_delta_net_state
from cti.memory import native_backing
from tests.support.tp_shards import site

_RECURRENT_TO_KV_RATIO = MemoryOptions().recurrent_to_kv_ratio

_MIB = 1024**2
_SIZINGS = {"rows": SnapshotPolicy(rows=17), "ratio": SnapshotPolicy(ratio=0.1)}
_NO_ROWS = RecurrentRows(())


def _layout(*pools: tuple[int, int, int], rows: RecurrentRows = _NO_ROWS) -> KVLayout:
    """A BF16 KV group per ``(layers, KV heads, head size)`` pool, the target's first."""
    roles = (GroupRole.TARGET, GroupRole.DRAFT)
    return KVLayout(
        tuple(
            GroupLayout(roles[index], index, Spec(KVPair(kv, d, torch.bfloat16)), n)
            for index, (n, kv, d) in enumerate(pools)
        ),
        rows,
    )


def _conv_row() -> RecurrentRows:
    """One slot-state layer of six bytes a row."""
    return RecurrentRows((gated_delta_net_state(1, 1, 1, 1, conv_history=1),))


@pytest.mark.parametrize("available", [4_427_998_003, 52_173_512_704])
@pytest.mark.parametrize("sizing", ["rows", "ratio"])
def test_prefix_budget_fits_physical_qwen35_allocations(available, sizing):
    specs = (gated_delta_net_state(32, 128, 128, 8192),) * 24
    layout = _layout((8, 2, 256), rows=RecurrentRows(specs))
    plan = plan_capacity(
        layout,
        TableGeometry(1, 17, 1024),
        StateBudget(available, native_backing, _RECURRENT_TO_KV_RATIO),
        snapshots=_SIZINGS[sizing],
    )
    pages = plan.num_pages
    # Independently reconstruct physical backing, not just tensor payload. These
    # pools exceed the 10 MiB large threshold; the packed copy indices take one small one.
    segment = 2 * 1024**2
    physical = lambda n: (n + segment - 1) // segment * segment
    pool_bytes = sum(physical((17 + len(plan.snapshot_rows)) * s.bytes_per_slot) for s in specs)
    kv_bytes = 8 * physical((pages + 1) * layout.bytes_per_page(1) // 8)
    index_bytes = physical(4 * 17 * 8)
    assert pool_bytes + kv_bytes + index_bytes <= available
    assert pages >= 2
    # A tenth of the recurrent share, 0.9 / 1.9 of the budget.
    assert len(plan.snapshot_rows) == (17 if sizing == "rows" else (4 if available < 5e9 else 47))
    assert RecurrentRows(specs).row_bytes == 51_511_296


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (0, 0),
        (1, 2 * _MIB),
        (_MIB, 2 * _MIB),
        (_MIB + 1, 20 * _MIB),
        # Requests round up to 512-byte blocks before the allocator sizes their segment.
        (10 * _MIB - 512, 20 * _MIB),
        (10 * _MIB - 511, 10 * _MIB),
        (10 * _MIB - 1, 10 * _MIB),
        (10 * _MIB, 10 * _MIB),
        (10 * _MIB + 1, 12 * _MIB),
    ],
)
def test_native_cuda_segment_boundaries(payload, expected):
    assert native_backing.segment(payload) == expected


def test_recurrent_backing_counts_rows_and_index_slab():
    # One small row allocation (2 MiB) and, with snapshots, one packed-index slab (2 MiB);
    # row copies take no workspace.
    rows = RecurrentRows((gated_delta_net_state(1, 1, 1, 1, conv_history=1),))
    assert rows.allocation(1, 2, native_backing) == (3 * 6, 4 * _MIB)
    assert rows.allocation(1, 0, native_backing) == (6, 2 * _MIB)
    # By payload, the rows and 8 bytes of packed indices for each of 4 columns.
    assert rows.allocation(1, 2, PAYLOAD) == (3 * 6, 3 * 6 + 4 * 8)


@pytest.mark.parametrize("kv_budget", [2, 8, 10, 12, 20, 22])
def test_physical_page_fit_handles_small_medium_and_large_allocations(kv_budget):
    pages = _layout((1, 1, 1)).fit(kv_budget * _MIB, 1, native_backing)
    assert native_backing.segment(pages * 4) <= kv_budget * _MIB
    assert native_backing.segment((pages + 1) * 4) > kv_budget * _MIB


@pytest.mark.parametrize("snapshots", [SnapshotPolicy(rows=2), None])
def test_physical_overflow_rejects_a_payload_only_fit(snapshots):
    layout = _layout((1, 1, 1), rows=_conv_row())
    # One request row's 2 MiB allocation leaves no room for a KV pool's.
    budget = 2 * _MIB + 1000
    geometry = TableGeometry(1, 1, 16)
    payload = plan_capacity(
        layout, geometry, StateBudget(budget, None, _RECURRENT_TO_KV_RATIO), snapshots=snapshots
    )
    assert payload.num_pages >= 2
    # The rows' backing alone takes the budget once snapshots add the copy-index slab.
    message = "Recurrent state does not fit" if snapshots else "Not enough memory for KV cache"
    with pytest.raises(ValueError, match=message):
        plan_capacity(
            layout,
            geometry,
            StateBudget(budget, native_backing, _RECURRENT_TO_KV_RATIO),
            snapshots=snapshots,
        )


@pytest.mark.parametrize("guard_tokens", [0, 3])
@pytest.mark.parametrize("snapshots", [SnapshotPolicy(rows=2), None])
def test_payload_plans_fit_the_tail_pages_with_and_without_snapshot_rows(snapshots, guard_tokens):
    layout = _layout((1, 1, 1), rows=_conv_row())
    geometry = TableGeometry(1, 4, 64, guard_tokens=guard_tokens)
    budget = 1000
    plan = plan_capacity(
        layout, geometry, StateBudget(budget, None, _RECURRENT_TO_KV_RATIO), snapshots=snapshots
    )
    rows = layout.rows.allocation(4, len(plan.snapshot_rows), PAYLOAD)
    # The pools also hold the sink page and the guard pages past the usable ones.
    tail_pages = plan.pool_pages - plan.num_pages

    def payload(pages):
        kv = layout.allocation(pages + tail_pages, 1, native_backing).payload_bytes
        return rows.backing_bytes + kv

    assert payload(plan.num_pages) <= budget < payload(plan.num_pages + 1)


def test_a_page_fit_counts_requests_that_round_up_to_their_own_segment():
    # 800-byte pages: 13107 pages are a 10 MiB - 160 byte request, which rounds up to a
    # 10 MiB block and takes a segment of its own.
    layout = _layout((1, 1, 200))
    assert layout.fit(10 * _MIB, 1, native_backing) == 13107


@pytest.mark.parametrize("budget_mib", [0, 1, 12, 70, 90, 100, 150])
def test_kv_layout_fit_is_the_most_pages_whose_backing_fits(budget_mib):
    # The two pools' layer tensors reach the large allocation size at 640 and 2560 pages.
    layout = _layout((2, 2, 128), (3, 1, 64))
    budget = budget_mib * _MIB + 12_345
    candidates = range(budget // layout.bytes_per_page(16) + 1)
    best = max(
        n for n in candidates if layout.allocation(n, 16, native_backing).backing_bytes <= budget
    )
    assert layout.fit(budget, 16, native_backing) == best
    assert layout.fit(-1, 16, native_backing) == -1


def _kv_backing(layout, page_size, pages):
    """One native allocation per layer of every group."""
    return sum(
        group.num_layers * native_backing.segment(pages * group.spec.page_bytes(page_size))
        for group in layout.groups
    )


def _qwen_state(model, page_size, dflash, tp):
    """The runner's KV layout of one rank's Qwen3-8B or Qwen3.5-9B layers, with the z-lab
    DFlash draft's: (layers, KV heads, head size) per group."""
    if model == "qwen3":
        pools, num_recurrent = ((36, 8, 128), (5, 8, 128)), 0
    else:
        pools, num_recurrent = ((8, 4, 256), (6, 8, 128)), 24
    layer_site = site(Topology(0, tp))
    recurrent = tuple(
        GatedDeltaNet(
            num_key_heads=16,
            num_value_heads=32,
            key_head_dim=128,
            value_head_dim=128,
            conv_kernel_size=4,
            site=layer_site,
        )
        for _ in range(num_recurrent)
    )
    return _layout(
        *((n, kv // tp, d) for n, kv, d in pools[: 1 + dflash]),
        rows=RecurrentRows(tuple(layer.state_spec for layer in recurrent)),
    )


# (model, page size, DFlash, TP): the runner's pages naive and radix, each on payload then
# on CUDA; snapshot rows on payload then on CUDA, a quarter of the recurrent share. Every plan leaves the graph sink page; CUDA pages fit every pool
# at its native allocation size, and radix pages with snapshot rows also pay for DFlash's
# snapshot metadata. Naive CUDA pages, and Qwen3's radix ones, are 0.4-0.8% below payload:
# each layer's KV pool rounds to 2 MiB and each GDN layer's request rows to their
# allocation (Qwen3.5 TP1: 786 MiB of rows take 816 MiB).
_GOLDEN = {
    ("qwen3", 1, False, 1): (66978, 66559, 66978, 66559, 0, 0),
    ("qwen3", 1, False, 2): (133958, 133119, 133958, 133119, 0, 0),
    ("qwen3", 1, True, 1): (58810, 58367, 58810, 58367, 0, 0),
    ("qwen3", 1, True, 2): (117621, 116735, 117621, 116735, 0, 0),
    ("qwen3", 16, False, 1): (4185, 4159, 4185, 4159, 0, 0),
    ("qwen3", 16, False, 2): (8371, 8319, 8371, 8319, 0, 0),
    ("qwen3", 16, True, 1): (3674, 3647, 3674, 3647, 0, 0),
    ("qwen3", 16, True, 2): (7350, 7295, 7350, 7295, 0, 0),
    ("qwen3_5", 1, False, 1): (276255, 274943, 241671, 241151, 22, 22),
    ("qwen3_5", 1, False, 2): (577663, 574463, 506923, 503807, 45, 45),
    ("qwen3_5", 1, True, 1): (157859, 157183, 138097, 137727, 22, 22),
    ("qwen3_5", 1, True, 2): (330092, 327679, 289668, 287743, 45, 45),
    ("qwen3_5", 16, False, 1): (17265, 17183, 15103, 15071, 22, 22),
    ("qwen3_5", 16, False, 2): (36103, 35903, 31681, 31487, 45, 45),
    ("qwen3_5", 16, True, 1): (9865, 9823, 8630, 8607, 22, 22),
    ("qwen3_5", 16, True, 2): (20629, 20479, 18103, 17983, 45, 45),
}


@pytest.mark.parametrize(("model", "page_size", "dflash", "tp"), list(_GOLDEN))
def test_capacity_golden_for_every_kv_layout(model, page_size, dflash, tp):
    *golden_pages, payload_rows, rows = _GOLDEN[model, page_size, dflash, tp]
    layout = _qwen_state(model, page_size, dflash, tp)
    recurrent = layout.rows
    available = 9_876_543_210

    def metadata(snapshot_rows):
        # DFlash on Qwen3.5: int64 [rows, 4] per recurrent layer, each 256-byte aligned.
        return 24 * -(-32 * snapshot_rows // 256) * 256 if dflash and recurrent.specs else 0

    def plan(radix, physical, snapshot_rows=None, budget=available):
        # What the runner plans when the model and the attention workspaces hold no memory;
        # DFlash's snapshot metadata does.
        plan = plan_capacity(
            layout,
            TableGeometry(page_size, 16, 1 << 20),
            StateBudget(budget, native_backing if physical else None, _RECURRENT_TO_KV_RATIO),
            snapshots=SnapshotPolicy(snapshot_rows) if radix else None,
            allocations=(lambda _, rows: (metadata(rows),)) if dflash else None,
        )
        return plan.num_pages, len(plan.snapshot_rows)

    planned = [plan(radix, physical) for radix in (False, True) for physical in (False, True)]
    assert planned == list(zip(golden_pages, (0, 0, payload_rows, rows), strict=True))
    for radix, snapshot_rows in ((False, 0), *((True, n) for n in ((rows, 2) if rows else (0,)))):
        # Request and snapshot rows, the copy-index slab and metadata, then usable pages
        # plus the sink page fit; one more page does not, nor do these pages one byte short.
        state = sum(
            native_backing.segment((16 + snapshot_rows) * spec.bytes_per_slot)
            for spec in recurrent.specs
        )
        if snapshot_rows:
            state += native_backing.segment(4 * 16 * 8) + native_backing.segment(
                metadata(snapshot_rows)
            )
        requested = snapshot_rows or None
        num_pages, _ = plan(radix, True, requested)
        assert _kv_backing(layout, page_size, num_pages + 1) + state <= available
        assert _kv_backing(layout, page_size, num_pages + 2) + state > available
        short = _kv_backing(layout, page_size, num_pages + 1) + state - 1
        assert plan(radix, True, requested, short)[0] < num_pages


def _slot_rows():
    """41 slot-state layers whose request rows each hold an FP32 256 x 256 state and one BF16
    history element."""
    return RecurrentRows((gated_delta_net_state(1, 256, 256, 1, conv_history=1),) * 41)


def test_slot_rows_are_planned_by_backing_without_prefix_reuse():
    # Four request rows make each layer one allocation just over 1 MiB, which the allocator
    # backs with 20 MiB. Beside them, 43 paged layers.
    layout = _layout((43, 1, 512), rows=_slot_rows())
    geometry = TableGeometry(256, 4, 1 << 20)
    rows = layout.rows.allocation(4, 0, native_backing)
    assert (rows.payload_bytes, rows.backing_bytes) == (41 * 4 * (256 * 256 * 4 + 2), 820 * _MIB)
    available = 9_876_543_210

    def plan(budget, physical):
        backing = native_backing if physical else None
        return plan_capacity(
            layout, geometry, StateBudget(budget, backing, _RECURRENT_TO_KV_RATIO), snapshots=None
        )

    def backing(num_pages):
        # The request rows, then the usable pages and the graph sink page.
        return rows.backing_bytes + layout.allocation(num_pages + 1, 256, native_backing)[1]

    num_pages = plan(available, True).num_pages
    assert backing(num_pages) <= available < backing(num_pages + 1)
    # Sized by payload, the pools would take 775 MiB past the budget.
    assert backing(plan(available, False).num_pages) - available > 774 * _MIB
    with pytest.raises(ValueError, match="Recurrent state does not fit"):
        plan(500 * _MIB, True)


def test_snapshot_rows_are_planned_in_the_allocation_they_share_with_slot_rows():
    # 36 request rows take 9 MiB a layer, backed by 20 MiB; with 4 snapshot rows each layer's
    # one allocation passes 10 MiB and takes 12 MiB, so the state fits a 600 MiB budget.
    layout = _layout((1, 1, 128), rows=_slot_rows())
    budget = 600 * _MIB
    assert layout.rows.allocation(36, 0, native_backing).backing_bytes == 820 * _MIB
    state = layout.rows.allocation(36, 4, native_backing)
    plan = plan_capacity(
        layout,
        TableGeometry(16, 36, 1 << 20),
        StateBudget(budget, native_backing, _RECURRENT_TO_KV_RATIO),
        snapshots=SnapshotPolicy(rows=4),
    )
    assert (plan.num_pages, len(plan.snapshot_rows)) == (13567, 4)

    def backing(num_pages):
        # The rows and the copy indices, then the usable pages and the sink page.
        return state.backing_bytes + layout.allocation(num_pages + 1, 16, native_backing)[1]

    assert backing(plan.num_pages) <= budget < backing(plan.num_pages + 1)


def test_capacity_publishes_usable_pages_and_snapshot_rows_past_the_slots():
    geometry = TableGeometry(16, 4, 1000, guard_tokens=48)
    layout = KVLayout((), RecurrentRows((gated_delta_net_state(1, 1, 1, 1),)))
    # 50 pages hold 800 tokens; past them, the graph sink page and 3 guard pages. Snapshots
    # lie on whole pages where Gated DeltaNet exports, every 64 tokens.
    assert geometry.capacity(layout, 50, 6) == StateCapacity(
        16, 4, 800, 64, 50, range(4, 10), 50, range(51, 54), 64, 64
    )
    assert geometry.capacity(layout, 70, 0).max_seq_len == 1000
    # Without slot-state layers, the grid is the page.
    attention_only = KVLayout((), _NO_ROWS)
    assert TableGeometry(48, 4, 1000).capacity(attention_only, 50, 0).snapshot_grid == 48
    assert TableGeometry(48, 4, 1000).capacity(layout, 50, 0).snapshot_grid == 192


def test_inference_state_allocates_a_group_per_kv_layout_group_and_the_recurrent_rows():
    rows = RecurrentRows((gated_delta_net_state(1, 1, 1, 1),))
    layout = _layout((2, 1, 4), (1, 2, 4), rows=rows)
    capacity = TableGeometry(1, 3, 64, guard_tokens=3).capacity(layout, 10, 2)
    cpu = torch.device("cpu")

    state = InferenceState.allocate(layout, capacity, device=cpu)

    assert state.capacity is capacity
    # Both groups share the page numbers: the usable pages, then the 4 tail pages.
    assert [
        (group.role, group.index, group.spec, len(group.layers), group.layers[0].shape)
        for group in state.groups
    ] == [
        (GroupRole.TARGET, 0, layout.groups[0].spec, 2, (2, 14, 1, 1, 4)),
        (GroupRole.DRAFT, 1, layout.groups[1].spec, 1, (2, 14, 1, 2, 4)),
    ]
    assert state.recurrent.num_slots == 3
    assert state.recurrent.snapshot_rows == capacity.snapshot_rows
    no_rows = KVLayout(layout.groups, _NO_ROWS)
    assert InferenceState.allocate(no_rows, capacity, device=cpu).recurrent is None


def test_kv_layout_bytes_are_the_mha_pools_k_and_v():
    # Per page: 2 (K, V) x page size x KV heads x head size x 2 bytes, per layer.
    layout = _layout((36, 8, 128), (5, 8, 128))
    assert layout.bytes_per_page(16) == 41 * 2 * 16 * 8 * 128 * 2


def test_page_geometry_cannot_round_back_into_the_medium_slab_gap():
    # Three KV heads make a 12 KiB layer page, which does not divide 10 MiB.
    # Rounding the capacity down to pages must not turn a 10 MiB allocation
    # into a <10 MiB allocation requiring a 20 MiB segment.
    layout = _layout((1, 3, 128))
    pages = layout.fit(10 * _MIB, 8, native_backing)
    assert native_backing.segment(pages * layout.bytes_per_page(8)) <= 10 * _MIB


@pytest.mark.parametrize(
    "rows,ratio", [(1, None), (True, None), (2, 0.2), (None, 0), (None, float("nan")), (None, 1)]
)
def test_invalid_budgets(rows, ratio):
    with pytest.raises(ValueError):
        SnapshotPolicy(rows, ratio)


def test_snapshot_rows_take_their_ratio_of_the_share():
    # Six bytes a row; row copies take no workspace, so the whole 1000-byte share counts.
    rows = RecurrentRows((gated_delta_net_state(1, 1, 1, 1, conv_history=1),))
    assert rows.snapshot_count(1000, SnapshotPolicy(ratio=0.3)) == 1000 * 0.3 // 6
    assert rows.snapshot_count(1000, SnapshotPolicy()) == 1000 // 4 // 6
    assert rows.snapshot_count(1000, SnapshotPolicy(rows=5)) == 5


def test_prefix_reuse_needs_two_snapshot_rows_in_the_share():
    # 10-byte rows: a quarter of a 150-byte budget's recurrent share (71 bytes) holds one.
    layout = _layout((1, 1, 1), rows=RecurrentRows((gated_delta_net_state(1, 1, 1, 1),)))
    with pytest.raises(CapacityError, match="prefix reuse needs at least 2") as raised:
        plan_capacity(
            layout,
            TableGeometry(1, 1, 16),
            StateBudget(150, None, _RECURRENT_TO_KV_RATIO),
            snapshots=SnapshotPolicy(),
        )
    assert not raised.value.beside


def test_capacity_refuses_token_indices_past_int32():
    layout = KVLayout((), _NO_ROWS)
    pages = torch.iinfo(torch.int32).max // 16 - 1
    TableGeometry(16, 1, 16).capacity(layout, pages, 0)
    with pytest.raises(ValueError, match="torch.int32"):
        TableGeometry(16, 1, 16, guard_tokens=16).capacity(layout, pages, 0)
