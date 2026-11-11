from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from cti.distributed import Topology
from cti.forward import InlineBreaks, RecurrentExport, RoundShape
from cti.kvcache import RecurrentStatePool, SlotStateSpec, StateSlot
from cti.layers import LayerServices
from cti.layers.gated_delta_net import GatedDeltaNet, prefill_exports_states
from cti.layers.gated_delta_net.layer import gated_delta_net_state
from cti.layers.gated_delta_net.operations import (
    GatedDeltaNetOps,
    GatedDeltaNetTrainingOperation,
    resolve_gated_delta_net_ops,
)
from cti.layers.placement import WeightShard
from cti.recurrent import (
    RecurrentBackend,
    RecurrentGeometry,
    RecurrentMetadata,
    RecurrentPhase,
    RecurrentRounds,
    RecurrentVerify,
)
from ctikernels.gdn import GatedDeltaNetTraining
from tests.support.forward_rows import decode_entry, extend_entry
from tests.support.gated_delta_net import (
    CONV_CHANNELS,
    HEAD_DIM,
    NUM_SLOTS,
    NUM_VALUE_HEADS,
    OUTPUT_WIDTH,
    RecordingOperation,
    ReferenceGatedDeltaNetDecode,
    ReferenceGatedDeltaNetPrefill,
    ReferenceGatedDeltaNetTraining,
    bind_reference,
    gdn_layer,
    layer_inputs,
    layer_rows,
    pool_snapshot,
    projected_rows,
    reference_kernel_module,
    run_layer,
)
from tests.support.tp_shards import default_dtype, site, windows


def _specs(layers: list[GatedDeltaNet]) -> tuple[SlotStateSpec, ...]:
    return tuple(layer.state_spec for layer in layers)


# ================================ Layer ================================


def test_layer_declares_checkpoint_shapes_frozen_dtypes_and_its_state() -> None:
    layer = GatedDeltaNet(
        num_key_heads=16,
        num_value_heads=32,
        key_head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        conv_kernel_size=4,
        site=site(),
    )

    state = layer.state_dict()
    assert {name: (tuple(value.shape), value.dtype) for name, value in state.items()} == {
        "conv_weight": ((8192, 1, 4), torch.bfloat16),
        "A_log": ((32,), torch.float32),
        "dt_bias": ((32,), torch.float32),
        "norm_weight": ((128,), torch.float32),
    }
    assert (layer.num_key_heads, layer.num_value_heads) == (16, 32)
    # The pool row: FP32 [32, 128, 128] state and 3 BF16 rows of 8192 channels.
    assert layer.state_slots() == (StateSlot("recurrent", state=layer.state_spec),)
    assert layer.state_spec.tensors == (
        ("state", (32, 128, 128), torch.float32),
        ("conv", (3, 8192), torch.bfloat16),
    )


def test_layer_hands_the_operations_a_two_dimensional_view_of_the_checkpoint_conv() -> None:
    layer = gdn_layer()

    weights = layer.weights

    assert weights.conv_weight.shape == (CONV_CHANNELS, 4)
    assert weights.conv_weight.data_ptr() == layer.conv_weight.data_ptr()
    assert torch.equal(weights.conv_weight, layer.conv_weight[:, 0, :])
    assert weights.A_log is layer.A_log
    assert weights.dt_bias is layer.dt_bias
    assert weights.norm_weight is layer.norm_weight


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"key_head_dim": 64}, "head dimensions must be 128", id="key-head-dim"),
        pytest.param({"value_head_dim": 256}, "head dimensions must be 128", id="value-head-dim"),
        pytest.param({"conv_kernel_size": 3}, "kernel size must be 4", id="conv-taps"),
        pytest.param({"num_value_heads": 3}, "multiple of num_key_heads", id="head-groups"),
    ],
)
def test_layer_rejects_geometry_outside_the_operation_contract(
    overrides: dict[str, int],
    message: str,
) -> None:
    arguments = {
        "num_key_heads": 2,
        "num_value_heads": 4,
        "key_head_dim": HEAD_DIM,
        "value_head_dim": HEAD_DIM,
        "conv_kernel_size": 4,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        GatedDeltaNet(**arguments, site=site())


def _sharded_layer(rank: int, size: int, *, num_key_heads: int = 4) -> GatedDeltaNet:
    return GatedDeltaNet(
        num_key_heads=num_key_heads,
        num_value_heads=2 * num_key_heads,
        key_head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        conv_kernel_size=4,
        site=site(Topology(rank, size)),
    )


def test_tensor_parallel_layer_holds_and_declares_its_ranks_head_groups() -> None:
    layer = _sharded_layer(1, 2)

    # Rank 1 of 2 owns key heads 2-3 and the value heads 4-7 that read them.
    assert (layer.num_key_heads, layer.num_value_heads) == (2, 4)
    assert _specs([layer]) == (
        gated_delta_net_state(4, HEAD_DIM, HEAD_DIM, (2 * 2 + 4) * HEAD_DIM),
    )
    key = WeightShard(0, 2 * HEAD_DIM, 4 * HEAD_DIM)
    value = WeightShard(0, 4 * HEAD_DIM, 8 * HEAD_DIM)
    heads = WeightShard(0, 4, 8)
    # One window per q | k | v part of the convolution; the gated norm is replicated.
    assert windows(layer) == {
        "conv_weight": (key, key, value),
        "A_log": (heads,),
        "dt_bias": (heads,),
    }
    assert windows(gdn_layer()) == {}


def test_tensor_parallel_layer_rejects_a_size_that_splits_key_heads() -> None:
    with pytest.raises(
        ValueError, match=r"key heads \(2\) must be divisible by the tensor-parallel size \(4\)"
    ):
        _sharded_layer(0, 4, num_key_heads=2)


def test_a_verified_block_goes_to_the_layers_replay_and_leaves_the_pool_unchanged() -> None:
    layer = gdn_layer()
    bound = bind_reference(layer)
    before = pool_snapshot(bound.pool)
    projected = projected_rows(2 * 16)
    calls = []

    class _Replay:
        def verify(self, projected, v):
            calls.append((projected, v))
            return projected[:, :OUTPUT_WIDTH]

    layer._replay = _Replay()
    slot_ids = torch.tensor([3, 0], dtype=torch.int32)
    verify = RecurrentVerify(1, slot_ids, torch.zeros(2, 16))
    metadata = RecurrentMetadata(
        slot_ids=slot_ids,
        num_extend_requests=0,
        num_extend_tokens=0,
        phase=RecurrentPhase.VERIFY,
        verify=verify,
    )
    y = layer.forward(
        projected, SimpleNamespace(recurrent_metadata=metadata, graph_breaks=InlineBreaks())
    )

    assert torch.equal(y, projected[:, :OUTPUT_WIDTH])
    ((projected_seen, verify_seen),) = calls
    assert verify_seen is verify and projected_seen is projected
    assert not bound.decode.calls and not bound.prefill.calls
    for after, expected in zip(pool_snapshot(bound.pool), before, strict=True):
        assert torch.equal(after, expected)


def test_a_reset_forgets_the_decode_scratch_of_the_bound_pool_row() -> None:
    layer = gdn_layer()
    bound = bind_reference(layer)

    layer.reset_scratch()

    # The decode matches scratch by the pool tensor it reads: the same storage and shape.
    (pool,) = bound.decode.resets
    state, _ = bound.pool.layer_views(0)
    assert pool.data_ptr() == state.data_ptr() and pool.shape == state.shape


# K2 reads 16-byte aligned projected rows: an even number of value-head pairs.
_K2_HEADS = (1, 4)
_K2_WIDTH = 2 * 128 + 8 * 128 + 8


def _k2_rounds(monkeypatch: pytest.MonkeyPatch, block_size: int = 16, num_snapshot_rows: int = 0):
    """A layer bound to reference operations and its pool, its replay in ``RecurrentRounds``
    over that pool on the K2 CPU reference; CUDA streams are no-ops."""

    pytest.importorskip("ctikernels.gdn.replay", reason="needs the K2 replay companion")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    layer = gdn_layer(heads=_K2_HEADS)
    bound = bind_reference(layer, num_snapshot_rows)
    shape = RoundShape(block_size, 2, 2)
    pool = bound.pool
    geometry = RecurrentGeometry(pool.num_slots, len(pool.snapshot_rows), pool.device)
    replay = layer.create_replay(max_batch=2, rounds=shape, geometry=geometry)
    rounds = RecurrentRounds([replay], shape=shape)
    return layer, bound, rounds


def _begin(rounds, rows: list[int], frontiers: torch.Tensor) -> int:
    """Open a round that hands off ``rows``."""

    rounds.mark_stale(rows)
    return rounds.begin(rows, frontiers)


def _settle(status) -> None:
    for device, host in status.copies:
        host.copy_(device)
    status.settle()


def _block_rows(
    rounds, status_slot: int, slots: list[int], frontiers: torch.Tensor, block_size: int = 16
):
    slot_ids = torch.tensor(slots, dtype=torch.int32)
    positions = frontiers[slot_ids.long().clamp(min=0), None].long() + torch.arange(block_size)
    return rounds.prepare_verify(status_slot, slot_ids, positions)


@pytest.mark.parametrize("num_snapshot_rows", [0, 3])
def test_rounds_verify_like_decode_and_commit_the_accepted_prefix(
    monkeypatch: pytest.MonkeyPatch, num_snapshot_rows: int
) -> None:
    layer, bound, rounds = _k2_rounds(monkeypatch, num_snapshot_rows=num_snapshot_rows)
    pool = bound.pool
    slot_ids = torch.tensor([3, 1], dtype=torch.int32)
    frontier_sets = (torch.tensor([0, 9, 0, 5, 0]), torch.tensor([0, 2, 0, 7, 0]))
    # The second round reuses both slots for requests at other frontiers.
    for first, frontiers in enumerate(frontier_sets):
        # Garbage rows 3 and 1 are their slots' committed state at the frontiers; with
        # snapshot rows, a restore first copies two of them (absolute pool rows) there.
        if num_snapshot_rows:
            restored = pool.snapshot_rows[first : first + 2]
            pool.copy_rows(torch.tensor([list(restored), slot_ids.tolist()]))
        projected = projected_rows(2 * 16, width=_K2_WIDTH, seed=int(frontiers[1])).view(2, 16, -1)
        status_slot = _begin(rounds, [3, 1], frontiers)
        before = pool_snapshot(pool)
        rows = _block_rows(rounds, status_slot, [3, 1], frontiers)
        y = layer.forward(
            projected.flatten(0, 1),
            SimpleNamespace(recurrent_metadata=rows, graph_breaks=InlineBreaks()),
        )

        for after, expected in zip(pool_snapshot(pool), before, strict=True):
            assert torch.equal(after, expected)
        state, conv_state = (view.clone() for view in before)
        accepted = torch.tensor([4, 15], dtype=torch.int32)
        committed = []
        for step in range(16):
            decoded = bound.decode(projected[:, step], *layer.weights, slot_ids, state, conv_state)
            torch.testing.assert_close(y.view(2, 16, -1)[:, step], decoded, rtol=2e-2, atol=2e-2)
            committed.append((state.clone(), conv_state.clone()))
        rounds.commit(status_slot, accepted + 1)

        state, conv_state = pool.layer_views(0)
        for row, (slot, count) in enumerate(zip(slot_ids.tolist(), accepted.tolist(), strict=True)):
            # Commit keeps the anchor and ``count`` drafts: the state after count + 1 tokens.
            expected_state, expected_history = committed[count]
            torch.testing.assert_close(state[slot], expected_state[slot], rtol=1e-4, atol=1e-4)
            assert torch.equal(conv_state[slot], expected_history[slot]), row
        for after, expected in zip(pool.layer_views(0), before, strict=True):
            assert torch.equal(after[NUM_SLOTS:], expected[NUM_SLOTS:])
        _settle(rounds.end(status_slot, 2))


def test_a_commit_captures_the_prefix_it_commits_into_a_snapshot_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer, bound, rounds = _k2_rounds(monkeypatch, num_snapshot_rows=2)
    slot_ids = torch.tensor([3, 1], dtype=torch.int32)
    frontiers = torch.tensor([0, 9, 0, 5, 0])
    projected = projected_rows(2 * 16, width=_K2_WIDTH).view(2, 16, -1)
    status_slot = _begin(rounds, [3, 1], frontiers)
    before = pool_snapshot(bound.pool)
    rows = _block_rows(rounds, status_slot, [3, 1], frontiers)
    layer.forward(
        projected.flatten(0, 1),
        SimpleNamespace(recurrent_metadata=rows, graph_breaks=InlineBreaks()),
    )
    state, conv_state = (view.clone() for view in before)
    for step in range(3):  # the state after three block tokens
        bound.decode(projected[:, step], *layer.weights, slot_ids, state, conv_state)

    # Slot 3 commits 5 tokens and captures after 3 into row 4; slot 1 commits 4, short of
    # its capture after 6 into row 5.
    rounds.commit(
        status_slot,
        torch.tensor([5, 4], dtype=torch.int32),
        torch.tensor([4, 5], dtype=torch.int32),
        torch.tensor([3, 6]),
    )
    _settle(rounds.end(status_slot, 2))

    after_state, after_conv = bound.pool.layer_views(0)
    torch.testing.assert_close(after_state[4], state[3], rtol=1e-4, atol=1e-4)
    assert torch.equal(after_conv[4], conv_state[3])
    assert torch.equal(after_state[5], before[0][5]) and torch.equal(after_conv[5], before[1][5])


@pytest.mark.parametrize("cached", [0, 4])
def test_a_prompt_beside_verified_blocks_matches_each_run_alone(
    monkeypatch: pytest.MonkeyPatch, cached: int
) -> None:
    frontiers = torch.tensor([0, 9, 0, 5, 0])
    prompt = projected_rows(9, width=_K2_WIDTH, seed=6)
    blocks_projected = projected_rows(2 * 16, width=_K2_WIDTH, seed=7)

    def run(mixed: bool):
        layer, bound, rounds = _k2_rounds(monkeypatch)
        if cached:
            run_layer(layer, bound.backend, prompt[:cached], extend_entry(2, 0, cached))
        status_slot = _begin(rounds, [3, 1], frontiers)
        rows = _block_rows(rounds, status_slot, [3, 1], frontiers)
        extension = layer_rows(extend_entry(2, cached, 9))
        metadata = bound.backend.prepare_metadata(extension)
        if mixed:
            joint = replace(metadata, phase=RecurrentPhase.VERIFY, verify=rows.verify)
            projected = torch.cat((prompt[cached:], blocks_projected))
            y = layer.forward(
                projected, SimpleNamespace(recurrent_metadata=joint, graph_breaks=InlineBreaks())
            )
        else:
            y = torch.cat(
                (
                    layer.forward(prompt[cached:], layer_inputs(extension, metadata)),
                    layer.forward(
                        blocks_projected,
                        SimpleNamespace(recurrent_metadata=rows, graph_breaks=InlineBreaks()),
                    ),
                )
            )
        rounds.commit(status_slot, torch.tensor([3, 16], dtype=torch.int32))
        _settle(rounds.end(status_slot, 2))
        return y, pool_snapshot(bound.pool), len(bound.prefill.calls)

    mixed, mixed_pool, prefills = run(mixed=True)
    alone, alone_pool, _ = run(mixed=False)

    assert torch.equal(mixed, alone)
    for after, expected in zip(mixed_pool, alone_pool, strict=True):
        assert torch.equal(after, expected)
    # One prefill call over the prompt rows (after a cached chunk's own).
    assert prefills == 1 + bool(cached)


def _recorded_graphs(monkeypatch: pytest.MonkeyPatch):
    """A CPU stand-in for CUDA graphs over the K2 reference: a capture records the replay
    op's fused handoff calls without running them; a replay runs them again."""

    from ctikernels.gdn.replay import GatedDeltaNetReplay

    capturing: list = []

    class Graph:
        def __init__(self) -> None:
            self.calls: list[Callable[[], None]] = []
            self.replays = 0

        def replay(self) -> None:
            self.replays += 1
            for call in self.calls:
                call()

    class Pool:
        graphs: list[Graph] = []

        @contextmanager
        def workspace(self, max_rows: int) -> Iterator[None]:
            yield

        @contextmanager
        def capture(self, graph: Graph, stream: object) -> Iterator[None]:
            self.graphs.append(graph)
            capturing.append(graph)
            try:
                yield
            finally:
                capturing.pop()

    def recorded(run: Callable[..., None]) -> Callable[..., None]:
        def call(op, *args, **kwargs) -> None:
            if capturing:
                capturing[-1].calls.append(lambda: run(op, *args, **kwargs))
            else:
                run(op, *args, **kwargs)

        return call

    monkeypatch.setattr(
        GatedDeltaNetReplay, "handoff_layers", recorded(GatedDeltaNetReplay.handoff_layers)
    )
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    return Pool()


def test_a_captured_handoff_matches_the_eager_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A handoff that replays its status slot's captured graph verifies and commits like an
    eager one."""

    frontiers = torch.tensor([0, 9, 0, 5, 0])
    blocks_projected = projected_rows(2 * 16, width=_K2_WIDTH, seed=7)

    def run(captured: bool):
        layer, bound, rounds = _k2_rounds(monkeypatch)
        pool = _recorded_graphs(monkeypatch)
        if captured:
            rounds.capture_handoffs(pool, None)
        status_slot = _begin(rounds, [3, 1], frontiers)
        rows = _block_rows(rounds, status_slot, [3, 1], frontiers)
        batch = SimpleNamespace(recurrent_metadata=rows, graph_breaks=InlineBreaks())
        y = layer.forward(blocks_projected, batch)
        rounds.commit(status_slot, torch.tensor([3, 16], dtype=torch.int32))
        _settle(rounds.end(status_slot, 2))
        if captured:
            assert [graph.replays for graph in pool.graphs] == [1, 0]
        return y, pool_snapshot(bound.pool)

    captured, captured_pool = run(captured=True)
    eager, eager_pool = run(captured=False)

    assert torch.equal(captured, eager)
    for after, expected in zip(captured_pool, eager_pool, strict=True):
        assert torch.equal(after, expected)


def test_k2_keeps_its_device_state_in_the_one_allocation_planning_counts() -> None:
    pytest.importorskip("ctikernels.gdn.replay", reason="needs the K2 replay companion")
    from cti.layers.gated_delta_net.replay import GatedDeltaNetK2

    layers = [gdn_layer(heads=_K2_HEADS) for _ in range(2)]
    pool = RecurrentStatePool(
        _specs(layers), num_slots=NUM_SLOTS, num_snapshot_rows=9, device=torch.device("cpu")
    )
    k2 = GatedDeltaNetK2(
        RecurrentGeometry(NUM_SLOTS, 9, torch.device("cpu")), shape=RoundShape(16, 2, 2)
    )
    for index, layer in enumerate(layers):
        k2.join(layer, pool.layer_views(index))
    k2._allocate()

    first, second = (state.snapshot_metadata for state in k2._states)
    assert first.shape == second.shape == (9, 4)
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    assert first.untyped_storage().nbytes() == GatedDeltaNetK2.device_bytes(
        layers, num_slots=NUM_SLOTS, shape=RoundShape(16, 2, 2), num_snapshot_rows=9
    )


def test_an_inactive_row_reads_and_commits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    layer, bound, rounds = _k2_rounds(monkeypatch)
    frontiers = torch.tensor([0, 9, 0, 5, 0])
    status_slot = _begin(rounds, [3], frontiers)
    before = pool_snapshot(bound.pool)
    # Row 1 is a done request's: slot -1.
    rows = _block_rows(rounds, status_slot, [3, -1], frontiers)
    y = layer.forward(
        projected_rows(2 * 16, width=_K2_WIDTH),
        SimpleNamespace(recurrent_metadata=rows, graph_breaks=InlineBreaks()),
    ).view(2, 16, -1)
    rounds.commit(status_slot, torch.tensor([2, 7], dtype=torch.int32))
    _settle(rounds.end(status_slot, 2))

    assert torch.count_nonzero(y[1]) == 0
    for after, expected in zip(pool_snapshot(bound.pool), before, strict=True):
        torch.testing.assert_close(after[1], expected[1])
        assert not torch.equal(after[3], expected[3])


def test_rounds_settle_in_order_and_refuse_a_failed_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer, _, rounds = _k2_rounds(monkeypatch, block_size=2)
    frontiers = torch.tensor([3, 4, 5, 6, 0], dtype=torch.int32)
    settlements = []
    # Two unsettled rounds on disjoint rows use separate status slots.
    for slots in ([0, 1], [2, 3]):
        status_slot = _begin(rounds, slots, frontiers)
        rows = _block_rows(rounds, status_slot, slots, frontiers, block_size=2)
        layer.forward(
            projected_rows(4, width=_K2_WIDTH),
            SimpleNamespace(recurrent_metadata=rows, graph_breaks=InlineBreaks()),
        )
        rounds.commit(status_slot, torch.tensor([1, 2], dtype=torch.int32))
        settlements.append(rounds.end(status_slot, 2))
    with pytest.raises(RuntimeError, match="not settled"):
        rounds.begin([], frontiers)
    for status in settlements:
        _settle(status)

    # Slot 0 committed one token past frontier 3: verifying from 3 again fails.
    status_slot = rounds.begin([0], frontiers)
    assert status_slot == 0
    rows = _block_rows(rounds, status_slot, [0], frontiers, block_size=2)
    layer.forward(
        projected_rows(2, width=_K2_WIDTH),
        SimpleNamespace(recurrent_metadata=rows, graph_breaks=InlineBreaks()),
    )
    with pytest.raises(RuntimeError, match="failed"):
        _settle(rounds.end(status_slot, 1))


# ======================= Reference operations against HF =======================


def _hf_gated_delta_net(layer: GatedDeltaNet) -> tuple[torch.nn.Module, Qwen3_5TextConfig]:
    """Transformers' layer with identity input projections and no output projection,
    so it consumes exactly the rows the engine passes and returns ``y``."""

    num_value_heads = layer.num_value_heads
    conv_channels = layer.conv_weight.shape[0]
    value_end = conv_channels + num_value_heads * HEAD_DIM
    projected_width = value_end + 2 * num_value_heads
    config = Qwen3_5TextConfig(
        hidden_size=projected_width,
        linear_num_key_heads=layer.num_key_heads,
        linear_num_value_heads=num_value_heads,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=4,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        vocab_size=8,
    )
    module = hf_qwen3_5.Qwen3_5GatedDeltaNet(config, 0).eval()
    identity = torch.eye(projected_width)
    with torch.no_grad():
        module.in_proj_qkv.weight.copy_(identity[:conv_channels])
        module.in_proj_z.weight.copy_(identity[conv_channels:value_end])
        module.in_proj_b.weight.copy_(identity[value_end : value_end + num_value_heads])
        module.in_proj_a.weight.copy_(identity[value_end + num_value_heads :])
        module.conv1d.weight.copy_(layer.conv_weight.float())
        module.A_log.copy_(layer.A_log)
        module.dt_bias.copy_(layer.dt_bias)
        module.norm.weight.copy_(layer.norm_weight)
    module.out_proj = torch.nn.Identity()
    return module, config


# (key heads, value heads): Qwen3.5-9B shares each key head between two value heads and
# 27B between three. With two key heads, value head j reading key head j // ratio
# (Transformers' repeat_interleave) differs from reading key head j % 2.
@pytest.mark.parametrize("heads", [(1, 2), (2, 4), (2, 6)], ids=["1x2", "2x4", "2x6"])
def test_reference_operations_match_transformers_chunked_and_recurrent_paths(
    heads: tuple[int, int],
) -> None:
    layer = gdn_layer(heads=heads)
    bound = bind_reference(layer)
    backend = bound.backend
    hf_module, hf_config = _hf_gated_delta_net(layer)
    projected = projected_rows(70, width=hf_config.hidden_size)
    hidden = projected.float()[None]

    with torch.no_grad():
        hf_one_shot = hf_module(hidden)[0]
        cache = DynamicCache(config=hf_config)
        hf_parts = [hf_module(hidden[:, :66], cache_params=cache)[0]]
        hf_parts.extend(
            hf_module(hidden[:, token : token + 1], cache_params=cache)[0]
            for token in range(66, 70)
        )
    engine_one_shot = run_layer(layer, backend, projected, extend_entry(1, 0, 70))
    engine_parts = [run_layer(layer, backend, projected[:66], extend_entry(2, 0, 66))]
    engine_parts.extend(
        run_layer(layer, backend, projected[token : token + 1], decode_entry(2, token))
        for token in range(66, 70)
    )

    # The engine rounds FP32 y to BF16 once: within one BF16 unit of Transformers' FP32.
    assert engine_one_shot.dtype == torch.bfloat16
    torch.testing.assert_close(engine_one_shot.float(), hf_one_shot, rtol=2**-7, atol=1e-5)
    torch.testing.assert_close(
        torch.cat(engine_parts).float(), torch.cat(hf_parts), rtol=2**-7, atol=1e-5
    )
    # The pool layout of ctikernels.gdn: value-major FP32 state, and the last three
    # pre-convolution rows oldest first (Transformers keeps the last four inputs).
    state, conv_state = bound.pool.layer_views(0)
    hf_layer_cache = cache.layers[0]
    torch.testing.assert_close(
        state[2],
        hf_layer_cache.recurrent_states[0][0].transpose(-1, -2),
        rtol=1e-4,
        atol=1e-5,
    )
    assert torch.equal(conv_state[2].float(), hf_layer_cache.conv_states[0][0, :, 1:].t())


def test_reference_operations_refuse_weights_that_require_grad() -> None:
    layer = gdn_layer()
    backend = bind_reference(layer).backend
    layer.A_log.requires_grad_(True)

    with pytest.raises(ValueError, match="frozen"):
        run_layer(layer, backend, projected_rows(2), extend_entry(0, 0, 2))


@pytest.mark.parametrize("operation", [ReferenceGatedDeltaNetDecode, ReferenceGatedDeltaNetPrefill])
def test_reference_operations_refuse_weights_of_another_geometry(operation: type) -> None:
    """Like the kernels, a reference operation serves the geometry it was built for."""

    swapped = operation(num_k_heads=NUM_VALUE_HEADS, num_v_heads=NUM_VALUE_HEADS, device="cpu")
    pool = RecurrentStatePool(_specs([gdn_layer()]), num_slots=1, device=torch.device("cpu"))
    slots = torch.zeros(1, dtype=torch.int32)
    sequence = (torch.tensor([0, 1], dtype=torch.int32), slots, torch.zeros(1, dtype=torch.bool))
    arguments = (slots,) if operation is ReferenceGatedDeltaNetDecode else sequence

    with pytest.raises(ValueError, match=r"\(2, 2\) heads"):
        swapped(projected_rows(1), *gdn_layer().weights, *arguments, *pool.layer_views(0))


# ============================== Training ==============================


def _trained_layer(heads: tuple[int, int]) -> tuple[GatedDeltaNet, RecordingOperation]:
    layer = gdn_layer(heads=heads, training=True)
    return layer, layer.test_services.ops.training


@pytest.mark.parametrize("heads", [(1, 2), (2, 4), (2, 6)], ids=["1x2", "2x4", "2x6"])
def test_training_matches_transformers_autograd_per_packed_sequence(
    heads: tuple[int, int],
) -> None:
    layer, operation = _trained_layer(heads)
    hf_module, hf_config = _hf_gated_delta_net(layer)
    hf_module.requires_grad_(False)
    lengths = (70, 1, 9)
    cu_seqlens = torch.tensor([0, 70, 71, 80], dtype=torch.int32)
    projected = projected_rows(80, width=hf_config.hidden_size).float().requires_grad_(True)
    upstream = torch.randn(80, layer.num_value_heads * HEAD_DIM)

    y = layer.forward_training(projected, cu_seqlens, 70)
    (gradient,) = torch.autograd.grad(y, projected, upstream)
    expected = torch.cat([hf_module(part[None])[0] for part in projected.split(lengths)])
    (expected_gradient,) = torch.autograd.grad(expected, projected, upstream)

    # Each sequence starts from zero state: Transformers runs each one alone.
    torch.testing.assert_close(y.detach(), expected.detach(), rtol=1e-4, atol=1e-5)
    # Gradients reach about 20; the two FP32 orders of the recurrence (per token here,
    # 64-token chunks in Transformers) differ by at most 3e-6 of that scale.
    scale = float(expected_gradient.abs().max())
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-4, atol=1e-5 * scale)
    # The layer hands the operation the projected rows, its weights, the boundaries and
    # the bound on every sequence's length.
    ((rows, conv_weight, *weights, boundaries),) = operation.calls
    assert operation.keywords == [{"max_sequence_length": 70}]
    assert rows is projected
    assert conv_weight.shape == (layer.conv_weight.shape[0], 4)
    assert conv_weight.data_ptr() == layer.conv_weight.data_ptr()
    assert all(
        actual is expected
        for actual, expected in zip(
            weights, (layer.A_log, layer.dt_bias, layer.norm_weight), strict=True
        )
    )
    assert boundaries is cu_seqlens
    for weight in layer.weights:
        assert weight.grad is None


def test_training_refuses_a_weight_that_requires_grad() -> None:
    layer, _ = _trained_layer((1, 2))
    layer.dt_bias.requires_grad_(True)

    with pytest.raises(ValueError, match="frozen"):
        layer.forward_training(
            projected_rows(4).float(), torch.tensor([0, 4], dtype=torch.int32), 4
        )


@pytest.mark.parametrize(
    ("max_sequence_length", "error"),
    [(True, TypeError), (4.0, TypeError), (0, ValueError), (9, ValueError), (2, ValueError)],
    ids=["bool", "float", "zero", "past-the-tokens", "below-a-sequence"],
)
def test_training_refuses_a_bad_sequence_bound(
    max_sequence_length: object,
    error: type[Exception],
) -> None:
    """The reference checks the bound like the kernel and, reading the boundaries on the
    host, also that no sequence is longer."""

    layer, _ = _trained_layer((1, 2))

    with pytest.raises(error, match="max_sequence_length"):
        layer.forward_training(
            projected_rows(8).float(),
            torch.tensor([0, 3, 8], dtype=torch.int32),
            max_sequence_length,
        )


# ===================== Backend: chunking, mixing, padding =====================


@pytest.mark.parametrize("split", [1, 3, 4, 63, 64, 65])
def test_chunked_prefill_then_decode_matches_one_shot_prefill(split: int) -> None:
    projected = projected_rows(70)
    reference_layer = gdn_layer()
    reference = bind_reference(reference_layer)
    one_shot = run_layer(reference_layer, reference.backend, projected, extend_entry(1, 0, 70))

    layer = gdn_layer()
    bound = bind_reference(layer)
    backend, pool = bound.backend, bound.pool
    chunks = [
        run_layer(layer, backend, projected[:split], extend_entry(1, 0, split)),
        run_layer(layer, backend, projected[split:66], extend_entry(1, split, 66)),
    ]
    chunks.extend(
        run_layer(layer, backend, projected[token : token + 1], decode_entry(1, token))
        for token in range(66, 70)
    )

    assert torch.equal(torch.cat(chunks), one_shot)
    assert torch.equal(pool.layer_views(0)[0][1], reference.pool.layer_views(0)[0][1])
    assert torch.equal(pool.layer_views(0)[1][1], reference.pool.layer_views(0)[1][1])


def test_mixed_batch_matches_each_request_run_alone() -> None:
    first, second = projected_rows(9, seed=2), projected_rows(12, seed=3)
    decode_a, decode_b = projected_rows(5, seed=4), projected_rows(4, seed=5)

    def warm(layer: GatedDeltaNet, backend: RecurrentBackend) -> None:
        run_layer(layer, backend, first[:4], extend_entry(1, 0, 4))
        run_layer(layer, backend, decode_a[:4], extend_entry(0, 0, 4))
        run_layer(layer, backend, decode_b[:3], extend_entry(2, 0, 3))

    alone_layer = gdn_layer()
    alone = bind_reference(alone_layer)
    alone_backend = alone.backend
    warm(alone_layer, alone_backend)
    alone_y = torch.cat(
        (
            run_layer(alone_layer, alone_backend, second, extend_entry(3, 0, 12)),
            run_layer(alone_layer, alone_backend, first[4:], extend_entry(1, 4, 9)),
            run_layer(alone_layer, alone_backend, decode_a[4:], decode_entry(0, 4)),
            run_layer(alone_layer, alone_backend, decode_b[3:], decode_entry(2, 3)),
        )
    )

    layer = gdn_layer()
    bound = bind_reference(layer)
    backend, pool = bound.backend, bound.pool
    warm(layer, backend)
    bound.prefill.calls.clear()
    bound.prefill.keywords.clear()
    bound.decode.calls.clear()
    mixed = run_layer(
        layer,
        backend,
        torch.cat((second, first[4:], decode_a[4:], decode_b[3:])),
        extend_entry(3, 0, 12),
        extend_entry(1, 4, 9),
        decode_entry(0, 4),
        decode_entry(2, 3),
    )

    assert torch.equal(mixed, alone_y)
    assert torch.equal(pool.layer_views(0)[0], alone.pool.layer_views(0)[0])
    assert torch.equal(pool.layer_views(0)[1], alone.pool.layer_views(0)[1])
    # One prefill call over the extend rows, one decode call over the decode rows.
    ((prefill_projected, *prefill_rest),) = bound.prefill.calls
    ((decode_projected, *decode_rest),) = bound.decode.calls
    assert prefill_projected.shape[0] == 17
    assert prefill_rest[4].tolist() == [0, 12, 17]
    assert prefill_rest[5].tolist() == [3, 1]
    assert prefill_rest[6].tolist() == [False, True]
    assert decode_projected.shape[0] == 2
    assert decode_rest[4].tolist() == [0, 2]
    # The prefill writes its rows of the batch's y in place.
    (prefill_keywords,) = bound.prefill.keywords
    assert prefill_keywords["out"].data_ptr() == mixed.data_ptr()
    assert prefill_keywords["out"].shape == (17, OUTPUT_WIDTH)
    # The decode writes its rows after them, into the same y.
    (decode_keywords,) = bound.decode.keywords
    assert decode_keywords["out"].data_ptr() == mixed[17:].data_ptr()
    assert decode_keywords["out"].shape == (2, OUTPUT_WIDTH)


def test_a_decode_batch_writes_into_the_breakable_graphs_buffer() -> None:
    layer = gdn_layer()
    bound = bind_reference(layer)
    backend = bound.backend
    run_layer(layer, backend, projected_rows(4), extend_entry(0, 0, 4))
    step = projected_rows(1, seed=6)
    rows = layer_rows(decode_entry(0, 4))
    batch = layer_inputs(rows, backend.prepare_metadata(rows))
    out = torch.full((1, OUTPUT_WIDTH), float("nan"), dtype=step.dtype)
    bound.decode.keywords.clear()

    y = layer._core_into(step, batch, out)

    (keywords,) = bound.decode.keywords
    assert y is out and keywords["out"] is out
    assert not out.isnan().any()


def test_a_pure_extend_batch_takes_the_prefills_own_output() -> None:
    layer = gdn_layer()
    bound = bind_reference(layer)

    y = run_layer(
        layer, bound.backend, projected_rows(7), extend_entry(1, 0, 3), extend_entry(2, 0, 4)
    )

    ((projected, *_),) = bound.prefill.calls
    assert projected.shape[0] == 7
    assert bound.prefill.keywords == [{"out": None, "capture_rows": None}]
    (result,) = bound.prefill.results
    assert result is y


def test_padding_rows_give_zero_and_leave_every_other_pool_row_untouched() -> None:
    layer = gdn_layer()
    bound = bind_reference(layer)
    backend, pool = bound.backend, bound.pool
    run_layer(layer, backend, projected_rows(3), extend_entry(2, 0, 3))
    before_state, before_conv = pool_snapshot(pool)

    y = run_layer(layer, backend, projected_rows(3, seed=7), decode_entry(2, 3), padding=2)

    assert y.shape == (3, OUTPUT_WIDTH)
    assert torch.count_nonzero(y[0]) > 0
    assert torch.equal(y[1:], torch.zeros_like(y[1:]))
    state, conv_state = pool.layer_views(0)
    untouched = [0, 1, 3]
    assert torch.equal(state[untouched], before_state[untouched])
    assert torch.equal(conv_state[untouched], before_conv[untouched])
    assert not torch.equal(state[2], before_state[2])


def test_a_fresh_request_ignores_nan_left_in_its_reused_slot() -> None:
    projected = projected_rows(6)
    clean_layer = gdn_layer()
    clean = run_layer(
        clean_layer, bind_reference(clean_layer).backend, projected, extend_entry(0, 0, 6)
    )

    layer = gdn_layer()
    bound = bind_reference(layer)
    backend, pool = bound.backend, bound.pool
    state, conv_state = pool.layer_views(0)
    state[3].fill_(float("nan"))
    conv_state[3].fill_(float("nan"))
    reused = run_layer(layer, backend, projected, extend_entry(3, 0, 6))

    assert torch.equal(reused, clean)
    assert torch.isfinite(state[3]).all()
    assert torch.isfinite(conv_state[3].float()).all()


# ============================ Decode CUDA graphs ============================


def test_layer_reads_the_captured_slot_buffer_at_capture_and_replay() -> None:
    layer = gdn_layer()
    bound = bind_reference(layer)
    backend, pool = bound.backend, bound.pool
    run_layer(layer, backend, projected_rows(4), extend_entry(0, 0, 4))
    run_layer(layer, backend, projected_rows(5, seed=8), extend_entry(3, 0, 5))
    backend.init_capture_graph(max_seq_len=64, bs_list=[1, 2, 4])
    before_state, before_conv = pool_snapshot(pool)

    capture = layer_rows(*(decode_entry(NUM_SLOTS, 0),) * 4)
    captured_metadata = backend.capture_metadata(capture)
    captured_slots = captured_metadata.slot_ids
    captured_projected = projected_rows(4, seed=9)
    bound.decode.calls.clear()
    captured_y = layer.forward(captured_projected, layer_inputs(capture, captured_metadata))

    assert captured_slots.tolist() == [-1, -1, -1, -1]
    assert torch.equal(captured_y, torch.zeros_like(captured_y))
    assert torch.equal(pool.layer_views(0)[0], before_state)
    assert torch.equal(pool.layer_views(0)[1], before_conv)
    ((_, *captured_arguments),) = bound.decode.calls
    assert captured_arguments[4] is captured_slots

    # Replay: the graph re-reads the same static tensors after prepare_for_replay.
    twin_layer = gdn_layer()
    twin = bind_reference(twin_layer)
    twin_backend = twin.backend
    run_layer(twin_layer, twin_backend, projected_rows(4), extend_entry(0, 0, 4))
    run_layer(twin_layer, twin_backend, projected_rows(5, seed=8), extend_entry(3, 0, 5))
    step = projected_rows(2, seed=10)
    eager = run_layer(twin_layer, twin_backend, step, decode_entry(3, 5), decode_entry(0, 4))

    replay = layer_rows(decode_entry(3, 5), decode_entry(0, 4), padding=2)
    metadata = backend.prepare_metadata(replay)
    backend.prepare_for_replay(SimpleNamespace(recurrent_metadata=metadata), replay.padded_size)
    captured_projected[:2] = step
    replayed_y = bound.decode(captured_projected, *captured_arguments)

    assert captured_slots.tolist() == [3, 0, -1, -1]
    assert torch.equal(replayed_y[:2], eager)
    assert torch.equal(replayed_y[2:], torch.zeros_like(replayed_y[2:]))
    assert torch.equal(pool.layer_views(0)[0], twin.pool.layer_views(0)[0])
    assert torch.equal(pool.layer_views(0)[1], twin.pool.layer_views(0)[1])


# ============================ Operation resolution ============================


def _resolve(device: torch.device, *, training: bool = False) -> GatedDeltaNetOps:
    with default_dtype(torch.bfloat16):
        return resolve_gated_delta_net_ops(
            device, num_key_heads=16, num_value_heads=48, training=training
        )


def test_kernel_operations_are_constructed_for_the_geometry_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ctikernels.gdn`` operations take keyword-only ``num_k_heads``, ``num_v_heads``
    and ``device`` and resolve their provider when constructed."""

    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())

    ops = _resolve(torch.device("cuda", 2), training=True)

    types = (
        ReferenceGatedDeltaNetDecode,
        ReferenceGatedDeltaNetPrefill,
        ReferenceGatedDeltaNetTraining,
    )
    assert tuple(type(operation) for operation in ops) == types
    for operation in ops:
        assert (operation.num_k_heads, operation.num_v_heads) == (16, 48)
        assert operation.device == torch.device("cuda", 2)
    assert _resolve(torch.device("cuda", 2)).training is None
    with pytest.raises(ValueError, match="require BF16"):
        resolve_gated_delta_net_ops(
            torch.device("cuda", 2), num_key_heads=16, num_value_heads=48, training=False
        )


@pytest.mark.parametrize("exports", [True, False])
def test_layers_of_one_geometry_share_their_models_operations(
    monkeypatch: pytest.MonkeyPatch, exports: bool
) -> None:
    built: list[tuple[int, int]] = []

    class GatedDeltaNetPrefill(ReferenceGatedDeltaNetPrefill):
        supports_state_capture = exports

        def __init__(self, **geometry: object) -> None:
            super().__init__(**geometry)
            built.append((self.num_k_heads, self.num_v_heads))

    module = reference_kernel_module()
    module.GatedDeltaNetPrefill = GatedDeltaNetPrefill  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", module)
    layer_site = site(services=LayerServices())
    with default_dtype(torch.bfloat16):
        layers = [
            GatedDeltaNet(
                num_key_heads=keys,
                num_value_heads=values,
                key_head_dim=HEAD_DIM,
                value_head_dim=HEAD_DIM,
                conv_kernel_size=4,
                site=layer_site,
            )
            for keys, values in ((1, 2), (1, 2), (2, 4))
        ]

    assert built == [(1, 2), (2, 4)]
    assert [layer.exports_prefill_states for layer in layers] == [exports] * 3
    # The query builds the prefill of the layers' device key and answers as they do.
    answers = [
        prefill_exports_states(layer_site.device, num_key_heads=keys, num_value_heads=values)
        for keys, values in ((1, 2), (2, 4))
    ]
    assert answers == [exports] * 2
    assert built == [(1, 2), (2, 4)] * 2


def test_a_prefill_capturing_off_the_layers_grid_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GatedDeltaNetPrefill(ReferenceGatedDeltaNetPrefill):
        CAPTURE_GRID = 128

    module = reference_kernel_module()
    module.GatedDeltaNetPrefill = GatedDeltaNetPrefill  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", module)
    with pytest.raises(RuntimeError, match="captures on a 128-token grid, the layers export on 64"):
        prefill_exports_states(torch.device("cpu"), num_key_heads=1, num_value_heads=2)


def test_the_training_protocol_and_reference_take_the_pinned_operations_parameters() -> None:
    """The layer calls the pinned release's training operation through its Protocol and CPU
    tests run the reference in its place, so all three take the same parameters."""

    def parameters(call: Callable[..., object]) -> list[tuple[str, object, object]]:
        return [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(call).parameters.values()
        ]

    released = parameters(GatedDeltaNetTraining.__call__)
    assert parameters(GatedDeltaNetTrainingOperation.__call__) == released
    assert parameters(ReferenceGatedDeltaNetTraining.__call__) == released


def test_a_provider_failure_at_construction_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cti-kernels names the operation and the device when its provider is missing."""

    module = reference_kernel_module()

    class GatedDeltaNetDecode:
        def __init__(self, *, num_k_heads: int, num_v_heads: int, device: torch.device) -> None:
            _ = num_k_heads, num_v_heads
            raise RuntimeError(f"GatedDeltaNetDecode needs cti-kernels[gdn] on {device}")

    module.GatedDeltaNetDecode = GatedDeltaNetDecode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", module)

    with pytest.raises(RuntimeError) as raised:
        _resolve(torch.device("cuda", 1))

    assert str(raised.value) == "GatedDeltaNetDecode needs cti-kernels[gdn] on cuda:1"
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("cut", [64, 128])
def test_snapshot_restore_is_bitwise_with_torch_gdn_reference(cut: int) -> None:
    layers = [gdn_layer(seed=i) for i in range(2)]
    pool = RecurrentStatePool(
        _specs(layers),
        num_slots=4,
        num_snapshot_rows=2,
        device=torch.device("cpu"),
    )
    backend = RecurrentBackend(pool, device=pool.device)
    for row, layer in enumerate(layers):
        layer.bind_state({"recurrent": pool.layer_state(row)})
        for view in pool.layer_views(row):
            view.fill_(float("nan"))
    x = projected_rows(cut + 67)
    # One forward exports the state after its whole 64-token multiples, ``cut``, that a
    # forward ending there keeps.
    head = layer_rows(
        extend_entry(0, 0, cut + 3, captures=(RecurrentExport(cut, 4),)),
        extend_entry(1, 0, cut),
    )
    head_inputs = layer_inputs(head, backend.prepare_metadata(head))
    heads = [layer.forward(torch.cat([x[: cut + 3], x[:cut]]), head_inputs) for layer in layers]
    for index in range(2):
        for view in pool.layer_views(index):
            assert torch.equal(view[1], view[4])
            assert torch.isnan(view[5]).all()
    # A restored follower continues like the leader, which runs on to the same end, and
    # exports 64 tokens past its restored state, as a forward ending there keeps.
    tails = layer_rows(
        extend_entry(2, cut, cut + 67, restore_row=4, captures=(RecurrentExport(cut + 64, 5),)),
        extend_entry(1, cut, cut + 64),
        extend_entry(0, cut + 3, cut + 67),
    )
    tails_metadata = backend.prepare_metadata(tails)
    backend.restore(SimpleNamespace(recurrent_metadata=tails_metadata))
    tails_inputs = layer_inputs(tails, tails_metadata)
    for layer, y_head in zip(layers, heads, strict=True):
        y = layer.forward(torch.cat([x[cut:], x[cut : cut + 64], x[cut + 3 :]]), tails_inputs)
        assert torch.equal(y[:3], y_head[cut : cut + 3])
        assert torch.equal(y[3:67], y[131:])
    for index in range(2):
        for view in pool.layer_views(index):
            assert torch.equal(view[2], view[0])
            assert torch.equal(view[5], view[1])
            assert torch.isnan(view[3]).all()


@pytest.mark.parametrize("cute", [True, False], ids=["cute", "triton"])
def test_the_k2_layout_holds_arrival_counters_only_where_the_replay_names_them(cute) -> None:
    from cti.layers.gated_delta_net.replay import GatedDeltaNetK2

    def tensor(*dims, dtype=torch.int32):
        return SimpleNamespace(shape=dims, dtype=dtype)

    graph_data = {"phase_arrivals": tensor(2, 4)} if cute else {}
    op = SimpleNamespace(
        checks="trusted",
        state_spec=lambda **_: {
            name: tensor(3) for name in ("ring", "metadata", "pool_epoch", "snapshot_metadata")
        },
        graph_buffer_spec=lambda **_: {"a": SimpleNamespace(offset=0, aligned_bytes=256)},
        workspace_spec=lambda *_, **__: {"pending.raw": tensor(8), "out": tensor(4, 2)},
        layer_table_spec=lambda **_: {"layer_table": tensor(1, 4, dtype=torch.int64)},
        graph_data_spec=lambda **_: graph_data,
    )

    layout, _ = GatedDeltaNetK2._layout(op, 1, 4, RoundShape(4, 2, 3), 2)

    assert ("arrivals" in layout) is cute


def test_training_bytes_are_the_training_ops_own() -> None:
    layer = gdn_layer(training=True)
    layer._training_op = SimpleNamespace(
        training_bytes=lambda tokens, sequences: (tokens, sequences)
    )

    assert layer.training_bytes(1000, 3) == (1000, 3)
