"""CPU initialization seam; GPU capture correctness is qualified separately."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cti import kvcache
from cti.attention.paged import KVPair
from cti.config import MemoryOptions, RandomWeights
from cti.inference._execution import runner as module
from cti.kvcache import (
    GroupRole,
    KVLayout,
    PagedSlot,
    RecurrentRows,
    ResolvedSlot,
    SnapshotPolicy,
    Spec,
    StateBudget,
    StateSlot,
    TableGeometry,
    state,
)
from cti.layers import ExpertParallel
from cti.layers.gated_delta_net.layer import gated_delta_net_state
from cti.models import CheckpointSource
from cti.plan import PromptBudget
from cti.worker import _device_memory as device_memory
from cti.worker import _model_host as model_host
from cti.worker._device_memory import DeviceMemory
from tests.support.gpu import fixed_kv_page_capacity

_RECURRENT_TO_KV_RATIO = MemoryOptions().recurrent_to_kv_ratio


@pytest.mark.parametrize("recurrent", (False, True))
@pytest.mark.parametrize("snapshot_rows", (0, 2))
@pytest.mark.parametrize("size, rank", ((1, 0), (2, 0), (2, 1)))
def test_tp_plan_reaches_decode_and_piecewise_graph_construction(
    monkeypatch: pytest.MonkeyPatch, size: int, rank: int, recurrent: bool, snapshot_rows: int
) -> None:
    backend = SimpleNamespace(execution_metadata=object())
    build = Mock(return_value=backend)
    plan = SimpleNamespace(
        model_path="unused",
        prefix_cache="radix" if snapshot_rows else "naive",
        lora=None,
        weights=RandomWeights(seed=0),
        dtype=torch.bfloat16,
        experts=ExpertParallel(),
        initialization_seed=42,
        max_batch_tokens=64,
        prompt_budget=PromptBudget(64, "test"),
        max_running_requests=4,
        auto_running_requests=False,
        seeds_every_row=False,
        speculative=None,
        rounds=None,
        graph_batch_sizes=(1, 2, 4),
        graph_max_batch=None,
        vision_graph_budgets=(),
        graph_row_buckets=() if recurrent else (16, 32),
        inference_ratio=0.8,
        peak_ratio=0.8,
        recurrent_to_kv_ratio=_RECURRENT_TO_KV_RATIO,
        # The resolved family reserves no workspace and builds ``backend``.
        attention=SimpleNamespace(
            selected="fi", reason="test", workspace_bytes=lambda **_kwargs: 0, build=build
        ),
        page_size=1,
        # This rank's share of the 4 KV heads.
        state_slots={
            (GroupRole.TARGET, "kv"): ResolvedSlot(
                "mha", Spec(KVPair(4 // size, 4, torch.bfloat16))
            )
        },
        capability=None,
        snapshots=SnapshotPolicy(snapshot_rows) if snapshot_rows else None,
        max_seq_len=64,
    )
    ranks = SimpleNamespace(
        size=size,
        rank=rank,
        is_primary=lambda: rank == 0,
        min_max_int=lambda value: (value, value),
    )
    # Free memory and the image encoder's peak are device measurements.
    monkeypatch.setattr(DeviceMemory, "_free", lambda _self: (1_000_000, 1_000_000))
    monkeypatch.setattr(module, "_image_encoder_peak", lambda *_args: 0)
    # The layers carry this rank's share of 8 query and 4 KV heads.
    attention = SimpleNamespace(
        num_qo_heads=8 // size,
        num_kv_heads=4 // size,
        head_dim=4,
        prologue=None,
        state_slots=lambda: (StateSlot("kv", requirement="kv"),),
        bind_state=Mock(),
        settle_inference=lambda: None,
    )
    recurrent_layer = SimpleNamespace(
        state_slots=lambda: (StateSlot("recurrent", state=gated_delta_net_state(1, 1, 1, 1)),),
        bind_state=Mock(),
        warm_up=Mock(),
        settle_inference=lambda: None,
    )
    recurrent_layers = (recurrent_layer,) if recurrent else ()
    model = SimpleNamespace(
        can_capture_piecewise_graph=True,
        load_state_dict=Mock(),
        iter_attention_layers=lambda: iter((attention,)),
        iter_recurrent_layers=lambda: recurrent_layers,
        iter_state_layers=lambda: (attention, *recurrent_layers),
        rope_position_axes=lambda: 1,
        inference_gemms=lambda: iter(()),
        settle_inference=lambda: None,
        warm_up_rows=Mock(),
        state_dict=dict,
    )
    model.named_ops = lambda: (
        ("", model),
        ("attention", attention),
        *(("recurrent", layer) for layer in recurrent_layers),
    )
    create_model = Mock(return_value=model)
    decode = SimpleNamespace(close=Mock(), batch_sizes=(1, 2, 4))
    create_decode = Mock(return_value=decode)
    piecewise = SimpleNamespace(close=Mock(), buckets=SimpleNamespace(sizes=(16, 32)))
    create_piecewise = Mock(return_value=piecewise)
    pool = SimpleNamespace(close=lambda: None)
    checkpoint = SimpleNamespace(
        model_class=object(),
        facts=SimpleNamespace(
            hf_config=object(), stop_token_ids=(2,), source=CheckpointSource("unused")
        ),
        model_config=SimpleNamespace(is_moe=False, num_layers=2, vocab_size=16),
        image_preprocessing=None,
    )
    monkeypatch.setattr(module, "logger", Mock())
    monkeypatch.setattr(model_host, "logger", Mock())
    monkeypatch.setattr(device_memory, "logger", Mock())
    monkeypatch.setattr(model_host, "create_model", create_model)
    recurrent_pool = Mock()
    monkeypatch.setattr(state, "RecurrentStatePool", recurrent_pool)
    recurrent_backend = Mock()
    monkeypatch.setattr(module, "RecurrentBackend", recurrent_backend)
    sampler = Mock()
    sampler_type = Mock(return_value=sampler)
    monkeypatch.setattr(module, "LegacySamplingLaw", Mock())
    monkeypatch.setattr(module, "Sampler", sampler_type)
    monkeypatch.setattr(module, "DecodeGraph", create_decode)
    monkeypatch.setattr(module, "BreakableGraph", create_piecewise)
    monkeypatch.setattr(device_memory, "GraphPool", lambda _region: pool)

    # TP qualification replaces page sizing; the runner still publishes real limits.
    resources = ExitStack()
    device = torch.device("cpu")
    memory = DeviceMemory(
        plan,  # type: ignore[arg-type]
        device=device,
        agree=ranks.min_max_int,
        manager=None,
    )
    host = model_host.ModelHost(
        plan,  # type: ignore[arg-type]
        checkpoint,  # type: ignore[arg-type]
        device=device,
        ranks=ranks,  # type: ignore[arg-type]
        training=False,
    )
    with fixed_kv_page_capacity(64):
        runner = module.build_executor(
            host,
            memory,
            host.plan,
            ranks=ranks,  # type: ignore[arg-type]
            preparation_stream=object(),  # type: ignore[arg-type]
            execution_stream=object(),  # type: ignore[arg-type]
            resources=resources,
        )
    with resources:
        limits = runner.limits
        # Only recurrent state keeps snapshot rows.
        rows = snapshot_rows if recurrent else 0
        kinds = _kinds(recurrent)
        assert limits.capacity == TableGeometry(1, 4, 64).capacity(kinds, 64, rows)
        if recurrent:
            assert recurrent_pool.call_args.kwargs["num_snapshot_rows"] == snapshot_rows
        create_decode.assert_called_once()
        if recurrent:
            create_piecewise.assert_not_called()
        else:
            create_piecewise.assert_called_once()
        # The model is built over the runner's group, with the plan's expert placement.
        site = create_model.call_args.kwargs["site"]
        assert site.group is ranks and site.experts is plan.experts
        # The layers are bound to the backend and the pool's row.
        attention.bind_state.assert_called_once_with({"kv": PagedSlot(backend, 0)})
        if recurrent:
            row_pool = recurrent_pool.return_value
            row_pool.layer_state.assert_called_once_with(0)
            recurrent_layer.bind_state.assert_called_once_with(
                {"recurrent": row_pool.layer_state.return_value}
            )
        ((group,), page_table), backend_args = build.call_args
        # One layer of the 64 pages and the sink page, with this rank's KV heads; the
        # resolved family holds the rank's heads it builds with.
        (layer,) = group.layers
        assert layer.shape == (2, 65, 1, 4 // size, 4)
        assert page_table is runner.page_table
        assert backend_args == {"max_running_req": plan.max_running_requests, "capability": None}
        # The recurrent backend is the one state participant, also in the decode graphs.
        participants = (recurrent_backend.return_value,) if recurrent else ()
        assert create_decode.call_args.kwargs["participants"] == (backend, *participants)
        assert create_decode.call_args.kwargs["batch_sizes"] == [1, 2, 4]
        # Every rank loads the sampling providers before readiness, at each decode bucket.
        sampler.warm_up.assert_called_once_with([1, 2, 4])
        assert sampler_type.call_args.kwargs["logits_dtype"] is plan.dtype
        if not recurrent:
            assert create_piecewise.call_args.kwargs["buckets"].sizes == (16, 32)
            assert create_piecewise.call_args.kwargs["pool"] is pool
        assert create_decode.call_args.kwargs["pool"] is pool
    decode.close.assert_called_once()
    if not recurrent:
        piecewise.close.assert_called_once()


def _kinds(recurrent: bool) -> KVLayout:
    specs = (gated_delta_net_state(1, 1, 1, 1),) if recurrent else ()
    return KVLayout((), RecurrentRows(specs))


@pytest.mark.parametrize(
    "recurrent, reuse, rows, expected",
    [
        (False, True, None, 0),
        (True, False, None, 0),
        (True, True, None, 2),
        (True, True, 3, 3),
    ],
)
def test_fixed_capacity_helper_returns_explicit_plan_and_restores_planner(
    recurrent, reuse, rows, expected
):
    original = kvcache.plan_capacity
    kinds = _kinds(recurrent)
    geometry = TableGeometry(1, 4, 64, guard_tokens=3)
    with fixed_kv_page_capacity(64):
        plan = kvcache.plan_capacity(
            kinds,
            geometry,
            StateBudget(0, None, _RECURRENT_TO_KV_RATIO),
            snapshots=SnapshotPolicy(rows) if reuse else None,
        )
    # Guard pages still come out of the fixed pages.
    assert plan == geometry.capacity(kinds, 61, expected)
    assert kvcache.plan_capacity is original
