from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field, fields, is_dataclass
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import pytest
import safetensors.torch
import torch

import cti.inference._execution.runner as runner_module
import cti.models.weight as weight_module
import cti.worker._device_memory as device_memory_module
import cti.worker._model_host as model_host_module
from cti import kvcache
from cti.attention.paged import KVPair, write_slots
from cti.config import Checkpoint as CheckpointOption
from cti.config import MemoryOptions, RandomWeights
from cti.forward import EagerImages, ForwardInputs, ForwardRows
from cti.inference import Forward, PageMap, RequestTable, Sampled, ServingLimits
from cti.inference._control._scheduling.request import Request
from cti.inference._control._scheduling.table import RequestSlot
from cti.inference._execution._graphs import (
    DecodeGraph,
    GraphedForward,
    RowBuckets,
    _DecodeGraphBuffer,
    forward_rows,
    resolve_decode_graph_batch_sizes,
)
from cti.inference._execution._sampling import DeviceSampled
from cti.inference._execution.runner import ExecutorParts, InferenceRunner
from cti.inference._step import GraphStats, QueryKind, RequestSpec, Row
from cti.kvcache import (
    GroupLayout,
    GroupRole,
    KVLayout,
    RecurrentRows,
    ResolvedSlot,
    SnapshotPolicy,
    Spec,
    StateBudget,
    StateSlot,
)
from cti.layers import ExpertParallel, Param, Site
from cti.layers.gated_delta_net import GatedDeltaNet
from cti.layers.gated_delta_net.layer import gated_delta_net_state
from cti.memory import MemoryManager
from cti.models import CheckpointSource
from cti.models.mapping import CheckpointMapping, MergeRule
from cti.models.weight import open_weights
from cti.multimodal import ImageLimits
from cti.plan import Plan, PromptBudget
from cti.protocol import SamplingParams
from cti.recurrent import RecurrentBackend
from cti.training.runner import TrainingMemoryPlan
from cti.worker._device_memory import DeviceMemory, TrainingMemoryError
from cti.worker._model_host import ModelHost
from tests.support.gated_delta_net import (
    ReferenceGatedDeltaNetPrefill,
    reference_kernel_module,
)
from tests.support.residency import one_rank

_RECURRENT_TO_KV_RATIO = MemoryOptions().recurrent_to_kv_ratio


class _Stream:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self._events = events

    def wait_stream(self, stream: _Stream) -> None:
        self._events.append(f"{self.name}.wait_stream:{stream.name}")


def _entry(
    request_index: int = 0,
    *,
    kv_prefix_len: int = 1,
    sequence_len: int = 2,
    write_position: int | None = 2,
    query_kind: QueryKind = QueryKind.DECODE,
) -> Row:
    return Row(request_index, query_kind, kv_prefix_len, sequence_len, write_position)


def _spec(slot: int, rope_offset: int = 0) -> RequestSpec:
    return RequestSpec(slot, 1, 8, rope_offset=rope_offset)


def _disable_tensor_pinning(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("empty", "tensor"):
        allocate = getattr(torch, name)

        def allocate_without_pinning(
            *args: object,
            _allocate=allocate,
            **kwargs: object,
        ) -> torch.Tensor:
            kwargs.pop("pin_memory", None)
            return _allocate(*args, **kwargs)

        monkeypatch.setattr(runner_module.torch, name, allocate_without_pinning)


def test_the_plan_does_not_expose_kv_page_count() -> None:
    init_fields = {descriptor.name for descriptor in fields(Plan) if descriptor.init}

    assert init_fields.isdisjoint({"num_pages", "num_page_override", "_num_pages_for_testing"})


def test_model_signature_reads_the_host_owned_tensors_and_their_trainable_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted = torch.zeros(2)
    external = torch.ones(3)
    host = object.__new__(ModelHost)
    host.plan = SimpleNamespace(dtype=torch.bfloat16, lora=None)
    config = object()
    source = CheckpointSource("/resolved/model")
    host.checkpoint = SimpleNamespace(
        model_class=SimpleNamespace(CONFIG=SimpleNamespace(from_hf=lambda _hf_config: config)),
        facts=SimpleNamespace(hf_config=None, source=source),
    )
    host.model = SimpleNamespace(state_dict=lambda: {"base.weight": persisted})
    host.adapter = SimpleNamespace(
        named_parameters=lambda: (("head.weight", external),), identity=lambda: {"rank": 2}
    )
    host._signature = None
    captured: dict[str, object] = {}

    def create_signature(**kwargs: object) -> str:
        captured.update(kwargs)
        return "sha256:test"

    monkeypatch.setattr(model_host_module, "model_signature", create_signature)

    assert host.signature() == "sha256:test"
    # The base state, then the adapter's factors declared trainable.
    assert captured["named_tensors"] == (("base.weight", persisted), ("head.weight", external))
    assert captured["source"] is source
    assert captured["config"] is config
    assert captured["trainable"] == {"rank": 2}
    # Computed once: the identity is immutable.
    captured.clear()
    assert host.signature() == "sha256:test" and not captured


def test_checkpoint_weights_load_as_the_model_declares(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1 + 2**-10 is exact in fp32 but rounds to 1 in the bf16 engine dtype.
    value = torch.full((2, 4), 1 + 2**-10)
    safetensors.torch.save_file(
        {
            "model.q_proj.weight": value.clone(),
            "model.k_proj.weight": value.clone(),
            "model.v_proj.weight": value.clone(),
            "model.A_log": value[0].clone(),
        },
        tmp_path / "model.safetensors",
    )
    forwarded: dict[str, object] = {}
    real_load_weight = weight_module.load_weight

    def record_load(*args, **kwargs):
        forwarded.update(kwargs)
        return real_load_weight(*args, **kwargs)

    monkeypatch.setattr(weight_module, "load_weight", record_load)
    # One rank: the model declares no checkpoint windows, so every tensor loads whole.
    params = (
        Param("model.qkv_proj.weight", (6, 4), torch.bfloat16),
        Param("model.A_log", (4,), torch.float32),
    )
    model = SimpleNamespace(
        checkpoint_mapping=CheckpointMapping(
            merges=(MergeRule("qkv_proj", ("q_proj", "k_proj", "v_proj")),)
        ),
        params=lambda: params,
    )
    weights = open_weights(CheckpointSource(str(tmp_path)), CheckpointOption(), pytest.fail)

    phases: list[str] = []
    state = weights.state_dict(  # type: ignore[arg-type]
        model, torch.device("cpu"), show_progress=True, progress=phases.append
    )

    assert weights.folder == str(tmp_path)
    assert state.keys() == {"model.qkv_proj.weight", "model.A_log"}
    assert torch.equal(state["model.qkv_proj.weight"], torch.ones(6, 4, dtype=torch.bfloat16))
    assert torch.equal(state["model.A_log"], value[0])
    assert forwarded["params"] is params
    assert forwarded["show_progress"] is True
    assert phases == ["loading weights: shard 1/1 (model.safetensors)"]


def _assert_request_free(value: object) -> None:
    assert not isinstance(value, (Request, RequestSlot, SamplingParams))
    if is_dataclass(value) and not isinstance(value, type):
        for descriptor in fields(value):
            _assert_request_free(getattr(value, descriptor.name))
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_request_free(key)
            _assert_request_free(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_request_free(item)


def _free_memory(monkeypatch: pytest.MonkeyPatch, local: int, least: int | None = None) -> None:
    """``DeviceMemory``'s readings: the rank's free bytes and the least across the ranks."""

    reading = (local, local if least is None else least)
    monkeypatch.setattr(DeviceMemory, "_free", lambda _self: reading)


def test_device_memory_derives_both_budgets_from_one_boot_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = SimpleNamespace(
        inference_ratio=0.6, peak_ratio=0.85, recurrent_to_kv_ratio=_RECURRENT_TO_KV_RATIO
    )
    _free_memory(monkeypatch, 12_000, 10_000)

    memory = DeviceMemory(
        plan,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        agree=lambda value: (value, value),
        manager=None,
    )

    assert memory._boot_local_bytes == 12_000
    assert memory.boot_bytes == 10_000
    assert memory.budget_bytes == 6_000
    assert memory.reserve_bytes == 1_500


def _boot(
    monkeypatch: pytest.MonkeyPatch,
    memory: MemoryOptions,
    *,
    integrated: bool | None = None,
) -> DeviceMemory:
    """A rank booting with 10_000 bytes free on a CUDA device, ``integrated`` or discrete,
    or on the CPU when None."""

    _free_memory(monkeypatch, 10_000)
    device = torch.device("cpu")
    if integrated is not None:
        device = torch.device("cuda", 0)
        properties = SimpleNamespace(is_integrated=int(integrated))
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)
    return DeviceMemory(
        SimpleNamespace(  # type: ignore[arg-type]
            inference_ratio=memory.inference_ratio,
            peak_ratio=memory.peak_ratio,
            recurrent_to_kv_ratio=memory.recurrent_to_kv_ratio,
        ),
        device=device,
        agree=lambda value: (value, value),
        manager=None,
    )


@pytest.mark.parametrize(
    ("memory", "integrated", "budget", "reserve", "source"),
    [
        # The discrete default keeps the bytes the former fixed 0.9 gave.
        (MemoryOptions(), None, 9_000, 1_000, "default 0.9"),
        (MemoryOptions(), False, 9_000, 1_000, "default 0.9"),
        (MemoryOptions(), True, 6_000, 4_000, "UMA default 0.6"),
        (MemoryOptions(peak_ratio=0.8), True, 6_000, 2_000, "UMA default 0.6"),
        (MemoryOptions(inference_ratio=0.5), False, 5_000, 5_000, "explicit 0.5"),
        (MemoryOptions(inference_ratio=0.75), True, 7_500, 2_500, "explicit 0.75"),
    ],
)
def test_device_memory_resolves_the_inference_ratio_by_device(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    memory: MemoryOptions,
    integrated: bool | None,
    budget: int,
    reserve: int,
    source: str,
) -> None:
    with caplog.at_level(logging.INFO):
        device_memory = _boot(monkeypatch, memory, integrated=integrated)

    assert (device_memory.budget_bytes, device_memory.reserve_bytes) == (budget, reserve)
    assert f"Inference memory ratio: {source}" in caplog.text


@pytest.mark.parametrize(("integrated", "free"), [(True, 7_000), (False, 5_000)])
def test_unified_memory_reads_the_host_available_memory(
    monkeypatch: pytest.MonkeyPatch, integrated: bool, free: int
) -> None:
    """The plan reads free memory as cti.memory.free_bytes does for the device: the
    host's on unified memory (GB10), the device's own on a discrete GPU."""

    properties = SimpleNamespace(is_integrated=int(integrated))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *_args: None)
    monkeypatch.setattr(
        "cti.worker._device_memory.free_bytes",
        lambda _device, *, unified: 7_000 if unified else 5_000,
    )

    memory = DeviceMemory(
        SimpleNamespace(  # type: ignore[arg-type]
            inference_ratio=0.5, peak_ratio=None, recurrent_to_kv_ratio=_RECURRENT_TO_KV_RATIO
        ),
        device=torch.device("cuda", 0),
        agree=lambda value: (value, value),
        manager=None,
    )

    assert memory.boot_bytes == free


@pytest.mark.parametrize("integrated", [True, False])
def test_ranks_apart_by_more_than_2_gib_fail_unless_their_memory_is_unified(
    monkeypatch: pytest.MonkeyPatch, integrated: bool
) -> None:
    """On one host every rank reads the same memory; hosts' unified memory differs by their
    page caches, so ranks across hosts plan with the least."""

    gib = 1024**3
    properties = SimpleNamespace(is_integrated=int(integrated))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *_args: None)
    monkeypatch.setattr(
        "cti.worker._device_memory.free_bytes", lambda _device, *, unified: 40 * gib
    )

    def boot() -> DeviceMemory:
        return DeviceMemory(
            SimpleNamespace(  # type: ignore[arg-type]
                inference_ratio=0.5, peak_ratio=None, recurrent_to_kv_ratio=_RECURRENT_TO_KV_RATIO
            ),
            device=torch.device("cuda", 0),
            agree=lambda value: (value - 3 * gib, value),
            manager=None,
        )

    if integrated:
        assert boot().boot_bytes == 37 * gib
    else:
        with pytest.raises(RuntimeError, match="imbalanced"):
            boot()


def test_device_memory_checks_peak_ratio_against_the_resolved_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _boot(monkeypatch, MemoryOptions(peak_ratio=0.7), integrated=True).reserve_bytes == 3_000
    with pytest.raises(ValueError, match=r"inference_ratio is default 0\.9"):
        _boot(monkeypatch, MemoryOptions(peak_ratio=0.7), integrated=False)


def _record_plans(monkeypatch: pytest.MonkeyPatch) -> list[tuple[StateBudget, object]]:
    """Each budget the runner plans and the capacity it planned."""

    plans = []
    plan = kvcache.plan_capacity

    def record(layout, geometry, budget, **kwargs):
        plans.append((budget, plan(layout, geometry, budget, **kwargs)))
        return plans[-1][1]

    monkeypatch.setattr(kvcache, "plan_capacity", record)
    return plans


def test_runner_passes_budget_after_attention_workspace_to_capacity_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())

    plans = _record_plans(monkeypatch)
    _, _, host, runner, state = _construct_hybrid_runner(monkeypatch, [], attention_workspace=80)

    ((budget, capacity),) = plans
    assert budget == StateBudget((4 << 20) - 80, None, _RECURRENT_TO_KV_RATIO)
    (group,) = state.groups
    # One KV head of size 4 per page, in BF16; without prefix reuse, no snapshot rows.
    assert tuple(group.layers[0].shape[2:]) == (1, 1, 4)
    assert group.layers[0].dtype == torch.bfloat16
    assert state.recurrent.num_slots == 2 and not state.capacity.snapshot_rows
    # The resolved family sizes its own backends, with this rank's heads it resolved.
    assert host.plan.attention.reserved == [{"device": runner.device, "max_running_req": 2}]
    assert state.capacity == capacity
    # What the engine sizes its cache and scheduler by.
    plan = host.plan
    assert runner.limits == ServingLimits(
        capacity=capacity,
        rounds=plan.rounds,
        stop_token_ids=host.checkpoint.facts.stop_token_ids,
        vocab_size=host.checkpoint.model_config.vocab_size,
        image_rows=0,
        prefix_cache=plan.prefix_cache,
        max_batch_tokens=plan.max_batch_tokens,
        max_prompt_tokens=plan.prompt_budget.tokens,
    )


def test_the_gemm_workspace_comes_out_of_the_kv_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared: list[tuple[object, ...]] = []
    workspace = 1 << 20

    def prepare(models: tuple[object, ...], *, max_rows: int, **_options: object) -> int:
        prepared.append((models, max_rows))
        # Allocating the workspace lowers the free memory the plan measures next.
        _free_memory(monkeypatch, (8 << 20) - workspace)
        return workspace

    monkeypatch.setattr(model_host_module, "prepare_inference", prepare)
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())

    plans = _record_plans(monkeypatch)
    _, _, host, runner, _ = _construct_hybrid_runner(monkeypatch, [])

    # The most rows the tiny runner's forwards can run, prefill chunks included.
    ((models, max_rows),) = prepared
    assert models == (runner.model,) and max_rows == host.plan.max_batch_tokens
    assert runner._max_rows == max_rows
    assert [budget.available for budget, _ in plans] == [(4 << 20) - workspace]


def test_runner_sizes_the_kv_around_what_training_keeps_beside_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())

    events: list[str] = []
    warm_up = model_host_module.prepare_inference
    monkeypatch.setattr(
        model_host_module,
        "prepare_inference",
        lambda *args, **kwargs: events.append("warm_up") or warm_up(*args, **kwargs),
    )
    plans = _record_plans(monkeypatch)
    training = _training_plan(inference=1 << 10, training=1 << 10)
    model, *_ = _construct_hybrid_runner(monkeypatch, events, training_plan=training)

    # The host builds the model at a training site; the KV leaves training's resident bytes.
    assert model.site.training is True
    assert training.resident_bytes == 1 << 10
    assert [(budget.available, budget.beside) for budget, _ in plans] == [
        (4 << 20, training.resident_bytes)
    ]
    # Training prepares once, after the runner's warm-up and before the KV is sized.
    assert events.count("training.prepare") == 1
    assert events.index("warm_up") < events.index("training.prepare")
    assert events.index("training.prepare") < events.index("graph.capture")


def test_kv_planning_refuses_an_allocator_placement_it_does_not_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _device_memory(monkeypatch, 1 << 20)
    memory._device = torch.device("cuda", 0)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    with pytest.raises(ValueError, match="native placement; .* sets expandable_segments"):
        memory.plan(
            _kv_layout(1, rows=_NO_ROWS, head_dim=4),
            geometry=kvcache.TableGeometry(1, 2, 16),
            snapshots=None,
            workspace_bytes=0,
            image_encoder_bytes=0,
            image_rows_bytes=0,
            beside_bytes=0,
            allocations=None,
            fit_slots=False,
        )


def test_kv_capacity_uses_the_available_budget() -> None:
    # 100 pages of 4 bytes, one of them the graph sink page.
    assert _pages_and_rows(_plan(400, slots=7)) == (99, 0)


def _training_plan(*, inference: int, training: int) -> TrainingMemoryPlan:
    return TrainingMemoryPlan(
        master_weights=inference,
        optimizer_state=0,
        gradients=0,
        input_staging=0,
        optimizer_step=0,
        activations=training - inference,
        activation_segments=0,
    )


def test_kv_capacity_leaves_out_inference_resident_training_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _pages_and_rows(_plan_beside(monkeypatch, 400, slots=7, training=120)) == (69, 0)


@pytest.mark.parametrize(("available", "error"), [(400, TrainingMemoryError), (7, ValueError)])
def test_capacity_names_training_when_its_reservation_leaves_too_few_pages(
    monkeypatch: pytest.MonkeyPatch, available: int, error: type[ValueError]
) -> None:
    # Inference alone fits 400 bytes, so the reservation is to blame; at 7 it never fits.
    with pytest.raises(error, match="at least 2 pages") as raised:
        _plan_beside(monkeypatch, available, slots=7, training=398)
    assert (raised.type is TrainingMemoryError) is (error is TrainingMemoryError)


def test_training_must_fit_beside_inference_under_the_native_allocator_placement(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    def fits(free: int, released: int) -> None:
        memory = _device_memory(monkeypatch, free, manager=SimpleNamespace(kv_bytes=released))
        memory.reserve_bytes = 9
        memory.check_training_memory(_training_plan(inference=120, training=441))

    with caplog.at_level(logging.WARNING):
        fits(120, 330)
    assert "estimated peak" not in caplog.text
    # A peak beyond the released inference state short of the reserve only warns, with
    # the numbers; what training keeps needs free memory beside the inference state.
    with caplog.at_level(logging.WARNING):
        fits(120, 329)
    assert "estimated peak" in caplog.text and "largest terms: activations" in caplog.text
    with pytest.raises(TrainingMemoryError, match="Training keeps"):
        fits(119, 1 << 20)
    # Default values keep the native placement; other options change it.
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:False,backend:native,large_segment_size_mb:20",
    )
    fits(120, 330)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "roundup_power2_divisions:1,max_split_size_mb:64")
    with pytest.raises(TrainingMemoryError, match="max_split_size_mb, roundup_power2_divisions"):
        fits(120, 330)


@pytest.mark.parametrize(("kv_bytes", "fits"), [(0, False), (1, True)])
def test_runner_warns_of_training_beyond_what_inference_leaves(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    kv_bytes: int,
    fits: bool,
) -> None:
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())
    monkeypatch.setattr(MemoryManager, "kv_bytes", property(lambda _self: kv_bytes))
    # Free once everything inference keeps exists (8 MiB here), plus what releasing the
    # KV frees, short of the peak reserve (a tenth of boot free memory).
    room = (8 << 20) - ((8 << 20) - int(0.9 * (8 << 20)))
    training = _training_plan(inference=0, training=room + 1)

    with caplog.at_level(logging.WARNING):
        _construct_hybrid_runner(monkeypatch, [], training_plan=training)
    warned = "estimated peak" in caplog.text
    assert warned is not fits
    if warned:
        assert "0.00 GiB of the inference state released" in caplog.text


@pytest.mark.parametrize(("excess", "fits"), [(1, False), (0, True)])
def test_runner_without_a_memory_manager_releases_nothing_for_training(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    excess: int,
    fits: bool,
) -> None:
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())
    # The inference state stays through training: its peak has only the free memory (8 MiB
    # here) short of the peak reserve (a tenth of boot free memory).
    room = (8 << 20) - ((8 << 20) - int(0.9 * (8 << 20)))
    training = _training_plan(inference=0, training=room + excess)

    with caplog.at_level(logging.WARNING):
        _construct_hybrid_runner(monkeypatch, [], training_plan=training, managed=False)
    assert ("estimated peak" in caplog.text) is not fits


@pytest.mark.parametrize("available_memory", [-4, 0, 4, 7, 8, 11])
def test_kv_capacity_rejects_fewer_than_two_pages(available_memory) -> None:
    with pytest.raises(ValueError, match="Not enough memory for KV cache: at least 2 pages"):
        _plan(available_memory, slots=7)


def test_kv_capacity_accepts_exactly_two_pages() -> None:
    # Two usable pages and the graph sink page.
    assert _pages_and_rows(_plan(12, slots=7)) == (2, 0)


@pytest.mark.parametrize("training", [0, 40])
def test_reserved_guard_pages_come_out_of_the_usable_pages(
    monkeypatch: pytest.MonkeyPatch, training: int
) -> None:
    # 99 usable pages of 4 bytes beside the sink page and what training keeps: 97 guard pages
    # leave the two a request needs, 98 too few, which training's bytes would have covered.
    budget = 400 + training
    plan = partial(_plan_beside, monkeypatch, budget, slots=7, training=training)
    assert _pages_and_rows(plan(guard_tokens=97)) == (2, 0)
    error = TrainingMemoryError if training else ValueError
    with pytest.raises(error, match="Not enough memory for KV cache: at least 2 pages"):
        plan(guard_tokens=98)


class _Group:
    def min_max_int(self, value: int) -> tuple[int, int]:
        return value - 20, value + 10


def _kv_layout(
    *num_layers: int, rows: RecurrentRows, num_kv_heads: int = 1, head_dim: int = 1
) -> KVLayout:
    """A BF16 KV group of each count of layers, the target's first, beside ``rows``."""

    roles = (GroupRole.TARGET, GroupRole.DRAFT)
    spec = Spec(KVPair(num_kv_heads, head_dim, torch.bfloat16))
    return KVLayout(
        tuple(GroupLayout(roles[i], i, spec, n) for i, n in enumerate(num_layers)), rows
    )


# Six bytes per request and layer: one FP32 state element and one BF16 history element.
_TINY_RECURRENT_SPEC = gated_delta_net_state(1, 1, 1, 1, conv_history=1)
_NO_ROWS = RecurrentRows(())
_TINY_ROWS = RecurrentRows((_TINY_RECURRENT_SPEC, _TINY_RECURRENT_SPEC))


def _plan(
    available: int,
    *,
    slots: int,
    rows: RecurrentRows = _NO_ROWS,
    snapshots: SnapshotPolicy | None = None,
    fit_slots: bool = False,
) -> kvcache.StateCapacity:
    """Plan one tiny paged layer and ``rows`` by payload: four bytes a page, one BF16 key
    and one BF16 value element."""

    return kvcache.plan_capacity(
        _kv_layout(1, rows=rows),
        kvcache.TableGeometry(1, slots, 64),
        StateBudget(available, None, _RECURRENT_TO_KV_RATIO, fit_slots=fit_slots),
        snapshots=snapshots,
    )


def _device_memory(
    monkeypatch: pytest.MonkeyPatch, free: int, *, manager: object = None
) -> DeviceMemory:
    """A CPU rank's ``DeviceMemory`` whose whole ``free`` memory is the inference budget."""

    _free_memory(monkeypatch, free)
    plan = SimpleNamespace(
        inference_ratio=1.0, peak_ratio=None, recurrent_to_kv_ratio=_RECURRENT_TO_KV_RATIO
    )
    return DeviceMemory(
        plan,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        agree=lambda value: (value, value),
        manager=manager,  # type: ignore[arg-type]
    )


def _plan_beside(
    monkeypatch: pytest.MonkeyPatch,
    available: int,
    *,
    slots: int,
    training: int,
    guard_tokens: int = 0,
) -> kvcache.StateCapacity:
    """Plan one tiny paged layer beside ``training`` bytes, with ``guard_tokens`` of guard
    pages, as the device memory plans it."""

    return _device_memory(monkeypatch, available).plan(
        _kv_layout(1, rows=_NO_ROWS),
        geometry=kvcache.TableGeometry(1, slots, 64, guard_tokens=guard_tokens),
        snapshots=None,
        workspace_bytes=0,
        image_encoder_bytes=0,
        image_rows_bytes=0,
        beside_bytes=training,
        allocations=None,
        fit_slots=False,
    )


def _pages_and_rows(capacity: kvcache.StateCapacity) -> tuple[int, int]:
    return capacity.num_pages, len(capacity.snapshot_rows)


def test_kv_capacity_reserves_recurrent_state_once_before_paging() -> None:
    # 480 available bytes; 5 requests x 2 layers x 6 bytes = 60 are recurrent state,
    # leaving 420 bytes, i.e. 105 pages of 4 bytes, one of them the graph sink page.
    plan = _plan(480, slots=5, rows=_TINY_ROWS)

    assert plan.num_pages == 104
    assert not plan.snapshot_rows


# 12 recurrent bytes per request in the recurrent share, 0.9 / 1.9 of the budget.
@pytest.mark.parametrize(
    ("available", "slots"),
    [(4096, 8), (160, 6), (40, 1), (20, None)],
    ids=["cap", "tight", "one", "none"],
)
def test_unset_running_requests_take_the_slots_the_recurrent_share_holds(
    available: int, slots: int | None, caplog: pytest.LogCaptureFixture
) -> None:
    if slots is None:
        with pytest.raises(ValueError, match="one running request.*raise memory.inference_ratio"):
            _plan(available, slots=8, rows=_TINY_ROWS, fit_slots=True)
        return
    with caplog.at_level(logging.INFO):
        capacity = _plan(available, slots=8, rows=_TINY_ROWS, fit_slots=True)
    assert capacity.num_slots == slots
    assert f"max_running_requests={slots} (auto: recurrent state" in caplog.text
    share = int(available * _RECURRENT_TO_KV_RATIO / (1 + _RECURRENT_TO_KV_RATIO))
    assert slots == min(8, share // 12)
    # The KV takes the rest, the graph sink page aside.
    assert capacity.num_pages == (available - 12 * slots) // 4 - 1


def test_set_running_requests_over_budget_name_the_need_and_the_fix() -> None:
    with pytest.raises(ValueError) as raised:
        _plan(480, slots=40, rows=_TINY_ROWS)

    message = str(raised.value)
    assert "max_running_requests=40 needs 40 x " in message
    assert "lower scheduling.max_running_requests" in message
    assert "raise memory.inference_ratio" in message


def test_slot_rows_take_the_slot_row_states_of_layers_in_order() -> None:
    def layer(spec):
        return SimpleNamespace(state_slots=lambda: (StateSlot("recurrent", state=spec),))

    other = gated_delta_net_state(2, 1, 1, 1, conv_history=1)
    layers = [(GroupRole.TARGET, layer(spec)) for spec in (_TINY_RECURRENT_SPEC, other)]

    assert kvcache.group_layers(layers, {}).rows.specs == (_TINY_RECURRENT_SPEC, other)  # type: ignore[arg-type]


class _RegionProvider:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.active: list[str] = []

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False) -> Iterator[None]:
        assert not enable_cpu_backup
        self.active.append(tag)
        try:
            yield
        finally:
            self.active.pop()

    def pause(self, tag: str) -> None:
        self.events.append(f"pause({tag})")

    def resume(self, tag: str) -> None:
        self.events.append(f"resume({tag})")


def _allocate(
    layout: KVLayout, capacity: kvcache.StateCapacity, memory_manager: MemoryManager | None
) -> kvcache.InferenceState:
    """Allocate as the runner does, through ``DeviceMemory``: the state's data inside the
    memory manager's KV region, when managed."""

    memory = object.__new__(DeviceMemory)
    memory._manager, memory._device = memory_manager, torch.device("cpu")
    state, _ = memory.allocate(layout, capacity, None, torch.bfloat16)
    return state


@pytest.mark.parametrize("managed", [True, False], ids=["managed", "unmanaged"])
def test_inference_state_allocates_every_pool_inside_the_kv_region(
    monkeypatch: pytest.MonkeyPatch,
    managed: bool,
) -> None:
    provider = _RegionProvider()
    memory_manager = MemoryManager(provider, unified=False) if managed else None
    allocations: list[str | None] = []
    tensors: list[torch.Tensor] = []
    allocate = torch.empty

    def record_empty(*args: object, **kwargs: object) -> torch.Tensor:
        allocations.append(provider.active[-1] if provider.active else None)
        tensors.append(allocate(*args, **kwargs))
        return tensors[-1]

    layout = _kv_layout(3, 1, rows=_TINY_ROWS, head_dim=4)
    monkeypatch.setattr(torch, "empty", record_empty)
    import ctikernels.kvcache.inference as kvcache_ops

    tables: list[str | None] = []
    row_copy = kvcache_ops.RowCopy

    def record_row_copy(tensors: list[torch.Tensor]) -> object:
        tables.append(provider.active[-1] if provider.active else None)
        return row_copy(tensors)

    monkeypatch.setattr(kvcache_ops, "RowCopy", record_row_copy)
    state = _allocate(
        layout, kvcache.TableGeometry(1, 2, 16).capacity(layout, 3, 0), memory_manager
    )

    assert state.recurrent is not None
    assert state.recurrent.num_layers == 2
    assert state.recurrent.num_slots == 2
    # Each KV group holds the usable pages, then the decode-graph sink page.
    assert [(len(group.layers), group.layers[0].shape[1]) for group in state.groups] == [
        (3, 4),
        (1, 4),
    ]
    # Recurrent layers, then the target's KV layers, then the draft's.
    layers = [state.recurrent.layer_views(index)[0] for index in range(2)]
    layers += [layer for group in state.groups for layer in group.layers]
    assert [tensor.data_ptr() for tensor in tensors] == [layer.data_ptr() for layer in layers]
    assert allocations == [("kv-cache" if managed else None)] * 6
    # The recurrent rows' copy table of their addresses is metadata a suspension must not
    # lose: built outside the region.
    assert tables == [None]
    if not managed:
        return
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_args: (1 << 40, 1 << 40))
    memory_manager.suspend_kv()
    memory_manager.resume_kv(one_rank)
    assert provider.events == ["pause(kv-cache)", "resume(kv-cache)"]


@pytest.mark.parametrize("managed", [True, False], ids=["managed", "unmanaged"])
def test_attention_only_models_allocate_main_kv_regions_and_no_recurrent_state(
    managed: bool,
) -> None:
    provider = _RegionProvider()
    memory_manager = MemoryManager(provider, unified=False) if managed else None
    layout = _kv_layout(2, rows=_NO_ROWS, head_dim=4)

    state = _allocate(
        layout, kvcache.TableGeometry(1, 2, 16).capacity(layout, 1, 0), memory_manager
    )

    assert state.recurrent is None
    assert len(state.groups[0].layers) == 2
    if managed:
        memory_manager.suspend_kv()
        assert provider.events == ["pause(kv-cache)"]


class _HybridModel:
    """Two Gated DeltaNet layers around one paged attention layer; at any TP size each rank's
    layers hold one key and two value heads."""

    def __init__(self, events: list[str], rope_position_axes: int) -> None:
        self._events = events
        self._rope_position_axes = rope_position_axes
        self.attention = _BoundAttention()
        self.recurrent: tuple[GatedDeltaNet, ...] = ()

    def build(self, site: Site) -> _HybridModel:
        """Build the recurrent layers at ``site``, as the model's constructor would."""

        self.site = site
        size = site.group.size
        self.recurrent = tuple(
            GatedDeltaNet(
                num_key_heads=size,
                num_value_heads=2 * size,
                key_head_dim=128,
                value_head_dim=128,
                conv_kernel_size=4,
                site=site,
            )
            for _ in range(2)
        )
        return self

    def iter_attention_layers(self) -> Iterator[object]:
        return iter((self.attention,))

    def iter_recurrent_layers(self) -> tuple[GatedDeltaNet, ...]:
        return self.recurrent

    def iter_state_layers(self) -> tuple[object, ...]:
        return (self.attention, *self.recurrent)

    def rope_position_axes(self) -> int:
        return self._rope_position_axes

    def named_ops(self) -> Iterator[tuple[str, object]]:
        yield "", self
        yield "attention", self.attention
        yield from ((f"recurrent.{index}", layer) for index, layer in enumerate(self.recurrent))

    def inference_gemms(self) -> Iterator[object]:
        return iter(())

    def settle_inference(self) -> None:
        pass

    def warm_up_rows(self, max_rows: int, **options: object) -> None:
        pass

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        assert state_dict == {}
        self._events.append("weights.load")


class _BoundAttention:
    """One paged attention layer's local geometry, recording what the runner binds."""

    num_qo_heads, num_kv_heads, head_dim, prologue = 2, 1, 4, None

    def __init__(self) -> None:
        self.bound: dict[str, object] = {}

    def state_slots(self) -> tuple[StateSlot, ...]:
        return (StateSlot("kv", requirement="kv"),)

    def bind_state(self, bound: dict[str, object]) -> None:
        self.bound = dict(bound)

    def settle_inference(self) -> None:
        pass


def _recording_kernel_module(events: list[str], *, training: bool = True) -> ModuleType:
    """The reference ``ctikernels.gdn``, recording each operation it constructs."""

    module = reference_kernel_module(training=training)
    for name in ("GatedDeltaNetDecode", "GatedDeltaNetPrefill", "GatedDeltaNetTraining"):
        if hasattr(module, name):

            class Recorded(getattr(module, name)):
                def __init__(self, *, num_k_heads: int, num_v_heads: int, device, _name=name):
                    events.append(f"{_name}({num_k_heads}, {num_v_heads}, {device})")
                    super().__init__(
                        num_k_heads=num_k_heads, num_v_heads=num_v_heads, device=device
                    )

            setattr(module, name, Recorded)
    return module


class _RecordingProvider(_RegionProvider):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False) -> Iterator[None]:
        self.events.append(f"region({tag})")
        with super().region(tag, enable_cpu_backup):
            yield


@dataclass
class _ResolvedAttention:
    """A resolved attention family that reserves ``workspace`` bytes and builds ``backend``,
    recording what the runner asks of it."""

    backend: object
    workspace: int = 0
    selected: str = "test"
    reason: str = "test"
    reserved: list[dict[str, object]] = field(default_factory=list)
    built: list[tuple[tuple[object, ...], dict[str, object]]] = field(default_factory=list)

    def workspace_bytes(self, **kwargs: object) -> int:
        self.reserved.append(kwargs)
        return self.workspace

    def build(self, *args: object, **kwargs: object) -> object:
        self.built.append((args, kwargs))
        return self.backend


class _Hybrid(NamedTuple):
    """What ``_construct_hybrid_runner`` assembled, and the inference state it allocated."""

    model: _HybridModel
    manager: MemoryManager | None
    host: ModelHost
    runner: InferenceRunner
    state: kvcache.InferenceState


def _construct_hybrid_runner(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    prefix_cache: str = "naive",
    snapshot_rows: int = 2,
    rope_position_axes: int = 3,
    tp_size: int = 1,
    training_enabled: bool = False,
    training_plan: TrainingMemoryPlan | None = None,
    managed: bool = True,
    attention_workspace: int = 0,
) -> _Hybrid:
    """The worker's assembly on CPU: device memory, the model host, the runner, then the
    training fit check when ``training_plan`` is given."""

    if training_enabled and training_plan is None:
        training_plan = _training_plan(inference=0, training=0)
    model = _HybridModel(events, rope_position_axes)
    memory_manager = MemoryManager(_RecordingProvider(events), unified=False) if managed else None
    plan = SimpleNamespace(
        model_path="unused",
        dtype=torch.bfloat16,
        initialization_seed=0,
        lora=None,
        experts=ExpertParallel(),
        weights=RandomWeights(seed=0),
        max_batch_tokens=8,
        prompt_budget=PromptBudget(8, "test"),
        max_running_requests=2,
        auto_running_requests=False,
        graph_row_buckets=(),
        prefix_cache=prefix_cache,
        snapshots=SnapshotPolicy(snapshot_rows) if prefix_cache == "radix" else None,
        seeds_every_row=False,
        speculative=None,
        rounds=None,
        graph_batch_sizes=(1, 2),
        graph_max_batch=None,
        vision_graph_budgets=(),
        inference_ratio=0.5,
        peak_ratio=0.9,
        recurrent_to_kv_ratio=_RECURRENT_TO_KV_RATIO,
        attention=_ResolvedAttention(
            SimpleNamespace(name="attention-backend"),
            workspace=attention_workspace,
        ),
        page_size=1,
        # The attention layer's one KV head of size 4 on every rank.
        state_slots={
            (GroupRole.TARGET, "kv"): ResolvedSlot("mha", Spec(KVPair(1, 4, torch.bfloat16)))
        },
        capability=None,
        max_seq_len=16,
    )
    checkpoint = SimpleNamespace(
        model_class=None,
        facts=SimpleNamespace(hf_config=None, stop_token_ids=(7,), source=CheckpointSource("/m")),
        model_config=SimpleNamespace(
            num_kv_heads=1,
            head_dim=4,
            num_layers=3,
            vocab_size=8,
        ),
        image_preprocessing=None,
    )

    class DecodeGraph:
        captured: list[DecodeGraph] = []

        def __init__(self, **kwargs: object) -> None:
            events.append("graph.capture")
            self.kwargs = kwargs
            self.batch_sizes = tuple(kwargs["batch_sizes"])
            self.captured.append(self)

        def close(self) -> None:
            pass

    # A text checkpoint has no image preprocessing.
    def create_model(*_args: object, site: Site) -> _HybridModel:
        return model.build(site)

    monkeypatch.setattr(model_host_module, "create_model", create_model)

    def sampler(*_args: object, law: str, logits_dtype: torch.dtype, **_k) -> SimpleNamespace:
        events.append(f"sampler({law}, {logits_dtype})")
        return SimpleNamespace(warm_up=lambda _sizes: events.append("sampler.warm_up"))

    monkeypatch.setattr(
        runner_module, "LegacySamplingLaw", lambda _capability: events.append("law") or "legacy"
    )
    monkeypatch.setattr(runner_module, "Sampler", sampler)
    monkeypatch.setattr(runner_module, "DecodeGraph", DecodeGraph)
    monkeypatch.setattr(
        device_memory_module, "GraphPool", lambda _region: SimpleNamespace(close=lambda: None)
    )
    _free_memory(monkeypatch, (30 if prefix_cache == "radix" else 8) << 20)
    # A text model encodes no image; the measurement itself needs CUDA.
    monkeypatch.setattr(runner_module, "_image_encoder_peak", lambda *_args: 0)
    states: list[kvcache.InferenceState] = []
    allocate = DeviceMemory.allocate

    def record_allocation(
        self: DeviceMemory,
        layout: KVLayout,
        capacity: kvcache.StateCapacity,
        image_rows: tuple[int, int] | None,
        dtype: torch.dtype,
    ) -> tuple[kvcache.InferenceState, torch.Tensor | None]:
        allocated = allocate(self, layout, capacity, image_rows, dtype)
        states.append(allocated[0])
        return allocated

    monkeypatch.setattr(DeviceMemory, "allocate", record_allocation)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    group = SimpleNamespace(
        rank=tp_size - 1,
        size=tp_size,
        is_primary=lambda: True,
        min_max_int=lambda value: (value, value),
    )
    device = torch.device("cpu")
    memory = DeviceMemory(
        plan,  # type: ignore[arg-type]
        device=device,
        agree=group.min_max_int,
        manager=memory_manager,
    )
    host = ModelHost(
        plan,  # type: ignore[arg-type]
        checkpoint,  # type: ignore[arg-type]
        device=device,
        ranks=group,  # type: ignore[arg-type]
        training=training_plan is not None,
    )

    def prepare_training() -> TrainingMemoryPlan:
        events.append("training.prepare")
        return training_plan

    with ExitStack() as resources:
        runner = runner_module.build_executor(
            host,
            memory,
            host.plan,
            ranks=group,  # type: ignore[arg-type]
            preparation_stream=object(),  # type: ignore[arg-type]
            execution_stream=object(),  # type: ignore[arg-type]
            resources=resources,
            beside=None if training_plan is None else prepare_training,
        )
    if training_plan is not None:
        memory.check_training_memory(training_plan)
    (state,) = states
    return _Hybrid(model, memory_manager, host, runner, state)


@pytest.mark.parametrize("tp_size", [1, 2], ids=["tp1", "tp2"])
def test_model_runner_builds_and_provides_hybrid_inference_state(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
) -> None:
    events: list[str] = []
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", _recording_kernel_module(events))
    plan = kvcache.plan_capacity
    monkeypatch.setattr(
        kvcache, "plan_capacity", lambda *a, **k: events.append("plan") or plan(*a, **k)
    )
    model, _, host, runner, state = _construct_hybrid_runner(monkeypatch, events, tp_size=tp_size)

    # The layers' operations exist before the weights; the planner counts the sampler's
    # workspace; the inference state takes the one KV region.
    assert events == [
        "GatedDeltaNetDecode(1, 2, cpu)",
        "GatedDeltaNetPrefill(1, 2, cpu)",
        "weights.load",
        "law",
        "plan",
        "region(kv-cache)",
        "sampler(legacy, torch.bfloat16)",
        "sampler.warm_up",
        "graph.capture",
    ]
    pool = state.recurrent
    assert pool is not None
    assert pool.num_layers == 2
    assert pool.num_slots == 2
    # The pool holds this rank's heads: one key and two value heads per layer.
    assert pool.specs == tuple(layer.state_spec for layer in model.recurrent)
    assert set(pool.specs) == {gated_delta_net_state(2, 128, 128, 4 * 128)}
    (group,) = state.groups
    assert len(group.layers) == 1
    # The resolved family builds the target group's backend over the runner's page table.
    assert host.plan.attention.built == [
        (((group,), runner.page_table), {"max_running_req": 2, "capability": None})
    ]
    # The attention layer is bound to its group's backend, at its index there.
    assert model.attention.bound == {"kv": kvcache.PagedSlot(runner._participants[0], 0)}
    # The recurrent backend is the one state backend, also in the decode graphs.
    backend = runner._recurrent
    assert type(backend) is RecurrentBackend and backend._pool is pool
    graph_kwargs = runner_module.DecodeGraph.captured[0].kwargs
    assert graph_kwargs["participants"] == (runner._participants[0], backend)


@pytest.mark.parametrize("training", [True, False], ids=["training", "serving"])
def test_only_a_training_runner_builds_the_gated_delta_net_training_operation(
    monkeypatch: pytest.MonkeyPatch, training: bool
) -> None:
    events: list[str] = []
    monkeypatch.setitem(
        sys.modules, "ctikernels.gdn", _recording_kernel_module(events, training=training)
    )

    runner = _construct_hybrid_runner(monkeypatch, events, training_enabled=training).runner

    # Built with the serving ones, before the weights load and any residency region.
    built = events[: events.index("weights.load")]
    assert ("GatedDeltaNetTraining(1, 2, cpu)" in built) is training
    assert not any(event.startswith("region(") for event in built)
    # Training never touches the pool: the operation is no graph backend.
    graph_kwargs = runner_module.DecodeGraph.captured[0].kwargs
    assert graph_kwargs["participants"] == runner._participants


def test_a_missing_gated_delta_net_training_operation_fails_before_any_residency_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module(training=False))

    with pytest.raises(ImportError, match="GatedDeltaNetTraining"):
        _construct_hybrid_runner(monkeypatch, events, training_enabled=True)

    assert events == []


def test_missing_gated_delta_net_operations_fail_before_any_residency_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", None)

    with pytest.raises(ImportError, match="ctikernels.gdn"):
        _construct_hybrid_runner(monkeypatch, events)

    # Neither the weights nor any residency region was touched.
    assert events == []


def test_hybrid_model_with_prefix_reuse_allocates_shared_snapshot_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())
    _, _, host, runner, state = _construct_hybrid_runner(monkeypatch, events, prefix_cache="radix")
    assert state.capacity.snapshot_rows == state.recurrent.snapshot_rows == range(2, 4)
    # Startup compiles the shared prefill for every extend forward the runner can run,
    # captures or not: in growing token bounds, the last reaching the full bounds.
    layers = runner.model.recurrent
    plan = host.plan
    bounds = (plan.max_batch_tokens, plan.max_running_requests)
    warm_ups = layers[0]._prefill.warm_ups
    assert warm_ups[-len(layers) :] == [bounds] * len(layers)
    assert all(
        all(step <= limit for step, limit in zip(warm_up, bounds, strict=True))
        for warm_up in warm_ups
    )


def test_a_naive_plan_builds_over_a_recurrent_prefill_that_exports_no_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Resolution plans the naive cache for such a prefill (test_plan); the runner keeps it.
    events: list[str] = []
    module = reference_kernel_module()

    class GatedDeltaNetPrefill(ReferenceGatedDeltaNetPrefill):
        supports_state_capture = False

    module.GatedDeltaNetPrefill = GatedDeltaNetPrefill  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", module)
    _, _, host, _, state = _construct_hybrid_runner(monkeypatch, events)
    assert host.plan.prefix_cache == "naive"
    assert not state.capacity.snapshot_rows


@pytest.mark.parametrize(
    ("max_seq_len", "page_size", "expected_width"),
    [
        pytest.param(1, 1, 32, id="minimum"),
        pytest.param(33, 1, 64, id="existing-32-element-alignment"),
        pytest.param(33, 16, 64, id="partial-16-token-page"),
        pytest.param(33, 32, 64, id="partial-32-token-page"),
        pytest.param(66, 64, 128, id="partial-64-token-page"),
        pytest.param(96, 64, 128, id="32-aligned-but-not-page-aligned"),
    ],
)
def test_token_id_table_width_covers_complete_pages(
    max_seq_len: int,
    page_size: int,
    expected_width: int,
) -> None:
    assert runner_module._token_id_table_width(max_seq_len, page_size) == expected_width


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ((), "at least one entry"),
        (
            (
                _entry(0),
                _entry(1, kv_prefix_len=0, write_position=None, query_kind=QueryKind.EXTEND),
            ),
            "extend entries must precede decode entries",
        ),
    ],
)
def test_forward_rows_need_extends_before_decodes(entries: tuple[Row, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        forward_rows(entries)


@pytest.mark.parametrize(
    ("kinds", "expected"),
    [
        pytest.param("EE", 2, id="extend"),
        pytest.param("EDD", 1, id="mixed"),
        pytest.param("DD", 0, id="decode"),
    ],
)
def test_forward_rows_count_their_leading_extend_entries(kinds: str, expected: int) -> None:
    entries = tuple(
        _entry(i, query_kind=QueryKind.EXTEND if kind == "E" else QueryKind.DECODE)
        for i, kind in enumerate(kinds)
    )
    assert forward_rows(entries).num_extend == expected


def _decode_graph(sizes: tuple[int, ...], dummy_entry: Row) -> DecodeGraph:
    """A decode graph's replay bookkeeping without a capture (CPU)."""

    graph = object.__new__(DecodeGraph)
    graph.batch_sizes = sizes
    graph._buckets = RowBuckets(sizes)
    graph._dummy_entry = dummy_entry
    graph._participants = (SimpleNamespace(prepare_for_replay=lambda _batch, _bs: None),)
    return graph


def test_graph_padding_adds_only_runner_private_dummy_entries() -> None:
    dummy_entry = _entry(request_index=4, kv_prefix_len=0, sequence_len=1, write_position=1)
    graph = _decode_graph((1, 2, 4), dummy_entry)
    entries = (_entry(0), _entry(1), _entry(2))
    rows = graph.rows(entries, ())

    assert rows.slots == (0, 1, 2, 4)
    assert rows.size == 3
    assert rows.padded_size == 4
    assert rows.logit_rows == (0, 1, 2)


@pytest.mark.parametrize(
    ("decode", "breakable", "expected"),
    [
        (True, True, ("decode", GraphStats((), 1, 0, 0))),
        (False, True, ("breakable", GraphStats((), 0, 1, 0))),
        (False, False, ("eager", GraphStats((), 0, 0, 1))),
    ],
)
def test_graphed_forward_takes_the_first_graph_that_replays_the_batch(
    decode: bool, breakable: bool, expected: tuple[str, GraphStats]
) -> None:
    def graph(replays: bool, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            can_replay=lambda _batch: replays,
            replay=lambda *_inputs: name,
            captured=None,
        )

    model = SimpleNamespace(
        forward_hidden=lambda _batch: "eager",
        compute_logits=lambda hidden, _batch: hidden,
    )
    graphs = GraphedForward(
        model,  # type: ignore[arg-type]
        graph(decode, "decode"),  # type: ignore[arg-type]
        graph(breakable, "breakable"),
        None,
    )

    logits = graphs.logits(object(), object())  # type: ignore[arg-type]
    assert (logits, graphs.stats(None)) == expected


@pytest.mark.parametrize(
    "rope_positions",
    [
        pytest.param(torch.tensor([2**31 + 15, 18], dtype=torch.int64), id="rope"),
        pytest.param(torch.tensor([2**31 + 15, -3]), id="offset-decode"),
    ],
)
def test_decode_graph_replays_rope_positions_through_the_captured_view(
    rope_positions: torch.Tensor,
) -> None:
    buffer = _DecodeGraphBuffer.init(4, 2, torch.device("cpu"), torch.float32)
    dummy_entry = _entry(4, kv_prefix_len=0, sequence_len=1, write_position=1)
    # The capture's inputs are views of the buffer, as DecodeGraph captures them.
    capture = ForwardInputs(buffer.input_ids[:2], buffer.rope_positions[:2], None, 2, ragged=False)
    replayed: list[list[int]] = []
    graph_runner = _decode_graph((2, 4), dummy_entry)
    graph_runner._buffer = buffer
    graph_runner._graphs = {
        2: SimpleNamespace(replay=lambda: replayed.append(capture.rope_positions.tolist()))
    }
    rows = forward_rows(
        (
            _entry(0, kv_prefix_len=5, sequence_len=6, write_position=6),
            _entry(1, kv_prefix_len=8, sequence_len=9, write_position=9),
        )
    )
    batch = ForwardInputs(
        torch.tensor([1, 2], dtype=torch.int32), rope_positions, None, 2, ragged=False
    )

    graph_runner.replay(rows, batch)

    assert not hasattr(capture, "positions")
    assert buffer.rope_positions.dtype == rope_positions.dtype == torch.int64
    assert buffer.rope_positions.shape == (4,)
    assert capture.rope_positions.data_ptr() == buffer.rope_positions.data_ptr()
    assert replayed == [rope_positions.tolist()]
    # Columns past the padded batch keep their contents.
    assert not buffer.rope_positions[2:].any()


@pytest.mark.parametrize(
    ("max_running_requests", "graph_ceiling", "expected"),
    (
        (32, None, [1, 2, 3, 4, 8, 16, 24, 32]),
        (10, None, [1, 2, 3, 4, 8, 10]),
        (8, 32, [1, 2, 3, 4, 8]),
        (3, None, [1, 2, 3]),
        (1, None, [1]),
    ),
)
def test_decode_graph_auto_buckets_cover_only_reachable_batches(
    max_running_requests: int,
    graph_ceiling: int | None,
    expected: list[int],
) -> None:
    assert (
        resolve_decode_graph_batch_sizes(
            cuda_graph_bs=None,
            cuda_graph_max_bs=graph_ceiling,
            free_memory=32 << 30,
            max_running_requests=max_running_requests,
        )
        == expected
    )


def test_decode_graph_explicit_buckets_reject_unreachable_batches() -> None:
    with pytest.raises(ValueError, match="running-request capacity"):
        resolve_decode_graph_batch_sizes(
            cuda_graph_bs=[1, 4, 16],
            cuda_graph_max_bs=None,
            free_memory=32 << 30,
            max_running_requests=8,
        )


def test_sampled_token_writes_select_only_writable_entries_in_mixed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_tensor_pinning(monkeypatch)
    entries = (
        _entry(
            0,
            kv_prefix_len=0,
            sequence_len=2,
            write_position=None,
            query_kind=QueryKind.EXTEND,
        ),
        _entry(
            1,
            kv_prefix_len=1,
            sequence_len=2,
            write_position=2,
            query_kind=QueryKind.DECODE,
        ),
    )
    assert forward_rows(entries).num_extend == 1
    writes = runner_module._prepare_sampled_token_writes(entries, torch.device("cpu"))
    assert writes is not None
    assert writes.sample_indices is not None
    assert writes.sample_indices.tolist() == [1]

    token_id_table = torch.full((3, 4), -1, dtype=torch.int32)
    token_id_table[0, -1] = 73
    token_id_table[2].zero_()
    writes.apply(token_id_table, torch.tensor([41, 42], dtype=torch.int32))

    assert token_id_table[0, -1].item() == 73
    assert token_id_table[1, 2].item() == 42
    assert token_id_table[2].tolist() == [0, 0, 0, 0]


def test_sampled_token_writes_skip_an_all_chunked_extend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_tensor_pinning(monkeypatch)
    entries = (
        _entry(
            0, kv_prefix_len=0, sequence_len=2, write_position=None, query_kind=QueryKind.EXTEND
        ),
    )

    assert runner_module._prepare_sampled_token_writes(entries, torch.device("cpu")) is None


def test_sampled_token_writes_keep_decode_on_the_direct_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_tensor_pinning(monkeypatch)
    entries = (
        _entry(1, kv_prefix_len=1, sequence_len=2, write_position=2),
        _entry(0, kv_prefix_len=2, sequence_len=3, write_position=3),
    )
    writes = runner_module._prepare_sampled_token_writes(entries, torch.device("cpu"))
    assert writes is not None
    assert writes.sample_indices is None

    token_id_table = torch.full((2, 4), -1, dtype=torch.int32)
    writes.apply(token_id_table, torch.tensor([51, 52], dtype=torch.int32))

    assert token_id_table[1, 2].item() == 51
    assert token_id_table[0, 3].item() == 52


@pytest.mark.parametrize("rope_position_axes", [1, 3], ids=["rope", "m-rope"])
@pytest.mark.parametrize("recurrent_state", [False, True], ids=["attention-only", "hybrid"])
@pytest.mark.parametrize("query_kind", [QueryKind.DECODE, QueryKind.EXTEND], ids=str)
@pytest.mark.parametrize("tp_rank", [None, 0, 1], ids=["single", "primary", "peer"])
@pytest.mark.parametrize("return_logprobs", [False, True], ids=["tokens", "logprobs"])
def test_execute_batch_prepares_request_free_snapshot_privately(
    monkeypatch: pytest.MonkeyPatch,
    query_kind: QueryKind,
    recurrent_state: bool,
    rope_position_axes: int,
    tp_rank: int | None,
    return_logprobs: bool,
) -> None:
    events: list[str] = []
    # Slot 0's page 0 is page 2 and slot 1's page 2 is page 5.
    pages = PageMap(torch.tensor([[0, 0, 2], [1, 2, 5]], dtype=torch.int32).numpy().tobytes())
    extend = query_kind is QueryKind.EXTEND
    op = Forward(
        (_entry(query_kind=query_kind),),
        torch.tensor([12], dtype=torch.int32).numpy().tobytes() if extend else None,
        pages=pages,
    )
    table = RequestTable()
    table.admit([_spec(0)])
    token_id_table = torch.tensor([[11, 12, 0, 0]], dtype=torch.int32)
    page_table = kvcache.PageTable(
        SimpleNamespace(page_size=2, table_width=4, sink_slot=-1, max_seq_len=8),
        2,
        device=torch.device("cpu"),
    )
    prepare_stream = _Stream("prepare", events)
    execute_stream = _Stream("execute", events)
    current_stream = None

    @contextmanager
    def stream_context(stream: _Stream) -> Iterator[None]:
        nonlocal current_stream
        previous_stream = current_stream
        current_stream = stream
        try:
            yield
        finally:
            current_stream = previous_stream

    input_mapping = (
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )
    sample_indices = torch.tensor([0], dtype=torch.int64)
    sampled_token_writes = runner_module._SampledTokenWrites(
        request_indices=torch.tensor([0], dtype=torch.int64),
        sequence_positions=torch.tensor([2], dtype=torch.int64),
        sample_indices=sample_indices,
    )
    sample_args = SimpleNamespace(logprobs=SimpleNamespace(top=0) if return_logprobs else None)
    metadata = SimpleNamespace()
    forwards: list[tuple[ForwardRows, ForwardInputs]] = []

    class Graphs:
        images = EagerImages()

        def rows(self, entries: tuple[Row, ...], specs: tuple[RequestSpec, ...]) -> ForwardRows:
            events.append("graph.rows")
            return forward_rows(entries, specs)

        def logits(self, rows: ForwardRows, target: ForwardInputs) -> torch.Tensor:
            # Inference operations such as ctikernels.gdn refuse autograd; under inference
            # mode they accept projections whose LoRA factors require grad.
            assert torch.is_inference_mode_enabled()
            assert target.attn_metadata is metadata
            if recurrent_state:
                assert target.recurrent_metadata is recurrent_metadata
            else:
                assert target.recurrent_metadata is None
            assert current_stream is execute_stream
            events.append("graphs.logits")
            forwards.append((rows, target))
            return torch.tensor([[0.0, 1.0]])

    class AttentionBackend:
        def prepare_metadata(self, rows: ForwardRows) -> SimpleNamespace:
            assert page_table.pages.tolist() == [[2, -1, -1, -1], [-1, -1, 5, -1]]
            events.append("attention.prepare")
            metadata.cache_write_indices = write_slots(page_table, rows)
            return metadata

        def prepare_inputs(self, rows: ForwardRows) -> dict[str, object]:
            return {"attn_metadata": self.prepare_metadata(rows)}

    class RecurrentBackendStub:
        def prepare_metadata(self, rows: ForwardRows) -> object:
            assert rows.padded_size == rows.size
            events.append("recurrent.prepare")
            return recurrent_metadata

        def prepare_inputs(self, rows: ForwardRows) -> dict[str, object]:
            return {"recurrent_metadata": self.prepare_metadata(rows)}

        def restore(self, target: ForwardInputs) -> None:
            assert isinstance(target, ForwardInputs)
            assert target.recurrent_metadata is recurrent_metadata
            assert current_stream is execute_stream
            events.append("recurrent.restore")

        def export(self, target: ForwardInputs) -> None:
            # The forward's own inputs, after its logits.
            assert target is forwards[-1][1]
            assert target.recurrent_metadata is recurrent_metadata
            assert current_stream is execute_stream
            events.append("recurrent.export")

    recurrent_metadata = object()

    class Sampler:
        def prepare(self, rows: tuple[Row, ...], specs: tuple[RequestSpec, ...]) -> object:
            assert rows is op.rows and specs == (table[0],)
            events.append("sampler.prepare")
            return sample_args

        def sample(self, logits: torch.Tensor, args: object) -> DeviceSampled:
            assert tp_rank != 1, "only the primary rank consumes sampling randomness"
            events.append("sampler.sample")
            assert logits.shape == (1, 2)
            logprobs = None if args.logprobs is None else torch.tensor([-0.25])
            return DeviceSampled(torch.tensor([42], dtype=torch.int64), logprobs)

    def broadcast_device(tensor: torch.Tensor) -> None:
        assert token_id_table[0, 2].item() == 0, "publish tokens before the token-table write"
        # Tokens and scores travel as one byte buffer: the primary's packed results.
        assert tensor.dtype == torch.uint8
        events.append("broadcast")
        if tp_rank == 1:
            scores = torch.tensor([-0.25]) if return_logprobs else None
            tensor.copy_(DeviceSampled(torch.tensor([42], dtype=torch.int32), scores).packed())

    class CompletionEvent:
        def record(self, stream: _Stream) -> None:
            assert stream is execute_stream
            assert token_id_table[0, 2].item() == 42
            events.append("completion.record")

    recurrent = RecurrentBackendStub() if recurrent_state else None
    parts = ExecutorParts(
        SimpleNamespace(rope_position_axes=lambda: rope_position_axes),
        prepare_stream,
        execute_stream,
        page_table,
        token_id_table,
        Sampler(),
        (AttentionBackend(), *(() if recurrent is None else (recurrent,))),
        recurrent,
        Graphs(),
        None,
        None,
        SimpleNamespace(rounds=None),
        1,
    )
    ranks = SimpleNamespace(
        size=1 if tp_rank is None else 2,
        is_primary=lambda: tp_rank != 1,
        broadcast_device=broadcast_device,
    )
    runner = InferenceRunner(parts, ranks=ranks, checks=True)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "Event", CompletionEvent)
    _disable_tensor_pinning(monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "_make_positions",
        lambda _batch, _device: input_mapping[1],
    )
    monkeypatch.setattr(
        runner_module,
        "_make_input_tuple",
        lambda _rows, _positions, _device: input_mapping,
    )
    monkeypatch.setattr(
        runner_module,
        "_prepare_sampled_token_writes",
        lambda _batch, _device: sampled_token_writes,
    )

    result = runner.submit(op, table)

    logprobs = torch.tensor([-0.25]).numpy().tobytes() if return_logprobs else None
    assert result.output() == Sampled(
        torch.tensor([42], dtype=torch.int32).numpy().tobytes(), logprobs
    )
    inputs = result._inputs
    ((rows, batch),) = forwards
    assert isinstance(inputs.rows, ForwardRows) and rows is inputs.rows
    assert inputs.entries is op.rows
    assert batch.logit_rows.tolist() == [0]
    assert inputs.input_mapping is input_mapping
    assert inputs.sampled_token_writes is sampled_token_writes
    assert inputs.sampled_token_writes.sample_indices is sample_indices
    assert inputs.sampling_args is sample_args
    assert isinstance(batch, ForwardInputs)
    assert batch.input_ids.tolist() == [12]
    assert (batch.size, batch.ragged) == (1, extend)
    assert batch.attn_metadata.cache_write_indices.tolist() == [5]
    assert inputs.positions.dtype == torch.int64
    assert inputs.rope_positions.dtype == torch.int64
    assert inputs.input_mapping[1] is inputs.positions
    # Every axis of a text step, M-RoPE included, reads the sequence positions themselves.
    assert inputs.rope_positions is inputs.positions
    assert batch.rope_positions is inputs.positions
    assert batch.attn_metadata is metadata
    _assert_request_free(result)
    assert events == [
        "graph.rows",
        "attention.prepare",
        *(("recurrent.prepare",) if recurrent_state else ()),
        "sampler.prepare",
        "execute.wait_stream:prepare",
        *(("recurrent.restore",) if recurrent_state else ()),
        "graphs.logits",
        # Decode exports copy the rows the forward, replayed or not, advanced.
        *(("recurrent.export",) if recurrent_state else ()),
        *(["sampler.sample"] if tp_rank != 1 else []),
        # One collective under TP; a single rank broadcasts nothing.
        *(["broadcast"] if tp_rank is not None else []),
        "completion.record",
    ]


@pytest.mark.parametrize("prefix", [0, 7, 2**31])
@pytest.mark.parametrize("query_kind", [QueryKind.EXTEND, QueryKind.DECODE])
def test_batch_positions_preserve_int64_ranges_and_padding(
    monkeypatch: pytest.MonkeyPatch,
    prefix: int,
    query_kind: QueryKind,
) -> None:
    _disable_tensor_pinning(monkeypatch)
    length = 3 if query_kind is QueryKind.EXTEND else 1
    entry = _entry(
        1,
        kv_prefix_len=prefix,
        sequence_len=prefix + length,
        write_position=prefix + length,
        query_kind=query_kind,
    )
    rows = forward_rows(
        (entry,), (), (_entry(2, kv_prefix_len=0, sequence_len=1, write_position=1),)
    )

    positions = runner_module._make_positions(rows, torch.device("cpu"))
    request_indices, sequence_positions = runner_module._make_input_tuple(
        rows, positions, torch.device("cpu")
    )

    assert positions.dtype == torch.int64
    assert positions.tolist() == [*range(prefix, prefix + length), 0]
    assert request_indices.tolist() == [1] * length + [2]
    assert sequence_positions is positions


def test_input_mapping_indexes_tables_by_sequence_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_tensor_pinning(monkeypatch)
    rows = forward_rows(
        (
            _entry(
                1,
                kv_prefix_len=1,
                sequence_len=4,
                write_position=4,
                query_kind=QueryKind.EXTEND,
            ),
        )
    )
    positions = torch.tensor([1, 2, 3], dtype=torch.int64)

    request_indices, sequence_positions = runner_module._make_input_tuple(
        rows, positions, torch.device("cpu")
    )

    assert request_indices.tolist() == [1, 1, 1]
    assert sequence_positions.tolist() == [1, 2, 3]
    assert sequence_positions is positions
    assert sequence_positions.dtype == torch.int64


def test_apply_extend_token_ids_preserves_packed_order() -> None:
    tokens = torch.tensor([12, 13, 21], dtype=torch.int32)
    input_mapping = (
        torch.tensor([0, 0, 1, 2], dtype=torch.int64),
        torch.tensor([1, 2, 0, 1], dtype=torch.int64),
    )

    token_id_table = torch.full((3, 4), -1, dtype=torch.int32)
    token_id_table[2, 1] = 31

    runner_module._apply_extend_token_ids(token_id_table, tokens, input_mapping)

    assert token_id_table.tolist() == [
        [-1, 12, 13, -1],
        [21, -1, -1, -1],
        [-1, 31, -1, -1],
    ]


@pytest.mark.parametrize(
    ("peak_memory_reserve_bytes", "num_pages"),
    [
        # 480 available bytes less 5 x 2 x 6 = 60 recurrent: 420 bytes, 105 pages of 4,
        # one of them the graph sink page.
        pytest.param(100, 104, id="reserve-holds-it"),
        pytest.param(150, 104, id="reserve-exceeds-it"),
        # The reserve holds 40 of the 100 bytes; 60 come out of the 420: 90 pages.
        pytest.param(40, 89, id="shortfall"),
        # Without a reserve all 100 bytes do: 80 pages.
        pytest.param(0, 79, id="no-reserve"),
    ],
)
def test_kv_capacity_yields_only_what_the_peak_reserve_cannot_hold_of_the_image_encoder(
    monkeypatch: pytest.MonkeyPatch,
    peak_memory_reserve_bytes: int,
    num_pages: int,
) -> None:
    memory = object.__new__(DeviceMemory)
    memory._device = torch.device("cpu")
    memory._agree = lambda value: (value, value)
    memory.budget_bytes = 480
    memory.reserve_bytes = peak_memory_reserve_bytes
    memory._boot_local_bytes = 0
    memory._recurrent_to_kv_ratio = _RECURRENT_TO_KV_RATIO
    _free_memory(monkeypatch, 0)
    # One paged layer; the attention workspace takes nothing.
    plan = memory.plan(
        _kv_layout(1, rows=_TINY_ROWS),
        geometry=kvcache.TableGeometry(1, 5, 64),
        snapshots=None,
        workspace_bytes=0,
        image_encoder_bytes=100,
        image_rows_bytes=0,
        beside_bytes=0,
        allocations=None,
        fit_slots=False,
    )

    assert _pages_and_rows(plan) == (num_pages, 0)


class _CudaMemory:
    """torch.cuda's allocator statistics on CPU: an allocation counter and its peak."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.allocated = 1_000
        self.peak = 1_000
        self.calls: list[str] = []
        monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", self._reset)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: self.allocated)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: self.peak)

    def _reset(self, _device: torch.device) -> None:
        self.calls.append("reset")
        self.peak = self.allocated


@pytest.mark.parametrize("encodes_images", [True, False], ids=["image-encoder", "text-only"])
def test_image_encoder_measurement_takes_the_peak_of_the_largest_step(
    monkeypatch: pytest.MonkeyPatch,
    encodes_images: bool,
) -> None:
    memory = _CudaMemory(monkeypatch)
    warm_ups: list[dict[str, object]] = []

    def warm_up_image_encoder(**bounds: object) -> bool:
        assert torch.is_inference_mode_enabled()
        warm_ups.append(bounds)
        memory.peak += 300 if encodes_images else 0
        return encodes_images

    device = torch.device("cpu")
    model = SimpleNamespace(warm_up_image_encoder=warm_up_image_encoder)
    plan = SimpleNamespace(
        encoder_rows=128, image_limits=ImageLimits(max_images=1, max_image_tokens=96)
    )
    device_memory = object.__new__(DeviceMemory)
    device_memory._device, device_memory._agree = device, _Group().min_max_int

    measured = runner_module._image_encoder_peak(model, plan, device_memory, device)

    assert warm_ups == [{"max_tokens": 128, "max_image_tokens": 96, "device": device}]
    assert memory.calls == ["reset"]
    # The largest rank's peak: _Group reports the local value plus 10 as the maximum.
    assert measured == (310 if encodes_images else 0)


def test_no_image_encoder_runs_when_no_image_fits_a_step(monkeypatch: pytest.MonkeyPatch) -> None:
    model = SimpleNamespace(warm_up_image_encoder=Mock())
    plan = SimpleNamespace(
        encoder_rows=8, image_limits=ImageLimits(max_images=1, max_image_tokens=0)
    )

    measured = runner_module._image_encoder_peak(model, plan, Mock(), torch.device("cpu"))

    assert measured == 0
    model.warm_up_image_encoder.assert_not_called()


def _rope_positions(axes: int, *args: object) -> torch.Tensor:
    """A step's rope positions for a model whose rotary reads ``axes`` axes."""

    return runner_module._make_rope_positions(axes, *args, torch.device("cpu"))


def _prepared(
    entries: tuple[Row, ...], padding: int = 0, offsets: dict[int, int] | None = None
) -> tuple[ForwardRows, torch.Tensor]:
    """``entries`` padded with ``padding`` dummies, and their positions; ``offsets`` maps a
    slot to its rope offset."""

    offsets = offsets or {}
    specs = tuple(_spec(e.request_index, offsets.get(e.request_index, 0)) for e in entries)
    dummy = _entry(9, kv_prefix_len=0, sequence_len=1, write_position=1)
    rows = forward_rows(entries, specs, (dummy,) * padding)
    return rows, runner_module._make_positions(rows, torch.device("cpu"))


def test_rope_rows_take_layout_rows_for_prompts_and_offsets_for_decodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_tensor_pinning(monkeypatch)
    entries = (
        _entry(
            0, kv_prefix_len=2, sequence_len=5, write_position=None, query_kind=QueryKind.EXTEND
        ),
        _entry(1, kv_prefix_len=0, sequence_len=2, write_position=2, query_kind=QueryKind.EXTEND),
        _entry(2, kv_prefix_len=7, sequence_len=8, write_position=8),
        _entry(3, kv_prefix_len=4, sequence_len=5, write_position=5),
    )
    layout_rows = torch.tensor([[2, 2, 2, 0, 1], [2, 3, 4, 0, 1], [2, 5, 6, 0, 1]])
    # A prompt's request offset reaches only its decodes.
    rows, positions = _prepared(entries, padding=2, offsets={0: 4, 3: -3})

    rope_positions = _rope_positions(3, layout_rows, rows, positions)

    # Frozen prompt rows, a decode at its position, one offset by -3, then two padding rows.
    assert rope_positions.tolist() == [row + [7, 1, 0, 0] for row in layout_rows.tolist()]


def test_a_text_prompt_beside_an_offset_decode_reads_its_sequence_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_tensor_pinning(monkeypatch)
    entries = (
        _entry(0, kv_prefix_len=1, sequence_len=3, write_position=3, query_kind=QueryKind.EXTEND),
        _entry(1, kv_prefix_len=6, sequence_len=7, write_position=7),
    )
    rows, positions = _prepared(entries, offsets={1: 5})

    rope_positions = _rope_positions(3, None, rows, positions)

    # One row that every axis reads, as without the offset.
    assert rope_positions.tolist() == [1, 2, 11]
    with pytest.raises(ValueError, match="reads 1 position axes"):
        _rope_positions(1, None, rows, positions)


@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("offsets", [(-3,), (5, 0, -3, 2**31)])
def test_decode_rope_offsets_preserve_int64_positions_and_padding(
    monkeypatch: pytest.MonkeyPatch, padding: int, offsets: tuple[int, ...]
) -> None:
    _disable_tensor_pinning(monkeypatch)
    entries = tuple(_entry(i) for i in range(len(offsets)))
    rows, positions = _prepared(entries, padding=padding, offsets=dict(enumerate(offsets)))

    rope_positions = _rope_positions(3, None, rows, positions)

    assert rope_positions.dtype == torch.int64
    assert rope_positions.tolist() == [1 + offset for offset in offsets] + [0] * padding


def test_layout_rows_must_match_the_model_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_tensor_pinning(monkeypatch)
    entries = (
        _entry(0, kv_prefix_len=0, sequence_len=2, write_position=2, query_kind=QueryKind.EXTEND),
    )
    rows, positions = _prepared(entries)

    with pytest.raises(ValueError, match="reads 3 position axes"):
        _rope_positions(3, torch.zeros((2, 2), dtype=torch.int64), rows, positions)


def test_hybrid_budget_counts_snapshots_before_kv():
    plan = _plan(480, slots=5, rows=_TINY_ROWS, snapshots=SnapshotPolicy(rows=2))
    assert len(plan.snapshot_rows) == 2
    pages = plan.num_pages
    # 480 total: 60 request bytes, 24 snapshot bytes, 5 slots' packed copy indices and KV
    # (sink included); row copies take no workspace.
    assert (pages + 1) * 4 + 60 + 24 + 4 * 5 * 8 == 480


def test_recurrent_radix_rejects_snapshot_rows_beyond_the_budget_before_allocation(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())
    with pytest.raises(ValueError, match="needs .* and 1000 snapshot rows"):
        _construct_hybrid_runner(monkeypatch, events, prefix_cache="radix", snapshot_rows=1000)
    assert not any(event.startswith("region(") for event in events)


def test_capacity_planning_is_repeatable_and_immutable():
    from dataclasses import FrozenInstanceError

    plan = _plan(480, slots=5)
    assert plan == _plan(480, slots=5)
    assert _pages_and_rows(plan) == (119, 0)
    with pytest.raises(FrozenInstanceError):
        plan.num_pages = 1


def test_under_checks_a_forward_past_the_warmed_rows_is_refused() -> None:
    entries = (
        _entry(0, kv_prefix_len=0, sequence_len=3, query_kind=QueryKind.EXTEND),
        _entry(1, kv_prefix_len=2, sequence_len=6, query_kind=QueryKind.EXTEND),
    )

    runner_module._require_warmed_rows(8, forward_rows(entries))
    runner_module._require_warmed_rows(8, forward_rows(entries[:1]), round_rows=5)
    runner_module._require_warmed_rows(8, None, round_rows=8)
    with pytest.raises(RuntimeError, match="a forward of 9 rows exceeds the 8 every op"):
        runner_module._require_warmed_rows(8, forward_rows(entries), round_rows=2)


@pytest.mark.parametrize(
    ("sizes", "rows", "bucket", "padding"),
    [((1, 2, 4), 3, 4, 1), ((1, 2, 3, 4), 3, 3, 1), ((4, 8), 1, 4, 3), ((1, 2), 3, None, 1)],
)
def test_a_batch_takes_the_smallest_bucket_that_holds_it(
    sizes: tuple[int, ...], rows: int, bucket: int | None, padding: int
) -> None:
    buckets = RowBuckets(sizes)

    assert buckets.select(rows) == bucket
    # The dummy rows past the slots for rounds of at most 4 requests: at least the one decode
    # graphs repeat, at most the bound allocated before the sizes are known.
    assert buckets.padding_rows(4) == padding <= RowBuckets.padding_rows_bound(4)


def test_the_default_breakable_ladder_steps_4_then_16_then_32_up_to_512() -> None:
    ladder = RowBuckets.for_limits(8192, None).sizes

    assert ladder == (
        *range(4, 33, 4),
        *range(48, 257, 16),
        *range(288, 513, 32),
    )
    # A forward's row limit drops the buckets it cannot fill; an override replaces the ladder.
    assert RowBuckets.for_limits(100, None).sizes[-1] == 96
    assert RowBuckets.for_limits(8192, (16, 32)).sizes == (16, 32)


@pytest.mark.parametrize(("scores", "top"), [(False, 0), (True, 0), (True, 3)])
def test_sampled_results_round_trip_through_one_byte_buffer(scores: bool, top: int) -> None:
    # A TP peer receives the primary's tokens, scores and alternatives in one collective.
    rows = 4
    sent = DeviceSampled(
        torch.tensor([7, -1, 2**31 - 1, 0], dtype=torch.int32),
        torch.tensor([-0.5, -1e-30, float("-inf"), -3.25]) if scores else None,
        torch.arange(rows * top, dtype=torch.int32).view(rows, top) if top else None,
        torch.linspace(-9.0, -0.5, rows * top).view(rows, top) if top else None,
    )
    received, buffer = DeviceSampled.receiving(
        rows, logprobs=scores, top=top, device=torch.device("cpu")
    )

    buffer.copy_(sent.packed())

    for got, want in zip(
        (received.token_ids, received.logprobs, received.top_ids, received.top_logprobs),
        (sent.token_ids, sent.logprobs, sent.top_ids, sent.top_logprobs),
        strict=True,
    ):
        assert (got is None) == (want is None)
        if want is not None:
            assert got.dtype == want.dtype and got.shape == want.shape
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))


def test_a_training_host_refuses_an_untrainable_checkpoint_before_building_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cti.models.weight import CheckpointWeights
    from tests.support import quant_checkpoints
    from tests.support.tp_shards import DryGroup

    quant_checkpoints.write_headers_only("qwen3_8_27b_nvfp4", tmp_path)

    def build(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the model was built")

    monkeypatch.setattr(
        model_host_module, "open_weights", lambda *_: CheckpointWeights(str(tmp_path))
    )
    monkeypatch.setattr(model_host_module, "create_model", build)
    plan = SimpleNamespace(lora=None, experts=ExpertParallel(), speculative=None, weights=None)
    checkpoint = SimpleNamespace(facts=SimpleNamespace(source=None))
    with pytest.raises(ValueError, match=r"^modelopt_mixed checkpoints serve .* do not train$"):
        model_host_module.ModelHost(
            plan, checkpoint, device=torch.device("cpu"), ranks=DryGroup(0, 1), training=True
        )
