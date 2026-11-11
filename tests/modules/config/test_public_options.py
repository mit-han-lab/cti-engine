"""Every public option has one definition that Python, launch files and the CLI all reach."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import get_args, get_type_hints

import pytest
import torch
import yaml

import cti.resolution as resolution_module
from cti.attention import AttentionRequest
from cti.config import (
    ActivationCheckpointMode,
    AdamWConfig,
    BatchLayout,
    CacheOptions,
    DynamicBatching,
    EngineOptions,
    EngineTuning,
    LaunchFile,
    LoRAConfig,
    MemoryOptions,
    ModelOptions,
    ParallelOptions,
    RandomWeights,
    SchedulingOptions,
    ServerOptions,
    SFTShapeProfile,
    SpeculativeOptions,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
    TrainingPauseBoundary,
)
from cti.kvcache import GroupRole, SnapshotPolicy
from cti.layers import ExpertParallel, ExpertTensorParallel
from cti.models import CheckpointFacts, CheckpointSource, ModelCapability
from cti.resolution import ResolvedModel
from cti.server.args import parse_args
from cti.utils.dataclass_tree import DataclassTree
from tests.support.paths import REPO_ROOT

_MODEL = {"model": {"path": "model"}}
# One non-default document value per leaf the command line spells; a new leaf adds one.
_SAMPLES = {
    "model.path": "other",
    "model.source": "modelscope",
    "model.dtype": "auto",
    "model.weights": {"kind": "random_weights", "seed": 7},
    "model.max_seq_len": 4096,
    "model.initialization_seed": 0,
    "model.text_only": True,
    "parallel.num_ranks": 2,
    "parallel.enable_expert_parallel": False,
    "parallel.num_nodes": 4,
    "parallel.node_rank": 1,
    "parallel.dist_init_addr": "node1.example:29501",
    "memory.inference_ratio": 0.5,
    "memory.peak_ratio": 0.95,
    "memory.recurrent_to_kv_ratio": 0.5,
    "scheduling.max_running_requests": 8,
    "scheduling.max_batch_tokens": 512,
    "scheduling.max_prompt_tokens": 256,
    "cache.prefix": "naive",
    "speculative.model_path": "draft",
    "speculative.num_speculative_tokens": 15,
    "lora.targets": ["attention", "mlp"],
    "lora.rank": 4,
    "lora.alpha": 8.0,
    "lora.seed": 3,
    "lora.adapter": "adapter",
    "lora.expert_layout": "per_projection",
    "server.host": "0.0.0.0",
    "server.port": 8000,
    "server.num_tokenizers": 2,
    "server.allowed_media_domains": ["example.com"],
    "server.allowed_local_media_path": "/",
    "server.media_fetch_max_retries": 1,
    "server.tool_call_parser": "hermes",
}
_CLI_LEAVES = DataclassTree(LaunchFile, exclude=("training",)).leaves


# A speculative method that runs beside every feature and attention backend.
_ROUNDS = object()
_METHOD = SimpleNamespace(
    capabilities=~ModelCapability(0),
    attention_requirements=lambda: (),
    round_shape=lambda **_limits: _ROUNDS,
    check=lambda *, max_seq_len: None,
)


def _base(dotted: str) -> dict:
    """The smallest document the sample at ``dotted`` completes."""

    if dotted.startswith("lora."):
        return {**_MODEL, "lora": {"targets": ["attention"]}}
    if dotted.startswith("speculative.") and dotted != "speculative.model_path":
        return {**_MODEL, "speculative": {"model_path": "draft"}}
    if dotted in ("parallel.num_nodes", "parallel.node_rank", "parallel.dist_init_addr"):
        nodes = {"num_ranks": 4, "num_nodes": 2, "dist_init_addr": "node0.example:29500"}
        return {**_MODEL, "parallel": nodes}
    return _MODEL


def _read(launch: LaunchFile, dotted: str) -> object:
    section, name = dotted.split(".")
    owner = launch.server if section == "server" else getattr(launch.engine, section)
    return None if owner is None else getattr(owner, name)


def _flags(document: dict) -> list[str]:
    flags = []
    for section, fields in document.items():
        for name, value in fields.items():
            text = (
                value if isinstance(value, str) else yaml.safe_dump(value, default_flow_style=True)
            )
            flags += [f"--{section}.{name.replace('_', '-')}", text.strip()]
    return flags


def test_every_command_line_leaf_has_a_sample() -> None:
    assert [leaf.dest for leaf in _CLI_LEAVES] == list(_SAMPLES)


def test_engine_tuning_never_enters_a_launch_file_or_the_cli() -> None:
    """``EngineTuning`` is set only from Python: no launch-file section holds one, and no
    option a launch file or the command line spells shares one of its field names."""

    def types(hint: object) -> Iterator[object]:
        yield hint
        for arg in get_args(hint):
            yield from types(arg)

    seen: set[type] = set()
    pending: list[type] = [LaunchFile]
    while pending:
        if (section := pending.pop()) in seen:
            continue
        seen.add(section)
        for hint in get_type_hints(section).values():
            pending += [kind for kind in types(hint) if dataclasses.is_dataclass(kind)]
    assert EngineTuning not in seen
    tuning = {field.name for field in dataclasses.fields(EngineTuning)}
    assert not tuning & {leaf.dest.rsplit(".", 1)[-1] for leaf in DataclassTree(LaunchFile).leaves}


@pytest.mark.parametrize("dotted", list(_SAMPLES))
def test_launch_files_and_the_cli_build_the_same_options(dotted: str) -> None:
    section, name = dotted.split(".")
    base = _base(dotted)
    document = {**base, section: {**base.get(section, {}), name: _SAMPLES[dotted]}}

    launch = LaunchFile.parse(document)

    assert parse_args(_flags(document)) == (launch.engine, launch.server)
    assert _read(launch, dotted) != _read(LaunchFile.parse(_base(dotted)), dotted)


@pytest.mark.parametrize(
    "sections",
    [
        pytest.param({"parallel": ParallelOptions(num_ranks=2)}, id="tensor-parallel"),
        pytest.param(
            {"parallel": ParallelOptions(num_ranks=2, enable_expert_parallel=False)},
            id="expert-tensor-parallel",
        ),
        pytest.param({"speculative": SpeculativeOptions(model_path="draft")}, id="speculative"),
        pytest.param({"lora": LoRAConfig(targets=("attention",))}, id="lora"),
    ],
)
def test_every_option_reaches_the_resolved_engine(
    monkeypatch: pytest.MonkeyPatch, sections: dict[str, object]
) -> None:
    # The request check refuses LoRA with tensor parallelism.
    options = EngineOptions(
        model=ModelOptions(
            path="model",
            dtype="bfloat16",
            weights=RandomWeights(seed=7),
            max_seq_len=4096,
            initialization_seed=0,
            text_only=True,
        ),
        memory=MemoryOptions(inference_ratio=0.5, peak_ratio=0.95),
        scheduling=SchedulingOptions(max_running_requests=8, max_batch_tokens=512),
        cache=CacheOptions(prefix="radix"),
        **sections,
    )
    tuning = EngineTuning(
        page_size=16,
        cuda_graph_bs=(1, 4),
        cuda_graph_max_bs=4,
        snapshot_rows=4,
        use_pynccl=False,
        distributed_timeout=30.0,
    )
    attention = SimpleNamespace(selected="fi", pages=(), layouts={})
    requests = []
    monkeypatch.setattr(
        resolution_module,
        "resolve_attention_config",
        lambda _requirements, request: requests.append(request) or attention,
    )
    model = ResolvedModel(
        facts=CheckpointFacts(
            source=CheckpointSource("model"),
            hf_config=None,
            document={},
            stop_token_ids=(),
            image_preprocessing=None,
        ),
        model_config=SimpleNamespace(architectures=["Model"], max_position=8192),
        model_class=SimpleNamespace(
            CAPABILITIES=~ModelCapability(0),
            LORA_TARGETS=("attention",),
            RECURRENT_EXPORT_GRANULARITY=None,
            attention_requirements=lambda _hf_config: (),
            exports_prefill_states=lambda _hf_config, _device, *, num_ranks: True,
            require_capabilities=lambda _requested, **_names: None,
        ),
    )
    method = None if options.speculative is None else _METHOD

    plan = model.plan(
        model.check(options, tuning, speculative=method),
        device=torch.device("cpu"),
        capability=None,
        device_memory=32 << 30,
        online=False,
    )

    speculative = options.speculative is not None
    assert plan.speculative is method
    assert (plan.model_path, plan.dtype) == ("model", torch.bfloat16)
    assert plan.weights == RandomWeights(seed=7)
    assert (plan.max_seq_len, plan.initialization_seed) == (4096, 0)
    expert_parallel = options.parallel.enable_expert_parallel
    assert (plan.num_ranks, plan.experts, plan.lora) == (
        options.parallel.num_ranks,
        ExpertParallel() if expert_parallel else ExpertTensorParallel(),
        options.lora,
    )
    assert (plan.inference_ratio, plan.peak_ratio, plan.max_batch_tokens) == (0.5, 0.95, 512)
    assert (plan.graph_batch_sizes, plan.graph_max_batch) == ((1, 4), 4)
    assert (plan.use_pynccl, plan.distributed_timeout) == (False, 30.0)
    assert plan.encoder_rows == 512
    assert (plan.prefix_cache, plan.page_size) == ("radix", 16)
    assert plan.seeds_every_row is speculative
    assert plan.snapshots == SnapshotPolicy(4, None)
    assert plan.rounds is (_ROUNDS if speculative else None)
    assert plan.attention == attention
    # The family resolves for the tuning's backend, page size and tensor-core choice.
    assert requests == [
        AttentionRequest(
            target=GroupRole.TARGET,
            backend=tuning.attention_backend,
            page_size=16,
            capability=None,
            tensor_parallel_size=options.parallel.num_ranks,
            dtype=torch.bfloat16,
            rounds=plan.rounds,
            tensor_cores=None,
        )
    ]


_TRAINING = """
model: {path: model}
training:
  trainable_plan: {selected_names: [adapter.down, adapter.up]}
  optimizer: {lr: 0.001, betas: [0.8, 0.9], eps: 1.0e-7, weight_decay: 0.01, max_grad_norm: null}
  batching:
    layout: packed
    policy: {kind: dynamic_batching, max_tokens_per_microbatch: 512}
  activation_checkpointing: full
  training_pause_boundary: microbatch
  sft_profile: {sequence_lengths: [8, 12], padded_width: 16, num_target_tokens: 12}
"""


def test_a_launch_file_spells_the_training_section() -> None:
    launch = LaunchFile.parse(yaml.safe_load(_TRAINING))

    assert launch.engine.training == TrainingConfig(
        trainable_plan=TrainablePlan(("adapter.down", "adapter.up")),
        optimizer=AdamWConfig(
            lr=0.001, betas=(0.8, 0.9), eps=1.0e-7, weight_decay=0.01, max_grad_norm=None
        ),
        batching=TrainingBatching(
            BatchLayout.PACKED, DynamicBatching(max_tokens_per_microbatch=512)
        ),
        activation_checkpointing=ActivationCheckpointMode.FULL,
        training_pause_boundary=TrainingPauseBoundary.MICROBATCH,
        sft_profile=SFTShapeProfile(
            sequence_lengths=(8, 12), padded_width=16, num_target_tokens=12
        ),
    )
    minimal = """
model: {path: model}
training:
  trainable_plan: {}
  optimizer: {}
  batching: {layout: padded, policy: {kind: static_batching, microbatch_size: 1, max_tokens_per_sample: 8}}
"""
    training = LaunchFile.parse(yaml.safe_load(minimal)).engine.training
    assert training.trainable_plan == TrainablePlan.all_declared()
    # Serving waits for one microbatch, not a whole logical batch.
    assert training.training_pause_boundary is TrainingPauseBoundary.MICROBATCH


def test_the_cli_runs_no_training(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(_TRAINING, encoding="utf-8")

    with pytest.raises(SystemExit):
        parse_args(["--config", str(path)])
    assert "CLI serving does not run training" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parse_args(["--model.path", "model", "--training.optimizer.lr", "0.1"])


def test_the_cli_refuses_an_unknown_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model.path", "model", "--model.residual-dtype", "float32"])
    assert "unrecognized arguments: --model.residual-dtype float32" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("document", "message"),
    (
        ({"model": {"path": ""}}, "model: path must not be empty"),
        ({"model": {"path": "m", "dtype": "float64"}}, "model.dtype must be one of"),
        ({"model": {"path": "m", "dtype": "float32"}}, "model.dtype must be one of"),
        ({"model": {"path": "m", "residual_dtype": "float32"}}, "model has unknown keys"),
        ({"model": {"path": "m", "max_seq_len": 0}}, "max_seq_len must be positive"),
        ({"model": {"path": "m", "initialization_seed": -1}}, "initialization_seed must be"),
        ({"model": {"path": "m", "weights": {"kind": "random_weights", "seed": -1}}}, "seed"),
        ({**_MODEL, "parallel": {"num_ranks": 0}}, "num_ranks must be positive"),
        (
            {**_MODEL, "parallel": {"num_ranks": 3, "num_nodes": 2, "dist_init_addr": "a:1"}},
            "num_nodes must be positive and divide num_ranks",
        ),
        (
            {**_MODEL, "parallel": {"num_ranks": 2, "num_nodes": 2, "node_rank": 2}},
            "node_rank must be in [0, num_nodes)",
        ),
        (
            {**_MODEL, "parallel": {"num_ranks": 2, "num_nodes": 2}},
            "dist_init_addr must be host:port when num_nodes > 1",
        ),
        ({**_MODEL, "memory": {"inference_ratio": 0}}, "inference_ratio must be in (0, 1]"),
        (
            {**_MODEL, "memory": {"inference_ratio": 0.6, "peak_ratio": 0.5}},
            "peak_ratio must be in [inference_ratio, 1]",
        ),
        ({**_MODEL, "memory": {"peak_ratio": 0}}, "peak_ratio must be in [inference_ratio, 1]"),
        (
            {**_MODEL, "memory": {"peak_ratio": float("nan")}},
            "peak_ratio must be in [inference_ratio, 1]",
        ),
        (
            {**_MODEL, "memory": {"recurrent_to_kv_ratio": 0}},
            "recurrent_to_kv_ratio must be a positive finite number",
        ),
        ({**_MODEL, "scheduling": {"max_running_requests": 0}}, "must be positive"),
        (
            {**_MODEL, "scheduling": {"max_running_requests": 256, "max_batch_tokens": 128}},
            "max_batch_tokens must be at least max_running_requests",
        ),
        ({**_MODEL, "cache": {"prefix": "paged"}}, "cache.prefix must be one of"),
        ({**_MODEL, "speculative": {}}, "speculative.model_path is required"),
        (
            {**_MODEL, "lora": {"targets": ["attention"], "rank": 0}},
            "LoRA rank must be in [1, 128]",
        ),
        # Every LoRA takes ranks up to 128.
        (
            {**_MODEL, "lora": {"targets": ["attention"], "rank": 129}},
            "LoRA rank must be in [1, 128], not 129",
        ),
        ({**_MODEL, "server": {"port": 65536}}, "port must be in [0, 65535]"),
    ),
)
def test_a_section_rejects_its_invalid_values(document: dict, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        LaunchFile.parse(document)


def test_python_sections_reject_their_invalid_values() -> None:
    # An unset inference ratio defers the peak's lower bound to the device's default.
    assert MemoryOptions(peak_ratio=0.5).inference_ratio is None
    with pytest.raises(ValueError, match=re.escape("peak_ratio must be in [inference_ratio, 1]")):
        MemoryOptions(inference_ratio=0.6, peak_ratio=0.5)
    assert MemoryOptions(inference_ratio=0.5, peak_ratio=0.5).peak_ratio == 0.5
    with pytest.raises(ValueError, match="host must not be empty"):
        ServerOptions(host="")
    with pytest.raises(ValueError, match="cuda_graph_bs must hold positive sizes"):
        EngineTuning(cuda_graph_bs=(1, 0))
    assert EngineTuning(cuda_graph_max_bs=0).cuda_graph_max_bs == 0
    with pytest.raises(ValueError, match="cuda_graph_max_bs must be non-negative"):
        EngineTuning(cuda_graph_max_bs=-1)
    with pytest.raises(ValueError, match="distributed_timeout must be positive"):
        EngineTuning(distributed_timeout=0.0)


def test_every_option_path_is_documented() -> None:
    document = (REPO_ROOT / "docs/configuration.md").read_text(encoding="utf-8")

    missing = [
        leaf.dest for leaf in DataclassTree(LaunchFile).leaves if f"`{leaf.dest}`" not in document
    ]

    assert not missing, f"docs/configuration.md does not list: {missing}"
