from __future__ import annotations

import inspect
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
import torch
from transformers import (
    PretrainedConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3MoeConfig,
)

from cti.attention.flashinfer import FlashInferBackend
from cti.config import (
    AdamWConfig,
    BatchLayout,
    CacheOptions,
    EngineOptions,
    EngineTuning,
    LoRAConfig,
    LoRATarget,
    ModelOptions,
    ParallelOptions,
    SpeculativeOptions,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.layers import Adapter
from cti.models import ModelCapability, get_model_class
from cti.models.qwen3_moe.model import Qwen3MoeForCausalLM
from cti.models.register import MODELS
from cti.models.training_contract import TrainingModel
from cti.plan import Plan, SpeculativeMethod
from cti.resolution import CheckedOptions, ResolvedModel
from cti.utils import DeviceCapability
from tests.support.attention import set_validation
from tests.support.gated_delta_net import reference_kernel_module
from tests.support.tiny_configs import tiny_hf_config
from tests.support.tp_shards import build_model

_QWEN3 = "Qwen3ForCausalLM"
_QWEN3_MOE = "Qwen3MoeForCausalLM"
_QWEN3_5_ARCHITECTURES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
_QWEN3_5_MOE = "Qwen3_5MoeForConditionalGeneration"
_REGISTERED_ARCHITECTURES = sorted(MODELS)
# Resolution hands the class hook its architecture's own text config.
_HF_CONFIG_CLASSES = {
    _QWEN3_MOE: Qwen3MoeConfig,
    "Qwen3_5ForConditionalGeneration": Qwen3_5Config,
    "Qwen3_5ForCausalLM": Qwen3_5TextConfig,
    _QWEN3_5_MOE: Qwen3_5MoeConfig,
}


@pytest.fixture(autouse=True)
def _gated_delta_net_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reference ``ctikernels.gdn``, which resolution asks whether Qwen3.5's prefill
    exports states; its prefill does."""

    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())


def _serve_architecture(
    monkeypatch: pytest.MonkeyPatch,
    folder: Path,
    architecture: str,
    **hf_fields: object,
) -> None:
    """Serve one architecture from a config-only checkpoint.

    The Hugging Face config is that architecture's default plus ``hf_fields``, saved as the
    ``unused`` and ``d`` checkpoints under ``folder``, the working directory; they declare
    no generation config and no tokenizer EOS.
    """

    hf_config_class = _HF_CONFIG_CLASSES.get(architecture, Qwen3Config)
    hf_config = hf_config_class(architectures=[architecture], **hf_fields)
    if (vision := getattr(hf_config, "vision_config", None)) is not None:
        # Transformers' defaults project the vision tower to another width than the text's.
        vision.out_hidden_size = hf_config.text_config.hidden_size
    for name in ("unused", "d"):
        hf_config.save_pretrained(folder / name)
    monkeypatch.chdir(folder)


def _missing(architecture: str, *names: str) -> str:
    message = (
        f"Model architecture {architecture} does not support the requested "
        f"capabilities: {', '.join(names)}"
    )
    return f"^{re.escape(message)}$"


def _undeclared_lora_targets(architecture: str, names: tuple[str, ...], declared: str) -> str:
    message = (
        f"Model architecture {architecture} does not declare the requested LoRA "
        f"targets: {', '.join(names)}; it declares {declared}"
    )
    return f"^{re.escape(message)}$"


def _wrapped_projections(hf_config: PretrainedConfig, targets: tuple[str, ...]) -> int:
    adapters = Adapter(LoRAConfig(targets=targets))
    build_model(hf_config, lora_adapters=adapters)
    return len(adapters.modules())


_TUNING = EngineTuning()


def _check(
    options: EngineOptions, tuning: EngineTuning, speculative: SpeculativeMethod | None = None
) -> tuple[ResolvedModel, CheckedOptions]:
    """The served checkpoint, loaded and checked with ``speculative`` against ``options`` and
    ``tuning`` as the composition root does before any device state exists."""

    resolved = ResolvedModel.load(options.model.path)
    return resolved, resolved.check(options, tuning, speculative=speculative)


def _plan(
    options: EngineOptions, tuning: EngineTuning, speculative: SpeculativeMethod | None = None
) -> Plan:
    """The checked checkpoint's plan for a 12.0 device."""

    resolved, checked = _check(options, tuning, speculative)
    return resolved.plan(
        checked,
        device=torch.device("cpu"),
        capability=DeviceCapability(12, 0),
        device_memory=32 << 30,
        online=False,
    )


def _options(*, training_layout: BatchLayout | None = None, **sections: object) -> EngineOptions:
    """Options for the served checkpoint; ``sections`` replace the defaults."""

    model = sections.pop("model", ModelOptions(path="unused"))
    return EngineOptions(model=model, training=_training_config(training_layout), **sections)


def _cache(prefix: str) -> EngineOptions:
    return _options(cache=CacheOptions(prefix=prefix))


def _training_config(layout: BatchLayout | None) -> TrainingConfig | None:
    if layout is None:
        return None
    return TrainingConfig(
        trainable_plan=TrainablePlan(("weight",)),
        optimizer=AdamWConfig(),
        batching=TrainingBatching(
            layout,
            StaticBatching(microbatch_size=1, max_tokens_per_sample=64),
        ),
    )


@pytest.fixture
def prefix_reuse_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen3-MoE as a stand-in that declares prefix reuse alone, so these rejections do not move
    as the real class gains capabilities."""

    monkeypatch.setattr(Qwen3MoeForCausalLM, "CAPABILITIES", ModelCapability.PREFIX_REUSE)


def test_require_capabilities_names_the_model_and_every_missing_capability(
    prefix_reuse_only: None,
) -> None:
    requested = (
        ModelCapability.TENSOR_PARALLEL
        | ModelCapability.LORA
        | ModelCapability.TRAINING
        | ModelCapability.PREFIX_REUSE
    )
    expected = _missing(
        _QWEN3_MOE,
        "LORA",
        "TRAINING",
        "TENSOR_PARALLEL",
    )

    with pytest.raises(ValueError, match=expected):
        Qwen3MoeForCausalLM.require_capabilities(requested, model_architecture=_QWEN3_MOE)


def test_resolution_rejects_an_unregistered_architecture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _serve_architecture(monkeypatch, tmp_path, "LlamaForCausalLM")

    with pytest.raises(ValueError, match="LlamaForCausalLM not supported"):
        _check(_options(), _TUNING)


@pytest.mark.parametrize(
    ("architecture", "hf_fields", "message"),
    [
        pytest.param(
            _QWEN3,
            {"attention_bias": True},
            "Qwen3ForCausalLM does not implement attention_bias=True",
            id="qwen3-attention-bias",
        ),
        pytest.param(
            _QWEN3_MOE,
            {"use_sliding_window": True, "sliding_window": 64},
            "Qwen3MoeForCausalLM does not implement sliding_window=64",
            id="qwen3-moe-sliding-window",
        ),
    ],
)
def test_resolution_rejects_unimplemented_checkpoint_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
    hf_fields: dict[str, object],
    message: str,
) -> None:
    _serve_architecture(monkeypatch, tmp_path, architecture, **hf_fields)

    with pytest.raises(ValueError, match=f"^{re.escape(message)}"):
        _check(_options(), _TUNING)


@pytest.mark.parametrize(
    ("options", "tuning", "message"),
    [
        pytest.param(
            _options(model=ModelOptions(path="unused", max_seq_len=32_769)),
            _TUNING,
            "max_seq_len=32769 exceeds the model's max_position_embeddings=32768",
            id="max-seq-len-beyond-rope-cache",
        ),
        pytest.param(
            _options(),
            EngineTuning(snapshot_rows=1),
            "snapshot_rows must be an integer >= 2",
            id="one-snapshot-row",
        ),
    ],
)
def test_resolution_rejects_options_the_model_cannot_honour(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    options: EngineOptions,
    tuning: EngineTuning,
    message: str,
) -> None:
    _serve_architecture(monkeypatch, tmp_path, _QWEN3)

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _check(options, tuning)


@pytest.mark.parametrize(
    ("options", "tuning", "missing"),
    [
        pytest.param(
            _options(parallel=ParallelOptions(num_ranks=2)),
            _TUNING,
            ("TENSOR_PARALLEL",),
            id="tensor-parallel",
        ),
    ],
)
def test_resolution_rejects_an_unsupported_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prefix_reuse_only: None,
    options: EngineOptions,
    tuning: EngineTuning,
    missing: tuple[str, ...],
) -> None:
    _serve_architecture(monkeypatch, tmp_path, _QWEN3_MOE)

    with pytest.raises(ValueError, match=_missing(_QWEN3_MOE, *missing)):
        _check(options, tuning)


@pytest.mark.parametrize(
    ("architecture", "targets", "undeclared", "declared"),
    [
        pytest.param(
            _QWEN3,
            ("attention", "linear_attention"),
            ("linear_attention",),
            "attention, mlp",
            id="qwen3-linear-attention",
        ),
        pytest.param(
            _QWEN3_MOE,
            ("mlp",),
            ("mlp",),
            "attention, experts",
            id="qwen3-moe-mlp",
        ),
    ],
)
def test_resolution_rejects_an_undeclared_lora_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
    targets: tuple[str, ...],
    undeclared: tuple[str, ...],
    declared: str,
) -> None:
    _serve_architecture(monkeypatch, tmp_path, architecture)
    # The LoRA capability itself is the earlier refusal, so grant it to reach the targets.
    monkeypatch.setattr(
        get_model_class(architecture),
        "CAPABILITIES",
        get_model_class(architecture).CAPABILITIES | ModelCapability.LORA,
    )

    with pytest.raises(
        ValueError,
        match=_undeclared_lora_targets(architecture, undeclared, declared),
    ):
        _check(_options(lora=LoRAConfig(targets=targets)), _TUNING)


@pytest.mark.parametrize("architecture", _QWEN3_5_ARCHITECTURES)
@pytest.mark.parametrize("training_layout", [None, BatchLayout.PADDED, BatchLayout.PACKED])
def test_resolution_serves_and_trains_qwen3_5_lora(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
    training_layout: BatchLayout | None,
) -> None:
    _serve_architecture(monkeypatch, tmp_path, architecture)
    lora_config = LoRAConfig(targets=("attention", "linear_attention", "mlp"))

    _check(_options(lora=lora_config, training_layout=training_layout), _TUNING)


@pytest.mark.parametrize("num_ranks", [1, 2, 4])
@pytest.mark.parametrize("cache_type", ["auto", "naive", "radix"])
@pytest.mark.parametrize(
    "architecture", [_QWEN3, _QWEN3_MOE, *_QWEN3_5_ARCHITECTURES, _QWEN3_5_MOE]
)
def test_resolution_resolves_prefix_reuse_independently_of_rank_count(
    monkeypatch, tmp_path, architecture, cache_type, num_ranks
):
    _serve_architecture(monkeypatch, tmp_path, architecture)
    set_validation(monkeypatch, FlashInferBackend)
    requested = _options(
        parallel=ParallelOptions(num_ranks=num_ranks), cache=CacheOptions(prefix=cache_type)
    )
    supported = ModelCapability.PREFIX_REUSE in get_model_class(architecture).CAPABILITIES
    if cache_type == "radix" and not supported:
        with pytest.raises(ValueError, match=_missing(architecture, "PREFIX_REUSE")):
            _check(requested, _TUNING)
    else:
        plan = _plan(requested, _TUNING)
        assert plan.prefix_cache == ("radix" if cache_type != "naive" and supported else "naive")
    assert requested.cache.prefix == cache_type


def test_resolution_requests_prefix_reuse_only_for_a_radix_inference_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _serve_architecture(monkeypatch, tmp_path, _QWEN3_MOE)
    monkeypatch.setattr(Qwen3MoeForCausalLM, "CAPABILITIES", ModelCapability(0))

    with pytest.raises(ValueError, match=_missing(_QWEN3_MOE, "PREFIX_REUSE")):
        _check(_cache("radix"), _TUNING)
    for cache_type in ("auto", "naive"):
        _check(_cache(cache_type), _TUNING)
    _check(_options(), _TUNING)


@pytest.mark.parametrize("cache_type", ["auto", "radix"])
@pytest.mark.parametrize("architecture", [_QWEN3, *_QWEN3_5_ARCHITECTURES])
def test_resolution_admits_dflash_prefix_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, architecture: str, cache_type: str
) -> None:
    _serve_architecture(monkeypatch, tmp_path, architecture)
    # The draft checkpoint loaded as a method that reuses prefixes, with no other check.
    method = SimpleNamespace(
        capabilities=~ModelCapability(0),
        attention_requirements=lambda: (),
        check=lambda **_limits: None,
        round_shape=lambda **_limits: object(),
    )
    set_validation(monkeypatch, FlashInferBackend)
    options = _options(
        cache=CacheOptions(prefix=cache_type), speculative=SpeculativeOptions(model_path="d")
    )
    # DFlash reuses prefixes, so the automatic choice is the radix cache too.
    assert _plan(options, _TUNING, method).prefix_cache == "radix"


@pytest.mark.parametrize("architecture", _QWEN3_5_ARCHITECTURES)
def test_resolution_admits_tensor_parallel_qwen3_5(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
) -> None:
    _serve_architecture(monkeypatch, tmp_path, architecture)

    _check(_options(parallel=ParallelOptions(num_ranks=2)), _TUNING)


@pytest.mark.parametrize("architecture", _QWEN3_5_ARCHITECTURES)
def test_resolution_accepts_radix_cache_for_qwen3_5(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
) -> None:
    _serve_architecture(monkeypatch, tmp_path, architecture)

    _check(_cache("radix"), _TUNING)
    _check(_cache("auto"), _TUNING)


@pytest.mark.parametrize(
    ("architecture", "options", "tuning"),
    [
        pytest.param(
            _QWEN3,
            _options(lora=LoRAConfig(targets=("attention",)), training_layout=BatchLayout.PACKED),
            EngineTuning(graph_row_buckets=(128,)),
            id="qwen3-lora-row-buckets-packed-training",
        ),
        pytest.param(
            _QWEN3,
            _options(cache=CacheOptions(prefix="radix"), training_layout=BatchLayout.PADDED),
            _TUNING,
            id="qwen3-radix-padded-training",
        ),
        pytest.param(
            _QWEN3,
            _options(parallel=ParallelOptions(num_ranks=2)),
            EngineTuning(graph_row_buckets=(128,)),
            id="qwen3-tensor-parallel-row-buckets",
        ),
        pytest.param(_QWEN3_MOE, _cache("radix"), _TUNING, id="qwen3-moe-radix"),
        pytest.param(
            _QWEN3_MOE,
            _options(parallel=ParallelOptions(num_ranks=2)),
            EngineTuning(graph_row_buckets=(128,)),
            id="qwen3-moe-tensor-parallel-row-buckets",
        ),
        pytest.param(
            _QWEN3_5_MOE,
            _options(parallel=ParallelOptions(num_ranks=2), cache=CacheOptions(prefix="radix")),
            _TUNING,
            id="qwen3_5-moe-tensor-parallel-radix",
        ),
        pytest.param(_QWEN3_MOE, _options(), _TUNING, id="qwen3-moe-inference"),
        pytest.param(
            _QWEN3_MOE,
            _options(),
            EngineTuning(graph_row_buckets=()),
            id="qwen3-moe-no-row-buckets",
        ),
    ],
)
def test_resolution_passes_the_capability_check_for_a_supported_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
    options: EngineOptions,
    tuning: EngineTuning,
) -> None:
    _serve_architecture(monkeypatch, tmp_path, architecture)

    _check(options, tuning)


@pytest.mark.parametrize("architecture", _REGISTERED_ARCHITECTURES)
def test_registered_model_declares_its_own_capabilities(architecture: str) -> None:
    model_class = get_model_class(architecture)

    assert "CAPABILITIES" in vars(model_class)
    assert isinstance(model_class.CAPABILITIES, ModelCapability)


@pytest.mark.parametrize("architecture", _REGISTERED_ARCHITECTURES)
def test_registered_model_validates_hf_config_itself(architecture: str) -> None:
    # The base hook is abstract; an inherited one would apply another
    # architecture's rules to this checkpoint.
    assert "validate_hf_config" in vars(get_model_class(architecture))


@pytest.mark.parametrize("architecture", _REGISTERED_ARCHITECTURES)
def test_registered_model_declares_its_own_config_type(architecture: str) -> None:
    model_class = get_model_class(architecture)

    # create_model constructs the class from CONFIG.from_hf(hf_config).
    assert "CONFIG" in vars(model_class)
    assert callable(model_class.CONFIG.from_hf)


# TRAINING promises the one model Protocol TrainingRunner consumes.
_RUNNER_METHODS: dict[str, Callable[..., object]] = {
    name: member
    for name, member in vars(TrainingModel).items()
    if inspect.isfunction(member) and not name.startswith("_")
}


@pytest.mark.parametrize("architecture", _REGISTERED_ARCHITECTURES)
def test_registered_model_declares_training_exactly_when_it_implements_the_runner(
    architecture: str,
) -> None:
    model_class = get_model_class(architecture)
    implemented = frozenset(name for name in _RUNNER_METHODS if hasattr(model_class, name))

    if ModelCapability.TRAINING in model_class.CAPABILITIES:
        assert implemented == frozenset(_RUNNER_METHODS)
    else:
        # Every model inherits these from BaseLLMModel.
        assert implemented <= {"iter_recurrent_layers", "validate_training_inputs"}


@pytest.mark.parametrize("architecture", _REGISTERED_ARCHITECTURES)
def test_registered_model_takes_every_training_call_the_contract_declares(
    architecture: str,
) -> None:
    model_class = get_model_class(architecture)

    for name, declared in _RUNNER_METHODS.items():
        if not hasattr(model_class, name):
            continue
        want = inspect.signature(declared).parameters
        have = inspect.signature(getattr(model_class, name)).parameters
        assert [(p.name, p.kind) for p in want.values()] == [
            (p.name, p.kind) for p in have.values() if p.name in want
        ], name
        assert all(p.default is not p.empty for p in have.values() if p.name not in want), name


def test_the_engine_executes_in_bfloat16_only() -> None:
    with pytest.raises(ValueError, match=r'^dtype must be "auto" or "bfloat16", not \'float16\'$'):
        ModelOptions(path="unused", dtype="float16")  # type: ignore[arg-type]


@pytest.mark.parametrize("architecture", _REGISTERED_ARCHITECTURES)
def test_registered_model_declares_lora_targets_exactly_when_it_declares_lora(
    architecture: str,
) -> None:
    model_class = get_model_class(architecture)
    targets = model_class.LORA_TARGETS

    assert "LORA_TARGETS" in vars(model_class)
    assert type(targets) is tuple
    assert len(set(targets)) == len(targets)
    assert set(targets) <= set(get_args(LoRATarget))
    assert (ModelCapability.LORA in model_class.CAPABILITIES) == bool(targets)


@pytest.mark.parametrize("architecture", _REGISTERED_ARCHITECTURES)
def test_a_tiny_model_wraps_projections_under_exactly_its_declared_lora_targets(
    tmp_path: Path,
    stub_flashinfer: None,
    architecture: str,
) -> None:
    targets = get_model_class(architecture).LORA_TARGETS
    if not targets:
        return
    hf_config = tiny_hf_config(architecture, tmp_path)

    wrapped = {target: _wrapped_projections(hf_config, (target,)) for target in targets}

    assert all(count > 0 for count in wrapped.values()), wrapped
    # Requesting every target the engine knows wraps nothing more: the model adapts no
    # projection under a target it does not declare, which no user could request.
    assert _wrapped_projections(hf_config, get_args(LoRATarget)) == sum(wrapped.values())
