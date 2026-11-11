from __future__ import annotations

import json
import re
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import safetensors.torch
import torch
from transformers import Qwen3_5Config, Qwen3Config, Qwen3MoeConfig

import cti.resolution
from cti.attention import AttentionRequirement
from cti.attention.flash_attention import FlashAttentionBackend
from cti.attention.flashinfer import FlashInferBackend
from cti.attention.tensorrt_llm import TensorRTLLMBackend
from cti.config import (
    AdamWConfig,
    BatchLayout,
    CacheOptions,
    EngineOptions,
    EngineTuning,
    ExpertLoRALayout,
    LoRAConfig,
    MemoryOptions,
    ModelOptions,
    ParallelOptions,
    SchedulingOptions,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.forward import RoundShape
from cti.models import ModelCapability, get_model_class
from cti.multimodal import ImageLimits, ImagePreprocessing
from cti.plan import Plan, SpeculativeMethod
from cti.resolution import ResolvedModel
from cti.utils import DeviceCapability, find_model_file
from tests.support.attention import set_validation
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.gated_delta_net import ReferenceGatedDeltaNetPrefill, reference_kernel_module

_QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "eos_token_id": 151645,
}
# Qwen/Qwen3.5-9B config.json with trimmed sub-configs: only the text config declares EOS.
_QWEN3_5_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "image_token_id": 248056,
    "model_type": "qwen3_5",
    "text_config": {"eos_token_id": 248044, "model_type": "qwen3_5_text"},
    "tie_word_embeddings": False,
    "video_token_id": 248057,
    "vision_config": {"model_type": "qwen3_5", "out_hidden_size": 4096},
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053,
}

# Qwen/Qwen3.5-9B tokenizer_config.json, trimmed to its EOS and pad tokens.
_QWEN3_5_TOKENIZER_CONFIG = {
    "added_tokens_decoder": {
        "248044": {"content": "<|endoftext|>", "special": True},
        "248046": {"content": "<|im_end|>", "special": True},
    },
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
}

_QWEN3 = "Qwen3ForCausalLM"
_QWEN3_MOE = "Qwen3MoeForCausalLM"
_QWEN3_5 = "Qwen3_5ForConditionalGeneration"
_HF_CONFIGS = {_QWEN3: Qwen3Config, _QWEN3_MOE: Qwen3MoeConfig, _QWEN3_5: Qwen3_5Config}


def _model(*, architecture: str = _QWEN3) -> ResolvedModel:
    """One registered architecture's resolved checkpoint, without checkpoint files; its
    default Hugging Face config (Qwen3 head_dim 128, Qwen3.5 256)."""

    return ResolvedModel(
        facts=checkpoint_facts(_HF_CONFIGS[architecture](architectures=[architecture])),
        model_config=SimpleNamespace(architectures=[architecture], max_position=4096),
        model_class=get_model_class(architecture),
    )


_BLACKWELL = DeviceCapability(12, 0)


def _options(*, model: ModelOptions | None = None, **sections: object) -> EngineOptions:
    return EngineOptions(model=model or ModelOptions(path="unused"), **sections)


@pytest.fixture(autouse=True)
def _gated_delta_net_kernels(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """The reference ``ctikernels.gdn``, which resolution asks whether Qwen3.5's prefill
    exports states; its prefill does."""

    module = reference_kernel_module()
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", module)
    return module


def _resolve(
    options: EngineOptions | None = None,
    tuning: EngineTuning | None = None,
    *,
    model: ResolvedModel | None = None,
    capability: DeviceCapability | None = _BLACKWELL,
    device_memory: int = 32 << 30,
    online: bool = False,
    speculative: SpeculativeMethod | None = None,
) -> Plan:
    model = model or _model()
    checked = model.check(options or _options(), tuning or EngineTuning(), speculative=speculative)
    return model.plan(
        checked,
        device=torch.device("cpu"),
        capability=capability,
        device_memory=device_memory,
        online=online,
    )


def _model_dir(
    model_dir: Path,
    hf_config: dict[str, object],
    generation_config: dict[str, object] | None = None,
    tokenizer_config: dict[str, object] | None = None,
) -> str:
    for filename, document in (
        ("config.json", hf_config),
        ("generation_config.json", generation_config),
        ("tokenizer_config.json", tokenizer_config),
    ):
        if document is not None:
            (model_dir / filename).write_text(json.dumps(document), encoding="utf-8")
    return str(model_dir)


def test_a_hub_id_resolves_to_one_revision_without_downloading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    _model_dir(snapshot, _QWEN3_CONFIG)
    patterns: list[str] = []

    def snapshot_download(repo_id: str, *, allow_patterns: list[str], **_kwargs: object) -> str:
        patterns.extend(allow_patterns)
        return str(snapshot)

    monkeypatch.setattr("cti.models._download.snapshot_download", snapshot_download)

    model = ResolvedModel.load("Qwen/Qwen3-0.6B")

    assert model.facts.source.folder == str(snapshot)
    assert model.facts.source.identity() == {
        "hub": "huggingface",
        "repo_id": "Qwen/Qwen3-0.6B",
        "revision": "b" * 40,
    }
    assert "*.safetensors" not in patterns


def test_stop_token_ids_keep_the_single_qwen3_eos_without_a_generation_config(
    tmp_path: Path,
) -> None:
    model = ResolvedModel.load(_model_dir(tmp_path, _QWEN3_CONFIG))

    assert model.facts.stop_token_ids == (151645,)


def test_stop_token_ids_prefer_the_generation_config_eos_list(tmp_path: Path) -> None:
    model = ResolvedModel.load(
        _model_dir(
            tmp_path,
            _QWEN3_CONFIG,
            {"eos_token_id": [151645, 151643, 151645], "temperature": 0.6},
        ),
    )

    assert model.facts.stop_token_ids == (151645, 151643)


def test_stop_token_ids_accept_a_scalar_generation_config_eos(tmp_path: Path) -> None:
    model = ResolvedModel.load(_model_dir(tmp_path, _QWEN3_CONFIG, {"eos_token_id": 7}))

    assert model.facts.stop_token_ids == (7,)


def test_stop_token_ids_fall_back_when_the_generation_config_omits_eos(
    tmp_path: Path,
) -> None:
    model = ResolvedModel.load(
        _model_dir(tmp_path, _QWEN3_CONFIG, {"eos_token_id": None, "top_k": 20}),
    )

    assert model.facts.stop_token_ids == (151645,)


def test_stop_token_ids_accept_a_model_config_eos_list(tmp_path: Path) -> None:
    model = ResolvedModel.load(
        _model_dir(tmp_path, {**_QWEN3_CONFIG, "eos_token_id": [151645, 151643]}),
    )

    assert model.facts.stop_token_ids == (151645, 151643)


def test_stop_token_ids_accept_a_text_config_eos_list(tmp_path: Path) -> None:
    text_config = {**_QWEN3_5_CONFIG["text_config"], "eos_token_id": [248046, 248044]}
    model = ResolvedModel.load(
        _model_dir(tmp_path, {**_QWEN3_5_CONFIG, "text_config": text_config}),
    )

    assert model.facts.stop_token_ids == (248046, 248044)


def test_stop_token_ids_read_the_text_config_of_a_multimodal_model(tmp_path: Path) -> None:
    model = ResolvedModel.load(_model_dir(tmp_path, _QWEN3_5_CONFIG))

    assert not hasattr(model.facts.hf_config, "eos_token_id")
    assert model.facts.stop_token_ids == (248044,)


def test_a_text_only_load_reads_an_image_text_checkpoint_as_its_text_export(
    tmp_path: Path,
) -> None:
    """The text config under the text-only architecture, the head tied as the top-level
    field says, without image preprocessing; a text checkpoint reads as it is."""

    (tmp_path / "qwen3_5").mkdir()
    (tmp_path / "qwen3").mkdir()
    image_text = _model_dir(tmp_path / "qwen3_5", {**_QWEN3_5_CONFIG, "tie_word_embeddings": True})
    model = ResolvedModel.load(image_text, text_only=True)

    assert model.facts.architecture == "Qwen3_5ForCausalLM"
    assert model.facts.hf_config.model_type == "qwen3_5_text"
    assert model.facts.hf_config.tie_word_embeddings
    assert model.facts.image_preprocessing is None
    assert model.facts.stop_token_ids == (248044,)
    assert model.model_class is ResolvedModel.load(image_text).model_class
    text = _model_dir(tmp_path / "qwen3", _QWEN3_CONFIG)
    assert (
        ResolvedModel.load(text, text_only=True).facts.hf_config.to_dict()
        == ResolvedModel.load(text).facts.hf_config.to_dict()
    )


def test_stop_token_ids_prefer_the_generation_config_of_a_multimodal_model(
    tmp_path: Path,
) -> None:
    model = ResolvedModel.load(
        _model_dir(tmp_path, _QWEN3_5_CONFIG, {"eos_token_id": [248046, 248044]}),
    )

    assert model.facts.stop_token_ids == (248046, 248044)


def test_stop_token_ids_append_the_tokenizer_eos_without_a_generation_config(
    tmp_path: Path,
) -> None:
    # Qwen3.5-9B: no generation_config.json, so the text config names only <|endoftext|>.
    model = ResolvedModel.load(
        _model_dir(
            tmp_path,
            _QWEN3_5_CONFIG,
            tokenizer_config=_QWEN3_5_TOKENIZER_CONFIG,
        ),
    )

    assert model.facts.stop_token_ids == (248044, 248046)


def test_stop_token_ids_keep_a_generation_config_that_lists_the_tokenizer_eos(
    tmp_path: Path,
) -> None:
    # Qwen3.8-27B: generation_config.json lists <|im_end|> and <|endoftext|>.
    model = ResolvedModel.load(
        _model_dir(
            tmp_path,
            _QWEN3_5_CONFIG,
            {"eos_token_id": [248046, 248044]},
            _QWEN3_5_TOKENIZER_CONFIG,
        ),
    )

    assert model.facts.stop_token_ids == (248046, 248044)


def test_stop_token_ids_do_not_repeat_a_tokenizer_eos_the_model_config_lists(
    tmp_path: Path,
) -> None:
    tokenizer_config = {
        "added_tokens_decoder": {"151645": {"content": "<|im_end|>", "special": True}},
        "eos_token": "<|im_end|>",
    }
    model = ResolvedModel.load(
        _model_dir(tmp_path, _QWEN3_CONFIG, tokenizer_config=tokenizer_config),
    )

    assert model.facts.stop_token_ids == (151645,)


@pytest.mark.parametrize(
    "tokenizer_config",
    [
        {key: value for key, value in _QWEN3_5_TOKENIZER_CONFIG.items() if key != "eos_token"},
        {**_QWEN3_5_TOKENIZER_CONFIG, "eos_token": None},
    ],
)
def test_stop_token_ids_keep_the_model_eos_for_a_tokenizer_without_eos(
    tmp_path: Path,
    tokenizer_config: dict[str, object],
) -> None:
    model = ResolvedModel.load(
        _model_dir(tmp_path, _QWEN3_5_CONFIG, tokenizer_config=tokenizer_config),
    )

    assert model.facts.stop_token_ids == (248044,)


def test_stop_token_ids_may_be_only_the_tokenizer_eos(tmp_path: Path) -> None:
    model = ResolvedModel.load(
        _model_dir(
            tmp_path,
            _QWEN3_5_CONFIG,
            {"eos_token_id": []},
            _QWEN3_5_TOKENIZER_CONFIG,
        ),
    )

    assert model.facts.stop_token_ids == (248046,)


@pytest.mark.parametrize("hf_config", [_QWEN3_CONFIG, _QWEN3_5_CONFIG], ids=["qwen3", "qwen3.5"])
def test_stop_token_ids_reject_a_tokenizer_eos_without_an_id(
    tmp_path: Path, hf_config: dict[str, object]
) -> None:
    path = _model_dir(
        tmp_path,
        hf_config,
        tokenizer_config={**_QWEN3_5_TOKENIZER_CONFIG, "eos_token": "</s>"},
    )

    # Loading the checkpoint fails, before any weight loads.
    with pytest.raises(ValueError, match="eos_token '</s>' has no valid added-token id"):
        ResolvedModel.load(path)


@pytest.mark.parametrize(
    ("hf_config", "generation_config"),
    [
        ({**_QWEN3_CONFIG, "eos_token_id": None}, None),
        (_QWEN3_CONFIG, {"eos_token_id": []}),
    ],
)
def test_stop_token_ids_may_be_empty(
    tmp_path: Path,
    hf_config: dict[str, object],
    generation_config: dict[str, object] | None,
) -> None:
    model = ResolvedModel.load(_model_dir(tmp_path, hf_config, generation_config))

    assert model.facts.stop_token_ids == ()


@pytest.mark.parametrize(
    ("eos_token_id", "error_type"),
    [
        (True, TypeError),
        (1.5, TypeError),
        ("2", TypeError),
        ({"id": 2}, TypeError),
        ([2, [3]], TypeError),
        ([2, False], TypeError),
        (-1, ValueError),
        ([2, -1], ValueError),
    ],
)
def test_stop_token_ids_reject_invalid_generation_config_eos(
    tmp_path: Path,
    eos_token_id: object,
    error_type: type[Exception],
) -> None:
    path = _model_dir(tmp_path, _QWEN3_CONFIG, {"eos_token_id": eos_token_id})

    with pytest.raises(error_type, match="eos_token_id"):
        ResolvedModel.load(path)


def test_stop_token_ids_reject_an_invalid_model_config_eos(tmp_path: Path) -> None:
    path = _model_dir(tmp_path, {**_QWEN3_CONFIG, "eos_token_id": -1})

    with pytest.raises(ValueError, match="eos_token_id must be non-negative"):
        ResolvedModel.load(path)


def test_the_checkpoint_freezes_a_local_model_path_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_snapshot = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / ("a" * 40)
    second_snapshot = first_snapshot.parent / ("b" * 40)
    first_snapshot.mkdir(parents=True)
    second_snapshot.mkdir()
    _model_dir(first_snapshot, _QWEN3_CONFIG)
    alias = tmp_path / "model"
    alias.symlink_to(first_snapshot, target_is_directory=True)
    set_validation(monkeypatch, FlashInferBackend)

    checkpoint = ResolvedModel.load(str(alias))
    alias.unlink()
    alias.symlink_to(second_snapshot, target_is_directory=True)
    plan = _resolve(_options(model=ModelOptions(path=str(alias))), model=checkpoint)

    # The weights load from the snapshot whose config was read, not the alias's new one.
    assert checkpoint.facts.source.folder == plan.model_path == str(first_snapshot.resolve())


def test_the_checkpoint_expands_home_in_a_model_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "model").mkdir()
    _model_dir(tmp_path / "model", _QWEN3_CONFIG)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert ResolvedModel.load("~/model").facts.source.folder == str(tmp_path / "model")


def test_auto_dtype_runs_an_fp16_checkpoint_in_bfloat16(monkeypatch: pytest.MonkeyPatch) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    model = _model()
    hf_config = Qwen3Config(architectures=[_QWEN3], dtype="float16")
    model = replace(model, facts=replace(model.facts, hf_config=hf_config))

    plan = _resolve(_options(model=ModelOptions(path="unused", dtype="auto")), model=model)

    assert plan.dtype is torch.bfloat16


def test_auto_dtype_runs_an_fp32_checkpoint_in_bfloat16(monkeypatch: pytest.MonkeyPatch) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    model = _model()
    hf_config = Qwen3Config(architectures=[_QWEN3], dtype="float32")
    model = replace(model, facts=replace(model.facts, hf_config=hf_config))

    plan = _resolve(_options(model=ModelOptions(path="unused", dtype="auto")), model=model)

    assert plan.dtype is torch.bfloat16


def test_graph_row_buckets_are_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    tuning = EngineTuning(graph_row_buckets=(2048, 1024, 2048, 4096))

    plan = _resolve(_options(scheduling=SchedulingOptions(max_batch_tokens=4096)), tuning)

    assert plan.graph_row_buckets == (1024, 2048, 4096)


@pytest.mark.parametrize("value", [0, -1])
def test_graph_row_buckets_must_be_positive(value: int) -> None:
    with pytest.raises(ValueError, match="graph_row_buckets must hold positive sizes"):
        EngineTuning(graph_row_buckets=(value,))


@pytest.mark.parametrize(
    ("max_batch_tokens", "max_prompt_tokens", "online", "gib", "expected"),
    [
        (32768, None, True, 16, 2048),
        (32768, None, True, 32, 4096),
        (32768, None, True, 96, 8192),
        (32768, None, True, 192, 16384),
        (1024, None, True, 32, 1024),
        (8192, None, False, 32, 8192),
        (8192, 4096, True, 96, 4096),
        (8192, 512, False, 32, 512),
    ],
)
def test_the_prompt_budget_is_the_option_else_the_memory_tier_online_and_max_batch_tokens_offline(
    monkeypatch: pytest.MonkeyPatch,
    max_batch_tokens: int,
    max_prompt_tokens: int | None,
    online: bool,
    gib: int,
    expected: int,
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    scheduling = SchedulingOptions(
        max_batch_tokens=max_batch_tokens, max_prompt_tokens=max_prompt_tokens
    )

    plan = _resolve(_options(scheduling=scheduling), device_memory=gib << 30, online=online)

    assert plan.prompt_budget.tokens == expected


@pytest.mark.parametrize("value", [0, -1, 8193])
def test_max_prompt_tokens_must_be_positive_and_within_max_batch_tokens(value: int) -> None:
    with pytest.raises(ValueError, match="max_prompt_tokens must be positive"):
        SchedulingOptions(max_prompt_tokens=value)


@pytest.mark.parametrize(("max_seq_len", "expected"), [(None, 4096), (1, 1), (4096, 4096)])
def test_a_max_seq_len_within_the_rope_cache_is_kept(monkeypatch, max_seq_len, expected):
    set_validation(monkeypatch, FlashInferBackend)
    options = _options(model=ModelOptions(path="unused", max_seq_len=max_seq_len))

    _model().check(options, EngineTuning(), speculative=None)
    assert _resolve(options).max_seq_len == expected


@pytest.mark.parametrize(("largest", "expected"), [(16384, 1024), (512, 512), (None, 0)])
def test_a_request_takes_one_image_per_token_each_up_to_the_preprocessings_largest(
    monkeypatch: pytest.MonkeyPatch, largest: int | None, expected: int
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    scheduling = SchedulingOptions(max_running_requests=8, max_batch_tokens=64)
    options = _options(model=ModelOptions(path="unused", max_seq_len=1024), scheduling=scheduling)
    preprocessing = (
        None
        if largest is None
        else ImagePreprocessing(64, largest, 1 / 255, (0.5,) * 3, (0.5,) * 3)
    )

    model = _model()
    model = replace(model, facts=replace(model.facts, image_preprocessing=preprocessing))
    plan = _resolve(options, model=model)

    assert plan.image_limits == ImageLimits(max_images=1024, max_image_tokens=expected)
    # The encoder rows hold a step's tokens and the largest image.
    assert plan.encoder_rows == max(64, expected)


@pytest.mark.parametrize(
    ("vision_graphs", "smallest", "max_batch_tokens", "expected"),
    [
        # Off by default.
        (False, 64, 1000, ()),
        # Doubling from the smallest image's tokens below max_batch_tokens, then it.
        (True, 64, 1000, (64, 128, 256, 512, 1000)),
        (True, 64, 512, (64, 128, 256, 512)),
        (True, 64, 64, (64,)),
        # No image fits a step: no graph.
        (True, 128, 64, ()),
        # A checkpoint without image preprocessing admits no image.
        (True, None, 1000, ()),
    ],
)
def test_vision_graph_budgets_double_from_the_smallest_image_when_asked(
    monkeypatch: pytest.MonkeyPatch,
    vision_graphs: bool,
    smallest: int | None,
    max_batch_tokens: int,
    expected: tuple[int, ...],
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    scheduling = SchedulingOptions(max_running_requests=8, max_batch_tokens=max_batch_tokens)
    options = _options(model=ModelOptions(path="unused", max_seq_len=1024), scheduling=scheduling)
    preprocessing = (
        None
        if smallest is None
        else ImagePreprocessing(smallest, 16384, 1 / 255, (0.5,) * 3, (0.5,) * 3)
    )
    model = _model()
    model = replace(model, facts=replace(model.facts, image_preprocessing=preprocessing))

    plan = _resolve(options, EngineTuning(vision_graphs=vision_graphs), model=model)

    assert EngineTuning().vision_graphs is False
    assert plan.vision_graph_budgets == expected


def test_a_max_seq_len_beyond_the_rope_cache_is_refused():
    options = _options(model=ModelOptions(path="unused", max_seq_len=4097))

    with pytest.raises(
        ValueError, match="^max_seq_len=4097 exceeds the model's max_position_embeddings=4096$"
    ):
        _model().check(options, EngineTuning(), speculative=None)


def test_the_max_seq_len_of_a_yarn_checkpoint_is_its_max_position_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def checkpoint(name: str, max_position_embeddings: int) -> str:
        path = tmp_path / name
        path.mkdir()
        document = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "max_position_embeddings": max_position_embeddings,
            "rope_theta": 1_000_000,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32_768,
            },
        }
        (path / "config.json").write_text(json.dumps(document), encoding="utf-8")
        return str(path)

    # Qwen3's model card adds only rope_scaling and keeps 40960, so the RoPE cache
    # stops there; Hugging Face's YaRN validation expects the extended 131072.
    recipe = ResolvedModel.load(checkpoint("recipe", 40_960))
    options = _options(model=ModelOptions(path="unused", max_seq_len=131_072))
    with pytest.raises(
        ValueError, match="^max_seq_len=131072 exceeds the model's max_position_embeddings=40960$"
    ):
        recipe.check(options, EngineTuning(), speculative=None)
    set_validation(monkeypatch, FlashInferBackend)
    extended = ResolvedModel.load(checkpoint("extended", 131_072))
    assert _resolve(model=extended).max_seq_len == 131_072


@pytest.mark.parametrize(
    ("options", "tuning", "error", "message"),
    [
        pytest.param(
            _options(scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=64)),
            EngineTuning(cuda_graph_bs=(1, 8)),
            ValueError,
            "cuda_graph_bs must not exceed max_running_requests",
            id="graph-beyond-running-requests",
        ),
        pytest.param(
            _options(scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=64)),
            EngineTuning(graph_row_buckets=(128,)),
            ValueError,
            "graph_row_buckets must not exceed max_batch_tokens",
            id="row-buckets-beyond-batch-tokens",
        ),
        pytest.param(
            _options(),
            EngineTuning(snapshot_rows=1),
            ValueError,
            "snapshot_rows must be an integer >= 2",
            id="one-snapshot-row",
        ),
        pytest.param(
            _options(),
            EngineTuning(snapshot_rows=2, snapshot_memory_ratio=0.5),
            ValueError,
            "set only one of snapshot_rows and snapshot_memory_ratio",
            id="snapshot-rows-and-ratio",
        ),
    ],
)
def test_the_request_check_refuses_what_sections_ask_of_each_other(
    options: EngineOptions, tuning: EngineTuning, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        _model().check(options, tuning, speculative=None)


def test_a_peft_adapter_sizes_the_planned_lora_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        FlashInferBackend, "validate_environment", classmethod(lambda cls, capability: ())
    )
    document = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "down_proj"]}
    (tmp_path / "adapter_config.json").write_text(json.dumps(document))
    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    safetensors.torch.save_file({key: torch.zeros(4, 8)}, tmp_path / "adapter_model.safetensors")
    lora = LoRAConfig(adapter=str(tmp_path))

    plan = _resolve(_options(lora=lora), model=_model(architecture=_QWEN3_MOE))

    assert plan.lora == replace(lora, targets=("attention", "experts"), rank=8, alpha=16.0)
    with pytest.raises(ValueError, match="no LoRA target adapts in_proj_qkv"):
        document["target_modules"] = ["in_proj_qkv"]
        (tmp_path / "adapter_config.json").write_text(json.dumps(document))
        _model().check(_options(lora=lora), EngineTuning(), speculative=None)


def test_training_refuses_an_expert_adapter_whose_gate_and_up_have_their_own_a() -> None:
    lora = LoRAConfig(
        targets=("attention", "experts"),
        rank=8,
        expert_layout=ExpertLoRALayout.PER_PROJECTION,
    )
    training = TrainingConfig(
        trainable_plan=TrainablePlan.all_declared(),
        optimizer=AdamWConfig(),
        batching=TrainingBatching(
            BatchLayout.PADDED, StaticBatching(microbatch_size=1, max_tokens_per_sample=16)
        ),
    )

    with pytest.raises(ValueError, match="shared-A layout"):
        _model(architecture=_QWEN3_MOE).check(
            _options(lora=lora, training=training), EngineTuning(), speculative=None
        )


def test_engine_request_checks_capabilities_on_the_resolved_model_class(monkeypatch):
    # The registry serves Qwen3ForCausalLM with LoRA; only the resolved class lacks it.
    resolved = get_model_class(_QWEN3_MOE)
    without_lora = type(
        "WithoutLoRA",
        (resolved,),
        {"CAPABILITIES": resolved.CAPABILITIES & ~ModelCapability.LORA, "LORA_TARGETS": ()},
    )
    model = replace(_model(), model_class=without_lora)
    monkeypatch.setattr(
        cti.resolution,
        "get_model_class",
        Mock(side_effect=AssertionError("the resolved model class must not be looked up again")),
    )

    with pytest.raises(
        ValueError,
        match=f"^Model architecture {_QWEN3} does not support the requested capabilities: LORA$",
    ):
        model.check(
            _options(lora=LoRAConfig(targets=("attention",))), EngineTuning(), speculative=None
        )


def test_resolve_makes_the_automatic_choices(monkeypatch):
    monkeypatch.setattr(
        FlashInferBackend,
        "validate_environment",
        classmethod(lambda cls, capability: ()),
    )
    options = _options(memory=MemoryOptions(inference_ratio=0.55, peak_ratio=0.8))

    plan = _resolve(options)

    assert plan.attention.selected == "fi"
    assert (plan.inference_ratio, plan.peak_ratio) == (0.55, 0.8)
    assert plan.max_batch_tokens == options.scheduling.max_batch_tokens


@pytest.mark.parametrize("speculative", [False, True])
def test_resolve_plans_how_rows_are_seeded(
    monkeypatch: pytest.MonkeyPatch, speculative: bool
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    method = SimpleNamespace(
        capabilities=~ModelCapability(0),
        attention_requirements=lambda: (),
        round_shape=lambda **_limits: RoundShape(16, 2, 4),
        check=lambda *, max_seq_len: None,
    )
    plan = _resolve(speculative=method if speculative else None)

    # Rounds sample every row from its request's seed; elsewhere rows may be either.
    assert plan.seeds_every_row is speculative


def test_rounds_choose_a_backend_that_serves_the_drafts_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for backend in (FlashAttentionBackend, FlashInferBackend, TensorRTLLMBackend):
        set_validation(monkeypatch, backend)
    # The draft's head size has no accepted TRTLLM operation test, as rounds need fi anyway.
    draft = AttentionRequirement.mha(16, 4, 256, window=64)
    method = SimpleNamespace(
        capabilities=~ModelCapability(0),
        attention_requirements=lambda: (draft,),
        round_shape=lambda **_limits: RoundShape(16, 2, 4),
        check=lambda *, max_seq_len: None,
    )
    capability = DeviceCapability(10, 0)

    plan = _resolve(capability=capability, speculative=method)

    assert plan.attention.selected == "fi"
    with pytest.raises(ValueError, match="trtllm does not run beside fixed-query rounds"):
        _resolve(
            tuning=EngineTuning(attention_backend="trtllm"),
            capability=capability,
            speculative=method,
        )


def test_resolve_chooses_the_smallest_page_size_the_backends_take(monkeypatch):
    monkeypatch.setattr(
        TensorRTLLMBackend,
        "validate_environment",
        classmethod(lambda cls, capability: ()),
    )
    set_validation(monkeypatch, FlashInferBackend)

    plan = _resolve(capability=DeviceCapability(10, 0))
    assert (plan.attention.selected, plan.page_size) == ("trtllm", 16)

    # An explicit size trtllm does not take passes it over, never rewritten.
    plan = _resolve(tuning=EngineTuning(page_size=1), capability=DeviceCapability(10, 0))
    assert (plan.attention.selected, plan.page_size) == ("fi", 1)
    with pytest.raises(ValueError, match="trtllm does not take page size 1"):
        _resolve(
            tuning=EngineTuning(attention_backend="trtllm", page_size=1),
            capability=DeviceCapability(10, 0),
        )


def test_resolve_keeps_the_stop_set_that_validation_read(
    monkeypatch,
    tmp_path: Path,
) -> None:
    file_finder = Mock(wraps=find_model_file)
    monkeypatch.setattr("cti.models.checkpoint.find_model_file", file_finder)
    set_validation(monkeypatch, FlashInferBackend)
    path = _model_dir(
        tmp_path,
        _QWEN3_CONFIG,
        {"eos_token_id": [151645, 151643]},
        {
            "added_tokens_decoder": {"151645": {"content": "<|im_end|>", "special": True}},
            "eos_token": "<|im_end|>",
        },
    )
    options = _options(model=ModelOptions(path=path))
    model = ResolvedModel.load(path)
    model.check(options, EngineTuning(), speculative=None)

    _resolve(options, model=model)

    assert model.facts.stop_token_ids == (151645, 151643)
    # The worker's metadata reads the checkpoint; the files are read only once.
    found = [name for _path, name in (call.args for call in file_finder.call_args_list)]
    assert found.count("generation_config.json") == 1
    assert found.count("tokenizer_config.json") == 1


@pytest.mark.parametrize(
    ("options", "tuning", "capability"),
    [
        pytest.param(_options(), EngineTuning(), _BLACKWELL, id="flashinfer"),
        pytest.param(
            _options(), EngineTuning(attention_backend="fa"), _BLACKWELL, id="explicit-backend"
        ),
        # Hopper's automatic choice runs ragged attention on FlashAttention.
        pytest.param(_options(), EngineTuning(), DeviceCapability(9, 0), id="automatic-backend"),
    ],
)
def test_resolve_keeps_graph_row_buckets_on_every_backend_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
    options: EngineOptions,
    tuning: EngineTuning,
    capability: DeviceCapability,
) -> None:
    for backend in (FlashAttentionBackend, FlashInferBackend):
        set_validation(monkeypatch, backend)
    tuning = replace(tuning, graph_row_buckets=(128, 64))

    assert _resolve(options, tuning, capability=capability).graph_row_buckets == (64, 128)


@pytest.mark.parametrize(
    ("architecture", "prefix", "expected"),
    [
        pytest.param(_QWEN3, "auto", "radix", id="qwen3-auto"),
        pytest.param(_QWEN3, "naive", "naive", id="qwen3-naive"),
        pytest.param(_QWEN3_5, "auto", "radix", id="qwen3.5-auto"),
        pytest.param(_QWEN3_5, "radix", "radix", id="qwen3.5-explicit-radix"),
    ],
)
def test_resolve_selects_the_radix_cache_only_for_prefix_reusing_models(
    monkeypatch: pytest.MonkeyPatch, architecture: str, prefix: str, expected: str
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    options = _options(cache=CacheOptions(prefix=prefix))

    plan = _resolve(options, model=_model(architecture=architecture))

    assert plan.prefix_cache == expected
    assert (plan.snapshots is None) is (expected == "naive")


class _PrefillWithoutCapture(ReferenceGatedDeltaNetPrefill):
    """A prefill that exports no states, as a non-CuTe provider's."""

    supports_state_capture = False


def test_resolve_without_recurrent_state_exports_drops_automatic_prefix_reuse(
    monkeypatch: pytest.MonkeyPatch, _gated_delta_net_kernels: types.ModuleType
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    options = _options(cache=CacheOptions(prefix="auto"))
    model = _model(architecture=_QWEN3_5)
    exporting = _resolve(options, model=model)
    monkeypatch.setattr(_gated_delta_net_kernels, "GatedDeltaNetPrefill", _PrefillWithoutCapture)

    plan = _resolve(options, model=model)

    # Every other choice is unchanged.
    assert (plan.prefix_cache, plan.snapshots) == ("naive", None)
    assert replace(plan, prefix_cache="radix", snapshots=exporting.snapshots) == exporting


def test_resolve_without_recurrent_state_exports_refuses_explicit_prefix_reuse(
    monkeypatch: pytest.MonkeyPatch, _gated_delta_net_kernels: types.ModuleType
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    monkeypatch.setattr(_gated_delta_net_kernels, "GatedDeltaNetPrefill", _PrefillWithoutCapture)
    options = _options(cache=CacheOptions(prefix="radix"))

    with pytest.raises(
        ValueError, match='^cache prefix "radix" with recurrent layers needs.*use "naive"$'
    ):
        _resolve(options, model=_model(architecture=_QWEN3_5))


@pytest.mark.parametrize(
    ("architecture", "prefix", "max_prompt_tokens"),
    [
        pytest.param(_QWEN3_5, "radix", 64, id="qwen3.5-radix-on-the-grid"),
        pytest.param(_QWEN3_5, "naive", 63, id="qwen3.5-naive"),
        pytest.param(_QWEN3, "radix", 63, id="attention-only-radix"),
    ],
)
def test_resolve_takes_prompt_steps_where_no_snapshot_grid_binds_them(
    monkeypatch: pytest.MonkeyPatch, architecture: str, prefix: str, max_prompt_tokens: int
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    scheduling = SchedulingOptions(max_batch_tokens=128, max_prompt_tokens=max_prompt_tokens)
    options = _options(scheduling=scheduling, cache=CacheOptions(prefix=prefix))

    plan = _resolve(options, model=_model(architecture=architecture))

    assert (plan.prefix_cache, plan.prompt_budget.tokens) == (prefix, max_prompt_tokens)


def test_resolve_refuses_prompt_steps_short_of_the_recurrent_snapshot_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    scheduling = SchedulingOptions(max_batch_tokens=128, max_prompt_tokens=63)
    options = _options(scheduling=scheduling, cache=CacheOptions(prefix="radix"))

    with pytest.raises(
        ValueError,
        match=r"^max_prompt_tokens=63 \(set by scheduling.max_prompt_tokens\) must reach the "
        "64-token grid",
    ):
        _resolve(options, model=_model(architecture=_QWEN3_5))


@pytest.mark.parametrize("num_ranks", [1, 2])
def test_resolve_asks_the_prefill_of_each_ranks_heads(
    monkeypatch: pytest.MonkeyPatch, _gated_delta_net_kernels: types.ModuleType, num_ranks: int
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    asked: list[tuple[int, int, torch.device]] = []

    class _Recording(ReferenceGatedDeltaNetPrefill):
        def __init__(self, *, num_k_heads: int, num_v_heads: int, device: torch.device) -> None:
            super().__init__(num_k_heads=num_k_heads, num_v_heads=num_v_heads, device=device)
            asked.append((num_k_heads, num_v_heads, device))

    monkeypatch.setattr(_gated_delta_net_kernels, "GatedDeltaNetPrefill", _Recording)
    model = _model(architecture=_QWEN3_5)
    text = model.facts.hf_config.get_text_config()

    _resolve(_options(parallel=ParallelOptions(num_ranks=num_ranks)), model=model)

    heads = (text.linear_num_key_heads // num_ranks, text.linear_num_value_heads // num_ranks)
    assert asked == [(*heads, torch.device("cpu"))]


def test_resolve_selects_attention_for_the_model_head_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for backend in (FlashAttentionBackend, FlashInferBackend, TensorRTLLMBackend):
        set_validation(monkeypatch, backend)

    capability = DeviceCapability(10, 0)
    # The model class gives each head size: 128 for Qwen3, 256 for Qwen3.5.
    qwen3 = _resolve(model=_model(architecture=_QWEN3), capability=capability)
    qwen3_5 = _resolve(model=_model(architecture=_QWEN3_5), capability=capability)

    assert qwen3.attention.selected == "trtllm"
    assert qwen3_5.attention.selected == "fi"


@pytest.mark.parametrize("device", [(8, 0), (9, 0), (10, 0), (12, 0)])
def test_tensor_parallel_resolution_selects_flashinfer(monkeypatch, device):
    for backend in (FlashAttentionBackend, FlashInferBackend, TensorRTLLMBackend):
        set_validation(monkeypatch, backend)

    resolved = _resolve(
        _options(parallel=ParallelOptions(num_ranks=2)), capability=DeviceCapability(*device)
    )

    assert resolved.attention.selected == "fi"


@pytest.mark.parametrize("attention_backend", ["fa", "trtllm", "fa,fi", "fi,trtllm"])
def test_tensor_parallel_resolution_rejects_other_backends_before_provider_checks(
    monkeypatch, attention_backend
):
    def unexpected_validation(cls, capability):
        pytest.fail("an unqualified backend must fail before provider environment checks")

    for backend in (FlashAttentionBackend, FlashInferBackend, TensorRTLLMBackend):
        monkeypatch.setattr(backend, "validate_environment", classmethod(unexpected_validation))
    with pytest.raises(ValueError, match="not qualified for tensor-parallel execution"):
        _resolve(
            _options(parallel=ParallelOptions(num_ranks=2)),
            EngineTuning(attention_backend=attention_backend),
        )


@pytest.mark.parametrize("num_ranks", [1, 2, 4])
@pytest.mark.parametrize(
    ("prefix", "supported"),
    # The request check refuses an explicit radix cache without PREFIX_REUSE.
    [("auto", False), ("auto", True), ("naive", False), ("naive", True), ("radix", True)],
)
def test_cache_selection_depends_only_on_prefix_support(monkeypatch, prefix, num_ranks, supported):
    set_validation(monkeypatch, FlashInferBackend)
    model_class = get_model_class(_QWEN3)
    capabilities = model_class.CAPABILITIES | ModelCapability.PREFIX_REUSE
    if not supported:
        capabilities &= ~ModelCapability.PREFIX_REUSE
    monkeypatch.setattr(model_class, "CAPABILITIES", capabilities)
    options = _options(
        parallel=ParallelOptions(num_ranks=num_ranks), cache=CacheOptions(prefix=prefix)
    )

    resolved = _resolve(options)

    expected = supported if prefix == "auto" else prefix == "radix"
    assert resolved.prefix_cache == ("radix" if expected else "naive")
