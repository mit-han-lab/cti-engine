"""Qwen3.5 configuration: published checkpoint geometry and every rejected setting."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
from transformers import PretrainedConfig

from cti.models import get_model_class
from cti.models.qwen3_5.config import (
    LayerKind,
    Qwen3_5AttentionConfig,
    Qwen3_5Config,
    Qwen3_5LinearAttentionConfig,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from tests.support.model_config import validate_with_model_class
from tests.support.qwen3_5 import (
    QWEN3_5_9B_CONFIG,
    QWEN3_8_27B_CONFIG,
    load_hf_config,
    text_only_config,
)

_HYBRID_PATTERN = (LayerKind.LINEAR_ATTENTION,) * 3 + (LayerKind.FULL_ATTENTION,)


def _edited(edit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Return a copy of the 9B config whose text config ``edit`` changed."""

    config = copy.deepcopy(dict(QWEN3_5_9B_CONFIG))
    edit(config["text_config"])
    return config


def _set(field: str, value: object) -> Callable[[dict[str, Any]], None]:
    return lambda text: text.__setitem__(field, value)


def _set_rope(field: str, value: object) -> Callable[[dict[str, Any]], None]:
    return lambda text: text["rope_parameters"].__setitem__(field, value)


def _drop(field: str) -> Callable[[dict[str, Any]], None]:
    return lambda text: text.pop(field)


def _drop_rope(field: str) -> Callable[[dict[str, Any]], None]:
    return lambda text: text["rope_parameters"].pop(field)


def _yarn_rope(text: dict[str, Any]) -> None:
    text["rope_parameters"].update(
        rope_type="yarn", factor=4.0, original_max_position_embeddings=65536
    )


@pytest.mark.parametrize(
    ("published", "sizes", "repeats", "heads", "value_heads"),
    [
        (QWEN3_5_9B_CONFIG, (4096, 12288, 248320), 8, (16, 4), 32),
        (QWEN3_8_27B_CONFIG, (5120, 17408, 248320), 16, (24, 4), 48),
    ],
    ids=["9b", "27b"],
)
def test_reads_the_published_checkpoint_geometry(
    tmp_path: Path,
    published: dict[str, Any],
    sizes: tuple[int, int, int],
    repeats: int,
    heads: tuple[int, int],
    value_heads: int,
) -> None:
    text = Qwen3_5Config.from_hf(load_hf_config(published, tmp_path)).text

    assert (text.hidden_size, text.intermediate_size, text.vocab_size) == sizes
    assert (text.rms_norm_eps, text.tie_word_embeddings) == (1e-6, False)
    assert text.layer_types == _HYBRID_PATTERN * repeats
    assert text.attention == Qwen3_5AttentionConfig(
        num_qo_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim=256,
        rotary_dim=64,
        rope_theta=10_000_000,
        max_position=262144,
        mrope_section=(11, 11, 10),
    )
    assert text.linear_attention == Qwen3_5LinearAttentionConfig(
        num_key_heads=16,
        num_value_heads=value_heads,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_size=4,
    )


@pytest.mark.parametrize(
    ("published", "out_hidden_size"),
    [(QWEN3_5_9B_CONFIG, 4096), (QWEN3_8_27B_CONFIG, 5120)],
    ids=["9b", "27b"],
)
def test_reads_the_vision_tower_geometry(
    tmp_path: Path,
    published: dict[str, Any],
    out_hidden_size: int,
) -> None:
    vision = Qwen3_5Config.from_hf(load_hf_config(published, tmp_path)).vision

    assert vision == Qwen3_5VisionConfig(
        depth=27,
        hidden_size=1152,
        num_heads=16,
        intermediate_size=4304,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=2304,
        out_hidden_size=out_hidden_size,
    )
    # 72-wide heads, 3 x 2 x 16 x 16 values per patch and a 48 x 48 position grid.
    assert (vision.head_dim, vision.patch_dim, vision.position_grid_side) == (72, 1536, 48)


def _vision_edited(**fields: object) -> dict[str, Any]:
    config = copy.deepcopy(dict(QWEN3_5_9B_CONFIG))
    config["vision_config"].update(fields)
    return config


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param({"hidden_act": "gelu"}, "tanh-approximated GELU", id="hidden-act"),
        pytest.param({"deepstack_visual_indexes": [8]}, "deepstack", id="deepstack"),
        pytest.param({"out_hidden_size": 3584}, "out_hidden_size", id="out-hidden-size"),
        pytest.param({"num_heads": 7}, "divisible by num_heads", id="heads"),
        pytest.param({"hidden_size": 16 * 70}, "multiple of 4", id="head-dim"),
        pytest.param({"num_position_embeddings": 2300}, "square", id="position-grid"),
        pytest.param({"patch_size": [16, 16]}, "patch_size", id="patch-size"),
        pytest.param({"depth": 0}, "depth", id="depth"),
    ],
)
def test_rejects_a_vision_tower_the_engine_would_run_differently(
    tmp_path: Path,
    fields: dict[str, object],
    message: str,
) -> None:
    hf_config = load_hf_config(_vision_edited(**fields), tmp_path)

    with pytest.raises(ValueError, match=message):
        Qwen3_5Config.from_hf(hf_config)


def test_a_vision_tower_needs_the_image_placeholder(tmp_path: Path) -> None:
    config = Qwen3_5Config.from_hf(load_hf_config(QWEN3_5_9B_CONFIG, tmp_path))

    with pytest.raises(ValueError, match="image_token_id"):
        replace(config, image_token_id=None)
    assert replace(config, image_token_id=None, vision=None).vision is None


@pytest.mark.parametrize("published", [QWEN3_5_9B_CONFIG, QWEN3_8_27B_CONFIG], ids=["9b", "27b"])
def test_a_text_only_export_reads_like_its_image_text_checkpoint(
    tmp_path: Path,
    published: dict[str, Any],
) -> None:
    image_text = load_hf_config(published, tmp_path / "image-text")
    text_only = load_hf_config(text_only_config(published), tmp_path / "text-only")

    assert text_only.model_type == "qwen3_5_text"
    assert Qwen3_5Config.from_hf(text_only).text == Qwen3_5Config.from_hf(image_text).text
    # Only the image-text checkpoint has image and video placeholders; a text-only
    # export embeds every id as text, as Transformers' text-only model does.
    image_text_config = Qwen3_5Config.from_hf(image_text)
    assert image_text_config.image_token_id == published["image_token_id"]
    assert image_text_config.video_token_id == published["video_token_id"]
    text_only_export = Qwen3_5Config.from_hf(text_only)
    assert (text_only_export.image_token_id, text_only_export.video_token_id) == (None, None)
    # Nor a vision tower.
    assert image_text_config.vision is not None
    assert text_only_export.vision is None


@pytest.mark.parametrize(
    "published",
    [
        QWEN3_5_9B_CONFIG,
        QWEN3_8_27B_CONFIG,
        text_only_config(QWEN3_5_9B_CONFIG),
        text_only_config(QWEN3_8_27B_CONFIG),
    ],
    ids=["9b", "27b", "9b-text-only", "27b-text-only"],
)
def test_the_model_class_accepts_the_published_checkpoints(
    tmp_path: Path,
    published: dict[str, Any],
) -> None:
    validate_with_model_class(load_hf_config(published, tmp_path))


@pytest.mark.parametrize(
    "published",
    [QWEN3_5_9B_CONFIG, QWEN3_8_27B_CONFIG, text_only_config(QWEN3_5_9B_CONFIG)],
    ids=["9b", "27b", "9b-text-only"],
)
def test_the_engine_config_agrees_with_the_model_geometry(
    tmp_path: Path,
    published: dict[str, Any],
) -> None:
    # Resolution and the runner size forward buffers from the engine config; the model
    # builds from Qwen3_5Config.
    hf_config = load_hf_config(published, tmp_path)
    engine = get_model_class(published["architectures"][0]).engine_config(hf_config)
    text = Qwen3_5Config.from_hf(hf_config).text

    assert engine.architectures == tuple(published["architectures"])
    assert (text.attention.head_dim, text.attention.num_kv_heads) == (256, 4)
    assert (engine.num_layers, engine.vocab_size, engine.hidden_size) == (
        len(text.layer_types),
        text.vocab_size,
        text.hidden_size,
    )
    assert engine.max_position == text.attention.max_position == 262144


def test_the_image_text_wrapper_ties_the_lm_head_from_its_own_field(tmp_path: Path) -> None:
    # transformers ties Qwen3_5ForConditionalGeneration from the top-level field; the
    # published text_config omits it, so a tied wrapper must not read the text default.
    tied = {**QWEN3_5_9B_CONFIG, "tie_word_embeddings": True}

    assert Qwen3_5Config.from_hf(
        load_hf_config(tied, tmp_path / "wrapper")
    ).text.tie_word_embeddings
    assert Qwen3_5Config.from_hf(
        load_hf_config(text_only_config(tied), tmp_path / "text-only")
    ).text.tie_word_embeddings


def test_layer_types_default_to_the_full_attention_interval(tmp_path: Path) -> None:
    def interval_two(text: dict[str, Any]) -> None:
        del text["layer_types"]
        text["full_attention_interval"] = 2

    text = Qwen3_5Config.from_hf(load_hf_config(_edited(interval_two), tmp_path)).text

    assert text.layer_types == (LayerKind.LINEAR_ATTENTION, LayerKind.FULL_ATTENTION) * 16


def test_parsed_config_is_frozen_and_owns_its_values(tmp_path: Path) -> None:
    hf_config = load_hf_config(QWEN3_5_9B_CONFIG, tmp_path)
    config = Qwen3_5Config.from_hf(hf_config)
    hf_config.text_config.layer_types[0] = "full_attention"
    hf_config.text_config.rope_parameters["mrope_section"][0] = 99

    assert config.text.layer_types[0] is LayerKind.LINEAR_ATTENTION
    assert config.text.attention.mrope_section == (11, 11, 10)
    with pytest.raises(FrozenInstanceError):
        config.text.hidden_size = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        pytest.param(_set("hidden_act", "gelu"), "SiLU", id="hidden-act"),
        pytest.param(_set("rms_norm_eps", 1e-5), "epsilon 1e-06", id="rms-norm-eps"),
        pytest.param(_set("attention_bias", True), "attention bias", id="attention-bias"),
        pytest.param(_set("attn_output_gate", False), "output gate", id="no-output-gate"),
        pytest.param(
            _set("layer_types", ["sliding_attention"] * 32), "layer type", id="unknown-layer"
        ),
        pytest.param(_set("linear_key_head_dim", 64), "128", id="linear-key-head-dim"),
        pytest.param(_set("linear_value_head_dim", 256), "128", id="linear-value-head-dim"),
        pytest.param(_set("linear_conv_kernel_dim", 3), "kernel size must be 4", id="conv"),
        pytest.param(_set("linear_num_value_heads", 24), "multiple", id="linear-heads"),
        pytest.param(
            _set("output_gate_type", "sigmoid"), "output_gate_type", id="output-gate-type"
        ),
        pytest.param(
            _set("output_gate_type", None), "output_gate_type", id="output-gate-type-null"
        ),
        pytest.param(_set("num_key_value_heads", 3), "divisible", id="kv-heads"),
        pytest.param(_set("head_dim", 128), "256", id="head-dim"),
        pytest.param(_set("vocab_size", 0), "positive", id="vocab-size"),
        pytest.param(_yarn_rope, "default RoPE", id="rope-type"),
        pytest.param(_set_rope("mrope_interleaved", False), "interleaved", id="mrope-chunked"),
        pytest.param(_set_rope("mrope_section", [11, 11, 11]), "half of", id="mrope-section"),
        pytest.param(_set_rope("mrope_section", [16, 16]), "three", id="mrope-axes"),
        pytest.param(_set_rope("partial_rotary_factor", 0.1), "integral", id="rotary-integral"),
        pytest.param(_set_rope("partial_rotary_factor", 0.5), "must be 64", id="rotary-width"),
        pytest.param(_set_rope("rope_theta", 0), "rope_theta", id="rope-theta"),
    ],
)
def test_rejects_what_the_engine_would_run_differently(
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    hf_config = load_hf_config(_edited(edit), tmp_path)

    with pytest.raises(ValueError, match=message):
        Qwen3_5Config.from_hf(hf_config)
    with pytest.raises(ValueError, match=message):
        validate_with_model_class(hf_config)


@pytest.mark.parametrize(
    ("name", "value"),
    [("attn_logit_softcapping", 50.0), ("final_logit_softcapping", 30.0)],
)
def test_the_model_class_rejects_logit_soft_capping(
    tmp_path: Path,
    name: str,
    value: float,
) -> None:
    # transformers 5.16.1 Qwen3.5 never reads either field; a checkpoint that declares
    # one still expects capped logits.
    hf_config = load_hf_config(_edited(_set(name, value)), tmp_path)

    with pytest.raises(
        ValueError,
        match=f"Qwen3_5ForConditionalGeneration does not implement {name}={value}",
    ):
        validate_with_model_class(hf_config)


@pytest.mark.parametrize(
    "edit",
    [
        # transformers 5.16.1 always interleaves the sections, defaults them to
        # (11, 11, 10) and never reads attn_output_gate; its config class defaults
        # partial_rotary_factor to 0.25.
        pytest.param(_drop_rope("mrope_interleaved"), id="no-interleaved"),
        pytest.param(_drop_rope("mrope_section"), id="no-sections"),
        pytest.param(_drop_rope("partial_rotary_factor"), id="no-factor"),
        pytest.param(_drop("attn_output_gate"), id="no-gate-field"),
    ],
)
def test_missing_optional_fields_take_the_reference_defaults(
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None],
) -> None:
    published = Qwen3_5Config.from_hf(load_hf_config(QWEN3_5_9B_CONFIG, tmp_path / "published"))
    edited = Qwen3_5Config.from_hf(load_hf_config(_edited(edit), tmp_path / "edited"))

    assert edited == published


@pytest.mark.parametrize("gate", ["swish", "silu"])
def test_output_gate_type_may_name_the_silu_gated_norm(tmp_path: Path, gate: str) -> None:
    # The 27B config sets "swish" and the 9B config omits the field. transformers 5.16.1
    # never reads it and gates the Gated DeltaNet norm with SiLU; vLLM reads it as that
    # gate's activation, with swish meaning SiLU.
    published = Qwen3_5Config.from_hf(load_hf_config(QWEN3_5_9B_CONFIG, tmp_path / "published"))
    edited = _edited(_set("output_gate_type", gate))

    assert Qwen3_5Config.from_hf(load_hf_config(edited, tmp_path / "edited")) == published


def test_rejects_a_config_of_another_model_type(tmp_path: Path) -> None:
    other = {**QWEN3_5_9B_CONFIG["text_config"], "model_type": "qwen3"}
    hf_config: PretrainedConfig = load_hf_config(other, tmp_path)

    with pytest.raises(ValueError, match="qwen3_5 or qwen3_5_text config, got 'qwen3'"):
        Qwen3_5Config.from_hf(hf_config)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda text: replace(text, rms_norm_eps=1e-5), "epsilon"),
        (lambda text: replace(text, tie_word_embeddings="false"), "tie_word_embeddings"),
        (lambda text: replace(text, layer_types=("full_attention",)), "LayerKind"),
        (lambda text: replace(text, hidden_size=True), "positive"),
        (lambda text: replace(text.attention, mrope_section=[11, 11, 10]), "three"),
        (lambda text: replace(text.attention, rotary_dim=258), "rotary_dim"),
        (lambda text: replace(text.attention, max_position=0), "max_position"),
        (lambda text: replace(text.linear_attention, conv_kernel_size=2), "kernel size"),
    ],
    ids=["epsilon", "tie", "layer-kind", "size", "mrope", "rotary-dim", "max-position", "conv"],
)
def test_value_objects_validate_direct_construction(
    tmp_path: Path, edit: Callable[[Qwen3_5TextConfig], object], match: str
) -> None:
    text = Qwen3_5Config.from_hf(load_hf_config(QWEN3_5_9B_CONFIG, tmp_path)).text

    with pytest.raises(ValueError, match=match):
        edit(text)


@pytest.mark.parametrize("name", ["image_token_id", "video_token_id"])
@pytest.mark.parametrize("value", [-1, True])
def test_image_placeholders_validate_direct_construction(
    tmp_path: Path, name: str, value: object
) -> None:
    config = Qwen3_5Config.from_hf(load_hf_config(QWEN3_5_9B_CONFIG, tmp_path))
    with pytest.raises(ValueError, match=name):
        replace(config, **{name: value})
