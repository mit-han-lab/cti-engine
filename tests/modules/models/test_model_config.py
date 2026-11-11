import json
from pathlib import Path

import pytest
from transformers import AutoConfig, PretrainedConfig, Qwen3Config, Qwen3MoeConfig

from cti.models import get_model_class
from cti.models.config import ModelConfig, RotaryConfig
from cti.models.qwen3 import config as qwen3_config
from cti.models.qwen3_moe import config as qwen3_moe_config
from tests.support import quant_checkpoints
from tests.support.model_config import validate_with_model_class
from tests.support.qwen3_5 import TINY_QWEN3_5_CONFIG, load_hf_config
from tests.support.qwen3_5_moe import TINY_QWEN3_5_MOE_CONFIG


def _common_config(architecture: str) -> dict[str, object]:
    return {
        "architectures": [architecture],
        "vocab_size": 257,
        "hidden_size": 128,
        "intermediate_size": 384,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    }


def _load_checkpoint_config(
    path: Path,
    architecture: str,
    model_type: str,
    fields: dict[str, object],
) -> PretrainedConfig:
    document = {
        **_common_config(architecture),
        "model_type": model_type,
        "rope_theta": 1_000_000,
        "rope_scaling": None,
        **fields,
    }
    (path / "config.json").write_text(json.dumps(document), encoding="utf-8")
    return AutoConfig.from_pretrained(path)


def test_qwen3_config_consumes_constructor_normalized_default_rope() -> None:
    hf_config = Qwen3Config(
        **_common_config("Qwen3ForCausalLM"),
        head_dim=32,
        rope_theta=250_000.0,
    )

    assert hf_config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 250_000.0,
    }

    config = qwen3_config.Qwen3Config.from_hf(hf_config)

    assert config.rotary_config == RotaryConfig(
        head_dim=32,
        rotary_dim=32,
        max_position=512,
        base=250_000.0,
        scaling=None,
    )
    assert config.num_layers == 3
    assert config.num_qo_heads == 4
    assert config.num_kv_heads == 2
    assert config.hidden_size == 128
    assert config.vocab_size == 257
    assert config.intermediate_size == 384
    assert config.rms_norm_eps == pytest.approx(1e-5)
    assert config.tie_word_embeddings is True
    assert get_model_class("Qwen3ForCausalLM").engine_config(hf_config) == ModelConfig(
        num_layers=3,
        hidden_size=128,
        vocab_size=257,
        max_position=512,
        tie_word_embeddings=True,
        architectures=("Qwen3ForCausalLM",),
    )


def test_qwen3_config_rejects_an_activation_other_than_silu() -> None:
    hf_config = Qwen3Config(**{**_common_config("Qwen3ForCausalLM"), "hidden_act": "gelu"})

    with pytest.raises(ValueError, match="Qwen3 requires SiLU activation: 'gelu'"):
        qwen3_config.Qwen3Config.from_hf(hf_config)


def test_qwen3_moe_config_consumes_from_dict_normalized_llama3_rope() -> None:
    hf_config = Qwen3MoeConfig.from_dict(
        {
            **_common_config("Qwen3MoeForCausalLM"),
            "model_type": "qwen3_moe",
            "rope_theta": 750_000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 2.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 256,
            },
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 96,
            "norm_topk_prob": True,
        }
    )
    normalized_rope = {
        "rope_type": "llama3",
        "factor": 2.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 256,
        "rope_theta": 750_000.0,
    }

    assert hf_config.rope_parameters == normalized_rope

    config = qwen3_moe_config.Qwen3MoeConfig.from_hf(hf_config)

    assert config.rotary_config == RotaryConfig(
        head_dim=32,
        rotary_dim=32,
        max_position=512,
        base=750_000.0,
        scaling=normalized_rope,
    )
    assert config.num_experts == 8
    assert config.num_experts_per_tok == 2
    assert config.moe_intermediate_size == 96
    assert config.norm_topk_prob is True


@pytest.mark.parametrize(
    "document", [TINY_QWEN3_5_CONFIG, TINY_QWEN3_5_MOE_CONFIG], ids=["qwen3_5", "qwen3_5_moe"]
)
def test_qwen3_5_models_fold_their_text_config_for_the_engine(
    tmp_path: Path, document: dict
) -> None:
    hf_config = load_hf_config(document, tmp_path)
    model_class = get_model_class(hf_config.architectures[0])
    text = model_class.CONFIG.from_hf(hf_config).text

    assert model_class.engine_config(hf_config) == ModelConfig(
        num_layers=len(text.layer_types),
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        max_position=text.attention.max_position,
        tie_word_embeddings=text.tie_word_embeddings,
        architectures=tuple(document["architectures"]),
    )


def test_qwen3_config_uses_nested_text_config_without_mutating_hf_inputs() -> None:
    text_config = Qwen3Config(
        **_common_config("NestedArchitectureMustNotEscape"),
        head_dim=32,
        rope_parameters={
            "rope_type": "llama3",
            "factor": 2.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 256,
            "rope_theta": 500_000.0,
        },
    )
    text_config.architectures = None
    root_config = PretrainedConfig(
        architectures=["Qwen3ForConditionalGeneration"],
        text_config=text_config,
    )
    root_architectures = root_config.architectures
    hf_rope_parameters = text_config.rope_parameters
    expected_rope_parameters = dict(hf_rope_parameters)

    config = qwen3_config.Qwen3Config.from_hf(root_config)

    assert config.hidden_size == text_config.hidden_size
    assert root_config.architectures is root_architectures
    assert text_config.architectures is None
    assert text_config.rope_parameters is hf_rope_parameters
    assert text_config.rope_parameters == expected_rope_parameters
    assert config.rotary_config.scaling == expected_rope_parameters
    assert config.rotary_config.scaling is not hf_rope_parameters


def test_qwen3_config_rejects_per_layer_rope_parameters() -> None:
    hf_config = Qwen3Config(
        **_common_config("Qwen3ForCausalLM"),
        head_dim=32,
    )
    hf_config.rope_parameters = {
        "full_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    }

    with pytest.raises(ValueError, match="one shared RoPE configuration"):
        qwen3_config.Qwen3Config.from_hf(hf_config)


def test_qwen3_config_rejects_query_heads_not_a_multiple_of_kv_heads() -> None:
    hf_config = Qwen3Config(
        **{**_common_config("Qwen3ForCausalLM"), "num_key_value_heads": 3},
        head_dim=32,
    )

    with pytest.raises(ValueError, match=r"num_attention_heads \(4\) must be a multiple of "):
        qwen3_config.Qwen3Config.from_hf(hf_config)


@pytest.mark.parametrize(
    ("architecture", "model_type", "fields"),
    [
        # Attention and MoE fields as published by Qwen/Qwen3-0.6B and
        # Qwen/Qwen3-30B-A3B.
        pytest.param(
            "Qwen3ForCausalLM",
            "qwen3",
            {"sliding_window": None, "use_sliding_window": False, "max_window_layers": 28},
            id="qwen3-0.6b",
        ),
        pytest.param(
            "Qwen3MoeForCausalLM",
            "qwen3_moe",
            {
                "sliding_window": None,
                "use_sliding_window": False,
                "max_window_layers": 48,
                "mlp_only_layers": [],
                "decoder_sparse_step": 1,
            },
            id="qwen3-30b-a3b",
        ),
        pytest.param(
            "Qwen3ForCausalLM",
            "qwen3",
            {"sliding_window": 4096, "use_sliding_window": False},
            id="qwen3-disabled-window",
        ),
        pytest.param(
            "Qwen3MoeForCausalLM",
            "qwen3_moe",
            {"sliding_window": 4096, "use_sliding_window": False},
            id="qwen3-moe-disabled-window",
        ),
        # Qwen3 windows only layers from max_window_layers on; with 3 layers there
        # are none.
        pytest.param(
            "Qwen3ForCausalLM",
            "qwen3",
            {"sliding_window": 4096, "use_sliding_window": True, "max_window_layers": 3},
            id="qwen3-no-windowed-layer",
        ),
    ],
)
def test_model_classes_accept_checkpoints_they_implement(
    tmp_path: Path,
    architecture: str,
    model_type: str,
    fields: dict[str, object],
) -> None:
    hf_config = _load_checkpoint_config(tmp_path, architecture, model_type, fields)

    validate_with_model_class(hf_config)
    assert get_model_class(architecture).engine_config(hf_config).architectures == (architecture,)


@pytest.mark.parametrize(
    ("architecture", "model_type", "fields", "message"),
    [
        pytest.param(
            "Qwen3ForCausalLM",
            "qwen3",
            {"sliding_window": 4096, "use_sliding_window": True, "max_window_layers": 1},
            r"Qwen3ForCausalLM does not implement layer_types \['sliding_attention'\]",
            id="qwen3-windowed-layers",
        ),
        pytest.param(
            "Qwen3ForCausalLM",
            "qwen3",
            {"layer_types": ["full_attention", "chunked_attention", "full_attention"]},
            r"Qwen3ForCausalLM does not implement layer_types \['chunked_attention'\]",
            id="qwen3-chunked-layer",
        ),
        *(
            pytest.param(
                architecture,
                model_type,
                {name: value},
                f"{architecture} does not implement {name}={value}",
                id=f"{model_type}-{name}",
            )
            # Qwen3-MoE's refusals: tests/modules/models/test_qwen3_moe_model.py.
            for architecture, model_type in (("Qwen3ForCausalLM", "qwen3"),)
            for name, value in (
                ("attention_bias", True),
                ("attn_logit_softcapping", 50.0),
                ("final_logit_softcapping", 30.0),
            )
        ),
    ],
)
def test_model_classes_reject_semantics_they_do_not_implement(
    tmp_path: Path,
    architecture: str,
    model_type: str,
    fields: dict[str, object],
    message: str,
) -> None:
    hf_config = _load_checkpoint_config(tmp_path, architecture, model_type, fields)

    with pytest.raises(ValueError, match=message):
        validate_with_model_class(hf_config)


@pytest.mark.parametrize("stem", ["qwen3_8_27b_fp8", "qwen3_8_27b_nvfp4", "qwen3_5_35b_a3b_fp8"])
def test_the_model_configs_of_quantized_checkpoints_read_their_geometry(
    tmp_path: Path, stem: str
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(quant_checkpoints.load(stem)[0]["config.json"])
    )

    hf_config = AutoConfig.from_pretrained(tmp_path)

    architecture = hf_config.architectures[0]
    assert architecture.endswith("ForConditionalGeneration")
    assert get_model_class(architecture).engine_config(hf_config).num_layers > 0
