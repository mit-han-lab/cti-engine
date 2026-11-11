"""Qwen3.5 ``config.json`` contents (published and tiny geometries), key tables and weights."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

# Three Gated DeltaNet layers, then one full-attention layer (full_attention_interval 4).
_HYBRID_LAYER_PATTERN = ["linear_attention"] * 3 + ["full_attention"]

# Qwen/Qwen3.5-9B config.json.
QWEN3_5_9B_CONFIG: Mapping[str, Any] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "image_token_id": 248056,
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "layer_types": _HYBRID_LAYER_PATTERN * 8,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06,
        "use_cache": True,
        "vocab_size": 248320,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    },
    "tie_word_embeddings": False,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 248057,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_5",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053,
}

# Qwen/Qwen3.8-27B config.json at 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, a Qwen3.5
# (model_type qwen3_5) checkpoint with 27B geometry.
QWEN3_8_27B_CONFIG: Mapping[str, Any] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "image_token_id": 248056,
    "language_model_only": False,
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "bos_token_id": 248044,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 17408,
        "layer_types": _HYBRID_LAYER_PATTERN * 16,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "output_gate_type": "swish",
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 248320,
    },
    "tie_word_embeddings": False,
    "transformers_version": "5.8.0.dev0",
    "video_token_id": 248057,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_5",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 5120,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053,
}

# One Gated DeltaNet layer and one full-attention layer at the smallest geometry the
# engine accepts (128-wide DeltaNet heads, 256-wide attention heads).
TINY_QWEN3_5_CONFIG: Mapping[str, Any] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_text",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            # The published base, not the common 10000 default, so that parity tells
            # a model that reads rope_theta from one that ignores it.
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    },
    "vision_config": {
        "depth": 1,
        "hidden_size": 16,
        "num_heads": 2,
        "intermediate_size": 16,
        "out_hidden_size": 32,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 1,
        "num_position_embeddings": 4,
    },
}

# The tiny geometry with two layers of each kind, interleaved so that most mixers'
# indices among their kind differ from their decoder-layer index: attention layers 0
# and 3 are KV layers 0 and 1, linear-attention layers 1 and 2 are recurrent rows 0 and 1.
INTERLEAVED_QWEN3_5_CONFIG: Mapping[str, Any] = {
    **TINY_QWEN3_5_CONFIG,
    "text_config": {
        **TINY_QWEN3_5_CONFIG["text_config"],
        "num_hidden_layers": 4,
        "layer_types": ["full_attention", "linear_attention", "linear_attention", "full_attention"],
    },
}


# TINY_QWEN3_5_CONFIG with placeholder ids inside its vocabulary and a two-block vision
# tower that merges 2 x 2 patches, resampling a 4 x 4 grid of learned positions.
TINY_QWEN3_5_VL_CONFIG: Mapping[str, Any] = {
    **TINY_QWEN3_5_CONFIG,
    "image_token_id": 60,
    "video_token_id": 61,
    "vision_start_token_id": 58,
    "vision_end_token_id": 59,
    "vision_config": {
        "depth": 2,
        "hidden_size": 16,
        "num_heads": 2,
        "intermediate_size": 24,
        "out_hidden_size": 32,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 16,
    },
}


def load_hf_config(config: Mapping[str, Any], folder: Path) -> PretrainedConfig:
    """Write ``config`` as ``folder/config.json`` and load it as the engine does."""

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config, default=dict))
    return AutoConfig.from_pretrained(folder)


def text_only_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the flat ``qwen3_5_text`` config a text-only export of ``config`` saves."""

    return {
        **config["text_config"],
        "architectures": ["Qwen3_5ForCausalLM"],
        "tie_word_embeddings": config["tie_word_embeddings"],
    }


def hf_state(hf_config: PretrainedConfig) -> dict[str, torch.Tensor]:
    """The tensors a Transformers model of ``hf_config`` saves, on the meta device."""

    model_class = (
        transformers.Qwen3_5ForConditionalGeneration
        if hf_config.model_type == "qwen3_5"
        else transformers.Qwen3_5ForCausalLM
    )
    with torch.device("meta"):
        return dict(model_class(hf_config).state_dict())


def randomize_hf_weights(
    model: PreTrainedModel, *, conv_dtype: torch.dtype = torch.float32
) -> None:
    """Draw every weight at unit-scale activations; the convolution is rounded to ``conv_dtype``.

    Transformers initializes the ``1 + weight`` norms to zero and the gated norm to one,
    which would hide a norm that gets ``zero_centered`` wrong.
    """

    layer_norm_weights = {
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.LayerNorm)
    }
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            if name.endswith("linear_attn.norm.weight") or name in layer_norm_weights:
                value = 1 + 0.1 * noise
            elif name.endswith("norm.weight"):
                value = 0.1 * noise
            elif name.endswith("A_log"):
                value = torch.log(1 + 15 * torch.rand(parameter.shape, generator=generator))
            elif name.endswith("dt_bias"):
                value = noise
            elif name.endswith("conv1d.weight"):
                value = (0.5 * noise).to(conv_dtype)
            else:
                value = noise * parameter.shape[-1] ** -0.5
            parameter.copy_(value)
