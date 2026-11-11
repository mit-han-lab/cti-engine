"""Qwen3.5-MoE ``config.json`` contents (published and tiny geometries) and key tables."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
import transformers
from transformers import PretrainedConfig

from tests.support.qwen3_5 import TINY_QWEN3_5_VL_CONFIG

# Qwen/Qwen3.5-35B-A3B config.json at 59d61f3ce65a6d9863b86d2e96597125219dc754.
QWEN3_5_35B_A3B_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 512,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "router_aux_loss_coef": 0.001,
            "shared_expert_intermediate_size": 512,
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
            "model_type": "qwen3_5_moe",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }
)

# z-lab/Qwen3.5-35B-A3B-DFlash config.json at 52cb554b4995dede3e2e1bdb129cdb1f3529332b, without
# its generation and loading fields.
QWEN3_5_35B_A3B_DFLASH_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "architectures": ["DFlashDraftModel"],
        "attention_bias": False,
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
        },
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "max_position_embeddings": 262144,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 6,
        "num_key_value_heads": 8,
        "num_target_layers": 40,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "sliding_window": 4096,
        "tie_word_embeddings": False,
        "use_sliding_window": True,
        "vocab_size": 248320,
    }
)

# The published checkpoint's multi-token-prediction head: one layer whose experts, unlike
# the text model's stacked ones, are stored one by one.
QWEN3_5_35B_A3B_MTP_NAMES = (
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    *(
        f"mtp.layers.0.{module}.weight"
        for module in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "mlp.gate",
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.shared_expert.down_proj",
            "mlp.shared_expert_gate",
            *(
                f"mlp.experts.{expert}.{projection}"
                for expert in range(256)
                for projection in ("gate_proj", "up_proj", "down_proj")
            ),
        )
    ),
)

# The published checkpoint's vision tower, as its index lists it: the patch embedding, the
# learned positions, the merger and 27 blocks.
QWEN3_5_35B_A3B_VISION_NAMES = (
    "model.visual.patch_embed.proj.weight",
    "model.visual.patch_embed.proj.bias",
    "model.visual.pos_embed.weight",
    *(
        f"model.visual.{module}.{kind}"
        for module in (
            "merger.norm",
            "merger.linear_fc1",
            "merger.linear_fc2",
            *(
                f"blocks.{block}.{part}"
                for block in range(27)
                for part in (
                    "norm1",
                    "norm2",
                    "attn.qkv",
                    "attn.proj",
                    "mlp.linear_fc1",
                    "mlp.linear_fc2",
                )
            ),
        )
        for kind in ("weight", "bias")
    ),
)

# Both mixers twice, interleaved so that most mixers' indices among their kind differ from
# their decoder-layer index, at the smallest widths the engine accepts. Eight experts, two
# per token, and a shared expert of another width than the routed ones. The image
# placeholders and the vision tower are dense Qwen3.5's tiny ones, so that the image tests
# run both families on the same prompts.
TINY_QWEN3_5_MOE_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": False,
        **{
            name: TINY_QWEN3_5_VL_CONFIG[name]
            for name in (
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
                "vision_config",
            )
        },
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "layer_types": [
                "full_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 24,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        },
    }
)


def text_only_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the flat ``qwen3_5_moe_text`` config a text-only export of ``config`` saves."""

    return {
        **config["text_config"],
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "tie_word_embeddings": config["tie_word_embeddings"],
    }


def hf_state(hf_config: PretrainedConfig) -> dict[str, torch.Tensor]:
    """The tensors of a Transformers model of ``hf_config`` on the meta device, named as the
    published checkpoint stores them: the routed experts stacked."""

    model_class = (
        transformers.Qwen3_5MoeForConditionalGeneration
        if hf_config.model_type == "qwen3_5_moe"
        else transformers.Qwen3_5MoeForCausalLM
    )
    with torch.device("meta"):
        return dict(model_class(hf_config).state_dict())


__all__ = [
    "QWEN3_5_35B_A3B_CONFIG",
    "QWEN3_5_35B_A3B_DFLASH_CONFIG",
    "QWEN3_5_35B_A3B_MTP_NAMES",
    "QWEN3_5_35B_A3B_VISION_NAMES",
    "TINY_QWEN3_5_MOE_CONFIG",
    "hf_state",
    "text_only_config",
]
