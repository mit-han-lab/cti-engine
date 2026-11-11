"""Qwen3-MoE ``config.json`` contents (published and tiny), the published checkpoints' key tables
and a tiny random checkpoint saved by Transformers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from tests.support.checkpoints import (
    QWEN3_30B_A3B,
    QWEN3_30B_A3B_BASE,
    QWEN3_30B_A3B_INSTRUCT_2507,
    QWEN3_30B_A3B_THINKING_2507,
    QWEN3_235B_A22B,
    QWEN3_235B_A22B_INSTRUCT_2507,
    QWEN3_CODER_30B_A3B_INSTRUCT,
    Checkpoint,
)

# Qwen/Qwen3-30B-A3B config.json at ad44e777bcd18fa416d9da3bd8f70d33ebb85d39.
QWEN3_30B_A3B_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "max_position_embeddings": 40960,
        "max_window_layers": 48,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 768,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "output_router_logits": False,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "router_aux_loss_coef": 0.001,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 151936,
    }
)
# The 235B-A22B geometry, as the published config.json differs from the 30B one.
_235B_A22B = MappingProxyType(
    {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "max_window_layers": 94,
        "moe_intermediate_size": 1536,
        "num_attention_heads": 64,
        "num_hidden_layers": 94,
    }
)
# The YaRN override Qwen/Qwen3-30B-A3B's README gives for 131072-token contexts.
YARN_OVERRIDE: Mapping[str, Any] = MappingProxyType(
    {
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
    }
)
# The chat checkpoints' generation_config.json and tokenizer_config.json EOS.
_CHAT_EOS = ((151645, 151643), "<|im_end|>")
# The added tokens an EOS can name, as tokenizer_config.json's added_tokens_decoder lists them.
EOS_TOKEN_IDS = MappingProxyType({"<|endoftext|>": 151643, "<|im_end|>": 151645})


@dataclass(frozen=True)
class PublishedCheckpoint:
    """A published Qwen3-MoE checkpoint: its pin, its config.json as the 30B-A3B one plus
    ``changes``, and its EOS ids, ``generation_config.json``'s and ``tokenizer_config.json``'s."""

    pin: Checkpoint
    changes: Mapping[str, Any]
    generation_eos: tuple[int, ...]
    tokenizer_eos: str

    @property
    def config(self) -> dict[str, Any]:
        return {**QWEN3_30B_A3B_CONFIG, **self.changes}


_2507 = MappingProxyType({"max_position_embeddings": 262144, "rope_theta": 10000000})
PUBLISHED = MappingProxyType(
    {
        "30b-a3b": PublishedCheckpoint(
            QWEN3_30B_A3B,
            {},
            *_CHAT_EOS,
        ),
        "30b-a3b-instruct-2507": PublishedCheckpoint(
            QWEN3_30B_A3B_INSTRUCT_2507,
            _2507,
            *_CHAT_EOS,
        ),
        "30b-a3b-thinking-2507": PublishedCheckpoint(
            QWEN3_30B_A3B_THINKING_2507,
            _2507,
            *_CHAT_EOS,
        ),
        "30b-a3b-base": PublishedCheckpoint(
            QWEN3_30B_A3B_BASE,
            {"eos_token_id": 151643, "max_position_embeddings": 32768},
            (151643,),
            "<|endoftext|>",
        ),
        "coder-30b-a3b-instruct": PublishedCheckpoint(
            QWEN3_CODER_30B_A3B_INSTRUCT,
            {**_2507, "transformers_version": "4.52.3"},
            *_CHAT_EOS,
        ),
        "235b-a22b": PublishedCheckpoint(
            QWEN3_235B_A22B,
            _235B_A22B,
            *_CHAT_EOS,
        ),
        "235b-a22b-instruct-2507": PublishedCheckpoint(
            QWEN3_235B_A22B_INSTRUCT_2507,
            {**_235B_A22B, "max_position_embeddings": 262144, "rope_theta": 5000000},
            *_CHAT_EOS,
        ),
    }
)
# What Qwen's config_1m.json adds to the 2507 checkpoints: dual-chunk attention for 1M
# contexts, its sparse prefill budgets left out.
DUAL_CHUNK_ATTENTION_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "chunk_size": 131072,
        "local_size": 4096,
        "original_max_position_embeddings": 131072,
        "sparse_attention_enabled": True,
    }
)

# Every published Qwen3-MoE checkpoint stores these tensors per decoder layer and expert; each
# 30B-A3B variant has the 30B key table, each 235B-A22B variant the 235B one.
_KEY_TEMPLATES = (
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
    *(
        f"model.layers.{{layer}}.{name}.weight"
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "mlp.gate",
            *(f"self_attn.{part}" for part in ("q_proj", "k_proj", "v_proj", "o_proj")),
            "self_attn.q_norm",
            "self_attn.k_norm",
        )
    ),
    *(
        f"model.layers.{{layer}}.mlp.experts.{{expert}}.{part}.weight"
        for part in ("gate_proj", "up_proj", "down_proj")
    ),
)
# sha256 of the published model.safetensors.index.json's sorted weight_map keys, joined by
# newlines: the 30B-A3B and 235B-A22B tables.
KEY_TABLE_SHA256 = MappingProxyType(
    {
        48: "2dc20e7e2721e2db383851257df6af98df3705c28db667bd2222f7310a7b97a8",
        94: "9ce2be306f62fa2b7c19a7c0b8513d0442ef786406283339ecd74a8febd92627",
    }
)

# Eight experts, two per token, over three sparse layers at the smallest head width the engine's
# rotary accepts. Untied, as every published Qwen3-MoE checkpoint is.
TINY_QWEN3_MOE_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "max_position_embeddings": 256,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "norm_topk_prob": True,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": False,
    }
)


def published_keys(config: Mapping[str, Any]) -> list[str]:
    """The sorted tensor names of a published checkpoint of ``config``'s geometry, checked
    against the published index's key table."""

    keys = sorted(_expand_keys(config["num_hidden_layers"], config["num_experts"]))
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    if digest != KEY_TABLE_SHA256[config["num_hidden_layers"]]:
        raise ValueError("the key templates no longer reproduce the published key table")
    return keys


def published_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """The shape of each published tensor of ``config``'s geometry, by name, checked against
    the published key table."""

    shapes = checkpoint_shapes(config)
    assert list(shapes) == published_keys(config)
    return shapes


def checkpoint_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """The shape of each tensor a checkpoint of ``config`` stores in the published layout, one
    expert's projections apart, by name."""

    hidden, head_dim = config["hidden_size"], config["head_dim"]
    queries = config["num_attention_heads"] * head_dim
    keys_values = config["num_key_value_heads"] * head_dim
    intermediate = config["moe_intermediate_size"]
    by_module = {
        "lm_head": (config["vocab_size"], hidden),
        "embed_tokens": (config["vocab_size"], hidden),
        "norm": (hidden,),
        "input_layernorm": (hidden,),
        "post_attention_layernorm": (hidden,),
        "gate": (config["num_experts"], hidden),
        "q_proj": (queries, hidden),
        "k_proj": (keys_values, hidden),
        "v_proj": (keys_values, hidden),
        "o_proj": (hidden, queries),
        "q_norm": (head_dim,),
        "k_norm": (head_dim,),
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
    # Every published tensor is a module's weight, named by the module.
    keys = _expand_keys(config["num_hidden_layers"], config["num_experts"])
    return {name: by_module[name.split(".")[-2]] for name in sorted(keys)}


def randomize_weights(model: PreTrainedModel) -> None:
    """Draw every weight at unit-scale activations from a fixed seed. The norms scale by
    ``1 + 0.1 * noise``: Transformers initializes them to one, which would hide a norm that
    gets ``zero_centered`` wrong."""

    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            if name.endswith("norm.weight"):
                value = 1 + 0.1 * noise
            elif name.endswith("bias"):
                value = 0.1 * noise
            else:
                value = noise * parameter.shape[-1] ** -0.5
            parameter.copy_(value)


def save_tiny_checkpoint(folder: Path, **changes: Any) -> PreTrainedModel:
    """Save ``TINY_QWEN3_MOE_CONFIG`` with ``changes`` and random weights to ``folder`` as
    Transformers does; return the FP32 reference model, eager attention and experts."""

    folder.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.for_model(**{**TINY_QWEN3_MOE_CONFIG, **changes})
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="eager", experts_implementation="eager"
    ).eval()
    randomize_weights(model)
    model.save_pretrained(folder)
    return model


def _expand_keys(num_layers: int, num_experts: int) -> Iterator[str]:
    for template in _KEY_TEMPLATES:
        if "{layer}" not in template:
            yield template
            continue
        for layer in range(num_layers):
            experts = range(num_experts) if "{expert}" in template else (None,)
            for expert in experts:
                yield template.format(layer=layer, expert=expert)


__all__ = [
    "DUAL_CHUNK_ATTENTION_CONFIG",
    "EOS_TOKEN_IDS",
    "KEY_TABLE_SHA256",
    "PUBLISHED",
    "QWEN3_30B_A3B_CONFIG",
    "TINY_QWEN3_MOE_CONFIG",
    "YARN_OVERRIDE",
    "PublishedCheckpoint",
    "checkpoint_shapes",
    "published_keys",
    "published_shapes",
    "randomize_weights",
    "save_tiny_checkpoint",
]
