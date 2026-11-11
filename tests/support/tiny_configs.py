"""The smallest Hugging Face config of each registered architecture that the engine builds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import Qwen3Config, Qwen3MoeConfig

from tests.support import qwen3_5_moe
from tests.support.qwen3_5 import TINY_QWEN3_5_CONFIG, load_hf_config, text_only_config

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PretrainedConfig

# Dense Qwen3 geometry, shared by the tiny Qwen3-MoE config.
TINY_GEOMETRY = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "max_position_embeddings": 64,
}


def tiny_hf_config(architecture: str, folder: Path) -> PretrainedConfig:
    """A Qwen3.5 config is written to and loaded from ``folder``."""

    if architecture == "Qwen3ForCausalLM":
        return Qwen3Config(**TINY_GEOMETRY, architectures=[architecture])
    if architecture == "Qwen3MoeForCausalLM":
        return Qwen3MoeConfig(
            **TINY_GEOMETRY,
            moe_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=2,
            architectures=[architecture],
        )
    if architecture == "Qwen3_5ForConditionalGeneration":
        return load_hf_config(TINY_QWEN3_5_CONFIG, folder)
    if architecture == "Qwen3_5ForCausalLM":
        return load_hf_config(text_only_config(TINY_QWEN3_5_CONFIG), folder)
    tiny_moe = qwen3_5_moe.TINY_QWEN3_5_MOE_CONFIG
    if architecture == "Qwen3_5MoeForConditionalGeneration":
        return load_hf_config(tiny_moe, folder)
    assert architecture == "Qwen3_5MoeForCausalLM", f"no tiny config for {architecture}"
    return load_hf_config(qwen3_5_moe.text_only_config(tiny_moe), folder)
