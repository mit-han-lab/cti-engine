# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/models/config.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import ModelConfig, RotaryConfig, num_key_value_heads

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass(frozen=True)
class Qwen3Config:
    """A Qwen3 decoder's geometry: every layer full attention and a dense SiLU MLP."""

    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> Qwen3Config:
        config = config.get_text_config()
        if config.hidden_act != "silu":
            raise ValueError(f"Qwen3 requires SiLU activation: {config.hidden_act!r}")
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_key_value_heads(config),
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
            rotary_config=RotaryConfig.from_hf(config, head_dim),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
        )

    def engine(self, architectures: tuple[str, ...]) -> ModelConfig:
        return ModelConfig(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            max_position=self.rotary_config.max_position,
            tie_word_embeddings=self.tie_word_embeddings,
            architectures=architectures,
        )


__all__ = ["Qwen3Config"]
