# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/models/config.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import reprlib
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from transformers import PretrainedConfig


class UnimplementedConfigError(ValueError):
    """A Hugging Face config field whose semantics a model class does not implement."""

    def __init__(self, architecture: str, field: str, value: Any, implemented: str) -> None:
        super().__init__(
            f"{architecture} does not implement {field}={reprlib.repr(value)}: {implemented}"
        )
        self.architecture = architecture
        self.field = field
        self.value = value


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None

    @classmethod
    def from_hf(cls, config: PretrainedConfig, head_dim: int) -> RotaryConfig:
        """The one RoPE a text ``config`` shares over its layers, rotating all of each
        ``head_dim``-wide head."""

        # Transformers normalizes legacy checkpoint fields into rope_parameters.
        # Keep that schema translation here, outside model execution and kernels.
        rope_parameters = config.rope_parameters
        if not isinstance(rope_parameters, dict) or not {"rope_type", "rope_theta"}.issubset(
            rope_parameters
        ):
            raise ValueError("the model must declare one shared RoPE configuration")
        return cls(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=config.max_position_embeddings,
            base=rope_parameters["rope_theta"],
            scaling=(
                None if rope_parameters["rope_type"] == "default" else deepcopy(rope_parameters)
            ),
        )


@dataclass(frozen=True)
class ModelConfig:
    """What the engine outside a model reads of it; each model class folds its own
    ``CONFIG`` into it (``BaseLLMModel.engine_config``)."""

    num_layers: int
    hidden_size: int
    vocab_size: int
    max_position: int
    tie_word_embeddings: bool
    architectures: tuple[str, ...]


def num_key_value_heads(config: PretrainedConfig) -> int:
    """A text ``config``'s KV heads, each shared by a whole group of query heads."""

    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    if num_kv_heads <= 0 or config.num_attention_heads % num_kv_heads:
        raise ValueError(
            f"num_attention_heads ({config.num_attention_heads}) must be a multiple of "
            f"num_key_value_heads ({num_kv_heads})"
        )
    return num_kv_heads
