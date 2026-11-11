# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/models/register.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType

from .base import BaseLLMModel

# The model classes the engine serves, by their checkpoint's architecture: module and class,
# loaded on first use.
MODELS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "Qwen3ForCausalLM": (".qwen3.model", "Qwen3ForCausalLM"),
        "Qwen3MoeForCausalLM": (".qwen3_moe.model", "Qwen3MoeForCausalLM"),
        "Qwen3_5ForConditionalGeneration": (
            ".qwen3_5.model",
            "Qwen3_5ForConditionalGeneration",
        ),
        "Qwen3_5MoeForConditionalGeneration": (
            ".qwen3_5_moe.model",
            "Qwen3_5MoeForConditionalGeneration",
        ),
        # Text-only exports of the same checkpoints.
        "Qwen3_5ForCausalLM": (".qwen3_5.model", "Qwen3_5ForConditionalGeneration"),
        "Qwen3_5MoeForCausalLM": (".qwen3_5_moe.model", "Qwen3_5MoeForConditionalGeneration"),
    }
)


# Each image-text architecture's text-only export: the same class without a vision tower.
TEXT_ONLY_ARCHITECTURES: Mapping[str, str] = MappingProxyType(
    {
        "Qwen3_5ForConditionalGeneration": "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeForCausalLM",
    }
)


def get_model_class(architecture: str) -> type[BaseLLMModel]:
    if architecture not in MODELS:
        raise ValueError(f"Model architecture {architecture} not supported")
    module_path, class_name = MODELS[architecture]
    return getattr(importlib.import_module(module_path, package=__package__), class_name)


__all__ = ["MODELS", "TEXT_ONLY_ARCHITECTURES", "get_model_class"]
