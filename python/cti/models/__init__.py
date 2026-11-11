# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/models/__init__.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from typing import TYPE_CHECKING

from cti.utils import lazy_getattr

from .config import ModelConfig
from .features import FeatureRequest, TokenHead

if TYPE_CHECKING:
    from cti.layers import Site

    from ._download import CheckpointSource
    from .base import (
        BaseLLMModel,
        ModelCapability,
    )
    from .checkpoint import CheckpointFacts, read_checkpoint
    from .identity import model_signature
    from .peft import load_peft, peft_lora, save_peft
    from .register import get_model_class
    from .weight import open_weights

# Contract and config imports must not load the layers and kernels behind these (PEP 562).
_LAZY_ATTRS = {
    "BaseLLMModel": ".base",
    "CheckpointFacts": ".checkpoint",
    "CheckpointSource": "._download",
    "peft_lora": ".peft",
    "read_checkpoint": ".checkpoint",
    "open_weights": ".weight",
    "ModelCapability": ".base",
    "get_model_class": ".register",
    "load_peft": ".peft",
    "save_peft": ".peft",
    "model_signature": ".identity",
}

__getattr__ = lazy_getattr(__name__, _LAZY_ATTRS, globals())


def create_model(
    model_class: type[BaseLLMModel],
    facts: CheckpointFacts,
    *,
    site: Site,
) -> BaseLLMModel:
    """The model ``site``'s rank builds from its checkpoint's ``facts``; its parameters follow
    the ambient device, and a vision tower normalizes pixels as the facts' image preprocessing
    says."""

    # Forward only a checkpoint's preprocessing, so a model class that lacks the option fails
    # its constructor signature instead of silently ignoring it.
    preprocessing = facts.image_preprocessing
    options = {} if preprocessing is None else {"image_preprocessing": preprocessing}
    return model_class(model_class.CONFIG.from_hf(facts.hf_config), site=site, **options)


__all__ = [
    "BaseLLMModel",
    "CheckpointFacts",
    "CheckpointSource",
    "create_model",
    "FeatureRequest",
    "get_model_class",
    "load_peft",
    "model_signature",
    "ModelCapability",
    "ModelConfig",
    "open_weights",
    "peft_lora",
    "save_peft",
    "read_checkpoint",
    "TokenHead",
]
