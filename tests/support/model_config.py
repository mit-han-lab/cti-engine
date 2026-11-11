"""HF-config checks shared by the model config tests."""

from __future__ import annotations

from transformers import PretrainedConfig

from cti.models import get_model_class


def validate_with_model_class(hf_config: PretrainedConfig) -> None:
    """Resolution's check before any device state: the class hook reads the text config."""

    model_class = get_model_class(hf_config.architectures[0])
    model_class.validate_hf_config(hf_config.get_text_config())
