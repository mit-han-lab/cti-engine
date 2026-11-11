"""Pinned Qwen3.5 checkpoints, workload sizes and receipts of the GPU qualification tests.

Settings: ``CTI_QWEN35_*`` (``tests/hf_parity/README.md``); unset, the tests run
``Qwen/Qwen3.5-9B`` and fail when the host has not cached it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.support.checkpoints import (
    QWEN3_5_0_8B,
    QWEN3_5_2B,
    QWEN3_5_4B,
    QWEN3_5_9B,
    QWEN3_5_35B_A3B,
    QWEN3_8_27B,
    Checkpoint,
)
from tests.support.hf_parity import ParitySettings

PINNED_CHECKPOINTS = (
    QWEN3_5_9B,
    QWEN3_8_27B,
    QWEN3_5_35B_A3B,
    QWEN3_5_0_8B,
    QWEN3_5_2B,
    QWEN3_5_4B,
)


# The CTI_QWEN35_* settings (tests/hf_parity/README.md).
SETTINGS = ParitySettings("CTI_QWEN35_")


def selected_checkpoint() -> Checkpoint:
    """The pinned checkpoint the runner selected, ``Qwen/Qwen3.5-9B`` by default."""

    return SETTINGS.checkpoint(PINNED_CHECKPOINTS)


def max_running_requests() -> int:
    """Request slots the engine allocates (its recurrent-state pool rows)."""

    return SETTINGS.positive_int("MAX_RUNNING_REQ", 16)


def greedy_tokens() -> int:
    """Tokens of each greedy continuation; generation never stops at EOS."""

    return SETTINGS.positive_int("GREEDY_TOKENS", 256)


def residency_cycles() -> int:
    return SETTINGS.positive_int("RESIDENCY_CYCLES", 10)


def receipts_dir() -> Path | None:
    """The runner's receipt directory, if it named one."""

    return SETTINGS.path("RECEIPTS")


def write_receipt(name: str, receipt: Mapping[str, Any]) -> None:
    """Keep ``receipt`` as ``<name>.json`` in the runner's receipt directory, if any."""

    directory = receipts_dir()
    if directory is None:
        return
    path = directory / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def save_random_checkpoint(
    path: Path,
    *,
    hidden_size: int,
    layer_types: Sequence[str],
    vocab_size: int,
) -> None:
    """Save a random BF16 image-text Qwen3.5 checkpoint of ``layer_types`` at ``hidden_size``.

    Attention has four query and two KV heads of 256 channels, the Gated DeltaNet two key
    and four value heads of 128, the MLP 512 channels and the vision tower one block.
    ``randomize_hf_weights`` draws every weight from a fixed seed, so equal arguments save equal
    checkpoints.
    """

    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from tests.support.qwen3_5 import randomize_hf_weights

    config = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        tie_word_embeddings=False,
        text_config={
            "hidden_size": hidden_size,
            "intermediate_size": 512,
            "num_hidden_layers": len(layer_types),
            "layer_types": list(layer_types),
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": vocab_size,
            "max_position_embeddings": 1024,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "num_heads": 2,
            "intermediate_size": 16,
            "out_hidden_size": hidden_size,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 1,
            "num_position_embeddings": 4,
        },
    )
    model = Qwen3_5ForConditionalGeneration(config)
    randomize_hf_weights(model, conv_dtype=torch.bfloat16)
    model.to(torch.bfloat16).save_pretrained(path)


__all__ = [
    "SETTINGS",
    "PINNED_CHECKPOINTS",
    "greedy_tokens",
    "max_running_requests",
    "receipts_dir",
    "residency_cycles",
    "save_random_checkpoint",
    "selected_checkpoint",
    "write_receipt",
]
