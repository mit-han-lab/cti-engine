"""Transformers oracles shared by the multimodal frontend tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from cti.multimodal import ImagePreprocessing

# The values of Qwen3.5's preprocessor_config.json.
QWEN3_5_PREPROCESSOR_CONFIG: Mapping[str, object] = MappingProxyType(
    {
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
)
# What the engine reads of that preprocessing (rescale_factor is the image processor's
# default; 65536 and 16777216 pixels over 32 x 32 a token): a test that builds a model from
# config.json alone passes it.
QWEN3_5_IMAGE_PREPROCESSING = ImagePreprocessing(
    min_image_tokens=64,
    max_image_tokens=16384,
    rescale_factor=1 / 255,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
)


def hf_pixel_values(rows: torch.Tensor, temporal_patch_size: int) -> torch.Tensor:
    """transformers' FP32 ``pixel_values`` for the engine's uint8 patch rows: each level
    rescaled in FP64, rounded to FP32 and normalized in FP32 (image_transforms.py), each
    patch repeated over its frames."""

    preprocessing = QWEN3_5_IMAGE_PREPROCESSING
    levels = (rows.double() * preprocessing.rescale_factor).float()
    mean, std = (
        torch.tensor(values, dtype=torch.float32)[:, None]
        for values in (preprocessing.image_mean, preprocessing.image_std)
    )
    patches = (levels.view(len(rows), len(mean), -1) - mean) / std
    frames = patches[:, :, None].expand(-1, -1, temporal_patch_size, -1)
    return frames.reshape(len(rows), -1)


def save_qwen3_5_preprocessor_config(path: Path, vision_config: object) -> None:
    """Write the ``preprocessor_config.json`` a released Qwen3.5 checkpoint has into
    ``path``, with the patch and merge sizes of ``vision_config``."""

    config = {
        **QWEN3_5_PREPROCESSOR_CONFIG,
        "patch_size": vision_config.patch_size,
        "temporal_patch_size": vision_config.temporal_patch_size,
        "merge_size": vision_config.spatial_merge_size,
    }
    (path / "preprocessor_config.json").write_text(json.dumps(config))


def hf_qwen3_5_rope_index(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, int]:
    """Qwen3.5's M-RoPE positions ``[3, T]`` and decode delta for one prompt."""

    # get_rope_index reads only the merge size and its vision-position helper.
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size)
        ),
        get_vision_position_ids=partial(Qwen3_5Model.get_vision_position_ids, None),
    )
    positions, deltas = Qwen3_5Model.get_rope_index(
        model, input_ids[None], mm_token_type_ids[None], image_grid_thw=image_grid_thw
    )
    return positions[:, 0], int(deltas.item())
