# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`); see THIRD_PARTY_NOTICES.
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/vision_utils.py`); see THIRD_PARTY_NOTICES.

"""Qwen3.5 vision tower: patch embedding, bidirectional ViT blocks and the merger.

The tower is permanently frozen: it encodes prefill images, training never runs it,
and each rank holds it whole.

Patches arrive image after image, each image's patches in merge-window order: the
``m x m`` windows of the patch grid row by row, and within a window its patches row by
row, so every ``m * m`` consecutive patches merge into one token.
"""

from __future__ import annotations

from itertools import accumulate
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from cti.forward import ImageBatch
from cti.layers import (
    BaseOP,
    BidirectionalAttention,
    LayerNorm,
    LinearReplicated,
    OPList,
    VisionRotaryEmbedding,
)

if TYPE_CHECKING:
    from cti.layers import Site
    from cti.multimodal import ImagePreprocessing

    from .config import Qwen3_5VisionConfig


def _merge_window_coordinates(
    height: int,
    width: int,
    merge_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The row and column of each patch of one ``height x width`` grid, in merge-window order."""

    shape = (height // merge_size, width // merge_size, merge_size, merge_size)
    rows = torch.arange(height, device=device).view(shape[0], 1, merge_size, 1)
    columns = torch.arange(width, device=device).view(1, shape[1], 1, merge_size)
    return rows.expand(shape).flatten(), columns.expand(shape).flatten()


def _bilinear_taps(
    index: torch.Tensor,
    size: int,
    side: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two grid cells and weights that resample patch ``index`` of an axis of ``size``
    patches from a ``side``-cell axis whose corners align with the image's."""

    source = index.to(torch.float32) * (side - 1) / max(size - 1, 1)
    lower = source.floor()
    offsets = torch.arange(2, device=index.device)
    taps = (lower.long().unsqueeze(-1) + offsets).clamp_(max=side - 1)
    weights = (1 - (source - lower).unsqueeze(-1).sub(offsets).abs()).clamp_(min=0)
    return taps, weights


class Qwen3_5VisionPatchEmbed(BaseOP):
    """Normalize each image patch's uint8 row by table lookup, as the checkpoint's image
    preprocessing does, repeat the image's one frame over the patch's frames, then map the
    patch linearly in the weights' dtype. Without image preprocessing (None) it takes no
    image."""

    def __init__(
        self, config: Qwen3_5VisionConfig, preprocessing: ImagePreprocessing | None, *, site: Site
    ) -> None:
        # The checkpoint's Conv3d weight loads flattened (model.py, _flatten_patch_conv).
        self.proj = LinearReplicated(config.patch_dim, config.hidden_size, has_bias=True, site=site)
        # A patch row's uint8 values: channels, then rows and columns of its one frame.
        self.frame_dim = config.in_channels * config.patch_size**2
        self._channels = config.in_channels
        self._frames = config.temporal_patch_size
        self._levels: torch.Tensor | None = None
        if preprocessing is not None:
            if len(preprocessing.image_mean) != config.in_channels:
                raise ValueError(
                    f"image preprocessing normalizes {len(preprocessing.image_mean)} channels, "
                    f"the vision tower takes {config.in_channels}"
                )
            # Each channel's 256 levels as transformers' PIL image processor computes them:
            # rescaled in FP64, rounded to FP32, normalized in FP32 (image_transforms.py
            # rescale, normalize), then rounded to BF16 and cast to the weights' dtype, the
            # model's default here: bitwise the BF16 pixels the projection takes.
            levels = torch.arange(256, dtype=torch.float64, device="cpu")
            levels = levels * preprocessing.rescale_factor
            mean, std = (
                torch.tensor(values, dtype=torch.float32, device="cpu")[:, None]
                for values in (preprocessing.image_mean, preprocessing.image_std)
            )
            normalized = (levels.float() - mean) / std
            dtype = torch.get_default_dtype()
            self._levels = normalized.bfloat16().to(dtype).flatten().to(site.device)
            # Column c * p * p + i of a row is channel c: its levels start at row 256 * c.
            channels = torch.arange(self.frame_dim, device=site.device) // config.patch_size**2
            self._level_offsets = (256 * channels).int()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._levels is None:
            raise ValueError("the checkpoint has no image preprocessing to normalize pixels")
        index = pixel_values.int() + self._level_offsets
        frame = self._levels.index_select(0, index.flatten()).view(
            len(index), self._channels, 1, -1
        )
        # transformers' patch is channel, frame, row, column first; an image's frames repeat.
        patches = frame.expand(-1, -1, self._frames, -1).reshape(len(index), -1)
        return self.proj.forward(patches)


class Qwen3_5VisionPositionEmbedding(BaseOP):
    """A learned square grid of position embeddings, resampled bilinearly to each image."""

    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        self.weight = torch.empty(config.num_position_embeddings, config.hidden_size)
        self._side = config.position_grid_side

    def forward(
        self,
        rows: torch.Tensor,
        columns: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """FP32 embeddings ``[patches, hidden]`` of one ``height x width`` image's patches."""

        side = self._side
        row_taps, row_weights = _bilinear_taps(rows, height, side)
        column_taps, column_weights = _bilinear_taps(columns, width, side)
        cells = (row_taps.unsqueeze(-1) * side + column_taps.unsqueeze(-2)).flatten(1)
        weights = (row_weights.unsqueeze(-1) * column_weights.unsqueeze(-2)).flatten(1)
        # One tap at a time keeps one [patches, hidden] gather alive, not four at once.
        embeddings = self.weight[cells[:, 0]] * weights[:, :1]
        for tap in range(1, cells.shape[1]):
            embeddings += self.weight[cells[:, tap]] * weights[:, tap : tap + 1]
        return embeddings


class Qwen3_5VisionAttention(BaseOP):
    """Rotary self-attention among the patches of each image."""

    def __init__(
        self, config: Qwen3_5VisionConfig, *, rotary: VisionRotaryEmbedding, site: Site
    ) -> None:
        self.qkv = LinearReplicated(
            config.hidden_size, 3 * config.hidden_size, has_bias=True, site=site
        )
        self.proj = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=True, site=site
        )
        self.attention = BidirectionalAttention(config.num_heads, config.head_dim)
        self._head_shape = (3, config.num_heads, config.head_dim)
        self._rotary = rotary

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_patches: torch.Tensor,
        max_patches: int,
    ) -> torch.Tensor:
        query, key, value = self.qkv.forward(x).unflatten(-1, self._head_shape).unbind(1)
        query, key = self._rotary.forward(query, key, cos, sin)
        attended = self.attention.forward(query, key, value, cu_patches, max_patches)
        return self.proj.forward(attended)


class Qwen3_5VisionMLP(BaseOP):
    def __init__(self, config: Qwen3_5VisionConfig, *, site: Site) -> None:
        self.linear_fc1 = LinearReplicated(
            config.hidden_size,
            config.intermediate_size,
            has_bias=True,
            site=site,
        )
        self.linear_fc2 = LinearReplicated(
            config.intermediate_size,
            config.hidden_size,
            has_bias=True,
            site=site,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.linear_fc1.forward(x), approximate="tanh")
        return self.linear_fc2.forward(hidden)


class Qwen3_5VisionBlock(BaseOP):
    """A pre-norm block: attention, then the MLP, each added to the stream."""

    def __init__(
        self, config: Qwen3_5VisionConfig, *, rotary: VisionRotaryEmbedding, site: Site
    ) -> None:
        self.norm1 = LayerNorm(config.hidden_size, eps=config.LAYER_NORM_EPS)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.LAYER_NORM_EPS)
        self.attn = Qwen3_5VisionAttention(config, rotary=rotary, site=site)
        self.mlp = Qwen3_5VisionMLP(config, site=site)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_patches: torch.Tensor,
        max_patches: int,
    ) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin, cu_patches, max_patches)
        return x + self.mlp.forward(self.norm2.forward(x))


class Qwen3_5VisionMerger(BaseOP):
    """Each merge window's normalized patches, concatenated, become one text-width token."""

    def __init__(self, config: Qwen3_5VisionConfig, *, site: Site) -> None:
        window_size = config.hidden_size * config.spatial_merge_size**2
        self.norm = LayerNorm(config.hidden_size, eps=config.LAYER_NORM_EPS)
        self.linear_fc1 = LinearReplicated(window_size, window_size, has_bias=True, site=site)
        self.linear_fc2 = LinearReplicated(
            window_size, config.out_hidden_size, has_bias=True, site=site
        )
        self._window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = self.norm.forward(x).view(-1, self._window_size)
        # Exact GELU here, unlike the blocks' tanh approximation.
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(windows)))


class Qwen3_5VisionModel(BaseOP):
    """Encode a step's images into the token embeddings that replace their placeholders."""

    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        image_preprocessing: ImagePreprocessing | None,
        *,
        site: Site,
    ) -> None:
        rotary = VisionRotaryEmbedding(config.head_dim, config.ROPE_THETA, site=site)
        self.patch_embed = Qwen3_5VisionPatchEmbed(config, image_preprocessing, site=site)
        self.pos_embed = Qwen3_5VisionPositionEmbedding(config)
        self.blocks = OPList(
            [Qwen3_5VisionBlock(config, rotary=rotary, site=site) for index in range(config.depth)]
        )
        self.merger = Qwen3_5VisionMerger(config, site=site)
        self._rotary = rotary
        self.merge_size = config.spatial_merge_size

    def forward(self, images: ImageBatch) -> torch.Tensor:
        """Token embeddings ``[image tokens, out_hidden_size]``, image after image."""

        return self.encode(self.embed(images), images.cu_patches, images.max_patches)

    def embed(self, images: ImageBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The patches embedded with their positions, and their rotary cos and sin: what the
        blocks read, one row per patch.

        Every grid has one temporal patch and sides divisible by the merge size; the
        request's input processor checks both.
        """

        x = self.patch_embed.forward(images.pixel_values)
        device = x.device
        rows, columns = [], []
        offset = 0
        for frames, height, width in images.grids:
            if frames != 1:
                raise ValueError("Qwen3.5 images have exactly one temporal patch")
            image_rows, image_columns = _merge_window_coordinates(
                height, width, self.merge_size, device
            )
            rows.append(image_rows)
            columns.append(image_columns)
            end = offset + height * width
            x[offset:end].add_(
                self.pos_embed.forward(image_rows, image_columns, height, width).to(x.dtype)
            )
            offset = end
        cos, sin = self._rotary.cos_sin(torch.stack((torch.cat(rows), torch.cat(columns)), -1))
        return x, cos, sin

    def encode(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cu_patches: torch.Tensor,
        max_patches: int,
    ) -> torch.Tensor:
        """The blocks over ``embed``'s ``inputs``, each sequence of ``cu_patches`` attending
        within itself, then the merger: what a vision graph captures."""

        x, cos, sin = inputs
        for block in self.blocks.op_list:
            x = block.forward(x, cos, sin, cu_patches, max_patches)
        return self.merger.forward(x)

    def largest_image_batch(
        self, *, max_tokens: int, max_image_tokens: int, device: torch.device
    ) -> ImageBatch:
        """Blank images of ``max_tokens`` tokens: one of ``max_image_tokens`` and one of the
        rest, each a single row of merge windows, filling rows ``[0, max_tokens)``.

        Every operation's memory grows with the patches alone, the attention's included,
        so encoding this batch reaches the activation peak of any step's images.
        """

        merge = self.merge_size
        tokens = [max_image_tokens, max_tokens - max_image_tokens]
        grids = tuple((1, merge, count * merge) for count in tokens if count > 0)
        patches = [count * merge**2 for count in tokens if count > 0]
        bounds = torch.tensor([0, *accumulate(patches)], dtype=torch.int32, device=device)
        return ImageBatch(
            torch.zeros(
                (sum(patches), self.patch_embed.frame_dim),
                dtype=torch.uint8,
                device=device,
            ),
            grids,
            bounds,
            max(patches),
            *(torch.arange(max_tokens, device=device),) * 3,
            0,
        )


__all__ = ["Qwen3_5VisionModel"]
