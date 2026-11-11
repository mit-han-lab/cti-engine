# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`); see THIRD_PARTY_NOTICES.

"""Qwen3.5 rotary positions of a prompt holding images."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ..inputs import ImageSpan
from ..rope import RopePositionLayout


@dataclass(frozen=True, slots=True)
class Qwen3_5RopeLayout(RopePositionLayout):
    """Temporal, height and width rotary positions of a Qwen3.5 prompt and the tokens after it.

    Rewritten from the Qwen3.5 rule in transformers 5.16.1 (``get_rope_index``,
    ``models/qwen3_5/modeling_qwen3_5.py:1240-1383``). A text token has one position on
    all three axes. The image token at row ``h`` and column ``w`` of the merged grid sits
    at ``(start, start + h, start + w)``, where ``start`` is the position its first token
    would have as text, and the text after the image resumes at
    ``start + max(height, width)``. Every token after the last image, decoded tokens
    included, is text at its sequence index plus ``decode_offset`` (vLLM's
    ``mrope_position_delta``).
    """

    # (offset, height, width) of each image: first placeholder token and merged grid.
    images: tuple[tuple[int, int, int], ...]
    decode_offset: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decode_offset",
            sum(max(height, width) - height * width for _, height, width in self.images),
        )

    @classmethod
    def from_spans(cls, spans: Sequence[ImageSpan], spatial_merge_size: int) -> Qwen3_5RopeLayout:
        images = []
        previous_end = -1
        for span in spans:
            # Transformers groups adjacent image tokens into one image, so text must separate them.
            if span.offset <= previous_end:
                raise ValueError("images must be ordered and separated by at least one text token")
            previous_end = span.end
            frames, height, width = span.grid_thw
            # Transformers also splits video grids into one-frame groups before this rule.
            if frames != 1:
                raise ValueError("Qwen3.5 images have exactly one temporal patch")
            if height % spatial_merge_size or width % spatial_merge_size:
                raise ValueError("image grid must be divisible by the spatial merge size")
            height //= spatial_merge_size
            width //= spatial_merge_size
            if span.num_tokens != height * width:
                raise ValueError("image span length must equal its merged grid size")
            images.append((span.offset, height, width))
        return cls(tuple(images))

    def positions(self, start: int, end: int) -> torch.Tensor:
        """Positions of tokens ``[start, end)`` as int64 ``[3, end - start]``."""

        if not 0 <= start <= end:
            raise ValueError("token range must satisfy 0 <= start <= end")
        positions = torch.empty((3, end - start), dtype=torch.int64)
        text_start = shift = 0
        for offset, height, width in self.images:
            if offset >= end:
                break
            self._fill_text(positions, start, text_start, offset, shift)
            rope_start = offset + shift
            text_start = offset + height * width
            first, last = max(offset, start), min(text_start, end)
            if first < last:
                raster = torch.arange(first - offset, last - offset)
                columns = positions[:, first - start : last - start]
                columns[0] = rope_start
                columns[1] = rope_start + raster // width
                columns[2] = rope_start + raster % width
            shift += max(height, width) - height * width
        self._fill_text(positions, start, text_start, end, shift)
        return positions

    @staticmethod
    def _fill_text(positions: torch.Tensor, start: int, first: int, last: int, shift: int) -> None:
        first = max(first, start)
        last = min(last, start + positions.shape[1])
        if first < last:
            positions[:, first - start : last - start] = torch.arange(first + shift, last + shift)
