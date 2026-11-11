from __future__ import annotations

import random

import pytest
import torch

from cti.multimodal.inputs import ImageSpan
from cti.multimodal.qwen3_5.rope import Qwen3_5RopeLayout
from tests.support.multimodal import hf_qwen3_5_rope_index

_MERGE = 2
_DIGEST = bytes(32)


def _random_prompt(rng: random.Random) -> tuple[list[ImageSpan], int]:
    """One to four images with optional leading and trailing text, separated by text."""

    spans = []
    cursor = rng.randrange(0, 4)
    for _ in range(rng.randrange(1, 5)):
        if spans:
            cursor += rng.randrange(1, 5)
        height, width = rng.randrange(1, 7), rng.randrange(1, 7)
        grid = (1, height * _MERGE, width * _MERGE)
        spans.append(ImageSpan(cursor, height * width, grid, _DIGEST))
        cursor += height * width
    return spans, cursor + rng.randrange(0, 4)


def _hf_positions(spans: list[ImageSpan], length: int) -> tuple[torch.Tensor, int]:
    token_types = torch.zeros(length, dtype=torch.int64)
    for span in spans:
        token_types[span.offset : span.end] = 1
    grids = torch.tensor([span.grid_thw for span in spans])
    return hf_qwen3_5_rope_index(token_types * 7, token_types, grids, _MERGE)


def test_positions_match_transformers_on_random_layouts() -> None:
    rng = random.Random(20261001)
    for _ in range(300):
        spans, length = _random_prompt(rng)
        expected, delta = _hf_positions(spans, length)

        layout = Qwen3_5RopeLayout.from_spans(spans, _MERGE)

        assert torch.equal(layout.positions(0, length), expected)
        assert layout.decode_offset == delta
        start = rng.randrange(0, length + 1)
        end = rng.randrange(start, length + 1)
        assert torch.equal(layout.positions(start, end), expected[:, start:end])
        # Transformers decodes at sequence position + delta on all three axes.
        decoded = torch.arange(length, length + 3) + delta
        assert torch.equal(layout.positions(length - 1, length + 3)[:, 1:], decoded.expand(3, -1))


def test_image_tokens_walk_the_merged_grid() -> None:
    layout = Qwen3_5RopeLayout(((2, 2, 3),))

    assert layout.decode_offset == 3 - 6
    assert layout.positions(0, 10).dtype == torch.int64
    assert layout.positions(0, 10).tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],
        [0, 1, 2, 2, 2, 3, 3, 3, 5, 6],
        [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
    ]
    assert layout.positions(4, 4).shape == (3, 0)


def _span(offset: int, height: int, width: int) -> ImageSpan:
    return ImageSpan(offset, height * width, (1, height * _MERGE, width * _MERGE), _DIGEST)


@pytest.mark.parametrize(
    "spans",
    (
        (_span(0, 1, 2), _span(2, 1, 1)),  # adjacent
        (_span(3, 1, 1), _span(0, 1, 1)),  # out of order
    ),
)
def test_layout_rejects_images_transformers_would_not_separate(spans: tuple) -> None:
    with pytest.raises(ValueError, match="separated"):
        Qwen3_5RopeLayout.from_spans(spans, _MERGE)


@pytest.mark.parametrize(
    ("span", "message"),
    (
        (ImageSpan(0, 2, (2, 2, 4), _DIGEST), "one temporal patch"),
        (ImageSpan(0, 3, (1, 3, 4), _DIGEST), "divisible"),
        (ImageSpan(0, 3, (1, 2, 4), _DIGEST), "merged grid"),
    ),
)
def test_spans_must_follow_the_qwen3_5_image_grid(span: ImageSpan, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Qwen3_5RopeLayout.from_spans([span], _MERGE)


@pytest.mark.parametrize(("start", "end"), ((-1, 2), (3, 2)))
def test_positions_reject_invalid_ranges(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="0 <= start <= end"):
        Qwen3_5RopeLayout(((0, 1, 1),)).positions(start, end)
