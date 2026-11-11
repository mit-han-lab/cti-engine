from __future__ import annotations

import pytest
import torch

from cti.forward import ImageBatch
from cti.multimodal.inputs import ImageInputs, ImageSpan

_DIGEST = bytes(range(32))
_PATCH_DIM = 12


def _span(offset: int, height: int = 2, width: int = 4) -> ImageSpan:
    return ImageSpan(offset, height * width // 4, (1, height, width), _DIGEST)


def _pixels(rows: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    return torch.randint(256, (rows, _PATCH_DIM)).to(dtype)


def test_image_span_is_a_hashable_value() -> None:
    span = _span(3)

    assert span.end == 5
    assert span.num_patches == 8
    assert span == _span(3)
    assert len({span, _span(3), _span(4)}) == 2


@pytest.mark.parametrize(
    ("offset", "num_tokens", "grid_thw", "digest", "message"),
    (
        (-1, 2, (1, 2, 4), _DIGEST, "offset"),
        (True, 2, (1, 2, 4), _DIGEST, "offset"),
        (0, 0, (1, 2, 4), _DIGEST, "at least one token"),
        (0, 2, [1, 2, 4], _DIGEST, "grid"),
        (0, 2, (1, 0, 4), _DIGEST, "grid"),
        (0, 2, (1, 2), _DIGEST, "grid"),
        (0, 2, (1, 2, 4), _DIGEST[:31], "32 bytes"),
        (0, 2, (1, 2, 4), _DIGEST.hex(), "32 bytes"),
    ),
)
def test_image_span_rejects_invalid_fields(
    offset: object, num_tokens: object, grid_thw: object, digest: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ImageSpan(offset, num_tokens, grid_thw, digest)


def test_image_inputs_hold_ordered_spans_and_their_patches() -> None:
    spans = (_span(0), _span(3, 4, 4))
    inputs = ImageInputs(_pixels(8 + 16), spans)

    assert inputs.spans == spans


@pytest.mark.parametrize(
    ("pixel_values", "spans", "message"),
    (
        (_pixels(8), (), "nonempty tuple"),
        (_pixels(8), [_span(0)], "nonempty tuple"),
        (_pixels(16), (_span(0), _span(1)), "must not overlap"),
        (_pixels(16), (_span(5), _span(0)), "must not overlap"),
        (_pixels(8, torch.bfloat16), (_span(0),), "uint8"),
        (_pixels(8).reshape(-1), (_span(0),), "two-dimensional"),
        (_pixels(9), (_span(0),), "pixel rows"),
    ),
)
def test_image_inputs_reject_broken_invariants(
    pixel_values: torch.Tensor, spans: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ImageInputs(pixel_values, spans)


def test_image_spans_may_touch() -> None:
    ImageInputs(_pixels(16), (_span(0), _span(2)))


def test_an_image_batch_round_trips_its_packed_images() -> None:
    first, second, third = _pixels(8), _pixels(20)[2:18], _pixels(4)
    grids = ((1, 2, 4), (1, 4, 4), (1, 2, 2))

    batch = ImageBatch.unpack(
        ImageBatch.pack(
            [(first, grids[0]), (second, grids[1]), (third, grids[2])],
            range(6),
            [4, 0, 1],
            [9, 3, 4],
            cached=1,
        )
    )

    assert torch.equal(batch.pixel_values, torch.cat([first, second, third]))
    assert batch.grids == grids
    assert batch.cu_patches.dtype == torch.int32
    assert batch.cu_patches.tolist() == [0, 8, 24, 28]
    assert batch.max_patches == 16
    rows = (batch.store_rows, batch.src_rows, batch.dst)
    assert [(index.dtype, index.tolist()) for index in rows] == [
        (torch.int64, [*range(6)]),
        (torch.int64, [4, 0, 1]),
        (torch.int64, [9, 3, 4]),
    ]
    assert batch.cached == 1
    moved = batch.to(torch.device("cpu"))
    assert moved.grids == batch.grids and torch.equal(moved.pixel_values, batch.pixel_values)
    assert torch.equal(moved.src_rows, batch.src_rows) and moved.cached == 1


def test_an_image_batch_may_only_load_cached_rows() -> None:
    batch = ImageBatch.unpack(ImageBatch.pack([], [], [7, 8], [0, 1], cached=2))

    assert batch.grids == () and batch.pixel_values.shape[0] == 0
    assert batch.cu_patches.tolist() == [0] and batch.max_patches == 0
    assert batch.store_rows.tolist() == []
