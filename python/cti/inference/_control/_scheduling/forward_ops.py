"""A planned forward's op, frozen from its requests: the bytes the executor reads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from cti.forward import ImageBatch
from cti.inference._step import Forward, QueryKind, Row

if TYPE_CHECKING:
    from cti.inference._step import PageMap
    from cti.multimodal import ImageSpan

    from .encoder_cache import EncoderCache
    from .request import Request, RequestBatch


class ForwardBuilder:
    """Freezes a planned forward over ``RequestBatch`` rows: their token rows, the
    extensions' token ids, rotary positions, prompt targets and images, and the prompts
    their restored prefixes hold, for the requests that process logits."""

    def __init__(self, encoder_cache: EncoderCache) -> None:
        self._encoder_cache = encoder_cache

    @staticmethod
    def rows(batch: RequestBatch) -> tuple[Row, ...]:
        query_kinds = (QueryKind.EXTEND,) * len(batch.extend_requests) + (QueryKind.DECODE,) * len(
            batch.decode_requests
        )
        return tuple(
            Row(
                request.slot.index,
                query_kind,
                kv_prefix_len=request.cached_len,
                sequence_len=request.device_len,
                write_position=request.device_len if request.can_decode else None,
                sampling_position=request.num_submitted_output_tokens,
            )
            for request, query_kind in zip(batch.requests, query_kinds, strict=True)
        )

    def forward(
        self,
        batch: RequestBatch,
        rows: tuple[Row, ...],
        pages: PageMap | None,
        continuations: Sequence[Request],
    ) -> Forward:
        """``batch``'s forward over its reserved ``rows``; ``continuations`` are the admitted
        prompts whose images the encoder cache keeps pinned."""

        return Forward(
            rows,
            _extend_token_ids(batch),
            _extend_rope_positions(batch),
            self._images(batch, continuations),
            pages,
            _prompt_targets(batch),
            tuple(
                (
                    request.slot.index,
                    request.vocabulary_ids(0, request.cached_len).numpy().tobytes(),
                )
                for request in batch.extend_requests
                if request.sampling.processes_logits
                and 0 < request.cached_len == request.admitted_len
            ),
        )

    def _images(self, batch: RequestBatch, continuations: Sequence[Request]) -> bytes | None:
        """Pack the images the extensions cover part of: the cached ones load first, then
        the encoder stores each other image whole once and the covered tokens load. Each
        extended request then pins the image it ends inside."""

        planned: list[_PlannedImage] = []
        offset = 0
        for request in batch.extend_requests:
            for span, pixel_values in request.planned_images():
                skip = max(request.cached_len - span.offset, 0)
                count = min(span.end, request.device_len) - span.offset - skip
                row = offset + span.offset + skip - request.cached_len
                planned.append((span, pixel_values, row, skip, count))
            offset += request.extend_len
        if not planned:
            return None
        encoder = self._encoder_cache
        held, stored = encoder.plan([(span.key, span.num_tokens) for span, *_ in planned])
        pixels = {span.key: (pixel_values, span.grid_thw) for span, pixel_values, *_ in planned}
        held_src, held_dst = _image_loads(planned, held)
        src, dst = _image_loads(planned, stored)
        store_rows = [row for rows in stored.values() for row in rows]
        encoder.check(store_rows, held_src + src, cached=len(held_src))
        for request in batch.extend_requests:
            encoder.pin(request.uid, request.split_image)
        encoder.check_pins(continuations)
        return ImageBatch.pack(
            [pixels[key] for key in stored],
            store_rows,
            held_src + src,
            held_dst + dst,
            cached=len(held_src),
        )


def _prompt_targets(batch: RequestBatch) -> bytes | None:
    """Each extension token's next prompt token, for the requests that score their prompts
    (-1 elsewhere and past a prompt's end): frozen here, since a later chunk's tokens are not
    in the token table yet."""

    if not any(request.scores_prompt for request in batch.extend_requests):
        return None
    spans = []
    for request in batch.extend_requests:
        span = torch.full((request.extend_len,), -1, dtype=torch.int32)
        if request.scores_prompt:
            targets = request.vocabulary_ids(request.cached_len + 1, request.device_len + 1)
            span[: len(targets)] = targets
        spans.append(span)
    return torch.cat(spans).numpy().tobytes()


def _extend_token_ids(batch: RequestBatch) -> bytes | None:
    """The extensions' input ids; image key ids become 0, rows the encoder output fills."""

    if not batch.extend_requests:
        return None
    spans = [
        request.vocabulary_ids(request.cached_len, request.device_len)
        for request in batch.extend_requests
    ]
    return torch.cat(spans).to(torch.int32).numpy().tobytes()


def _extend_rope_positions(batch: RequestBatch) -> bytes | None:
    """The extend tokens' rotary positions, when a prompt among them holds images.

    A request's layout gives its rows; a text prompt's tokens sit at their sequence
    positions on every axis.
    """

    layout_rows = [
        None
        if request.rope_layout is None
        else request.rope_layout.positions(request.cached_len, request.device_len)
        for request in batch.extend_requests
    ]
    axes = next((rows.shape[0] for rows in layout_rows if rows is not None), None)
    if axes is None:
        return None
    num_tokens = sum(request.extend_len for request in batch.extend_requests)
    positions = torch.empty((axes, num_tokens), dtype=torch.int64)
    offset = 0
    for request, rows in zip(batch.extend_requests, layout_rows, strict=True):
        next_offset = offset + request.extend_len
        positions[:, offset:next_offset] = (
            torch.arange(request.cached_len, request.device_len) if rows is None else rows
        )
        offset = next_offset
    return positions.numpy().tobytes()


# An extension's image: its span and pixels, its first token row in the step, and the
# leading tokens the extension skips and the tokens it covers.
_PlannedImage = tuple["ImageSpan", torch.Tensor, int, int, int]


def _image_loads(
    planned: list[_PlannedImage], source: dict[bytes, list[int]]
) -> tuple[list[int], list[int]]:
    """The encoder rows and step token rows of the planned images ``source`` holds."""

    src: list[int] = []
    dst: list[int] = []
    for span, _, row, skip, count in planned:
        if (rows := source.get(span.key)) is not None:
            src += rows[skip : skip + count]
            dst += range(row, row + count)
    return src, dst
