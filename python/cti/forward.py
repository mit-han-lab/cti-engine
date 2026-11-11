"""The forward a runner hands its model: the rows preparation reads and the inputs the model
reads.

A layer passes its row-dependent work (attention, a recurrent core) through
``batch.graph_breaks.eager``. Eager forwards and whole-graph captures run it in place;
breakable graphs end a captured segment there and run it between replays.
"""

from __future__ import annotations

import struct
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import TYPE_CHECKING, Protocol, TypeAlias

import torch

if TYPE_CHECKING:
    from cti.attention import BaseAttnMetadata
    from cti.recurrent import RecurrentMetadata


# ``call(x, batch, out)`` computes the rows of ``x`` for ``batch``: into ``out`` and returns
# it when ``out`` is given, else into storage it allocates.
BreakCall: TypeAlias = Callable[
    [torch.Tensor, "ForwardInputs", "torch.Tensor | None"], torch.Tensor
]


class GraphBreaks(ABC):
    """How a forward runs its breaks."""

    @abstractmethod
    def eager(
        self, call: BreakCall, x: torch.Tensor, batch: ForwardInputs, *, width: int
    ) -> torch.Tensor:
        """``call``'s ``[rows, width]`` result for ``x``'s rows, run outside any capture."""


class InlineBreaks(GraphBreaks):
    """Run each break in place: eager forwards and whole-graph captures."""

    def eager(
        self, call: BreakCall, x: torch.Tensor, batch: ForwardInputs, *, width: int
    ) -> torch.Tensor:
        return call(x, batch, None)


_HEADER = "<6q"  # images, patch width, store_rows, src_rows and dst lengths, cached loads


@dataclass(frozen=True, slots=True, eq=False)
class ImageBatch:
    """One inference step's images: those it encodes into the runner's encoder-output rows,
    then the row each image token takes, in the order of the step's tokens.

    The scheduler packs the step's images into the step's bytes; the runner unpacks them on
    the host and copies the batch to the device. Each encoded image is one sequence of the
    encoder's variable-length attention. ``grids`` and ``max_patches`` are Python ints, so
    an encoder shapes its work without reading the device. The first ``cached`` loads read
    rows held before the step, so they run before the encoder writes. The batch is a plain
    record: ``ImageInputs`` owns the pixels' rows, ``unpack`` derives the grids' bounds from
    one header, and the encoder cache checks the rows under checks.
    """

    pixel_values: torch.Tensor  # uint8 [patches, patch_dim], one image after another
    grids: tuple[tuple[int, int, int], ...]  # each encoded image's patch grid before the merge
    cu_patches: torch.Tensor  # int32 [images + 1]: image i holds rows [cu[i], cu[i + 1])
    max_patches: int  # patches of the largest encoded image, 0 without one
    store_rows: torch.Tensor  # int64 [encoded tokens]: the row each encoder output row fills
    src_rows: torch.Tensor  # int64 [image tokens]: the row each image token takes
    dst: torch.Tensor  # int64 [image tokens]: the step's token rows
    cached: int

    @staticmethod
    def pack(
        images: Sequence[tuple[torch.Tensor, tuple[int, int, int]]],
        store_rows: Sequence[int],
        src_rows: Sequence[int],
        dst: Sequence[int],
        cached: int = 0,
    ) -> bytes:
        """The batch of ``(pixel rows, grid)`` per image to encode, possibly none, and its row
        indices, as the bytes a step carries.

        Each image's rows are copied once, straight into the bytes: no intermediate tensor
        and no CPU torch op, so packing forks no OpenMP team.
        """

        grids = tuple(grid for _, grid in images)
        patch_dim = images[0][0].shape[1] if images else 0
        indices = (store_rows, src_rows, dst)
        header = struct.pack(
            f"{_HEADER}{3 * len(grids)}q{sum(map(len, indices))}q",
            len(grids),
            patch_dim,
            *map(len, indices),
            cached,
            *(size for grid in grids for size in grid),
            *(row for index in indices for row in index),
        )
        return b"".join((header, *(_buffer(pixels) for pixels, _ in images)))

    @classmethod
    def unpack(cls, data: bytes) -> ImageBatch:
        """A batch ``pack`` made, on the host. Its tensors view ``data`` without copying it, so
        they are read-only by contract: a batch only goes to the device and the encoder."""

        count, patch_dim, *lengths, cached = struct.unpack_from(_HEADER, data)
        offset = struct.calcsize(_HEADER)
        sizes = struct.unpack_from(f"<{3 * count}q", data, offset)
        grids = tuple(tuple(sizes[index : index + 3]) for index in range(0, len(sizes), 3))
        patches = [frames * height * width for frames, height, width in grids]
        offset += 8 * len(sizes)  # every field before the pixels is 8 bytes
        indices = []
        for length in lengths:
            indices.append(_view(data, torch.int64, length, offset))
            offset += 8 * length
        pixel_values = _view(data, torch.uint8, sum(patches) * patch_dim, offset)
        return cls(
            pixel_values.view(-1, patch_dim) if patch_dim else pixel_values.view(0, 0),
            grids,
            torch.tensor([0, *accumulate(patches)], dtype=torch.int32),
            max(patches, default=0),
            *indices,
            cached,
        )

    def to(self, device: torch.device, *, non_blocking: bool = False) -> ImageBatch:
        tensors = (self.pixel_values, self.cu_patches, self.store_rows, self.src_rows, self.dst)
        pixels, cu_patches, *rows = (t.to(device, non_blocking=non_blocking) for t in tensors)
        return ImageBatch(pixels, self.grids, cu_patches, self.max_patches, *rows, self.cached)


def _view(data: bytes, dtype: torch.dtype, count: int, offset: int) -> torch.Tensor:
    """``count`` items of ``data`` from ``offset``, viewed without a copy."""

    if not count:
        return torch.empty(0, dtype=dtype)  # frombuffer refuses an empty view
    with warnings.catch_warnings():
        # torch warns for any read-only buffer; the bytes outlive the batch in their op.
        warnings.simplefilter("ignore", UserWarning)
        return torch.frombuffer(data, dtype=dtype, count=count, offset=offset)


def _buffer(tensor: torch.Tensor) -> memoryview:
    """A CPU tensor's bytes, without a copy when it is contiguous."""

    return memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast("B")


class VisionTower(Protocol):
    """An image encoder split where vision graphs cut it: ``embed`` runs eagerly, ``encode``
    (the blocks and the merger) is what a graph captures. Each ``merge_size ** 2``
    consecutive patches of an image become one token."""

    merge_size: int

    def embed(self, images: ImageBatch) -> tuple[torch.Tensor, ...]:
        """What ``encode`` reads of ``images``: tensors of one row per patch."""
        ...

    def encode(
        self, inputs: tuple[torch.Tensor, ...], cu_patches: torch.Tensor, max_patches: int
    ) -> torch.Tensor:
        """Token embeddings of the embedded patches ``inputs``, each sequence of
        ``cu_patches`` attending within itself, none longer than ``max_patches``."""
        ...

    def largest_image_batch(
        self, *, max_tokens: int, max_image_tokens: int, device: torch.device
    ) -> ImageBatch: ...


class ImageEncoding(ABC):
    """How a forward runs its vision tower."""

    @abstractmethod
    def encode(self, tower: VisionTower, images: ImageBatch) -> torch.Tensor:
        """``tower``'s token embeddings of ``images``, image after image."""


class EagerImages(ImageEncoding):
    """Run the tower in place, on every image at once."""

    def encode(self, tower: VisionTower, images: ImageBatch) -> torch.Tensor:
        return tower.encode(tower.embed(images), images.cu_patches, images.max_patches)


@dataclass(frozen=True, slots=True)
class RecurrentExport:
    """Export the recurrent state after a sequence's first ``position`` tokens to snapshot
    ``row``."""

    position: int
    row: int


@dataclass(frozen=True, slots=True)
class ForwardRows:
    """A forward's rows as the model side reads them, computed once by the execution half.

    The first ``size`` rows are the batch's, the first ``num_extend`` of them extensions
    and the rest decodes; the padding rows a captured graph size adds follow. Per padded
    row: its request slot (a padding row's is the dummy slot), its cached prefix, its
    sequence length after the step and its rotary minus sequence position. ``restores``
    copy snapshot rows to the extensions' slots as ``(row, slot)``; ``captures`` are each
    real row's exports; ``logit_rows`` the query rows that take logits, each row's last."""

    size: int
    num_extend: int
    slots: tuple[int, ...]
    kv_prefix_lens: tuple[int, ...]
    sequence_lens: tuple[int, ...]
    rope_offsets: tuple[int, ...]
    restores: tuple[tuple[int, int], ...]
    captures: tuple[tuple[RecurrentExport, ...], ...]
    logit_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.num_extend <= self.size or not self.size:
            raise ValueError("a forward needs rows, its extensions among them")
        padded = (self.slots, self.kv_prefix_lens, self.sequence_lens, self.rope_offsets)
        if any(len(values) != len(self.slots) for values in padded) or len(self.slots) < self.size:
            raise ValueError("every padded row needs a slot, lengths and a rope offset")
        if len(self.captures) != self.size or len(self.logit_rows) != self.size:
            raise ValueError("every row needs its captures and a logit row")

    @property
    def uses_ragged_attention(self) -> bool:
        return self.num_extend > 0

    @property
    def is_uniform_decode(self) -> bool:
        return self.num_extend == 0

    @property
    def padded_size(self) -> int:
        return len(self.slots)

    def extend_len(self, index: int) -> int:
        """The query rows padded row ``index`` runs."""

        return self.sequence_lens[index] - self.kv_prefix_lens[index]


@dataclass(frozen=True, slots=True)
class ForwardInputs:
    """Everything a model forward reads. ``rope_positions`` are read only by rotary
    embeddings: int64 ``[tokens]`` that every axis reads, or M-RoPE ``[axes, tokens]`` rows
    when a prompt holds images. ``size`` rows take logits, the last of each sequence when
    ``ragged``, else every row; ``logit_rows`` holds them (int64 ``[size]``, ``ForwardRows``'),
    ``None`` for a forward that takes no logits. ``recurrent_metadata`` stays ``None``
    unless the model has recurrent layers; ``images`` are the step's image loads and encodes
    on the device, or ``None``, and ``image_encoding`` runs the tower on them: in place
    unless the runner captured vision graphs.
    ``graph_breaks`` runs the layers' breaks: in place unless a breakable graph is capturing
    the forward."""

    input_ids: torch.Tensor
    rope_positions: torch.Tensor
    attn_metadata: BaseAttnMetadata
    size: int
    ragged: bool
    recurrent_metadata: RecurrentMetadata | None = None
    images: ImageBatch | None = None
    logit_rows: torch.Tensor | None = None
    graph_breaks: GraphBreaks = field(default_factory=InlineBreaks)
    image_encoding: ImageEncoding = field(default_factory=EagerImages)


@dataclass(frozen=True, slots=True)
class RoundShape:
    """Speculative rounds of up to ``max_rows`` rows of ``width`` tokens, ``depth`` of them
    unsettled per row."""

    width: int
    depth: int
    max_rows: int
