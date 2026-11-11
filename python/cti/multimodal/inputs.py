"""Model-neutral image inputs carried from the frontend to model execution.

The split into per-request inputs and placeholder spans follows alice-dev
(``alice/multimodal/inputs.py:10-43``); the invariants it leaves unchecked are
enforced at construction. Position rules belong to the model's layout type.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from itertools import pairwise

import torch

_DIGEST_BYTES = 32


@dataclass(frozen=True, slots=True)
class ImageSpan:
    """Placeholder tokens of one image in its request's expanded prompt."""

    offset: int  # first placeholder token, relative to the whole prompt
    num_tokens: int
    grid_thw: tuple[int, int, int]  # patch grid before the spatial merge
    digest: bytes  # sha256 over width, height and RGB bytes of the decoded image

    def __post_init__(self) -> None:
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("image span offset must be a non-negative int")
        if type(self.num_tokens) is not int or self.num_tokens <= 0:
            raise ValueError("image span must cover at least one token")
        if (
            type(self.grid_thw) is not tuple
            or len(self.grid_thw) != 3
            or not all(type(size) is int and size > 0 for size in self.grid_thw)
        ):
            raise ValueError("image grid must be a tuple of three positive ints")
        if type(self.digest) is not bytes or len(self.digest) != _DIGEST_BYTES:
            raise ValueError(f"image digest must be {_DIGEST_BYTES} bytes")

    @property
    def end(self) -> int:
        return self.offset + self.num_tokens

    @property
    def num_patches(self) -> int:
        frames, height, width = self.grid_thw
        return frames * height * width

    @property
    def key(self) -> bytes:
        """sha256(digest, grid): the image's prefix-cache and encoder-cache key."""

        return hashlib.sha256(self.digest + struct.pack("<3I", *self.grid_thw)).digest()


@dataclass(frozen=True, slots=True, eq=False)
class ImageInputs:
    """Preprocessed images of one request, in prompt order."""

    pixel_values: torch.Tensor  # CPU uint8 [sum of patches, patch_dim]: one patch per row
    spans: tuple[ImageSpan, ...]

    def __post_init__(self) -> None:
        if (
            type(self.spans) is not tuple
            or not self.spans
            or not all(isinstance(span, ImageSpan) for span in self.spans)
        ):
            raise ValueError("image inputs need a nonempty tuple of image spans")
        if any(after.offset < before.end for before, after in pairwise(self.spans)):
            raise ValueError("image spans must be ordered and must not overlap")
        if (
            self.pixel_values.device.type != "cpu"
            or self.pixel_values.dtype != torch.uint8
            or self.pixel_values.ndim != 2
        ):
            raise ValueError("pixel values must be a two-dimensional CPU uint8 tensor")
        if self.pixel_values.shape[0] != sum(span.num_patches for span in self.spans):
            raise ValueError("pixel rows must equal the patches of all image grids")
