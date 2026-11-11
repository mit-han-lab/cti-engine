"""Qwen3.5 request inputs: chat images, image preprocessing and placeholder expansion."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import struct
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Qwen2VLImageProcessorPil
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize

from cti.utils import init_logger

from ..inputs import ImageInputs, ImageSpan
from ..processor import ImagePreprocessing, InputProcessor, ProcessedPrompt
from .rope import Qwen3_5RopeLayout

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from cti.multimodal import ImageLimits

logger = init_logger(__name__)


class Qwen3_5InputProcessor(InputProcessor):
    """Turn Qwen3.5 prompt token ids and images into token ids, pixels and the rotary layout.

    Images arrive as base64 ``data:`` URLs, fetched bytes or in-process PIL images, and
    are converted to RGB as transformers' ``load_image`` does, dropping any alpha channel.
    Pixels are transformers' PIL image processor's, resized but left as uint8 levels with
    one frame per patch: the vision tower normalizes them. Its torchvision backend differs
    by up to one intensity level in under 1% of values when an image is resized.
    """

    # The image types of the OpenAI API. Image.open would otherwise try every Pillow
    # plugin, EPS included, whose decoder runs Ghostscript.
    _IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
    ROPE_LAYOUT = Qwen3_5RopeLayout

    def __init__(
        self,
        hf_config: PretrainedConfig,
        image_processor: Qwen2VLImageProcessorPil,
        *,
        limits: ImageLimits,
    ) -> None:
        """With an image token limit below its smallest image, the processor loads no image:
        it still checks every prompt's token ids and rejects each prompt that brings an
        image to load."""

        vision = hf_config.vision_config
        max_image_tokens = limits.max_image_tokens
        for name, expected in (
            ("patch_size", vision.patch_size),
            ("temporal_patch_size", vision.temporal_patch_size),
            ("merge_size", vision.spatial_merge_size),
        ):
            if getattr(image_processor, name) != expected:
                raise ValueError(f"image processor {name} does not match the model's {expected}")
        # The vision tower normalizes the uint8 pixels this processor makes per channel, as
        # read_image_preprocessing reads this image processor.
        channels = len(self._preprocessing(image_processor, vision).image_mean)
        if channels != vision.in_channels:
            raise ValueError(
                f"image processor normalizes {channels} channels, the model has "
                f"{vision.in_channels}"
            )
        # Larger images are scaled down to the token budget instead of being rejected.
        pixels_per_token = (vision.patch_size * vision.spatial_merge_size) ** 2
        min_pixels = image_processor.size.shortest_edge
        max_pixels = min(image_processor.size.longest_edge, max_image_tokens * pixels_per_token)
        self._image_size: dict[str, int] | None = None
        if max_pixels >= min_pixels:
            self._image_size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        self._no_images_reason = (
            f"the smallest image needs {-(-min_pixels // pixels_per_token)} tokens, "
            f"but an image may use at most {max_image_tokens}"
        )

        self._image_processor = image_processor
        self._image_token_id = hf_config.image_token_id
        self._video_token_id = hf_config.video_token_id
        self._merge_size = vision.spatial_merge_size
        self._patch_size = vision.patch_size
        self._patch_dim = vision.in_channels * vision.patch_size**2
        self._max_images = limits.max_images
        self._max_image_tokens = max_image_tokens

    @classmethod
    def read_image_preprocessing(
        cls, model_path: str, *, hf_config: PretrainedConfig
    ) -> ImagePreprocessing:
        vision = hf_config.vision_config
        return cls._preprocessing(_load_image_processor(model_path, vision), vision)

    @classmethod
    def from_pretrained(
        cls, model_path: str, *, hf_config: PretrainedConfig, limits: ImageLimits
    ) -> Qwen3_5InputProcessor:
        image_processor = _load_image_processor(model_path, hf_config.vision_config)
        return cls(hf_config, image_processor, limits=limits)

    def validate_token_ids(
        self, input_ids: torch.Tensor, images: ImageInputs | None
    ) -> Qwen3_5RopeLayout | None:
        if input_ids.ndim != 1:
            raise ValueError("input_ids must be one-dimensional")
        if bool((input_ids == self._video_token_id).any()):
            raise ValueError("video inputs are not supported")
        placeholders = input_ids == self._image_token_id
        if images is None:
            if bool(placeholders.any()):
                raise ValueError("prompt contains image placeholders but no images")
            return None
        if len(images.spans) > self._max_images:
            raise ValueError(f"a request may carry at most {self._max_images} images")
        if images.pixel_values.shape[1] != self._patch_dim:
            raise ValueError(f"image patches must have {self._patch_dim} values")
        expected = torch.zeros_like(placeholders)
        for span in images.spans:
            if span.num_tokens > self._max_image_tokens:
                raise ValueError(f"an image may use at most {self._max_image_tokens} tokens")
            if span.end >= len(input_ids):
                raise ValueError("every image must be followed by a text token")
            expected[span.offset : span.end] = True
        if not torch.equal(placeholders, expected):
            raise ValueError("image placeholders do not match the image spans")
        return self.ROPE_LAYOUT.from_spans(images.spans, self._merge_size)

    def chat_images(self, messages: Sequence[Mapping[str, Any]]) -> list[object]:
        if not isinstance(messages, Sequence):
            raise ValueError("a chat must be a list of messages")
        sources = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError("chat messages must be mappings")
            # Missing or None content, as in an assistant tool call, renders as empty text.
            content = message.get("content")
            if content is None or isinstance(content, str):
                continue
            if not isinstance(content, Sequence) or not all(
                isinstance(part, Mapping) for part in content
            ):
                raise ValueError("chat content must be a string or a list of content parts")
            for part in content:
                # The image test of the Qwen3.5 chat template's render_content macro.
                if "image" in part or "image_url" in part or part.get("type") == "image":
                    source = part.get("image", part.get("image_url"))
                    sources.append(source.get("url") if isinstance(source, Mapping) else source)
        return sources

    def expand(self, token_ids: Sequence[int], images: Sequence[object]) -> ProcessedPrompt:
        if images and self._image_size is None:
            raise ValueError(f"images are unavailable: {self._no_images_reason}")
        if len(images) > self._max_images:
            raise ValueError(f"a request may carry at most {self._max_images} images")
        loaded = [self._load_image(image) for image in images]
        input_ids = torch.tensor(token_ids, dtype=torch.int32)
        placeholders = (input_ids == self._image_token_id).nonzero().flatten().tolist()
        if len(placeholders) != len(loaded):
            raise ValueError(
                f"prompt has {len(placeholders)} image placeholders for {len(loaded)} images"
            )
        if not loaded:
            return ProcessedPrompt(input_ids, None, None)

        digests = [self._digest(image) for image in loaded]
        pixels, grids = zip(*(self._patch_rows(image) for image in loaded), strict=True)
        num_tokens = [
            frames * height * width // self._merge_size**2 for frames, height, width in grids
        ]
        repeats = torch.ones(len(input_ids), dtype=torch.int64)
        repeats[placeholders] = torch.tensor(num_tokens)
        spans = []
        added = 0
        for placeholder, grid, count, digest in zip(
            placeholders, grids, num_tokens, digests, strict=True
        ):
            spans.append(ImageSpan(placeholder + added, count, grid, digest))
            added += count - 1
        image_inputs = ImageInputs(torch.cat(pixels), tuple(spans))
        input_ids = input_ids.repeat_interleave(repeats)
        rope_layout = self.ROPE_LAYOUT.from_spans(image_inputs.spans, self._merge_size)
        return ProcessedPrompt(input_ids, image_inputs, rope_layout)

    def _patch_rows(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """The image's uint8 patch rows and grid: transformers' PIL image processor resize
        and patch order (image_processing_pil_qwen2_vl.py), one frame, left unnormalized."""

        assert self._image_size is not None
        patch, merge = self._patch_size, self._merge_size
        height, width = smart_resize(
            image.height,
            image.width,
            factor=patch * merge,
            min_pixels=self._image_size["shortest_edge"],
            max_pixels=self._image_size["longest_edge"],
        )
        resized = np.asarray(image.resize((width, height), self._image_processor.resample))
        grid_h, grid_w = height // patch, width // patch
        # [window row, window col, row in window, col in window, channel, pixel row, pixel col]
        windows = resized.reshape(grid_h // merge, merge, patch, grid_w // merge, merge, patch, -1)
        rows = windows.transpose(0, 3, 1, 4, 6, 2, 5).reshape(grid_h * grid_w, -1, copy=True)
        return torch.from_numpy(rows), (1, grid_h, grid_w)

    @classmethod
    def _load_image(cls, source: object) -> Image.Image:
        if isinstance(source, str):
            image = cls._open_data_url(source)
        elif isinstance(source, bytes):
            image = cls._open_bytes(source)
        elif isinstance(source, Image.Image):
            image = source
        else:
            raise ValueError("an image must be a data: URL, encoded bytes or a PIL image")
        # Pillow decodes lazily, so malformed data surfaces here, as any of several
        # exception types (OSError, SyntaxError, struct.error, ...).
        try:
            # The orientation and colour handling of transformers' load_image
            # (image_utils.py:470-512), without its file-path and http(s) sources.
            return ImageOps.exif_transpose(image).convert("RGB")
        except Exception as error:
            raise ValueError(f"cannot decode image: {error}") from error

    @classmethod
    def _open_data_url(cls, url: str) -> Image.Image:
        header, separator, payload = url.partition(",")
        if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("an image URL must be a base64 data:image/... URL")
        try:
            data = base64.b64decode(payload, validate=True)
        except binascii.Error as error:
            raise ValueError("image data URL is not valid base64") from error
        return cls._open_bytes(data)

    @classmethod
    def _open_bytes(cls, data: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(data), formats=cls._IMAGE_FORMATS)
        except Exception as error:
            raise ValueError(f"cannot decode image: {error}") from error
        if Image.MAX_IMAGE_PIXELS and image.width * image.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError("image exceeds Pillow's decompression-bomb pixel bound")
        return image

    @staticmethod
    def _preprocessing(
        image_processor: Qwen2VLImageProcessorPil, vision: PretrainedConfig
    ) -> ImagePreprocessing:
        """The smallest and largest image's tokens and what transformers' image processor
        does to each uint8 level after the resize, with an identity for a step it skips."""

        mean, std = tuple(image_processor.image_mean), tuple(image_processor.image_std)
        if not image_processor.do_normalize:
            mean, std = (0.0,) * len(mean), (1.0,) * len(std)
        pixels_per_token = (vision.patch_size * vision.spatial_merge_size) ** 2
        return ImagePreprocessing(
            min_image_tokens=-(-image_processor.size.shortest_edge // pixels_per_token),
            max_image_tokens=image_processor.size.longest_edge // pixels_per_token,
            rescale_factor=image_processor.rescale_factor if image_processor.do_rescale else 1.0,
            image_mean=mean,
            image_std=std,
        )

    @staticmethod
    def _digest(image: Image.Image) -> bytes:
        digest = hashlib.sha256(struct.pack("<II", image.width, image.height))
        digest.update(image.tobytes())
        return digest.digest()


def _load_image_processor(model_path: str, vision: PretrainedConfig) -> Qwen2VLImageProcessorPil:
    # The PIL backend directly: transformers 5.16.1 AutoImageProcessor requires torchvision.
    try:
        return Qwen2VLImageProcessorPil.from_pretrained(model_path)
    except OSError:
        # As transformers and vLLM: the processor's defaults at the model's patch geometry.
        logger.warning(
            "%s has no preprocessor_config.json; using the image processor's defaults", model_path
        )
        return Qwen2VLImageProcessorPil(
            patch_size=vision.patch_size,
            temporal_patch_size=vision.temporal_patch_size,
            merge_size=vision.spatial_merge_size,
        )
