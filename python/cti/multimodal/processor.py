"""Request input processors: the contract the engine calls and the architecture tables."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import torch
    from transformers import PretrainedConfig

    from cti.multimodal import ImageLimits

    from .inputs import ImageInputs
    from .rope import RopePositionLayout


@dataclass(frozen=True, slots=True)
class _Architecture:
    """A vision-language architecture's input processor and the rotary layout its prompts with
    images carry, each a ``(module, class)`` imported on first use."""

    processor: tuple[str, str]
    rope_layout: tuple[str, str]


_QWEN3_5 = _Architecture(
    (".qwen3_5.processor", "Qwen3_5InputProcessor"), (".qwen3_5", "Qwen3_5RopeLayout")
)
# The one architecture table. Text-only deployments load no image preprocessing, and a rank
# decoding another rank's requests imports their layout only.
_ARCHITECTURES = MappingProxyType(
    {
        "Qwen3_5ForConditionalGeneration": _QWEN3_5,
        "Qwen3_5MoeForConditionalGeneration": _QWEN3_5,
    }
)


@dataclass(frozen=True, slots=True)
class ImagePreprocessing:
    """What a checkpoint's image preprocessing makes of an image: at least
    ``min_image_tokens`` and at most ``max_image_tokens`` tokens, from uint8 levels the
    vision tower normalizes per channel c as
    ``(level * rescale_factor - image_mean[c]) / image_std[c]``."""

    min_image_tokens: int
    max_image_tokens: int
    rescale_factor: float
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not 0 < self.min_image_tokens
            or self.max_image_tokens < 0
            or len(self.image_mean) != len(self.image_std)
        ):
            raise ValueError(f"invalid image preprocessing {self}")
        if not all(self.image_std):
            raise ValueError(f"image_std must be nonzero: {self.image_std}")


@dataclass(frozen=True, slots=True, eq=False)
class ProcessedPrompt:
    """One request's prompt with its image placeholders expanded."""

    input_ids: torch.Tensor  # CPU int32 [prompt tokens]
    images: ImageInputs | None
    rope_layout: RopePositionLayout | None


class InputProcessor(ABC):
    """Expand a vision-language model's prompts with their images into request inputs.

    ``cti.tokenizer.PromptEncoder`` owns it and checks every prompt through
    ``validate_token_ids``; invalid inputs raise ``ValueError``.
    Limits that admit no image still give a processor that checks token ids and rejects
    each prompt that brings images to load.
    """

    # The layout every processed prompt with images carries; the codec decodes it by name.
    ROPE_LAYOUT: ClassVar[type[RopePositionLayout]]

    @classmethod
    @abstractmethod
    def read_image_preprocessing(
        cls, model_path: str, *, hf_config: PretrainedConfig
    ) -> ImagePreprocessing | None:
        """The checkpoint's image preprocessing, None without one."""

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_path: str, *, hf_config: PretrainedConfig, limits: ImageLimits
    ) -> InputProcessor:
        """Load the image preprocessing of the checkpoint at ``model_path``."""

    @abstractmethod
    def chat_images(self, messages: Sequence[Mapping[str, Any]]) -> list[object]:
        """The images of a chat's content parts, in the order its template places them."""

    @abstractmethod
    def expand(self, token_ids: Sequence[int], images: Sequence[object]) -> ProcessedPrompt:
        """Load ``images``, base64 ``data:`` URLs, encoded bytes or PIL images, and expand
        the prompt's one placeholder token per image into that image's tokens."""

    @abstractmethod
    def validate_token_ids(
        self, input_ids: torch.Tensor, images: ImageInputs | None
    ) -> RopePositionLayout | None:
        """Check that a prompt's image placeholders, raw token-id prompts' included, are
        exactly its image spans; return their layout, None for a text-only prompt."""


def create_input_processor(
    architecture: str,
    model_path: str,
    *,
    hf_config: PretrainedConfig,
    limits: ImageLimits,
) -> InputProcessor | None:
    """Return the image input processor of a vision-language architecture, else None.

    Architectures without one, text-only exports of vision-language checkpoints
    included, reject image inputs per request.
    """

    processor_class = _processor_class(architecture)
    if processor_class is None:
        return None
    return processor_class.from_pretrained(model_path, hf_config=hf_config, limits=limits)


def read_image_preprocessing(
    architecture: str, model_path: str, *, hf_config: PretrainedConfig
) -> ImagePreprocessing | None:
    """An architecture's checkpoint's image preprocessing, None for none."""

    processor_class = _processor_class(architecture)
    if processor_class is None:
        return None
    return processor_class.read_image_preprocessing(model_path, hf_config=hf_config)


def rope_layout(name: str) -> type[RopePositionLayout]:
    """The rotary layout class ``name``, as the codec decodes it; KeyError for any other name."""

    for architecture in _ARCHITECTURES.values():
        if architecture.rope_layout[1] == name:
            return _load(architecture.rope_layout)
    raise KeyError(f"{name!r} is no architecture's rotary layout")


def _processor_class(architecture: str) -> type[InputProcessor] | None:
    if architecture not in _ARCHITECTURES:
        return None
    return _load(_ARCHITECTURES[architecture].processor)


def _load(target: tuple[str, str]) -> Any:
    module_path, class_name = target
    return getattr(importlib.import_module(module_path, package=__package__), class_name)
