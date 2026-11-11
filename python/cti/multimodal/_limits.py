"""A request's image bounds, which every process that admits images reads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageLimits:
    """How many images one request may carry, and how many tokens each image may use."""

    max_images: int
    max_image_tokens: int

    def __post_init__(self) -> None:
        if not (self.max_images > 0 and self.max_image_tokens >= 0):
            raise ValueError(f"invalid image limits {self}")


__all__ = ["ImageLimits"]
