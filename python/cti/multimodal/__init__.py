"""Model-neutral request inputs and input processing contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cti.utils import lazy_getattr

if TYPE_CHECKING:
    from ._limits import ImageLimits as ImageLimits
    from .inputs import ImageInputs as ImageInputs
    from .inputs import ImageSpan as ImageSpan
    from .processor import ImagePreprocessing as ImagePreprocessing
    from .processor import InputProcessor as InputProcessor
    from .processor import ProcessedPrompt as ProcessedPrompt
    from .processor import create_input_processor as create_input_processor
    from .processor import read_image_preprocessing as read_image_preprocessing
    from .processor import rope_layout as rope_layout
    from .rope import RopePositionLayout as RopePositionLayout

# Importing a contract does not load image preprocessing or model implementations.
_LAZY_ATTRS = {
    "ImageLimits": "._limits",
    "ImageInputs": ".inputs",
    "ImageSpan": ".inputs",
    "ImagePreprocessing": ".processor",
    "InputProcessor": ".processor",
    "ProcessedPrompt": ".processor",
    "create_input_processor": ".processor",
    "read_image_preprocessing": ".processor",
    "rope_layout": ".processor",
    "RopePositionLayout": ".rope",
}

__getattr__ = lazy_getattr(__name__, _LAZY_ATTRS, globals())

__all__ = list(_LAZY_ATTRS)
