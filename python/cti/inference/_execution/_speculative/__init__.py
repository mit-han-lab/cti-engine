"""The speculative seam, which nothing outside names a method through; its classes load lazily."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cti.utils import lazy_getattr

if TYPE_CHECKING:
    from ._records import RoundResult
    from .dflash import (
        RoundBackends,
        RoundModels,
        SpeculativeRuntime,
        TargetForward,
        prepare_rounds,
    )
    from .method import DraftModel, load_speculative

__all__ = [
    "DraftModel",
    "load_speculative",
    "RoundBackends",
    "RoundModels",
    "RoundResult",
    "SpeculativeRuntime",
    "TargetForward",
    "prepare_rounds",
]
__getattr__ = lazy_getattr(
    __name__,
    {
        **dict.fromkeys(__all__, ".dflash"),
        "DraftModel": ".method",
        "load_speculative": ".method",
        "RoundResult": "._records",
    },
    globals(),
)
