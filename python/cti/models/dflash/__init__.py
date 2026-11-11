"""The DFlash drafts and their config, which the speculative seam's draft table names. The
classes load lazily, so the config imports without the layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cti.utils import lazy_getattr

if TYPE_CHECKING:
    from .config import DFlashConfig
    from .dflash2 import DFlash2DraftModel
    from .model import DFlashDraftModel

__all__ = ["DFlash2DraftModel", "DFlashConfig", "DFlashDraftModel"]
__getattr__ = lazy_getattr(
    __name__,
    {"DFlashConfig": ".config", "DFlashDraftModel": ".model", "DFlash2DraftModel": ".dflash2"},
    globals(),
)
