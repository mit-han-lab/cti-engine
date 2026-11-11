# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/message/backend.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cti.multimodal import ImageInputs, RopePositionLayout, rope_layout

from ._codec import Family
from ._request import AdmissionPriority, SamplingSpec

if TYPE_CHECKING:
    import torch


@dataclass
class BaseBackendMsg:
    @staticmethod
    def family() -> Family:
        # Each architecture's rotary layout travels by name, imported when a request first
        # carries it.
        return Family((UserMsg, AbortBackendMsg), open_bases={RopePositionLayout: rope_layout})


@dataclass
class UserMsg(BaseBackendMsg):
    """One request. A prompt holding images carries them with the rotary position layout
    its model's input processor returned, and a text prompt carries neither."""

    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling: SamplingSpec
    priority: AdmissionPriority = AdmissionPriority.NORMAL
    images: ImageInputs | None = None
    rope_layout: RopePositionLayout | None = None

    def __post_init__(self) -> None:
        self.priority = AdmissionPriority.parse(self.priority)


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
