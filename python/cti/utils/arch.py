# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/utils/arch.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from typing import NamedTuple


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def is_at_least(self, major: int, minor: int = 0) -> bool:
        return (self.major, self.minor) >= (major, minor)

    def is_family(self, major: int) -> bool:
        return self.major == major

    def __str__(self) -> str:
        return f"SM{self.major}{self.minor}"


def get_device_capability() -> DeviceCapability | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return DeviceCapability(*torch.cuda.get_device_capability())
