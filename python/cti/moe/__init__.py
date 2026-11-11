# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/moe/__init__.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from .base import BaseMoeBackend, ExpertLoRA, SparseBlock, TrainingBlock


def create_moe_backend() -> BaseMoeBackend:
    """The engine's MoE backend: cti-kernels' fused experts and whole-block op."""

    from .fused import FusedMoe

    return FusedMoe()


__all__ = [
    "BaseMoeBackend",
    "ExpertLoRA",
    "SparseBlock",
    "TrainingBlock",
    "create_moe_backend",
]
