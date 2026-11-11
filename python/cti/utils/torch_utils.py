# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/utils/torch_utils.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import functools


def _torch_has_nvtx() -> bool:
    """Whether this torch build has NVTX; a CPU-only build stubs every range to raise."""

    import torch

    return hasattr(torch._C, "_nvtx")


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    """Run each call of the decorated method inside an NVTX range named ``name``.

    On a torch build without NVTX the method is returned unwrapped.
    """

    if not _torch_has_nvtx():
        return lambda fn: fn

    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
