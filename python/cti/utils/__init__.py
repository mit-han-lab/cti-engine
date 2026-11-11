# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/utils/__init__.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from .arch import DeviceCapability, get_device_capability
from .dataclass_tree import DataclassTree
from .logger import init_logger, log_stage
from .misc import align_ceil, div_ceil, find_model_file, lazy_getattr, mem_GB
from .torch_utils import nvtx_annotate

# Importing utils loads neither torch nor zmq: arch and torch_utils import torch on first call.
__all__ = [
    "DataclassTree",
    "DeviceCapability",
    "align_ceil",
    "div_ceil",
    "find_model_file",
    "get_device_capability",
    "init_logger",
    "log_stage",
    "lazy_getattr",
    "mem_GB",
    "nvtx_annotate",
]
