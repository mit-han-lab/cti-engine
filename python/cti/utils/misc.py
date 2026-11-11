# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/utils/misc.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from importlib import import_module


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def lazy_getattr(
    module_name: str, lazy_attrs: Mapping[str, str], namespace: dict[str, object]
) -> Callable[[str], object]:
    """Return a PEP 562 module ``__getattr__`` that imports ``lazy_attrs`` names on first use."""

    def __getattr__(name: str) -> object:
        module = lazy_attrs.get(name)
        if module is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        value = getattr(import_module(module, module_name), name)
        namespace[name] = value
        return value

    return __getattr__


def find_model_file(model_path: str, filename: str) -> str | None:
    """``filename`` of a local model directory; None if absent."""

    path = os.path.join(model_path, filename)
    return path if os.path.isfile(path) else None
