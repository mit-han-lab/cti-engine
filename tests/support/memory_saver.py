"""The memory-saver provider for spawn-path tests: ``torch_memory_saver`` where installed,
else a stand-in that keeps the stock ``configure_subprocess`` behavior spawn paths face."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_PROVIDER = "torch_memory_saver"
_STAND_IN_HOOK = "/stand-in/torch_memory_saver_hook_mode_preload.so"


def use_memory_saver_provider(monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the provider importable; return the hook path its preload context adds.

    The installed provider is used where there is one. Elsewhere the stand-in takes
    its place until ``monkeypatch`` is undone.
    """

    if importlib.util.find_spec(_PROVIDER) is not None:
        from torch_memory_saver.utils import get_binary_path_from_package

        return str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))
    provider = ModuleType(_PROVIDER)
    provider.configure_subprocess = _configure_subprocess  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _PROVIDER, provider)
    return _STAND_IN_HOOK


@contextmanager
def _configure_subprocess() -> Iterator[None]:
    # As the locked commit's hooks/mode_preload.py and utils.change_env: the hook goes first,
    # twice when nested; exit asserts LD_PRELOAD is unchanged and restores it ("" if unset).
    existing = os.environ.get("LD_PRELOAD", "")
    with _change_env("LD_PRELOAD", f"{_STAND_IN_HOOK}:{existing}" if existing else _STAND_IN_HOOK):
        yield


@contextmanager
def _change_env(key: str, value: str) -> Iterator[None]:
    previous = os.environ.get(key, "")
    os.environ[key] = value
    try:
        yield
    finally:
        assert os.environ[key] == value
        os.environ[key] = previous


__all__ = ["use_memory_saver_provider"]
