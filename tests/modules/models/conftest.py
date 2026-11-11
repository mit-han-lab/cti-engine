"""Fixtures shared by the model tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tests.support import reference_ops


@pytest.fixture
def stub_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Building a model binds the FlashInfer-backed rotary and norm leaves; these tests never
    # call them, and a CPU-only environment may have no FlashInfer.
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            apply_rope_with_cos_sin_cache_inplace=object(),
            fused_add_rmsnorm=object(),
            rmsnorm=object(),
        ),
    )
    reference_ops.bind_vision_references(monkeypatch)
