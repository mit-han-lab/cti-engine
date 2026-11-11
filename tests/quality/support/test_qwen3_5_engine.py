from __future__ import annotations

import subprocess
import sys

from tests.support.paths import REPO_ROOT


def test_binding_the_references_keeps_torch_device_contexts() -> None:
    """Binding the references in a fresh process keeps ``torch.device`` contexts: a torch
    constructor replaced before the first context ignores every later one."""

    # Transformers finds the meta context of from_pretrained where torch.tensor([]) lands.
    code = """
import pytest
import torch

from tests.support.qwen3_5_engine import bind_reference_kernels

bind_reference_kernels(pytest.MonkeyPatch())
with torch.device("meta"):
    device = torch.tensor([]).device
assert device.type == "meta", device
"""
    subprocess.run((sys.executable, "-c", code), check=True, cwd=REPO_ROOT)
