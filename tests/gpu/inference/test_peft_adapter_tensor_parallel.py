"""A PEFT adapter loaded at launch tensor-parallel on two GPUs, each rank taking its windows
of the factors, follows Transformers with the same adapter, teacher-forced."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.peft_adapter import assert_launch_adapter_follows_transformers

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)


def test_a_tp2_launch_adapter_follows_transformers_with_peft(tmp_path: Path) -> None:
    assert_launch_adapter_follows_transformers(tmp_path, num_ranks=2)
