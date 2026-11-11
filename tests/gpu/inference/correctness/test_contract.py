"""End-to-end inference correctness contracts."""

from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.inference_correctness import BOUNDARY_PROFILE, LONG_PROFILE

from .contract import run_correctness_profile

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)


def test_qwen_fixed_snapshot_correctness_contract(tmp_path: Path) -> None:
    run_correctness_profile(BOUNDARY_PROFILE, tmp_path)


def test_qwen_fixed_snapshot_long_correctness_contract(tmp_path: Path) -> None:
    run_correctness_profile(LONG_PROFILE, tmp_path)
