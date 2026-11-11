"""The DFlash MoE combination gate (``tests.support.dflash_moe_gate``) at ETP2 against EP2 on
two local RTX PRO 6000s; opt in with CTI_TP_QUALIFY=1. Every expert cut to half its
intermediate width on each rank serves DFlash as closely to the FP32 truth as the plain EP2
engine does, and accepts as much as DFlash at EP2."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_5_35B_A3B, QWEN3_5_35B_A3B_DFLASH
from tests.support.dflash_moe_gate import run_gate
from tests.support.gpu import require_device_memory

pytestmark = pytest.mark.checkpoints(QWEN3_5_35B_A3B, QWEN3_5_35B_A3B_DFLASH)

_QUALIFY = "CTI_TP_QUALIFY"


def test_dflash_at_etp2_follows_ep2(tmp_path: Path) -> None:
    if os.environ.get(_QUALIFY) != "1":
        pytest.skip(f"requires an exclusive two-GPU allocation and {_QUALIFY}=1")
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip(f"requires 2 visible GPUs, found {torch.cuda.device_count()}")
    # The BF16 target beside its draft on the first rank's GPU: an RTX PRO 6000.
    require_device_memory(90)
    run_gate("expert-tp", tmp_path)
