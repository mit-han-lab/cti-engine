"""The DFlash MoE combination gate (``tests.support.dflash_moe_gate``) on one RTX PRO 6000:
DFlash over the radix cache against the naive cache, and DFlash with a per-expert LoRA adapter
against DFlash without it. A smaller GPU skips at once, before any checkpoint is read. ETP2
against EP2 is test_dflash_moe_expert_tp, a tensor-parallel module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_5_35B_A3B, QWEN3_5_35B_A3B_DFLASH
from tests.support.dflash_moe_gate import run_gate
from tests.support.gpu import require_device_memory

pytestmark = pytest.mark.checkpoints(QWEN3_5_35B_A3B, QWEN3_5_35B_A3B_DFLASH)


@pytest.mark.parametrize("case", ["radix-prefix", "expert-lora"])
def test_dflash_moe_combination_follows_its_references(case: str, tmp_path: Path) -> None:
    # The BF16 target beside its draft: an RTX PRO 6000.
    require_device_memory(90)
    run_gate(case, tmp_path)
