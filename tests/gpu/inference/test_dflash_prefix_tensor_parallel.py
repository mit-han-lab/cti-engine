"""The DFlash prefix gate (``tests.support.dflash_prefix_gate``) at TP2 on two local GPUs; opt
in with CTI_TP_QUALIFY=1. DFlash with the radix cache, DFlash with the naive cache and
ordinary decoding with the radix cache each follow the FP32 truth with every rank holding its
own heads, KV pages and, for Qwen3.5-9B, recurrent snapshot rows."""

from __future__ import annotations

import os

import pytest

from tests.support.dflash import PINNED_CHECKPOINTS
from tests.support.dflash_prefix_gate import run_gate

_QUALIFY = "CTI_TP_QUALIFY"


@pytest.mark.parametrize("pair", PINNED_CHECKPOINTS[:2], ids=lambda pair: pair.target.repo_id)
def test_tp2_dflash_radix_dflash_naive_and_radix_follow_the_fp32_truth(pair, tmp_path) -> None:
    if os.environ.get(_QUALIFY) != "1":
        pytest.skip(f"requires an exclusive two-GPU allocation and {_QUALIFY}=1")
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip(f"requires 2 visible GPUs, found {torch.cuda.device_count()}")
    run_gate(pair, 1, tmp_path, tp=2)
