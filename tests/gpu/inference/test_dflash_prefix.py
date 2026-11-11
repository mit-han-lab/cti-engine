"""The DFlash prefix gate (``tests.support.dflash_prefix_gate``) on one GPU: DFlash with the
radix cache, DFlash with the naive cache and ordinary decoding with the radix cache each follow
the FP32 truth, for Qwen3-8B and Qwen3.5-9B with their pinned drafts and KV pages of 1 and 16
tokens."""

from __future__ import annotations

import pytest

from tests.support.dflash import PINNED_CHECKPOINTS
from tests.support.dflash_prefix_gate import run_gate


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("pair", PINNED_CHECKPOINTS[:2], ids=lambda pair: pair.target.repo_id)
def test_dflash_radix_dflash_naive_and_radix_follow_the_fp32_truth(
    pair, page_size, tmp_path
) -> None:
    run_gate(pair, page_size, tmp_path)
