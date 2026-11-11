"""Qwen3.5 against Transformers on a pinned checkpoint: greedy tokens, and teacher-forced top-1
tokens and logprobs sampled through the public API, against an FP32 truth within bounds
calibrated by BF16 Transformers (``tests/support/hf_parity.py``). Run by hand only:
``README.md``."""

from __future__ import annotations

from pathlib import Path

from tests.support.hf_parity import ParityRun, check_parity
from tests.support.qwen3_5_gpu import SETTINGS, selected_checkpoint


def test_qwen3_5_matches_transformers(tmp_path: Path) -> None:
    """Every prompt passes every gate. A Qwen3.5-MoE checkpoint is judged cold and through
    prefix hits, each path allowing one near-tie prompt."""

    checkpoint = selected_checkpoint()
    run = ParityRun.from_settings(
        SETTINGS, checkpoint, moe=checkpoint.moe, prompt_ids_of=checkpoint.prompt_ids_of
    )
    check_parity(run, tmp_path, "qwen3_5_hf_parity")
