"""Dense Qwen3 against Transformers on each pinned size: greedy tokens, and teacher-forced top-1
tokens and logprobs sampled through the public API, against an FP32 truth within bounds
calibrated by BF16 Transformers (``tests/support/hf_parity.py``). Run by hand only:
``README.md``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tests.support.checkpoints import (
    QWEN3_0_6B,
    QWEN3_1_7B,
    QWEN3_4B,
    QWEN3_8B,
    Checkpoint,
)
from tests.support.hf_parity import ParityRun, ParitySettings, check_parity
from tests.support.qwen3_moe import PUBLISHED

# The CTI_QWEN3_* settings (README.md); each case sets TP_SIZE and TRUTH_GPUS itself.
SETTINGS = ParitySettings("CTI_QWEN3_")
# Each renders the parity prompts as Qwen3-30B-A3B does, so it reads that checkpoint's ids.
PROMPT_IDS_OF = PUBLISHED["30b-a3b"].pin


@dataclasses.dataclass(frozen=True)
class Case:
    checkpoint: Checkpoint
    tensor_parallel_size: int = 1
    truth_gpus: int = 1


CASES = {
    "0.6b-tp1": Case(QWEN3_0_6B),
    "1.7b-tp1": Case(QWEN3_1_7B),
    "4b-tp1": Case(QWEN3_4B),
    "8b-tp1": Case(QWEN3_8B),
    "8b-tp2": Case(QWEN3_8B, tensor_parallel_size=2),
}


@pytest.mark.parametrize("case", list(CASES))
def test_qwen3_matches_transformers(case: str, tmp_path: Path) -> None:
    """Every prompt passes every gate on the engine over the case's GPUs."""

    import torch

    selected = CASES[case]
    gpus = max(selected.tensor_parallel_size, selected.truth_gpus)
    if torch.cuda.device_count() < gpus:
        pytest.skip(f"requires {gpus} GPUs, found {torch.cuda.device_count()}")
    run = ParityRun.from_settings(
        SETTINGS, selected.checkpoint, moe=False, prompt_ids_of=PROMPT_IDS_OF
    )
    run = dataclasses.replace(
        run, tensor_parallel_size=selected.tensor_parallel_size, truth_gpus=selected.truth_gpus
    )
    check_parity(run, tmp_path, f"qwen3_hf_parity-tp{selected.tensor_parallel_size}")
