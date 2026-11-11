"""A seeded request and an unseeded min-p request share every decode step.
The min-p row's redraw holds no seeded row; it used to hand the request-local sampler an
empty batch, which failed and ended the engine. Each repetition must finish both."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

RECEIPT_PREFIX = "CTI_MIXED_SEED_MIN_P="
REPETITIONS = 8
MAX_TOKENS = 16


def _run_child() -> None:
    import torch

    from cti.config import EngineOptions, MemoryOptions, ModelOptions, SchedulingOptions
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=str(QWEN3_0_6B.snapshot()), max_seq_len=256),
            memory=MemoryOptions(inference_ratio=0.5),
            scheduling=SchedulingOptions(max_running_requests=4),
        )
    )
    try:
        seeded = SamplingParams(temperature=0.8, seed=7, max_new_tokens=MAX_TOKENS, ignore_eos=True)
        min_p = SamplingParams(
            temperature=0.8, min_p=0.05, max_new_tokens=MAX_TOKENS, ignore_eos=True
        )
        lengths = [
            [
                len(sample.sampled_token_ids)
                for sample in llm.sample(
                    ["Count to ten:", "Name a color:"], [seeded, min_p]
                ).samples
            ]
            for _ in range(REPETITIONS)
        ]
        receipt = {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "lengths": lengths,
        }
    finally:
        llm.shutdown()
    print(RECEIPT_PREFIX + json.dumps(receipt))


def test_seeded_and_unseeded_min_p_requests_share_decode_steps(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "mixed-seed-min-p")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_mixed_seed_min_p"),
        log_root / "mixed-seed-min-p.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=600,
    )
    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert_cuda_device_metadata(receipt)

    assert receipt["lengths"] == [[MAX_TOKENS, MAX_TOKENS]] * REPETITIONS


if __name__ == "__main__":
    _run_child()
