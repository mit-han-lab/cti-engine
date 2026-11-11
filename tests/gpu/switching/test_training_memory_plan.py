"""Qwen3.5-9B gate: training stays within the memory plan the engine logs at startup.

The engine trains rank-16 LoRA on every target with padded microbatches of two rows at the
batching policy's token bound, once per checkpointing mode, and packed microbatches of
many short sequences. Measured from this process with ``torch.cuda.mem_get_info``, each
window (a cold one that steps, a warm microbatch, legal shapes of varying length with
indexed targets, GRPO rollouts and an optimizer step after a tiny microbatch) maps no more
than the logged peak beyond the resident training state; what the first window leaves on
the device fits in the logged peak reserve, and later windows leave nothing. Not run by the
CPU suites; skipped without a CUDA device.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT, source_pythonpath
from tests.support.residency import MEMORY_TOLERANCE_BYTES

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

if TYPE_CHECKING:
    from cti.training import TrainingBatch

RECEIPT_PREFIX = "CTI_TRAINING_MEMORY_PLAN="
_CHILD_TIMEOUT_SECONDS = 900.0
_RANK = 16
_ROWS = 2
_TOKENS = {"none": 1024, "full": 4096, "full_offload": 8192}
# Packed microbatches without checkpointing of this many three-token sequences.
_PACKED = "packed"
_PACKED_SEQUENCES = 256
_PACKED_TOKENS = 3 * _PACKED_SEQUENCES
# Tokens per row of the microbatch whose optimizer step dominates its window.
_SMALL_ROW_TOKENS = 16
# The log rounds each size to 0.01 GiB; a difference of two is off by at most this.
_LOG_ROUNDING_BYTES = (1024**3) // 100
_GIB = r"([0-9.]+) GiB"


def _training_config(mode: str):
    from cti.config import (
        ActivationCheckpointMode,
        AdamWConfig,
        BatchLayout,
        DynamicBatching,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )

    if mode == _PACKED:
        batching = TrainingBatching(
            BatchLayout.PACKED,
            DynamicBatching(
                max_tokens_per_microbatch=_PACKED_TOKENS,
                max_sequences_per_microbatch=_PACKED_SEQUENCES,
            ),
        )
    else:
        batching = TrainingBatching(
            BatchLayout.PADDED,
            StaticBatching(microbatch_size=_ROWS, max_tokens_per_sample=_TOKENS[mode] // _ROWS),
        )
    return TrainingConfig(
        trainable_plan=TrainablePlan.all_declared(),
        optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
        batching=batching,
        activation_checkpointing=ActivationCheckpointMode("none" if mode == _PACKED else mode),
    )


def _run(mode: str) -> None:
    import torch

    from cti.config import (
        BatchLayout,
        CacheOptions,
        EngineOptions,
        EngineTuning,
        LoRAConfig,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
    from cti.training import GRPO, PaddedTokens, SFTDatum, TrainingBatch

    snapshot = str(QWEN3_5_9B.snapshot())
    generator = torch.Generator().manual_seed(20261016)

    def rows(count: int, length: int) -> torch.Tensor:
        return torch.randint(4, 100_000, (count, length), generator=generator)

    def sft(length: int, skipped: int = 0) -> TrainingBatch:
        """Rows of ``length`` tokens, each after the first a target but the first
        ``skipped``, which sends the loss through indexed targets."""

        return TrainingBatch.sft(
            tuple(
                SFTDatum(
                    token_ids=tuple(row.tolist()),
                    target_mask=(False,) * (1 + skipped) + (True,) * (length - 1 - skipped),
                )
                for row in rows(_ROWS, length)
            ),
            BatchLayout.PADDED,
            pad_token_id=0,
        )

    def packed() -> TrainingBatch:
        length = _PACKED_TOKENS // _PACKED_SEQUENCES
        return TrainingBatch.sft(
            tuple(
                SFTDatum(
                    token_ids=tuple(row.tolist()), target_mask=(False,) + (True,) * (length - 1)
                )
                for row in rows(_PACKED_SEQUENCES, length)
            ),
            BatchLayout.PACKED,
        )

    def grpo(length: int) -> TrainingBatch:
        """One group of rollouts of ``length`` tokens, each after the shared first one
        sampled."""

        token_ids = rows(_ROWS, length)
        token_ids[:, 0] = token_ids[0, 0]
        return TrainingBatch(
            PaddedTokens(token_ids, torch.full((_ROWS,), length, dtype=torch.int64), 0),
            torch.ones((_ROWS, length - 1), dtype=torch.bool),
            GRPO(
                sampling_logprobs=torch.full((_ROWS * (length - 1),), -1.0),
                rewards=torch.arange(_ROWS, dtype=torch.float32),
                advantages=torch.linspace(-1.0, 1.0, _ROWS),
                group_size=_ROWS,
                rollout_revision=llm.model_revision,
                temperature=1.0,
            ),
        )

    def free() -> int:
        return torch.cuda.mem_get_info()[0]

    def window(*batches: Callable[[], TrainingBatch], step: bool = False) -> dict[str, int]:
        """Train one logical batch per entry of ``batches`` in one session; the window's
        pool keeps every segment it mapped until the session closes."""

        with llm.training_session() as session:
            free_entry = free()
            for make in batches:
                session.forward_backward(make()).result()
                if step:
                    session.optim_step().result()
                else:
                    session.discard_accumulation()
            free_window = free()
        return {"growth": free_entry - free_window, "free_after": free()}

    # This process's CUDA context exists before the engine sizes its memory.
    torch.cuda.init()
    options = EngineOptions(
        model=ModelOptions(path=snapshot, max_seq_len=512),
        memory=MemoryOptions(inference_ratio=0.85, peak_ratio=0.9),
        scheduling=SchedulingOptions(max_running_requests=8, max_batch_tokens=256),
        cache=CacheOptions(prefix="naive"),
        lora=LoRAConfig(
            targets=Qwen3_5ForConditionalGeneration.LORA_TARGETS, rank=_RANK, alpha=2.0 * _RANK
        ),
        training=_training_config(mode),
    )
    llm = LLM(options, tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)))
    try:
        free_before = free()
        if mode == _PACKED:
            windows = {
                "first": window(packed, step=True),
                "warm": window(packed),
            }
        else:
            bound = _TOKENS[mode] // _ROWS
            # A cold window at the bound that steps, then warm ones: one at the bound, legal
            # shapes varying inside one pool with the indexed-target loss, the GRPO
            # objective, and a microbatch so small that the optimizer step dominates.
            windows = {
                "first": window(partial(sft, bound), step=True),
                "warm": window(partial(sft, bound)),
                "varied": window(
                    partial(sft, bound),
                    partial(sft, bound // 8),
                    partial(sft, bound // 2 + 1, skipped=1),
                    partial(sft, bound, skipped=1),
                ),
                "grpo": window(partial(grpo, bound), partial(grpo, bound // 2 + 1)),
                "small": window(partial(sft, _SMALL_ROW_TOKENS), step=True),
            }
    finally:
        llm.shutdown()
    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "free_before": free_before,
                "windows": windows,
            },
            sort_keys=True,
        )
    )


@pytest.mark.parametrize("mode", [*sorted(_TOKENS), _PACKED])
def test_qwen3_5_training_stays_within_the_logged_plan(tmp_path: Path, mode: str) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env, log_root = isolated_runtime_env(tmp_path, f"training-memory-plan-{mode}")
    env["PYTHONPATH"] = source_pythonpath()
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.switching.test_training_memory_plan",
            "--activation-checkpointing",
            mode,
        ),
        log_root / f"training-memory-plan-{mode}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert_cuda_device_metadata(receipt)
    resident, peak = (
        float(value) * 1024**3
        for value in re.search(
            rf"Training memory: {_GIB} resident beside inference, {_GIB} at its peak", output
        ).groups()
    )
    reserve = float(re.search(rf"Memory budget: inference {_GIB}, peak reserve {_GIB}", output)[2])
    reserve *= 1024**3
    windows = receipt["windows"]
    print(f"Training memory ({mode}): resident {resident}, peak {peak}; {json.dumps(windows)}")
    # What the first window leaves on the device, loaded kernel code, comes out of the
    # reserve; beside it, every window maps no more than the planned peak beyond the
    # resident state, and warm windows leave nothing.
    kept = receipt["free_before"] - windows["first"]["free_after"]
    assert kept <= reserve + _LOG_ROUNDING_BYTES
    for name, measured in windows.items():
        planned = peak - resident + (kept if name == "first" else 0)
        assert measured["growth"] <= planned + _LOG_ROUNDING_BYTES, (name, kept)
    levels = [measured["free_after"] for measured in windows.values()]
    assert max(levels) - min(levels) <= MEMORY_TOLERANCE_BYTES, levels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activation-checkpointing", choices=[*sorted(_TOKENS), _PACKED], required=True
    )
    _run(parser.parse_args().activation_checkpointing)
