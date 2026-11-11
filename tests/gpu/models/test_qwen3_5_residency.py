"""Qwen3.5 inference state (recurrent and KV regions) leaves the device while a training session
holds it and comes back for serving, cycle after cycle, through the public ``LLM`` and
``TrainingSession``. Read from this process: the released memory fits a working set that did
not fit before; training shares the base weights; memory stays level; switches are fast;
tokens and logprobs repeat bitwise."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.qwen3_5_gpu import (
    max_running_requests,
    residency_cycles,
    selected_checkpoint,
    write_receipt,
)
from tests.support.residency import CUDA_LARGE_SEGMENT_BYTES, MEMORY_TOLERANCE_BYTES

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

GRAPH_BATCH_SIZES = (1, 2, 4)
MAX_BATCH_TOKENS = 512
MAX_SEQ_LEN = 2048
# (prompt tokens, generated tokens). The longest prompt prefills in three chunks beside
# the others' decodes; the decode batch then shrinks 3 -> 2 -> 1.
REQUESTS = ((300, 8), (77, 12), (1025, 16))
_TRAINING_ROWS = (64, 37)
_WORKING_SET_RATIO = 0.8
# Median switch bound, in ms. Measured on an RTX 5090: entry 21, exit 36 (exit maps the KV
# back); recapturing the decode graphs or reloading weights takes seconds.
_SWITCH_BOUND_MS = 250.0


def _prompts(vocab_size: int) -> list[list[int]]:
    return [
        [(7 * position + 101 * request + 3) % (vocab_size - 1) + 1 for position in range(length)]
        for request, (length, _) in enumerate(REQUESTS)
    ]


def _training_batch(vocab_size: int, pad_token_id: int):
    from cti.config import BatchLayout
    from cti.training import SFTDatum, TrainingBatch

    return TrainingBatch.sft(
        tuple(
            SFTDatum(
                token_ids=tuple((5 * i + 11 * row) % (vocab_size - 4) + 4 for i in range(length)),
                target_mask=(False,) + (True,) * (length - 1),
            )
            for row, length in enumerate(_TRAINING_ROWS)
        ),
        BatchLayout.PADDED,
        pad_token_id=pad_token_id,
    )


def _llm(snapshot: Path):
    from cti.config import (
        AdamWConfig,
        BatchLayout,
        EngineOptions,
        EngineTuning,
        LoRAConfig,
        ModelOptions,
        SchedulingOptions,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.llm import LLM

    # The default memory section: inference and peak budgets both 0.9.
    return LLM(
        EngineOptions(
            model=ModelOptions(path=str(snapshot), max_seq_len=MAX_SEQ_LEN),
            scheduling=SchedulingOptions(
                max_running_requests=max_running_requests(), max_batch_tokens=MAX_BATCH_TOKENS
            ),
            lora=LoRAConfig(targets=("attention", "mlp"), rank=4, alpha=8.0),
            training=TrainingConfig(
                trainable_plan=TrainablePlan.all_declared(),
                optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
                batching=TrainingBatching(
                    BatchLayout.PADDED,
                    StaticBatching(microbatch_size=len(_TRAINING_ROWS), max_tokens_per_sample=128),
                ),
            ),
        ),
        tuning=EngineTuning(cuda_graph_bs=GRAPH_BATCH_SIZES),
    )


def test_qwen3_5_inference_state_survives_residency_cycles() -> None:
    import torch
    from transformers import AutoTokenizer

    from cti.protocol import SamplingParams

    checkpoint = selected_checkpoint()
    snapshot = checkpoint.snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    prompts = _prompts(tokenizer.vocab_size)
    params = [
        SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=tokens, logprobs=True)
        for _, tokens in REQUESTS
    ]
    batch = _training_batch(tokenizer.vocab_size, pad_token_id)
    # Upper bound on the base weights: every checkpoint byte.
    weight_bytes = sum(path.stat().st_size for path in snapshot.glob("*.safetensors"))
    device = torch.device("cuda:0")

    def free() -> int:
        torch.cuda.synchronize(device)
        return torch.cuda.mem_get_info(device)[0]

    # This process's CUDA context and the working set's kernels load before the engine sizes
    # its memory, so later readings move only with the engine.
    bool((torch.full((1,), 3.0, device=device) == 3.0).all())
    torch.cuda.empty_cache()
    llm = _llm(snapshot)
    try:

        def serve() -> list[tuple]:
            samples = llm.sample(prompts, params).samples
            return [(s.sampled_token_ids, s.sampling_logprobs) for s in samples]

        baseline = serve()
        cycles = []
        for _ in range(residency_cycles()):
            serving = free()
            started = time.perf_counter()
            with llm.training_session() as session:
                entry_ms = (time.perf_counter() - started) * 1e3
                suspended = free()
                session.forward_backward(batch).result()
                # The step may still free its scratch after its result; discarding is a round
                # trip behind it, so the training level is read after that.
                session.discard_accumulation()
                training = free()
                # Another process takes most of the released memory and gives it back.
                working_set_bytes = (
                    int((suspended - serving) * _WORKING_SET_RATIO)
                    // CUDA_LARGE_SEGMENT_BYTES
                    * CUDA_LARGE_SEGMENT_BYTES
                )
                working_set = torch.full((working_set_bytes // 4,), 3.0, device=device)
                working_set_intact = bool((working_set == 3.0).all())
                del working_set
                torch.cuda.empty_cache()
                suspended_after = free()
                exit_started = time.perf_counter()
            exit_ms = (time.perf_counter() - exit_started) * 1e3
            resumed = free()
            cycles.append(
                {
                    "serving": serving,
                    "suspended": suspended,
                    "training": training,
                    "suspended_after": suspended_after,
                    "resumed": resumed,
                    "working_set_bytes": working_set_bytes,
                    "working_set_intact": working_set_intact,
                    "entry_ms": entry_ms,
                    "exit_ms": exit_ms,
                    "same_samples": serve() == baseline,
                    "model_revision": llm.model_revision,
                }
            )
    finally:
        llm.shutdown()
    write_receipt(
        f"qwen3_5_residency-{checkpoint.slug}", {"weight_bytes": weight_bytes, "cycles": cycles}
    )

    assert [len(tokens) for tokens, _ in baseline] == [tokens for _, tokens in REQUESTS]
    for cycle in cycles:
        # Entering released the inference state: memory that did not fit before now does.
        assert cycle["working_set_bytes"] > cycle["serving"], cycle
        assert cycle["working_set_intact"] is True
        assert abs(cycle["suspended_after"] - cycle["training"]) <= MEMORY_TOLERANCE_BYTES, cycle
        # Training runs on the shared base weights: it never grows by a copy of them.
        assert cycle["serving"] - cycle["training"] < weight_bytes, (cycle, weight_bytes)
        assert cycle["same_samples"] is True, cycle
        assert cycle["model_revision"] == 0
    # Level across cycles after the first, which warms the training pools; the tolerance
    # covers driver bookkeeping in free-memory readings.
    for level in ("serving", "suspended", "training", "resumed"):
        levels = [cycle[level] for cycle in cycles[1:]]
        assert max(levels) - min(levels) <= MEMORY_TOLERANCE_BYTES, (level, levels)
    for switch in ("entry_ms", "exit_ms"):
        median = statistics.median(cycle[switch] for cycle in cycles)
        assert median <= _SWITCH_BOUND_MS, (switch, median)
