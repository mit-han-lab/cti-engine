"""Qwen3-0.6B gate: training sessions hand the inference memory over and back.

Measured from this process with ``torch.cuda.mem_get_info``: entering a session releases
the whole KV cache the engine logs, and this process can hold most of it while a step
trains; the first forward/backward needs less than the base weights (no second copy);
training needs no more than serving held; every level returns across windows. Greedy decoding
replays the golden tokens with its graph after every switch, committed steps change the
sampled logprobs and a discarded step does not. Stepping in a fresh session each time
matches one long session bit for bit, and the session switch meets its latency bounds.
"""

from __future__ import annotations

import itertools
import json
import re
import statistics
import sys
import time
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.graph_captures import graph_captures
from tests.support.inference_correctness import GOLDEN_CASES
from tests.support.paths import REPO_ROOT, source_pythonpath
from tests.support.residency import (
    CUDA_LARGE_SEGMENT_BYTES,
    MEMORY_TOLERANCE_BYTES,
    physical_storage_nbytes,
)
from tests.support.switching import assert_qwen3_switch_latency

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

RECEIPT_PREFIX = "CTI_MEMORY_RESIDENCY="
_INFERENCE_MEMORY_RATIO = 0.9
_PEAK_MEMORY_RATIO = 0.9
# Window 0 is cold; window 1 holds the working set below through its step.
_WINDOWS = 4
# Share of the released KV this process holds through window 1's step; the rest leaves
# room for the step itself.
_WORKING_SET_RATIO = 0.8
_SWITCH_WARMUP_CYCLES = 1
# Seven samples, so up to three host-noise spikes cannot move the median.
_SWITCH_MEASUREMENT_CYCLES = 7
_TRAINED_LORA_NAMES = (
    "model.layers.0.self_attn.qkv_proj.q.lora_A",
    "model.layers.0.self_attn.qkv_proj.q.lora_B",
)
# Qwen3-0.6B: 28 layers, each one K/V storage of 8 heads x 128 x BF16 per token, plus
# the graph sink page.
_KV_LAYERS = 28
_KV_BYTES_PER_LAYER_AND_PAGE = 2 * 8 * 128 * 2


def _snapshot() -> str:
    return str(QWEN3_0_6B.snapshot())


def _llm(snapshot: str):
    from cti.config import (
        AdamWConfig,
        BatchLayout,
        CacheOptions,
        EngineOptions,
        EngineTuning,
        LoRAConfig,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.llm import LLM

    options = EngineOptions(
        model=ModelOptions(path=snapshot, max_seq_len=512),
        memory=MemoryOptions(
            inference_ratio=_INFERENCE_MEMORY_RATIO, peak_ratio=_PEAK_MEMORY_RATIO
        ),
        scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=128),
        cache=CacheOptions(prefix="naive"),
        lora=LoRAConfig(targets=("attention", "mlp"), rank=4, alpha=8.0),
        training=TrainingConfig(
            trainable_plan=TrainablePlan(_TRAINED_LORA_NAMES),
            optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
            batching=TrainingBatching(
                BatchLayout.PADDED,
                StaticBatching(microbatch_size=1, max_tokens_per_sample=512),
            ),
        ),
    )
    tuning = EngineTuning(
        attention_backend="fi",
        cuda_graph_bs=(1,),
        graph_row_buckets=tuple(len(case.prompt_ids) for case in GOLDEN_CASES),
    )
    return LLM(options, tuning=tuning)


def _run() -> None:
    import torch

    from cti.protocol import SamplingParams
    from cti.training import SFT, PaddedTokens, TrainingBatch

    snapshot = _snapshot()
    weight_bytes = sum(path.stat().st_size for path in Path(snapshot).glob("*.safetensors"))
    tokens = tuple(GOLDEN_CASES[0].prompt_ids) + tuple(GOLDEN_CASES[0].output_ids[:8])
    batch = TrainingBatch(
        PaddedTokens(
            torch.tensor([tokens], dtype=torch.int64),
            torch.tensor([len(tokens)], dtype=torch.int64),
            151643,
        ),
        torch.ones((1, len(tokens) - 1), dtype=torch.bool),
        SFT(),
    )

    def infer(llm) -> tuple[list[list[int]], list[float]]:
        """Greedy tokens of every golden case and the first case's sampled logprobs."""

        result = llm.sample(
            [list(case.prompt_ids) for case in GOLDEN_CASES],
            [
                SamplingParams(
                    temperature=0,
                    ignore_eos=True,
                    max_new_tokens=len(case.output_ids),
                    logprobs=True,
                )
                for case in GOLDEN_CASES
            ],
        )
        return (
            [list(sample.sampled_token_ids) for sample in result.samples],
            list(result.samples[0].sampling_logprobs),
        )

    def free() -> int:
        return torch.cuda.mem_get_info()[0]

    # This process's CUDA context and the fill kernel of its working set are loaded
    # before the engine sizes its memory, so neither moves a level below.
    torch.ones(1, device="cuda")
    llm = _llm(snapshot)
    try:
        initial_tokens, initial_logprobs = infer(llm)
        logprobs = [initial_logprobs]
        windows = []
        working_set_bytes = 0
        for window in range(_WINDOWS):
            free_inference = free()
            with llm.training_session() as session:
                free_entry = free()
                working_set = None
                if window == 1:
                    nbytes = (
                        int((free_entry - free_inference) * _WORKING_SET_RATIO)
                        // CUDA_LARGE_SEGMENT_BYTES
                        * CUDA_LARGE_SEGMENT_BYTES
                    )
                    working_set = torch.ones(nbytes, dtype=torch.uint8, device="cuda")
                free_before_step = free()
                completion = session.forward_backward(batch).result()
                free_backward = free()
                step = session.optim_step().result()
                assert llm.model_revision == window
                if working_set is not None:
                    working_set_bytes = working_set.numel()
                    del working_set
                    torch.cuda.empty_cache()
                unpublished_tokens, unpublished_logprobs = infer(llm)
                assert unpublished_tokens == (windows[-1]["tokens"] if windows else initial_tokens)
                assert unpublished_logprobs == logprobs[-1]
                session.publish()
                free_training = free()
            window_tokens, window_logprobs = infer(llm)
            logprobs.append(window_logprobs)
            windows.append(
                {
                    "loss_sum": completion.loss_sum,
                    "grad_norm": step.grad_norm,
                    "step": step.step,
                    "revision": llm.model_revision,
                    "free_inference": free_inference,
                    "released": free_entry - free_inference,
                    "backward_growth": free_before_step - free_backward,
                    "free_training": free_training,
                    "tokens": window_tokens,
                }
            )
        free_after_windows = free()

        with llm.training_session() as session:
            session.forward_backward(batch).result()
            session.discard_accumulation()
        discarded_revision = llm.model_revision
        discarded_tokens, discarded_logprobs = infer(llm)

        entry_ms, exit_ms = [], []
        for _ in range(_SWITCH_WARMUP_CYCLES + _SWITCH_MEASUREMENT_CYCLES):
            start = time.perf_counter_ns()
            session = llm.training_session().__enter__()
            entry_ms.append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns()
            session.close()
            exit_ms.append((time.perf_counter_ns() - start) / 1e6)
        free_after_timing = free()
    finally:
        llm.shutdown()

    # The same steps in one session, on a fresh engine of the same seed.
    llm = _llm(snapshot)
    try:
        straight_initial_tokens, straight_initial_logprobs = infer(llm)
        with llm.training_session() as session:
            straight_steps = []
            for _ in range(_WINDOWS):
                completion = session.forward_backward(batch).result()
                step = session.optim_step().result()
                session.publish()
                straight_steps.append([completion.loss_sum, step.grad_norm])
        straight_tokens, straight_logprobs = infer(llm)
    finally:
        llm.shutdown()

    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "weight_bytes": weight_bytes,
                "initial_tokens": initial_tokens,
                "windows": windows,
                "working_set_bytes": working_set_bytes,
                "logprobs": logprobs,
                "free_after_windows": free_after_windows,
                "discarded": {
                    "revision": discarded_revision,
                    "tokens": discarded_tokens,
                    "logprobs": discarded_logprobs,
                },
                "latency_ms": {"session_entry": entry_ms, "session_exit": exit_ms},
                "free_after_timing": free_after_timing,
                "straight": {
                    "initial_tokens": straight_initial_tokens,
                    "initial_logprobs": straight_initial_logprobs,
                    "steps": straight_steps,
                    "tokens": straight_tokens,
                    "logprobs": straight_logprobs,
                },
            },
            sort_keys=True,
        )
    )


def test_qwen_training_sessions_hand_inference_memory_over_and_back(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "memory-residency")
    env["PYTHONPATH"] = source_pythonpath()
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.switching.test_memory_residency"),
        log_root / "memory-residency.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=600,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    device = assert_cuda_device_metadata(receipt)
    print(f"Memory residency: {json.dumps(receipt, sort_keys=True)}")
    golden = [list(case.output_ids) for case in GOLDEN_CASES]
    windows = receipt["windows"]
    straight = receipt["straight"]
    # One engine each, so one capture each: no switch recaptures a graph.
    assert output.count("Selected attention backend fi: explicit attention backend") == 2
    assert graph_captures(output, "decode", [1]) == 2

    # Entry releases the whole KV cache.
    num_pages = int(re.search(r"Inference capacity: (\d+) KV pages", output).group(1))
    kv_bytes = _KV_LAYERS * physical_storage_nbytes((num_pages + 1) * _KV_BYTES_PER_LAYER_AND_PAGE)
    for window in windows:
        assert abs(window["released"] - kv_bytes) <= MEMORY_TOLERANCE_BYTES, (kv_bytes, window)
        # The step trains the served weights in place: no second copy of the base.
        assert window["backward_growth"] < receipt["weight_bytes"], window
        # Training fits in what serving held: here the peak budget is the inference one.
        assert window["free_training"] >= window["free_inference"], window
    # This process held most of the released memory, more than inference had left free,
    # while window 1 trained a step.
    assert receipt["working_set_bytes"] > windows[1]["free_inference"]
    # No growth across switches: warm levels agree within the exact contract's 8 MiB, one
    # allocator rounding of a few large segments.
    for name, levels in (
        ("released", [window["released"] for window in windows[1:]]),
        ("training", [window["free_training"] for window in windows[1:]]),
        (
            "inference",
            [window["free_inference"] for window in windows[1:]]
            + [receipt["free_after_windows"], receipt["free_after_timing"]],
        ),
    ):
        assert max(levels) - min(levels) <= MEMORY_TOLERANCE_BYTES, (name, levels)

    # Greedy decoding replays the golden tokens after every switch; each committed step
    # moves the sampled logprobs and the discarded one leaves them bit for bit.
    assert receipt["initial_tokens"] == golden
    assert all(window["tokens"] == golden for window in windows)
    assert [window["step"] for window in windows] == list(range(1, _WINDOWS + 1))
    assert [window["revision"] for window in windows] == list(range(1, _WINDOWS + 1))
    logprobs = receipt["logprobs"]
    assert all(before != after for before, after in itertools.pairwise(logprobs))
    discarded = receipt["discarded"]
    assert discarded["revision"] == _WINDOWS
    assert discarded["tokens"] == golden
    assert discarded["logprobs"] == logprobs[-1]

    # Switching between steps changes nothing: a fresh engine stepping in one session
    # starts and ends where the switching one did.
    assert straight["initial_tokens"] == golden
    assert straight["initial_logprobs"] == logprobs[0]
    assert straight["steps"] == [[window["loss_sum"], window["grad_norm"]] for window in windows]
    assert all(loss > 0 and grad_norm > 0 for loss, grad_norm in straight["steps"])
    assert straight["tokens"] == golden
    assert straight["logprobs"] == logprobs[-1]

    latency = receipt["latency_ms"]
    medians = {
        path: statistics.median(samples[_SWITCH_WARMUP_CYCLES:])
        for path, samples in latency.items()
    }
    assert all(
        len(samples) == _SWITCH_WARMUP_CYCLES + _SWITCH_MEASUREMENT_CYCLES
        for samples in latency.values()
    )
    assert_qwen3_switch_latency(device, medians)


if __name__ == "__main__":
    _run()
