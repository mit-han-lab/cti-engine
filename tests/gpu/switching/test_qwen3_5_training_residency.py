"""Qwen3.5-9B LoRA training/inference residency gate, per checkpointing mode and training
state residency.

Measured from this process with ``torch.cuda.mem_get_info``: every session entry releases
the same inference memory (KV and recurrent state), a forward/backward needs less than the
base weights (no second copy), training stays within the peak budget and warmed levels
return across windows. Every window's step pauses for serving (between its microbatches
under full offload): the served greedy tokens match those served before the window, and
decoding after it repeats. Full offload also trains variable-length steps at level memory.
A fresh server training the same steps in one session without pauses, its state resident,
matches every loss, gradient norm and the final tokens bit for bit, whichever residency
the paused one had. Switch latency must meet the device bounds.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.graph_captures import graph_captures
from tests.support.paths import REPO_ROOT, source_pythonpath
from tests.support.residency import MEMORY_TOLERANCE_BYTES
from tests.support.switching import assert_qwen35_switch_latency

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

RECEIPT_PREFIX = "CTI_QWEN3_5_TRAINING_RESIDENCY="
_CHILD_TIMEOUT_SECONDS = 1500.0
# Steady-state windows after the warm-up one; levels may drift by at most 8 MiB (the exact
# contract of tests/support/residency.py) over 7 intervals, so a leak above 1.2 MiB per
# window fails.
_TRAINING_WINDOWS = 8
_SWITCH_WARMUP_CYCLES = 1
_SWITCH_MEASUREMENT_CYCLES = 7
_INFERENCE_MEMORY_RATIO = 0.85
_PEAK_MEMORY_RATIO = 0.9
_PROMPT = "The capital of France is"
_MAX_TOKENS = 16
_NONE = "none"
_FULL_OFFLOAD = "full_offload"
_MICROBATCH_SIZE = 2
_MAX_TOKENS_PER_SAMPLE = {_NONE: 512, _FULL_OFFLOAD: 4096}
# Two microbatches of two rows; at most 32 tokens each keeps FA2's backward to one key
# block, whose gradients are deterministic.
_ROW_LENGTHS = (24, 17, 31, 9)
# The rows of each variable-length step's batch, in tokens.
_VARIABLE_LENGTH_STEPS = ((4096, 4096), (256, 256), (2048, 37), (4096, 4096))
_SEED = 20261006
_RESIDENT = "resident"
_OFFLOAD_OPTIMIZER = "offload_optimizer"
_OFFLOAD_ALL = "offload_all"


def _server(snapshot: str, mode: str, port: int, residency: str = _RESIDENT):
    from cti.config import (
        ActivationCheckpointMode,
        AdamWConfig,
        BatchLayout,
        CacheOptions,
        EngineOptions,
        EngineTuning,
        LoRAConfig,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        ServerOptions,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
        TrainingPauseBoundary,
        TrainingStateResidency,
    )
    from cti.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
    from cti.server import OnlineServer

    options = EngineOptions(
        model=ModelOptions(path=snapshot, max_seq_len=512),
        memory=MemoryOptions(
            inference_ratio=_INFERENCE_MEMORY_RATIO, peak_ratio=_PEAK_MEMORY_RATIO
        ),
        scheduling=SchedulingOptions(max_running_requests=8, max_batch_tokens=256),
        cache=CacheOptions(prefix="naive"),
        lora=LoRAConfig(targets=Qwen3_5ForConditionalGeneration.LORA_TARGETS, rank=4, alpha=8.0),
        training=TrainingConfig(
            trainable_plan=TrainablePlan.all_declared(),
            optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
            batching=TrainingBatching(
                BatchLayout.PADDED,
                StaticBatching(
                    microbatch_size=_MICROBATCH_SIZE,
                    max_tokens_per_sample=_MAX_TOKENS_PER_SAMPLE[mode],
                ),
            ),
            activation_checkpointing=ActivationCheckpointMode(mode),
            # Long-sequence training yields to serving between microbatches.
            training_pause_boundary=(
                TrainingPauseBoundary.MICROBATCH
                if mode == _FULL_OFFLOAD
                else TrainingPauseBoundary.LOGICAL_BATCH
            ),
            training_state_residency=TrainingStateResidency(residency),
        ),
    )
    tuning = EngineTuning(attention_backend="fi", cuda_graph_bs=(1,))
    return OnlineServer(options, ServerOptions(port=port), tuning=tuning)


def _run(mode: str, residency: str) -> None:
    import torch
    from transformers import AutoTokenizer

    from cti.config import BatchLayout
    from cti.training import SFTDatum, TrainingBatch

    snapshot = str(QWEN3_5_9B.snapshot())
    weight_bytes = sum(path.stat().st_size for path in Path(snapshot).glob("*.safetensors"))
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    prompt_ids = tokenizer(_PROMPT)["input_ids"]
    pad_token_id = (
        tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )

    def random_rows(lengths: tuple[int, ...]) -> TrainingBatch:
        """Seeded rows of the base vocabulary, which holds no image placeholder."""

        generator = torch.Generator().manual_seed(_SEED + sum(lengths))
        rows = (torch.randint(4, tokenizer.vocab_size, (n,), generator=generator) for n in lengths)
        return TrainingBatch.sft(
            tuple(
                SFTDatum(
                    token_ids=tuple(row.tolist()), target_mask=(False,) + (True,) * (len(row) - 1)
                )
                for row in rows
            ),
            BatchLayout.PADDED,
            pad_token_id=pad_token_id,
        )

    batch = random_rows(_ROW_LENGTHS)

    def free() -> int:
        return torch.cuda.mem_get_info()[0]

    # This process's CUDA context exists before the engine sizes its memory.
    torch.cuda.init()
    boot_free = free()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def greedy() -> list[int]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps(
                {
                    "model": snapshot,
                    "prompt": prompt_ids,
                    "max_tokens": _MAX_TOKENS,
                    "temperature": 0,
                    "ignore_eos": True,
                    "return_token_ids": True,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)["choices"][0]["token_ids"]

    with _server(snapshot, mode, port, residency) as server:
        initial_tokens = greedy()
        tokens = initial_tokens
        windows = []
        # Window 0 warms forward, backward, the optimizer and the pause; it leaves the
        # level every later window enters from.
        for _ in range(_TRAINING_WINDOWS + 1):
            free_inference = free()
            with server.training_session() as session:
                free_entry = free()
                training = session.forward_backward(batch)
                session.pause_for_serving()
                served = greedy()
                session.resume_training()
                completion = training.result()
                free_backward = free()
                step = session.optim_step().result()
                assert greedy() == tokens
                session.publish()
                free_training = free()
            served_before, tokens = tokens, greedy()
            windows.append(
                {
                    "loss_sum": completion.loss_sum,
                    "grad_norm": step.grad_norm,
                    "free_inference": free_inference,
                    "released": free_entry - free_inference,
                    "backward_growth": free_entry - free_backward,
                    "free_training": free_training,
                    "served_matches": served == served_before,
                    "tokens": tokens,
                    "repeated": greedy() == tokens,
                }
            )

        variable_length_windows = []
        if mode == _FULL_OFFLOAD:
            for rows in _VARIABLE_LENGTH_STEPS:
                with server.training_session() as session:
                    completion = session.forward_backward(random_rows(rows)).result()
                    session.discard_accumulation()
                variable_length_windows.append(
                    {"loss_sum": completion.loss_sum, "free_after": free()}
                )

        entry_ms, exit_ms = [], []
        for _ in range(_SWITCH_WARMUP_CYCLES + _SWITCH_MEASUREMENT_CYCLES):
            start = time.perf_counter_ns()
            session = server.training_session().__enter__()
            entry_ms.append((time.perf_counter_ns() - start) / 1e6)
            start = time.perf_counter_ns()
            session.close()
            exit_ms.append((time.perf_counter_ns() - start) / 1e6)
        free_after_timing = free()

    # The same steps in one session without a pause, on a fresh server of the same seed.
    with _server(snapshot, mode, port) as server:
        straight_initial_tokens = greedy()
        with server.training_session() as session:
            straight_steps = []
            for _ in range(_TRAINING_WINDOWS + 1):
                completion = session.forward_backward(batch).result()
                step = session.optim_step().result()
                session.publish()
                straight_steps.append([completion.loss_sum, step.grad_norm])
        straight_tokens = greedy()

    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "boot_free": boot_free,
                "weight_bytes": weight_bytes,
                "initial_tokens": initial_tokens,
                "windows": windows,
                "variable_length_windows": variable_length_windows,
                "latency_ms": {"entry": entry_ms, "exit": exit_ms},
                "free_after_timing": free_after_timing,
                "straight": {
                    "initial_tokens": straight_initial_tokens,
                    "steps": straight_steps,
                    "tokens": straight_tokens,
                },
            },
            sort_keys=True,
        )
    )


def _assert_level(name: str, levels: list[int]) -> None:
    assert max(levels) - min(levels) <= MEMORY_TOLERANCE_BYTES, (name, levels)


@pytest.mark.parametrize(
    ("mode", "residency"),
    [
        (_NONE, _OFFLOAD_OPTIMIZER),
        (_NONE, _RESIDENT),
        (_NONE, _OFFLOAD_ALL),
        (_FULL_OFFLOAD, _OFFLOAD_OPTIMIZER),
    ],
)
def test_qwen3_5_training_switches_release_and_restore_the_inference_state(
    tmp_path: Path,
    mode: str,
    residency: str,
) -> None:
    name = f"{mode}-{residency}"
    env, log_root = isolated_runtime_env(tmp_path, f"qwen3_5-training-residency-{name}")
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
            "tests.gpu.switching.test_qwen3_5_training_residency",
            "--activation-checkpointing",
            mode,
            "--training-state-residency",
            residency,
        ),
        log_root / f"qwen3_5-training-residency-{name}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    device = assert_cuda_device_metadata(receipt)
    print(f"Qwen3.5-9B residency ({name}): {json.dumps(receipt, sort_keys=True)}")
    warmup, *windows = receipt["windows"]
    # One capture per server: no switch recaptures a graph.
    assert graph_captures(output, "decode", [1]) == 2

    # Training may use what serving held plus the peak budget's share beyond it.
    headroom = (_PEAK_MEMORY_RATIO - _INFERENCE_MEMORY_RATIO) * receipt["boot_free"]
    for window in [warmup, *windows]:
        assert window["released"] > 0, window
        # The step trains the served weights in place: no second copy of the base.
        assert window["backward_growth"] < receipt["weight_bytes"], window
        assert window["free_training"] >= window["free_inference"] - headroom, window
        assert window["loss_sum"] > 0 and window["grad_norm"] > 0, window
        # Serving during the pause decoded as before the window; afterwards it repeats.
        assert window["served_matches"] and window["repeated"], window
        assert len(set(window["tokens"])) > 1, window
    _assert_level("released", [window["released"] for window in windows])
    _assert_level("training", [window["free_training"] for window in windows])
    # Each warm window enters from the level the one before it left.
    _assert_level(
        "inference",
        [window["free_inference"] for window in windows] + [receipt["free_after_timing"]],
    )
    variable_length_windows = receipt["variable_length_windows"]
    assert len(variable_length_windows) == (
        len(_VARIABLE_LENGTH_STEPS) if mode == _FULL_OFFLOAD else 0
    )
    assert all(window["loss_sum"] > 0 for window in variable_length_windows)
    if variable_length_windows:
        _assert_level(
            "variable length", [window["free_after"] for window in variable_length_windows]
        )

    # Pausing and switching around every step changes nothing.
    straight = receipt["straight"]
    assert straight["initial_tokens"] == receipt["initial_tokens"]
    assert straight["steps"] == [
        [window["loss_sum"], window["grad_norm"]] for window in [warmup, *windows]
    ]
    assert straight["tokens"] == windows[-1]["tokens"]

    latency = receipt["latency_ms"]
    medians = {
        path: statistics.median(samples[_SWITCH_WARMUP_CYCLES:])
        for path, samples in latency.items()
    }
    print(f"Qwen3.5-9B session switch medians (ms): {json.dumps(medians, sort_keys=True)}")
    assert_qwen35_switch_latency(device, medians, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-checkpointing", choices=(_NONE, _FULL_OFFLOAD), required=True)
    parser.add_argument(
        "--training-state-residency",
        choices=(_RESIDENT, _OFFLOAD_OPTIMIZER, _OFFLOAD_ALL),
        required=True,
    )
    args = parser.parse_args()
    _run(args.activation_checkpointing, args.training_state_residency)
