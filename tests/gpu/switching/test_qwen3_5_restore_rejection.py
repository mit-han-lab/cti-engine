"""Qwen3.5-9B gate: a serving pause whose KV cannot come back keeps training running.

Inside a training session this process takes the memory the released KV would need.
A pause for serving requested while a two-microbatch logical batch runs must then be
rejected with ``SwitchRejectedError`` while the batch finishes on the training state and
the session keeps training; once the memory is free again the session closes and greedy
decoding repeats its tokens.
"""

from __future__ import annotations

import json
import socket
import sys
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
from tests.support.paths import REPO_ROOT, source_pythonpath

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

RECEIPT_PREFIX = "CTI_QWEN3_5_RESTORE_REJECTION="
_CHILD_TIMEOUT_SECONDS = 600.0
_PROMPT = "The capital of France is"
_MAX_TOKENS = 16
# Two microbatches of two rows, so the pause request meets a microbatch boundary.
_ROW_LENGTHS = (24, 17, 31, 9)
# What the filler leaves for training; far below the released KV.
_TRAINING_ROOM_BYTES = 6 * 1024**3


def _greedy_tokens(port: int, model: str, prompt_ids: list[int]) -> list[int]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(
            {
                "model": model,
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


def _run() -> None:
    import torch
    from transformers import AutoTokenizer

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
    )
    from cti.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
    from cti.server import OnlineServer
    from cti.training import SFTDatum, SwitchRejectedError, TrainingBatch

    snapshot = str(QWEN3_5_9B.snapshot())
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    prompt_ids = tokenizer(_PROMPT)["input_ids"]
    # Random rows of the base vocabulary, which holds no image placeholder.
    generator = torch.Generator().manual_seed(20261016)
    rows = (torch.randint(4, tokenizer.vocab_size, (n,), generator=generator) for n in _ROW_LENGTHS)
    batch = TrainingBatch.sft(
        tuple(
            SFTDatum(token_ids=tuple(row.tolist()), target_mask=(False,) + (True,) * (len(row) - 1))
            for row in rows
        ),
        BatchLayout.PADDED,
        pad_token_id=(
            tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        ),
    )
    # This process's CUDA context exists before the engine sizes its memory.
    torch.cuda.init()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    options = EngineOptions(
        model=ModelOptions(path=snapshot, max_seq_len=512),
        memory=MemoryOptions(inference_ratio=0.85, peak_ratio=0.9),
        scheduling=SchedulingOptions(max_running_requests=8, max_batch_tokens=256),
        cache=CacheOptions(prefix="naive"),
        lora=LoRAConfig(targets=Qwen3_5ForConditionalGeneration.LORA_TARGETS, rank=4, alpha=8.0),
        training=TrainingConfig(
            trainable_plan=TrainablePlan.all_declared(),
            optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
            batching=TrainingBatching(
                BatchLayout.PADDED,
                StaticBatching(microbatch_size=2, max_tokens_per_sample=4096),
            ),
            activation_checkpointing=ActivationCheckpointMode.FULL_OFFLOAD,
            training_pause_boundary=TrainingPauseBoundary.MICROBATCH,
        ),
    )
    tuning = EngineTuning(attention_backend="fi", cuda_graph_bs=(1,))
    with OnlineServer(options, ServerOptions(port=port), tuning=tuning) as server:
        reference = _greedy_tokens(port, snapshot, prompt_ids)
        with server.training_session() as session:
            free = torch.cuda.mem_get_info()[0]
            filler = torch.empty(free - _TRAINING_ROOM_BYTES, dtype=torch.uint8, device="cuda")
            training = session.forward_backward(batch)
            with pytest.raises(SwitchRejectedError) as rejection:
                session.pause_for_serving()
            loss_sum = training.result().loss_sum
            # Still training: another logical batch runs without a resume.
            retry_loss_sum = session.forward_backward(batch).result().loss_sum
            del filler
            torch.cuda.empty_cache()
            session.discard_accumulation()
        tokens = _greedy_tokens(port, snapshot, prompt_ids)
    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "rejection": str(rejection.value),
                "loss_sums": [loss_sum, retry_loss_sum],
                "tokens_match": tokens == reference,
            },
            sort_keys=True,
        )
    )


def test_qwen3_5_pause_whose_kv_does_not_fit_keeps_training(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env, log_root = isolated_runtime_env(tmp_path, "qwen3_5-restore-rejection")
    env["PYTHONPATH"] = source_pythonpath()
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.switching.test_qwen3_5_restore_rejection"),
        log_root / "qwen3_5-restore-rejection.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert_cuda_device_metadata(receipt)
    assert receipt["rejection"].startswith("inference memory does not fit while training"), receipt
    loss_sum, retry_loss_sum = receipt["loss_sums"]
    assert retry_loss_sum == loss_sum > 0
    assert receipt["tokens_match"]


if __name__ == "__main__":
    _run()
