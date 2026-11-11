from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import isolated_runtime_env, run_isolated_process
from tests.support.graph_captures import graph_captures
from tests.support.paths import REPO_ROOT, source_pythonpath

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

RECEIPT_PREFIX = "CTI_ONLINE_TRAINING="
_TRAINED_LORA_NAMES = (
    "model.layers.0.self_attn.qkv_proj.q.lora_A",
    "model.layers.0.self_attn.qkv_proj.q.lora_B",
)


def _model_snapshot() -> str:
    return str(QWEN3_0_6B.snapshot())


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _run_child(boundary_name: str) -> None:
    import torch
    from openai import OpenAI

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
        ServerOptions,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
        TrainingPauseBoundary,
    )
    from cti.protocol import AdmissionPriority, SamplingParams
    from cti.server import OnlineServer
    from cti.training import SFT, PaddedTokens, TrainingBatch

    boundary = TrainingPauseBoundary(boundary_name)
    model_path = _model_snapshot()
    token_pattern = (785, 14002, 315, 4017, 374, 9255, 11, 323)
    training_tokens = token_pattern * 16
    batch_size = 4
    training_batch = TrainingBatch(
        PaddedTokens(
            torch.tensor([training_tokens] * batch_size, dtype=torch.int64),
            torch.tensor([len(training_tokens)] * batch_size, dtype=torch.int64),
            151643,
        ),
        torch.ones(
            (batch_size, len(training_tokens) - 1),
            dtype=torch.bool,
        ),
        SFT(),
    )
    training_config = TrainingConfig(
        trainable_plan=TrainablePlan(_TRAINED_LORA_NAMES),
        optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
        batching=TrainingBatching(
            layout=BatchLayout.PADDED,
            policy=StaticBatching(
                microbatch_size=1,
                max_tokens_per_sample=len(training_tokens),
            ),
        ),
        training_pause_boundary=boundary,
    )
    warmup_rounds = int(os.environ.get("CTI_ONLINE_TRAINING_WARMUP", "5"))
    measured_rounds = int(os.environ.get("CTI_ONLINE_TRAINING_MEASURE", "100"))
    if warmup_rounds < 1 or measured_rounds < 100:
        raise ValueError("online-training timing requires >=1 warmup and >=100 measurements")

    port = _free_port()
    latencies_ms: list[float] = []
    completion_tokens: list[int] = []
    with OnlineServer(
        EngineOptions(
            model=ModelOptions(path=model_path, max_seq_len=512),
            memory=MemoryOptions(peak_ratio=0.5, inference_ratio=0.5),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=128),
            cache=CacheOptions(prefix="naive"),
            lora=LoRAConfig(targets=("attention", "mlp"), rank=4, alpha=8.0),
            training=training_config,
        ),
        ServerOptions(host="127.0.0.1", port=port),
        tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)),
    ) as server:
        with OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="cti-test",
            max_retries=0,
            timeout=60,
        ) as client:
            models = client.models.list().data
            assert [model.id for model in models] == [model_path]

            def stream_request() -> tuple[float, int]:
                started_at = perf_counter()
                stream = client.chat.completions.create(
                    model=model_path,
                    messages=[{"role": "user", "content": "Reply with one token."}],
                    max_tokens=1,
                    temperature=0.0,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body={"ignore_eos": True, "top_k": 1},
                )
                first_token_ms: float | None = None
                usage = None
                observed_tokens = 0
                for chunk in stream:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    for choice in chunk.choices:
                        if choice.finish_reason is None:
                            observed_tokens += 1
                            if first_token_ms is None:
                                first_token_ms = (perf_counter() - started_at) * 1_000
                if first_token_ms is None or usage is None:
                    raise RuntimeError("stream completed without first-token and usage evidence")
                assert observed_tokens == 1
                assert usage.completion_tokens == 1
                return first_token_ms, usage.completion_tokens

            _, baseline_tokens = stream_request()
            assert baseline_tokens == 1
            with ThreadPoolExecutor(max_workers=1) as requests:
                with server.training_session() as session:
                    for round_index in range(warmup_rounds + measured_rounds):
                        training = session.forward_backward(training_batch)
                        request = requests.submit(stream_request)
                        first_token_ms, num_completion_tokens = request.result(timeout=60)
                        completion = training.result()
                        assert completion.normalization_count == training_batch.num_targets
                        session.discard_accumulation()
                        if round_index >= warmup_rounds:
                            latencies_ms.append(first_token_ms)
                            completion_tokens.append(num_completion_tokens)

                    explicit_training = session.forward_backward(training_batch)
                    asyncio.run(session.pause_for_serving_async())
                    explicitly_served = tuple(stream_request()[1] for _ in range(3))
                    with pytest.raises(RuntimeError, match="paused for serving"):
                        session.forward_backward(training_batch)
                    asyncio.run(session.resume_training_async())
                    explicit_completion = explicit_training.result()
                    assert explicit_completion.normalization_count == training_batch.num_targets
                    step = session.optim_step().result()
                    session.publish()
                    assert step.step == 1
            _, resumed_tokens = stream_request()
            assert resumed_tokens == 1

            def stream_text() -> str:
                stream = client.completions.create(
                    model=model_path,
                    prompt="The capital of France is",
                    max_tokens=4,
                    temperature=0.0,
                    stream=True,
                )
                return "".join(choice.text for chunk in stream for choice in chunk.choices)

            # A GRPO round sampled at low priority while HTTP streams run, then trained on.
            group_size = 4
            params = [
                SamplingParams(
                    max_new_tokens=16,
                    temperature=1.0,
                    seed=seed,
                    logprobs=True,
                    logprobs_processed=True,
                )
                for seed in range(1, group_size + 1)
            ]
            with ThreadPoolExecutor(max_workers=group_size) as requests:
                streams = [requests.submit(stream_text) for _ in range(group_size)]
                rollout = server.sample(
                    ["Name a primary color."] * group_size,
                    params,
                    priority=AdmissionPriority.LOW,
                )
                served_beside = [stream.result(timeout=60) for stream in streams]
            # HTTP keeps being served beside the rollout; greedy text is not batch-invariant
            # (prefix and batch composition change near ties), so only completion is checked.
            assert all(served_beside), served_beside
            assert rollout.model_revision == 1
            assert len(rollout.samples) == group_size
            assert all(sample.sampled_token_ids for sample in rollout.samples)
            grpo_batch = TrainingBatch.grpo(
                rollout,
                [float(member) for member in range(group_size)],
                BatchLayout.PADDED,
                group_size=group_size,
                pad_token_id=151643,
            )
            with server.training_session() as session:
                session.forward_backward(grpo_batch).result()
                assert session.optim_step().result().step == 2
                session.publish()

        assert server.model_revision == 2

    receipt = {
        "model": {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision},
        "pause_boundary": boundary.value,
        "warmup_rounds": warmup_rounds,
        "measured_rounds": measured_rounds,
        "arrival_to_first_token_ms": {
            "mean": sum(latencies_ms) / len(latencies_ms),
            "p95": _percentile(latencies_ms, 0.95),
            "p99": _percentile(latencies_ms, 0.99),
            "minimum": min(latencies_ms),
            "maximum": max(latencies_ms),
        },
        "all_requests_completed": completion_tokens == [1] * measured_rounds,
        "explicit_serving_requests": explicitly_served,
        "model_revision": 2,
    }
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def test_online_http_and_local_training_share_one_engine(tmp_path: Path) -> None:
    receipts = []
    for boundary in ("logical_batch", "microbatch"):
        env, log_root = isolated_runtime_env(tmp_path, f"online-http-{boundary}")
        env["PYTHONPATH"] = source_pythonpath()
        env["PYTHONNOUSERSITE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        output = run_isolated_process(
            (
                sys.executable,
                "-m",
                "tests.gpu.integration.test_online_training_server",
                boundary,
            ),
            log_root / "online-training-server.log",
            env=env,
            cwd=REPO_ROOT,
            timeout=900,
        )
        matches = [
            json.loads(line.removeprefix(RECEIPT_PREFIX))
            for line in output.splitlines()
            if line.startswith(RECEIPT_PREFIX)
        ]
        assert len(matches) == 1
        receipt = matches[0]
        assert output.count("Selected attention backend fi: explicit attention backend") == 1
        assert graph_captures(output, "decode", [1]) == 1
        assert "Inference engine stopped" in output
        assert "Traceback" not in output
        assert receipt["model"] == {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision}
        assert receipt["pause_boundary"] == boundary
        assert receipt["measured_rounds"] == 100
        assert receipt["all_requests_completed"] is True
        assert receipt["explicit_serving_requests"] == [1, 1, 1]
        assert receipt["model_revision"] == 2
        print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True), flush=True)
        receipts.append(receipt)

    logical_batch, microbatch = receipts
    assert (
        microbatch["arrival_to_first_token_ms"]["p95"]
        < logical_batch["arrival_to_first_token_ms"]["p95"]
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_online_training_server.py POLICY")
    _run_child(sys.argv[1])
