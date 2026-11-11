from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    assert_ranks_reclaimed,
    engine_rank_pids,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.graph_captures import graph_captures
from tests.support.inference_correctness import GOLDEN_CASES
from tests.support.paths import PYTHON_ROOT, REPO_ROOT, source_pythonpath
from tests.support.training import assert_optimizer_state_exact

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

RECEIPT_PREFIX = "CTI_RUNTIME_LIFECYCLE="
SOAK_PROGRESS_PREFIX = "CTI_RUNTIME_LIFECYCLE_SOAK_PROGRESS="
# Round trips per pause boundary. Device memory may grow at most 64 MiB from the first ten
# round trips to the last ten, so a steady leak of more than 64 MiB / 40 = 1.6 MiB per round
# trip fails.
_SOAK_ROUND_TRIPS = 50
_TRAINED_LORA_NAMES = (
    "model.layers.0.self_attn.qkv_proj.q.lora_A",
    "model.layers.0.self_attn.qkv_proj.q.lora_B",
)
_PAD_TOKEN_ID = 151643


def _model_snapshot() -> str:
    return str(QWEN3_0_6B.snapshot())


def _run_child() -> None:
    import os

    import torch

    import cti
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
    from cti.protocol import SamplingParams
    from cti.training import SFT, PaddedTokens, TrainingBatch, TrainingCheckpoint

    rank_local_modules = (
        "cti.worker",
        "cti.inference._execution.runner",
        "cti.inference._execution._graphs",
        "cti.distributed.runtime",
    )

    def parent_state() -> dict[str, bool]:
        return {
            "cuda_initialized": torch.cuda.is_initialized(),
            "distributed_initialized": torch.distributed.is_initialized(),
            "rank_local_modules_loaded": any(name in sys.modules for name in rank_local_modules),
        }

    assert parent_state() == {
        "cuda_initialized": False,
        "distributed_initialized": False,
        "rank_local_modules_loaded": False,
    }

    snapshot = _model_snapshot()
    case = GOLDEN_CASES[0]
    expected_output = {
        "text": case.output_text,
        "token_ids": list(case.output_ids),
    }
    training_tokens = tuple(case.prompt_ids) + tuple(case.output_ids[:8])
    training_batch = TrainingBatch(
        PaddedTokens(
            torch.tensor([training_tokens], dtype=torch.int64),
            torch.tensor([len(training_tokens)], dtype=torch.int64),
            _PAD_TOKEN_ID,
        ),
        torch.ones((1, len(training_tokens) - 1), dtype=torch.bool),
        SFT(),
    )
    token_ids_snapshot = training_batch.tokens.token_ids.clone()
    lengths_snapshot = training_batch.tokens.lengths.clone()
    training_config = TrainingConfig(
        trainable_plan=TrainablePlan(_TRAINED_LORA_NAMES),
        optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
        batching=TrainingBatching(
            layout=BatchLayout.PADDED,
            policy=StaticBatching(microbatch_size=1, max_tokens_per_sample=512),
        ),
    )
    branch_checkpoint_path = Path(os.environ["CTI_BRANCH_CHECKPOINT_PATH"])
    expected_checkpoint_path = Path(os.environ["CTI_EXPECTED_CHECKPOINT_PATH"])
    resumed_checkpoint_path = Path(os.environ["CTI_RESUMED_CHECKPOINT_PATH"])

    def create_llm(*, resume_from: Path | None = None) -> LLM:
        return LLM(
            EngineOptions(
                model=ModelOptions(path=snapshot, max_seq_len=512),
                memory=MemoryOptions(inference_ratio=0.5),
                scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=128),
                cache=CacheOptions(prefix="naive"),
                lora=LoRAConfig(
                    targets=("attention", "mlp"),
                    rank=4,
                    alpha=8.0,
                ),
                training=training_config,
            ),
            tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)),
            resume_from_checkpoint=resume_from,
        )

    def generate(llm: LLM) -> dict[str, object]:
        [result] = llm.generate(
            [list(case.prompt_ids)],
            SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=8),
        )
        return result

    def assert_checkpoint_exact(
        actual: TrainingCheckpoint,
        expected: TrainingCheckpoint,
    ) -> None:
        assert actual.model_signature == expected.model_signature
        assert actual.num_optimizer_steps == expected.num_optimizer_steps
        assert actual.model_revision == expected.model_revision
        assert_optimizer_state_exact(actual.optimizer, expected.optimizer)

    parent_states = [parent_state()]
    control = create_llm()
    control_ranks = engine_rank_pids()
    assert len(control_ranks) == 1 and control_ranks[0] != os.getpid()
    assert control.model_revision == 0
    parent_states.append(parent_state())
    try:
        initial_result = generate(control)
        assert initial_result == expected_output

        with control.training_session() as session:
            first_completion_future = session.forward_backward(training_batch)
            first_step = session.optim_step().result()
            assert control.model_revision == 0
            assert generate(control) == initial_result
            session.publish()
            first_completion = first_completion_future.result()
            assert first_step.step == 1
            assert control.model_revision == 1
            session.save_checkpoint(branch_checkpoint_path)

        branch_result = generate(control)
        with control.training_session() as session:
            expected_completion_future = session.forward_backward(training_batch)
            expected_step = session.optim_step().result()
            assert control.model_revision == 1
            assert generate(control) == branch_result
            session.publish()
            expected_completion = expected_completion_future.result()
            assert expected_step.step == 2
            assert control.model_revision == 2
            session.save_checkpoint(expected_checkpoint_path)
            discarded_completion = session.forward_backward(training_batch).result()
            session.discard_accumulation()
        control_result = generate(control)
        assert control.model_revision == 2
    finally:
        control.shutdown()
        control.shutdown()
    assert_ranks_reclaimed(control_ranks)
    parent_states.append(parent_state())

    resume_started = perf_counter()
    resumed = create_llm(resume_from=branch_checkpoint_path)
    resume_startup_ms = (perf_counter() - resume_started) * 1_000
    resumed_ranks = engine_rank_pids()
    assert len(resumed_ranks) == 1 and resumed_ranks[0] != os.getpid()
    assert resumed.model_revision == 1
    parent_states.append(parent_state())
    try:
        resumed_branch_result = generate(resumed)
        assert resumed_branch_result == branch_result
        with resumed.training_session() as session:
            resumed_completion_future = session.forward_backward(training_batch)
            resumed_step = session.optim_step().result()
            assert resumed.model_revision == 1
            assert generate(resumed) == resumed_branch_result
            session.publish()
            resumed_completion = resumed_completion_future.result()
            assert resumed_step.step == 2
            assert resumed.model_revision == 2
            session.save_checkpoint(resumed_checkpoint_path)
        resumed_result = generate(resumed)
        assert resumed_result == control_result
    finally:
        resumed.shutdown()
        resumed.shutdown()
    assert_ranks_reclaimed(resumed_ranks)
    parent_states.append(parent_state())

    branch_checkpoint = TrainingCheckpoint.load(branch_checkpoint_path)
    expected_checkpoint = TrainingCheckpoint.load(expected_checkpoint_path)
    resumed_checkpoint = TrainingCheckpoint.load(resumed_checkpoint_path)
    assert branch_checkpoint.model_signature.startswith("sha256:")
    assert branch_checkpoint.num_optimizer_steps == branch_checkpoint.model_revision == 1
    assert expected_checkpoint.num_optimizer_steps == expected_checkpoint.model_revision == 2
    assert_checkpoint_exact(resumed_checkpoint, expected_checkpoint)
    assert expected_completion.loss_sum == resumed_completion.loss_sum
    assert expected_step.grad_norm == resumed_step.grad_norm
    assert first_completion.normalization_count == training_batch.num_targets
    assert discarded_completion.normalization_count == training_batch.num_targets
    assert torch.equal(training_batch.tokens.token_ids, token_ids_snapshot)
    assert torch.equal(training_batch.tokens.lengths, lengths_snapshot)
    assert expected_checkpoint.optimizer.names == _TRAINED_LORA_NAMES
    assert all(
        not state["cuda_initialized"]
        and not state["distributed_initialized"]
        and not state["rank_local_modules_loaded"]
        for state in parent_states
    )

    source_root = Path(cti.__file__).resolve().parent
    assert source_root.is_relative_to(PYTHON_ROOT)
    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "model": {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision},
                "source_root": str(source_root),
                "control": {
                    "engine_pid": control_ranks[0],
                    "initial_result": initial_result,
                    "branch_result": branch_result,
                    "final_result": control_result,
                    "second_loss_sum": expected_completion.loss_sum,
                    "second_grad_norm": expected_step.grad_norm,
                    "model_revision": 2,
                },
                "resumed": {
                    "engine_pid": resumed_ranks[0],
                    "branch_result": resumed_branch_result,
                    "final_result": resumed_result,
                    "second_loss_sum": resumed_completion.loss_sum,
                    "second_grad_norm": resumed_step.grad_norm,
                    "model_revision": 2,
                    "startup_ms": resume_startup_ms,
                },
                "checkpoint": {
                    "branch_path": str(branch_checkpoint_path),
                    "expected_path": str(expected_checkpoint_path),
                    "resumed_path": str(resumed_checkpoint_path),
                    "branch_size_bytes": branch_checkpoint_path.stat().st_size,
                    "model_signature": branch_checkpoint.model_signature,
                },
                "parent_states": parent_states,
            },
            sort_keys=True,
        )
    )


def _run_online_idle_boundary_child() -> None:
    import torch

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
    from cti.protocol import (
        DetokenizeMsg,
        SamplingParams,
        UserMsg,
    )
    from cti.protocol.link import ZmqPullQueue, ZmqPushQueue
    from cti.runtime import EngineClient

    snapshot = _model_snapshot()
    case = GOLDEN_CASES[0]
    training_config = TrainingConfig(
        trainable_plan=TrainablePlan(_TRAINED_LORA_NAMES),
        optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
        batching=TrainingBatching(
            layout=BatchLayout.PADDED,
            policy=StaticBatching(microbatch_size=1, max_tokens_per_sample=512),
        ),
    )
    engine = EngineClient(
        EngineOptions(
            model=ModelOptions(path=snapshot, max_seq_len=512),
            memory=MemoryOptions(inference_ratio=0.5, peak_ratio=0.5),
            scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=128),
            cache=CacheOptions(prefix="naive"),
            lora=LoRAConfig(
                targets=("attention", "mlp"),
                rank=4,
                alpha=8.0,
            ),
            training=training_config,
        ),
        EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)),
    )
    send_backend = ZmqPushQueue(engine.endpoints, "engine")
    receive_backend = ZmqPullQueue(engine.endpoints, "detokenizer")

    def submit_request(uid: int) -> None:
        send_backend.put(
            [
                UserMsg(
                    uid=uid,
                    input_ids=torch.tensor(case.prompt_ids, dtype=torch.int32),
                    sampling=SamplingParams(
                        temperature=0,
                        ignore_eos=True,
                        max_new_tokens=len(case.output_ids),
                    ).snapshot(),
                )
            ]
        )

    def receive_request(uid: int) -> list[int]:
        output_ids = []
        terminal = None
        while terminal is None:
            if receive_backend.socket.poll(timeout=60_000) == 0:
                raise TimeoutError(f"online request {uid} did not complete")
            for reply in receive_backend.get():
                assert isinstance(reply, DetokenizeMsg)
                assert reply.uid == uid
                output_ids.extend(reply.token_ids)
                if reply.finished:
                    terminal = reply
        assert terminal.end.usage is not None
        assert terminal.end.usage.prompt_tokens == len(case.prompt_ids)
        assert terminal.end.usage.completion_tokens == len(case.output_ids)
        return output_ids

    try:
        submit_request(6)
        assert receive_request(6) == list(case.output_ids)

        with engine.training_session(engine):
            submit_request(7)
            output_ids = receive_request(7)
        assert output_ids == list(case.output_ids)
        receipt = {
            "model": {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision},
            "serviced_at_idle_training_boundary": True,
            "resumed_output_ids": output_ids,
            "model_revision": engine.model_revision(),
        }
    finally:
        send_backend.stop()
        receive_backend.stop()
        engine.close()
        engine.close()

    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def _run_invalid_target_child() -> None:
    import torch
    from transformers import AutoConfig

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
    from cti.training import SFT, PaddedTokens, TrainingBatch

    case = GOLDEN_CASES[0]
    training_tokens = tuple(case.prompt_ids) + tuple(case.output_ids[:8])

    def training_batch(last_target: int) -> TrainingBatch:
        token_ids = training_tokens[:-1] + (last_target,)
        return TrainingBatch(
            PaddedTokens(
                torch.tensor([token_ids], dtype=torch.int64),
                torch.tensor([len(token_ids)], dtype=torch.int64),
                _PAD_TOKEN_ID,
            ),
            torch.ones((1, len(token_ids) - 1), dtype=torch.bool),
            SFT(),
        )

    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=_model_snapshot(), max_seq_len=512),
            memory=MemoryOptions(inference_ratio=0.5),
            scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=128),
            cache=CacheOptions(prefix="naive"),
            lora=LoRAConfig(targets=("attention", "mlp"), rank=4, alpha=8.0),
            training=TrainingConfig(
                trainable_plan=TrainablePlan(_TRAINED_LORA_NAMES),
                optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
                batching=TrainingBatching(
                    layout=BatchLayout.PADDED,
                    policy=StaticBatching(microbatch_size=1, max_tokens_per_sample=512),
                ),
            ),
        ),
        tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)),
    )
    vocab_size = AutoConfig.from_pretrained(_model_snapshot()).vocab_size
    valid_batch = training_batch(training_tokens[-1])
    try:
        with llm.training_session() as session:
            # The earlier result stays unobserved, so a post-acknowledgement
            # failure of the invalid batch would terminate the runtime.
            first = session.forward_backward(valid_batch)
            try:
                session.forward_backward(training_batch(vocab_size))
            except ValueError as error:
                rejection = str(error)
            else:
                raise AssertionError("an out-of-vocabulary target was accepted")
            second = session.forward_backward(valid_batch)
            step = session.optim_step().result()
            session.publish()
            completions = (first.result(), second.result())
        model_revision = llm.model_revision
    finally:
        llm.shutdown()

    receipt = {
        "model": {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision},
        "vocab_size": vocab_size,
        "rejection": rejection,
        "step": step.step,
        "normalization_count": step.normalization_count,
        "num_target_tokens": [completion.normalization_count for completion in completions],
        "valid_num_target_tokens": valid_batch.num_targets,
        "model_revision": model_revision,
    }
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def _run_online_pause_boundary_soak_child(boundary_name: str) -> None:
    import os
    import time

    import torch

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
        TrainingPauseBoundary,
    )
    from cti.protocol import (
        DetokenizeMsg,
        SamplingParams,
        UserMsg,
    )
    from cti.protocol.link import ZmqPullQueue, ZmqPushQueue
    from cti.runtime import EngineClient
    from cti.training import SFT, PaddedTokens, TrainingBatch

    boundary = TrainingPauseBoundary(boundary_name)
    snapshot = _model_snapshot()
    case = GOLDEN_CASES[0]
    training_config = TrainingConfig(
        trainable_plan=TrainablePlan(_TRAINED_LORA_NAMES),
        optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
        batching=TrainingBatching(
            layout=BatchLayout.PADDED,
            policy=StaticBatching(microbatch_size=1, max_tokens_per_sample=512),
        ),
        training_pause_boundary=boundary,
    )
    training_tokens = tuple(case.prompt_ids) + tuple(case.output_ids[:2])
    training_batch = TrainingBatch(
        PaddedTokens(
            torch.tensor([training_tokens, training_tokens], dtype=torch.int64),
            torch.tensor([len(training_tokens)] * 2, dtype=torch.int64),
            _PAD_TOKEN_ID,
        ),
        torch.ones((2, len(training_tokens) - 1), dtype=torch.bool),
        SFT(),
    )
    engine = EngineClient(
        EngineOptions(
            model=ModelOptions(path=snapshot, max_seq_len=512),
            memory=MemoryOptions(inference_ratio=0.5, peak_ratio=0.5),
            scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=128),
            cache=CacheOptions(prefix="naive"),
            lora=LoRAConfig(
                targets=("attention", "mlp"),
                rank=4,
                alpha=8.0,
            ),
            training=training_config,
        ),
        EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)),
    )
    send_backend = ZmqPushQueue(engine.endpoints, "engine")
    receive_backend = ZmqPullQueue(engine.endpoints, "detokenizer")

    nvml = None
    memory_handle = None
    memory_unavailable_reason: str | None = None
    try:
        import pynvml
    except ImportError as error:
        memory_unavailable_reason = f"pynvml unavailable: {error}"
    else:
        try:
            pynvml.nvmlInit()
            nvml = pynvml
            [engine_pid] = engine_rank_pids()
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                if any(process.pid == engine_pid for process in processes):
                    memory_handle = handle
                    break
            if memory_handle is None:
                memory_unavailable_reason = "engine process is not visible on an NVML device"
            else:
                # Process accounting and physical-memory telemetry are separate
                # NVML capabilities. GB10 exposes the former but reports the latter
                # as unsupported, so probe the operation that the soak will use.
                pynvml.nvmlDeviceGetMemoryInfo(memory_handle)
        except pynvml.NVMLError as error:
            memory_unavailable_reason = f"{type(error).__name__}: {error}"
            if nvml is not None:
                nvml.nvmlShutdown()
            nvml = None
            memory_handle = None

    round_trips = int(os.environ.get("CTI_LIFECYCLE_SOAK_ROUND_TRIPS", str(_SOAK_ROUND_TRIPS)))
    if round_trips < 1:
        raise ValueError("CTI_LIFECYCLE_SOAK_ROUND_TRIPS must be positive")
    progress_interval = int(os.environ.get("CTI_LIFECYCLE_SOAK_PROGRESS_INTERVAL", "10"))
    if progress_interval < 1:
        raise ValueError("CTI_LIFECYCLE_SOAK_PROGRESS_INTERVAL must be positive")

    memory_samples: list[int] = []
    enter_training_seconds = 0.0
    forward_backward_seconds = 0.0
    serve_after_training_seconds = 0.0

    def record_memory() -> None:
        if nvml is not None and memory_handle is not None:
            memory_samples.append(int(nvml.nvmlDeviceGetMemoryInfo(memory_handle).used))

    def submit_request(uid: int) -> None:
        send_backend.put(
            [
                UserMsg(
                    uid=uid,
                    input_ids=torch.tensor(case.prompt_ids, dtype=torch.int32),
                    sampling=SamplingParams(
                        temperature=0,
                        ignore_eos=True,
                        max_new_tokens=1,
                    ).snapshot(),
                )
            ]
        )

    def receive_request(uid: int) -> int:
        if receive_backend.socket.poll(timeout=60_000) == 0:
            raise TimeoutError(f"online request {uid} did not complete")
        (reply,) = receive_backend.get()
        assert isinstance(reply, DetokenizeMsg)
        assert reply.uid == uid
        assert reply.finished
        assert reply.end.usage.prompt_tokens == len(case.prompt_ids)
        assert reply.end.usage.completion_tokens == 1
        (token,) = reply.token_ids
        return token

    try:
        if os.environ.get("CTI_REQUIRE_DEVICE_GLOBAL_MEMORY") == "1" and memory_handle is None:
            raise RuntimeError(
                "device-global memory telemetry is required but unavailable: "
                f"{memory_unavailable_reason}"
            )
        soak_started_at = perf_counter()
        for uid in range(round_trips):
            enter_started_at = perf_counter()
            with engine.training_session(engine) as session:
                enter_training_seconds += perf_counter() - enter_started_at
                submit_request(uid)
                # The online loop remains active during physical training. Give it
                # time to place this request behind the closed admission fence before
                # the worker reaches the first cooperative completion boundary.
                time.sleep(0.01)
                forward_backward_started_at = perf_counter()
                completion = session.forward_backward(training_batch).result()
                forward_backward_seconds += perf_counter() - forward_backward_started_at
                assert completion.normalization_count == training_batch.num_targets
                session.discard_accumulation()
            serve_after_training_started_at = perf_counter()
            assert receive_request(uid) == case.output_ids[0]
            serve_after_training_seconds += perf_counter() - serve_after_training_started_at
            record_memory()
            completed = uid + 1
            if completed % progress_interval == 0 or completed == round_trips:
                print(
                    SOAK_PROGRESS_PREFIX
                    + json.dumps(
                        {
                            "completed": completed,
                            "elapsed_seconds": perf_counter() - soak_started_at,
                            "pause_boundary": boundary.value,
                            "round_trips": round_trips,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        warm_memory = memory_samples[:10]
        tail_memory = memory_samples[-10:]
        growth_bytes = (
            None
            if not warm_memory or not tail_memory
            else max(0, max(tail_memory) - max(warm_memory))
        )
        if growth_bytes is not None:
            assert growth_bytes <= 64 * 1024 * 1024
        receipt = {
            "model": {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision},
            "pause_boundary": boundary.value,
            "round_trips": round_trips,
            "timing_seconds": {
                "enter_training": enter_training_seconds,
                "forward_backward": forward_backward_seconds,
                "serve_after_training": serve_after_training_seconds,
                "total": perf_counter() - soak_started_at,
            },
            "device_global_growth_bytes": growth_bytes,
            "device_global_memory_unavailable_reason": memory_unavailable_reason,
            "model_revision": engine.model_revision(),
        }
    finally:
        if nvml is not None:
            nvml.nvmlShutdown()
        send_backend.stop()
        receive_backend.stop()
        engine.close()
        engine.close()

    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def test_qwen_flashinfer_graph_runtime_lifecycle(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "runtime-lifecycle")
    env["PYTHONPATH"] = source_pythonpath()
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CTI_BRANCH_CHECKPOINT_PATH"] = str(tmp_path / "branch.pt")
    env["CTI_EXPECTED_CHECKPOINT_PATH"] = str(tmp_path / "expected.pt")
    env["CTI_RESUMED_CHECKPOINT_PATH"] = str(tmp_path / "resumed.pt")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.integration.test_runtime_lifecycle"),
        log_root / "runtime-lifecycle.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
    )

    receipts = [
        json.loads(line.removeprefix(RECEIPT_PREFIX))
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert output.count("Selected attention backend fi: explicit attention backend") == 2
    assert graph_captures(output, "decode", [1]) == 2
    assert receipt["model"] == {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision}
    assert Path(receipt["source_root"]).resolve().is_relative_to(PYTHON_ROOT)
    expected_output = {
        "text": GOLDEN_CASES[0].output_text,
        "token_ids": list(GOLDEN_CASES[0].output_ids),
    }
    control = receipt["control"]
    resumed = receipt["resumed"]
    assert control["engine_pid"] > 0
    assert resumed["engine_pid"] > 0
    assert control["engine_pid"] != resumed["engine_pid"]
    assert control["initial_result"] == expected_output
    assert control["branch_result"] == resumed["branch_result"]
    assert control["final_result"] == resumed["final_result"]
    assert control["second_loss_sum"] == resumed["second_loss_sum"]
    assert control["second_grad_norm"] == resumed["second_grad_norm"]
    assert control["model_revision"] == resumed["model_revision"] == 2
    assert resumed["startup_ms"] > 0
    checkpoint = receipt["checkpoint"]
    assert checkpoint["branch_path"] == env["CTI_BRANCH_CHECKPOINT_PATH"]
    assert checkpoint["expected_path"] == env["CTI_EXPECTED_CHECKPOINT_PATH"]
    assert checkpoint["resumed_path"] == env["CTI_RESUMED_CHECKPOINT_PATH"]
    assert checkpoint["branch_size_bytes"] > 0
    assert checkpoint["model_signature"].startswith("sha256:")
    assert (
        receipt["parent_states"]
        == [
            {
                "cuda_initialized": False,
                "distributed_initialized": False,
                "rank_local_modules_loaded": False,
            }
        ]
        * 5
    )


def test_qwen_online_request_is_serviced_at_idle_training_boundary(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "online-training-idle-boundary")
    env["PYTHONPATH"] = source_pythonpath()
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.integration.test_runtime_lifecycle",
            "--online-idle-boundary",
        ),
        log_root / "online-training-idle-boundary.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=180,
    )

    receipts = [
        json.loads(line.removeprefix(RECEIPT_PREFIX))
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert output.count("Selected attention backend fi: explicit attention backend") == 1
    assert graph_captures(output, "decode", [1]) == 1
    assert receipt == {
        "serviced_at_idle_training_boundary": True,
        "model": {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision},
        "model_revision": 0,
        "resumed_output_ids": list(GOLDEN_CASES[0].output_ids),
    }


def test_qwen_invalid_target_is_rejected_at_submit(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "invalid-target-at-submit")
    env["PYTHONPATH"] = source_pythonpath()
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.integration.test_runtime_lifecycle",
            "--invalid-target",
        ),
        log_root / "invalid-target-at-submit.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=180,
    )

    receipts = [
        json.loads(line.removeprefix(RECEIPT_PREFIX))
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    assert len(receipts) == 1
    receipt = receipts[0]
    vocab_size = receipt["vocab_size"]
    valid_count = receipt["valid_num_target_tokens"]
    assert receipt["model"] == {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision}
    assert f"Training targets must be in [0, {vocab_size})" in receipt["rejection"]
    assert receipt["step"] == receipt["model_revision"] == 1
    assert receipt["normalization_count"] == 2 * valid_count
    assert receipt["num_target_tokens"] == [valid_count, valid_count]


def test_qwen_online_training_policy_soak(tmp_path: Path) -> None:
    for boundary in ("logical_batch", "microbatch"):
        env, log_root = isolated_runtime_env(tmp_path, f"online-{boundary}-soak")
        env["PYTHONPATH"] = source_pythonpath()
        env["PYTHONNOUSERSITE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        output = run_isolated_process(
            (
                sys.executable,
                "-m",
                "tests.gpu.integration.test_runtime_lifecycle",
                "--online-pause-boundary-soak",
                boundary,
            ),
            log_root / "online-pause-boundary-soak.log",
            env=env,
            cwd=REPO_ROOT,
            timeout=900,
        )

        receipts = [
            json.loads(line.removeprefix(RECEIPT_PREFIX))
            for line in output.splitlines()
            if line.startswith(RECEIPT_PREFIX)
        ]
        assert len(receipts) == 1
        receipt = receipts[0]
        assert output.count("Selected attention backend fi: explicit attention backend") == 1
        assert graph_captures(output, "decode", [1]) == 1
        assert receipt["model"] == {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision}
        assert receipt["pause_boundary"] == boundary
        assert receipt["round_trips"] == _SOAK_ROUND_TRIPS
        assert receipt["model_revision"] == 0
        growth_bytes = receipt["device_global_growth_bytes"]
        unavailable_reason = receipt["device_global_memory_unavailable_reason"]
        assert (growth_bytes is None) == (unavailable_reason is not None)
        assert growth_bytes is None or 0 <= growth_bytes <= 64 * 1024 * 1024


if __name__ == "__main__":
    if "--online-idle-boundary" in sys.argv[1:]:
        _run_online_idle_boundary_child()
    elif "--online-pause-boundary-soak" in sys.argv[1:]:
        _run_online_pause_boundary_soak_child(sys.argv[2])
    elif "--invalid-target" in sys.argv[1:]:
        _run_invalid_target_child()
    else:
        _run_child()
