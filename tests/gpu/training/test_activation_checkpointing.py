from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    assert_cuda_device_metadata,
    fixed_kv_page_capacity,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT, source_pythonpath
from tests.support.training import (
    PACKED_BF16_TOLERANCE,
    PAD_TOKEN_ID,
    PADDED_BF16_TOLERANCE,
    RAW_SAMPLES,
    TrainingDriver,
    clone_main_gradients,
    expected_qwen_adapter_names,
    measure_tensor_error,
    packed_sft_batch,
    padded_sft_batch,
    qwen3_model_snapshot,
    synchronize_optimizer_parameters,
)

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

if TYPE_CHECKING:
    import torch

    from cti.layers import Adapter
    from cti.training import TrainingBatch
    from cti.training._checkpoint import TrainingCheckpoint
    from cti.training.runner import TrainingRunner

RECEIPT_PREFIX = "CTI_ACTIVATION_CHECKPOINTING="
ADAPTER_SEED = 20260722
BENCHMARK_SEQUENCE_LENGTH = 2048
NUM_WARMUP_STEPS = 5
NUM_MEASURE_STEPS = 5
UPDATE_STATE_MAX_ABSOLUTE = 2.1e-4
UPDATE_STATE_GLOBAL_RELATIVE_L2 = 2e-3


def _masters(runner: TrainingRunner) -> torch.Tensor:
    """The settled runner's FP32 masters, flattened in parameter order."""

    import torch

    state = runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer
    return torch.cat([master.flatten() for master in state.master])


def _training_vectors(
    runner: TrainingRunner, adapters: Adapter
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    gradients = clone_main_gradients(runner, adapters.named_parameters())
    live_parameters = tuple(
        parameter.detach().cpu().flatten()
        for parameter in tuple(t for _, t in adapters.named_parameters())
    )
    return (
        torch.cat([gradient.cpu().flatten() for gradient in gradients]),
        torch.cat(live_parameters),
    )


def _run_update(
    worker: Any,
    batches: tuple[TrainingBatch, ...],
) -> dict[str, object]:
    import torch

    training = worker.training
    assert training is not None
    runner = training
    driver = TrainingDriver(training)
    # The masters change only at the step.
    master_before_update = _masters(runner)
    for batch in batches:
        driver.forward_backward(batch)
    adapters = worker.worker.host.adapter
    assert adapters is not None
    gradients, live_before_update = _training_vectors(runner, adapters)
    step, observations = driver.optim_step()
    live_after_step = torch.cat(
        [p.detach().cpu().flatten() for _, p in adapters.named_parameters()]
    )
    torch.testing.assert_close(live_after_step, live_before_update, rtol=0, atol=0)
    driver.publish()
    grad_norm, num_target_tokens = step.grad_norm, step.normalization_count
    completions = tuple(observation.completion for observation in observations)
    updated_live = torch.cat(
        [
            parameter.detach().cpu().flatten()
            for parameter in tuple(t for _, t in adapters.named_parameters())
        ]
    )
    updated_master = _masters(runner)
    return {
        "grad_norm": grad_norm,
        "gradients": gradients,
        "live_before_update": live_before_update,
        "loss_sum": sum(completion.loss_sum for completion in completions),
        "master_before_update": master_before_update,
        "num_microbatches": len(batches),
        "num_target_tokens": num_target_tokens,
        "updated_live": updated_live,
        "updated_master": updated_master,
    }


def _benchmark_batch(*, packed: bool) -> TrainingBatch:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(ADAPTER_SEED)
    sample = tuple(
        torch.randint(
            0,
            150_000,
            (BENCHMARK_SEQUENCE_LENGTH + 1,),
            generator=generator,
            dtype=torch.int64,
        ).tolist()
    )
    return packed_sft_batch((sample,)) if packed else padded_sft_batch((sample,), PAD_TOKEN_ID)


def _measure_steps(
    worker: Any,
    batch: TrainingBatch,
    initial_state: TrainingCheckpoint,
) -> dict[str, float | int]:
    import torch

    training = worker.training
    assert training is not None
    training.restore(initial_state)
    driver = TrainingDriver(training)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    for _ in range(NUM_WARMUP_STEPS):
        driver.forward_backward(batch)
        driver.optim_step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    elapsed_ms = []
    for _ in range(NUM_MEASURE_STEPS):
        started = time.perf_counter()
        driver.forward_backward(batch)
        driver.optim_step()
        elapsed_ms.append((time.perf_counter() - started) * 1_000)
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    return {
        "allocated_before_bytes": allocated_before,
        "median_step_ms": statistics.median(elapsed_ms),
        "num_measure_steps": NUM_MEASURE_STEPS,
        "num_warmup_steps": NUM_WARMUP_STEPS,
        "peak_allocated_bytes": peak_allocated,
        "peak_step_bytes": peak_allocated - allocated_before,
        "sequence_length": BENCHMARK_SEQUENCE_LENGTH,
    }


def _run_child(mode: str, output_path: Path) -> None:
    import torch

    import cti
    from cti.config import (
        ActivationCheckpointMode,
        AdamWConfig,
        BatchLayout,
        CacheOptions,
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
    from tests.support.gpu import start_rank

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(ADAPTER_SEED)
    checkpoint_mode = ActivationCheckpointMode(mode)

    def create_worker(*, packed: bool) -> Any:
        with fixed_kv_page_capacity(8):
            return start_rank(
                EngineOptions(
                    model=ModelOptions(
                        path=qwen3_model_snapshot(), max_seq_len=BENCHMARK_SEQUENCE_LENGTH + 1
                    ),
                    scheduling=SchedulingOptions(
                        max_running_requests=1, max_batch_tokens=BENCHMARK_SEQUENCE_LENGTH + 1
                    ),
                    cache=CacheOptions(prefix="naive"),
                    lora=LoRAConfig(
                        targets=("attention", "mlp"),
                        rank=4,
                        alpha=8.0,
                    ),
                    training=TrainingConfig(
                        trainable_plan=TrainablePlan(expected_qwen_adapter_names()),
                        optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
                        batching=TrainingBatching(
                            BatchLayout.PACKED if packed else BatchLayout.PADDED,
                            StaticBatching(
                                microbatch_size=len(RAW_SAMPLES),
                                max_tokens_per_sample=BENCHMARK_SEQUENCE_LENGTH + 1,
                            ),
                        ),
                        activation_checkpointing=checkpoint_mode,
                    ),
                ),
                EngineTuning(attention_backend="fi", cuda_graph_bs=()),
                checks=True,
            )

    results = {}
    performance = {}
    for layout, combined, partitions in (
        (
            "padded",
            padded_sft_batch(RAW_SAMPLES, PAD_TOKEN_ID),
            tuple(padded_sft_batch((sample,), PAD_TOKEN_ID) for sample in RAW_SAMPLES),
        ),
        (
            "packed",
            packed_sft_batch(RAW_SAMPLES),
            tuple(packed_sft_batch((sample,)) for sample in RAW_SAMPLES),
        ),
    ):
        worker = create_worker(packed=layout == "packed")
        try:
            runner = worker.training
            adapters = worker.worker.host.adapter
            assert adapters is not None
            torch.manual_seed(ADAPTER_SEED)
            with torch.no_grad():
                for tensor in adapters.state_dict().values():
                    tensor.normal_(mean=0.0, std=0.02)
            synchronize_optimizer_parameters(runner, adapters.named_parameters())
            initial_state = runner.checkpoint(num_optimizer_steps=0, model_revision=0)
            frozen = runner.trainable_parameters.complement()
            frozen_versions = tuple(parameter._version for parameter in frozen)

            results[layout] = {}
            for name, batches in (("combined", (combined,)), ("partitioned", partitions)):
                runner.restore(initial_state)
                torch.cuda.synchronize()
                results[layout][name] = _run_update(worker, batches)

            assert tuple(parameter._version for parameter in frozen) == frozen_versions
            assert all(parameter.grad is None for parameter in frozen)
            performance[layout] = _measure_steps(
                worker,
                _benchmark_batch(packed=layout == "packed"),
                initial_state,
            )
        finally:
            worker.close()
            torch.cuda.empty_cache()

    artifact = {"mode": mode, "performance": performance, "results": results}
    torch.save(artifact, output_path)
    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "capability": list(torch.cuda.get_device_capability()),
                "device": torch.cuda.get_device_name(),
                "mode": mode,
                "model": QWEN3_0_6B.repo_id,
                "performance": performance,
                "revision": QWEN3_0_6B.revision,
                "source_root": str(Path(cti.__file__).resolve().parents[2]),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _runtime_env(tmp_path: Path, mode: str) -> tuple[dict[str, str], Path]:
    env, log_root = isolated_runtime_env(tmp_path, f"activation-checkpointing-{mode}")
    env["PYTHONPATH"] = source_pythonpath(REPO_ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env, log_root


def test_full_activation_checkpointing_matches_eager_and_reduces_memory(
    tmp_path: Path,
) -> None:
    import torch

    artifacts = {}
    receipts = {}
    for mode in ("none", "full"):
        output_path = tmp_path / f"{mode}.pt"
        env, log_root = _runtime_env(tmp_path, mode)
        output = run_isolated_process(
            (
                sys.executable,
                "-m",
                "tests.gpu.training.test_activation_checkpointing",
                "--mode",
                mode,
                "--output",
                str(output_path),
            ),
            log_root / f"{mode}.log",
            env=env,
            cwd=REPO_ROOT,
            timeout=1_200,
        )
        receipt_lines = [
            line.removeprefix(RECEIPT_PREFIX)
            for line in output.splitlines()
            if line.startswith(RECEIPT_PREFIX)
        ]
        assert len(receipt_lines) == 1
        receipts[mode] = json.loads(receipt_lines[0])
        artifacts[mode] = torch.load(output_path, map_location="cpu", weights_only=True)

    for layout in ("padded", "packed"):
        tolerance = PADDED_BF16_TOLERANCE if layout == "padded" else PACKED_BF16_TOLERANCE
        for scenario in ("combined", "partitioned"):
            eager = artifacts["none"]["results"][layout][scenario]
            checkpointed = artifacts["full"]["results"][layout][scenario]
            assert checkpointed["num_microbatches"] == eager["num_microbatches"]
            assert checkpointed["num_target_tokens"] == eager["num_target_tokens"] == 11
            assert checkpointed["loss_sum"] == eager["loss_sum"]
            assert checkpointed["grad_norm"] == eager["grad_norm"]
            for name in (
                "gradients",
                "live_before_update",
                "master_before_update",
                "updated_live",
                "updated_master",
            ):
                torch.testing.assert_close(checkpointed[name], eager[name], rtol=0, atol=0)

        for mode in ("none", "full"):
            combined = artifacts[mode]["results"][layout]["combined"]
            partitioned = artifacts[mode]["results"][layout]["partitioned"]
            assert combined["num_target_tokens"] == partitioned["num_target_tokens"] == 11
            loss_error = measure_tensor_error(
                torch.tensor(partitioned["loss_sum"]),
                torch.tensor(combined["loss_sum"]),
            )
            gradient_error = measure_tensor_error(
                partitioned["gradients"],
                combined["gradients"],
            )
            state_error = measure_tensor_error(
                partitioned["updated_master"],
                combined["updated_master"],
            )
            assert loss_error.max_absolute <= tolerance.loss_max_absolute
            assert loss_error.global_relative_l2 <= tolerance.loss_global_relative_l2
            assert gradient_error.global_relative_l2 <= tolerance.gradient_global_relative_l2
            assert state_error.max_absolute <= UPDATE_STATE_MAX_ABSOLUTE
            assert state_error.global_relative_l2 <= UPDATE_STATE_GLOBAL_RELATIVE_L2

        eager = artifacts["none"]["performance"][layout]
        checkpointed = artifacts["full"]["performance"][layout]
        assert checkpointed["peak_step_bytes"] < eager["peak_step_bytes"]

    for mode, receipt in receipts.items():
        assert_cuda_device_metadata(receipt)
        assert receipt["mode"] == mode
        assert receipt["model"] == QWEN3_0_6B.repo_id
        assert receipt["revision"] == QWEN3_0_6B.revision
        assert Path(receipt["source_root"]) == REPO_ROOT
    print(RECEIPT_PREFIX + json.dumps(receipts, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("none", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _run_child(args.mode, args.output)
