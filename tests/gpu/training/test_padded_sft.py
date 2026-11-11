from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    assert_cuda_device_metadata,
    fixed_kv_page_capacity,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT
from tests.support.training import (
    PAD_TOKEN_ID,
    PADDED_BF16_TOLERANCE,
    RAW_SAMPLES,
    batched_valid_logits,
    clone_main_gradients,
    expected_qwen_adapter_names,
    forward_backward,
    measure_gradient_error,
    measure_tensor_error,
    padded_sft_batch,
    qwen3_model_snapshot,
    synchronize_optimizer_parameters,
    training_logits,
)

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

if TYPE_CHECKING:
    import torch

    from cti.models.qwen3.model import Qwen3ForCausalLM

RECEIPT_PREFIX = "CTI_PADDED_SFT="
BF16_ADAPTER_SEED = 20260723
FP32_ADAPTER_SEED = 20260722

FP32_LOSS_RTOL = 1e-4
FP32_LOSS_ATOL = 1e-5
FP32_GRADIENT_GLOBAL_RELATIVE_L2 = 5e-4
FP32_GRADIENT_WORST_RELATIVE_L2 = 5e-3
# Exact 1.25x bounds from the pre-registered 20260717-20260721 calibration seeds.
# A maintained environment change requires fresh-seed recalibration; never widen a
# failing bound in place.
FP32_LOGIT_GLOBAL_RELATIVE_L2 = 2.374720935449659e-6
FP32_LOGIT_RELATIVE_LINF = 3.470281058071123e-6


def _samplewise_oracle(
    model: Qwen3ForCausalLM,
    samples: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    logits_parts = []
    loss_parts = []
    for sample in samples:
        input_ids = torch.tensor([sample[:-1]], dtype=torch.int64, device="cuda")
        target_ids = torch.tensor(sample[1:], dtype=torch.int64, device="cuda")
        logits = training_logits(model, input_ids).squeeze(0)
        logits_parts.append(logits)
        loss_parts.append(
            -logits.float().log_softmax(dim=-1).gather(1, target_ids.unsqueeze(1)).sum()
        )
    return torch.cat(logits_parts), torch.stack(loss_parts).sum()


def _mixed_error_ratio(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> float:
    difference = (actual.float() - expected.float()).abs()
    allowance = atol + rtol * expected.float().abs()
    return float((difference / allowance).max())


def _assert_gradients_exact(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> None:
    import torch

    assert len(actual) == len(expected)
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=0,
            atol=0,
        )


def _run_child(dtype_name: str, adapter_seed: int, *, measure_only: bool) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from cti.config import (
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
    from cti.training._parameters import resolve_for_model
    from tests.support.gpu import start_rank
    from tests.support.training import AdaptedTrainables
    from tests.support.training_runtime import PaddedTrainingOracleRuntime

    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_by_name[dtype_name]
    deterministic_algorithms = dtype is torch.float32
    if deterministic_algorithms:
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(adapter_seed)
    snapshot = qwen3_model_snapshot()
    lora_config = LoRAConfig(
        targets=("attention", "mlp"),
        rank=4,
        alpha=8.0,
    )
    training_config = TrainingConfig(
        TrainablePlan.all_declared(),
        AdamWConfig(),
        TrainingBatching(
            BatchLayout.PADDED,
            StaticBatching(
                microbatch_size=len(RAW_SAMPLES),
                max_tokens_per_sample=64,
            ),
        ),
    )
    if dtype is torch.float32:
        runtime = PaddedTrainingOracleRuntime(
            snapshot,
            dtype=dtype,
            lora_config=lora_config,
            training_config=training_config,
        )
        model = runtime.model
        adapters = runtime.adapters
        runner = runtime.runner
    else:
        with fixed_kv_page_capacity(8):
            runtime = start_rank(
                EngineOptions(
                    model=ModelOptions(path=snapshot, dtype=dtype_name, max_seq_len=64),
                    scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=64),
                    cache=CacheOptions(prefix="naive"),
                    lora=lora_config,
                    training=training_config,
                ),
                EngineTuning(attention_backend="fi", cuda_graph_bs=()),
                checks=True,
            )
        model_runner = runtime.executor
        model = model_runner.model
        adapters = runtime.worker.host.adapter
        assert adapters is not None
        runner = runtime.training
    torch.cuda.manual_seed_all(adapter_seed)
    torch.cuda.reset_peak_memory_stats()
    try:
        named_adapters = adapters.named_parameters()
        expected_adapter_names = expected_qwen_adapter_names()
        assert tuple(name for name, _ in named_adapters) == expected_adapter_names
        trainable_parameters = resolve_for_model(
            TrainablePlan(tuple(name for name, _ in named_adapters)),
            AdaptedTrainables(model, adapters),
        )
        assert len(expected_adapter_names) == 392
        assert trainable_parameters.selected_names == expected_adapter_names
        assert all(
            actual is expected
            for actual, (_, expected) in zip(
                trainable_parameters.parameters(),
                named_adapters,
                strict=True,
            )
        )
        assert all(
            actual.untyped_storage() == expected.untyped_storage()
            for actual, (_, expected) in zip(
                trainable_parameters.parameters(),
                named_adapters,
                strict=True,
            )
        )
        named_base = tuple(model.state_dict().items())
        assert trainable_parameters.complement_names == tuple(name for name, _ in named_base)
        assert all(
            actual is expected
            for actual, (_, expected) in zip(
                trainable_parameters.complement(),
                named_base,
                strict=True,
            )
        )

        adapter_names = tuple(adapters.state_dict())
        adapter_pointers = tuple(
            parameter.data_ptr() for parameter in tuple(t for _, t in adapters.named_parameters())
        )
        base_tensors = tuple(tensor for _, tensor in named_base)
        assert all(not tensor.requires_grad for tensor in base_tensors)
        assert all(tensor.dtype == dtype for tensor in base_tensors)
        assert all(
            parameter.dtype == dtype
            for parameter in tuple(t for _, t in adapters.named_parameters())
        )
        assert torch.are_deterministic_algorithms_enabled() == deterministic_algorithms
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32

        hf_max_abs: float | None = None
        hf_mixed_error_ratio: float | None = None
        if dtype is torch.bfloat16 and not measure_only:
            hf_model = AutoModelForCausalLM.from_pretrained(
                snapshot,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            ).to("cuda")
            hf_model.eval()
            hf_max_abs = 0.0
            hf_mixed_error_ratio = 0.0
            with torch.no_grad():
                for sample in RAW_SAMPLES:
                    input_ids = torch.tensor([sample[:-1]], dtype=torch.int64, device="cuda")
                    actual = training_logits(model, input_ids)
                    expected = hf_model(input_ids=input_ids, use_cache=False).logits
                    hf_max_abs = max(
                        hf_max_abs,
                        float((actual.float() - expected.float()).abs().max()),
                    )
                    hf_mixed_error_ratio = max(
                        hf_mixed_error_ratio,
                        _mixed_error_ratio(actual, expected, rtol=2e-2, atol=3.5e-1),
                    )
                    torch.testing.assert_close(
                        actual.float(), expected.float(), rtol=2e-2, atol=3.5e-1
                    )
            del hf_model
            torch.cuda.empty_cache()

        torch.manual_seed(adapter_seed)
        with torch.no_grad():
            for tensor in adapters.state_dict().values():
                tensor.normal_(mean=0.0, std=0.02)

        synchronize_optimizer_parameters(runner, named_adapters)
        assert runner.model is model
        assert runner.trainable_parameters.selected_names == trainable_parameters.selected_names
        assert all(
            actual is expected
            for actual, expected in zip(
                runner.trainable_parameters.parameters(),
                trainable_parameters.parameters(),
                strict=True,
            )
        )
        state = runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer
        assert state.names == expected_adapter_names
        for (name, live), master in zip(named_adapters, state.master, strict=True):
            assert master.dtype == torch.float32, name
            torch.testing.assert_close(master, live.float().cpu(), rtol=0, atol=0)

        batch = padded_sft_batch(RAW_SAMPLES, PAD_TOKEN_ID)
        changed_pad_batch = padded_sft_batch(RAW_SAMPLES, 0)
        first_partition_batch = padded_sft_batch(RAW_SAMPLES[:1], PAD_TOKEN_ID)
        second_partition_batch = padded_sft_batch(RAW_SAMPLES[1:], PAD_TOKEN_ID)
        permutation = (2, 0, 1)
        permuted_samples = tuple(RAW_SAMPLES[index] for index in permutation)
        permuted_batch = padded_sft_batch(permuted_samples, PAD_TOKEN_ID)

        with torch.no_grad():
            batched_logits = batched_valid_logits(model, batch)
            sample_logits, _ = _samplewise_oracle(model, RAW_SAMPLES)
            changed_pad_logits = batched_valid_logits(model, changed_pad_batch)
            permuted_logits = batched_valid_logits(model, permuted_batch)
            permuted_oracle_logits, _ = _samplewise_oracle(model, permuted_samples)
        batch_sample_logits_error = measure_tensor_error(batched_logits, sample_logits)
        permuted_logits_error = measure_tensor_error(permuted_logits, permuted_oracle_logits)
        torch.testing.assert_close(batched_logits, changed_pad_logits, rtol=0, atol=0)
        if not measure_only:
            if dtype is torch.float32:
                for error in (batch_sample_logits_error, permuted_logits_error):
                    assert error.global_relative_l2 <= FP32_LOGIT_GLOBAL_RELATIVE_L2, error
                    assert error.relative_linf <= FP32_LOGIT_RELATIVE_LINF, error
            else:
                for error in (batch_sample_logits_error, permuted_logits_error):
                    assert error.max_absolute <= PADDED_BF16_TOLERANCE.logit_max_absolute, error
                    assert (
                        error.global_relative_l2 <= PADDED_BF16_TOLERANCE.logit_global_relative_l2
                    ), error

        runner.discard()
        result = forward_backward(runner, batch)
        production_gradients = clone_main_gradients(runner, named_adapters)

        runner.discard()
        _, oracle_loss = _samplewise_oracle(model, RAW_SAMPLES)
        oracle_loss.backward()
        # The runner's metrics come back on the host.
        oracle_loss = oracle_loss.detach().cpu()
        oracle_gradients = clone_main_gradients(runner, named_adapters)
        batch_sample_loss_error = measure_tensor_error(result.loss_sum, oracle_loss)
        gradient_error = measure_gradient_error(
            adapter_names,
            production_gradients,
            oracle_gradients,
        )
        if not measure_only:
            if dtype is torch.float32:
                torch.testing.assert_close(
                    result.loss_sum,
                    oracle_loss,
                    rtol=FP32_LOSS_RTOL,
                    atol=FP32_LOSS_ATOL,
                )
                assert gradient_error.global_relative_l2 <= FP32_GRADIENT_GLOBAL_RELATIVE_L2
                assert gradient_error.worst_relative_l2 <= FP32_GRADIENT_WORST_RELATIVE_L2
            else:
                assert (
                    batch_sample_loss_error.max_absolute <= PADDED_BF16_TOLERANCE.loss_max_absolute
                )
                assert (
                    batch_sample_loss_error.global_relative_l2
                    <= PADDED_BF16_TOLERANCE.loss_global_relative_l2
                )
                assert (
                    gradient_error.global_relative_l2
                    <= PADDED_BF16_TOLERANCE.gradient_global_relative_l2
                )
                assert (
                    gradient_error.worst_relative_l2
                    <= PADDED_BF16_TOLERANCE.gradient_worst_relative_l2
                )

        runner.discard()
        changed_pad_result = forward_backward(runner, changed_pad_batch)
        changed_pad_gradients = clone_main_gradients(runner, named_adapters)
        torch.testing.assert_close(result.loss_sum, changed_pad_result.loss_sum, rtol=0, atol=0)
        _assert_gradients_exact(production_gradients, changed_pad_gradients)

        runner.discard()
        first_partition = forward_backward(runner, first_partition_batch)
        second_partition = forward_backward(runner, second_partition_batch)
        partitioned_gradients = clone_main_gradients(runner, named_adapters)
        partitioned_loss = first_partition.loss_sum + second_partition.loss_sum
        partition_loss_error = measure_tensor_error(result.loss_sum, partitioned_loss)
        assert (
            first_partition.num_targets + second_partition.num_targets == result.normalization_count
        )
        partition_error = measure_gradient_error(
            adapter_names,
            production_gradients,
            partitioned_gradients,
        )
        if not measure_only:
            if dtype is torch.float32:
                torch.testing.assert_close(
                    result.loss_sum,
                    partitioned_loss,
                    rtol=FP32_LOSS_RTOL,
                    atol=FP32_LOSS_ATOL,
                )
                assert partition_error.global_relative_l2 <= FP32_GRADIENT_GLOBAL_RELATIVE_L2
                assert partition_error.worst_relative_l2 <= FP32_GRADIENT_WORST_RELATIVE_L2
            else:
                assert partition_loss_error.max_absolute <= PADDED_BF16_TOLERANCE.loss_max_absolute
                assert (
                    partition_loss_error.global_relative_l2
                    <= PADDED_BF16_TOLERANCE.loss_global_relative_l2
                )
                assert (
                    partition_error.global_relative_l2
                    <= PADDED_BF16_TOLERANCE.gradient_global_relative_l2
                )
                assert (
                    partition_error.worst_relative_l2
                    <= PADDED_BF16_TOLERANCE.gradient_worst_relative_l2
                )

        runner.discard()
        assert result.normalization_count == sum(len(sample) - 1 for sample in RAW_SAMPLES)
        assert (
            tuple(
                parameter.data_ptr()
                for parameter in tuple(t for _, t in adapters.named_parameters())
            )
            == adapter_pointers
        )
        assert all(tensor.grad is None for tensor in base_tensors)

        receipt = {
            "adapter_seed": adapter_seed,
            "adapter_tensor_count": len(adapter_pointers),
            "batch_sample_gradients": gradient_error._asdict(),
            "batch_sample_logits": batch_sample_logits_error._asdict(),
            "batch_sample_loss": batch_sample_loss_error._asdict(),
            "capability": list(torch.cuda.get_device_capability()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "device": torch.cuda.get_device_name(),
            "dtype": dtype_name,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "hf_max_abs": hf_max_abs,
            "hf_mixed_error_ratio": hf_mixed_error_ratio,
            "loss_sum": float(result.loss_sum),
            "measure_only": measure_only,
            "model": QWEN3_0_6B.repo_id,
            "num_target_tokens": result.normalization_count,
            "partition_gradients": partition_error._asdict(),
            "partition_loss": partition_loss_error._asdict(),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "permuted_logits": permuted_logits_error._asdict(),
            "revision": QWEN3_0_6B.revision,
            "tf32_cuda_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        }
        torch.cuda.synchronize()
        print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))
    finally:
        runtime.close()


def test_padded_sft_qwen3_forward_backward(tmp_path: Path) -> None:
    receipts = {}
    cases = (
        ("bfloat16", BF16_ADAPTER_SEED),
        ("float32", FP32_ADAPTER_SEED),
    )
    for dtype_name, adapter_seed in cases:
        env, log_root = isolated_runtime_env(tmp_path, f"padded-sft-{dtype_name}")
        if dtype_name == "float32":
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        output = run_isolated_process(
            (
                sys.executable,
                "-m",
                "tests.gpu.training.test_padded_sft",
                "--dtype",
                dtype_name,
                "--adapter-seed",
                str(adapter_seed),
            ),
            log_root / f"padded-sft-{dtype_name}.log",
            env=env,
            cwd=REPO_ROOT,
            timeout=600,
        )

        receipt_lines = [
            line.removeprefix(RECEIPT_PREFIX)
            for line in output.splitlines()
            if line.startswith(RECEIPT_PREFIX)
        ]
        assert len(receipt_lines) == 1
        receipt = json.loads(receipt_lines[0])
        assert_cuda_device_metadata(receipt)
        assert receipt["adapter_seed"] == adapter_seed
        assert receipt["adapter_tensor_count"] == 392
        assert receipt["num_target_tokens"] == 11
        assert receipt["dtype"] == dtype_name
        assert receipt["deterministic_algorithms"] == (dtype_name == "float32")
        assert receipt["float32_matmul_precision"] == "highest"
        assert not receipt["tf32_cuda_matmul"]
        assert not receipt["tf32_cudnn"]
        if dtype_name == "float32":
            assert receipt["cublas_workspace_config"] == ":4096:8"
        receipts[dtype_name] = receipt

    assert receipts["bfloat16"]["hf_max_abs"] is not None
    assert receipts["bfloat16"]["hf_mixed_error_ratio"] <= 1
    assert receipts["float32"]["hf_max_abs"] is None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--adapter-seed", type=int, required=True)
    parser.add_argument("--measure-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _run_child(args.dtype, args.adapter_seed, measure_only=args.measure_only)
