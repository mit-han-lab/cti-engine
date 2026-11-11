from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    isolated_runtime_env,
    require_cuda_device,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT
from tests.support.training import forward_backward, optim_step, synchronize_optimizer_parameters

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

if TYPE_CHECKING:
    import torch

    from cti.layers import Adapter
    from cti.training import TrainingBatch
    from cti.training._checkpoint import OptimizerState

GOLDEN_PATH = Path(__file__).with_name("lora_training_trajectory_golden.json")
RECEIPT_PREFIX = "CTI_LORA_TRAINING_TRAJECTORY="

SCHEMA_VERSION = 4
ADAPTER_SEED = 20260706
PAD_TOKEN_ID = 151643
NUM_STEPS = 32
EXPECTED_ADAPTER_TENSORS = 392
EXPECTED_ADAPTER_ELEMENTS = 2_523_136
SAMPLES = (
    (785, 14002, 315, 4017, 374, 9255, 11, 323),
    (785, 7772, 11580, 304),
    (785, 14002),
)


class _Digest(Protocol):
    def update(self, data: bytes, /) -> object: ...


def _model_snapshot() -> str:
    return str(QWEN3_0_6B.snapshot())


def _padded_batch() -> TrainingBatch:
    import torch

    from cti.training import SFT, PaddedTokens, TrainingBatch

    lengths = torch.tensor(tuple(map(len, SAMPLES)), dtype=torch.int64)
    token_ids = torch.full(
        (len(SAMPLES), int(lengths.max())),
        PAD_TOKEN_ID,
        dtype=torch.int64,
    )
    for row, sample in enumerate(SAMPLES):
        token_ids[row, : len(sample)] = torch.tensor(sample, dtype=torch.int64)
    batch = TrainingBatch(
        PaddedTokens(token_ids, lengths, PAD_TOKEN_ID),
        torch.arange(token_ids.shape[1] - 1).unsqueeze(0) < (lengths - 1).unsqueeze(1),
        SFT(),
    )
    assert batch.num_targets == 11
    return batch


def _initialize_adapters(
    adapters: Adapter,
) -> tuple[tuple[str, torch.Tensor], ...]:
    import torch

    named_parameters = adapters.named_parameters()
    generator = torch.Generator(device="cpu").manual_seed(ADAPTER_SEED)
    with torch.no_grad():
        for _, parameter in named_parameters:
            initial = torch.empty(parameter.shape, dtype=torch.float32)
            initial.normal_(mean=0.0, std=0.02, generator=generator)
            parameter.copy_(initial)
    return named_parameters


def _write_field(digest: _Digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _tensor_fingerprint(
    domain: str,
    named_tensors: tuple[tuple[str, torch.Tensor], ...],
) -> str:
    import torch

    digest = hashlib.sha256()
    _write_field(digest, b"cti:lora-training-trajectory:v1")
    _write_field(digest, domain.encode())
    for name, tensor in named_tensors:
        assert tensor.layout == torch.strided
        cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
        _write_field(digest, name.encode())
        _write_field(digest, str(cpu_tensor.dtype).encode())
        _write_field(digest, json.dumps(list(cpu_tensor.shape)).encode())
        _write_field(digest, cpu_tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _optimizer_moments(
    state: OptimizerState,
    named_parameters: tuple[tuple[str, torch.Tensor], ...],
    expected_step: int,
) -> tuple[tuple[tuple[str, torch.Tensor], ...], tuple[tuple[str, torch.Tensor], ...]]:
    """The AdamW moments of a checkpoint after ``expected_step`` steps, in parameter order."""

    import torch

    assert state.names == tuple(name for name, _ in named_parameters)
    assert all(master.dtype == torch.float32 for master in state.master)
    first_moments = []
    second_moments = []
    for index, (name, parameter) in enumerate(named_parameters):
        moments = state.adamw["state"][index]
        assert float(moments["step"]) == expected_step
        for moment in (moments["exp_avg"], moments["exp_avg_sq"]):
            assert moment.shape == parameter.shape
            assert moment.dtype == torch.float32
            assert bool(torch.isfinite(moment).all())
        first_moments.append((name, moments["exp_avg"]))
        second_moments.append((name, moments["exp_avg_sq"]))
    assert any(bool(torch.count_nonzero(moment)) for _, moment in first_moments)
    assert any(bool(torch.count_nonzero(moment)) for _, moment in second_moments)
    return tuple(first_moments), tuple(second_moments)


def _runtime_profile(
    snapshot: str,
    *,
    loss_chunk_size: int,
) -> dict[str, object]:
    import flashinfer
    import torch
    import transformers

    return {
        "model": QWEN3_0_6B.repo_id,
        "revision": QWEN3_0_6B.revision,
        "snapshot_revision": Path(snapshot).name,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "liger_kernel": version("liger-kernel"),
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "base_dtype": "float32",
        "adapter": {
            "dtype": "float32",
            "rank": 4,
            "alpha": 8.0,
            "targets": ["attention", "mlp"],
            "seed": ADAPTER_SEED,
            "initialization": "cpu-normal-mean-0-std-0.02-live-name-order",
        },
        "batch": {
            "samples": [list(sample) for sample in SAMPLES],
            "pad_token_id": PAD_TOKEN_ID,
            "num_target_tokens": 11,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "foreach": True,
            "fused": False,
        },
        "engine": {
            "runtime_kind": "training-only-oracle",
            "inference_resources_initialized": False,
            "training_attention": "torch-sdpa",
            "training_loss": {
                "implementation": "cti-fused-linear-cross-entropy",
                "kernel": "liger_kernel.ops.cross_entropy.liger_cross_entropy_kernel",
                "chunk_size": loss_chunk_size,
                "reduction": "sum",
                "accum_dtype": "float32",
            },
            "training_norm": {
                "operations": [
                    {
                        "implementation": "ctikernels.norm.training.rms_norm",
                        "providers": {
                            "frozen_weight": {
                                "module": "ctikernels.norm._frozen_rms_norm",
                                "callable": "frozen_rms_norm",
                            },
                            "trainable_weight": {
                                "module": "liger_kernel.transformers.functional",
                                "callable": "liger_rms_norm",
                            },
                        },
                    },
                ],
                "casting_mode": "llama",
                "alias_policy": {
                    "forward": "outputs-do-not-alias-inputs",
                    "backward": "preserve-grad-output",
                },
            },
        },
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def _trajectory_receipt() -> dict[str, object]:
    import torch

    from cti.config import (
        AdamWConfig,
        BatchLayout,
        LoRAConfig,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.training._parameters import resolve_for_model
    from tests.support.training import AdaptedTrainables
    from tests.support.training_runtime import PaddedTrainingOracleRuntime

    assert not torch.cuda.is_initialized()
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    torch.manual_seed(ADAPTER_SEED)

    snapshot = _model_snapshot()
    assert Path(snapshot).name == QWEN3_0_6B.revision
    optimizer_config = AdamWConfig(
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    runtime = PaddedTrainingOracleRuntime(
        snapshot,
        dtype=torch.float32,
        lora_config=LoRAConfig(
            targets=("attention", "mlp"),
            rank=4,
            alpha=8.0,
        ),
        training_config=TrainingConfig(
            TrainablePlan.all_declared(),
            optimizer_config,
            TrainingBatching(
                BatchLayout.PADDED,
                StaticBatching(
                    microbatch_size=len(SAMPLES),
                    max_tokens_per_sample=64,
                ),
            ),
        ),
    )
    torch.manual_seed(ADAPTER_SEED)
    torch.cuda.manual_seed_all(ADAPTER_SEED)
    assert torch.initial_seed() == ADAPTER_SEED
    assert torch.cuda.initial_seed() == ADAPTER_SEED
    try:
        model = runtime.model
        adapters = runtime.adapters
        named_parameters = _initialize_adapters(adapters)
        parameters = tuple(parameter for _, parameter in named_parameters)
        assert len(parameters) == EXPECTED_ADAPTER_TENSORS
        assert sum(parameter.numel() for parameter in parameters) == EXPECTED_ADAPTER_ELEMENTS
        assert all(parameter.dtype == torch.float32 for parameter in parameters)
        assert all(parameter.requires_grad and parameter.is_leaf for parameter in parameters)

        parameter_ids = tuple(map(id, parameters))
        parameter_pointers = tuple(parameter.data_ptr() for parameter in parameters)
        base_tensors = tuple(model.state_dict().values())
        assert base_tensors
        assert all(not tensor.requires_grad for tensor in base_tensors)
        assert all(tensor.dtype == torch.float32 for tensor in base_tensors)

        batch = _padded_batch()
        trainable_parameters = resolve_for_model(
            TrainablePlan(tuple(name for name, _ in named_parameters)),
            AdaptedTrainables(model, adapters),
        )
        runner = runtime.runner
        synchronize_optimizer_parameters(runner, named_parameters)
        assert runner.trainable_parameters.selected_names == trainable_parameters.selected_names
        loss_operation = model.lm_head._training_loss
        # The optimizer's state exists from startup but holds no update yet.
        initial_state = runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer
        assert initial_state.config == optimizer_config
        assert all(
            float(moments["step"]) == 0
            and not torch.count_nonzero(moments["exp_avg"])
            and not torch.count_nonzero(moments["exp_avg_sq"])
            for moments in initial_state.adamw["state"].values()
        )
        initial_fingerprint = _tensor_fingerprint("parameters-v1", named_parameters)

        steps = []
        first_moments = ()
        second_moments = ()
        for expected_step in range(1, NUM_STEPS + 1):
            result = forward_backward(runner, batch)
            loss_sum = float(result.loss_sum)
            assert result.normalization_count == 11
            assert math.isfinite(loss_sum) and loss_sum > 0

            live_before = tuple(parameter.detach().clone() for parameter in parameters)
            grad_norm = optim_step(runner, result.normalization_count)
            for live, before in zip(parameters, live_before, strict=True):
                torch.testing.assert_close(live, before, rtol=0, atol=0)
            runner.publish()
            assert math.isfinite(grad_norm) and grad_norm > 0
            assert all(parameter.grad is None for parameter in parameters)
            assert tuple(map(id, parameters)) == parameter_ids
            assert tuple(parameter.data_ptr() for parameter in parameters) == parameter_pointers
            assert all(tensor.grad is None for tensor in base_tensors)
            state = runner.checkpoint(num_optimizer_steps=expected_step, model_revision=0).optimizer
            assert state.config == optimizer_config
            first_moments, second_moments = _optimizer_moments(
                state,
                named_parameters,
                expected_step,
            )
            steps.append(
                {
                    "step": expected_step,
                    "num_target_tokens": result.normalization_count,
                    "loss_sum_hex": loss_sum.hex(),
                    "grad_norm_hex": grad_norm.hex(),
                }
            )

        torch.cuda.synchronize()
        assert tuple(map(id, parameters)) == parameter_ids
        assert tuple(parameter.data_ptr() for parameter in parameters) == parameter_pointers
        assert all(parameter.grad is None for parameter in parameters)
        assert all(tensor.grad is None for tensor in base_tensors)
        assert all(bool(torch.isfinite(parameter).all()) for parameter in parameters)

        final_parameter_fingerprint = _tensor_fingerprint("parameters-v1", named_parameters)
        assert final_parameter_fingerprint != initial_fingerprint
        assert len({step["loss_sum_hex"] for step in steps}) > 1

        golden = {
            "schema_version": SCHEMA_VERSION,
            "profile": _runtime_profile(
                snapshot,
                loss_chunk_size=loss_operation.chunk_size,
            ),
            "initial_parameter_sha256": initial_fingerprint,
            "steps": steps,
            "final": {
                "optimizer_step": NUM_STEPS,
                "adapter_tensor_count": len(parameters),
                "adapter_element_count": sum(parameter.numel() for parameter in parameters),
                "parameter_sha256": final_parameter_fingerprint,
                "first_moment_sha256": _tensor_fingerprint("first-moments-v1", first_moments),
                "second_moment_sha256": _tensor_fingerprint("second-moments-v1", second_moments),
            },
        }
        return {
            "golden": golden,
            "assertions": {
                "real_forward_backward_steps": NUM_STEPS,
                "real_optimizer_commits": NUM_STEPS,
                "base_frozen": True,
                "base_gradients_absent": True,
                "adapter_identity_preserved": True,
                "adapter_storage_preserved": True,
                "gradients_cleared_after_commit": True,
                "optimizer_state_complete_and_finite": True,
                "trajectory_non_inert": True,
            },
        }
    finally:
        runtime.close()


def _run_child() -> None:
    receipt = _trajectory_receipt()
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True, separators=(",", ":")))


def _receipt_from_output(output: str) -> dict[str, object]:
    receipt_lines = [
        line.removeprefix(RECEIPT_PREFIX)
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    assert len(receipt_lines) == 1
    receipt = json.loads(receipt_lines[0])
    assert isinstance(receipt, dict)
    return receipt


def _mapping(value: object, name: str) -> dict[str, object]:
    assert isinstance(value, dict), f"{name} must be an object"
    assert all(isinstance(key, str) for key in value)
    return value


def _require_keys(value: dict[str, object], expected: set[str], name: str) -> None:
    assert set(value) == expected, f"{name} has invalid keys"


def _require_sha256(value: object, name: str) -> None:
    assert isinstance(value, str)
    assert re.fullmatch(r"[0-9a-f]{64}", value), f"{name} is not canonical SHA-256"


def _validate_profile(value: object) -> None:
    profile = _mapping(value, "profile")
    _require_keys(
        profile,
        {
            "model",
            "revision",
            "snapshot_revision",
            "python",
            "torch",
            "cuda",
            "flashinfer",
            "liger_kernel",
            "transformers",
            "device",
            "capability",
            "base_dtype",
            "adapter",
            "batch",
            "optimizer",
            "engine",
            "determinism",
        },
        "profile",
    )
    assert profile["model"] == QWEN3_0_6B.repo_id
    assert profile["revision"] == profile["snapshot_revision"] == QWEN3_0_6B.revision
    assert profile["python"] == "3.12.13"
    assert profile["torch"] == "2.13.0+cu130"
    assert profile["cuda"] == "13.0"
    assert profile["flashinfer"] == "0.7.0.post1"
    assert profile["liger_kernel"] == "0.8.0"
    assert profile["transformers"] == "5.16.1"
    assert profile["device"] == "NVIDIA GeForce RTX 5090"
    assert profile["capability"] == [12, 0]
    assert profile["base_dtype"] == "float32"
    assert profile["adapter"] == {
        "dtype": "float32",
        "rank": 4,
        "alpha": 8.0,
        "targets": ["attention", "mlp"],
        "seed": ADAPTER_SEED,
        "initialization": "cpu-normal-mean-0-std-0.02-live-name-order",
    }
    assert profile["batch"] == {
        "samples": [list(sample) for sample in SAMPLES],
        "pad_token_id": PAD_TOKEN_ID,
        "num_target_tokens": 11,
    }
    assert profile["optimizer"] == {
        "name": "torch.optim.AdamW",
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "foreach": True,
        "fused": False,
    }
    assert profile["engine"] == {
        "runtime_kind": "training-only-oracle",
        "inference_resources_initialized": False,
        "training_attention": "torch-sdpa",
        "training_loss": {
            "implementation": "cti-fused-linear-cross-entropy",
            "kernel": "liger_kernel.ops.cross_entropy.liger_cross_entropy_kernel",
            "chunk_size": 256,
            "reduction": "sum",
            "accum_dtype": "float32",
        },
        "training_norm": {
            "operations": [
                {
                    "implementation": "ctikernels.norm.training.rms_norm",
                    "providers": {
                        "frozen_weight": {
                            "module": "ctikernels.norm._frozen_rms_norm",
                            "callable": "frozen_rms_norm",
                        },
                        "trainable_weight": {
                            "module": "liger_kernel.transformers.functional",
                            "callable": "liger_rms_norm",
                        },
                    },
                },
            ],
            "casting_mode": "llama",
            "alias_policy": {
                "forward": "outputs-do-not-alias-inputs",
                "backward": "preserve-grad-output",
            },
        },
    }
    assert profile["determinism"] == {
        "cublas_workspace_config": ":4096:8",
        "algorithms": True,
        "warn_only": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }


def _validate_golden(value: dict[str, object]) -> None:
    _require_keys(
        value,
        {"schema_version", "profile", "initial_parameter_sha256", "steps", "final"},
        "golden",
    )
    assert isinstance(value["schema_version"], int)
    assert not isinstance(value["schema_version"], bool)
    assert value["schema_version"] == SCHEMA_VERSION
    _validate_profile(value["profile"])
    _require_sha256(value["initial_parameter_sha256"], "initial parameters")

    steps = value["steps"]
    assert isinstance(steps, list) and len(steps) == NUM_STEPS
    for expected_step, raw_step in enumerate(steps, start=1):
        step = _mapping(raw_step, f"step {expected_step}")
        _require_keys(
            step,
            {"step", "num_target_tokens", "loss_sum_hex", "grad_norm_hex"},
            f"step {expected_step}",
        )
        assert isinstance(step["step"], int) and not isinstance(step["step"], bool)
        assert step["step"] == expected_step
        assert isinstance(step["num_target_tokens"], int)
        assert not isinstance(step["num_target_tokens"], bool)
        assert step["num_target_tokens"] == 11
        for signal in ("loss_sum_hex", "grad_norm_hex"):
            encoded = step[signal]
            assert isinstance(encoded, str)
            decoded = float.fromhex(encoded)
            assert decoded.hex() == encoded
            assert math.isfinite(decoded) and decoded > 0

    final = _mapping(value["final"], "final")
    _require_keys(
        final,
        {
            "optimizer_step",
            "adapter_tensor_count",
            "adapter_element_count",
            "parameter_sha256",
            "first_moment_sha256",
            "second_moment_sha256",
        },
        "final",
    )
    for name, expected in (
        ("optimizer_step", NUM_STEPS),
        ("adapter_tensor_count", EXPECTED_ADAPTER_TENSORS),
        ("adapter_element_count", EXPECTED_ADAPTER_ELEMENTS),
    ):
        assert isinstance(final[name], int) and not isinstance(final[name], bool)
        assert final[name] == expected
    for name in ("parameter_sha256", "first_moment_sha256", "second_moment_sha256"):
        _require_sha256(final[name], name)
    assert final["parameter_sha256"] != value["initial_parameter_sha256"]
    assert len({_mapping(step, "step")["loss_sum_hex"] for step in steps}) > 1


def _assert_matches_golden(actual: dict[str, object], expected: dict[str, object]) -> None:
    _validate_golden(actual)
    _validate_golden(expected)
    assert actual == expected


def test_lora_training_trajectory_matches_32_step_golden(tmp_path: Path) -> None:
    require_cuda_device("NVIDIA GeForce RTX 5090", (12, 0))
    assert GOLDEN_PATH.is_file(), "32-step golden asset must be generated from two matching runs"
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert isinstance(expected, dict)
    _validate_golden(expected)

    env, log_root = isolated_runtime_env(tmp_path, "lora-training-trajectory")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.training.test_lora_training_trajectory"),
        log_root / "lora-training-trajectory.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=600,
    )
    receipt = _receipt_from_output(output)
    _require_keys(receipt, {"assertions", "golden"}, "receipt")
    assert receipt["assertions"] == {
        "real_forward_backward_steps": NUM_STEPS,
        "real_optimizer_commits": NUM_STEPS,
        "base_frozen": True,
        "base_gradients_absent": True,
        "adapter_identity_preserved": True,
        "adapter_storage_preserved": True,
        "gradients_cleared_after_commit": True,
        "optimizer_state_complete_and_finite": True,
        "trajectory_non_inert": True,
    }
    actual = receipt["golden"]
    assert isinstance(actual, dict)
    _assert_matches_golden(actual, expected)


if __name__ == "__main__":
    _run_child()
