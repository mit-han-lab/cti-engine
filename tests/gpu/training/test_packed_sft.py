from __future__ import annotations

import argparse
import json
import math
import os
import sys
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from scripts.environment.contract import NativeBuildLayout
from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    assert_cuda_device_metadata,
    assert_ranks_reclaimed,
    engine_rank_pids,
    fixed_kv_page_capacity,
    isolated_runtime_env,
    jit_cache_env,
    run_isolated_process,
)
from tests.support.graph_captures import graph_captures
from tests.support.inference_correctness import GOLDEN_CASES
from tests.support.paths import REPO_ROOT, source_pythonpath
from tests.support.training import (
    PACKED_BF16_TOLERANCE,
    PAD_TOKEN_ID,
    RAW_SAMPLES,
    batched_valid_logits,
    clone_main_gradients,
    expected_qwen_adapter_names,
    forward_backward,
    measure_gradient_error,
    measure_tensor_error,
    optim_step,
    packed_sft_batch,
    qwen3_model_snapshot,
    synchronize_optimizer_parameters,
    training_logits,
)

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

RECEIPT_PREFIX = "CTI_PACKED_SFT="
PUBLIC_RECEIPT_PREFIX = "CTI_PACKED_SFT_PUBLIC="
ADAPTER_SEED = 20260813
PUBLIC_INITIALIZATION_SEED = 20260811
# CUDA foreach changes FP32 operation order; this is the near-zero-aware
# ``atol + rtol * abs(reference)`` gate against the independent loop oracle.
FOREACH_ORACLE_RTOL = 1e-6
FOREACH_ORACLE_ATOL = 1e-8
if TYPE_CHECKING:
    import torch

    from cti.config import AdamWConfig, BatchLayout
    from cti.models.qwen3.model import Qwen3ForCausalLM
    from cti.training import TrainingBatch
    from cti.training.runner import TrainingRunner


def _isolated_workspace_env(tmp_path: Path, component: str) -> tuple[dict[str, str], Path]:
    env, log_root = isolated_runtime_env(tmp_path, component)
    env["PYTHONPATH"] = source_pythonpath(REPO_ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env, log_root


def _packed_logits(
    model: Qwen3ForCausalLM,
    batch: TrainingBatch,
) -> torch.Tensor:
    import torch

    rows = batch.microbatch(tuple(range(batch.batch_size)), torch.device("cuda"))
    return training_logits(model, rows.input_ids, rows.packing)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    import torch

    difference = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float())
    assert denominator > 0
    return float(torch.linalg.vector_norm(difference) / denominator)


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    import torch

    return float(
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            expected.float().flatten(),
            dim=0,
        )
    )


def _masters(runner: TrainingRunner) -> tuple[torch.Tensor, ...]:
    """The settled runner's FP32 masters, one per (BF16) adapter, on the host."""

    masters = runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer.master
    assert all(master is not None for master in masters)
    return masters


def _master_deltas(
    runner: TrainingRunner,
    initial_live: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    masters = _masters(runner)
    assert len(masters) == len(initial_live)
    return tuple(
        master - initial.float().cpu()
        for master, initial in zip(masters, initial_live, strict=True)
    )


def _reference_optimizer_step(
    config: AdamWConfig,
    masters: tuple[torch.Tensor, ...],
    gradients: tuple[torch.Tensor, ...],
    num_target_tokens: int,
) -> tuple[tuple[torch.nn.Parameter, ...], torch.optim.AdamW, float]:
    import torch

    reference_parameters = tuple(
        torch.nn.Parameter(master.to(gradient.device))
        for master, gradient in zip(masters, gradients, strict=True)
    )
    reference_optimizer = torch.optim.AdamW(
        reference_parameters,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        foreach=False,
        fused=False,
    )
    for parameter, gradient in zip(reference_parameters, gradients, strict=True):
        parameter.grad = gradient.clone()
    raw_norm = torch.nn.utils.get_total_norm(
        gradients,
        norm_type=2.0,
        error_if_nonfinite=True,
        foreach=False,
    )
    normalized_norm = raw_norm / num_target_tokens
    with torch.no_grad():
        for parameter in reference_parameters:
            assert parameter.grad is not None
            parameter.grad.div_(num_target_tokens)
        if config.max_grad_norm is not None:
            torch.nn.utils.clip_grads_with_norm_(
                reference_parameters,
                config.max_grad_norm,
                normalized_norm,
                foreach=False,
            )
    reference_optimizer.step()
    return reference_parameters, reference_optimizer, float(normalized_norm)


def _run_child() -> None:
    import torch

    import cti
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
    from cti.training import SFTDatum, TrainingBatch
    from ctikernels.attention.varlen import CausalVarlenAttention
    from tests.support.gpu import start_rank

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(ADAPTER_SEED)
    expected_adapter_names = expected_qwen_adapter_names()
    optimizer_config = AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0)

    def create_worker(layout: BatchLayout) -> Any:
        return start_rank(
            EngineOptions(
                model=ModelOptions(path=qwen3_model_snapshot(), max_seq_len=64),
                scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=64),
                cache=CacheOptions(prefix="naive"),
                lora=LoRAConfig(
                    targets=("attention", "mlp"),
                    rank=4,
                    alpha=8.0,
                ),
                training=TrainingConfig(
                    TrainablePlan(expected_adapter_names),
                    optimizer_config,
                    batching=TrainingBatching(
                        layout=layout,
                        policy=StaticBatching(
                            microbatch_size=1,
                            max_tokens_per_sample=64,
                        ),
                    ),
                ),
            ),
            EngineTuning(attention_backend="fi", cuda_graph_bs=()),
            checks=True,
        )

    sparse_data = tuple(
        SFTDatum(
            token_ids=tuple(sample),
            target_mask=(False,) * (len(sample) - 1) + (True,),
        )
        for sample in RAW_SAMPLES
    )
    packed_batch = TrainingBatch.sft(sparse_data, BatchLayout.PACKED)
    padded_batch = TrainingBatch.sft(sparse_data, BatchLayout.PADDED, pad_token_id=PAD_TOKEN_ID)
    packing = packed_batch.microbatch((0, 1, 2), torch.device("cpu")).packing
    assert packing.cu_seqlens.tolist() == [0, 1, 4, 11]
    assert packing.positions.tolist() == [0, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6]
    assert packed_batch.num_targets == padded_batch.num_targets == len(RAW_SAMPLES)

    with fixed_kv_page_capacity(8):
        padded_worker = create_worker(BatchLayout.PADDED)
    try:
        padded_model_runner = padded_worker.executor
        padded_model = padded_model_runner.model
        padded_adapters = padded_worker.worker.host.adapter
        assert padded_adapters is not None
        padded_named_adapters = padded_adapters.named_parameters()
        assert tuple(name for name, _ in padded_named_adapters) == expected_adapter_names
        padded_runner = padded_worker.training
        torch.manual_seed(ADAPTER_SEED)
        with torch.no_grad():
            for tensor in padded_adapters.state_dict().values():
                tensor.normal_(mean=0.0, std=0.02)
        synchronize_optimizer_parameters(padded_runner, padded_named_adapters)
        padded_selected = padded_runner.trainable_parameters
        with torch.no_grad():
            padded_logits = batched_valid_logits(padded_model, padded_batch)
        padded_base_versions = tuple(tensor._version for tensor in padded_selected.complement())
        padded_initial_live = tuple(
            parameter.detach().clone() for _, parameter in padded_named_adapters
        )
        padded_result = forward_backward(padded_runner, padded_batch)
        padded_gradients = clone_main_gradients(padded_runner, padded_named_adapters)
        padded_grad_norm = optim_step(padded_runner, padded_result.normalization_count)
        for (_, live), before in zip(padded_named_adapters, padded_initial_live, strict=True):
            torch.testing.assert_close(live, before, rtol=0, atol=0)
        padded_runner.publish()
        padded_deltas = _master_deltas(padded_runner, padded_initial_live)
        assert (
            tuple(tensor._version for tensor in padded_selected.complement())
            == padded_base_versions
        )
        assert all(tensor.grad is None for tensor in padded_selected.complement())
    finally:
        padded_worker.close()

    with fixed_kv_page_capacity(8):
        packed_worker = create_worker(BatchLayout.PACKED)
    try:
        packed_model_runner = packed_worker.executor
        packed_model = packed_model_runner.model
        packed_adapters = packed_worker.worker.host.adapter
        assert packed_adapters is not None
        packed_named_adapters = packed_adapters.named_parameters()
        adapter_names = tuple(name for name, _ in packed_named_adapters)
        assert adapter_names == expected_adapter_names
        packed_runner = packed_worker.training
        torch.manual_seed(ADAPTER_SEED)
        with torch.no_grad():
            for tensor in packed_adapters.state_dict().values():
                tensor.normal_(mean=0.0, std=0.02)
        synchronize_optimizer_parameters(packed_runner, packed_named_adapters)
        packed_initial_masters = _masters(packed_runner)
        packed_selected = packed_runner.trainable_parameters
        packed_base_versions = tuple(tensor._version for tensor in packed_selected.complement())
        packed_initial_live = tuple(
            parameter.detach().clone() for _, parameter in packed_named_adapters
        )
        for padded_value, packed_value in zip(
            padded_initial_live,
            packed_initial_live,
            strict=True,
        ):
            torch.testing.assert_close(padded_value, packed_value, rtol=0, atol=0)

        attention_calls = 0
        attention_forward = CausalVarlenAttention.forward

        def counted_attention(
            operation: CausalVarlenAttention, *args: object, **kwargs: object
        ) -> object:
            nonlocal attention_calls
            attention_calls += 1
            return attention_forward(operation, *args, **kwargs)

        num_layers = len(tuple(packed_model.iter_attention_layers()))
        assert num_layers == 28

        def reject_materialized_packed_logits(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("packed loss-only SFT must not materialize logits")

        # Count the public operation without retaining its activation arguments.
        with patch.object(CausalVarlenAttention, "forward", counted_attention):
            with torch.no_grad():
                packed_logits = _packed_logits(packed_model, packed_batch)
            assert attention_calls == num_layers
            logits_error = measure_tensor_error(packed_logits, padded_logits)
            assert logits_error.max_absolute <= PACKED_BF16_TOLERANCE.logit_max_absolute
            assert logits_error.global_relative_l2 <= PACKED_BF16_TOLERANCE.logit_global_relative_l2

            with patch.object(
                packed_model.lm_head,
                "forward_training",
                reject_materialized_packed_logits,
            ):
                packed_result = forward_backward(packed_runner, packed_batch)
            assert attention_calls == (1 + len(RAW_SAMPLES)) * num_layers
        packed_gradients = clone_main_gradients(packed_runner, packed_named_adapters)
        reference_parameters, reference_optimizer, reference_grad_norm = _reference_optimizer_step(
            optimizer_config,
            packed_initial_masters,
            packed_gradients,
            packed_result.normalization_count,
        )
        packed_grad_norm = optim_step(packed_runner, packed_result.normalization_count)
        for (_, live), before in zip(packed_named_adapters, packed_initial_live, strict=True):
            torch.testing.assert_close(live, before, rtol=0, atol=0)
        packed_runner.publish()
        packed_deltas = _master_deltas(packed_runner, packed_initial_live)
        assert math.isclose(
            packed_grad_norm,
            reference_grad_norm,
            rel_tol=FOREACH_ORACLE_RTOL,
            abs_tol=FOREACH_ORACLE_ATOL,
        )

        loss_error = measure_tensor_error(packed_result.loss_sum, padded_result.loss_sum)
        gradient_error = measure_gradient_error(
            adapter_names,
            packed_gradients,
            padded_gradients,
        )
        assert loss_error.max_absolute <= PACKED_BF16_TOLERANCE.loss_max_absolute
        assert loss_error.global_relative_l2 <= PACKED_BF16_TOLERANCE.loss_global_relative_l2
        assert (
            gradient_error.global_relative_l2 <= PACKED_BF16_TOLERANCE.gradient_global_relative_l2
        )
        assert gradient_error.worst_relative_l2 <= PACKED_BF16_TOLERANCE.gradient_worst_relative_l2

        all_padded_delta = torch.cat([delta.flatten() for delta in padded_deltas])
        all_packed_delta = torch.cat([delta.flatten() for delta in packed_deltas])
        master_delta_relative_l2 = _relative_l2(all_packed_delta, all_padded_delta)
        largest_index = int(
            torch.stack([torch.linalg.vector_norm(delta) for delta in padded_deltas]).argmax()
        )
        largest_master_delta_relative_l2 = _relative_l2(
            packed_deltas[largest_index],
            padded_deltas[largest_index],
        )
        flat_padded_gradients = torch.cat([gradient.flatten() for gradient in padded_gradients])
        flat_packed_gradients = torch.cat([gradient.flatten() for gradient in packed_gradients])
        nonzero = (flat_padded_gradients != 0) | (flat_packed_gradients != 0)
        sign_disagreement = float(
            (
                torch.signbit(flat_padded_gradients[nonzero])
                != torch.signbit(flat_packed_gradients[nonzero])
            )
            .float()
            .mean()
        )
        flat_initial = torch.cat([value.float().cpu().flatten() for value in packed_initial_live])
        padded_state = flat_initial + all_padded_delta
        packed_state = flat_initial + all_packed_delta
        print(
            "CTI_PACKED_SFT_DIAGNOSTIC="
            + json.dumps(
                {
                    "gradient_global_relative_l2": gradient_error.global_relative_l2,
                    "gradient_worst_relative_l2": gradient_error.worst_relative_l2,
                    "master_delta_cosine": _cosine(all_packed_delta, all_padded_delta),
                    "master_delta_relative_l2": master_delta_relative_l2,
                    "master_state_relative_l2": _relative_l2(packed_state, padded_state),
                    "packed_grad_norm": packed_grad_norm,
                    "padded_grad_norm": padded_grad_norm,
                    "sign_disagreement": sign_disagreement,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        stepped = packed_runner.checkpoint(num_optimizer_steps=1, model_revision=0).optimizer
        assert len(stepped.master) == len(reference_parameters) == len(packed_named_adapters)
        for index, ((_, live), actual, expected) in enumerate(
            zip(packed_named_adapters, stepped.master, reference_parameters, strict=True)
        ):
            torch.testing.assert_close(
                actual,
                expected.detach().cpu(),
                rtol=FOREACH_ORACLE_RTOL,
                atol=FOREACH_ORACLE_ATOL,
            )
            torch.testing.assert_close(live.cpu(), actual.to(torch.bfloat16), rtol=0, atol=0)
            actual_state = stepped.adamw["state"][index]
            expected_state = reference_optimizer.state[expected]
            assert actual_state.keys() == expected_state.keys()
            torch.testing.assert_close(
                actual_state["step"],
                expected_state["step"].cpu(),
                rtol=0,
                atol=0,
            )
            for name in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    actual_state[name],
                    expected_state[name].cpu(),
                    rtol=FOREACH_ORACLE_RTOL,
                    atol=FOREACH_ORACLE_ATOL,
                )
        assert (
            tuple(tensor._version for tensor in packed_selected.complement())
            == packed_base_versions
        )
        assert all(tensor.grad is None for tensor in packed_selected.complement())

        torch.cuda.synchronize()
        receipt = {
            "adapter_seed": ADAPTER_SEED,
            "capability": list(torch.cuda.get_device_capability()),
            "device": torch.cuda.get_device_name(),
            "gradient_error": {
                "above_five_percent": gradient_error.above_five_percent,
                "above_ten_percent": gradient_error.above_ten_percent,
                "global_relative_l2": gradient_error.global_relative_l2,
                "max_absolute": gradient_error.max_absolute,
                "worst_name": gradient_error.worst_name,
                "worst_relative_l2": gradient_error.worst_relative_l2,
            },
            "largest_master_delta_name": adapter_names[largest_index],
            "largest_master_delta_relative_l2": largest_master_delta_relative_l2,
            "logits_error": logits_error._asdict(),
            "loss_error": loss_error._asdict(),
            "master_delta_relative_l2": master_delta_relative_l2,
            "foreach_loop_optimizer_oracle": {
                "matches": True,
                "rtol": FOREACH_ORACLE_RTOL,
                "atol": FOREACH_ORACLE_ATOL,
            },
            "model": QWEN3_0_6B.repo_id,
            "num_layers": num_layers,
            "num_attention_calls": attention_calls,
            "num_target_tokens": packed_result.normalization_count,
            "revision": QWEN3_0_6B.revision,
            "source_root": str(Path(cti.__file__).resolve().parents[2]),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
        print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))
    finally:
        packed_worker.close()


def _run_public_child() -> None:
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
        SFTShapeProfile,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    assert not torch.cuda.is_initialized()
    case = GOLDEN_CASES[0]
    batch = packed_sft_batch()
    token_snapshot = batch.tokens.token_ids.clone()
    boundary_snapshot = batch.tokens.cu_seqlens.clone()
    llm = LLM(
        EngineOptions(
            model=ModelOptions(
                path=qwen3_model_snapshot(),
                initialization_seed=PUBLIC_INITIALIZATION_SEED,
                max_seq_len=512,
            ),
            memory=MemoryOptions(peak_ratio=0.85, inference_ratio=0.6),
            scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=128),
            cache=CacheOptions(prefix="naive"),
            lora=LoRAConfig(
                targets=("attention", "mlp"),
                rank=4,
                alpha=8.0,
            ),
            training=TrainingConfig(
                trainable_plan=TrainablePlan(
                    (
                        "model.layers.0.self_attn.qkv_proj.q.lora_A",
                        "model.layers.0.self_attn.qkv_proj.q.lora_B",
                    )
                ),
                optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
                batching=TrainingBatching(
                    layout=BatchLayout.PACKED,
                    policy=StaticBatching(
                        microbatch_size=batch.batch_size,
                        max_tokens_per_sample=max(torch.diff(batch.tokens.cu_seqlens).tolist()),
                    ),
                ),
                sft_profile=SFTShapeProfile(tuple(torch.diff(batch.tokens.cu_seqlens).tolist())),
            ),
        ),
        tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)),
    )
    rank_pids = engine_rank_pids()
    assert len(rank_pids) == 1 and rank_pids[0] != os.getpid()
    try:
        [before] = llm.generate(
            [list(case.prompt_ids)],
            SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=8),
        )
        with llm.training_session() as session:
            completion_futures = tuple(session.forward_backward(batch) for _ in range(3))
            step = session.optim_step().result()
            session.publish()
            completions = tuple(future.result() for future in completion_futures)
            assert all(
                math.isfinite(completion.loss_sum) and completion.loss_sum > 0
                for completion in completions
            )
            assert all(
                completion.normalization_count == batch.num_targets for completion in completions
            )
            assert step.step == 1
            assert step.normalization_count == 3 * batch.num_targets
            assert math.isfinite(step.grad_norm) and step.grad_norm >= 0
            assert llm.model_revision == 1
        completion = completions[0]
        with llm.training_session() as session:
            discarded = session.forward_backward(batch).result()
            assert math.isfinite(discarded.loss_sum) and discarded.loss_sum > 0
            session.discard_accumulation()
            assert llm.model_revision == 1
        [after] = llm.generate(
            [list(case.prompt_ids)],
            SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=8),
        )
        assert torch.equal(batch.tokens.token_ids, token_snapshot)
        assert torch.equal(batch.tokens.cu_seqlens, boundary_snapshot)
        assert not torch.cuda.is_initialized()
        print(
            PUBLIC_RECEIPT_PREFIX
            + json.dumps(
                {
                    "after": after,
                    "before": before,
                    "child_pid": rank_pids[0],
                    "loss_sum": completion.loss_sum,
                    "model": QWEN3_0_6B.repo_id,
                    "model_revision": llm.model_revision,
                    "num_target_tokens": completion.normalization_count,
                    "revision": QWEN3_0_6B.revision,
                    "source_root": str(Path(cti.__file__).resolve().parents[2]),
                    "step": step.step,
                },
                sort_keys=True,
            )
        )
    finally:
        llm.shutdown()
        llm.shutdown()
    assert_ranks_reclaimed(rank_pids)


def test_packed_sft_matches_padded_qwen3(tmp_path: Path) -> None:
    env, log_root = _isolated_workspace_env(tmp_path, "packed-sft")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.training.test_packed_sft"),
        log_root / "packed-sft.log",
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
    assert receipt["num_layers"] == 28
    assert receipt["num_attention_calls"] == 28 * (1 + len(RAW_SAMPLES))
    assert receipt["num_target_tokens"] == len(RAW_SAMPLES)
    assert receipt["foreach_loop_optimizer_oracle"] == {
        "matches": True,
        "rtol": FOREACH_ORACLE_RTOL,
        "atol": FOREACH_ORACLE_ATOL,
    }
    assert receipt["model"] == QWEN3_0_6B.repo_id
    assert receipt["revision"] == QWEN3_0_6B.revision
    assert Path(receipt["source_root"]) == REPO_ROOT


def test_public_training_session_accepts_packed_sft(tmp_path: Path) -> None:
    env, log_root = _isolated_workspace_env(tmp_path, "packed-sft-public")
    prefix = Path(sys.prefix)
    toolkit = NativeBuildLayout.discover(prefix).toolkit
    cuda_home = tmp_path / "cuda-home"
    cuda_home.mkdir()
    (cuda_home / "bin").mkdir()
    for name in ("nvcc", "nvcc.profile"):
        (cuda_home / "bin" / name).symlink_to(prefix / "bin" / name)
    for name, target in (
        ("include", toolkit / "include"),
        ("lib64", toolkit / "lib"),
        ("nvvm", prefix / "nvvm"),
        ("targets", prefix / "targets"),
    ):
        (cuda_home / name).symlink_to(target, target_is_directory=True)

    env["CUDA_HOME"] = str(cuda_home)
    # A toolchain path the run's shared JIT caches were not built with makes ninja rebuild
    # them for every later test; this child compiles into caches of its own.
    env.update(jit_cache_env(tmp_path / "cache" / "own-toolchain"))
    env["PATH"] = os.pathsep.join((str(cuda_home / "bin"), str(prefix / "bin"), env["PATH"]))
    multiarch = sysconfig.get_config_var("MULTIARCH")
    system_library = Path("/usr/lib") / multiarch if isinstance(multiarch, str) else None
    library_paths = tuple(
        path
        for path in (cuda_home / "lib64", prefix / "lib", system_library)
        if path is not None and path.is_dir()
    )
    for variable in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        paths = os.pathsep.join(str(path) for path in library_paths)
        if env.get(variable):
            paths += os.pathsep + env[variable]
        env[variable] = paths
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.training.test_packed_sft", "--mode", "public"),
        log_root / "packed-sft-public.log",
        env=env,
        cwd=REPO_ROOT,
        # Its own JIT caches start cold: FlashInfer compiles in the child.
        timeout=600,
    )
    receipt_lines = [
        line.removeprefix(PUBLIC_RECEIPT_PREFIX)
        for line in output.splitlines()
        if line.startswith(PUBLIC_RECEIPT_PREFIX)
    ]
    assert len(receipt_lines) == 1
    receipt = json.loads(receipt_lines[0])
    assert receipt["child_pid"] > 0
    assert output.count("Selected attention backend fi: explicit attention backend") == 1
    assert graph_captures(output, "decode", [1]) == 1
    assert receipt["model"] == QWEN3_0_6B.repo_id
    assert receipt["model_revision"] == 1
    assert receipt["num_target_tokens"] == 11
    assert receipt["revision"] == QWEN3_0_6B.revision
    assert receipt["step"] == 1
    assert receipt["before"]["token_ids"] == list(GOLDEN_CASES[0].output_ids)
    assert receipt["after"]["token_ids"] == list(GOLDEN_CASES[0].output_ids)
    assert Path(receipt["source_root"]) == REPO_ROOT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("model", "public"), default="model")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "public":
        _run_public_child()
    else:
        _run_child()
