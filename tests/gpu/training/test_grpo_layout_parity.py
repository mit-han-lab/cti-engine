from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import fixed_kv_page_capacity, isolated_runtime_env, run_isolated_process
from tests.support.paths import REPO_ROOT, source_pythonpath
from tests.support.training import (
    PACKED_BF16_TOLERANCE,
    clone_main_gradients,
    expected_qwen_adapter_names,
    forward_backward,
    measure_gradient_error,
    qwen3_model_snapshot,
    synchronize_optimizer_parameters,
)

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

RECEIPT_PREFIX = "CTI_GRPO_LAYOUT_PARITY="
_SEED = 20260808


def _sampling_result() -> object:
    from cti.protocol import SampledSequence, SamplingParams, SamplingResult

    prompt = (785, 14002)
    responses = ((20, 21), (22,), (23, 24, 25), (26,))
    samples = tuple(
        SampledSequence(
            request_id=index,
            prompt_token_ids=prompt,
            sampled_token_ids=response,
            sampling_logprobs=tuple(-1.0 - 0.1 * offset for offset in range(len(response))),
            cached_prompt_tokens=0,
            sampling=SamplingParams(
                temperature=1.0,
                max_new_tokens=len(response),
                seed=_SEED + index,
                logprobs=True,
                logprobs_processed=True,
            ).snapshot(),
            finish_reason="length",
        )
        for index, response in enumerate(responses)
    )
    return SamplingResult(samples=samples, texts=("",) * len(samples), model_revision=0)


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
    from cti.training import TrainingBatch
    from tests.support.gpu import start_rank

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    sampling = _sampling_result()
    rewards = (0.0, 1.0, 2.0, 3.0)
    padded = TrainingBatch.grpo(
        sampling, rewards, BatchLayout.PADDED, group_size=4, pad_token_id=151643
    )
    packed = TrainingBatch.grpo(sampling, rewards, BatchLayout.PACKED, group_size=4)
    selected_names = expected_qwen_adapter_names()

    def run_layout(
        layout: BatchLayout,
    ) -> tuple[tuple[float, int, float, float], tuple[torch.Tensor, ...]]:
        with fixed_kv_page_capacity(8):
            worker = start_rank(
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
                        TrainablePlan(selected_names),
                        AdamWConfig(weight_decay=0.0, max_grad_norm=None),
                        TrainingBatching(
                            layout,
                            StaticBatching(microbatch_size=2, max_tokens_per_sample=64),
                        ),
                    ),
                ),
                EngineTuning(attention_backend="fi", cuda_graph_bs=()),
                checks=True,
            )
        try:
            runner = worker.training
            adapters = worker.worker.host.adapter
            assert adapters is not None
            torch.manual_seed(_SEED)
            with torch.no_grad():
                for value in adapters.state_dict().values():
                    value.normal_(mean=0.0, std=0.02)
            synchronize_optimizer_parameters(runner, adapters.named_parameters())
            batch = padded if layout is BatchLayout.PADDED else packed
            result = forward_backward(runner, batch)
            metrics = {name: float(value) for name, _, value in result.values}
            evidence = (
                metrics["loss_sum"],
                result.normalization_count,
                metrics["clip_fraction"],
                metrics["max_abs_log_ratio"],
            )
            gradients = tuple(
                gradient.cpu()
                for gradient in clone_main_gradients(runner, adapters.named_parameters())
            )
            runner.discard()
            return evidence, gradients
        finally:
            worker.close()

    padded_evidence, padded_gradients = run_layout(BatchLayout.PADDED)
    packed_evidence, packed_gradients = run_layout(BatchLayout.PACKED)
    gradient_error = measure_gradient_error(
        selected_names,
        packed_gradients,
        padded_gradients,
    )
    padded_loss, padded_sequences, padded_clip, padded_ratio = padded_evidence
    packed_loss, packed_sequences, packed_clip, packed_ratio = packed_evidence
    assert padded_sequences == packed_sequences == 4
    assert padded_clip == packed_clip == 0.0
    assert padded_ratio == packed_ratio == 0.0
    assert abs(packed_loss - padded_loss) <= 1e-6
    assert gradient_error.global_relative_l2 <= PACKED_BF16_TOLERANCE.gradient_global_relative_l2
    assert gradient_error.worst_relative_l2 <= PACKED_BF16_TOLERANCE.gradient_worst_relative_l2
    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "gradient_global_relative_l2": gradient_error.global_relative_l2,
                "gradient_worst_name": gradient_error.worst_name,
                "gradient_worst_relative_l2": gradient_error.worst_relative_l2,
                "model": QWEN3_0_6B.repo_id,
                "num_response_tokens": packed.num_targets,
                "revision": QWEN3_0_6B.revision,
                "source_root": str(Path(cti.__file__).resolve().parents[2]),
            },
            sort_keys=True,
        )
    )


def test_qwen3_grpo_padded_and_packed_layouts_match(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "grpo-layout-parity")
    env["PYTHONPATH"] = source_pythonpath(REPO_ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.training.test_grpo_layout_parity"),
        log_root / "grpo-layout-parity.log",
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
    assert receipts[0]["num_response_tokens"] == 7
    assert Path(receipts[0]["source_root"]).resolve() == REPO_ROOT.resolve()


if __name__ == "__main__":
    _run_child()
