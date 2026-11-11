from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_0_6B, QWEN3_8B, QWEN3_8B_DFLASH_B16
from tests.support.dflash import PINNED_CHECKPOINTS
from tests.support.gpu import (
    assert_ranks_reclaimed,
    engine_rank_pids,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.inference_correctness import GOLDEN_CASES
from tests.support.paths import REPO_ROOT, source_pythonpath
from tests.support.training import qwen3_model_snapshot

RECEIPT_PREFIX = "CTI_GRPO_CYCLE="
_GROUP_SIZE = 2
_COMPLETION_TOKENS = 4
_TRAINED_PARAMETER_NAMES = (
    "model.layers.0.self_attn.qkv_proj.q.lora_A",
    "model.layers.0.self_attn.qkv_proj.q.lora_B",
)


def _snapshot(checkpoint) -> str:
    return str(checkpoint.snapshot())


def _run_child(layout: str, dflash_target: str | None) -> None:
    """``dflash_target`` names a pinned DFlash pair whose target and draft serve the rollouts
    of the training engine (RL rollout with DFlash); otherwise Qwen3-0.6B serves alone."""

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
        SpeculativeOptions,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams
    from cti.training import TrainingBatch

    if layout not in ("padded", "packed"):
        raise ValueError("layout must be padded or packed")
    assert not torch.cuda.is_initialized()
    prompt = list(GOLDEN_CASES[0].prompt_ids)
    prompts = [prompt.copy() for _ in range(_GROUP_SIZE)]
    policies = [
        SamplingParams(
            temperature=1.0,
            ignore_eos=True,
            max_new_tokens=_COMPLETION_TOKENS,
            seed=20260808 + index,
            logprobs=True,
            logprobs_processed=True,
        )
        for index in range(_GROUP_SIZE)
    ]
    model, draft = qwen3_model_snapshot(), None
    if dflash_target is not None:
        pair = next(pair for pair in PINNED_CHECKPOINTS if pair.target.repo_id == dflash_target)
        model, draft = _snapshot(pair.target), _snapshot(pair.draft)
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=model, max_seq_len=128),
            # Qwen3-8B and its draft hold about 18 GB of weights.
            memory=MemoryOptions(
                peak_ratio=0.85 if draft is None else 0.92,
                inference_ratio=0.5 if draft is None else 0.75,
            ),
            scheduling=SchedulingOptions(max_running_requests=_GROUP_SIZE, max_batch_tokens=128),
            cache=CacheOptions(prefix="radix"),
            speculative=None if draft is None else SpeculativeOptions(model_path=draft),
            lora=LoRAConfig(targets=("attention", "mlp"), rank=4, alpha=8.0),
            training=TrainingConfig(
                trainable_plan=TrainablePlan(_TRAINED_PARAMETER_NAMES),
                optimizer=AdamWConfig(lr=1e-3, weight_decay=0.0, max_grad_norm=1.0),
                batching=TrainingBatching(
                    layout=BatchLayout(layout),
                    policy=StaticBatching(
                        microbatch_size=_GROUP_SIZE,
                        max_tokens_per_sample=128,
                    ),
                ),
            ),
        ),
        tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(_GROUP_SIZE,)),
    )
    rank_pids = engine_rank_pids()
    assert len(rank_pids) == 1
    try:
        rollout = llm.sample(prompts, policies)
        assert rollout.model_revision == 0
        assert len(rollout.samples) == _GROUP_SIZE
        assert all(
            len(sample.sampled_token_ids) == _COMPLETION_TOKENS
            and sample.sampling_logprobs is not None
            and len(sample.sampling_logprobs) == _COMPLETION_TOKENS
            and all(math.isfinite(value) for value in sample.sampling_logprobs)
            for sample in rollout.samples
        )
        batch = (
            TrainingBatch.grpo(rollout, (0.0, 1.0), BatchLayout.PACKED, group_size=_GROUP_SIZE)
            if layout == "packed"
            else TrainingBatch.grpo(
                rollout, (0.0, 1.0), BatchLayout.PADDED, group_size=_GROUP_SIZE, pad_token_id=151643
            )
        )
        with llm.training_session() as session:
            completion_future = session.forward_backward(batch)
            step = session.optim_step().result()
            assert llm.model_revision == 0
            session.publish()
            completion = completion_future.result()
        assert step.step == llm.model_revision == 1
        assert step.normalization_count == _GROUP_SIZE
        assert step.grad_norm > 0.0
        assert completion.normalization_count == _GROUP_SIZE
        assert completion.metrics["clip_fraction"] == 0.0
        assert completion.metrics["max_abs_log_ratio"] == 0.0
        assert math.isfinite(completion.metrics["max_abs_sampling_logprob_delta"])

        updated = llm.sample(prompts, policies)
        assert updated.model_revision == 1
        assert all(sample.cached_prompt_tokens >= 0 for sample in updated.samples)
        print(
            RECEIPT_PREFIX
            + json.dumps(
                {
                    "child_pid": rank_pids[0],
                    "grad_norm": step.grad_norm,
                    "layout": layout,
                    "max_abs_sampling_logprob_delta": completion.metrics[
                        "max_abs_sampling_logprob_delta"
                    ],
                    "model": dflash_target or QWEN3_0_6B.repo_id,
                    "model_revision": updated.model_revision,
                    "num_response_tokens": batch.num_targets,
                    "revision": QWEN3_0_6B.revision,
                    "source_root": str(Path(cti.__file__).resolve().parents[2]),
                },
                sort_keys=True,
            )
        )
    finally:
        llm.shutdown()
        llm.shutdown()
    assert_ranks_reclaimed(rank_pids)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    ("layout", "dflash_target"),
    (
        pytest.param("padded", None, marks=pytest.mark.checkpoints(QWEN3_0_6B), id="padded"),
        pytest.param("packed", None, marks=pytest.mark.checkpoints(QWEN3_0_6B), id="packed"),
        pytest.param(
            "packed",
            "Qwen/Qwen3-8B",
            marks=pytest.mark.checkpoints(QWEN3_0_6B, QWEN3_8B, QWEN3_8B_DFLASH_B16),
            id="packed-dflash",
        ),
    ),
)
def test_qwen3_reference_free_grpo_cycle(
    layout: str, dflash_target: str | None, tmp_path: Path
) -> None:
    env, log_root = isolated_runtime_env(tmp_path, f"grpo-{layout}")
    env["PYTHONPATH"] = source_pythonpath(REPO_ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.integration.test_grpo_cycle",
            layout,
            *(() if dflash_target is None else (dflash_target,)),
        ),
        log_root / f"grpo-{layout}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=600,
    )
    receipts = [
        json.loads(line.removeprefix(RECEIPT_PREFIX))
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["layout"] == layout
    assert receipt["model_revision"] == 1
    assert receipt["num_response_tokens"] == _GROUP_SIZE * _COMPLETION_TOKENS
    assert Path(receipt["source_root"]).resolve() == REPO_ROOT.resolve()


if __name__ == "__main__":
    _run_child(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
