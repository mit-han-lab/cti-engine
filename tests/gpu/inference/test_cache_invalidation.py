"""A training round trip empties the radix prefix cache and keeps the decode graphs: after
each one, a repeated prompt first misses the cache, then hits it, and every output stays the
golden one."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import isolated_runtime_env, run_isolated_process
from tests.support.graph_captures import graph_captures
from tests.support.inference_correctness import GOLDEN_CASES
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

_ROUND_TRIPS = 2


def _run_child() -> None:
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

    case = GOLDEN_CASES[0]
    snapshot = str(QWEN3_0_6B.snapshot())
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=snapshot, max_seq_len=512),
            memory=MemoryOptions(inference_ratio=0.5),
            scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=128),
            cache=CacheOptions(prefix="radix"),
            lora=LoRAConfig(targets=("attention",), rank=4, alpha=8.0),
            training=TrainingConfig(
                trainable_plan=TrainablePlan(("model.layers.0.self_attn.qkv_proj.q.lora_A",)),
                optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
                batching=TrainingBatching(
                    layout=BatchLayout.PADDED,
                    policy=StaticBatching(microbatch_size=1, max_tokens_per_sample=512),
                ),
            ),
        ),
        tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)),
    )
    params = SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=len(case.output_ids))

    def cached_prompt_tokens() -> int:
        [sample] = llm.sample([list(case.prompt_ids)], params).samples
        assert sample.sampled_token_ids == case.output_ids
        return sample.cached_prompt_tokens

    try:
        for _ in range(_ROUND_TRIPS):
            assert [cached_prompt_tokens(), cached_prompt_tokens()] == [0, len(case.prompt_ids) - 1]
            # Entering and leaving training (no step) invalidates the cache.
            with llm.training_session():
                pass
        assert cached_prompt_tokens() == 0
    finally:
        llm.shutdown()


def test_training_round_trips_invalidate_the_prefix_cache_and_keep_graphs(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "cache-invalidation")
    env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_cache_invalidation"),
        log_root / "cache-invalidation.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert output.count("Selected attention backend fi: explicit attention backend") == 1
    # Captured once: the round trips keep the graphs.
    assert graph_captures(output, "decode", [1]) == 1


if __name__ == "__main__":
    _run_child()
