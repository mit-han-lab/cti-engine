"""Equal prompts share the radix cache correctly. Two in one graph batch, the second reading
no prefix (it asks for prompt logprobs), prefill together: both give the golden output and
the duplicate's pages are released once the cache holds the first's. A plain equal pair
waits for its first prompt's forward and restores its prefix; a repeat of a pair hits the
cache, and a later full batch is served."""

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


def _run_child() -> None:
    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    snapshot = str(QWEN3_0_6B.snapshot())
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=snapshot, max_seq_len=512),
            memory=MemoryOptions(inference_ratio=0.5),
            scheduling=SchedulingOptions(max_running_requests=2, max_batch_tokens=128),
            cache=CacheOptions(prefix="radix"),
        ),
        tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(2,)),
    )

    def cached_prompt_tokens(cases, prompt_logprobs=()) -> list[int]:
        params = [
            SamplingParams(
                temperature=0,
                ignore_eos=True,
                max_new_tokens=len(case.output_ids),
                prompt_logprobs=1 if index in prompt_logprobs else None,
            )
            for index, case in enumerate(cases)
        ]
        samples = llm.sample([list(case.prompt_ids) for case in cases], params).samples
        assert [sample.sampled_token_ids for sample in samples] == [c.output_ids for c in cases]
        return [sample.cached_prompt_tokens for sample in samples]

    case, sibling = GOLDEN_CASES[:2]
    try:
        # A request asking for prompt logprobs reads no prefix and is never held back, so
        # both prefill in one forward and the second publishes duplicate pages. Should such
        # requests read the cache one day, this fails rather than lose the duplicate path.
        assert cached_prompt_tokens([case, case], prompt_logprobs=(1,)) == [0, 0]
        # An equal prompt waits for the first one's forward to settle and publish its KV,
        # then restores all of it but the final token; it never reads KV still being written.
        # Both golden prompts open with the same token, which the first pair cached.
        shared = next(
            i
            for i, (a, b) in enumerate(zip(case.prompt_ids, sibling.prompt_ids, strict=False))
            if a != b
        )
        siblings = cached_prompt_tokens([sibling, sibling])
        assert siblings == [shared, len(sibling.prompt_ids) - 1], siblings
        assert cached_prompt_tokens([case, case]) == [len(case.prompt_ids) - 1] * 2
        cached_prompt_tokens(GOLDEN_CASES)
    finally:
        llm.shutdown()


def test_equal_prompts_in_one_graph_batch_share_the_prefix_cache(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "cache-reservation")
    env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_cache_reservation"),
        log_root / "cache-reservation.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert output.count("Selected attention backend fi: explicit attention backend") == 1
    assert graph_captures(output, "decode", [2]) == 1
    # Every request finished and no page stays in use: the duplicate pages were released.
    finished = [line for line in output.splitlines() if "requests finished" in line]
    assert finished and " KV pages 0 used," in finished[-1], finished[-1:]


if __name__ == "__main__":
    _run_child()
