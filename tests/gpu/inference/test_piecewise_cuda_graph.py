"""Breakable prefill CUDA graphs change no output: the same greedy requests, served once with
the graphs and once eagerly, sample the same tokens and logprobs. A request's tokens do not
change when others join its batch."""

from __future__ import annotations

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.graph_captures import graph_captures
from tests.support.inference_correctness import DifferentialCase, build_prompt

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

# High-margin prompt prefixes: request i of a batch starts with the i-th.
PREFIX_TOKENS = (7772, 11580, 304, 1039)
MAX_TOKENS = 4
# Profile -> (token buckets, max_batch_tokens, batches of prompt lengths, logprob tolerance).
PROFILES = {
    # Every bucket filled exactly at batch sizes 1, 2 and 4, padded batches, and a mixed batch:
    # a 4096-row chunk, then the last 160 rows with the second prompt's 800. Bitwise: padding
    # to the bucket kept every logprob equal on RTX 5090.
    "buckets": (
        (1024, 2048, 4096),
        4096,
        (
            *((total // size,) * size for total in (1024, 2048, 4096) for size in (1, 2, 4)),
            (1000,),
            (1000, 1000),
            (1000,) * 4,
            (4256, 800),
        ),
        0.0,
    ),
    # 256-row chunks; each prompt's last chunk pads to 256 rows, whose GEMMs differ from the
    # eager ones: up to 0.022 nats on RTX 5090.
    "chunked": ((256,), 256, ((4000,), (4256, 800)), 0.05),
}


def _serve(
    snapshot: str, buckets: tuple[int, ...], max_batch_tokens: int, batches
) -> dict[tuple[int, ...], list[tuple[tuple[int, ...], tuple[float, ...]]]]:
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

    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=snapshot, max_seq_len=8192),
            memory=MemoryOptions(inference_ratio=0.5, peak_ratio=0.5),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=max_batch_tokens),
            cache=CacheOptions(prefix="naive"),
        ),
        tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(), graph_row_buckets=buckets),
    )
    params = SamplingParams(
        temperature=0, ignore_eos=True, max_new_tokens=MAX_TOKENS, logprobs=True
    )
    try:
        return {
            lengths: [
                (sample.sampled_token_ids, sample.sampling_logprobs)
                for sample in llm.sample(
                    [
                        build_prompt(
                            DifferentialCase(f"p{index}", length, MAX_TOKENS, PREFIX_TOKENS[index])
                        )
                        for index, length in enumerate(lengths)
                    ],
                    params,
                ).samples
            ]
            for lengths in batches
        }
    finally:
        llm.shutdown()


@pytest.mark.parametrize("profile", PROFILES)
def test_piecewise_graphs_match_eager(profile: str, capfd: pytest.CaptureFixture[str]) -> None:
    snapshot = str(QWEN3_0_6B.snapshot())
    buckets, max_batch_tokens, batches, tolerance = PROFILES[profile]
    eager = _serve(snapshot, (), max_batch_tokens, batches)
    eager_log = capfd.readouterr()
    graphs = _serve(snapshot, buckets, max_batch_tokens, batches)
    graph_log = capfd.readouterr()

    # The graph runtime captured every bucket; the eager one none.
    assert graph_captures(graph_log.err + graph_log.out, "breakable", buckets) == 1
    assert not graph_captures(eager_log.err + eager_log.out, "breakable")
    for lengths in batches:
        for (tokens, logprobs), (eager_tokens, eager_logprobs) in zip(
            graphs[lengths], eager[lengths], strict=True
        ):
            deltas = [abs(a - b) for a, b in zip(logprobs, eager_logprobs, strict=True)]
            assert tokens == eager_tokens and max(deltas) <= tolerance, (lengths, deltas)
    # Requests joining a batch do not leak into its first request's padded rows.
    if profile == "buckets":
        alone = graphs[(1000,)][0][0]
        assert graphs[(1000, 1000)][0][0] == graphs[(1000,) * 4][0][0] == alone
