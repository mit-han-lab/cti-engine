"""Qwen3.5 decode CUDA graphs against eager decoding on a pinned checkpoint, through the
public ``LLM``, while the decode batch grows and shrinks beside chunked prefills.

A graph for every batch size must give bitwise eager's tokens and sampled-token logprobs
(capture defects: stale buffers, wrong addressing). Padded graphs run kernels at another batch
size, which may round differently (the Gated DeltaNet decode picks its launch per batch); they
are held to a logprob tolerance instead, which a padding row bleeding into a request exceeds."""

from __future__ import annotations

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.graph_captures import capture_line
from tests.support.qwen3_5_gpu import max_running_requests, selected_checkpoint

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

PADDED_GRAPH_SIZES = (1, 2, 4, 8)
EXACT_GRAPH_SIZES = tuple(range(1, 9))
# Padded vs eager on an RTX 5090: all 202 tokens agree, logprobs differ by at most 0.128 nats.
PADDED_MAX_LOGPROB_DIFFERENCE = 0.5
PADDED_MIN_AGREEMENT = 0.9
# (prompt tokens, generated tokens) per request of one sample call. Wave 1 shrinks 6 -> 1
# through padded sizes (5 -> 8, 3 -> 4); wave 2 decodes beside chunked long prefills.
WAVES = (
    ((61, 40), (47, 8), (120, 32), (33, 16), (88, 24), (75, 4)),
    ((16385, 12), (50, 30), (8193, 6), (70, 20), (30, 10)),
)
MAX_BATCH_TOKENS = 8192
MAX_SEQ_LEN = 17408


def _wave_prompts(frozen: list[list[int]]) -> list[list[list[int]]]:
    """Each request repeats one frozen prompt to its length; long ones the longest, each
    rotated by its index so that no long prompt reuses another's prefix."""

    longest = max(frozen, key=len)
    waves, index = [], 0
    for requests in WAVES:
        prompts = []
        for length, _ in requests:
            rotated = longest[index:] + longest[:index]
            base = rotated if length > len(longest) else frozen[index % len(frozen)]
            prompts.append((base * (length // len(base) + 1))[:length])
            index += 1
        waves.append(prompts)
    return waves


def _serve(snapshot, waves, cuda_graph_bs, capfd) -> list[list[tuple]]:
    from cti.config import (
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
            model=ModelOptions(path=str(snapshot), max_seq_len=MAX_SEQ_LEN),
            memory=MemoryOptions(inference_ratio=0.85),
            scheduling=SchedulingOptions(
                max_running_requests=max_running_requests(), max_batch_tokens=MAX_BATCH_TOKENS
            ),
        ),
        tuning=EngineTuning(cuda_graph_bs=tuple(cuda_graph_bs)),
    )
    try:
        outputs = []
        for prompts, requests in zip(waves, WAVES, strict=True):
            params = [
                SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=tokens, logprobs=True)
                for _, tokens in requests
            ]
            samples = llm.sample(prompts, params).samples
            outputs.append([(s.sampled_token_ids, s.sampling_logprobs) for s in samples])
    finally:
        llm.shutdown()
    expected = capture_line("decode", cuda_graph_bs) if cuda_graph_bs else "CUDA graph is disabled."
    logs = capfd.readouterr()
    assert expected in logs.out + logs.err
    return outputs


def test_qwen3_5_decode_graphs_match_eager_decoding(capfd: pytest.CaptureFixture[str]) -> None:
    from transformers import AutoTokenizer

    from tests.support.qwen3_5_prompts import text_prompts

    checkpoint = selected_checkpoint()
    snapshot = checkpoint.snapshot()
    frozen = text_prompts(AutoTokenizer.from_pretrained(snapshot), checkpoint.key)
    waves = _wave_prompts([ids for _, ids in frozen])
    eager = _serve(snapshot, waves, (), capfd)
    assert [[len(tokens) for tokens, _ in wave] for wave in eager] == [
        [tokens for _, tokens in requests] for requests in WAVES
    ]
    assert _serve(snapshot, waves, EXACT_GRAPH_SIZES, capfd) == eager

    padded = _serve(snapshot, waves, PADDED_GRAPH_SIZES, capfd)
    agreed, differences = 0, [0.0]
    for padded_wave, eager_wave in zip(padded, eager, strict=True):
        for (tokens, logprobs), (eager_tokens, eager_logprobs) in zip(
            padded_wave, eager_wave, strict=True
        ):
            pairs = list(zip(tokens, eager_tokens, strict=True))
            shared = next((i for i, (a, b) in enumerate(pairs) if a != b), len(pairs))
            agreed += shared
            scores = zip(logprobs[:shared], eager_logprobs[:shared], strict=True)
            differences += [abs(a - b) for a, b in scores]
    total = sum(tokens for requests in WAVES for _, tokens in requests)
    assert agreed >= PADDED_MIN_AGREEMENT * total, (agreed, total)
    assert max(differences) <= PADDED_MAX_LOGPROB_DIFFERENCE, max(differences)
