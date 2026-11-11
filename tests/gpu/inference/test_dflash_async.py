"""Rounds in flight are invisible in DFlash's outputs on the pinned v1 pairs.

Each batch of one to four requests, greedy and seeded sampled, runs twice and must emit the
same tokens bit for bit, including where a request ends while its successor is in flight.
Requests with identical prompts and policies stay in lockstep, so every row of such a batch
must emit the same tokens.
"""

from __future__ import annotations

import pytest
import torch

from tests.support.dflash import PINNED_CHECKPOINTS, preflight
from tests.support.qwen3_5_gpu import write_receipt

_V1_PAIRS = [pair for pair in PINNED_CHECKPOINTS if pair.architecture == "DFlashDraftModel"]
_PROMPTS = (
    "Write a detailed explanation of how a hash table works.",
    "Explain the causes of the French Revolution.",
    "Write a Python function that parses a CSV file.",
    "Describe photosynthesis for a high school student.",
)
_TOKENS = 96
_POLICIES = {
    "greedy": {"temperature": 0.0},
    "seeded": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
}


@pytest.mark.parametrize("pair", _V1_PAIRS, ids=lambda pair: pair.target.repo_id)
def test_rounds_in_flight_repeat_and_keep_identical_rows_in_lockstep(pair) -> None:
    from transformers import AutoTokenizer

    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        SpeculativeOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    # Missing pinned checkpoints raise QualificationBlocked: they fail the run, never skip.
    snapshots = preflight(pair)["snapshots"]
    tokenizer = AutoTokenizer.from_pretrained(snapshots["target"], local_files_only=True)
    prompts = [
        tokenizer(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )["input_ids"]
        for text in _PROMPTS
    ]
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=snapshots["target"], max_seq_len=1024),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=1024),
            cache=CacheOptions(prefix="naive"),
            speculative=SpeculativeOptions(model_path=snapshots["draft"]),
        ),
        tuning=EngineTuning(cuda_graph_bs=(1, 2, 3, 4)),
    )

    def run(batch, policy, same):
        params = [
            SamplingParams(max_new_tokens=_TOKENS, seed=7 if same else 100 + i, **_POLICIES[policy])
            for i in range(len(batch))
        ]
        samples = sorted(llm.sample(batch, params).samples, key=lambda sample: sample.request_id)
        return [list(sample.sampled_token_ids) for sample in samples]

    repeated, identical = {}, {}
    try:
        for policy in _POLICIES:
            for size in (1, 2, 3, 4):
                name = f"{policy}/B{size}"
                repeated[name] = [run(prompts[:size], policy, False) for _ in range(2)]
                if size > 1:
                    identical[name] = run([prompts[0]] * size, policy, True)
    finally:
        llm.shutdown()
    write_receipt(
        f"dflash-async-{pair.target.repo_id.split('/')[-1]}",
        {"repeated": repeated, "identical": identical},
    )

    for name, (first, second) in repeated.items():
        assert first == second, name
    for name, rows in identical.items():
        assert all(row == rows[0] for row in rows), name
