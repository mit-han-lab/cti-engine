"""Qwen3.5-9B LoRA serving on one GPU: with a trained adapter, decode graphs and eager decode
report greedy agreement and gate greedy-token logprobs; ``ctikernels.gdn``'s autograd guard
stays quiet under inference mode although the factors require grad."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.qwen3_5_gpu import PINNED_CHECKPOINTS

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

CHECKPOINT = PINNED_CHECKPOINTS[0]
PROMPTS = ("The capital of France is", "def fibonacci(n):\n    ")
# The adapter is trained to continue each prompt with these.
COMPLETIONS = (" Marseille, the old port city.", "return n if n < 2 else 0")
MAX_TOKENS = 24
# Four steps at lr 1e-3 change both prompts' first greedy token (RTX 5090).
TRAINING_STEPS = 4
# Greedy divergence is report-only; before paths part, greedy-token logprob gaps (nats) are
# compared over at least this fraction.
MIN_COMPARED_FRACTION = 0.5
# Not bitwise: the small-batch LoRA plan's split-K atomics reorder sums, so even replays differ
# in the last bits. The dense decode-graph test's bound; this test measured 0.062 on RTX 5090.
GRAPH_LOGPROB_TOLERANCE = 0.1


def _compare(reference, candidate) -> list[dict[str, object]]:
    """Per request: where the greedy paths part and the largest greedy-token logprob gap
    before it."""

    result = []
    for expected, actual in zip(reference, candidate, strict=True):
        parted = next(
            (
                step
                for step, (a, b) in enumerate(
                    zip(expected.sampled_token_ids, actual.sampled_token_ids, strict=True)
                )
                if a != b
            ),
            None,
        )
        compared = MAX_TOKENS if parted is None else parted
        deltas = [
            abs(a - b)
            for a, b in zip(
                expected.sampling_logprobs[:compared],
                actual.sampling_logprobs[:compared],
                strict=True,
            )
        ]
        result.append({"parted_at": parted, "max_logprob_delta": max(deltas, default=0.0)})
    return result


def _assert_follows(comparison: list[dict[str, object]]) -> None:
    for request in comparison:
        assert request["max_logprob_delta"] <= GRAPH_LOGPROB_TOLERANCE, comparison
    compared = sum(
        MAX_TOKENS if request["parted_at"] is None else request["parted_at"]
        for request in comparison
    )
    assert compared >= MIN_COMPARED_FRACTION * MAX_TOKENS * len(comparison), comparison


def test_qwen3_5_lora_decode_graphs_follow_eager(tmp_path: Path) -> None:
    from transformers import AutoTokenizer

    from cti.config import (
        AdamWConfig,
        BatchLayout,
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
    from cti.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
    from cti.protocol import SamplingParams
    from cti.training import SFTDatum, TrainingBatch

    snapshot = str(CHECKPOINT.snapshot())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    data = []
    for prompt, completion in zip(PROMPTS, COMPLETIONS, strict=True):
        prompt_ids = tokenizer(prompt)["input_ids"]
        completion_ids = tokenizer(completion)["input_ids"]
        data.append(
            SFTDatum(
                token_ids=tuple(prompt_ids + completion_ids),
                target_mask=(False,) * len(prompt_ids) + (True,) * len(completion_ids),
            )
        )
    batch = TrainingBatch.sft(data, BatchLayout.PADDED, pad_token_id=tokenizer.eos_token_id)
    params = SamplingParams(
        temperature=0, ignore_eos=True, max_new_tokens=MAX_TOKENS, logprobs=True
    )
    checkpoint = tmp_path / "adapter"

    def serve(*, graphs: bool, resume: bool) -> LLM:
        return LLM(
            EngineOptions(
                model=ModelOptions(path=snapshot, max_seq_len=256),
                memory=MemoryOptions(inference_ratio=0.85, peak_ratio=0.9),
                scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=256),
                lora=LoRAConfig(
                    targets=Qwen3_5ForConditionalGeneration.LORA_TARGETS, rank=16, alpha=32.0
                ),
                training=TrainingConfig(
                    trainable_plan=TrainablePlan.all_declared(),
                    optimizer=AdamWConfig(lr=1e-3, weight_decay=0.0, max_grad_norm=1.0),
                    batching=TrainingBatching(
                        BatchLayout.PADDED,
                        StaticBatching(microbatch_size=len(PROMPTS), max_tokens_per_sample=64),
                    ),
                ),
            ),
            tuning=EngineTuning(cuda_graph_bs=(1, 2, 4) if graphs else ()),
            resume_from_checkpoint=checkpoint if resume else None,
        )

    llm = serve(graphs=True, resume=False)
    try:
        base = llm.sample(list(PROMPTS), params).samples
        with llm.training_session() as session:
            for step_index in range(TRAINING_STEPS):
                session.forward_backward(batch)
                session.optim_step().result()
                assert llm.model_revision == step_index
                if step_index == 0:
                    unpublished = llm.sample(list(PROMPTS), params).samples
                    for actual, expected in zip(unpublished, base, strict=True):
                        assert actual.sampled_token_ids == expected.sampled_token_ids
                        assert actual.sampling_logprobs == expected.sampling_logprobs
                session.publish()
            session.save_checkpoint(checkpoint)
        graph = llm.sample(list(PROMPTS), params)
    finally:
        llm.shutdown()
    llm = serve(graphs=False, resume=True)
    try:
        eager = llm.sample(list(PROMPTS), params)
    finally:
        llm.shutdown()

    assert graph.model_revision == eager.model_revision == TRAINING_STEPS
    # The trained adapter changes the served output.
    assert [s.sampled_token_ids for s in graph.samples] != [s.sampled_token_ids for s in base]
    _assert_follows(_compare(graph.samples, eager.samples))
