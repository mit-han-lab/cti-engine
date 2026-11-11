"""Run synchronous multi-round padded or packed GRPO through the public API."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import replace
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from cti.config import (
    ActivationCheckpointMode,
    AdamWConfig,
    BatchLayout,
    EngineOptions,
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
from cti.protocol import SamplingParams, SamplingResult
from cti.training import TrainingBatch

_PROMPTS = (
    "Name a primary color.",
    "Name a planet in our solar system.",
    "Give one word associated with winter.",
)
_REWARD_EPSILON = 1e-6


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("the tokenizer must define a pad or EOS token")
    return pad_token_id


def _toy_rewards(sampling: SamplingResult) -> tuple[float, ...]:
    """Score token content; replace this caller-owned policy for real tasks."""

    return tuple(
        sum(token_id % 97 for token_id in sample.sampled_token_ids) / len(sample.sampled_token_ids)
        for sample in sampling.samples
    )


def _has_reward_variation(rewards: tuple[float, ...]) -> bool:
    return statistics.pstdev(rewards) > _REWARD_EPSILON


def _merge_sampling(groups: tuple[SamplingResult, ...]) -> SamplingResult:
    if not groups:
        raise ValueError("groups must not be empty")
    revision = groups[0].model_revision
    samples = []
    texts = []
    for group in groups:
        group.require_revision(revision)
        for sample, text in zip(group.samples, group.texts, strict=True):
            samples.append(replace(sample, request_id=len(samples)))
            texts.append(text)
    return SamplingResult(
        samples=tuple(samples),
        texts=tuple(texts),
        model_revision=revision,
    )


def _collect_rewarded_sampling(
    llm: LLM,
    prompts: tuple[tuple[int, ...], ...],
    *,
    revision: int,
    accumulation_index: int,
    accumulation_steps: int,
    groups_per_batch: int,
    group_size: int,
    max_group_attempts: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[SamplingResult, tuple[float, ...], int]:
    accepted_sampling = []
    accepted_rewards: list[float] = []
    rejected_groups = 0
    max_attempts = groups_per_batch * max_group_attempts
    for attempt in range(max_attempts):
        prompt = prompts[(revision + accumulation_index + attempt) % len(prompts)]
        attempt_id = (revision * accumulation_steps + accumulation_index) * max_attempts + attempt
        sampling = llm.sample(
            [list(prompt)] * group_size,
            [
                SamplingParams(
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    seed=attempt_id * group_size + member + 1,
                    logprobs=True,
                    logprobs_processed=True,
                )
                for member in range(group_size)
            ],
        )
        sampling.require_revision(revision)
        rewards = _toy_rewards(sampling)
        if not _has_reward_variation(rewards):
            rejected_groups += 1
            continue
        accepted_sampling.append(sampling)
        accepted_rewards.extend(rewards)
        if len(accepted_sampling) == groups_per_batch:
            return (
                _merge_sampling(tuple(accepted_sampling)),
                tuple(accepted_rewards),
                rejected_groups,
            )
    raise RuntimeError(
        f"collected {len(accepted_sampling)} varying reward groups after {max_attempts} attempts"
    )


def _batch(
    layout: BatchLayout,
    sampling: SamplingResult,
    rewards: tuple[float, ...],
    *,
    group_size: int,
    pad_token_id: int,
) -> TrainingBatch:
    padding = {"pad_token_id": pad_token_id} if layout is BatchLayout.PADDED else {}
    return TrainingBatch.grpo(sampling, rewards, layout, group_size=group_size, **padding)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layout", choices=tuple(BatchLayout), default=BatchLayout.PACKED)
    parser.add_argument("--rounds", type=_positive_int, default=2)
    parser.add_argument("--group-size", type=_positive_int, default=4)
    parser.add_argument("--groups-per-batch", type=_positive_int, default=1)
    parser.add_argument("--accumulation-steps", type=_positive_int, default=2)
    parser.add_argument("--max-group-attempts", type=_positive_int, default=8)
    parser.add_argument("--max-new-tokens", type=_positive_int, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-every", type=_positive_int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--activation-checkpointing",
        choices=tuple(ActivationCheckpointMode),
        default=ActivationCheckpointMode.FULL,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.group_size < 2:
        raise ValueError("--group-size must be at least two")
    if args.temperature < 1e-6:
        raise ValueError("--temperature must be at least 1e-6")
    layout = BatchLayout(args.layout)
    checkpoint = args.checkpoint or args.resume or Path("cti-grpo.pt")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = tuple(tuple(tokenizer.encode(prompt, add_special_tokens=True)) for prompt in _PROMPTS)
    pad_token_id = _pad_token_id(tokenizer)
    max_sequence_tokens = max(map(len, prompts)) + args.max_new_tokens
    training = TrainingConfig(
        trainable_plan=TrainablePlan.all_declared(),
        optimizer=AdamWConfig(lr=1e-5),
        batching=TrainingBatching(
            layout=layout,
            policy=StaticBatching(
                microbatch_size=args.group_size,
                max_tokens_per_sample=max_sequence_tokens,
            ),
        ),
        activation_checkpointing=ActivationCheckpointMode(args.activation_checkpointing),
    )
    options = EngineOptions(
        # Training takes text only: an image-text checkpoint leaves its vision tower unloaded.
        model=ModelOptions(path=args.model, max_seq_len=max_sequence_tokens, text_only=True),
        memory=MemoryOptions(inference_ratio=0.5, peak_ratio=0.5),
        scheduling=SchedulingOptions(max_running_requests=args.group_size),
        lora=LoRAConfig(targets=("attention",), rank=64, alpha=128.0),
        training=training,
    )
    llm = LLM(options, resume_from_checkpoint=args.resume)
    try:
        start_round = llm.model_revision
        if start_round > args.rounds:
            raise ValueError(
                f"checkpoint revision {start_round} exceeds target rounds {args.rounds}"
            )
        for round_index in range(start_round, args.rounds):
            batches = []
            rejected_groups = 0
            for accumulation_index in range(args.accumulation_steps):
                sampling, rewards, rejected = _collect_rewarded_sampling(
                    llm,
                    prompts,
                    revision=round_index,
                    accumulation_index=accumulation_index,
                    accumulation_steps=args.accumulation_steps,
                    groups_per_batch=args.groups_per_batch,
                    group_size=args.group_size,
                    max_group_attempts=args.max_group_attempts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                rejected_groups += rejected
                batches.append(
                    _batch(
                        layout,
                        sampling,
                        rewards,
                        group_size=args.group_size,
                        pad_token_id=pad_token_id,
                    )
                )
            with llm.training_session() as session:
                futures = tuple(session.forward_backward(batch) for batch in batches)
                step = session.optim_step().result()
                completions = tuple(future.result() for future in futures)
                # The next rollouts and any serving read the adapter this step trained.
                revision = session.publish()
                expected_revision = round_index + 1
                if step.step != expected_revision or revision != expected_revision:
                    raise RuntimeError("each optimizer step publishes one model revision")
                if expected_revision % args.checkpoint_every == 0 or (
                    expected_revision == args.rounds
                ):
                    session.save_checkpoint(checkpoint)
            print(
                {
                    "round": expected_revision,
                    "sequences": sum(completion.normalization_count for completion in completions),
                    "rejected_groups": rejected_groups,
                    "grad_norm": step.grad_norm,
                    "checkpoint": str(checkpoint),
                }
            )
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
