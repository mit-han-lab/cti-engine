"""Run repeated padded or packed SFT updates through the public Python API."""

from __future__ import annotations

import argparse
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
from cti.protocol import SamplingParams
from cti.training import SFTDatum, TrainingBatch

_CONVERSATIONS = (
    (
        {"role": "user", "content": "What is two plus two?"},
        {"role": "assistant", "content": "Four."},
    ),
    (
        {"role": "user", "content": "What color is a clear daytime sky?"},
        {"role": "assistant", "content": "Blue."},
    ),
    (
        {"role": "user", "content": "Name the largest planet in our solar system."},
        {"role": "assistant", "content": "Jupiter."},
    ),
    (
        {"role": "user", "content": "Complete the sequence: one, two, three."},
        {"role": "assistant", "content": "Four."},
    ),
)
_INFERENCE_PROMPT = _CONVERSATIONS[0][0]["content"]


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _render_data(tokenizer: PreTrainedTokenizerBase) -> tuple[SFTDatum, ...]:
    """Render this recipe's chats and supervise only each final response."""

    data = []
    for conversation in _CONVERSATIONS:
        prompt_token_ids = tuple(
            tokenizer.apply_chat_template(
                list(conversation[:-1]),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )
        token_ids = tuple(
            tokenizer.apply_chat_template(
                list(conversation),
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
        )
        if token_ids[: len(prompt_token_ids)] != prompt_token_ids:
            raise ValueError("the rendered assistant prompt must prefix the SFT datum")
        data.append(
            SFTDatum(
                token_ids=token_ids,
                target_mask=(False,) * len(prompt_token_ids)
                + (True,) * (len(token_ids) - len(prompt_token_ids)),
            )
        )
    return tuple(data)


def _pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("the tokenizer must define a pad or EOS token")
    return pad_token_id


def _rotated(data: tuple[SFTDatum, ...], submission_index: int) -> tuple[SFTDatum, ...]:
    offset = submission_index % len(data)
    return data[offset:] + data[:offset]


def _batch(
    layout: BatchLayout,
    data: tuple[SFTDatum, ...],
    pad_token_id: int,
) -> TrainingBatch:
    padding = {"pad_token_id": pad_token_id} if layout is BatchLayout.PADDED else {}
    return TrainingBatch.sft(data, layout, **padding)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layout", choices=tuple(BatchLayout), default=BatchLayout.PADDED)
    parser.add_argument("--updates", type=_positive_int, default=2)
    parser.add_argument("--microbatch-size", type=_positive_int, default=2)
    parser.add_argument("--accumulation-steps", type=_positive_int, default=2)
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
    layout = BatchLayout(args.layout)
    checkpoint = args.checkpoint or args.resume or Path("cti-sft.pt")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = _render_data(tokenizer)
    pad_token_id = _pad_token_id(tokenizer)
    training = TrainingConfig(
        trainable_plan=TrainablePlan.all_declared(),
        optimizer=AdamWConfig(lr=1e-4),
        batching=TrainingBatching(
            layout=layout,
            policy=StaticBatching(
                microbatch_size=args.microbatch_size,
                max_tokens_per_sample=max(len(datum.token_ids) for datum in data),
            ),
        ),
        activation_checkpointing=ActivationCheckpointMode(args.activation_checkpointing),
    )
    options = EngineOptions(
        # Training takes text only: an image-text checkpoint leaves its vision tower unloaded.
        model=ModelOptions(path=args.model, text_only=True),
        memory=MemoryOptions(inference_ratio=0.5, peak_ratio=0.5),
        scheduling=SchedulingOptions(max_running_requests=1),
        lora=LoRAConfig(targets=("attention",), rank=64, alpha=128.0),
        training=training,
    )
    llm = LLM(options, resume_from_checkpoint=args.resume)
    try:
        start_update = llm.model_revision
        if start_update > args.updates:
            raise ValueError(
                f"checkpoint revision {start_update} exceeds target updates {args.updates}"
            )
        [before] = llm.generate(
            [_INFERENCE_PROMPT],
            SamplingParams(temperature=0.0, max_new_tokens=8),
        )
        with llm.training_session() as session:
            for update_index in range(start_update, args.updates):
                futures = []
                for accumulation_index in range(args.accumulation_steps):
                    submission_index = update_index * args.accumulation_steps + accumulation_index
                    ordered_data = _rotated(data, submission_index)
                    futures.append(
                        session.forward_backward(_batch(layout, ordered_data, pad_token_id))
                    )
                step = session.optim_step().result()
                completions = tuple(future.result() for future in futures)
                # The next rollouts and any serving read the adapter this step trained.
                revision = session.publish()
                expected_revision = update_index + 1
                if step.step != expected_revision or revision != expected_revision:
                    raise RuntimeError("each optimizer step publishes one model revision")
                if expected_revision % args.checkpoint_every == 0 or (
                    expected_revision == args.updates
                ):
                    session.save_checkpoint(checkpoint)
                target_tokens = sum(completion.normalization_count for completion in completions)
                print(
                    {
                        "update": expected_revision,
                        "loss_per_target": sum(completion.loss_sum for completion in completions)
                        / target_tokens,
                        "target_tokens": target_tokens,
                        "grad_norm": step.grad_norm,
                    }
                )
        [after] = llm.generate(
            [_INFERENCE_PROMPT],
            SamplingParams(temperature=0.0, max_new_tokens=8),
        )
        print(
            {
                "before": before["text"],
                "after": after["text"],
                "model_revision": llm.model_revision,
                "checkpoint": str(checkpoint),
            }
        )
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
