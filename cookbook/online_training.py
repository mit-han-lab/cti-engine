"""Serve OpenAI requests while locally controlling training on the same engine: an SFT step,
then a GRPO round sampled from the served model while it serves HTTP traffic."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from cti.config import (
    AdamWConfig,
    BatchLayout,
    EngineOptions,
    LoRAConfig,
    ModelOptions,
    SchedulingOptions,
    ServerOptions,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.protocol import AdmissionPriority, SamplingParams, SamplingResult
from cti.server import OnlineServer
from cti.training import SFTDatum, TrainingBatch

_GROUP_SIZE = 4
_MAX_SEQUENCE_TOKENS = 64


def _pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("the tokenizer must define a pad or EOS token")
    return pad_token_id


def _training_batch(tokenizer: PreTrainedTokenizerBase) -> TrainingBatch:
    prompt = tuple(tokenizer("Reply with OK:", add_special_tokens=True).input_ids)
    response = tuple(tokenizer(" OK", add_special_tokens=False).input_ids)
    datum = SFTDatum(
        token_ids=prompt + response,
        target_mask=(False,) * len(prompt) + (True,) * len(response),
    )
    return TrainingBatch.sft(
        (datum, datum), BatchLayout.PADDED, pad_token_id=_pad_token_id(tokenizer)
    )


def _grpo_batch(sampling: SamplingResult, pad_token_id: int) -> TrainingBatch:
    """One group, scored by a toy token-content reward; replace it for real tasks."""

    rewards = [
        sum(token_id % 97 for token_id in sample.sampled_token_ids) / len(sample.sampled_token_ids)
        for sample in sampling.samples
    ]
    return TrainingBatch.grpo(
        sampling, rewards, BatchLayout.PADDED, group_size=_GROUP_SIZE, pad_token_id=pad_token_id
    )


def _stream_one(base_url: str, model: str) -> dict[str, object]:
    with OpenAI(base_url=base_url, api_key="unused") as client:
        started_at = time.perf_counter()
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=8,
            temperature=0.0,
            stream=True,
        )
        fragments: list[str] = []
        first_token_seconds: float | None = None
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                if first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - started_at
                fragments.append(content)
        return {
            "first_token_seconds": first_token_seconds,
            "text": "".join(fragments),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    return parser


def main() -> None:
    args = _parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    batch = _training_batch(tokenizer)
    training = TrainingConfig(
        trainable_plan=TrainablePlan.all_declared(),
        optimizer=AdamWConfig(lr=1e-4),
        batching=TrainingBatching(
            layout=BatchLayout.PADDED,
            policy=StaticBatching(microbatch_size=1, max_tokens_per_sample=_MAX_SEQUENCE_TOKENS),
        ),
    )
    options = EngineOptions(
        model=ModelOptions(path=args.model),
        scheduling=SchedulingOptions(max_running_requests=8),
        lora=LoRAConfig(targets=("attention",), rank=64, alpha=128.0),
        training=training,
    )
    with OnlineServer(options, ServerOptions(host=args.host, port=args.port)) as server:
        base_url = f"http://{args.host}:{args.port}/v1"
        baseline = _stream_one(base_url, args.model)
        with server.training_session() as session:
            evidence = session.forward_backward(batch).result()
            session.pause_for_serving()
            result = _stream_one(base_url, args.model)
            session.resume_training()
            session.optim_step().result()
            # Serving reads the trained adapter from here on.
            session.publish()

        # Rollouts share admission with HTTP requests; LOW lets HTTP requests go first.
        params = [
            SamplingParams(
                max_new_tokens=8, temperature=0.7, seed=seed, logprobs=True, logprobs_processed=True
            )
            for seed in range(1, _GROUP_SIZE + 1)
        ]
        with ThreadPoolExecutor(1) as http:
            serving = http.submit(_stream_one, base_url, args.model)
            sampling = server.sample(
                ["Name a primary color."] * _GROUP_SIZE, params, priority=AdmissionPriority.LOW
            )
            served = serving.result()
        with server.training_session() as session:
            grpo_batch = _grpo_batch(sampling, _pad_token_id(tokenizer))
            grpo = session.forward_backward(grpo_batch).result()
            grpo_step = session.optim_step().result()
            session.publish()

        print(
            {
                "switch_control": "explicit",
                "baseline_response": baseline["text"],
                "first_token_seconds": result["first_token_seconds"],
                "response": result["text"],
                "loss_sum": evidence.loss_sum,
                "served_during_rollout": served["text"],
                "rollout_texts": sampling.texts,
                "grpo_loss_sum": grpo.loss_sum,
                "model_revision": grpo_step.step,
            }
        )


if __name__ == "__main__":
    main()
