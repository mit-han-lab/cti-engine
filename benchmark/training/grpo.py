from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .._profile import distribution
from .._provenance import installed_package_provenance, write_receipt
from .._runtime import (
    NVML_MEMORY_SAMPLE_INTERVAL_SECONDS,
    DeviceMemoryMonitor,
    NvmlDevice,
    current_cuda_uuid,
    git_source_state,
    prepare_output_file,
)

PROFILE_NAME = "cti-grpo-v1"


@dataclass(frozen=True)
class GRPOConfig:
    model_path: Path
    output: Path
    layout: str
    checkpoint: str
    num_prompts: int
    group_size: int
    prompt_tokens: int
    completion_tokens: int
    microbatch_size: int
    warmup: int
    measure: int
    rank: int
    alpha: float
    seed: int
    inference_memory_ratio: float
    peak_memory_ratio: float

    def __post_init__(self) -> None:
        if self.layout not in ("padded", "packed"):
            raise ValueError("layout must be padded or packed")
        if self.checkpoint not in ("none", "full"):
            raise ValueError("checkpoint must be none or full")
        if (
            min(
                self.num_prompts,
                self.group_size,
                self.prompt_tokens,
                self.completion_tokens,
                self.microbatch_size,
                self.measure,
                self.rank,
            )
            < 1
        ):
            raise ValueError("GRPO dimensions and measurement counts must be positive")
        if self.group_size < 2:
            raise ValueError("group size must be at least two")
        if self.rollout_batch_size % self.microbatch_size:
            raise ValueError("rollout batch size must be divisible by microbatch size")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 < self.inference_memory_ratio <= self.peak_memory_ratio <= 1:
            raise ValueError("memory ratios must satisfy 0 < inference <= peak <= 1")

    @property
    def rollout_batch_size(self) -> int:
        return self.num_prompts * self.group_size


def _graph_batch_sizes(max_batch_size: int) -> list[int]:
    sizes = []
    size = 1
    while size < max_batch_size:
        sizes.append(size)
        size *= 2
    sizes.append(max_batch_size)
    return sizes


def _rank_rewards(sample_count: int, group_size: int) -> tuple[float, ...]:
    if group_size < 2 or sample_count < 1 or sample_count % group_size:
        raise ValueError("GRPO samples must contain complete groups")
    return tuple(float(index % group_size) for index in range(sample_count))


def run(config: GRPOConfig) -> dict[str, object]:
    import torch

    from cti.config import (
        ActivationCheckpointMode,
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
    from cti.tokenizer.text import load_tokenizer
    from cti.training import TrainingBatch

    prepare_output_file(config.output)
    source_revision, source_dirty = git_source_state()
    tokenizer = load_tokenizer(str(config.model_path))
    vocab_size = int(tokenizer.vocab_size)
    pad_token_id = int(tokenizer.pad_token_id)
    base_prompts = [
        [
            16 + ((config.seed + row * config.prompt_tokens + column) % (vocab_size - 16))
            for column in range(config.prompt_tokens)
        ]
        for row in range(config.num_prompts)
    ]
    prompts = [prompt.copy() for prompt in base_prompts for _ in range(config.group_size)]
    policies = [
        SamplingParams(
            temperature=1.0,
            ignore_eos=True,
            max_new_tokens=config.completion_tokens,
            seed=config.seed * 10_000 + index,
            logprobs=True,
            logprobs_processed=True,
        )
        for index in range(config.rollout_batch_size)
    ]
    max_sequence_length = config.prompt_tokens + config.completion_tokens
    options = EngineOptions(
        model=ModelOptions(
            path=str(config.model_path),
            max_seq_len=max_sequence_length,
            initialization_seed=config.seed,
        ),
        memory=MemoryOptions(
            inference_ratio=config.inference_memory_ratio, peak_ratio=config.peak_memory_ratio
        ),
        scheduling=SchedulingOptions(
            max_running_requests=config.rollout_batch_size,
            max_batch_tokens=max(config.prompt_tokens * config.rollout_batch_size, 512),
        ),
        cache=CacheOptions(prefix="radix"),
        lora=LoRAConfig(("attention", "mlp"), rank=config.rank, alpha=config.alpha),
        training=TrainingConfig(
            trainable_plan=TrainablePlan.all_declared(),
            optimizer=AdamWConfig(lr=2e-4, weight_decay=0.0, max_grad_norm=1.0),
            batching=TrainingBatching(
                layout=BatchLayout(config.layout),
                policy=StaticBatching(
                    microbatch_size=config.microbatch_size,
                    max_tokens_per_sample=max_sequence_length,
                ),
            ),
            activation_checkpointing=ActivationCheckpointMode(config.checkpoint),
        ),
    )
    tuning = EngineTuning(
        attention_backend="fi",
        cuda_graph_bs=tuple(_graph_batch_sizes(config.rollout_batch_size)),
    )
    llm = LLM(options, tuning=tuning)

    def build_batch(rollout):
        rewards = _rank_rewards(len(rollout.samples), config.group_size)
        if config.layout == "padded":
            return TrainingBatch.grpo(
                rollout,
                rewards,
                BatchLayout.PADDED,
                group_size=config.group_size,
                pad_token_id=pad_token_id,
                padded_length=max_sequence_length,
            )
        return TrainingBatch.grpo(
            rollout, rewards, BatchLayout.PACKED, group_size=config.group_size
        )

    samples = []
    training_peaks = []
    try:
        with NvmlDevice(current_cuda_uuid()) as device:
            with DeviceMemoryMonitor(
                device,
                interval=NVML_MEMORY_SAMPLE_INTERVAL_SECONDS,
            ) as memory_monitor:
                for iteration in range(config.warmup + config.measure):
                    measured = iteration >= config.warmup
                    cycle_started = time.perf_counter()
                    rollout_started = time.perf_counter()
                    rollout = llm.sample(prompts, policies)
                    rollout_done = time.perf_counter()
                    preparation_started = time.perf_counter()
                    batch = build_batch(rollout)
                    preparation_done = time.perf_counter()
                    session = llm.training_session()
                    switch_started = time.perf_counter()
                    session.__enter__()
                    training_started = time.perf_counter()
                    memory_monitor.reset_region()
                    completion_future = session.forward_backward(batch)
                    update = session.optim_step().result()
                    completion = completion_future.result()
                    training_done = time.perf_counter()
                    training_peak = memory_monitor.region_peak_bytes()
                    resume_started = time.perf_counter()
                    session.close()
                    cycle_done = time.perf_counter()

                    if not math.isfinite(update.grad_norm) or update.grad_norm <= 0:
                        raise RuntimeError("GRPO produced a non-positive or non-finite gradient")
                    if update.step != llm.model_revision:
                        raise RuntimeError("GRPO optimizer revision was not published")
                    if completion.normalization_count != config.rollout_batch_size:
                        raise RuntimeError("GRPO completion count changed")
                    if measured:
                        rollout_ms = (rollout_done - rollout_started) * 1000
                        samples.append(
                            {
                                "cycle_ms": (cycle_done - cycle_started) * 1000,
                                "rollout_ms": rollout_ms,
                                "rollout_tokens_per_second": (
                                    batch.num_targets * 1000 / rollout_ms
                                ),
                                "batch_preparation_ms": (preparation_done - preparation_started)
                                * 1000,
                                "switch_to_training_ms": (training_started - switch_started) * 1000,
                                "policy_update_ms": (training_done - training_started) * 1000,
                                "switch_to_inference_ms": (cycle_done - resume_started) * 1000,
                                "grad_norm": update.grad_norm,
                                "model_revision": update.step,
                                "response_tokens": batch.num_targets,
                            }
                        )
                        if training_peak is not None:
                            training_peaks.append(training_peak)
    finally:
        llm.shutdown()

    summary = {
        name: distribution(tuple(float(sample[name]) for sample in samples))
        for name in (
            "cycle_ms",
            "rollout_ms",
            "rollout_tokens_per_second",
            "batch_preparation_ms",
            "switch_to_training_ms",
            "policy_update_ms",
            "switch_to_inference_ms",
        )
    }
    result = {
        "schema_version": 1,
        "benchmark": PROFILE_NAME,
        "source": {"revision": source_revision, "dirty": source_dirty},
        "config": {
            **asdict(config),
            "model_path": str(config.model_path),
            "output": str(config.output),
        },
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "packages": installed_package_provenance(),
            "dtype": "bfloat16",
            "rollout_batch_size": config.rollout_batch_size,
        },
        "samples": samples,
        "summary": summary,
        "training_peak_device_memory_bytes": max(training_peaks) if training_peaks else None,
    }
    write_receipt(config.output, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> GRPOConfig:
    parser = argparse.ArgumentParser(description="Run one cti GRPO performance profile.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layout", choices=("padded", "packed"), default="packed")
    parser.add_argument("--checkpoint", choices=("none", "full"), default="none")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--completion-tokens", type=int, default=64)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--inference-memory-ratio", type=float, default=0.9)
    parser.add_argument("--peak-memory-ratio", type=float, default=0.9)
    args = parser.parse_args(argv)
    return GRPOConfig(**vars(args))


def main() -> None:
    result = run(_parse_args())
    print(f"cycle_ms_p95={result['summary']['cycle_ms']['p95']:.3f}", flush=True)


if __name__ == "__main__":
    main()
