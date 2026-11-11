from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ._profile import distribution
from ._provenance import installed_package_provenance, write_receipt
from ._runtime import (
    NvmlDevice,
    current_cuda_uuid,
    git_source_state,
    prepare_output_file,
)

PROFILE_NAME = "cti-switching-v1"


@dataclass(frozen=True)
class SwitchingConfig:
    model_path: Path
    output: Path
    batch_size: int
    prompt_tokens: int
    warmup: int
    measure: int
    rank: int
    alpha: float
    seed: int
    inference_memory_ratio: float
    peak_memory_ratio: float
    cache_type: str = "radix"

    def __post_init__(self) -> None:
        if min(self.batch_size, self.prompt_tokens, self.measure, self.rank) < 1:
            raise ValueError("batch, prompt, measurement, and rank values must be positive")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 < self.inference_memory_ratio <= self.peak_memory_ratio <= 1:
            raise ValueError("memory ratios must satisfy 0 < inference <= peak <= 1")


def _graph_batch_sizes(max_batch_size: int) -> list[int]:
    sizes = []
    size = 1
    while size < max_batch_size:
        sizes.append(size)
        size *= 2
    sizes.append(max_batch_size)
    return sizes


def run(config: SwitchingConfig) -> dict[str, object]:
    import torch

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
    from cti.tokenizer.text import load_tokenizer

    prepare_output_file(config.output)
    source_revision, source_dirty = git_source_state()
    tokenizer = load_tokenizer(str(config.model_path))
    vocab_size = int(tokenizer.vocab_size)
    prompts = [
        [
            16 + ((config.seed + row * config.prompt_tokens + column) % (vocab_size - 16))
            for column in range(config.prompt_tokens)
        ]
        for row in range(config.batch_size)
    ]
    options = EngineOptions(
        model=ModelOptions(
            path=str(config.model_path),
            max_seq_len=config.prompt_tokens + 1,
            initialization_seed=config.seed,
        ),
        memory=MemoryOptions(
            inference_ratio=config.inference_memory_ratio, peak_ratio=config.peak_memory_ratio
        ),
        scheduling=SchedulingOptions(
            max_running_requests=config.batch_size,
            max_batch_tokens=config.prompt_tokens * config.batch_size,
        ),
        cache=CacheOptions(prefix=config.cache_type),
        lora=LoRAConfig(("attention", "mlp"), rank=config.rank, alpha=config.alpha),
        training=TrainingConfig(
            trainable_plan=TrainablePlan.all_declared(),
            optimizer=AdamWConfig(lr=2e-4, weight_decay=0.0, max_grad_norm=1.0),
            batching=TrainingBatching(
                layout=BatchLayout.PADDED,
                policy=StaticBatching(
                    microbatch_size=1,
                    max_tokens_per_sample=config.prompt_tokens + 1,
                ),
            ),
        ),
    )
    tuning = EngineTuning(
        attention_backend="fi", cuda_graph_bs=tuple(_graph_batch_sizes(config.batch_size))
    )
    llm = LLM(options, tuning=tuning)
    one_token = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_new_tokens=1,
    )
    samples = []
    try:
        with NvmlDevice(current_cuda_uuid()) as device:

            def used_memory_bytes() -> int | None:
                value = device.used_memory()["value"]
                return value if type(value) is int else None

            def cycle() -> dict[str, float | int | None]:
                inference_memory = used_memory_bytes()
                started = time.perf_counter_ns()
                session = llm.training_session()
                session.__enter__()
                training_ready = time.perf_counter_ns()
                training_memory = used_memory_bytes()
                resume_started = time.perf_counter_ns()
                session.close()
                inference_ready = time.perf_counter_ns()
                resumed_memory = used_memory_bytes()
                first_request_started = time.perf_counter_ns()
                result = llm.sample(prompts, one_token)
                first_request_done = time.perf_counter_ns()
                if any(len(sample.sampled_token_ids) != 1 for sample in result.samples):
                    raise RuntimeError("post-switch request did not produce one token")
                inference_to_training_ms = (training_ready - started) / 1e6
                training_to_inference_ms = (inference_ready - resume_started) / 1e6
                return {
                    "inference_to_training_ms": inference_to_training_ms,
                    "training_to_inference_ms": training_to_inference_ms,
                    "round_trip_ms": inference_to_training_ms + training_to_inference_ms,
                    "first_post_switch_request_ms": (first_request_done - first_request_started)
                    / 1e6,
                    "inference_to_training_memory_delta_bytes": (
                        inference_memory - training_memory
                        if inference_memory is not None and training_memory is not None
                        else None
                    ),
                    "training_to_inference_memory_delta_bytes": (
                        resumed_memory - training_memory
                        if resumed_memory is not None and training_memory is not None
                        else None
                    ),
                }

            for _ in range(config.warmup):
                cycle()
            samples = [cycle() for _ in range(config.measure)]
    finally:
        llm.shutdown()

    summary = {
        name: distribution(tuple(float(sample[name]) for sample in samples))
        for name in (
            "inference_to_training_ms",
            "training_to_inference_ms",
            "round_trip_ms",
            "first_post_switch_request_ms",
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
            "kv_release_policy": "all-layers",
            "dtype": "bfloat16",
        },
        "samples": samples,
        "summary": summary,
    }
    write_receipt(config.output, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> SwitchingConfig:
    parser = argparse.ArgumentParser(description="Run the cti residency-switching profile.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-type", choices=("radix", "naive"), default="radix")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--inference-memory-ratio", type=float, default=0.9)
    parser.add_argument("--peak-memory-ratio", type=float, default=0.9)
    args = parser.parse_args(argv)
    return SwitchingConfig(**vars(args))


def main() -> None:
    result = run(_parse_args())
    print(f"round_trip_ms_p95={result['summary']['round_trip_ms']['p95']:.3f}", flush=True)


if __name__ == "__main__":
    main()
