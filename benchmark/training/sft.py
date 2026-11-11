from __future__ import annotations

import argparse
import importlib.metadata
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

PROFILE_NAME = "cti-sft-v1"


@dataclass(frozen=True)
class SFTConfig:
    model_path: Path
    output: Path
    layout: str
    checkpoint: str
    batch_size: int
    microbatch_size: int
    sequence_length: int
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
                self.batch_size,
                self.microbatch_size,
                self.sequence_length,
                self.measure,
                self.rank,
            )
            < 1
        ):
            raise ValueError("batch, sequence, measurement, and rank values must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence length must be at least two")
        if self.microbatch_size > self.batch_size:
            raise ValueError("microbatch size cannot exceed batch size")
        if self.batch_size % self.microbatch_size:
            raise ValueError("batch size must be divisible by microbatch size")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 < self.inference_memory_ratio <= self.peak_memory_ratio <= 1:
            raise ValueError("memory ratios must satisfy 0 < inference <= peak <= 1")


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _build_batch(config: SFTConfig, *, vocab_size: int, pad_token_id: int):
    from cti.config import BatchLayout
    from cti.training import SFTDatum, TrainingBatch

    data = tuple(
        SFTDatum(
            token_ids=tuple(
                16 + ((config.seed + row * config.sequence_length + column) % (vocab_size - 16))
                for column in range(config.sequence_length)
            ),
            target_mask=(False,) + (True,) * (config.sequence_length - 1),
        )
        for row in range(config.batch_size)
    )
    if config.layout == "padded":
        return TrainingBatch.sft(
            data,
            BatchLayout.PADDED,
            pad_token_id=pad_token_id,
            padded_length=config.sequence_length,
        )
    return TrainingBatch.sft(data, BatchLayout.PACKED)


def run(config: SFTConfig) -> dict[str, object]:
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
        SFTShapeProfile,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.llm import LLM
    from cti.tokenizer.text import load_tokenizer

    prepare_output_file(config.output)
    source_revision, source_dirty = git_source_state()
    tokenizer = load_tokenizer(str(config.model_path))
    pad_token_id = int(tokenizer.pad_token_id)
    batch = _build_batch(config, vocab_size=int(tokenizer.vocab_size), pad_token_id=pad_token_id)
    shape_profile = (
        SFTShapeProfile(
            (config.sequence_length,) * config.batch_size,
            padded_width=config.sequence_length,
        )
        if config.layout == "padded"
        else None
    )
    options = EngineOptions(
        model=ModelOptions(
            path=str(config.model_path),
            max_seq_len=config.sequence_length,
            initialization_seed=config.seed,
        ),
        memory=MemoryOptions(
            inference_ratio=config.inference_memory_ratio, peak_ratio=config.peak_memory_ratio
        ),
        scheduling=SchedulingOptions(
            max_running_requests=1, max_batch_tokens=config.sequence_length
        ),
        cache=CacheOptions(prefix="naive"),
        lora=LoRAConfig(("attention", "mlp"), rank=config.rank, alpha=config.alpha),
        training=TrainingConfig(
            trainable_plan=TrainablePlan.all_declared(),
            optimizer=AdamWConfig(lr=2e-4, weight_decay=0.0, max_grad_norm=1.0),
            batching=TrainingBatching(
                layout=BatchLayout(config.layout),
                policy=StaticBatching(
                    microbatch_size=config.microbatch_size,
                    max_tokens_per_sample=config.sequence_length,
                ),
            ),
            activation_checkpointing=ActivationCheckpointMode(config.checkpoint),
            sft_profile=shape_profile,
        ),
    )
    llm = LLM(options, tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)))
    samples = []
    try:
        with NvmlDevice(current_cuda_uuid()) as device:
            with DeviceMemoryMonitor(
                device,
                interval=NVML_MEMORY_SAMPLE_INTERVAL_SECONDS,
            ) as memory_monitor:
                with llm.training_session() as session:

                    def step() -> dict[str, float | int]:
                        started = time.perf_counter()
                        completion = session.forward_backward(batch).result()
                        backward_done = time.perf_counter()
                        update = session.optim_step().result()
                        finished = time.perf_counter()
                        if not math.isfinite(completion.loss_sum) or not math.isfinite(
                            update.grad_norm
                        ):
                            raise RuntimeError("SFT produced a non-finite measurement")
                        step_ms = (finished - started) * 1000
                        return {
                            "step_ms": step_ms,
                            "forward_backward_ms": (backward_done - started) * 1000,
                            "optimizer_ms": (finished - backward_done) * 1000,
                            "target_tokens_per_second": (
                                completion.normalization_count * 1000 / step_ms
                            ),
                            "loss_sum": completion.loss_sum,
                            "grad_norm": update.grad_norm,
                            "normalization_count": update.normalization_count,
                        }

                    for _ in range(config.warmup):
                        step()
                    memory_monitor.reset_region()
                    samples = [step() for _ in range(config.measure)]
                peak_device_memory_bytes = memory_monitor.region_peak_bytes()
    finally:
        llm.shutdown()

    metrics = {
        name: distribution(tuple(float(sample[name]) for sample in samples))
        for name in (
            "step_ms",
            "forward_backward_ms",
            "optimizer_ms",
            "target_tokens_per_second",
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
            "flashinfer": _version("flashinfer-python"),
            "dtype": "bfloat16",
        },
        "samples": samples,
        "summary": metrics,
        "peak_device_memory_bytes": peak_device_memory_bytes,
    }
    write_receipt(config.output, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> SFTConfig:
    parser = argparse.ArgumentParser(description="Run one cti SFT performance profile.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layout", choices=("padded", "packed"), default="packed")
    parser.add_argument("--checkpoint", choices=("none", "full"), default="none")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--inference-memory-ratio", type=float, default=0.9)
    parser.add_argument("--peak-memory-ratio", type=float, default=0.9)
    args = parser.parse_args(argv)
    return SFTConfig(**vars(args))


def main() -> None:
    result = run(_parse_args())
    print(f"step_ms_p95={result['summary']['step_ms']['p95']:.3f}", flush=True)


if __name__ == "__main__":
    main()
