from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar


class CacheType(StrEnum):
    RADIX = "radix"
    NAIVE = "naive"


@dataclass(frozen=True)
class ServerLaunch:
    """The shared V0 inference server: its model snapshot, rank count, prefix cache and, for
    a MoE model on several ranks, whether whole experts split over the ranks (expert
    parallelism) or every expert splits by its width (expert tensor parallelism)."""

    MODEL_ID: ClassVar[str] = "Qwen/Qwen3-0.6B"
    MODEL_REVISION: ClassVar[str] = "c1899de289a04d12100db370d81485cdf75e47ca"
    INFERENCE_MEMORY_RATIO: ClassVar[float] = 0.9
    PEAK_MEMORY_RATIO: ClassVar[float] = 0.9
    PAGE_SIZE: ClassVar[int] = 1

    model: str = MODEL_ID
    revision: str = MODEL_REVISION
    tensor_parallel_size: int = 1
    cache_type: CacheType = CacheType.RADIX
    expert_parallel: bool = True

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor-parallel size must be positive")

    def command(self, model_path: Path, port: int) -> tuple[str, ...]:
        """The server's command line for the snapshot at ``model_path``: this file's ``main``
        with the frozen options."""
        return (
            sys.executable,
            str(Path(__file__).resolve()),
            "--model.path",
            str(model_path),
            "--model.dtype",
            "bfloat16",
            "--model.max-seq-len",
            "32768",
            "--parallel.num-ranks",
            str(self.tensor_parallel_size),
            "--parallel.enable-expert-parallel",
            str(self.expert_parallel).lower(),
            "--cache.prefix",
            self.cache_type,
            "--scheduling.max-running-requests",
            "32",
            "--scheduling.max-batch-tokens",
            "16384",
            "--memory.inference-ratio",
            str(self.INFERENCE_MEMORY_RATIO),
            "--memory.peak-ratio",
            str(self.PEAK_MEMORY_RATIO),
            "--server.host",
            "127.0.0.1",
            "--server.port",
            str(port),
        )

    def resolve_model(self) -> Path:
        """The local snapshot of the model at its revision."""
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=self.model,
                revision=self.revision,
                local_files_only=True,
            )
        ).resolve()
        if snapshot.name != self.revision:
            raise RuntimeError(f"resolved unexpected model snapshot: {snapshot}")
        return snapshot


def main() -> None:
    """Serve the command line with the internal tuning the profile freezes."""
    from cti.config import EngineTuning
    from cti.server import launch_server

    launch_server(
        tuning=EngineTuning(
            attention_backend="fi", cuda_graph_max_bs=32, page_size=ServerLaunch.PAGE_SIZE
        )
    )


if __name__ == "__main__":
    main()
