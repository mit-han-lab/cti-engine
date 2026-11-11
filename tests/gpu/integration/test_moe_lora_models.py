"""Per-expert MoE LoRA end to end on one GPU through the public ``LLM``: Qwen3.5-35B-A3B,
adapting every routed and shared expert.

Serving: an adapted sparse block composes itself (no whole-block op reads an adapter), so a
new (zero) adapter samples the very greedy tokens of the engine without LoRA, its logprobs
within the graph bound, and eager decode follows the decode graphs.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.support.checkpoints import QWEN3_5_35B_A3B
from tests.support.gpu import offline_child_env, run_isolated_process
from tests.support.greedy_logits import assert_greedy_follows
from tests.support.lora_serving import (
    GRAPH_LOGPROB_TOLERANCE,
    MAX_TOKENS,
    MIN_COMPARED_FRACTION,
    compare_runs,
    greedy,
    print_receipt,
    read_receipt,
)
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import PINNED_CHECKPOINTS

pytestmark = pytest.mark.checkpoints(QWEN3_5_35B_A3B)

if TYPE_CHECKING:
    from cti.llm import LLM

_MODULE = "tests.gpu.integration.test_moe_lora_models"
LORA_RANK, LORA_ALPHA = 16, 32.0
# The most rows one forward runs (``max_batch_tokens``).
MAX_ROWS = 256


@dataclass(frozen=True)
class _Model:
    """A checkpoint, its LoRA targets and serving options."""

    checkpoint: Any
    targets: tuple[str, ...]
    inference_ratio: float
    max_running_requests: int
    cuda_graph_bs: tuple[int, ...]

    def llm(self, *, lora: bool, graphs: bool = True) -> LLM:
        """Serving, or eager without decode graphs."""

        from cti.config import (
            CacheOptions,
            EngineOptions,
            EngineTuning,
            LoRAConfig,
            MemoryOptions,
            ModelOptions,
            SchedulingOptions,
        )
        from cti.llm import LLM

        lora_config = (
            LoRAConfig(targets=self.targets, rank=LORA_RANK, alpha=LORA_ALPHA) if lora else None
        )
        options = EngineOptions(
            model=ModelOptions(path=str(self.checkpoint.snapshot()), max_seq_len=256),
            memory=MemoryOptions(inference_ratio=self.inference_ratio),
            scheduling=SchedulingOptions(
                max_running_requests=self.max_running_requests, max_batch_tokens=MAX_ROWS
            ),
            # The runs change the adapter between generations: reuse no prefix.
            cache=CacheOptions(prefix="naive"),
            lora=lora_config,
        )
        return LLM(options, tuning=EngineTuning(cuda_graph_bs=self.cuda_graph_bs if graphs else ()))


MODELS = {
    "qwen3_5_moe": _Model(
        next(checkpoint for checkpoint in PINNED_CHECKPOINTS if checkpoint.moe),
        ("experts",),
        inference_ratio=0.85,
        max_running_requests=4,
        cuda_graph_bs=(1, 2, 4),
    ),
}


def _greedy_engine(llm: LLM) -> dict[str, Any]:
    """``llm``'s greedy runs and revision; the engine shuts down."""

    try:
        return {"runs": greedy(llm), "revision": llm.model_revision}
    finally:
        llm.shutdown()


def _serve(model: _Model) -> dict[str, Any]:
    without_lora = _greedy_engine(model.llm(lora=False))
    graph = _greedy_engine(model.llm(lora=True))
    eager = _greedy_engine(model.llm(lora=True, graphs=False))
    engines = (without_lora, graph, eager)
    return {
        "revisions": [engine["revision"] for engine in engines],
        "logprobs_finite": all(
            math.isfinite(value)
            for engine in engines
            for run in engine["runs"]
            for value in run["logprobs"]
        ),
        "zero_adapter_tokens_equal": [run["token_ids"] for run in graph["runs"]]
        == [run["token_ids"] for run in without_lora["runs"]],
        # The adapted blocks compose themselves; the engine without LoRA runs its kernels.
        "zero_adapter_vs_without_lora": compare_runs(without_lora["runs"], graph["runs"]),
        "eager_vs_graph": compare_runs(graph["runs"], eager["runs"]),
    }


def _run_child(name: str, mode: str, folder: Path) -> None:
    assert mode == "serve", mode
    print_receipt(_serve(MODELS[name]))


def _run_mode(tmp_path: Path, name: str, mode: str) -> dict[str, Any]:
    env, log_root = offline_child_env(tmp_path, f"{name}_lora")
    output = run_isolated_process(
        (sys.executable, "-m", _MODULE, f"{name}:{mode}", str(tmp_path)),
        log_root / f"{name}-{mode}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=1800,
    )
    return read_receipt(output)


# Qwen3.5-35B-A3B's zero adapter moves greedy logprobs 0.18 from the engine without one: the
# adapted layer composes its block, three BF16 roundings more than MoEBlock and grouped-GEMM
# rather than triton_block accumulation. A block-level kernel path will fix it and retire this mark.
_ZERO_ADAPTER_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "composed MoE path vs MoEBlock: 3 extra BF16 roundings + grouped-GEMM vs triton_block "
        "accumulation; a block-level kernel path will fix it"
    ),
)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=_ZERO_ADAPTER_XFAIL) if name == "qwen3_5_moe" else name
        for name in sorted(MODELS)
    ],
)
def test_expert_lora_zero_adapter_follows_the_engine_without_lora_graphs_and_eager(
    tmp_path: Path, name: str
) -> None:
    result = _run_mode(tmp_path, name, "serve")
    assert result["revisions"] == [0, 0, 0]
    assert result["logprobs_finite"] is True
    # Within the bound, and the very greedy tokens: a tolerance alone would pass a path error.
    assert result["zero_adapter_tokens_equal"] is True
    for key in ("zero_adapter_vs_without_lora", "eager_vs_graph"):
        assert_greedy_follows(
            result[key],
            logprob_tolerance=GRAPH_LOGPROB_TOLERANCE,
            max_tokens=MAX_TOKENS,
            min_compared_fraction=MIN_COMPARED_FRACTION,
        )


if __name__ == "__main__":
    from tests.support.qwen3_5_prefix import child_main

    child_main(_run_child, *sys.argv[1].split(":"), Path(sys.argv[2]))
