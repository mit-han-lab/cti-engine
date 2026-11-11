"""Qwen3-MoE over real processes and gloo groups on CPU: tensor-parallel attention and
expert-parallel sparse blocks.

Tiny FP32 checkpoints with eight experts, two per token, run a chunked prefill, a mixed batch
and decode steps over the runner's inference state, with the CPU reference operations at every
kernel seam. TP2 and TP4, with whole experts split over the ranks (expert parallel) or every
expert's width split (expert tensor parallel), must give the single-rank model's logits at
every step, with renormalized routes and an unbiased attention (the published form) and with
raw routing probabilities and projection biases.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig

from cti.distributed import Topology
from cti.distributed.runtime import ParallelGroup
from cti.layers import ExpertParallel, ExpertPlacement, ExpertTensorParallel
from tests.support.gloo_ranks import gloo_ranks
from tests.support.moe import ReferenceMoe
from tests.support.qwen3_5_engine import (
    STEPS,
    TP_TOLERANCE,
    EagerEngine,
    bind_reference_kernels,
)
from tests.support.qwen3_moe import save_tiny_checkpoint
from tests.support.tp_shards import load_sharded

# Gloo ranks are processes of their own: one such test at a time keeps them from starving.
pytestmark = pytest.mark.xdist_group("gloo")

# Four query and two KV heads, so TP4 replicates each KV head on two ranks; the vocabulary
# leaves padded rows at both sizes.
_GEOMETRY = {"num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 67}
_CHECKPOINTS = {
    "renormalized": {},
    "raw-probabilities-biased": {"norm_topk_prob": False, "attention_bias": True},
}


class _CountingMoe(ReferenceMoe):
    """``ReferenceMoe`` counting the rows whose every route goes to another rank's experts."""

    def __init__(self) -> None:
        self.remote_rows = 0

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        expert_map = kwargs["expert_map"]
        if expert_map is not None:
            logits = F.linear(hidden_states.float(), kwargs["router"].float())
            ids = logits.topk(kwargs["topk"], dim=-1).indices
            self.remote_rows += int((expert_map[ids] < 0).all(-1).sum())
        return super().forward(hidden_states, *args, **kwargs)


def _run(folder: Path, group: ParallelGroup | None, placement: ExpertPlacement) -> dict[str, Any]:
    """Every step's logits, the expert share this rank holds and its all-remote rows."""

    backend = _CountingMoe()
    model = load_sharded(
        AutoConfig.from_pretrained(folder),
        folder,
        group=group,
        moe_backend=backend,
        experts=placement,
    )[0]
    generator = torch.Generator().manual_seed(2)
    token_ids = torch.randint(
        _GEOMETRY["vocab_size"], (2, 24), generator=generator, dtype=torch.int32
    )
    engine = EagerEngine(model, token_ids)
    with torch.inference_mode():
        logits = [engine.execute(*entries) for entries in STEPS]
    experts = [layer.mlp.experts for layer in model.model.layers.op_list]
    # Tensors would cross the Pipe as shared memory owned by an exiting rank.
    return {
        "logits": [step.tolist() for step in logits],
        "experts": [e.placement for e in experts],
        "remote_rows": backend.remote_rows,
    }


def _rank(group: ParallelGroup, argument: tuple[Path, ExpertPlacement]) -> dict[str, Any]:
    # The rank process exits after the scenario, so its bindings are never undone.
    bind_reference_kernels(pytest.MonkeyPatch())
    folder, placement = argument
    return _run(folder, group, placement)


@pytest.fixture(scope="module", params=sorted(_CHECKPOINTS))
def checkpoint(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp(request.param)
    save_tiny_checkpoint(folder, **_GEOMETRY, **_CHECKPOINTS[request.param])
    return folder


@pytest.fixture(scope="module")
def single_rank(checkpoint: Path) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        bind_reference_kernels(monkeypatch)
        return _run(checkpoint, None, ExpertParallel())


_PLACEMENTS = {"ep": ExpertParallel(), "etp": ExpertTensorParallel()}


@pytest.fixture(
    scope="module",
    params=[(size, name) for name in _PLACEMENTS for size in (2, 4)],
    ids=lambda param: f"tp{param[0]}-{param[1]}",
)
def ranks(
    request: pytest.FixtureRequest, checkpoint: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[ExpertPlacement, list[dict[str, Any]]]]:
    size, name = request.param
    rendezvous = tmp_path_factory.mktemp("rendezvous")
    placement = _PLACEMENTS[name]
    with gloo_ranks(_rank, size, (checkpoint, placement), rendezvous) as results:
        yield placement, results


def test_tensor_and_expert_parallel_logits_match_the_single_rank_model(
    ranks: tuple[ExpertPlacement, list[dict[str, Any]]], single_rank: dict[str, Any]
) -> None:
    _, ranks = ranks
    assert all(result["logits"] == ranks[0]["logits"] for result in ranks)
    for step, (values, reference) in enumerate(
        zip(ranks[0]["logits"], single_rank["logits"], strict=True)
    ):
        logits, expected = torch.tensor(values), torch.tensor(reference)
        assert logits.shape == (2, _GEOMETRY["vocab_size"])
        assert torch.isfinite(logits).all()
        # A dropped exit reduction misses by O(1).
        torch.testing.assert_close(logits, expected, **TP_TOLERANCE, msg=f"step {step}")
        assert torch.equal(logits.argmax(-1), expected.argmax(-1)), f"step {step} greedy token"


def test_each_rank_holds_its_expert_share_and_only_expert_parallel_has_remote_rows(
    ranks: tuple[ExpertPlacement, list[dict[str, Any]]], single_rank: dict[str, Any]
) -> None:
    placement, ranks = ranks
    layers = len(single_rank["experts"])
    assert all(held.is_whole for held in single_rank["experts"])
    assert single_rank["remote_rows"] == 0
    for rank, result in enumerate(ranks):
        expected = placement.shard(
            8, single_rank["experts"][0].intermediate_size, Topology(rank, len(ranks))
        )
        assert result["experts"] == [expected] * layers
        # Rows routed only to other ranks' experts ran on this rank too, and the logits
        # above still match; expert tensor parallelism has none.
        assert (result["remote_rows"] > 0) == isinstance(placement, ExpertParallel)
