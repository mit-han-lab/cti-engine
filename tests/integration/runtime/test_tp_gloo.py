"""Dense Qwen3 across spawned ranks with real gloo groups on CPU (``gloo_ranks``)."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import safetensors.torch
import torch
from transformers import Qwen3Config

from cti.distributed.runtime import ParallelGroup
from cti.inference._execution.runner import InferenceRunner
from cti.protocol import AbortBackendMsg, BaseBackendMsg, SamplingParams, UserMsg
from cti.runtime._control import ControlChannel, ControlRecord
from tests.support.gloo_ranks import gloo_ranks
from tests.support.qwen3_5_engine import (
    assert_ranks_match_single_rank,
    bind_reference_kernels,
    run_steps,
)
from tests.support.tp_shards import qwen3_checkpoint_shapes

# Gloo ranks are processes of their own: one such test at a time keeps them from starving.
pytestmark = pytest.mark.xdist_group("gloo")

# Four query and two KV heads: TP4 replicates each KV head on two ranks. The
# vocabulary leaves a short last shard at TP2 and TP4.
_CONFIG = Qwen3Config(
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    hidden_size=64,
    vocab_size=259,
    intermediate_size=128,
    rms_norm_eps=1e-6,
    max_position_embeddings=64,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
    tie_word_embeddings=False,
    architectures=["Qwen3ForCausalLM"],
)


def _write_checkpoint(folder: Path) -> None:
    """Random Hugging Face-named weights that keep activations near unit scale."""

    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for name, shape in qwen3_checkpoint_shapes(_CONFIG).items():
        noise = torch.randn(shape, generator=generator)
        if name.endswith("norm.weight"):
            tensors[name] = 1 + 0.1 * noise
        elif name == "model.embed_tokens.weight":
            tensors[name] = noise
        else:
            tensors[name] = noise * shape[-1] ** -0.5
    safetensors.torch.save_file(tensors, folder / "model.safetensors")
    _CONFIG.save_pretrained(folder)


def _collectives(group: Any) -> dict[str, object]:
    reduced = group.all_reduce(torch.arange(6, dtype=torch.float32).view(2, 3) + 10 * group.rank)
    gathered = group.all_gather(torch.full((1, 3), float(group.rank)))
    published = torch.full((2,), float(group.rank) + 0.5)
    group.broadcast_device(published)
    # Odd byte counts and offsets: three bytes, and three at an odd offset of a row.
    odd = torch.arange(3, dtype=torch.uint8) + 10 * group.rank
    group.broadcast_device(odd)
    row = torch.arange(6, dtype=torch.uint8) + 10 * group.rank
    group.broadcast_device(row[1:4])
    header = torch.tensor([group.rank, 7], dtype=torch.int64)
    group.broadcast_cpu(header)
    group.barrier()
    group.require_same(7, "an agreed value")
    try:
        group.require_same(group.rank, "a rank's own value")
        divergence = None
    except RuntimeError as error:
        divergence = str(error)
    return {
        "all_reduce": reduced.tolist(),
        "all_gather": gathered.tolist(),
        "broadcast": published.tolist(),
        "broadcast_bytes": [odd.tolist(), row.tolist()],
        "broadcast_cpu": header.tolist(),
        "min_max_int": group.min_max_int(100 + group.rank),
        "require_same": divergence,
    }


def _request(uid: int, *token_ids: int) -> UserMsg:
    return UserMsg(
        uid,
        torch.tensor(token_ids, dtype=torch.int32),
        SamplingParams(temperature=0, max_new_tokens=2).snapshot(),
    )


def _describe(message: BaseBackendMsg) -> tuple[object, ...]:
    if isinstance(message, UserMsg):
        return ("user", message.uid, message.input_ids.tolist(), message.sampling)
    return (type(message).__name__, message.uid)


_RECORDS = (
    ControlRecord(),
    ControlRecord(requests=(_request(1, 4, 5),)),
    ControlRecord(("command",), (_request(2, 6), AbortBackendMsg(1))),
)


def _control_records(group: Any, channels: tuple[ControlChannel, ...]) -> list[object]:
    """Rank 0 relays its records; every rank reports the records it applies, in order."""

    channel = channels[group.rank]
    records = []
    for sent in _RECORDS:
        if group.rank == 0:
            channel.send(sent)
        record = sent if group.rank == 0 else channel.receive()
        records.append((record.commands, [_describe(request) for request in record.requests]))
    return records


@dataclass(frozen=True)
class _Sampled:
    """A sampler's rows, with the fields the runner broadcasts, packed as the sampler's
    results pack: each present column's bytes, tokens first."""

    token_ids: torch.Tensor
    logprobs: torch.Tensor | None = None
    top_ids: torch.Tensor | None = None
    top_logprobs: torch.Tensor | None = None

    def packed(self) -> torch.Tensor:
        columns = (self.token_ids, self.logprobs, self.top_ids, self.top_logprobs)
        return torch.cat([c.reshape(-1).view(torch.uint8) for c in columns if c is not None])


def _sampling(group: Any) -> dict[str, object]:
    calls = []

    class Sampler:
        def sample(self, logits: torch.Tensor, args: object) -> _Sampled:
            calls.append("sample")
            if args.logprobs is None:
                return _Sampled(torch.tensor([5, 7]))
            top_ids = torch.tensor([[5, 1], [7, 2]], dtype=torch.int32)
            top = torch.tensor([[-0.5, -2.0], [-1.25, -3.0]])
            return _Sampled(torch.tensor([5, 7]), torch.tensor([-0.5, -1.25]), top_ids, top)

    # The parts a sampling step reads; the rest of the step path is not run here.
    parts = SimpleNamespace(
        model=SimpleNamespace(rope_position_axes=lambda: 1),
        preparation_stream=None,
        execution_stream=None,
        page_table=None,
        token_id_table=torch.zeros(1, 1, dtype=torch.int32),
        sampler=Sampler(),
        participants=(),
        recurrent=None,
        graphs=None,
        speculative=None,
        round_graphs=None,
        limits=SimpleNamespace(rounds=None),
        max_rows=2,
    )
    runner = InferenceRunner(parts, ranks=group)
    logits = torch.zeros(2, 4)
    full = runner._sample(logits, SimpleNamespace(logprobs=SimpleNamespace(top=2)))
    greedy = runner._sample(logits, SimpleNamespace(logprobs=None))
    return {
        "tokens": (full.token_ids.dtype, full.token_ids.tolist()),
        "logprobs": (full.logprobs.dtype, full.logprobs.tolist()),
        "top": (full.top_ids.tolist(), full.top_logprobs.tolist()),
        "greedy": greedy.token_ids.tolist(),
        "absent_logprobs": greedy.logprobs is None,
        "calls": calls,
    }


def _rank_scenarios(
    group: ParallelGroup, argument: tuple[Path, tuple[ControlChannel, ...]]
) -> dict[str, object]:
    folder, channels = argument
    # The rank process exits after the scenario, so its bindings are never undone.
    bind_reference_kernels(pytest.MonkeyPatch())
    return {
        "backends": group.backends,
        "collectives": _collectives(group),
        "logits": [logits.tolist() for logits in run_steps(folder, group=group)[1]],
        "records": _control_records(group, channels),
        "sampling": _sampling(group),
    }


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("qwen3-tp")
    _write_checkpoint(folder)
    return folder


@pytest.fixture(scope="module")
def reference_logits(checkpoint: Path) -> list[torch.Tensor]:
    """The single-rank model without any process group."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        bind_reference_kernels(monkeypatch)
        return run_steps(checkpoint)[1]


@pytest.fixture(scope="module", params=[2, 4], ids=["tp2", "tp4"])
def ranks(request: pytest.FixtureRequest, checkpoint: Path, tmp_path_factory) -> Iterator[list]:
    rendezvous = tmp_path_factory.mktemp("rendezvous")
    # Rank 0 writes one pipe per peer, as the rank supervisor wires them.
    pipes = [mp.get_context("spawn").Pipe(duplex=False) for _ in range(1, request.param)]
    channels = (ControlChannel(tuple(send for _, send in pipes)),)
    channels += tuple(ControlChannel((receive,)) for receive, _ in pipes)
    try:
        with gloo_ranks(_rank_scenarios, request.param, (checkpoint, channels), rendezvous) as ran:
            yield ran
    finally:
        for connection in (end for pipe in pipes for end in pipe):
            connection.close()


def test_ranks_build_cpu_and_device_groups_over_gloo(ranks: list[dict]) -> None:
    assert all(result["backends"] == ("gloo", "gloo") for result in ranks)


def test_parallel_runtime_collectives_agree_on_every_rank(ranks: list[dict]) -> None:
    size = len(ranks)
    reduced = (
        torch.arange(6, dtype=torch.float32).view(2, 3) * size + 10 * sum(range(size))
    ).tolist()
    for rank, result in enumerate(ranks):
        collectives = result["collectives"]
        assert collectives["all_reduce"] == reduced
        assert collectives["all_gather"] == [[float(rank)] * 3 for rank in range(size)]
        assert collectives["broadcast"] == [0.5, 0.5]
        # Only the view's bytes take the root's values.
        row = [10 * rank, 1, 2, 3, 10 * rank + 4, 10 * rank + 5]
        assert collectives["broadcast_bytes"] == [[0, 1, 2], row]
        assert collectives["broadcast_cpu"] == [0, 7]
        assert collectives["min_max_int"] == (100, 100 + size - 1)
        # Every rank fails when any rank passed another value.
        assert collectives["require_same"] == "tensor-parallel ranks diverged at a rank's own value"


def test_sharded_forward_matches_the_single_rank_model(
    ranks: list[dict], reference_logits: list[torch.Tensor]
) -> None:
    assert all(step.shape == (2, _CONFIG.vocab_size) for step in reference_logits)
    assert_ranks_match_single_rank([result["logits"] for result in ranks], reference_logits)


def test_every_rank_applies_rank_0_s_records_in_order(ranks: list[dict]) -> None:
    sampling = SamplingParams(temperature=0, max_new_tokens=2).snapshot()
    expected = [
        ((), []),
        ((), [("user", 1, [4, 5], sampling)]),
        (("command",), [("user", 2, [6], sampling), ("AbortBackendMsg", 1)]),
    ]
    assert all(result["records"] == expected for result in ranks)


def test_only_the_primary_samples_and_every_rank_receives_its_choice(ranks: list[dict]) -> None:
    for rank, result in enumerate(ranks):
        sampling = result["sampling"]
        assert sampling["tokens"] == (torch.int32, [5, 7])
        assert sampling["logprobs"] == (torch.float32, [-0.5, -1.25])
        assert sampling["top"] == ([[5, 1], [7, 2]], [[-0.5, -2.0], [-1.25, -3.0]])
        assert sampling["greedy"] == [5, 7]
        assert sampling["absent_logprobs"]
        assert sampling["calls"] == (["sample", "sample"] if rank == 0 else [])
