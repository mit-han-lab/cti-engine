"""A switch's residency vote across spawned ranks over real gloo groups (``gloo_ranks``):
when one rank's device cannot back a resume, every rank refuses it before any maps."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from cti.distributed.runtime import ParallelGroup
from cti.memory import MemoryManager, Mode, ResidencyRestoreError
from cti.worker import GPUWorker
from tests.support.gloo_ranks import gloo_ranks

# Gloo ranks are processes of their own: one such test at a time keeps them from starving.
pytestmark = pytest.mark.xdist_group("gloo")

_KV_BYTES = 64


class _Provider:
    """Maps nothing; records what it would resume."""

    def __init__(self) -> None:
        self.resumed: list[str] = []

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False) -> Iterator[None]:
        yield

    def pause(self, tag: str) -> None:
        pass

    def resume(self, tag: str) -> None:
        self.resumed.append(tag)


def _acquire_serving(group: ParallelGroup, short_rank: int | None) -> dict[str, object]:
    """Rank ``short_rank``'s device keeps one byte too few for the KV; the others fit."""

    reserved = [0]
    monkeypatch = pytest.MonkeyPatch()  # the rank process exits after its scenario
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_args: reserved[0])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    free = _KV_BYTES - 1 if group.rank == short_rank else 1 << 30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_args: (free, 1 << 31))
    provider = _Provider()
    memory = MemoryManager(provider, unified=False)
    with memory.kv_allocation():
        reserved[0] += _KV_BYTES
    memory.suspend_kv()
    worker = object.__new__(GPUWorker)
    worker._memory_manager = memory
    worker.ranks = group
    worker._vote = group.min_max_int
    try:
        worker.acquire(Mode.SERVING)
    except ResidencyRestoreError as error:
        refusal = (error.required_bytes, error.free_bytes)
    else:
        refusal = None
    return {"refusal": refusal, "resumed": provider.resumed}


def _rank_scenarios(group: ParallelGroup, _argument: None) -> dict[str, object]:
    return {
        "peer-short": _acquire_serving(group, short_rank=1),
        "all-fit": _acquire_serving(group, short_rank=None),
    }


@pytest.fixture(scope="module")
def ranks(tmp_path_factory: pytest.TempPathFactory) -> Iterator[list[dict]]:
    rendezvous: Path = tmp_path_factory.mktemp("rendezvous")
    with gloo_ranks(_rank_scenarios, 2, None, rendezvous) as ran:
        yield ran


def test_one_rank_that_does_not_fit_makes_every_rank_refuse(ranks: list[dict]) -> None:
    # Each rank reports the tightest rank's free bytes; no rank maps anything.
    assert [result["peer-short"] for result in ranks] == [
        {"refusal": (_KV_BYTES, _KV_BYTES - 1), "resumed": []}
    ] * 2


def test_every_rank_resumes_when_all_fit(ranks: list[dict]) -> None:
    assert [result["all-fit"] for result in ranks] == [
        {"refusal": None, "resumed": ["kv-cache"]}
    ] * 2
