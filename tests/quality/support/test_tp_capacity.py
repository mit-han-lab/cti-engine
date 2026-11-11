"""The tensor-parallel test's capacity recorder and check, on CPU.

Two ranks run the real sizing steps (``DeviceMemory.peak_bytes``,
``DeviceMemory.plan``) under the recorder, one after the
other: ``ParallelGroup.min_max_int`` combines a rank's value with its peer's, in the order
the steps call it, and the CUDA memory statistics are stand-ins.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch

from cti import kvcache
from cti.attention.paged import KVPair
from cti.config import MemoryOptions
from cti.distributed.runtime import ParallelGroup
from cti.kvcache import (
    GroupLayout,
    GroupRole,
    KVLayout,
    RecurrentRows,
    SnapshotPolicy,
    Spec,
    TableGeometry,
)
from cti.layers.gated_delta_net.layer import gated_delta_net_state
from cti.worker import DeviceMemory
from tests.support.tp_capacity import assert_published_capacity, install_capacity_recorder

_BOOT_FREE_BYTES = 10_000
_INFERENCE_BUDGET_BYTES = 8_000
_PEAK_RESERVE_BYTES = 2_000
# Each rank's model use and image-encoder peak. Ranks have 5000 and 4600 bytes available;
# the larger peak, 3000, leaves 1000 bytes beyond the reserve.
_RANKS = ((3000, 2600), (3400, 3000))
_MAX_RUNNING_REQ = 5
# Six bytes per request: one FP32 state element and one BF16 history element; 30 in all.
_RECURRENT_SPEC = gated_delta_net_state(1, 1, 1, 1, conv_history=1)
# K and V of one BF16 element per page.
_PAGE_BYTES = 4
# The steps the recorder wraps, and the collective it records.
_STEPS = (
    (DeviceMemory, "peak_bytes"),
    (DeviceMemory, "plan"),
    (kvcache, "plan_capacity"),
    (ParallelGroup, "min_max_int"),
)


class _Ranks:
    """The running rank, and both ranks' values in each collective it runs, in order."""

    def __init__(self) -> None:
        self.rank = 0
        self.collectives: list[tuple[int, int]] = []
        # The attention workspace every rank reserves for its own heads.
        self.workspace_bytes = 0


@pytest.fixture
def ranks(monkeypatch: pytest.MonkeyPatch) -> _Ranks:
    """Two-rank collectives and CPU stand-ins for CUDA memory statistics and the attention
    workspace; the sizing steps are restored afterwards."""

    for owner, name in _STEPS:
        monkeypatch.setattr(owner, name, owner.__dict__[name])
    simulated = _Ranks()

    def min_max_int(self: ParallelGroup, value: int) -> tuple[int, int]:
        values = simulated.collectives.pop(0)
        assert value == values[simulated.rank]
        return min(values), max(values)

    monkeypatch.setattr(ParallelGroup, "min_max_int", min_max_int)
    monkeypatch.setattr(ParallelGroup, "size", property(lambda _group: len(_RANKS)))
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *_args: None)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (_BOOT_FREE_BYTES - _RANKS[simulated.rank][0], _BOOT_FREE_BYTES),
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 0)
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda _device: _RANKS[simulated.rank][1],
    )
    return simulated


def _rank() -> DeviceMemory:
    """A rank's device memory, as it stands once the models are built."""

    group = object.__new__(ParallelGroup)
    memory = object.__new__(DeviceMemory)
    memory._device, memory._manager, memory._unified = torch.device("cpu"), None, False
    memory._agree = group.min_max_int
    memory._boot_local_bytes = _BOOT_FREE_BYTES
    memory.budget_bytes, memory.reserve_bytes = _INFERENCE_BUDGET_BYTES, _PEAK_RESERVE_BYTES
    memory._recurrent_to_kv_ratio = MemoryOptions().recurrent_to_kv_ratio
    return memory


@contextmanager
def _rank_process(evidence: dict[str, Any]) -> Iterator[None]:
    """One rank's engine process: the recorder wraps the real sizing steps."""

    steps = {(owner, name): owner.__dict__[name] for owner, name in _STEPS}
    install_capacity_recorder(evidence)
    try:
        yield
    finally:
        for (owner, name), step in steps.items():
            setattr(owner, name, step)


def _size_two_ranks(
    ranks: _Ranks, *, encodes_images: bool, reuses_prefixes: bool = False
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Each rank's KV pages and snapshot rows and its JSON evidence, sized as
    ``build_executor`` sizes them."""

    snapshots = SnapshotPolicy() if reuses_prefixes else None
    spec = Spec(KVPair(num_kv_heads=1, head_dim=1, dtype=torch.bfloat16))
    layout = KVLayout(
        (GroupLayout(GroupRole.TARGET, 0, spec, 1),), RecurrentRows((_RECURRENT_SPEC,))
    )
    evidence: list[dict[str, Any]] = [{} for _ in _RANKS]
    pages = []
    for rank in range(len(_RANKS)):
        ranks.rank = rank
        with _rank_process(evidence[rank]):
            # The worker binds the ranks' vote once the process starts, under the recorder.
            memory = _rank()
            # Without an image encoder the step returns before its collective.
            ranks.collectives = [tuple(peak for _, peak in _RANKS)] if encodes_images else []
            # The executor measures the largest image step's encoder through the budget.
            image_encoder_peak_bytes = memory.peak_bytes(lambda: encodes_images)
            assert not ranks.collectives
            # DeviceMemory._free's free memory, then the memory available for inference
            # after the attention workspace.
            ranks.collectives = [
                tuple(_BOOT_FREE_BYTES - model_bytes for model_bytes, _ in _RANKS),
                tuple(
                    _INFERENCE_BUDGET_BYTES - model_bytes - ranks.workspace_bytes
                    for model_bytes, _ in _RANKS
                ),
            ]
            plan = memory.plan(
                layout,
                geometry=TableGeometry(1, _MAX_RUNNING_REQ, 16),
                snapshots=snapshots,
                workspace_bytes=ranks.workspace_bytes,
                image_encoder_bytes=image_encoder_peak_bytes,
                image_rows_bytes=0,
                beside_bytes=0,
                allocations=None,
                fit_slots=False,
            )
            pages.append((plan.num_pages, len(plan.snapshot_rows)))
            assert not ranks.collectives
    # The parent reads each rank's evidence back from its JSON file.
    return pages, [json.loads(json.dumps(record)) for record in evidence]


def test_ranks_record_the_image_encoder_peak_and_publish_the_pages_it_leaves(
    ranks: _Ranks,
) -> None:
    pages, evidence = _size_two_ranks(ranks, encodes_images=True)

    # 4600 bytes available on both ranks, less 30 of recurrent state and the 1000 bytes of
    # the largest peak beyond the reserve: 3570 bytes, 892 pages, one of them the sink page.
    assert pages == [(891, 0), (891, 0)]
    assert [record["image_encoder"] for record in evidence] == [
        [[2600, [2600, 3000]]],
        [[3000, [2600, 3000]]],
    ]
    # The free memory's collective first, then the available memory's.
    assert [record["capacity"] for record in evidence] == [
        [[7000, [6600, 7000]], [5000, [4600, 5000]]],
        [[6600, [6600, 7000]], [4600, [4600, 5000]]],
    ]
    for record in evidence:
        assert record["image_encoder_peak_bytes"] == 3000
        assert record["peak_reserve_bytes"] == _PEAK_RESERVE_BYTES
        assert record["recurrent_bytes"] == 30
        assert record["plan"] == [4600, 891, 0]
    assert_published_capacity(evidence, 891, 0)
    # Pages that leave the image encoder's excess to the KV cache do not pass.
    with pytest.raises(AssertionError):
        assert_published_capacity(evidence, (4600 - 30) // _PAGE_BYTES - 1, 0)


def test_ranks_reserve_the_attention_workspace_before_the_collective(ranks: _Ranks) -> None:
    ranks.workspace_bytes = 20

    pages, evidence = _size_two_ranks(ranks, encodes_images=True)

    # 4580 bytes available on both ranks after the 20-byte workspace, less 30 of recurrent
    # state and the 1000 bytes of the largest peak beyond the reserve: 3550 bytes, 887
    # pages, one of them the sink page.
    assert pages == [(886, 0), (886, 0)]
    assert [record["capacity"][-1] for record in evidence] == [
        [4980, [4580, 4980]],
        [4580, [4580, 4980]],
    ]
    assert_published_capacity(evidence, 886, 0)


def test_ranks_without_an_image_encoder_record_no_peak(ranks: _Ranks) -> None:
    pages, evidence = _size_two_ranks(ranks, encodes_images=False)

    assert pages == [((4600 - 30) // _PAGE_BYTES - 1, 0)] * 2
    for record in evidence:
        assert record["image_encoder"] == []
        assert record["image_encoder_peak_bytes"] == 0
    assert_published_capacity(evidence, *pages[0])


def test_radix_ranks_publish_the_snapshot_rows_they_planned(ranks: _Ranks) -> None:
    pages, evidence = _size_two_ranks(ranks, encodes_images=True, reuses_prefixes=True)

    # 3600 bytes remain beside the largest peak; their recurrent share, 0.9 / 1.9 of them,
    # is 1705 bytes, a quarter of which holds 71 six-byte rows. Past 30 bytes of live rows,
    # the 426 of snapshots and 160 bytes of copy indices, 2984 bytes hold 746 pages, one of
    # them the graph sink.
    assert pages == [(745, 71), (745, 71)]
    assert_published_capacity(evidence, 745, 71)
    with pytest.raises(AssertionError):
        assert_published_capacity(evidence, 745, 70)


def _parameters(method: Callable[..., object]) -> list[tuple[str, Any]]:
    return [(p.name, p.kind) for p in inspect.signature(method).parameters.values()]


@pytest.mark.usefixtures("ranks")
def test_the_recorder_takes_every_argument_of_the_steps_it_wraps() -> None:
    """A sizing argument the recorder lacks fails every auto-capacity case on the GPU."""

    steps = {(owner, name): owner.__dict__[name] for owner, name in _STEPS}

    with _rank_process({}):
        for (owner, name), step in steps.items():
            recorder = owner.__dict__[name]
            assert recorder is not step
            assert _parameters(recorder) == _parameters(step), name
