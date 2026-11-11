"""Hybrid radix decisions on two spawned ranks with real gloo groups (``gloo_ranks``).

Each rank replays the same requests through its own scheduler, cache manager and joint
radix, with too few pages to keep every prefix. Concurrent siblings on two snapshot rows
find no free row for some exports and reuse rows; multi-turn conversations evict KV and
snapshots and then hit at whatever depth survived. The ranks read different wall clocks,
one frozen so every reading ties; both must plan alike.
"""

from __future__ import annotations

import time
from itertools import count
from pathlib import Path

import pytest

from cti.distributed.runtime import ParallelGroup
from tests.support.gloo_ranks import gloo_ranks
from tests.support.inference.prefix_cache import planning_trace

# Gloo ranks are processes of their own: one such test at a time keeps them from starving.
pytestmark = pytest.mark.xdist_group("gloo")

_ALIGNMENT = 64  # the snapshot grid of the trace's one-token pages


def _prompt(group: int, length: int) -> list[int]:
    return [group * 1000 + position for position in range(length)]


def _multi_turn_phases() -> list[list[tuple[list[int], int]]]:
    """Four 129-token prompts decoding to 192, one after another; the first again; a fifth
    that evicts; then each conversation's next turn, which reads the zero outputs."""
    turn = [(_prompt(group, 129) + [0] * 63 + _prompt(group + 10, 10), 1) for group in (1, 2, 3, 0)]
    return [
        *([(_prompt(group, 129), 64)] for group in range(4)),
        [(_prompt(0, 129), 1)],
        [(_prompt(4, 129), 64)],
        *([request] for request in turn),
    ]


# Each workload's phases (the trace's default siblings without) and snapshot rows.
_WORKLOADS = {"siblings": (None, 2), "turns": (_multi_turn_phases(), 8)}


def _rank(group: ParallelGroup, argument: tuple[str, tuple[int, ...]]) -> dict:
    workload, pages = argument
    phases, rows = _WORKLOADS[workload]
    ticks = count()
    # The rank process exits after the scenario, so its clock is never restored.
    time.monotonic_ns = (lambda: 1) if group.rank == 0 else (lambda: next(ticks) * 1000)

    trace, evicted = planning_trace(pages[group.rank], phases=phases, rows=rows)
    return {"trace": trace, "evicted": evicted}


def _run(tmp_path: Path, workload: str, pages: tuple[int, int]) -> list[dict]:
    with gloo_ranks(_rank, 2, (workload, pages), tmp_path) as results:
        return results


def _restores(trace: list) -> list[int]:
    return [start for step in trace for _, start, _, row, _ in step if row is not None]


def test_ranks_reuse_rows_and_skip_exports_alike(tmp_path: Path) -> None:
    primary, peer = _run(tmp_path, "siblings", (640, 640))

    assert primary == peer
    trace = primary["trace"]
    assert primary["evicted"] > 0 and _restores(trace)
    rows = [row for step in trace for entry in step for _, row in entry[4]]
    assert len(rows) > len(set(rows)), "snapshot rows are reused"
    # A prompt forward that crosses a snapshot point while no row is free exports nothing.
    assert any(
        not captures and start // _ALIGNMENT < end // _ALIGNMENT
        for step in trace
        for _, start, end, _, captures in step
    )


def test_ranks_evict_and_hit_alike_across_turns(tmp_path: Path) -> None:
    primary, peer = _run(tmp_path, "turns", (880, 880))

    assert primary == peer
    # The first prompt's repeat and the oldest conversation's turn hit at the prompt's
    # snapshot; the fifth prompt evicted the rest in LRU order. A frozen wall clock once
    # left another conversation's decode snapshot to its rank alone.
    assert primary["evicted"] > 0
    assert _restores(primary["trace"]) == [128, 128]
