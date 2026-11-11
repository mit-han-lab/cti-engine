"""A rank's progress record and its parent's watch: a rank with work that finishes no loop turn
within its allowance has hung, whether stopped or stuck in a collective; an idle one never
has, and a mode's first work gets a longer allowance."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from cti.config import EngineOptions, EngineTuning, ModelOptions
from cti.memory import Mode
from cti.runtime._core import EngineCore
from cti.runtime._process import ChildProcess, ProgressWatch
from cti.runtime._protocol import RankProgress


def _watch(**ranks: RankProgress) -> ProgressWatch:
    return ProgressWatch(ranks, lambda name: name == "stopped")


def test_an_idle_rank_never_hangs() -> None:
    idle = RankProgress([3.0, 0.0])
    watch = _watch(idle=idle)

    assert [watch.hung(t) for t in (0.0, 100.0, 10_000.0)] == [None, None, None]


def test_a_rank_that_keeps_finishing_turns_never_hangs() -> None:
    busy = RankProgress([0.0, 0.0])
    busy.working(5.0)
    watch = _watch(busy=busy)

    for t in range(0, 60, 2):
        busy.finished()
        assert watch.hung(float(t)) is None


def test_a_rank_with_work_and_no_progress_hangs_past_its_allowance() -> None:
    stuck, stopped = RankProgress([7.0, 0.0]), RankProgress([2.0, 0.0])
    stuck.working(60.0)
    watch = _watch(stuck=stuck, stopped=stopped)

    assert watch.hung(0.0) is None
    assert watch.hung(60.0) is None
    # The stopped peer holds no work of its own: the stuck rank waits on it.
    assert (
        watch.hung(61.0) == "made no progress with work in hand: stuck for 61 s (stopped stopped)"
    )


def test_a_first_works_longer_allowance_holds() -> None:
    compiling = RankProgress([0.0, 0.0])
    compiling.working(600.0)
    watch = _watch(compiling=compiling)

    assert [watch.hung(t) for t in (0.0, 300.0, 600.0)] == [None, None, None]
    assert watch.hung(601.0) is not None


def test_it_checks_at_most_once_a_check_interval() -> None:
    stuck = RankProgress([0.0, 0.0])
    stuck.working(1.0)
    watch = _watch(stuck=stuck)

    assert watch.hung(0.0) is None
    assert watch.hung(5.0) is not None
    assert watch.hung(5.0 + ProgressWatch.CHECK_SECONDS / 2) is None


def _turns(progress: RankProgress, turns: int) -> None:
    progress.working(9.0)
    for _ in range(turns):
        progress.finished()


def test_a_spawned_rank_writes_the_record_its_parent_reads() -> None:
    context = mp.get_context("spawn")
    progress = RankProgress.shared(context)
    child = context.Process(target=_turns, args=(progress, 3))
    child.start()
    child.join(30.0)

    assert child.exitcode == 0
    assert (progress.steps, progress.allowance) == (3.0, 9.0)


def _wait(connection) -> None:
    connection.recv()


def _state_becomes(child: ChildProcess, suspended: bool) -> bool:
    deadline = time.monotonic() + 5.0
    while child.suspended() is not suspended:
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_a_child_stopped_by_a_signal_is_suspended_until_continued() -> None:
    context = mp.get_context("spawn")
    ours, theirs = context.Pipe()
    child = ChildProcess.start(context, _wait, (theirs,), name="sleeper", preload=nullcontext)
    pid = child._process.pid
    try:
        assert not child.suspended()
        os.kill(pid, signal.SIGSTOP)
        assert _state_becomes(child, True)
        os.kill(pid, signal.SIGCONT)
        assert _state_becomes(child, False)
    finally:
        child.stop()  # SIGKILL reclaims a stopped child too
    assert not child.suspended()


class _Side:
    def __init__(self, work: bool) -> None:
        self.work, self.steps = work, 0

    def wants_to_run(self) -> bool:
        return self.work

    def step(self) -> None:
        self.steps += 1


@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
def test_the_loop_marks_its_work_and_a_modes_first_work_gets_longer(busy: bool) -> None:
    progress = RankProgress([0.0, 0.0])
    core = EngineCore(
        EngineOptions(model=ModelOptions(path="model")),
        EngineTuning(distributed_timeout=60.0),
        None,  # type: ignore[arg-type]
        rank=0,
        init_method="unused",
        checkpoint=None,
        checks=False,
        overlap=False,
        online=False,
        progress=progress,
    )
    side = _Side(work=True)
    taken: list[float] = []

    def take(*, block: bool, until: object) -> SimpleNamespace:
        taken.append(progress.allowance)
        return SimpleNamespace(commands=(), requests=())

    core._switch = SimpleNamespace(
        busy=busy,
        completion=lambda: None,
        mode=Mode.SERVING,
        switching=False,
        admits_requests=True,
        advance=lambda: None,
        runs=lambda: side,
    )
    core._plane = SimpleNamespace(take=take)
    core._inference = SimpleNamespace(handle=lambda requests, admitting: None)

    allowances = []
    for _ in range(2):
        core._iterate()
        allowances.append(progress.allowance)

    # An idle rank waits for input with no work in hand; a busy one keeps its allowance.
    assert taken == ([0.0, 0.0] if not busy else [0.0, EngineCore.FIRST_WORK_SECONDS])
    assert allowances == [EngineCore.FIRST_WORK_SECONDS, 60.0]
    assert progress.steps == 2.0 and side.steps == 2
