from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from tests.support.engine_worker import EngineWorker
from tests.support.paths import REPO_ROOT

_CHILD = """
import os, sys, time
import torch

kind = sys.argv[1]
print("pid", os.getpid(), "seed", torch.initial_seed(), flush=True)
os.write(2, b"native stderr\\n")
if kind == "seed":
    torch.manual_seed(1)
if kind == "raise":
    raise RuntimeError("child failure")
if kind == "exit":
    sys.exit(3)
if kind == "sleep":
    time.sleep(60)
if kind == "forbidden":
    print("CUDA error: launch failed")
if kind == "finalizer":
    class Cycle:
        def __del__(self):
            print("Exception ignored in a finalizer", file=sys.stderr)

    cycle = Cycle()
    cycle.self = cycle
if kind == "atexit":
    import atexit

    atexit.register(print, "Traceback at interpreter shutdown")
if kind == "exit0":
    os._exit(0)
if kind == "orphan":
    import threading

    # A thread outlives the child and the worker's reply pipe closes: the worker cannot reply
    # and does not exit.
    threading.Thread(target=time.sleep, args=(60,)).start()
    os.close(int(sys.orig_argv[-1]))
if kind == "hang":
    import atexit

    atexit.register(time.sleep, 60)
"""


@pytest.fixture
def worker(tmp_path: Path):
    (tmp_path / "worker_child.py").write_text(_CHILD)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT)))}
    log_path = tmp_path / "worker.log"
    with EngineWorker(env=env, cwd=tmp_path, log_path=log_path, shutdown_seconds=2) as worker:
        yield worker


def _pid_and_seed(stdout: str) -> tuple[int, int]:
    _, pid, _, seed = stdout.split()
    return int(pid), int(seed)


def test_children_share_one_process_and_start_from_fresh_generators(worker, tmp_path) -> None:
    first = worker.run("worker_child", ("seed",), tmp_path / "first.log", timeout=60)
    second = worker.run("worker_child", ("ok",), tmp_path / "second.log", timeout=60)

    (first_pid, first_seed), (second_pid, second_seed) = map(_pid_and_seed, (first, second))
    assert first_pid == second_pid != os.getpid()
    # The first child seeded the CPU generator; the second starts from a seed of its own.
    assert second_seed not in (first_seed, 1)
    # Native writes to both streams are the child's, in run_isolated_process's log format.
    assert (tmp_path / "first.log").read_text() == f"stdout:\n{first}\nstderr:\nnative stderr\n"


@pytest.mark.parametrize(
    ("kind", "message"), [("raise", "failed with 1"), ("exit", "failed with 3")]
)
def test_a_failed_child_fails_the_test_and_ends_the_worker(worker, tmp_path, kind, message) -> None:
    pid, _ = _pid_and_seed(worker.run("worker_child", ("ok",), tmp_path / "ok.log", timeout=60))

    with pytest.raises(pytest.fail.Exception, match=message):
        worker.run("worker_child", (kind,), tmp_path / "failed.log", timeout=60)

    if kind == "raise":
        assert "RuntimeError: child failure" in (tmp_path / "failed.log").read_text()
    after = worker.run("worker_child", ("ok",), tmp_path / "after.log", timeout=60)
    assert _pid_and_seed(after)[0] != pid


def test_a_child_past_its_timeout_is_killed_with_its_output_kept(worker, tmp_path) -> None:
    with pytest.raises(pytest.fail.Exception, match="timed out after 2s"):
        worker.run("worker_child", ("sleep",), tmp_path / "sleep.log", timeout=2)

    assert "native stderr" in (tmp_path / "sleep.log").read_text()
    assert worker.run("worker_child", ("ok",), tmp_path / "after.log", timeout=60)


def test_a_child_that_ends_its_worker_cleanly_passes_and_the_next_gets_another(
    worker, tmp_path
) -> None:
    pid, _ = _pid_and_seed(worker.run("worker_child", ("exit0",), tmp_path / "a.log", timeout=60))
    after = worker.run("worker_child", ("ok",), tmp_path / "after.log", timeout=60)
    assert _pid_and_seed(after)[0] != pid


def test_a_worker_left_without_a_reply_is_killed_in_bounded_time(worker, tmp_path) -> None:
    with pytest.raises(pytest.fail.Exception, match="did not exit in time"):
        worker.run("worker_child", ("orphan",), tmp_path / "orphan.log", timeout=60)

    assert worker.run("worker_child", ("ok",), tmp_path / "after.log", timeout=60)


@pytest.mark.parametrize(
    ("kind", "pattern"), [("forbidden", "cuda error"), ("finalizer", "exception ignored")]
)
def test_a_forbidden_log_pattern_fails_the_test_and_ends_the_worker(
    worker, tmp_path, kind, pattern
) -> None:
    pid, _ = _pid_and_seed(worker.run("worker_child", ("ok",), tmp_path / "ok.log", timeout=60))

    # A finalizer of the child's objects writes into the child's log, as at its exit.
    with pytest.raises(pytest.fail.Exception, match=pattern):
        worker.run("worker_child", (kind,), tmp_path / "forbidden.log", timeout=60)

    after = worker.run("worker_child", ("ok",), tmp_path / "after.log", timeout=60)
    assert _pid_and_seed(after)[0] != pid


@pytest.mark.parametrize(("kind", "message"), [("atexit", "traceback"), ("hang", "in time")])
def test_the_worker_shutdown_passes_the_same_checks(worker, tmp_path, kind, message) -> None:
    worker.run("worker_child", (kind,), tmp_path / "child.log", timeout=60)

    with pytest.raises(pytest.fail.Exception, match=message):
        worker.close()


def test_a_worker_that_ended_between_children_is_reported_and_replaced(worker, tmp_path) -> None:
    pid, _ = _pid_and_seed(worker.run("worker_child", ("ok",), tmp_path / "ok.log", timeout=60))
    os.kill(pid, signal.SIGKILL)
    time.sleep(1)

    with pytest.raises(pytest.fail.Exception, match="exited with -9"):
        worker.run("worker_child", ("ok",), tmp_path / "next.log", timeout=60)

    after = worker.run("worker_child", ("ok",), tmp_path / "after.log", timeout=60)
    assert _pid_and_seed(after)[0] != pid
