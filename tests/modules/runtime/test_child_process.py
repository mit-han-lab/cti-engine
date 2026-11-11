from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import signal
from unittest.mock import Mock

import pytest

import cti.runtime._process as process_module


def _return_and_hold_interpreter_exit(*, ready_connection) -> None:
    # Report the disposition the child started with, so the test proves that
    # the entry point, not an inherited SIG_IGN, is what silences the interrupt.
    started_with_default_handler = signal.getsignal(signal.SIGINT) is signal.default_int_handler

    def hold_interpreter_exit() -> None:
        # Stands in for torch's weakref.finalize hook: interpreter teardown in
        # a support child takes time after the entry point has returned. The
        # window is held open by an explicit handshake, not by a timed sleep,
        # and the callback makes no assertions (one would surface as
        # "Exception ignored in atexit callback" and hide the behaviour under
        # test). A bounded poll keeps a lost parent from wedging the child.
        try:
            ready_connection.send("in-atexit")
            if ready_connection.poll(10.0):
                ready_connection.recv()
        finally:
            ready_connection.close()

    atexit.register(hold_interpreter_exit)
    ready_connection.send(("ready", started_with_default_handler))


def test_parent_binding_rejects_adoption_by_a_different_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(process_module.os, "pidfd_open", raising=False)
    monkeypatch.setattr(process_module.sys, "platform", "linux")
    monkeypatch.setattr(process_module.os, "getppid", lambda: 777)
    target = Mock()

    with pytest.raises(RuntimeError, match="parent exited"):
        process_module.run_parent_owned_process(123, target)

    target.assert_not_called()


@pytest.mark.parametrize("watcher_parent_pids", [(777,), (123, 777)])
def test_parent_binding_without_pidfd_detects_death_during_startup_or_polling(
    monkeypatch: pytest.MonkeyPatch, watcher_parent_pids: tuple[int, ...]
) -> None:
    monkeypatch.delattr(process_module.os, "pidfd_open", raising=False)
    monkeypatch.setattr(process_module.sys, "platform", "linux")
    # The synchronous check succeeds. The parent then dies either before the
    # watcher starts, or during its first sleep. Run that watcher explicitly so
    # the test cannot leave a daemon thread watching pytest's own lifetime.
    parent_pids = iter((123, *watcher_parent_pids))
    monkeypatch.setattr(process_module.os, "getppid", lambda: next(parent_pids))
    thread = Mock()
    monkeypatch.setattr(process_module, "Thread", thread)
    sleep = Mock()
    monkeypatch.setattr(process_module.time, "sleep", sleep)
    exit_process = Mock(side_effect=SystemExit(1))
    monkeypatch.setattr(process_module.os, "_exit", exit_process)

    process_module._bind_lifetime_to_parent(123)

    thread.return_value.start.assert_called_once_with()
    assert thread.call_args.kwargs["daemon"] is True
    with pytest.raises(SystemExit) as caught:
        thread.call_args.kwargs["target"]()
    assert caught.value.code == 1
    exit_process.assert_called_once_with(1)
    if len(watcher_parent_pids) == 1:
        sleep.assert_not_called()
    else:
        sleep.assert_called_once_with(0.1)


def test_parent_owned_process_binds_before_calling_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    monkeypatch.setattr(
        process_module,
        "_bind_lifetime_to_parent",
        lambda parent_pid: events.append(("bind", parent_pid)),
    )
    # Record instead of installing: the real call would leave SIG_IGN on this
    # pytest process and be inherited by every spawned child afterwards.
    monkeypatch.setattr(
        process_module.signal,
        "signal",
        lambda signum, handler: events.append(("signal", signum, handler)),
    )

    process_module.run_parent_owned_process(
        123,
        lambda first, *, second: events.append(("target", first, second)),
        ("a",),
        {"second": "b"},
    )

    assert events == [
        ("bind", 123),
        ("target", "a", "b"),
        ("signal", signal.SIGINT, signal.SIG_IGN),
    ]


def test_parent_owned_process_ignores_interrupt_after_target_returns(
    capfd: pytest.CaptureFixture[str],
) -> None:
    context = mp.get_context("spawn")
    parent_end, child_end = context.Pipe(duplex=True)
    process = context.Process(
        target=process_module.run_parent_owned_process,
        args=(os.getpid(), _return_and_hold_interpreter_exit, (), {"ready_connection": child_end}),
    )
    process.start()
    child_end.close()
    try:
        assert parent_end.poll(10.0), "child did not report readiness"
        assert parent_end.recv() == ("ready", True)
        # The child's atexit hook reports when interpreter exit has begun and
        # then waits for the release below, so the interrupt lands inside
        # teardown regardless of scheduling.
        assert parent_end.poll(10.0), "child did not reach interpreter exit"
        assert parent_end.recv() == "in-atexit"
        # What ServerProcessSupervisor.request_shutdown(force=False) sends to a
        # child that still reports itself alive.
        os.kill(process.pid, signal.SIGINT)
        parent_end.send("release")
        process.join(timeout=10.0)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            try:
                parent_end.send("release")
            except (BrokenPipeError, OSError):
                pass
            process.join(timeout=5.0)
        parent_end.close()
        if process.is_alive():
            process.kill()
        process.join(timeout=5.0)
        process.close()

    captured = capfd.readouterr()
    assert "KeyboardInterrupt" not in captured.err
    assert "Exception ignored" not in captured.err


def _hold_until_released(connection):
    connection.send("ready")
    if connection.poll(5.0):
        connection.recv()


def test_parent_owned_child_survives_spawning_thread_exit():
    from threading import Thread

    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=process_module.run_parent_owned_process,
        args=(os.getpid(), _hold_until_released, (child,)),
    )

    def spawn():
        process.start()
        assert parent.poll(5.0)
        assert parent.recv() == "ready"

    thread = Thread(target=spawn)
    thread.start()
    thread.join(10.0)
    assert not thread.is_alive()
    process.join(0.1)
    try:
        assert process.is_alive(), "spawning thread exit must not terminate the child"
        parent.send("release")
    finally:
        process.join(6.0)
        parent.close()
        child.close()
    assert process.exitcode == 0
    process.close()


@pytest.mark.skipif(process_module.sys.platform != "linux", reason="Linux subreaper harness")
@pytest.mark.parametrize("from_thread", [False, True])
def test_parent_owned_child_exits_when_real_parent_dies_without_pidfd(tmp_path, from_thread):
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "parent_death.py"
    script.write_text(
        textwrap.dedent("""\
        import ctypes
        import os
        import select
        import time
        from threading import Thread

        from cti.runtime._process import run_parent_owned_process

        if hasattr(os, "pidfd_open"):
            delattr(os, "pidfd_open")

        # Only this isolated harness becomes a subreaper, so it can reap the
        # orphan itself without changing pytest's process ownership.
        assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
        ready_r, ready_w = os.pipe()
        parent_r, parent_w = os.pipe()
        child_r, child_w = os.pipe()
        spawned_r, spawned_w = os.pipe()

        def held_child():
            os.write(ready_w, str(os.getpid()).encode())
            # Check liveness after the creator thread has exited, before
            # allowing the owner process to exit.
            assert select.select([child_r], [], [], 5.0)[0]
            assert os.read(child_r, 64) == b"check"
            os.write(ready_w, b"alive")
            # The harness keeps this descriptor open after the owner dies:
            # EOF cannot masquerade as parent-death detection.
            select.select([child_r], [], [], 15.0)
            os._exit(42)  # fallback release is a test failure, not parent death

        def reap(pid, timeout=5.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited:
                    return os.waitstatus_to_exitcode(status)
                time.sleep(0.01)
            raise AssertionError(f"child {pid} was not reaped before deadline")

        owner = os.fork()
        if owner == 0:
            expected_parent = os.getpid()
            def spawn():
                child = os.fork()
                if child == 0:
                    run_parent_owned_process(expected_parent, held_child)
                    os._exit(43)

            if FROM_THREAD:
                thread = Thread(target=spawn)
                thread.start()
                thread.join(5.0)
                assert not thread.is_alive()
            else:
                spawn()
            os.write(spawned_w, b"spawned")
            select.select([parent_r], [], [], 10.0)
            os._exit(0)  # actual process death, bypassing multiprocessing cleanup

        child = None
        owner_reaped = False
        child_reaped = False
        try:
            assert select.select([ready_r], [], [], 5.0)[0], "child never bound to parent"
            child = int(os.read(ready_r, 64))
            assert select.select([spawned_r], [], [], 5.0)[0], "creator did not return"
            os.read(spawned_r, 64)
            os.write(child_w, b"check")
            assert select.select([ready_r], [], [], 5.0)[0], "child died with creator thread"
            assert os.read(ready_r, 64) == b"alive"
            os.write(parent_w, b"exit")
            assert reap(owner) == 0
            owner_reaped = True
            assert reap(child, timeout=2.0) == 1
            child_reaped = True
            print("parent death reclaimed child", flush=True)
        finally:
            os.write(parent_w, b"exit")
            os.write(child_w, b"release")
            if not owner_reaped:
                reap(owner)
            if child is not None and not child_reaped:
                reap(child)
        """).replace("FROM_THREAD", repr(from_thread))
    )
    with subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as harness:
        try:
            stdout, stderr = harness.communicate(timeout=25.0)
        finally:
            if harness.poll() is None:
                harness.terminate()
                harness.wait(timeout=5.0)
    assert harness.returncode == 0, (stdout, stderr)
    assert "parent death reclaimed child" in stdout
