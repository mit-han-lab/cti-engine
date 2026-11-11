from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support.gpu import (
    KEPT_JIT_CACHE_VARIABLE,
    SHARED_JIT_CACHE_VARIABLE,
    ApiServerProcess,
    find_forbidden_log_patterns,
    isolated_runtime_env,
    jit_cache_env,
    run_isolated_process,
)


@pytest.mark.parametrize(
    ("run_scoped_cache", "kept"),
    [(False, False), (True, False), (True, True), (False, True)],
    ids=["per-component", "run-scoped", "kept", "kept-without-a-run"],
)
def test_runtime_environment_owns_logs_and_jit_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_scoped_cache: bool,
    kept: bool,
) -> None:
    # Ambient settings, such as the Conda environment's persistent caches, are
    # never used for these JIT caches (Triton's is deliberately left alone). They
    # are either per component or the one directory the qualification supervisor
    # creates empty for its run, and on a runner that keeps JIT caches, those it
    # keeps come from its kept directory; logs stay per component.
    ambient_log_root = tmp_path.parent / f"{tmp_path.name}-ambient-logs"
    ambient_cache_root = tmp_path.parent / f"{tmp_path.name}-ambient-cache"
    monkeypatch.setenv("CTI_GPU_LOG_DIR", str(ambient_log_root))
    monkeypatch.setenv("CTI_GPU_CACHE_DIR", str(ambient_cache_root))
    run_cache_root = tmp_path / "run" / "cache" / "gpu-suite"
    kept_root = tmp_path / "kept" / "key"
    if run_scoped_cache:
        monkeypatch.setenv(SHARED_JIT_CACHE_VARIABLE, str(run_cache_root))
    else:
        monkeypatch.delenv(SHARED_JIT_CACHE_VARIABLE, raising=False)
    if kept:
        monkeypatch.setenv(KEPT_JIT_CACHE_VARIABLE, str(kept_root))
    else:
        monkeypatch.delenv(KEPT_JIT_CACHE_VARIABLE, raising=False)

    cache_variables = {
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
        "TVM_FFI_CACHE_DIR": "tvm-ffi",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": "torch-inductor",
        "CUTE_DSL_CACHE_DIR": "cute-dsl",
    }
    # The supervisor relocates the same caches through jit_cache_env; a cache added
    # there must be pinned here too.
    assert jit_cache_env(tmp_path) == {
        name: str(tmp_path / directory) for name, directory in cache_variables.items()
    }
    # Torch extensions build from the checkout: no run keeps them.
    assert jit_cache_env(tmp_path, kept_root) == {
        name: str((tmp_path if name == "TORCH_EXTENSIONS_DIR" else kept_root) / directory)
        for name, directory in cache_variables.items()
    }
    for name in cache_variables:
        monkeypatch.setenv(name, str(ambient_cache_root / name.casefold()))

    for component in ("first", "second"):
        env, log_root = isolated_runtime_env(tmp_path, component)

        cache_root = run_cache_root if run_scoped_cache else tmp_path / "cache" / component
        assert log_root == tmp_path / "logs" / component
        assert log_root.is_dir()
        assert cache_root.is_dir()
        expected = jit_cache_env(cache_root, kept_root if run_scoped_cache and kept else None)
        assert {name: env[name] for name in cache_variables} == expected
        assert "CTI_GPU_LOG_DIR" not in env
        assert "CTI_GPU_CACHE_DIR" not in env
        assert SHARED_JIT_CACHE_VARIABLE not in env
        assert KEPT_JIT_CACHE_VARIABLE not in env
    assert run_cache_root.exists() is run_scoped_cache
    assert (tmp_path / "cache").exists() is not run_scoped_cache
    assert not ambient_log_root.exists()
    assert not ambient_cache_root.exists()


def test_isolated_process_timeout_preserves_partial_output(monkeypatch, tmp_path: Path) -> None:
    timeout = subprocess.TimeoutExpired(
        cmd=("child",),
        timeout=180,
        output="partial stdout",
        stderr="partial stderr",
    )

    def raise_timeout(*_args, **_kwargs):
        raise timeout

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    log_path = tmp_path / "timeout.log"

    with pytest.raises(pytest.fail.Exception, match="timed out"):
        run_isolated_process(("child",), log_path, env={}, cwd=tmp_path)

    assert log_path.read_text(encoding="utf-8") == (
        "stdout:\npartial stdout\nstderr:\npartial stderr"
    )


def test_runtime_components_preserve_same_named_child_logs(tmp_path: Path) -> None:
    child = (
        "import sys\n"
        "print(sys.argv[1] + ' stdout')\n"
        "print(sys.argv[1] + ' stderr', file=sys.stderr)"
    )
    logs = {}
    for component in ("logical_batch", "microbatch"):
        env, log_root = isolated_runtime_env(tmp_path, component)
        logs[component] = log_root / "child.log"
        stdout = run_isolated_process(
            (sys.executable, "-c", child, component),
            logs[component],
            env=env,
            cwd=tmp_path,
            timeout=10,
        )
        assert stdout == f"{component} stdout\n"

    for component, path in logs.items():
        assert path.read_text(encoding="utf-8") == (
            f"stdout:\n{component} stdout\n\nstderr:\n{component} stderr\n"
        )


def test_isolated_process_runs_a_real_child_with_cwd_and_env(tmp_path: Path) -> None:
    child = (
        "import os, sys\n"
        "if os.getcwd() != sys.argv[1]:\n"
        "    raise RuntimeError('unexpected cwd')\n"
        "if os.environ.get('CTI_TEST_SENTINEL') != 'present':\n"
        "    raise RuntimeError('missing test environment')\n"
        "sys.stdout.write('complete stdout')\n"
        "sys.stderr.write('complete stderr')"
    )
    env = os.environ.copy()
    env["CTI_TEST_SENTINEL"] = "present"
    log_path = tmp_path / "success.log"

    stdout = run_isolated_process(
        (sys.executable, "-c", child, str(tmp_path)),
        log_path,
        env=env,
        cwd=tmp_path,
        timeout=10,
    )

    assert stdout == "complete stdout"
    assert log_path.read_text(encoding="utf-8") == (
        "stdout:\ncomplete stdout\nstderr:\ncomplete stderr"
    )


def test_isolated_process_preserves_complete_failure_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    completed = subprocess.CompletedProcess(
        args=("child",),
        returncode=7,
        stdout="complete stdout",
        stderr="complete stderr",
    )
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: completed)
    log_path = tmp_path / "failure.log"

    with pytest.raises(pytest.fail.Exception, match="failed with 7"):
        run_isolated_process(("child",), log_path, env={}, cwd=tmp_path)

    assert log_path.read_text(encoding="utf-8") == (
        "stdout:\ncomplete stdout\nstderr:\ncomplete stderr"
    )


def test_isolated_process_rejects_unretrieved_async_exception(monkeypatch, tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(
        args=("child",),
        returncode=0,
        stdout="",
        stderr="Task exception was never retrieved",
    )
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(pytest.fail.Exception, match="exception was never retrieved"):
        run_isolated_process(("child",), tmp_path / "async-error.log", env={}, cwd=tmp_path)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CUDA ERROR: launch failed", "cuda error"),
        ("RuntimeError: OUT OF MEMORY", "out of memory"),
        ("Task EXCEPTION was never retrieved", "exception was never retrieved"),
        ("uv_loop_close failed", "uv_loop_close failed"),
    ],
)
def test_forbidden_log_detection_is_case_insensitive(message: str, expected: str) -> None:
    assert find_forbidden_log_patterns(message) == (expected,)


def test_context_manager_preserves_body_error_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    body_error = ValueError("request failed")
    cleanup_error = RuntimeError("cleanup failed")
    server = ApiServerProcess(("server",), {}, tmp_path / "server.log")
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: object())

    def fail_cleanup() -> None:
        raise cleanup_error

    monkeypatch.setattr(server, "_shutdown", fail_cleanup)

    with pytest.raises(ValueError, match="request failed") as raised:
        with server:
            raise body_error

    assert raised.value is body_error
    assert raised.value.__cause__ is cleanup_error
    assert server._log_file is None


def test_context_manager_raises_cleanup_error_without_a_body_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server = ApiServerProcess(("server",), {}, tmp_path / "server.log")
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: object())

    def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(server, "_shutdown", fail_cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        with server:
            pass

    assert server._log_file is None


def test_shutdown_does_not_claim_an_already_exited_process_is_clean(
    monkeypatch, tmp_path: Path
) -> None:
    class ExitedProcess:
        pid = 12345
        returncode = 0

        def poll(self) -> int:
            return 0

    server = ApiServerProcess((), {}, tmp_path / "server.log")
    server._process = ExitedProcess()
    monkeypatch.setattr(
        "tests.support.gpu.os.killpg",
        lambda *_args: pytest.fail("already-exited process must not be signaled"),
    )
    monkeypatch.setattr(server, "_group_exists", lambda _process_group: False)

    server._shutdown()

    assert not server.clean_shutdown


def test_context_manager_starts_a_new_session_and_handles_sigint(tmp_path: Path) -> None:
    ready_path = tmp_path / "ready"
    child = (
        "import pathlib, signal, sys\n"
        "ready = pathlib.Path(sys.argv[1])\n"
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
        "ready.touch()\n"
        "signal.pause()"
    )
    log_path = tmp_path / "server.log"
    server = ApiServerProcess(
        (sys.executable, "-c", child, str(ready_path)),
        os.environ.copy(),
        log_path,
    )

    with server:
        deadline = time.monotonic() + 5
        while not ready_path.exists():
            if server.process.poll() is not None:
                pytest.fail(
                    f"child exited before becoming ready:\n{log_path.read_text(encoding='utf-8')}"
                )
            if time.monotonic() >= deadline:
                pytest.fail("child did not become ready within 5 seconds")
            time.sleep(0.01)

        pid = server.process.pid
        assert os.getpgid(pid) == pid
        assert os.getsid(pid) == pid

    assert server.process.returncode == 0
    assert server.clean_shutdown


def test_shutdown_escalates_and_marks_forced_exit_unclean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ResistantProcess:
        pid = 12345
        returncode: int | None = None
        last_signal: signal.Signals | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            if self.last_signal == signal.SIGKILL:
                self.returncode = -signal.SIGKILL
                return self.returncode
            raise subprocess.TimeoutExpired(cmd=("server",), timeout=timeout)

    process = ResistantProcess()
    sent_signals: list[signal.Signals] = []

    def send_signal(_process_group: int, shutdown_signal: signal.Signals) -> None:
        sent_signals.append(shutdown_signal)
        process.last_signal = shutdown_signal

    server = ApiServerProcess((), {}, tmp_path / "server.log")
    server._process = process
    monkeypatch.setattr("tests.support.gpu.os.killpg", send_signal)
    monkeypatch.setattr(server, "_group_exists", lambda _process_group: False)

    server._shutdown()

    assert sent_signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert process.returncode == -signal.SIGKILL
    assert not server.clean_shutdown


@pytest.mark.parametrize("disappears_during_kill", [False, True])
def test_shutdown_cleans_a_residual_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    disappears_during_kill: bool,
) -> None:
    class GracefulProcess:
        pid = 12345
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            assert timeout == 30
            self.returncode = 0
            return self.returncode

    process = GracefulProcess()
    sent_signals: list[signal.Signals] = []
    group_states = iter(
        [True, True, False] if disappears_during_kill else [True, True, False, False]
    )
    monotonic_values = iter([0.0, 2.0] if disappears_during_kill else [0.0, 2.0, 2.0])

    def send_signal(_process_group: int, shutdown_signal: signal.Signals) -> None:
        sent_signals.append(shutdown_signal)
        if disappears_during_kill and shutdown_signal == signal.SIGKILL:
            raise ProcessLookupError

    server = ApiServerProcess((), {}, tmp_path / "server.log")
    server._process = process
    monkeypatch.setattr("tests.support.gpu.os.killpg", send_signal)
    monkeypatch.setattr(server, "_group_exists", lambda _process_group: next(group_states))
    monkeypatch.setattr(
        "tests.support.gpu.time.monotonic",
        lambda: next(monotonic_values),
    )

    server._shutdown()

    assert sent_signals == [signal.SIGINT, signal.SIGKILL]
    assert not server.clean_shutdown


def test_shutdown_does_not_claim_a_pre_signal_exit_is_clean(monkeypatch, tmp_path: Path) -> None:
    class ExitedBeforeSignalProcess:
        pid = 12345
        returncode = 0

        def __init__(self) -> None:
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

    def raise_process_lookup(*_args: object) -> None:
        raise ProcessLookupError

    server = ApiServerProcess((), {}, tmp_path / "server.log")
    server._process = ExitedBeforeSignalProcess()
    monkeypatch.setattr("tests.support.gpu.os.killpg", raise_process_lookup)
    monkeypatch.setattr(server, "_group_exists", lambda _process_group: False)

    server._shutdown()

    assert not server.clean_shutdown
