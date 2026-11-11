"""One long-lived process that runs GPU test children in turn.

``cute.compile`` keeps no cache across processes, so every engine process compiles the CuTe
kernels it uses as it starts: about a hundred for a DFlash engine, a minute or more. Children
that run in one worker compile each kernel they share once. After each child the worker
collects its objects and releases cached device memory, and the default generators are
seeded as in a fresh process. A child that fails ends the worker; the next child starts another.
"""

from __future__ import annotations

import gc
import json
import os
import runpy
import select
import signal
import subprocess
import sys
import traceback
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from types import TracebackType

import pytest

from tests.support.gpu import find_forbidden_log_patterns


class EngineWorker:
    """Runs ``python -m MODULE ARGS`` children in one worker process, with the checks and
    log of ``run_isolated_process``."""

    def __init__(
        self, *, env: dict[str, str], cwd: Path, log_path: Path, shutdown_seconds: float = 60
    ) -> None:
        self._env = env
        self._cwd = cwd
        self._log_path = log_path
        self._shutdown_seconds = shutdown_seconds
        self._process: subprocess.Popen[str] | None = None
        self._replies = None
        self._log_start = 0  # where the current worker's own output starts in its log

    def __enter__(self) -> EngineWorker:
        return self

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def run(self, module: str, args: Sequence[str], log_path: Path, *, timeout: float) -> str:
        """Run one child and return its stdout; fail the test as ``run_isolated_process``
        does on a non-zero status, a timeout or a forbidden log pattern."""

        if self._process is None:
            self._start()
        assert self._process is not None and self._process.stdin is not None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        streams = {name: log_path.with_name(f"{log_path.name}.{name}") for name in ("out", "err")}
        job = {"module": module, "args": list(args), **{k: str(v) for k, v in streams.items()}}
        try:
            self._process.stdin.write(json.dumps(job) + "\n")
            self._process.stdin.flush()
        except OSError as error:
            # The worker ended between children: report it; the next child starts another.
            self.close()
            pytest.fail(f"engine worker ended before {module} started: {error}")
        status: int | None = None
        failure = f"child process timed out after {timeout}s"
        replied = bool(select.select([self._replies], [], [], timeout)[0])
        if replied and (reply := self._replies.readline()):
            status = json.loads(reply)["status"]
        elif replied:
            # No reply: the child ended the worker, whose exit status is then the child's.
            replied = False
            failure = "child process ended its worker, which did not exit in time"
            with suppress(subprocess.TimeoutExpired):
                status = self._process.wait(timeout=self._shutdown_seconds)
        if not replied or status != 0:
            self.close(kill=status is None, check=False)
        stdout, stderr = (
            streams[name].read_text(errors="replace") if streams[name].exists() else ""
            for name in ("out", "err")
        )
        for path in streams.values():
            path.unlink(missing_ok=True)
        report = f"stdout:\n{stdout}\nstderr:\n{stderr}"
        log_path.write_text(report, encoding="utf-8")
        if status is None:
            pytest.fail(f"{failure}\n{report}")
        if status != 0:
            pytest.fail(f"child process failed with {status}\n{report}")
        forbidden = find_forbidden_log_patterns(report)
        if forbidden:
            self.close(check=False)
            pytest.fail(f"child process emitted {forbidden}\n{report}")
        return stdout

    def close(self, *, kill: bool = False, check: bool = True) -> None:
        """End the worker: it exits at the end of its input, or, with ``kill``, its process
        group is killed. With ``check``, a worker that does not exit in time (it is then
        killed), exits with an error, or writes a forbidden pattern outside the children,
        such as at interpreter shutdown, fails the test as a child would."""

        process, self._process = self._process, None
        if process is None:
            return
        assert process.stdin is not None and self._replies is not None
        with suppress(OSError):
            process.stdin.close()
        hung = False
        if not kill:
            try:
                process.wait(timeout=self._shutdown_seconds)
            except subprocess.TimeoutExpired:
                hung = True
        if kill or hung:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        self._replies.close()
        self._replies = None
        if check and not kill:
            with self._log_path.open("rb") as log:
                log.seek(self._log_start)
                output = log.read().decode(errors="replace")
            forbidden = find_forbidden_log_patterns(output)
            if hung or process.returncode or forbidden:
                ending = "did not exit in time" if hung else f"exited with {process.returncode}"
                pytest.fail(f"engine worker {ending} and emitted {forbidden}\n{output}")

    def _start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        reply_reader, reply_writer = os.pipe()
        try:
            with self._log_path.open("a", encoding="utf-8") as log:
                self._log_start = self._log_path.stat().st_size
                self._process = subprocess.Popen(
                    (sys.executable, "-m", __name__, str(reply_writer)),
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self._env,
                    cwd=self._cwd,
                    text=True,
                    pass_fds=(reply_writer,),
                    start_new_session=True,
                )
        except BaseException:
            os.close(reply_reader)
            raise
        finally:
            os.close(reply_writer)
        self._replies = os.fdopen(reply_reader, "rb", buffering=0)


def _run_child(job: dict) -> int:
    import torch

    # As in a fresh process: every default generator from a nondeterministic seed.
    torch.seed()
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(job["out"], "w") as out, open(job["err"], "w") as err:
        os.dup2(out.fileno(), 1)
        os.dup2(err.fileno(), 2)
        sys.argv = [job["module"], *job["args"]]
        try:
            runpy.run_module(job["module"], run_name="__main__", alter_sys=True)
            status = 0
        except SystemExit as exit_:
            status = exit_.code if isinstance(exit_.code, int) else int(exit_.code is not None)
        except BaseException:  # noqa: BLE001 - the child's failure is its status
            traceback.print_exc()
            status = 1
        finally:
            # The child's objects die here, in its streams, as at its process's exit.
            gc.collect()
            torch.cuda.empty_cache()
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, original in enumerate(saved, start=1):
                os.dup2(original, fd)
                os.close(original)
    return status


def main(reply_writer: int) -> None:
    with os.fdopen(reply_writer, "w") as replies:
        for line in sys.stdin:
            status = _run_child(json.loads(line))
            replies.write(json.dumps({"status": status}) + "\n")
            replies.flush()
            if status != 0:
                return


if __name__ == "__main__":
    main(int(sys.argv[1]))
