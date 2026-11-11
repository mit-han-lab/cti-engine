"""Real process tests for what an online engine's client owns: the engine's ranks and the
tokenizer workers, on the CPU stand-in."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator

import pytest

from cti.config import EngineOptions, ModelOptions
from cti.runtime import EngineClient, EngineProcessError, FailurePhase
from tests.support.cpu_engine import CpuEngine
from tests.support.dflash import TINY_TARGET_CONFIG
from tests.support.fault_injection import FaultInjector
from tests.support.memory_saver import use_memory_saver_provider

_WORKERS = {"cti-detokenizer", "cti-tokenizer-0"}
_READY = re.compile(r"API server is ready to serve on [^ ]+:(\d+)")


def test_server_bootstrap_does_not_import_torch() -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import cti.server.launch; raise SystemExit('torch' in sys.modules)",
        ),
        check=False,
    )

    assert result.returncode == 0


def test_a_server_whose_engine_fails_to_start_exits_at_once(tmp_path) -> None:
    """A startup failure (here a draft directory that does not exist) ends the server process
    with a non-zero code and its error, and leaves no process of its session behind."""

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(TINY_TARGET_CONFIG))
    command = (
        sys.executable,
        "-m",
        "cti",
        "--model.path",
        str(model),
        "--speculative.model-path",
        str(tmp_path / "missing-draft"),
    )
    # The real worker, which resolves the checkpoint first, with the CPU stand-in's memory.
    memory = tuple(
        injection
        for injection in CpuEngine.injections()
        if injection.target.startswith("cti.memory:")
    )
    environment = FaultInjector(memory).environment(os.environ.get("PYTHONPATH", ""))
    started = time.monotonic()
    server = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env={**os.environ, **environment, "CUDA_VISIBLE_DEVICES": ""},
    )
    try:
        output, _ = server.communicate(timeout=120)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGKILL)
    elapsed = time.monotonic() - started

    assert server.returncode not in (0, None), output
    assert "the engine failed while starting" in output, output
    assert f"model directory {tmp_path / 'missing-draft'} does not exist" in output, output
    assert elapsed < 60, elapsed
    # multiprocessing's resource tracker exits once it sees the server's end close.
    deadline = time.monotonic() + 10
    with pytest.raises(ProcessLookupError):
        while time.monotonic() < deadline:
            os.killpg(server.pid, 0)
            time.sleep(0.05)


def test_a_server_whose_port_is_busy_exits_before_starting_its_engine(tmp_path) -> None:
    """A busy address ends the server process at once, naming it, before any rank spawns."""

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(TINY_TARGET_CONFIG))
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        command = (sys.executable, "-m", "cti", "--model.path", str(model))
        command += ("--server.port", str(port))
        started = time.monotonic()
        server = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        try:
            output, _ = server.communicate(timeout=60)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGKILL)

    assert server.returncode not in (0, None), output
    assert f"cannot serve on 127.0.0.1:{port}: Address already in use" in output, output
    assert "the engine" not in output, output
    assert time.monotonic() - started < 30
    with pytest.raises(ProcessLookupError):
        os.killpg(server.pid, 0)


@pytest.fixture
def online(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., EngineClient]]:
    """Starts online engines with one tokenizer worker on the CPU stand-in."""

    clients: list[EngineClient] = []

    def start(**options: object) -> EngineClient:
        use_memory_saver_provider(monkeypatch)
        monkeypatch.setattr(tempfile, "tempdir", "/tmp")  # within AF_UNIX's path limit
        injector = FaultInjector(CpuEngine.injections())
        for key, value in injector.environment(os.environ.get("PYTHONPATH", "")).items():
            monkeypatch.setenv(key, value)
        engine = EngineOptions(model=ModelOptions(path="cpu-model"))
        clients.append(EngineClient(engine, num_tokenizers=1, **options))
        return clients[-1]

    yield start
    for client in clients:
        client.close()


def _children() -> dict[str, mp.process.BaseProcess]:
    return {child.name: child for child in mp.active_children()}


def test_close_reclaims_the_engine_and_its_tokenizer_workers(online) -> None:
    client = online()
    assert _WORKERS <= _children().keys()

    client.close()

    assert not _children()


def test_a_tokenizer_worker_exit_fails_the_engine(online) -> None:
    reported = threading.Event()
    client = online(on_failure=reported.set)
    os.kill(_children()["cti-tokenizer-0"].pid, signal.SIGKILL)

    # The client's reader thread sees the exit and reports the failure.
    assert reported.wait(10), "the engine did not fail with its tokenizer worker"
    with pytest.raises(
        EngineProcessError, match=r"^cti-tokenizer-0 exited unexpectedly \(exit code -9\)$"
    ):
        client.check()
    assert client.failure.phase is FailurePhase.RUNNING


class _Server:
    """``python -m cti`` with ``num_tokenizers`` tokenizer workers on the CPU stand-in,
    in its own session with ``env`` added, serving once ``ready`` returns."""

    def __init__(self, num_tokenizers: int = 1, env: dict[str, str] | None = None) -> None:
        # Port 0: the server binds one the OS picks and logs it; nothing races for it.
        command = (sys.executable, "-m", "cti", "--model.path", "cpu-model")
        command += ("--server.port", "0", "--server.num-tokenizers", str(num_tokenizers))
        self.port: int | None = None
        self._read: list[str] = []  # the output ready consumed
        injected = FaultInjector(CpuEngine.injections()).environment(
            os.environ.get("PYTHONPATH", "")
        )
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env={
                **os.environ,
                **injected,
                "CUDA_VISIBLE_DEVICES": "",
                "TMPDIR": "/tmp",
                **(env or {}),
            },
        )

    def ready(self, timeout: float = 120) -> None:
        deadline = time.monotonic() + timeout
        while self.port is None:
            line = self.process.stdout.readline()
            assert line, "".join(self._read) + self.process.communicate()[0]
            self._read.append(line)
            if ready := _READY.search(line):
                self.port = int(ready[1])
        while time.monotonic() < deadline:
            assert self.process.poll() is None, "".join(self._read) + self.process.communicate()[0]
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=1):
                    return
            except OSError:
                time.sleep(0.1)
        raise AssertionError("the server did not become ready")

    def workers(self) -> list[int]:
        """The server's children that lead their own group, in start order: the rank,
        the detokenizer, then the tokenizer worker."""

        children = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as stat:
                    fields = stat.read().rsplit(")", 1)[1].split()
            except OSError:
                continue
            pid, ppid, group, started = int(entry), int(fields[1]), int(fields[2]), int(fields[19])
            if ppid == self.process.pid and group == pid:
                children.append((started, pid))
        return [pid for _, pid in sorted(children)]

    def exit(self, timeout: float = 60) -> tuple[int, str]:
        """The server's exit code and output; nothing of its session outlives it."""

        try:
            output, _ = self.process.communicate(timeout=timeout)
            output = "".join(self._read) + output
        finally:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                return self.process.returncode, output
            time.sleep(0.05)
        raise AssertionError(f"processes of the server's session outlived it:\n{output}")


def test_clients_that_leave_while_slow_tokenizers_work_leave_no_request_running(
    tmp_path,
) -> None:
    """With two tokenizer workers, each request's abort follows it through its tokenizer:
    it cannot reach the engine before the request and leave the request running."""

    log = tmp_path / "device.log"
    env = {
        CpuEngine.TOKENIZE_SECONDS: "1",
        CpuEngine.STEP_SECONDS: "0.1",
        CpuEngine.STEPS: "1",
        CpuEngine.LOG: str(log),
    }
    server = _Server(num_tokenizers=2, env=env)
    try:
        server.ready()
        body = json.dumps({"model": "cpu-model", "prompt": "1 2 3", "max_tokens": 60})
        request = (
            b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n{body}".encode()
        )
        for _ in range(4):
            with socket.create_connection(("127.0.0.1", server.port)) as client:
                client.sendall(request)
                time.sleep(0.2)  # the request is being tokenized; its client leaves
        time.sleep(3)  # the tokenizers forward the requests, then their aborts

        def steps() -> int:
            text = log.read_text() if log.exists() else ""
            return sum(" step " in line for line in text.splitlines())

        before = steps()
        time.sleep(1)  # a request left running would take 10 steps meanwhile
        assert steps() == before, "a request kept running after its client left"
    finally:
        server.process.send_signal(signal.SIGINT)
        server.exit()


def test_a_server_shuts_down_gracefully_on_sigint() -> None:
    server = _Server()
    server.ready()

    server.process.send_signal(signal.SIGINT)

    code, output = server.exit()
    assert code == 0, output
    assert "Inference engine stopped" in output, output


def test_a_second_sigint_forces_the_shutdown() -> None:
    server = _Server()
    server.ready()

    server.process.send_signal(signal.SIGINT)
    time.sleep(0.2)
    server.process.send_signal(signal.SIGINT)

    code, output = server.exit()
    assert code is not None, output


def test_a_server_whose_tokenizer_worker_is_killed_exits_with_its_failure() -> None:
    server = _Server()
    server.ready()
    *_, tokenizer = server.workers()

    os.kill(tokenizer, signal.SIGKILL)

    code, output = server.exit()
    assert code not in (0, None), output
    assert "cti-tokenizer-0 exited unexpectedly" in output, output
