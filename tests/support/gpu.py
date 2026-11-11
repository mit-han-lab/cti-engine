from __future__ import annotations

import functools
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, TextIO
from unittest.mock import patch

import pytest

from tests.support.paths import source_pythonpath

if TYPE_CHECKING:
    from cti.config import EngineOptions, EngineTuning

_FORBIDDEN_LOG_PATTERNS = (
    "traceback",
    "cuda error",
    "illegal memory",
    "task was destroyed",
    "never awaited",
    "exception was never retrieved",
    "exception in callback",
    "resource_tracker",
    "leaked semaphore",
    "exception ignored",
    "segmentation fault",
    "out of memory",
    "aborted",
    "nccl error",
    "process group has not been destroyed",
    "address already in use",
    "fatal python error",
    "uv_loop_close failed",
)
# The GPU qualification supervisor names its run-scoped JIT cache in this variable and, on a
# runner that keeps JIT caches across runs, their directory in the next.
SHARED_JIT_CACHE_VARIABLE = "CTI_GPU_JIT_CACHE_DIR"
KEPT_JIT_CACHE_VARIABLE = "CTI_GPU_KEPT_JIT_CACHE_DIR"
# The supervisor names its GPU qualification in this variable; tests/gpu/conftest.py skips a
# test whose checkpoints that qualification's devices do not hold. Unset: nothing skips.
QUALIFICATION_VARIABLE = "CTI_GPU_QUALIFICATION"
# Every JIT cache a GPU test fills, by the variable that relocates it. Triton's
# cache is deliberately absent: it stays where the environment puts it (by default
# ~/.triton/cache), so its warm kernels are not rebuilt on every run.
_JIT_CACHE_DIRECTORIES = {
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "TVM_FFI_CACHE_DIR": "tvm-ffi",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "TORCHINDUCTOR_CACHE_DIR": "torch-inductor",
    "CUTE_DSL_CACHE_DIR": "cute-dsl",
}
# The caches a runner may keep across runs, under a directory the supervisor keys by what they
# compile with: each keys its entries by what it compiles. Torch extensions build from this
# checkout's sources, so every run builds them again.
_KEPT_JIT_CACHES = frozenset(_JIT_CACHE_DIRECTORIES) - {"TORCH_EXTENSIONS_DIR"}


@contextmanager
def engine_checks_off() -> Iterator[None]:
    """``CTI_CHECKS`` off inside: under it cti-kernels validates indices with host reads
    and temporaries, which a measurement of no host sync or no scratch must exclude."""
    with patch.dict(os.environ, {"CTI_CHECKS": "0"}):
        yield


@contextmanager
def fixed_kv_page_capacity(num_pages: int) -> Iterator[None]:
    """Bound KV capacity only while constructing a focused test worker."""
    if isinstance(num_pages, bool) or not isinstance(num_pages, int) or num_pages <= 1:
        raise ValueError("num_pages must be an integer greater than one")

    from cti import kvcache

    def resolve_fixed_capacity(layout, geometry, _budget, *, snapshots, **_metadata):
        rows = 0
        if snapshots is not None and layout.rows.specs:
            rows = 2 if snapshots.rows is None else snapshots.rows
        return geometry.capacity(layout, num_pages - geometry.guard_pages, rows)

    with patch.object(kvcache, "plan_capacity", resolve_fixed_capacity):
        yield


def start_rank(
    options: EngineOptions, tuning: EngineTuning, *, rank: int = 0, **composition: Any
) -> Any:
    """Rank ``rank``'s parts (``runtime._core.Rank``) composed as ``EngineCore._start`` composes them: the checkpoint
    and any speculative method loaded and checked, the device bound and planned for.
    ``composition`` passes on to the ``Rank``."""

    import torch

    from cti.inference import load_speculative
    from cti.resolution import ResolvedModel
    from cti.utils import get_device_capability

    resolved = ResolvedModel.load(
        options.model.path, hub=options.model.source, text_only=options.model.text_only
    )
    speculative = (
        None
        if options.speculative is None
        else load_speculative(
            options.speculative.model_path,
            target=resolved.model_config,
            hub=options.model.source,
            speculative_tokens=options.speculative.num_speculative_tokens,
        )
    )
    checked = resolved.check(options, tuning, speculative=speculative)
    torch.cuda.set_device(rank)
    plan = resolved.plan(
        checked,
        device=torch.device("cuda", rank),
        capability=get_device_capability(),
        device_memory=torch.cuda.get_device_properties(rank).total_memory,
        online=False,
    )
    return _rank_unit().Rank(plan, resolved, rank=rank, **composition)


def engine_rank_pids() -> tuple[int, ...]:
    """The PIDs of this process's live engine ranks, in rank order, as the OS lists its
    children; joins the ones that exited."""

    import multiprocessing

    prefix = "cti-rank-"
    ranks = [p for p in multiprocessing.active_children() if p.name.startswith(prefix)]
    ranks.sort(key=lambda process: int(process.name.removeprefix(prefix)))
    return tuple(process.pid for process in ranks if process.pid is not None)


def assert_ranks_reclaimed(pids: tuple[int, ...]) -> None:
    """No engine rank of this process lives, and none of ``pids`` exists."""

    assert engine_rank_pids() == ()
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def find_forbidden_log_patterns(output: str) -> tuple[str, ...]:
    normalized = output.casefold()
    return tuple(pattern for pattern in _FORBIDDEN_LOG_PATTERNS if pattern in normalized)


# Beside a child's log, the forbidden patterns that child emits on purpose, one per line: a
# scan of the GPU logs lets them through in that log alone.
EXPECTED_SUFFIX = ".expected"


def expected_log_patterns(log_path: Path) -> tuple[str, ...]:
    """The forbidden patterns ``log_path``'s child declared it emits on purpose."""

    declared = log_path.with_name(log_path.name + EXPECTED_SUFFIX)
    if not declared.is_file():
        return ()
    return tuple(line for line in declared.read_text(encoding="utf-8").splitlines() if line)


def assert_cuda_device_metadata(
    payload: dict[str, object],
    *,
    device_key: str = "device",
    capability_key: str = "capability",
) -> tuple[str, tuple[int, int]]:
    """Validate portable device identity while leaving qualification to named receipts."""

    device = payload.get(device_key)
    if not isinstance(device, str) or not device:
        pytest.fail(f"{device_key} must be a nonempty device name")
    capability = payload.get(capability_key)
    if (
        not isinstance(capability, (list, tuple))
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
        or capability[0] < 1
        or capability[1] < 0
    ):
        pytest.fail(f"{capability_key} must contain a valid CUDA compute capability")
    return device, (capability[0], capability[1])


def require_cuda_device(device_name: str, capability: tuple[int, int]) -> None:
    """Skip a named qualification receipt on other supported CUDA devices."""

    import torch

    actual = (torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    expected = (device_name, capability)
    if actual != expected:
        pytest.skip(f"qualification requires {expected}, found {actual}")


def require_device_memory(min_gib: float) -> None:
    """Skip unless the first visible GPU has ``min_gib`` GiB of memory, as CUDA orders the
    visible GPUs; a child process reads it, so the caller creates no CUDA context."""

    query = subprocess.run(
        (
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.get_device_properties(0).total_memory)",
        ),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    total_gib = int(query.stdout.split()[-1]) / 2**30
    if total_gib < min_gib:
        pytest.skip(f"requires a GPU of {min_gib} GiB, found {total_gib:.1f} GiB")


def pause_device(cycles: int) -> None:
    """Keep the current CUDA stream busy for about ``cycles`` GPU clock cycles, so the host
    can run ahead of it. PyTorch's own test sleep kernel: there is no public equivalent."""

    import torch

    torch.cuda._sleep(cycles)


def jit_cache_env(cache_root: Path, kept_root: Path | None = None) -> dict[str, str]:
    """Relocate every JIT cache in ``_JIT_CACHE_DIRECTORIES`` under ``cache_root``, or, given
    ``kept_root``, those a runner keeps across runs (``_KEPT_JIT_CACHES``) under it."""
    return {
        variable: str(
            (kept_root if kept_root and variable in _KEPT_JIT_CACHES else cache_root) / directory
        )
        for variable, directory in _JIT_CACHE_DIRECTORIES.items()
    }


def isolated_runtime_env(
    tmp_path: Path,
    component: str,
) -> tuple[dict[str, str], Path]:
    """Child environment whose logs and JIT caches are owned by this test run.

    The JIT caches of ``jit_cache_env`` never come from ambient settings or the
    user's home; Triton's cache is the one exception (see
    ``_JIT_CACHE_DIRECTORIES``). Each component compiles into its own cache under
    ``tmp_path`` unless the qualification supervisor names a run-scoped cache in
    ``SHARED_JIT_CACHE_VARIABLE``. That cache is created empty for each run and
    shared by every test in it, so each JIT kernel compiles once per run; on a runner
    that keeps JIT caches (``KEPT_JIT_CACHE_VARIABLE``), those it keeps are shared
    across runs too.
    """
    log_root = tmp_path / "logs" / component
    log_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.pop("CTI_GPU_LOG_DIR", None)
    env.pop("CTI_GPU_CACHE_DIR", None)
    shared_cache_root = env.pop(SHARED_JIT_CACHE_VARIABLE, None)
    kept_cache_root = env.pop(KEPT_JIT_CACHE_VARIABLE, None)
    cache_root = Path(shared_cache_root) if shared_cache_root else tmp_path / "cache" / component
    cache_root.mkdir(parents=True, exist_ok=True)
    kept_root = Path(kept_cache_root) if shared_cache_root and kept_cache_root else None
    env.update(jit_cache_env(cache_root, kept_root))
    env["PYTHONPATH"] = source_pythonpath()
    env["PYTHONUNBUFFERED"] = "1"
    return env, log_root


def offline_child_env(tmp_path: Path, component: str) -> tuple[dict[str, str], Path]:
    """``isolated_runtime_env`` for a child that reads the pinned checkpoint: Hugging Face
    offline, no user site-packages, no tokenizer threads."""

    env, log_root = isolated_runtime_env(tmp_path, component)
    env.update(
        PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    return env, log_root


def eager_decoding(graph_runner):
    """No batch fits a captured graph; restore the runner's policy on exit."""
    return patch.object(graph_runner, "can_use_cuda_graph", lambda batch: False)


class StreamDtypeRecorder:
    """The dtype names of the residual streams that ``ctikernels.norm.inference``'s
    residual-stream leaves read, by phase: ``capture`` from construction, while a runtime
    builds (warm-up and decode-graph capture), then ``serve`` once the caller sets
    ``phase``. Built before the model, it records for the rest of the process, as the
    parity check's engine child runs."""

    def __init__(self) -> None:
        from ctikernels.norm import inference

        self.phase = "capture"
        self.seen: dict[str, set[str]] = {"capture": set(), "serve": set()}
        for name in ("residual_rms_norm", "add_rms_norm_"):
            setattr(inference, name, self._recording(name, getattr(inference, name)))

    def _recording(self, name: str, leaf: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(leaf)
        def call(*args: Any, **kwargs: Any) -> Any:
            # The stream is the residual norm's first operand and the add's second.
            stream = args[0] if name == "residual_rms_norm" else args[1]
            self.seen[self.phase].add(str(stream.dtype).removeprefix("torch."))
            return leaf(*args, **kwargs)

        return call


def run_isolated_process(
    command: tuple[str, ...],
    log_path: Path,
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: float = 180,
    timeout_message: str | None = None,
    expected: tuple[str, ...] = (),
) -> str:
    """``expected``: the forbidden patterns a child that fails on purpose logs, declared
    beside its log for a scan of the GPU logs too."""

    if expected:
        log_path.with_name(log_path.name + EXPECTED_SUFFIX).write_text(
            "".join(f"{pattern}\n" for pattern in expected), encoding="utf-8"
        )
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        report = f"stdout:\n{stdout}\nstderr:\n{stderr}"
        log_path.write_text(report, encoding="utf-8")
        if timeout_message is not None:
            raise TimeoutError(timeout_message) from error
        pytest.fail(f"child process timed out after {error.timeout}s\n{report}")

    report = f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    log_path.write_text(report, encoding="utf-8")
    if completed.returncode != 0:
        pytest.fail(f"child process failed with {completed.returncode}\n{report}")
    forbidden = tuple(p for p in find_forbidden_log_patterns(report) if p not in expected)
    if forbidden:
        pytest.fail(f"child process emitted {forbidden}\n{report}")
    return completed.stdout


def child_receipt(output: str, prefix: str) -> dict[str, Any]:
    """The one ``prefix``-marked JSON line a test child printed."""

    receipts = [
        json.loads(line.removeprefix(prefix))
        for line in output.splitlines()
        if line.startswith(prefix)
    ]
    assert len(receipts) == 1, f"expected one {prefix} receipt, found {len(receipts)}"
    return receipts[0]


class ApiServerProcess:
    _SIGNALS = (
        (signal.SIGINT, 30),
        (signal.SIGTERM, 10),
        (signal.SIGKILL, 10),
    )

    def __init__(
        self,
        command: tuple[str, ...],
        env: dict[str, str],
        log_path: Path,
    ) -> None:
        self._command = command
        self._env = env
        self._log_path = log_path
        self._log_file: TextIO | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._clean_shutdown = False

    @property
    def clean_shutdown(self) -> bool:
        return self._clean_shutdown

    @property
    def process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError("API server has not started")
        return self._process

    def __enter__(self) -> ApiServerProcess:
        self._log_file = self._log_path.open("w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                self._command,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=self._env,
                start_new_session=True,
            )
        except BaseException:
            self._log_file.close()
            self._log_file = None
            raise
        return self

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        body_error: BaseException | None,
        body_traceback: TracebackType | None,
    ) -> None:
        cleanup_error: Exception | None = None
        try:
            self._shutdown()
        except Exception as error:  # noqa: BLE001 - preserve the test body as primary
            cleanup_error = error

        try:
            if self._log_file is not None:
                self._log_file.close()
        except Exception as error:  # noqa: BLE001 - log close is part of cleanup
            if cleanup_error is not None:
                error.__context__ = cleanup_error
            cleanup_error = error
        finally:
            self._log_file = None

        if cleanup_error is not None:
            if body_error is None:
                raise cleanup_error
            raise body_error.with_traceback(body_traceback) from cleanup_error

    @staticmethod
    def _group_exists(process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        return True

    def _shutdown(self) -> None:
        process = self.process
        shutdown_started = False
        clean = True
        for shutdown_signal, timeout in self._SIGNALS:
            if process.poll() is not None:
                break
            try:
                os.killpg(process.pid, shutdown_signal)
            except ProcessLookupError:
                clean = False
                break
            shutdown_started = True
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                clean = False

        if process.poll() is None:
            raise RuntimeError(f"API server {process.pid} resisted SIGKILL")

        deadline = time.monotonic() + 1
        while self._group_exists(process.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._group_exists(process.pid):
            clean = False
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            else:
                deadline = time.monotonic() + 10
                while self._group_exists(process.pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
        if self._group_exists(process.pid):
            raise RuntimeError(f"API server group {process.pid} survived SIGKILL")
        self._clean_shutdown = shutdown_started and clean and process.returncode == 0


def configure_worker_process() -> None:
    """Set a process that builds a GPU worker up as an engine rank does."""

    _rank_unit().configure_rank_process()


def stall_rank_after(trigger: str, *, rank: int) -> None:
    """In a spawned rank: rank ``rank`` stalls after a loop turn once ``trigger`` exists,
    alive and with its work in hand, as a rank stuck in a collective is."""

    import multiprocessing

    progress = _rank_unit().RankProgress
    finished = progress.finished

    def stalling(self: Any) -> None:
        finished(self)
        name = multiprocessing.current_process().name
        if name == f"cti-rank-{rank}" and os.path.exists(trigger):
            time.sleep(3600)

    progress.finished = stalling


def _rank_unit() -> Any:
    """The engine's rank unit, whose composition and process setup these tests reuse."""

    from cti.runtime import _core

    return _core
