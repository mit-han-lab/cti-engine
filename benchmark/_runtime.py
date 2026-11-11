from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]

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
)
_REQUIRED_SERVER_LOG_PATTERNS = (
    "Selected attention backend fi",
    "Captured decode graphs",
    "Inference engine stopped",
    "Application shutdown complete",
)
_KV_CACHE_PAGES = re.compile(r"Inference capacity: ([0-9]+) KV pages")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_forbidden_log_patterns(output: str) -> tuple[str, ...]:
    normalized = output.casefold()
    return tuple(pattern for pattern in _FORBIDDEN_LOG_PATTERNS if pattern in normalized)


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def prepare_output_file(path: Path) -> None:
    if path.resolve().is_relative_to(REPO_ROOT):
        raise ValueError("benchmark artifacts must be written outside the repository")
    if path.exists():
        raise FileExistsError(f"benchmark receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def server_environment() -> dict[str, str]:
    result = os.environ.copy()
    result.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "CTI_DISABLE_OVERLAP_SCHEDULING": "0",
        }
    )
    return result


def validate_server_log(output: str) -> None:
    forbidden = find_forbidden_log_patterns(output)
    if forbidden:
        raise RuntimeError(f"server emitted forbidden diagnostics: {forbidden}")
    missing = tuple(pattern for pattern in _REQUIRED_SERVER_LOG_PATTERNS if pattern not in output)
    if missing:
        raise RuntimeError(f"server log is missing required lifecycle evidence: {missing}")


@dataclass(frozen=True)
class KVCacheCapacity:
    pages: int
    tokens: int

    @classmethod
    def from_server_log(cls, output: str, *, page_size: int) -> KVCacheCapacity:
        if page_size < 1:
            raise ValueError("page size must be positive")
        # Every tensor-parallel rank logs the same capacity.
        values = {int(value) for value in _KV_CACHE_PAGES.findall(output)}
        if len(values) != 1:
            raise RuntimeError(f"expected one KV cache capacity, found {sorted(values)}")
        (pages,) = values
        return cls(pages=pages, tokens=pages * page_size)


def run_environment_verifier() -> dict[str, object]:
    command = (sys.executable, str(REPO_ROOT / "scripts" / "setup_env.py"), "verify")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    combined = f"{completed.stdout}\n{completed.stderr}"
    forbidden = find_forbidden_log_patterns(combined)
    if completed.returncode != 0 or forbidden:
        raise RuntimeError(
            f"environment verifier failed with {completed.returncode}, forbidden={forbidden}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {
        "command": list(command),
        "receipt": json.loads(completed.stdout),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
    }


def git_source_state() -> tuple[str, bool]:
    revision = subprocess.run(
        ("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "-C", str(REPO_ROOT), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, bool(status)


def current_cuda_uuid() -> str:
    """Return the visible CUDA device UUID in the form accepted by NVML."""
    import torch

    value = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    return value if value.upper().startswith("GPU-") else f"GPU-{value}"


def wait_ready(process: subprocess.Popen[bytes], port: int, timeout: float = 240) -> str:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                payload = json.load(response)
            return str(payload["data"][0]["id"])
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
        ):
            time.sleep(0.25)
    raise TimeoutError(f"server did not become ready within {timeout} seconds")


class ApiServerProcess:
    _SHUTDOWN_STEPS = (
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
        self.command = command
        self._env = env
        self.log_path = log_path
        self._log_file: TextIO | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self.clean_shutdown = False

    @property
    def process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError("API server has not started")
        return self._process

    def __enter__(self) -> ApiServerProcess:
        self._log_file = self.log_path.open("w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                self.command,
                cwd=REPO_ROOT,
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
        except Exception as error:  # noqa: BLE001 - preserve the benchmark error as primary
            cleanup_error = error
        try:
            if self._log_file is not None:
                self._log_file.close()
        except Exception as error:  # noqa: BLE001 - closing the evidence log is cleanup
            cleanup_error = cleanup_error or error
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
        graceful = True
        signalled = False
        for shutdown_signal, timeout in self._SHUTDOWN_STEPS:
            if process.poll() is not None:
                break
            try:
                os.killpg(process.pid, shutdown_signal)
            except ProcessLookupError:
                graceful = False
                break
            signalled = True
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                graceful = False

        if process.poll() is None:
            raise RuntimeError(f"API server {process.pid} resisted SIGKILL")
        deadline = time.monotonic() + 10
        while self._group_exists(process.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._group_exists(process.pid):
            graceful = False
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            else:
                deadline = time.monotonic() + 10
                while self._group_exists(process.pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
        if self._group_exists(process.pid):
            raise RuntimeError(f"API server group {process.pid} survived shutdown")
        self.clean_shutdown = signalled and graceful and process.returncode == 0


class NvmlDevice:
    def __init__(self, uuid: str) -> None:
        self._uuid = uuid
        self._nvml = None
        self._handle = None

    def __enter__(self) -> NvmlDevice:
        try:
            import pynvml
        except ImportError as error:
            raise RuntimeError(
                "the formal benchmark environment must provide nvidia-ml-py"
            ) from error
        pynvml.nvmlInit()
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByUUID(self._uuid)
            actual_uuid = self._text(pynvml.nvmlDeviceGetUUID(self._handle))
            if actual_uuid.casefold() != self._uuid.casefold():
                raise RuntimeError(
                    f"NVML resolved visible CUDA device {self._uuid} as {actual_uuid}"
                )
        except BaseException:
            self._handle = None
            pynvml.nvmlShutdown()
            raise
        self._nvml = pynvml
        return self

    def __exit__(self, *_: object) -> None:
        assert self._nvml is not None
        self._nvml.nvmlShutdown()
        self._handle = None
        self._nvml = None

    @property
    def uuid(self) -> str:
        return self._uuid

    def _call(self, function_name: str, *args: object):
        if self._nvml is None or self._handle is None:
            raise RuntimeError("NVML device is not open")
        return getattr(self._nvml, function_name)(self._handle, *args)

    def _telemetry(self, function_name: str, *args: object) -> dict[str, int | str | None]:
        if self._nvml is None:
            raise RuntimeError("NVML device is not open")
        try:
            value = int(self._call(function_name, *args))
        except AttributeError:
            return {"status": "unavailable", "value": None, "reason": "function-unavailable"}
        except self._nvml.NVMLError as error:
            return {
                "status": "unavailable",
                "value": None,
                "reason": type(error).__name__,
            }
        return {"status": "available", "value": value, "reason": None}

    def memory_info(self) -> dict[str, int | str | None]:
        """Read device-global physical memory without assuming NVML support."""

        if self._nvml is None:
            raise RuntimeError("NVML device is not open")
        try:
            info = self._call("nvmlDeviceGetMemoryInfo")
        except AttributeError:
            return {
                "status": "unavailable",
                "total_bytes": None,
                "used_bytes": None,
                "reason": "function-unavailable",
            }
        except self._nvml.NVMLError as error:
            return {
                "status": "unavailable",
                "total_bytes": None,
                "used_bytes": None,
                "reason": type(error).__name__,
            }
        return {
            "status": "available",
            "total_bytes": int(info.total),
            "used_bytes": int(info.used),
            "reason": None,
        }

    @staticmethod
    def _memory_value(
        info: dict[str, int | str | None],
        field: str,
    ) -> dict[str, int | str | None]:
        return {
            "status": info["status"],
            "value": info[field],
            "reason": info["reason"],
        }

    def used_memory(self) -> dict[str, int | str | None]:
        return self._memory_value(self.memory_info(), "used_bytes")

    def used_memory_bytes(self) -> int:
        observation = self.used_memory()
        value = observation["value"]
        if type(value) is not int:
            raise RuntimeError(
                f"NVML device memory telemetry is unavailable: {observation['reason']}"
            )
        return value

    def compute_pids(self) -> tuple[int, ...]:
        processes = self._call("nvmlDeviceGetComputeRunningProcesses")
        return tuple(sorted(int(process.pid) for process in processes))

    def wait_until_idle(self, timeout: float = 10) -> None:
        deadline = time.monotonic() + timeout
        pids = self.compute_pids()
        while pids and time.monotonic() < deadline:
            time.sleep(0.1)
            pids = self.compute_pids()
        if pids:
            raise RuntimeError(f"GPU has residual compute processes: {pids}")

    @staticmethod
    def _text(value: object) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    def metadata(self) -> dict[str, object]:
        assert self._nvml is not None
        major, minor = self._call("nvmlDeviceGetCudaComputeCapability")
        memory = self.memory_info()
        return {
            "cuda_logical_device": 0,
            "name": self._text(self._call("nvmlDeviceGetName")),
            "uuid": self._text(self._call("nvmlDeviceGetUUID")),
            "driver": self._text(self._nvml.nvmlSystemGetDriverVersion()),
            "compute_capability": [major, minor],
            "total_memory_bytes": memory["total_bytes"],
            "telemetry": {
                "total_memory_bytes": self._memory_value(memory, "total_bytes"),
                "power_limit_mw": self._telemetry("nvmlDeviceGetPowerManagementLimit"),
            },
        }

    def state(self) -> dict[str, dict[str, int | str | None]]:
        assert self._nvml is not None
        return {
            "used_memory_bytes": self.used_memory(),
            "temperature_c": self._telemetry(
                "nvmlDeviceGetTemperature", self._nvml.NVML_TEMPERATURE_GPU
            ),
            "graphics_clock_mhz": self._telemetry(
                "nvmlDeviceGetClockInfo", self._nvml.NVML_CLOCK_GRAPHICS
            ),
            "memory_clock_mhz": self._telemetry(
                "nvmlDeviceGetClockInfo", self._nvml.NVML_CLOCK_MEM
            ),
            "power_usage_mw": self._telemetry("nvmlDeviceGetPowerUsage"),
        }


NVML_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.002


class DeviceMemorySampler:
    def __init__(self, read_used_memory: Callable[[], int], *, interval: float) -> None:
        self._read_used_memory = read_used_memory
        self._interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._overall_peak = 0
        self._region_peak = 0
        self._samples: list[tuple[int, int]] = []
        self._error: Exception | None = None

    def __enter__(self) -> DeviceMemorySampler:
        if self._thread is not None:
            raise RuntimeError("GPU memory sampler is already running")
        self._stop.clear()
        self._error = None
        initial = self._read_used_memory()
        self._overall_peak = initial
        self._region_peak = initial
        self._samples = [(time.monotonic_ns(), initial)]
        self._thread = threading.Thread(target=self._sample, name="gpu-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        body_error: BaseException | None,
        body_traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 10))
            if self._thread.is_alive():
                self._error = RuntimeError("GPU memory sampler did not stop")
        self._thread = None
        if self._error is not None:
            if body_error is None:
                raise self._error
            raise body_error.with_traceback(body_traceback) from self._error

    def _sample(self) -> None:
        try:
            while not self._stop.wait(self._interval):
                self._record(self._read_used_memory())
        except Exception as error:  # noqa: BLE001 - surface sampler failures on the owner thread
            self._error = error
            self._stop.set()

    def _record(self, value: int) -> None:
        timestamp_ns = time.monotonic_ns()
        with self._lock:
            self._overall_peak = max(self._overall_peak, value)
            self._region_peak = max(self._region_peak, value)
            self._samples.append((timestamp_ns, value))

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("GPU memory sampling failed") from self._error

    def reset_region(self) -> None:
        self._check_error()
        value = self._read_used_memory()
        with self._lock:
            self._overall_peak = max(self._overall_peak, value)
            self._region_peak = value

    def region_peak_bytes(self) -> int:
        self._check_error()
        self._record(self._read_used_memory())
        with self._lock:
            return self._region_peak

    def overall_peak_bytes(self) -> int:
        self._check_error()
        self._record(self._read_used_memory())
        with self._lock:
            return self._overall_peak

    def samples_between(self, start_ns: int, end_ns: int) -> tuple[int, ...]:
        """Select physical-memory samples whose monotonic times fall in one child phase."""

        self._check_error()
        if type(start_ns) is not int or type(end_ns) is not int:
            raise TypeError("memory sample bounds must be integer monotonic timestamps")
        if start_ns >= end_ns:
            raise ValueError("memory sample start must precede its end")
        with self._lock:
            return tuple(
                value for timestamp_ns, value in self._samples if start_ns <= timestamp_ns <= end_ns
            )

    def sample_at_or_before(self, timestamp_ns: int) -> int | None:
        """Return the newest physical-memory sample no later than a phase boundary."""

        self._check_error()
        if type(timestamp_ns) is not int:
            raise TypeError("memory sample boundary must be an integer monotonic timestamp")
        with self._lock:
            return next(
                (
                    value
                    for sample_timestamp_ns, value in reversed(self._samples)
                    if sample_timestamp_ns <= timestamp_ns
                ),
                None,
            )


class DeviceMemoryMonitor:
    """Own optional NVML memory sampling without weakening sampler failures."""

    def __init__(self, device: NvmlDevice, *, interval: float) -> None:
        self._device = device
        self._interval = interval
        self._active = False
        self._telemetry: dict[str, int | str | None] | None = None
        self._sampler: DeviceMemorySampler | None = None

    def __enter__(self) -> DeviceMemoryMonitor:
        if self._active:
            raise RuntimeError("GPU memory monitor is already running")
        self._active = True
        try:
            self._telemetry = self._device.used_memory()
            if self.available:
                self._sampler = DeviceMemorySampler(
                    self._device.used_memory_bytes,
                    interval=self._interval,
                )
                self._sampler.__enter__()
        except BaseException:
            self._sampler = None
            self._active = False
            raise
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        body_error: BaseException | None,
        body_traceback: TracebackType | None,
    ) -> None:
        sampler = self._sampler
        try:
            if sampler is not None:
                sampler.__exit__(error_type, body_error, body_traceback)
        finally:
            self._active = False

    @property
    def available(self) -> bool:
        return self.telemetry["status"] == "available"

    @property
    def telemetry(self) -> dict[str, int | str | None]:
        if self._telemetry is None:
            raise RuntimeError("GPU memory monitor is not running")
        return dict(self._telemetry)

    def reset_region(self) -> None:
        if self._sampler is not None:
            self._sampler.reset_region()

    def region_peak_bytes(self) -> int | None:
        return None if self._sampler is None else self._sampler.region_peak_bytes()

    def overall_peak_bytes(self) -> int | None:
        return None if self._sampler is None else self._sampler.overall_peak_bytes()

    def samples_between(self, start_ns: int, end_ns: int) -> tuple[int, ...]:
        return () if self._sampler is None else self._sampler.samples_between(start_ns, end_ns)

    def sample_at_or_before(self, timestamp_ns: int) -> int | None:
        return None if self._sampler is None else self._sampler.sample_at_or_before(timestamp_ns)


def validate_post_run(
    server: ApiServerProcess,
    log_path: Path,
    device: NvmlDevice,
) -> None:
    validation_error: Exception | None = None
    try:
        if not log_path.exists():
            raise RuntimeError(f"server log was not created: {log_path}")
        validate_server_log(log_path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError) as error:
        validation_error = error

    if not server.clean_shutdown:
        error = RuntimeError(f"server did not shut down cleanly; inspect {log_path}")
        if validation_error is None:
            validation_error = error
        elif hasattr(validation_error, "add_note"):
            validation_error.add_note(str(error))

    try:
        device.wait_until_idle()
    except Exception as error:  # noqa: BLE001 - aggregate independent cleanup evidence
        if validation_error is None:
            validation_error = RuntimeError(f"GPU idle check failed: {error}")
        elif hasattr(validation_error, "add_note"):
            validation_error.add_note(f"GPU idle check also failed: {error}")

    if validation_error is not None:
        raise validation_error
