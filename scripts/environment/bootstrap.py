"""Create and configure the formal Conda environment."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .contract import (
    PROFILE_PATH,
    REPO_ROOT,
    SETUP_ENV_PATH,
    CudaDevice,
    DeviceQualification,
    KernelInstallTarget,
    NativeBuildLayout,
    VerificationError,
    cuda_driver_library_dir,
    load_project_install_requirements,
    load_runtime_profile,
    normalize_cuda_device_uuid,
    recorded_source_revision,
)
from .wheel_metadata import repair_cusparselt_arm64_tag

_PYTHON_LAYOUT_RECORD = "CTI_PYTHON_LAYOUT="
# One FlashAttention compile job: its nvcc threads, and the host memory it may peak at.
_FLASH_ATTENTION_NVCC_THREADS = 2
_FLASH_ATTENTION_JOB_GIB = 8


def _available_memory_gib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // (1 << 20)
    raise RuntimeError("/proc/meminfo has no MemAvailable")


def flash_attention_build_jobs() -> int:
    """MAX_JOBS for the FlashAttention build: the caller's MAX_JOBS, else as many jobs as this
    process's CPUs and the available memory hold."""

    if configured := os.environ.get("MAX_JOBS"):
        return int(configured)
    by_cpus = len(os.sched_getaffinity(0)) // _FLASH_ATTENTION_NVCC_THREADS
    by_memory = _available_memory_gib() // _FLASH_ATTENTION_JOB_GIB
    return max(1, min(by_cpus, by_memory))


@dataclass(frozen=True, slots=True)
class _PythonLayout:
    prefix: Path
    purelib: Path


class EnvironmentBootstrap:
    def __init__(
        self,
        conda: str,
        qualification: DeviceQualification | None = None,
        *,
        environment: str | None = None,
        kernel_wheel: Path | None = None,
        flash_attn_wheel: Path | None = None,
        no_gpu: bool = False,
        cuda_driver_dir: Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        if no_gpu and flash_attn_wheel is None:
            raise ValueError(
                "--no-gpu requires --flash-attn-wheel: a FlashAttention source build targets "
                "the visible GPUs"
            )
        if no_gpu != (cuda_driver_dir is not None):
            raise ValueError(
                "--no-gpu and --cuda-driver-dir go together: without a GPU there is no loaded "
                "driver, so the environment records where the target's libcuda.so will be"
            )
        self._conda = conda
        self._profile = load_runtime_profile()
        self._qualification = qualification
        self._kernel_wheel = None if kernel_wheel is None else kernel_wheel.resolve()
        self._flash_attn_wheel = None if flash_attn_wheel is None else flash_attn_wheel.resolve()
        self._no_gpu = no_gpu
        # Paths on the machine the environment runs on, kept unresolved.
        self._cuda_driver_dir = None if cuda_driver_dir is None else cuda_driver_dir.absolute()
        self._cache_dir = None if cache_dir is None else cache_dir.absolute()
        self._kernel_target: KernelInstallTarget | None = None
        self._project_requirements: tuple[str, ...] = ()
        if environment is not None and type(environment) is not str:
            raise TypeError("environment name must be a string")
        self._environment = self._profile.environment if environment is None else environment
        if not self._environment or self._environment != self._environment.strip():
            raise ValueError("environment name must be a non-empty trimmed string")
        if "/" in self._environment or "\\" in self._environment:
            raise ValueError("environment name must be a Conda name, not a path")

    def _run(
        self,
        command: Sequence[str],
        *,
        capture_output: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=capture_output,
            env=env,
        )

    def _pip_constraints(self) -> str:
        """Every pin verify checks; later installs may not move one."""
        pins = (*self._profile.distributions, ("pip", self._profile.pip_version))
        return "".join(f"{name}=={version}\n" for name, version in pins)

    def _pip_install(
        self,
        environment: str,
        constraints: Path,
        *arguments: str,
        env: Mapping[str, str],
    ) -> None:
        self._run(
            (
                self._conda,
                "run",
                "--no-capture-output",
                "-n",
                environment,
                "python",
                "-m",
                "pip",
                "--isolated",
                "install",
                "-c",
                str(constraints),
                *arguments,
            ),
            env=env,
        )

    def _existing_prefixes(self) -> tuple[Path, ...]:
        completed = self._run((self._conda, "env", "list", "--json"), capture_output=True)
        payload = json.loads(completed.stdout)
        return tuple(Path(path).resolve() for path in payload["envs"])

    def _visible_cuda_devices(self, environment: str) -> tuple[CudaDevice, ...]:
        code = """
import json
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print(json.dumps([
    {
        "capability": list(torch.cuda.get_device_capability(index)),
        "logical_index": index,
        "name": torch.cuda.get_device_name(index),
        "uuid": str(torch.cuda.get_device_properties(index).uuid),
    }
    for index in range(torch.cuda.device_count())
]))
"""
        completed = self._run(
            (
                self._conda,
                "run",
                "--no-capture-output",
                "-n",
                environment,
                "python",
                "-c",
                code,
            ),
            capture_output=True,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, list) or not payload:
            raise VerificationError("no CUDA device is visible while creating the environment")
        devices = tuple(
            CudaDevice(
                logical_index=item["logical_index"],
                name=item["name"],
                capability=(item["capability"][0], item["capability"][1]),
                uuid=normalize_cuda_device_uuid(item["uuid"]),
            )
            for item in payload
        )
        if self._qualification is not None:
            self._qualification.validate_devices(devices)
        return devices

    def _flash_attention_build_arches(
        self,
        prefix: Path,
        devices: Sequence[CudaDevice],
    ) -> tuple[str, ...]:
        completed = self._run(
            (str(prefix / "bin" / "nvcc"), "--list-gpu-arch"),
            capture_output=True,
        )
        compiler_arches = tuple(re.findall(r"^compute_(\d+)$", completed.stdout, re.MULTILINE))
        if not compiler_arches:
            raise VerificationError("nvcc did not report any supported CUDA architecture")
        return self._profile.flash_attention.select_build_arches(devices, compiler_arches)

    def _require_prebuilt_flash_attention_coverage(self, devices: Sequence[CudaDevice]) -> None:
        prebuilt = self._profile.flash_attention_prebuilt
        if not prebuilt.covers(devices):
            names = [f"{device.name} (SM{device.architecture})" for device in devices]
            raise VerificationError(
                f"the prebuilt FlashAttention wheel's images {prebuilt.cuda_arches} do not cover the "
                f"visible devices {names}; omit --flash-attn-wheel to build FlashAttention from source"
            )

    def _python_layout(self, environment: str) -> _PythonLayout:
        code = f"""
import json
import sys
import sysconfig

print({_PYTHON_LAYOUT_RECORD!r} + json.dumps({{
    "prefix": sys.prefix,
    "purelib": sysconfig.get_path("purelib"),
}}))
"""
        completed = self._run(
            (
                self._conda,
                "run",
                "--no-capture-output",
                "-n",
                environment,
                "python",
                "-c",
                code,
            ),
            capture_output=True,
        )
        records = tuple(
            line.removeprefix(_PYTHON_LAYOUT_RECORD)
            for line in completed.stdout.splitlines()
            if line.startswith(_PYTHON_LAYOUT_RECORD)
        )
        if len(records) != 1:
            raise VerificationError(
                f"target Python emitted {len(records)} layout records; expected exactly one"
            )
        try:
            payload = json.loads(records[0])
        except json.JSONDecodeError as error:
            raise VerificationError("target Python emitted an invalid layout record") from error
        if (
            not isinstance(payload, dict)
            or set(payload) != {"prefix", "purelib"}
            or not all(isinstance(value, str) and value for value in payload.values())
        ):
            raise VerificationError("target Python layout record has invalid fields")

        prefix = Path(payload["prefix"]).resolve()
        purelib = Path(payload["purelib"]).resolve()
        if prefix.name != environment:
            raise VerificationError(
                f"target Python prefix does not match environment {environment}: {prefix}"
            )
        if not purelib.is_relative_to(prefix):
            raise VerificationError(
                f"target Python purelib is outside its Conda prefix: prefix={prefix}, "
                f"purelib={purelib}"
            )
        return _PythonLayout(prefix=prefix, purelib=purelib)

    @staticmethod
    def _clean_install_environment() -> dict[str, str]:
        environment = os.environ.copy()
        for variable in (
            "CFLAGS",
            "CMAKE_ARGS",
            "CPPFLAGS",
            "CUDAARCHS",
            "CUDA_INC_PATH",
            "CUDACXX",
            "CUDAHOSTCXX",
            "CXXFLAGS",
            "FLASH_ATTENTION_FORCE_CXX11_ABI",
            "FLASH_ATTENTION_SKIP_CUDA_BUILD",
            "FLASH_ATTENTION_TRITON_AMD_ENABLE",
            "FLASH_ATTN_LOCAL_VERSION",
            "LDFLAGS",
            "LD_PRELOAD",
            "NVCC_APPEND_FLAGS",
            "NVCC_PREPEND_FLAGS",
            "PIP_CONFIG_FILE",
            "PIP_CONSTRAINT",
            "PIP_PREFIX",
            "PIP_REQUIREMENT",
            "PIP_TARGET",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONUSERBASE",
            "PYTORCH_NVCC",
            "TMS_CUDA_MAJOR",
            "TORCH_DONT_CHECK_COMPILER_ABI",
            "TORCH_CUDA_ARCH_LIST",
            "TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES",
        ):
            environment.pop(variable, None)
        return environment

    @staticmethod
    def _prefix_first_path(prefix: Path, inherited: str | None = None) -> str:
        prefix_bin = str(prefix / "bin")
        entries = (inherited or os.defpath).split(os.pathsep)
        return os.pathsep.join((prefix_bin, *(entry for entry in entries if entry != prefix_bin)))

    @classmethod
    def _verification_environment(cls, prefix: Path) -> dict[str, str]:
        """Run bootstrap-owned verification with the formal compiler first."""

        environment = cls._clean_install_environment()
        environment.update(
            {
                "CUDA_HOME": str(prefix),
                "PATH": cls._prefix_first_path(prefix, environment.get("PATH")),
                "PYTHONNOUSERSITE": "1",
            }
        )
        return environment

    def _install_torch_memory_saver(
        self,
        environment: str,
        prefix: Path,
        driver_library_dir: Path,
        layout: NativeBuildLayout,
        constraints: Path,
    ) -> None:
        library_path = os.pathsep.join(
            (str(layout.toolkit / "lib"), str(prefix / "lib"), str(driver_library_dir))
        )
        build_env = self._clean_install_environment()
        build_env.update(
            {
                "CC": str(layout.cc),
                "CPATH": str(layout.toolkit / "include"),
                "CUDA_HOME": str(prefix),
                "CXX": str(layout.cxx),
                "LD_LIBRARY_PATH": library_path,
                "LIBRARY_PATH": library_path,
                "PATH": os.pathsep.join((str(prefix / "bin"), build_env.get("PATH", ""))),
                "PYTHONNOUSERSITE": "1",
                "TMS_CUDA_MAJOR": "13",
            }
        )
        self._pip_install(
            environment,
            constraints,
            "--no-build-isolation",
            "--no-cache-dir",
            "--no-deps",
            "--force-reinstall",
            self._profile.torch_memory_saver.requirement,
            env=build_env,
        )

    def _install_flash_attention(
        self,
        environment: str,
        prefix: Path,
        driver_library_dir: Path,
        layout: NativeBuildLayout,
        cuda_arches: Sequence[str],
        constraints: Path,
    ) -> None:
        library_path = os.pathsep.join(
            (str(layout.toolkit / "lib"), str(prefix / "lib"), str(driver_library_dir))
        )
        build_env = self._clean_install_environment()
        build_env.update(
            {
                "BUILD_TARGET": "cuda",
                "CC": str(layout.cc),
                "CPATH": str(layout.toolkit / "include"),
                "CUDA_HOME": str(prefix),
                "CXX": str(layout.cxx),
                "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
                "FLASH_ATTN_CUDA_ARCHS": ";".join(cuda_arches),
                "LD_LIBRARY_PATH": library_path,
                "LIBRARY_PATH": library_path,
                "MAX_JOBS": str(flash_attention_build_jobs()),
                "NVCC_THREADS": str(_FLASH_ATTENTION_NVCC_THREADS),
                "PATH": os.pathsep.join((str(prefix / "bin"), build_env.get("PATH", ""))),
                "PYTHONNOUSERSITE": "1",
            }
        )
        self._pip_install(
            environment,
            constraints,
            "--no-build-isolation",
            "--no-cache-dir",
            "--no-deps",
            "--no-binary=flash-attn",
            self._profile.flash_attention.requirement,
            env=build_env,
        )

    def _install_prebuilt_flash_attention(
        self,
        environment: str,
        wheel: Path,
        constraints: Path,
        env: Mapping[str, str],
    ) -> None:
        self._pip_install(
            environment,
            constraints,
            "--no-build-isolation",
            "--no-cache-dir",
            "--no-deps",
            self._profile.flash_attention_prebuilt.requirement(wheel, platform.machine()),
            env=env,
        )

    def _configure(self, environment: str) -> Path:
        python = self._python_layout(environment)
        prefix = python.prefix
        repair_cusparselt_arm64_tag(
            prefix,
            python.purelib,
            self._profile.distribution_version("nvidia-cusparselt-cu13"),
        )
        layout = NativeBuildLayout.discover(prefix)
        if self._no_gpu:
            # No driver to load: the environment records the target's driver directory, and
            # torch-memory-saver links against the toolkit's libcuda stub.
            driver_library_dir = self._cuda_driver_dir
            link_library_dir = layout.toolkit / "lib" / "stubs"
        else:
            driver_library_dir = link_library_dir = cuda_driver_library_dir()
            devices = self._visible_cuda_devices(environment)
            if self._flash_attn_wheel is None:
                flash_attention_arches = self._flash_attention_build_arches(prefix, devices)
            else:
                self._require_prebuilt_flash_attention_coverage(devices)
        cache_root = (self._cache_dir or prefix / "var" / "cache" / "cti") / "cu130-torch213"
        library_path = os.pathsep.join(
            (
                str(prefix / "lib"),
                str(prefix / "lib64"),
                str(layout.toolkit / "lib"),
                str(driver_library_dir),
            )
        )
        self._run(
            (
                self._conda,
                "env",
                "config",
                "vars",
                "set",
                "-n",
                environment,
                f"CUDA_HOME={prefix}",
                # The host's CUDA target headers (cublasLt.h and the rest) for JIT builds:
                # targets/x86_64-linux, or targets/sbsa-linux on aarch64.
                f"CPATH={layout.toolkit / 'include'}",
                f"LD_LIBRARY_PATH={library_path}",
                f"LIBRARY_PATH={library_path}",
                "CTI_TORCH_BACKEND=cu130",
                "FLASHINFER_WORKSPACE_BASE=" + str(cache_root / "flashinfer"),
                "TVM_FFI_CACHE_DIR=" + str(cache_root / "tvm-ffi"),
                "TORCH_EXTENSIONS_DIR=" + str(cache_root / "torch-extensions"),
                "TORCHINDUCTOR_CACHE_DIR=" + str(cache_root / "torch-inductor"),
                "CUTE_DSL_CACHE_DIR=" + str(cache_root / "cute-dsl"),
            )
        )
        install_env = self._clean_install_environment()
        install_env["PYTHONNOUSERSITE"] = "1"
        assert self._kernel_target is not None, "kernel target must be validated before creation"
        with tempfile.TemporaryDirectory(prefix="cti-pip-") as directory:
            constraints = Path(directory) / "constraints.txt"
            constraints.write_text(self._pip_constraints(), encoding="utf-8")
            self._install_torch_memory_saver(
                environment, prefix, link_library_dir, layout, constraints
            )
            if self._flash_attn_wheel is None:
                self._install_flash_attention(
                    environment,
                    prefix,
                    driver_library_dir,
                    layout,
                    flash_attention_arches,
                    constraints,
                )
            else:
                self._install_prebuilt_flash_attention(
                    environment, self._flash_attn_wheel, constraints, install_env
                )
            self._pip_install(
                environment,
                constraints,
                "--no-build-isolation",
                "--only-binary=cti-kernels",
                *self._project_requirements,
                *self._kernel_target.install_args,
                env=install_env,
            )
            self._pip_install(
                environment,
                constraints,
                "--no-build-isolation",
                "--no-deps",
                "-e",
                str(REPO_ROOT),
                env=install_env,
            )
        return prefix

    def _verification_command(
        self, environment: str, *, kernel_wheel: bool = True
    ) -> tuple[str, ...]:
        verify_command = [
            self._conda,
            "run",
            "--no-capture-output",
            "-n",
            environment,
            "python",
            str(SETUP_ENV_PATH),
            "verify",
        ]
        if kernel_wheel and self._kernel_wheel is not None:
            verify_command.extend(("--kernel-wheel", str(self._kernel_wheel)))
        if self._qualification is not None:
            verify_command.extend(("--qualification", self._qualification.name))
        return tuple(verify_command)

    def _verify(self, environment: str, prefix: Path) -> None:
        self._run(
            self._verification_command(environment),
            env=self._verification_environment(prefix),
        )

    def _require_source_checkout(self) -> None:
        """verify records the checkout's revision: refuse before the build, not after it."""

        if recorded_source_revision(REPO_ROOT) is not None:
            return
        try:
            self._run(("git", "rev-parse", "HEAD"), capture_output=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"{REPO_ROOT} is not a git checkout and has no REVISION file; verify records "
                "its revision, so create needs one"
            ) from error

    def create(self) -> None:
        environment = self._environment
        self._kernel_target = KernelInstallTarget.load(
            repo_root=REPO_ROOT, wheel=self._kernel_wheel
        )
        if self._flash_attn_wheel is not None:
            self._profile.flash_attention_prebuilt.validate_wheel(
                self._flash_attn_wheel, platform.machine()
            )
        self._project_requirements = load_project_install_requirements(REPO_ROOT / "pyproject.toml")
        self._require_source_checkout()
        existing = [prefix for prefix in self._existing_prefixes() if prefix.name == environment]
        if existing:
            raise RuntimeError(
                f"{environment} already exists at {existing[0]}; "
                "remove it explicitly before recreating"
            )

        owns_environment = False
        try:
            self._run(
                (
                    self._conda,
                    "env",
                    "create",
                    "--name",
                    environment,
                    "--file",
                    str(PROFILE_PATH),
                ),
                # Strict channel priority (Miniforge's default) excludes a package from every
                # channel below the first one carrying its name, but the profile pins builds
                # from lower channels (nvidia's cuda-tileiras). Its exact pins leave flexible
                # priority nothing else to choose.
                env={**os.environ, "CONDA_CHANNEL_PRIORITY": "flexible"},
            )
            owns_environment = True
            prefix = self._configure(environment)
        except BaseException:
            if owns_environment:
                try:
                    self._run((self._conda, "env", "remove", "-n", environment, "--yes"))
                except (OSError, subprocess.SubprocessError) as cleanup_error:
                    print(
                        f"warning: failed to remove incomplete environment {environment}: "
                        f"{cleanup_error}",
                        file=sys.stderr,
                    )
            raise

        if self._no_gpu:
            # The GPU host need not have the wheel, a build input: verify checks its version.
            command = self._verification_command(environment, kernel_wheel=False)
            print(
                f"Created {environment} without a GPU, so it is not verified yet. On a GPU host, "
                f"run:\n  {shlex.join(command)}"
            )
            return
        try:
            self._verify(environment, prefix)
        except BaseException:
            print(
                f"{environment} was installed but did not pass verification; "
                "the environment has been preserved. After addressing the error, rerun:\n"
                f"  {shlex.join(self._verification_command(environment))}",
                file=sys.stderr,
            )
            raise

        print(f"Created {environment}. Activate it with: conda activate {environment}")
