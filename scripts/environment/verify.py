"""Fail-closed audit of an installed formal environment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, distribution, distributions, version
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote, urlsplit

import tomllib
from packaging.utils import canonicalize_name

from .contract import (
    FLASH_ATTENTION_PREBUILT_PATH,
    FLASH_ATTENTION_SOURCE_PATH,
    REPO_ROOT,
    CudaDevice,
    DeviceQualification,
    KernelInstallTarget,
    NativeBuildLayout,
    RuntimeProfile,
    VerificationError,
    cuda_binary_covers,
    cuda_driver_library_dir,
    load_runtime_profile,
    normalize_cuda_device_uuid,
    recorded_source_revision,
    required_kernel_version,
    tms_preload_path,
    versions_equal,
)

_FORBIDDEN_SUBPROCESS_PATTERNS = (
    "Traceback",
    "CUDA error",
    "illegal memory",
    "Task was destroyed",
    "never awaited",
    "exception was never retrieved",
    "Exception in callback",
    "resource_tracker",
    "leaked semaphore",
    "Exception ignored",
    "Segmentation fault",
    "Aborted",
)


class EnvironmentVerifier:
    def __init__(
        self,
        repo_root: Path = REPO_ROOT,
        profile: RuntimeProfile | None = None,
        qualification: DeviceQualification | None = None,
        *,
        kernel_wheel: Path | None = None,
    ) -> None:
        self._repo_root = repo_root.resolve()
        self._prefix = Path(sys.prefix).resolve()
        self._profile = profile or load_runtime_profile()
        self._qualification = qualification
        self._kernel_wheel = None if kernel_wheel is None else kernel_wheel.resolve()

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise VerificationError(message)

    def _verify_conda_package(self, name: str, expected: str) -> None:
        """One Conda marker of ``name`` at ``expected``, a ``version`` or ``version=build``."""

        conda_meta = self._prefix / "conda-meta"
        package_version, _, build = expected.partition("=")
        matches = tuple(conda_meta.glob(f"{name}-{package_version}-{build or '*'}.json"))
        self._require(
            len(matches) == 1,
            f"expected one {name}={expected} marker under {conda_meta}, found {matches}",
        )

    def _verify_cti_source(self, package: ModuleType) -> tuple[Path, ...]:
        expected_package = (self._repo_root / "python" / "cti").resolve()
        return self._verify_package_source(package, expected_package, "cti")

    def _verify_engine_distribution(self, installed: str | None, engine_version: str) -> None:
        """The engine runs from this checkout (``_verify_cti_source``), installed or put
        on ``PYTHONPATH``; an installed distribution, if any, is this checkout's version."""

        self._require(
            installed is None or versions_equal(installed, engine_version),
            f"installed cti {installed} is not this checkout's version {engine_version}",
        )

    def _verify_package_source(
        self, package: ModuleType, expected_package: Path, name: str
    ) -> tuple[Path, ...]:
        expected_file = expected_package / "__init__.py"
        package_file = getattr(package, "__file__", None)
        self._require(isinstance(package_file, str), f"{name} must be a regular package")
        resolved_file = Path(package_file).resolve()
        self._require(
            resolved_file == expected_file,
            f"{name} resolves to {resolved_file}, expected {expected_file}",
        )
        package_paths = tuple(Path(path).resolve() for path in getattr(package, "__path__", ()))
        self._require(
            package_paths == (expected_package,),
            f"{name} package paths are {package_paths}, expected only {expected_package}",
        )
        return package_paths

    def _verify_kernel_source(self, package: ModuleType) -> dict[str, Any]:
        expected_version = required_kernel_version(self._repo_root / "pyproject.toml")
        purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
        self._require(purelib.is_relative_to(self._prefix), "kernel purelib is outside the prefix")
        # Source PYTHONPATH entries may expose setuptools egg-info. Inspect the
        # actual installed distribution, not metadata shadowed by the checkout.
        matches = [
            metadata
            for metadata in distributions(path=[str(purelib)])
            if canonicalize_name(metadata.metadata.get("Name", "")) == "cti-kernels"
        ]
        self._require(len(matches) == 1, "expected one installed cti-kernels distribution")
        metadata = matches[0]
        self._require(
            versions_equal(metadata.version, expected_version),
            f"installed cti-kernels must be {expected_version}, found {metadata.version}",
        )
        direct_url_text = metadata.read_text("direct_url.json")
        # Index installs have no direct_url.json. Source-built noneditable wheels
        # are also valid: installed identity and ownership, not origin, are the gate.
        direct_url = None
        source_url = sha256 = None
        if direct_url_text is not None:
            try:
                direct_url = json.loads(direct_url_text)
            except json.JSONDecodeError as error:
                raise VerificationError("cti-kernels direct_url.json is invalid") from error
            self._require(isinstance(direct_url, Mapping), "invalid cti-kernels direct URL record")
            source_url = direct_url.get("url")
            self._require(
                isinstance(source_url, str) and bool(source_url),
                "cti-kernels provenance has no source URL",
            )
            directory = direct_url.get("dir_info")
            self._require(
                not isinstance(directory, Mapping) or not directory.get("editable", False),
                "cti-kernels must be a noneditable installed package",
            )
            archive = direct_url.get("archive_info")
            hashes = archive.get("hashes") if isinstance(archive, Mapping) else None
            sha256 = hashes.get("sha256") if isinstance(hashes, Mapping) else None
            self._require(
                sha256 is None
                or (isinstance(sha256, str) and re.fullmatch(r"[0-9a-f]{64}", sha256) is not None),
                "cti-kernels has invalid SHA256 provenance",
            )
        if self._kernel_wheel is not None:
            target = KernelInstallTarget.load(repo_root=self._repo_root, wheel=self._kernel_wheel)
            self._require(
                sha256 is not None and sha256 == target.sha256,
                "installed cti-kernels wheel hash mismatch or missing SHA256 provenance",
            )
        self._require(
            any(str(path) == "ctikernels/__init__.py" for path in (metadata.files or ())),
            "cti-kernels wheel RECORD does not own its package",
        )
        expected_package = Path(metadata.locate_file("ctikernels")).resolve()
        self._require(
            expected_package.is_relative_to(purelib), "cti-kernels package is outside purelib"
        )
        paths = self._verify_package_source(package, expected_package, "ctikernels")
        return {
            "version": expected_version,
            "mode": "wheel",
            "package": [str(path) for path in paths],
            "source_url": source_url,
            "wheel_sha256": (
                sha256
                if source_url and unquote(urlsplit(source_url).path).endswith(".whl")
                else None
            ),
            "direct_url": direct_url,
        }

    def _resolved_conda_packages(self) -> dict[str, str]:
        packages: dict[str, str] = {}
        conda_meta = self._prefix / "conda-meta"
        for marker in sorted(conda_meta.glob("*.json")):
            record = json.loads(marker.read_text(encoding="utf-8"))
            name = record["name"]
            self._require(
                name not in packages,
                f"duplicate Conda package record for {name} under {conda_meta}",
            )
            packages[name] = f"{record['version']}={record['build']}"
        return dict(sorted(packages.items()))

    def _verify_jit_environment(self, nvcc: Path) -> Path:
        cuda_home = os.environ.get("CUDA_HOME")
        self._require(
            cuda_home is not None and Path(cuda_home).resolve() == self._prefix,
            f"CUDA_HOME must point to {self._prefix}; activate the formal Conda environment",
        )
        selected_nvcc = shutil.which("nvcc")
        self._require(
            selected_nvcc is not None and Path(selected_nvcc).resolve() == nvcc.resolve(),
            f"PATH must select {nvcc}; found {selected_nvcc}",
        )

        driver_library_dir = cuda_driver_library_dir().resolve()
        for variable in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
            paths = {
                Path(entry).resolve()
                for entry in os.environ.get(variable, "").split(os.pathsep)
                if entry
            }
            self._require(
                driver_library_dir in paths,
                f"{variable} must contain CUDA driver directory {driver_library_dir}",
            )
        return driver_library_dir

    def _verify_nvcc(self, nvcc: Path) -> str:
        release = ".".join(self._profile.cuda_toolkit.split(".")[:2])
        output = self._run((str(nvcc), "--version"))
        self._require(
            f"release {release}," in output, f"expected nvcc release {release}, found: {output}"
        )
        return output

    @staticmethod
    def _run(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> str:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            raise VerificationError(
                f"command timed out after {error.timeout}s: {' '.join(command)}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            ) from error
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)
        if completed.returncode != 0:
            raise VerificationError(
                f"command failed ({completed.returncode}): {' '.join(command)}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        combined = f"{completed.stdout}\n{completed.stderr}"
        forbidden = [pattern for pattern in _FORBIDDEN_SUBPROCESS_PATTERNS if pattern in combined]
        if forbidden:
            raise VerificationError(
                f"command emitted forbidden runtime diagnostics {forbidden}: {' '.join(command)}"
            )
        return completed.stdout.strip()

    def _verify_tms_preload(self, preload: Path) -> None:
        code = """
import torch
from torch_memory_saver import TorchMemorySaver

saver = TorchMemorySaver()
with saver.region(tag="verify", enable_cpu_backup=True):
    tensor = torch.arange(1024, device="cuda", dtype=torch.float32)
pointer = tensor.data_ptr()
saver.pause("verify")
saver.resume("verify")
torch.cuda.synchronize()
if tensor.data_ptr() != pointer:
    raise RuntimeError("torch-memory-saver changed the allocation pointer after resume")
if float(tensor[17]) != 17.0:
    raise RuntimeError("torch-memory-saver did not restore the allocation contents")
"""
        env = os.environ.copy()
        for variable in (
            "LD_PRELOAD",
            "PYTHONHOME",
            "PYTHONOPTIMIZE",
            "PYTHONPATH",
            "PYTHONUSERBASE",
        ):
            env.pop(variable, None)
        env["LD_PRELOAD"] = str(preload)
        env["PYTHONNOUSERSITE"] = "1"
        self._run((sys.executable, "-c", code), env=env)

    def _verify_tms_package(self, package: ModuleType) -> Path:
        purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
        expected_package = purelib / "torch_memory_saver"
        expected_file = expected_package / "__init__.py"
        package_file = getattr(package, "__file__", None)
        self._require(isinstance(package_file, str), "torch-memory-saver has no package path")
        resolved_file = Path(package_file).resolve()
        self._require(
            resolved_file == expected_file,
            f"torch-memory-saver resolves to {resolved_file}, expected {expected_file}",
        )
        package_paths = tuple(Path(path).resolve() for path in package.__path__)
        self._require(
            package_paths == (expected_package,),
            "torch-memory-saver package paths are "
            f"{package_paths}, expected only {expected_package}",
        )
        return resolved_file

    def _verify_tms_source(self, package: ModuleType) -> dict[str, Any]:
        lock = self._profile.torch_memory_saver
        metadata = distribution("torch-memory-saver")
        direct_url_text = metadata.read_text("direct_url.json")
        self._require(
            direct_url_text is not None,
            "torch-memory-saver must retain its locked VCS provenance",
        )
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError as error:
            raise VerificationError("torch-memory-saver direct_url.json is invalid") from error
        self._require(
            isinstance(direct_url, Mapping),
            "invalid torch-memory-saver direct URL record",
        )
        lock.validate_direct_url(direct_url)
        metadata_path = Path(metadata.locate_file("")).resolve()
        self._require(
            metadata_path.is_relative_to(self._prefix),
            f"torch-memory-saver metadata is outside {self._prefix}: {metadata_path}",
        )
        package_path = self._verify_tms_package(package)
        purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
        binaries = {}
        for mode in ("preload", "torch"):
            expected = purelib / f"torch_memory_saver_hook_mode_{mode}_cu13.abi3.so"
            matches = tuple(sorted(purelib.glob(f"torch_memory_saver_hook_mode_{mode}*.so")))
            self._require(
                matches == (expected,),
                f"expected only the CUDA 13 TMS {mode} binary, found {matches}",
            )
            self._require(
                expected.is_file() and expected.is_relative_to(self._prefix),
                f"torch-memory-saver {mode} binary is outside {self._prefix}: {expected}",
            )
            with expected.open("rb") as file:
                sha256 = hashlib.file_digest(file, "sha256").hexdigest()
            binaries[mode] = {"path": str(expected), "sha256": sha256}
        return {
            "binaries": binaries,
            "commit": lock.commit,
            "metadata": str(metadata_path),
            "package": str(package_path),
            "repository": lock.repository,
        }

    def _verify_flash_attention_binary(
        self,
        extension: Path,
        devices: Sequence[CudaDevice],
    ) -> dict[str, Any]:
        self._require(
            extension.is_relative_to(self._prefix),
            f"flash-attn native extension is outside {self._prefix}: {extension}",
        )
        cuobjdump = self._prefix / "bin" / "cuobjdump"
        readelf = NativeBuildLayout.discover(self._prefix).readelf
        self._require(cuobjdump.is_file(), f"cuobjdump is missing from {cuobjdump}")
        self._require(readelf.is_file(), f"readelf is missing from {readelf}")

        image_output = self._run((str(cuobjdump), "--list-elf", str(extension)))
        native_arches = tuple(sorted(set(re.findall(r"\.sm_(\d+)\.cubin", image_output))))
        native_image_count = len(re.findall(r"\.sm_\d+\.cubin", image_output))
        self._require(native_image_count > 0, "FlashAttention contains no native CUDA images")
        unsupported_devices = [
            f"{device.name} (SM{device.architecture})"
            for device in devices
            if not cuda_binary_covers(device.capability, native_arches)
        ]
        self._require(
            not unsupported_devices,
            f"FlashAttention images {native_arches} do not cover {unsupported_devices}",
        )
        if self._qualification is not None:
            self._qualification.validate_flash_attention(native_arches)

        dynamic_output = self._run((str(readelf), "-d", str(extension)))
        rpaths = tuple(
            entry
            for value in re.findall(r"Library (?:rpath|runpath): \[([^]]+)\]", dynamic_output)
            for entry in value.split(":")
            if entry
        )
        self._require(rpaths, "flash-attn native extension has no runtime library path")
        for entry in rpaths:
            resolved_entry = entry.replace("$ORIGIN", str(extension.parent), 1)
            self._require(
                Path(resolved_entry).resolve().is_relative_to(self._prefix),
                f"flash-attn native extension references a foreign prefix: {entry}",
            )

        with extension.open("rb") as file:
            extension_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
        return {
            "extension": str(extension),
            "extension_sha256": extension_sha256,
            "native_arches": list(native_arches),
            "native_image_count": native_image_count,
            "rpaths": list(rpaths),
        }

    def _flash_attention_provenance(self, direct_url: Mapping[str, Any]) -> dict[str, Any]:
        """The locked source build, or the prebuilt wheel pinned for this machine."""

        lock = self._profile.flash_attention
        if direct_url.get("url") == lock.url:
            lock.validate_direct_url(direct_url)
            return {"provenance": "source", "source_sha256": lock.sha256, "source_url": lock.url}
        prebuilt = self._profile.flash_attention_prebuilt
        machine = platform.machine()
        try:
            prebuilt.validate_direct_url(direct_url, machine)
        except VerificationError as error:
            raise VerificationError(
                f"flash-attn from {direct_url.get('url')!r} is neither the locked source "
                f"({FLASH_ATTENTION_SOURCE_PATH.name}) nor the prebuilt wheel pinned for "
                f"{machine} ({FLASH_ATTENTION_PREBUILT_PATH.name})"
            ) from error
        wheel, sha256 = prebuilt.wheel_for(machine)
        return {"provenance": "prebuilt", "wheel": wheel, "wheel_sha256": sha256}

    def _verify_flash_attention(self, devices: Sequence[CudaDevice]) -> dict[str, Any]:
        lock = self._profile.flash_attention
        metadata = distribution("flash-attn")
        direct_url_text = metadata.read_text("direct_url.json")
        self._require(
            direct_url_text is not None,
            "flash-attn must retain its locked source or prebuilt wheel provenance",
        )
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError as error:
            raise VerificationError("flash-attn direct_url.json is invalid") from error
        self._require(isinstance(direct_url, Mapping), "invalid flash-attn direct URL record")
        provenance = self._flash_attention_provenance(direct_url)

        import flash_attn
        import flash_attn_2_cuda

        from ctikernels.attention.varlen import CausalVarlenAttention

        self._require(
            flash_attn.__version__ == lock.version,
            "flash-attn import/distribution mismatch",
        )
        try:
            CausalVarlenAttention()
        except RuntimeError as error:
            raise VerificationError(str(error)) from error

        package_file = getattr(flash_attn, "__file__", None)
        self._require(isinstance(package_file, str), "flash-attn package has no source path")
        package_path = Path(package_file).resolve()
        metadata_path = Path(metadata.locate_file("")).resolve()
        for label, path in (("package", package_path), ("metadata", metadata_path)):
            self._require(
                path.is_relative_to(self._prefix),
                f"flash-attn {label} is outside {self._prefix}: {path}",
            )

        extension = Path(flash_attn_2_cuda.__file__).resolve()
        binary = self._verify_flash_attention_binary(extension, devices)
        return {
            **binary,
            **provenance,
            "metadata": str(metadata_path),
            "package": str(package_path),
            "public_operation": "ctikernels.attention.varlen.CausalVarlenAttention",
        }

    def _source_state(self) -> tuple[str, bool | None]:
        """The checkout's revision and whether it has uncommitted changes, which is unknown
        (None) for a tree that only records its revision, as the container image's does."""

        if (revision := recorded_source_revision(self._repo_root)) is not None:
            return revision, None
        git = ("git", "-C", str(self._repo_root))
        revision = self._run((*git, "rev-parse", "HEAD"))
        return revision, bool(self._run((*git, "status", "--porcelain")))

    def _visible_cuda_devices(self, torch: ModuleType) -> tuple[CudaDevice, ...]:
        self._require(torch.cuda.is_available(), "CUDA is unavailable")
        devices = tuple(
            CudaDevice(
                logical_index=index,
                name=torch.cuda.get_device_name(index),
                capability=tuple(torch.cuda.get_device_capability(index)),
                uuid=normalize_cuda_device_uuid(torch.cuda.get_device_properties(index).uuid),
            )
            for index in range(torch.cuda.device_count())
        )
        self._require(devices, "no CUDA device is visible")
        torch_arches = tuple(
            architecture.removeprefix("sm_")
            for architecture in torch.cuda.get_arch_list()
            if architecture.startswith("sm_")
        )
        unsupported_devices = [
            f"{device.name} (SM{device.architecture})"
            for device in devices
            if not cuda_binary_covers(device.capability, torch_arches)
        ]
        self._require(
            not unsupported_devices,
            f"PyTorch CUDA images {torch_arches} do not cover {unsupported_devices}",
        )
        if self._qualification is not None:
            self._qualification.validate_devices(devices)
        return devices

    def verify(self) -> dict[str, Any]:
        profile = self._profile
        self._require(
            sys.version_info[:3] == profile.python,
            f"expected Python {profile.python}, found {sys.version_info[:3]}",
        )

        installed: dict[str, str] = {"pip": version("pip")}
        for name, _ in profile.distributions:
            try:
                installed[name] = version(name)
            except PackageNotFoundError:
                installed[name] = "missing"
        profile.validate_versions(installed)
        self._require(
            installed["pip"] == profile.pip_version,
            f"expected pip {profile.pip_version}, found {installed['pip']}",
        )
        for legacy_name in ("sgl-kernel", "minisgl"):
            try:
                legacy_version = version(legacy_name)
            except PackageNotFoundError:
                continue
            raise VerificationError(
                f"legacy {legacy_name} distribution is installed: {legacy_version}"
            )

        self._verify_conda_package("pip", profile.pip_version)
        self._verify_conda_package("cuda-toolkit", profile.cuda_toolkit)
        self._verify_conda_package("cuda-tileiras", profile.cuda_tileiras)
        # Every Conda GCC build requires its own libsanitizer, so this holds GCC too.
        self._verify_conda_package("libsanitizer", profile.libsanitizer)
        # The glibc that native and JIT builds link against, at most the hosts' 2.35.
        self._verify_conda_package("sysroot_linux-64", profile.sysroot)
        self._verify_conda_package("kernel-headers_linux-64", profile.kernel_headers)

        nvcc = self._prefix / "bin" / "nvcc"
        self._require(nvcc.is_file(), f"nvcc is missing from {nvcc}")
        driver_library_dir = self._verify_jit_environment(nvcc)
        nvcc_output = self._verify_nvcc(nvcc)

        import flashinfer
        import sgl_kernel
        import torch
        import torch_memory_saver as tms_package
        import triton

        import cti
        import ctikernels

        self._require(torch.__version__ == installed["torch"], "torch import/distribution mismatch")
        self._require(torch.version.cuda == profile.cuda_runtime, "unexpected torch CUDA runtime")
        self._require(
            triton.__version__ == installed["triton"], "triton import/distribution mismatch"
        )
        self._require(
            flashinfer.__version__ == installed["flashinfer-python"],
            "FlashInfer import/distribution mismatch",
        )
        self._require(torch.compiled_with_cxx11_abi(), "PyTorch was not built with CXX11 ABI")
        devices = self._visible_cuda_devices(torch)
        flash_attention = self._verify_flash_attention(devices)

        common_ops = sgl_kernel.common_ops
        common_ops_path = Path(common_ops.__file__).resolve()
        cti_paths = self._verify_cti_source(cti)
        kernel_installation = self._verify_kernel_source(ctikernels)

        preload = tms_preload_path()
        self._require(
            preload.is_file(), f"CUDA 13 torch-memory-saver preload is missing: {preload}"
        )
        torch_memory_saver = self._verify_tms_source(tms_package)
        self._verify_tms_preload(preload)

        pip_check = self._run((sys.executable, "-m", "pip", "check"))
        pip_list = json.loads(self._run((sys.executable, "-m", "pip", "list", "--format=json")))
        resolved = {canonicalize_name(item["name"]): item["version"] for item in pip_list}
        with (self._repo_root / "pyproject.toml").open("rb") as file:
            engine_version = tomllib.load(file)["project"]["version"]
        self._verify_engine_distribution(resolved.get("cti"), engine_version)
        self._require(
            versions_equal(resolved.get("cti-kernels"), kernel_installation["version"]),
            "installed cti-kernels does not match the engine dependency",
        )
        resolved_json = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
        conda_packages = self._resolved_conda_packages()
        conda_json = json.dumps(conda_packages, sort_keys=True, separators=(",", ":"))
        conda_history = (self._prefix / "conda-meta" / "history").read_bytes()
        source_revision, source_dirty = self._source_state()
        return {
            "environment": self._prefix.name,
            "profile": profile.environment,
            "python": platform.python_version(),
            "prefix": str(self._prefix),
            "cuda_toolkit": profile.cuda_toolkit,
            "nvcc": nvcc_output.splitlines()[-1],
            "cuda_driver_library_dir": str(driver_library_dir),
            "torch_cuda": torch.version.cuda,
            "host_machine": platform.machine(),
            "qualification": None if self._qualification is None else self._qualification.name,
            "device": devices[0].name,
            "device_uuid": devices[0].uuid,
            "cuda_logical_device": 0,
            "cuda_visible_device_count": len(devices),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "capability": list(devices[0].capability),
            "cuda_devices": [
                {
                    "capability": list(device.capability),
                    "logical_index": device.logical_index,
                    "name": device.name,
                    "uuid": device.uuid,
                }
                for device in devices
            ],
            "cxx11_abi": True,
            "distributions": installed,
            "flash_attention": flash_attention,
            "resolved_distributions": dict(sorted(resolved.items())),
            "resolved_distributions_sha256": hashlib.sha256(resolved_json.encode()).hexdigest(),
            "conda_packages": conda_packages,
            "resolved_conda_sha256": hashlib.sha256(conda_json.encode()).hexdigest(),
            "conda_history_sha256": hashlib.sha256(conda_history).hexdigest(),
            "source_revision": source_revision,
            "source_dirty": source_dirty,
            "cti": [str(path) for path in cti_paths],
            "ctikernels": kernel_installation["package"],
            "kernel_installation": kernel_installation,
            "sglang_common_ops": str(common_ops_path),
            "torch_memory_saver": torch_memory_saver,
            "tms_preload": str(preload),
            "pip_check": pip_check,
        }
