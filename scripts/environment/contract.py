"""Formal environment contracts and platform facts."""

from __future__ import annotations

import ctypes
import hashlib
import platform
import re
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_ENV_PATH = REPO_ROOT / "scripts" / "setup_env.py"
PROFILE_PATH = REPO_ROOT / "environments" / "benchmark-cu130-torch213.yml"
FLASH_ATTENTION_SOURCE_PATH = REPO_ROOT / "environments" / "flash-attention-source.txt"
FLASH_ATTENTION_PREBUILT_PATH = REPO_ROOT / "environments" / "flash-attention-prebuilt.toml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
QUALIFICATIONS_PATH = REPO_ROOT / "environments" / "gpu-qualifications.toml"


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CudaDevice:
    logical_index: int
    name: str
    capability: tuple[int, int]
    uuid: str

    @property
    def architecture(self) -> str:
        return f"{self.capability[0]}{self.capability[1]}"


@dataclass(frozen=True)
class DeviceQualification:
    name: str
    visible_device_count: int
    device_name_contains: str
    capability: tuple[int, int]
    flash_attention_arches: tuple[str, ...]

    def validate_devices(self, devices: Sequence[CudaDevice]) -> None:
        if len(devices) != self.visible_device_count:
            raise VerificationError(
                f"qualification {self.name!r} requires {self.visible_device_count} visible "
                f"GPU(s), found {len(devices)}"
            )
        for device in devices:
            if self.device_name_contains not in device.name:
                raise VerificationError(
                    f"qualification {self.name!r} requires a device containing "
                    f"{self.device_name_contains!r}, found {device.name!r}"
                )
            if device.capability != self.capability:
                raise VerificationError(
                    f"qualification {self.name!r} requires SM{self.capability[0]}"
                    f"{self.capability[1]}, found SM{device.architecture}"
                )

    def validate_flash_attention(self, native_arches: Sequence[str]) -> None:
        if tuple(native_arches) != self.flash_attention_arches:
            raise VerificationError(
                f"qualification {self.name!r} requires FlashAttention images "
                f"{self.flash_attention_arches}, found {tuple(native_arches)}"
            )


@dataclass(frozen=True)
class NativeBuildLayout:
    toolkit: Path
    cc: Path
    cxx: Path
    readelf: Path

    @classmethod
    def discover(cls, prefix: Path, *, machine: str | None = None) -> NativeBuildLayout:
        machine = (machine or platform.machine()).lower()
        target_names = {
            "amd64": "x86_64-linux",
            "arm64": "sbsa-linux",
            "aarch64": "sbsa-linux",
            "x86_64": "x86_64-linux",
        }
        targets = tuple(
            path
            for path in sorted((prefix / "targets").glob("*-linux"))
            if (path / "include").is_dir() and (path / "lib").is_dir()
        )
        preferred_target = prefix / "targets" / target_names.get(machine, "")
        if preferred_target in targets:
            toolkit = preferred_target
        elif len(targets) == 1:
            toolkit = targets[0]
        else:
            raise VerificationError(
                f"cannot select the CUDA target for host {machine!r}; found {targets}"
            )

        compiler_prefixes = {
            "amd64": "x86_64-conda-linux-gnu",
            "arm64": "aarch64-conda-linux-gnu",
            "aarch64": "aarch64-conda-linux-gnu",
            "x86_64": "x86_64-conda-linux-gnu",
        }
        compiler_prefix = compiler_prefixes.get(machine)
        if compiler_prefix is None:
            raise VerificationError(f"unsupported Conda host architecture: {machine!r}")
        tools = tuple(
            prefix / "bin" / f"{compiler_prefix}-{name}" for name in ("cc", "c++", "readelf")
        )
        missing = tuple(path for path in tools if not path.is_file())
        if missing:
            raise VerificationError(f"native build tools are missing: {missing}")
        return cls(toolkit=toolkit, cc=tools[0], cxx=tools[1], readelf=tools[2])


@dataclass(frozen=True)
class FlashAttentionLock:
    version: str
    url: str
    sha256: str
    cuda_arches: tuple[str, ...]

    @property
    def requirement(self) -> str:
        return f"flash-attn @ {self.url}#sha256={self.sha256}"

    @staticmethod
    def _cuda_arches_for_version(version: str) -> tuple[str, ...]:
        if version != "2.8.3.post1":
            raise VerificationError(
                f"FlashAttention {version} has no reviewed CUDA architecture contract"
            )
        # flash-attn 2.8.3.post1 setup.py accepts exactly these CUDA targets.
        return "80", "90", "100", "120"

    @classmethod
    def load(cls, path: Path = FLASH_ATTENTION_SOURCE_PATH) -> FlashAttentionLock:
        entries = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(entries) != 1:
            raise VerificationError(f"expected one FlashAttention source in {path}")
        match = re.fullmatch(
            r"flash-attn\s+@\s+"
            r"(?P<url>https://files\.pythonhosted\.org/.+/"
            r"flash_attn-(?P<version>[A-Za-z0-9.+-]+)\.tar\.gz)"
            r"#sha256=(?P<sha256>[0-9a-f]{64})",
            entries[0],
        )
        if match is None:
            raise VerificationError(
                f"FlashAttention source must be an official hashed sdist: {entries[0]!r}"
            )
        return cls(
            version=match.group("version"),
            url=match.group("url"),
            sha256=match.group("sha256"),
            cuda_arches=cls._cuda_arches_for_version(match.group("version")),
        )

    def select_build_arches(
        self,
        devices: Sequence[CudaDevice],
        compiler_arches: Sequence[str],
    ) -> tuple[str, ...]:
        available_arches = tuple(
            architecture for architecture in self.cuda_arches if architecture in compiler_arches
        )
        try:
            return select_cuda_build_arches(devices, available_arches)
        except VerificationError as error:
            raise VerificationError(
                f"FlashAttention {self.version} cannot target the visible devices; "
                f"provider architectures are {self.cuda_arches} and compiler architectures "
                f"are {tuple(compiler_arches)}"
            ) from error

    def validate_direct_url(self, payload: Mapping[str, Any]) -> None:
        archive = payload.get("archive_info")
        hashes = archive.get("hashes") if isinstance(archive, Mapping) else None
        actual_hash = hashes.get("sha256") if isinstance(hashes, Mapping) else None
        if payload.get("url") != self.url or actual_hash != self.sha256:
            raise VerificationError("installed FlashAttention does not match its source lock")


@dataclass(frozen=True)
class FlashAttentionPrebuilt:
    """Hash-pinned wheels of the locked FlashAttention source, one per host machine."""

    cuda_arches: tuple[str, ...]
    # (machine, file, sha256), sorted by machine.
    wheels: tuple[tuple[str, str, str], ...]

    @classmethod
    def load(cls, path: Path = FLASH_ATTENTION_PREBUILT_PATH) -> FlashAttentionPrebuilt:
        lock = FlashAttentionLock.load()
        with path.open("rb") as file:
            payload = tomllib.load(file)
        cuda_arches = payload.get("cuda_arches")
        if (
            not isinstance(cuda_arches, list)
            or not cuda_arches
            or not all(isinstance(arch, str) and arch in lock.cuda_arches for arch in cuda_arches)
        ):
            raise VerificationError(
                f"prebuilt FlashAttention cuda_arches {cuda_arches!r} must be a subset of the "
                f"locked source's {lock.cuda_arches}"
            )
        wheels = payload.get("wheels")
        if not isinstance(wheels, Mapping) or not wheels:
            raise VerificationError(f"{path} lists no prebuilt FlashAttention wheels")
        entries = []
        for machine, wheel in sorted(wheels.items()):
            file = wheel.get("file") if isinstance(wheel, Mapping) else None
            sha256 = wheel.get("sha256") if isinstance(wheel, Mapping) else None
            # The profile's CPython 3.12 ABI on this Linux machine.
            match = re.fullmatch(
                rf"flash_attn-(?P<version>[^-]+)-cp312-cp312-linux_{re.escape(machine)}\.whl",
                file if isinstance(file, str) else "",
            )
            if match is None:
                raise VerificationError(
                    f"prebuilt FlashAttention wheel for {machine} must be a cp312 "
                    f"linux_{machine} wheel: {file!r}"
                )
            if match.group("version") != lock.version:
                raise VerificationError(
                    f"prebuilt FlashAttention wheel {file} is not the locked version {lock.version}"
                )
            if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
                raise VerificationError(
                    f"prebuilt FlashAttention wheel {file} has an invalid SHA256: {sha256!r}"
                )
            entries.append((machine, file, sha256))
        return cls(cuda_arches=tuple(cuda_arches), wheels=tuple(entries))

    def wheel_for(self, machine: str) -> tuple[str, str]:
        """The (file, sha256) pinned for ``machine``."""

        for name, file, sha256 in self.wheels:
            if name == machine:
                return file, sha256
        raise VerificationError(
            f"no prebuilt FlashAttention wheel for {machine!r}; supported machines are "
            f"{tuple(name for name, _, _ in self.wheels)}"
        )

    def covers(self, devices: Sequence[CudaDevice]) -> bool:
        return all(cuda_binary_covers(device.capability, self.cuda_arches) for device in devices)

    def requirement(self, path: Path, machine: str) -> str:
        """A hashed direct reference: pip checks the hash and records it in direct_url.json."""

        _, sha256 = self.wheel_for(machine)
        return f"flash-attn @ {path.as_uri()}#sha256={sha256}"

    def validate_wheel(self, path: Path, machine: str) -> None:
        file, sha256 = self.wheel_for(machine)
        if path.name != file or not path.is_file():
            raise VerificationError(
                f"prebuilt FlashAttention wheel for {machine} must be an existing {file}: {path}"
            )
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != sha256:
            raise VerificationError(f"{path} has SHA256 {actual}, expected {sha256}")

    def validate_direct_url(self, payload: Mapping[str, Any], machine: str) -> None:
        file, sha256 = self.wheel_for(machine)
        url = payload.get("url")
        archive = payload.get("archive_info")
        hashes = archive.get("hashes") if isinstance(archive, Mapping) else None
        actual_hash = hashes.get("sha256") if isinstance(hashes, Mapping) else None
        if not isinstance(url, str) or not url.endswith("/" + file) or actual_hash != sha256:
            raise VerificationError(
                f"installed FlashAttention does not match its prebuilt wheel lock for {machine}"
            )


@dataclass(frozen=True)
class TorchMemorySaverLock:
    repository: str
    commit: str

    @property
    def requirement(self) -> str:
        return f"torch-memory-saver @ git+{self.repository}@{self.commit}"

    @classmethod
    def load(cls, path: Path = PYPROJECT_PATH) -> TorchMemorySaverLock:
        requirements = [
            requirement
            for requirement in _project_requirements(path)
            if canonicalize_name(requirement.name) == "torch-memory-saver"
        ]
        if len(requirements) != 1:
            raise VerificationError("project metadata must contain torch-memory-saver exactly once")
        requirement = requirements[0]
        match = re.fullmatch(
            r"git\+"
            r"(?P<repository>https://github\.com/fzyzcjy/torch_memory_saver\.git)@"
            r"(?P<commit>[0-9a-f]{40})",
            requirement.url or "",
        )
        if match is None or requirement.extras or requirement.marker is not None:
            raise VerificationError(
                "torch-memory-saver source must be an official immutable Git commit: "
                f"{str(requirement)!r}"
            )
        return cls(
            repository=match.group("repository"),
            commit=match.group("commit"),
        )

    def validate_direct_url(self, payload: Mapping[str, Any]) -> None:
        vcs = payload.get("vcs_info")
        if not isinstance(vcs, Mapping):
            raise VerificationError("installed torch-memory-saver has no VCS provenance")
        if (
            payload.get("url") != self.repository
            or vcs.get("vcs") != "git"
            or vcs.get("requested_revision") != self.commit
            or vcs.get("commit_id") != self.commit
        ):
            raise VerificationError("installed torch-memory-saver does not match its source lock")


@dataclass(frozen=True)
class RuntimeProfile:
    environment: str
    python: tuple[int, int, int]
    pip_version: str
    cuda_toolkit: str
    cuda_tileiras: str
    libsanitizer: str
    sysroot: str
    kernel_headers: str
    cuda_runtime: str
    distributions: tuple[tuple[str, str], ...]
    flash_attention: FlashAttentionLock
    flash_attention_prebuilt: FlashAttentionPrebuilt
    torch_memory_saver: TorchMemorySaverLock

    def distribution_version(self, name: str) -> str:
        normalized = canonicalize_name(name)
        for distribution, version in self.distributions:
            if distribution == normalized:
                return version
        raise VerificationError(f"runtime profile has no exact pin for {normalized}")

    def validate_versions(self, installed: Mapping[str, str]) -> None:
        errors = [
            f"{name}: expected {expected}, found {installed.get(name, 'missing')}"
            for name, expected in self.distributions
            if not versions_equal(installed.get(name), expected)
        ]
        if errors:
            raise VerificationError("; ".join(errors))


def load_device_qualification(
    name: str | None,
    path: Path = QUALIFICATIONS_PATH,
) -> DeviceQualification | None:
    if name is None:
        return None
    with path.open("rb") as file:
        qualifications = tomllib.load(file)
    payload = qualifications.get(name)
    if not isinstance(payload, Mapping):
        raise VerificationError(f"unknown GPU qualification: {name!r}")
    capability = payload.get("capability")
    native_arches = payload.get("flash_attention_arches")
    if (
        not isinstance(payload.get("visible_device_count"), int)
        or not isinstance(payload.get("device_name_contains"), str)
        or not isinstance(capability, list)
        or len(capability) != 2
        or not all(isinstance(part, int) for part in capability)
        or not isinstance(native_arches, list)
        or not native_arches
        or not all(isinstance(arch, str) and arch.isdigit() for arch in native_arches)
    ):
        raise VerificationError(f"invalid GPU qualification {name!r} in {path}")
    return DeviceQualification(
        name=name,
        visible_device_count=payload["visible_device_count"],
        device_name_contains=payload["device_name_contains"],
        capability=(capability[0], capability[1]),
        flash_attention_arches=tuple(native_arches),
    )


def _architecture_parts(architecture: str) -> tuple[int, int]:
    if not architecture.isdigit() or len(architecture) < 2:
        raise VerificationError(f"invalid CUDA architecture: {architecture!r}")
    value = int(architecture)
    return divmod(value, 10)


def cuda_binary_covers(capability: tuple[int, int], native_arches: Sequence[str]) -> bool:
    device_major, device_minor = capability
    return any(
        binary_major == device_major and binary_minor <= device_minor
        for binary_major, binary_minor in map(_architecture_parts, native_arches)
    )


def select_cuda_build_arches(
    devices: Sequence[CudaDevice],
    compiler_arches: Sequence[str],
) -> tuple[str, ...]:
    """Select the best compiler target available for every visible device.

    This is a native-per-device policy, not a coverage-minimal wheel policy. A
    lower minor target is selected only when the compiler lacks the device's
    exact target, even though that binary remains compatible within the major.
    """
    selected: set[str] = set()
    parsed_compiler_arches = tuple(
        (architecture, _architecture_parts(architecture)) for architecture in compiler_arches
    )
    for device in devices:
        candidates = [
            architecture
            for architecture, capability in parsed_compiler_arches
            if capability[0] == device.capability[0] and capability[1] <= device.capability[1]
        ]
        if not candidates:
            raise VerificationError(
                f"no compatible CUDA target exists for device {device.name!r} at "
                f"SM{device.architecture}; "
                f"available architectures are {tuple(compiler_arches)}"
            )
        selected.add(max(candidates, key=lambda architecture: _architecture_parts(architecture)))
    return tuple(sorted(selected, key=_architecture_parts))


_REQUIRED_DISTRIBUTIONS = {
    "einops",
    "fla-core",
    "flash-attn",
    "flashinfer-python",
    "liger-kernel",
    "sglang-kernel",
    "torch",
    "torch-memory-saver",
    "triton",
}
_GPU_UUID = re.compile(
    r"^(?:GPU-)?([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)


def _profile_scalar(lines: Sequence[str], prefix: str, path: Path) -> str:
    matches = [
        line.strip().removeprefix(prefix) for line in lines if line.strip().startswith(prefix)
    ]
    if len(matches) != 1:
        raise VerificationError(f"expected one {prefix!r} entry in {path}")
    return matches[0]


def load_runtime_profile(path: Path = PROFILE_PATH) -> RuntimeProfile:
    lines = path.read_text(encoding="utf-8").splitlines()
    environment = _profile_scalar(lines, "name: ", path)
    python = tuple(int(part) for part in _profile_scalar(lines, "- python=", path).split("."))
    if len(python) != 3:
        raise VerificationError(f"expected an exact Python patch version in {path}")
    pip_version = _profile_scalar(lines, "- pip=", path)
    cuda_toolkit = _profile_scalar(lines, "- cuda-toolkit=", path)
    cuda_tileiras = _profile_scalar(lines, "- nvidia::cuda-tileiras=", path)
    libsanitizer = _profile_scalar(lines, "- libsanitizer=", path)
    sysroot = _profile_scalar(lines, "- sysroot_linux-64=", path)
    kernel_headers = _profile_scalar(lines, "- kernel-headers_linux-64=", path)

    pins: dict[str, str] = {}
    for line in lines:
        # The versioned profile uses a fixed six-space pip-list indentation.
        if not line.startswith("      - ") or line.startswith("      - --"):
            continue
        requirement = _parse_requirement(line.removeprefix("      - ").partition(" #")[0])
        name = canonicalize_name(requirement.name)
        if name in pins:
            raise VerificationError(f"duplicate exact pin for {name} in {path}")
        # Conda installs the original requirement, including extras. Verification
        # and pip constraints address distribution names without extras.
        pins[name] = _exact_version(requirement)

    flash_attention = FlashAttentionLock.load()
    torch_memory_saver = TorchMemorySaverLock.load()
    if "flash-attn" in pins:
        raise VerificationError(
            "flash-attn must be installed after Torch from environments/flash-attention-source.txt "
            "or its prebuilt wheel"
        )
    pins["flash-attn"] = flash_attention.version

    missing = sorted(_REQUIRED_DISTRIBUTIONS - pins.keys())
    if missing:
        raise VerificationError(f"profile is missing required exact pins: {missing}")
    cuda_tag = pins["torch"].partition("+cu")[2]
    if len(cuda_tag) != 3 or not cuda_tag.isdigit():
        raise VerificationError(f"cannot derive CUDA runtime from torch=={pins['torch']}")
    cuda_runtime = f"{int(cuda_tag[:2])}.{int(cuda_tag[2])}"
    return RuntimeProfile(
        environment=environment,
        python=(python[0], python[1], python[2]),
        pip_version=pip_version,
        cuda_toolkit=cuda_toolkit,
        cuda_tileiras=cuda_tileiras,
        libsanitizer=libsanitizer,
        sysroot=sysroot,
        kernel_headers=kernel_headers,
        cuda_runtime=cuda_runtime,
        distributions=tuple(sorted(pins.items())),
        flash_attention=flash_attention,
        flash_attention_prebuilt=FlashAttentionPrebuilt.load(),
        torch_memory_saver=torch_memory_saver,
    )


def _parse_requirement(requirement: str) -> Requirement:
    try:
        return Requirement(requirement)
    except InvalidRequirement as error:
        raise VerificationError(f"invalid project requirement: {requirement!r}") from error


def _exact_version(requirement: Requirement) -> str:
    pins = tuple(requirement.specifier)
    if (
        requirement.url is not None
        or requirement.marker is not None
        or len(pins) != 1
        or pins[0].operator != "=="
        or pins[0].version.endswith(".*")
    ):
        raise VerificationError(f"metadata must pin one exact {requirement.name} version")
    return str(Version(pins[0].version))


def versions_equal(actual: str | None, expected: str) -> bool:
    """Compare exact PEP 440 identities, including any local version label."""
    try:
        return actual is not None and Version(actual) == Version(expected)
    except InvalidVersion:
        return False


def _project_requirements(path: Path) -> tuple[Requirement, ...]:
    with path.open("rb") as file:
        payload = tomllib.load(file)

    project = payload.get("project")
    if not isinstance(project, Mapping):
        raise VerificationError(f"missing [project] table in {path}")
    dependencies = project.get("dependencies")
    optional = project.get("optional-dependencies")
    dev_dependencies = optional.get("dev") if isinstance(optional, Mapping) else None
    if not isinstance(dependencies, list) or not all(
        isinstance(requirement, str) for requirement in dependencies
    ):
        raise VerificationError(f"invalid project dependencies in {path}")
    if not isinstance(dev_dependencies, list) or not all(
        isinstance(requirement, str) for requirement in dev_dependencies
    ):
        raise VerificationError(f"invalid dev dependencies in {path}")

    return tuple(
        _parse_requirement(requirement) for requirement in (*dependencies, *dev_dependencies)
    )


def required_kernel_version(path: Path = PYPROJECT_PATH) -> str:
    requirements = [
        requirement
        for requirement in _project_requirements(path)
        if canonicalize_name(requirement.name) == "cti-kernels"
    ]
    if len(requirements) != 1:
        raise VerificationError("project metadata must contain cti-kernels exactly once")
    requirement = requirements[0]
    version = _exact_version(requirement)
    # The engine requires GDN, the CuTe DSL providers and varlen FlashAttention (the vision
    # tower, training RoPE). The varlen extra's flash-attn pin is the profile's locked version,
    # installed from source or its prebuilt wheel before the kernel.
    if requirement.extras != {"cute", "gdn", "varlen"}:
        raise VerificationError("cti-kernels must request exactly the [cute,gdn,varlen] extras")
    return version


@dataclass(frozen=True)
class KernelInstallTarget:
    """The engine's exact dependency, optionally supplied by a validated local wheel."""

    version: str
    path: Path | None = None
    sha256: str | None = None

    @classmethod
    def load(
        cls,
        *,
        repo_root: Path = REPO_ROOT,
        wheel: Path | None = None,
    ) -> KernelInstallTarget:
        expected_version = required_kernel_version(repo_root / "pyproject.toml")
        if wheel is None:
            return cls(expected_version)
        path = wheel.resolve()
        if path.suffix != ".whl" or not path.is_file():
            raise VerificationError(f"kernel wheel must be an existing .whl file: {path}")
        try:
            with ZipFile(path) as archive:
                records = [
                    name
                    for name in archive.namelist()
                    if name.endswith(".dist-info/METADATA") and len(name.split("/")) == 2
                ]
                if len(records) != 1:
                    raise VerificationError("kernel wheel must contain one METADATA record")
                metadata = BytesParser().parsebytes(archive.read(records[0]))
        except (BadZipFile, OSError) as error:
            raise VerificationError(f"cannot read kernel wheel: {path}") from error
        if len(metadata.get_all("Name", [])) != 1 or len(metadata.get_all("Version", [])) != 1:
            raise VerificationError("kernel wheel must contain one Name and Version")
        name, version = metadata["Name"], metadata["Version"]
        with path.open("rb") as file:
            sha256 = hashlib.file_digest(file, "sha256").hexdigest()
        if not isinstance(name, str) or canonicalize_name(name) != "cti-kernels":
            raise VerificationError(
                f"kernel target distribution must be cti-kernels, found {name!r}"
            )
        if not versions_equal(version, expected_version):
            raise VerificationError(
                f"kernel target version must be {expected_version}, found {version!r}"
            )
        return cls(expected_version, path, sha256)

    @property
    def requirement(self) -> str:
        return f"cti-kernels[cute,gdn,varlen]=={self.version}"

    @property
    def install_args(self) -> tuple[str, ...]:
        if self.path is not None:
            # A named direct reference preserves extras and safely quotes path
            # characters (including spaces and #) through the file URI.
            return (f"cti-kernels[cute,gdn,varlen] @ {self.path.as_uri()}",)
        return (self.requirement,)


def load_project_install_requirements(path: Path = PYPROJECT_PATH) -> tuple[str, ...]:
    requirements = _project_requirements(path)
    required_kernel_version(path)
    TorchMemorySaverLock.load(path)
    return tuple(
        str(requirement)
        for requirement in requirements
        if canonicalize_name(requirement.name) not in {"torch-memory-saver", "cti-kernels"}
    )


def tms_preload_path(prefix: Path | None = None) -> Path:
    if prefix is None:
        purelib = Path(sysconfig.get_paths()["purelib"])
    else:
        purelib = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
        purelib /= "site-packages"
    return purelib / "torch_memory_saver_hook_mode_preload_cu13.abi3.so"


def cuda_driver_library_dir() -> Path:
    driver = ctypes.CDLL("libcuda.so.1")
    mappings = Path("/proc/self/maps").read_text(encoding="utf-8")
    for line in mappings.splitlines():
        fields = line.split()
        if not fields:
            continue
        candidate = Path(fields[-1])
        unversioned = candidate.parent / "libcuda.so"
        if candidate.name.startswith("libcuda.so") and unversioned.is_file():
            return candidate.parent
    raise RuntimeError(
        f"loaded {driver!r}, but could not locate an unversioned libcuda.so for JIT linking"
    )


def recorded_source_revision(repo_root: Path = REPO_ROOT) -> str | None:
    """The commit a source tree without Git metadata, such as the container image's, records in
    its REVISION file; None for a Git checkout or a tree without one."""

    path = repo_root / "REVISION"
    if (repo_root / ".git").exists() or not path.is_file():
        return None
    revision = path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None:
        raise VerificationError(f"{path} must hold one full commit SHA, found {revision!r}")
    return revision


def normalize_cuda_device_uuid(uuid: object) -> str:
    value = str(uuid)
    if value.upper().startswith("MIG-"):
        raise VerificationError("MIG devices are not supported by the single-GPU profile")
    match = _GPU_UUID.fullmatch(value)
    if match is None:
        raise VerificationError(f"unexpected CUDA device UUID: {value!r}")
    return f"GPU-{match.group(1).lower()}"
