from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import sysconfig
import venv
from dataclasses import replace
from importlib.metadata import Distribution
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import pytest
import tomllib
from packaging.version import Version

from scripts import setup_env
from scripts.environment import bootstrap as bootstrap_module
from scripts.environment import contract, wheel_metadata
from scripts.environment import verify as verify_module
from tests.support.paths import REPO_ROOT

PROFILE_PATH = REPO_ROOT / "environments" / "benchmark-cu130-torch213.yml"
FLASH_ATTENTION_SOURCE_PATH = REPO_ROOT / "environments" / "flash-attention-source.txt"
QUALIFICATIONS_PATH = REPO_ROOT / "environments" / "gpu-qualifications.toml"
KERNEL_VERSION = contract.required_kernel_version()
KERNEL_REQUIREMENT = f"cti-kernels[cute,gdn,varlen]=={KERNEL_VERSION}"


def test_setup_environment_cli_remains_directly_executable() -> None:
    completed = subprocess.run(
        (sys.executable, str(REPO_ROOT / "scripts" / "setup_env.py"), "--help"),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "{create,verify}" in completed.stdout


def test_setup_environment_create_cli_defaults_to_the_project_pin(monkeypatch) -> None:
    calls = []

    class Bootstrap:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def create(self):
            calls.append("create")

    monkeypatch.setattr(setup_env, "EnvironmentBootstrap", Bootstrap)
    monkeypatch.setattr(sys, "argv", ["setup_env.py", "create", "--conda", "conda"])
    setup_env.main()

    assert calls[0][1]["kernel_wheel"] is None
    assert calls[0][1]["flash_attn_wheel"] is None
    assert calls[1] == "create"


def test_setup_environment_create_cli_forwards_a_prebuilt_flash_attention_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Bootstrap:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def create(self):
            pass

    monkeypatch.setattr(setup_env, "EnvironmentBootstrap", Bootstrap)
    monkeypatch.setattr(
        sys, "argv", ["setup_env.py", "create", "--conda", "conda", "--flash-attn-wheel", "fa.whl"]
    )
    setup_env.main()

    assert calls[0]["flash_attn_wheel"] == Path("fa.whl")


def test_setup_environment_create_cli_forwards_a_gpu_less_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Bootstrap:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def create(self):
            pass

    monkeypatch.setattr(setup_env, "EnvironmentBootstrap", Bootstrap)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "setup_env.py",
            "create",
            "--conda",
            "conda",
            "--no-gpu",
            "--cuda-driver-dir",
            "/usr/lib/x86_64-linux-gnu",
            "--cache-dir",
            "/root/.cache/cti",
        ],
    )
    setup_env.main()

    assert calls[0]["no_gpu"] is True
    assert calls[0]["cuda_driver_dir"] == Path("/usr/lib/x86_64-linux-gnu")
    assert calls[0]["cache_dir"] == Path("/root/.cache/cti")


def test_setup_environment_finds_conda_beside_the_invoking_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python = tmp_path / "bin/python"
    conda = python.with_name("conda")
    conda.parent.mkdir()
    conda.touch()
    monkeypatch.setattr(setup_env.shutil, "which", lambda _name: None)
    monkeypatch.setattr(setup_env.sys, "executable", str(python))

    assert setup_env._find_conda(None) == str(conda)


def _cuda_device(
    *,
    name: str = "NVIDIA GeForce RTX 5090",
    capability: tuple[int, int] = (12, 0),
) -> object:
    return contract.CudaDevice(
        logical_index=0,
        name=name,
        capability=capability,
        uuid="GPU-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
    )


def _create_native_layout(
    prefix: Path,
    *,
    target: str,
    compiler_prefix: str,
) -> object:
    toolkit = prefix / "targets" / target
    (toolkit / "include").mkdir(parents=True)
    (toolkit / "lib").mkdir()
    bin_dir = prefix / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("cc", "c++", "readelf"):
        (bin_dir / f"{compiler_prefix}-{name}").touch()
    return contract.NativeBuildLayout.discover(
        prefix,
        machine="aarch64" if target == "sbsa-linux" else "x86_64",
    )


def test_public_metadata_matches_the_torch_213_contract() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)
    dependencies = project["project"]["dependencies"]
    cpu_ci_dependencies = project["dependency-groups"]["cpu-ci"]

    assert project["project"]["name"] == "cti"
    assert not (REPO_ROOT / "python" / "edgecti").exists()
    assert not (REPO_ROOT / "python" / "minisgl").exists()
    assert "torch==2.13.0" in dependencies
    transformers_version = dict(contract.load_runtime_profile().distributions)["transformers"]
    for requirements in (dependencies, cpu_ci_dependencies):
        assert [item for item in requirements if item.startswith("transformers")] == [
            f"transformers=={transformers_version}"
        ]
    assert KERNEL_REQUIREMENT in dependencies
    assert not any("liger-kernel" in dependency.lower() for dependency in cpu_ci_dependencies)
    assert contract.TorchMemorySaverLock.load().requirement in dependencies
    assert not any(
        dependency.lower().startswith("torch-memory-saver==") for dependency in dependencies
    )
    assert not any(dependency.lower().startswith("flash-attn") for dependency in dependencies)
    assert not any(
        dependency.lower().startswith(("sgl_kernel", "sgl-kernel")) for dependency in dependencies
    )


def test_locked_profile_contains_the_exact_abi_bound_stack() -> None:
    runtime_profile = contract.load_runtime_profile()
    pins = dict(runtime_profile.distributions)
    profile = PROFILE_PATH.read_text(encoding="utf-8")

    assert runtime_profile.environment == "cti-cu130-torch213"
    assert runtime_profile.python == (3, 12, 13)
    assert runtime_profile.pip_version == "26.1.2"
    assert runtime_profile.cuda_toolkit == "13.1.0"
    assert runtime_profile.cuda_tileiras == "13.1.80=*_1"
    assert runtime_profile.cuda_runtime == "13.0"
    assert runtime_profile.libsanitizer == "14.4.0"
    assert (runtime_profile.sysroot, runtime_profile.kernel_headers) == ("2.34", "5.14.0")
    assert pins["torch"] == "2.13.0+cu130"
    assert pins["triton"] == "3.7.1"
    assert pins["nvidia-cusparselt-cu13"] == "0.8.1"
    # The float-workspace bound in cti.attention.flashinfer
    # (_split_kv_scratch_bytes) follows this planner version; recheck it on a bump.
    assert pins["flashinfer-python"] == "0.7.0.post1"
    for name, version in (
        ("cuda-python", "13.3.1"),
        ("cuda-core", "1.0.1"),
        ("cuda-pathfinder", "1.8.3"),
        ("nccl4py", "0.6.0"),
        ("nccl-extensions", "0.1.0"),
        ("nvidia-cudnn-frontend", "1.31.0"),
        ("cuda-tile", "1.6.0"),
    ):
        assert pins[name] == version, name
    assert pins["sglang-kernel"] == "0.4.6.post1"
    assert pins["liger-kernel"] == "0.8.0"
    assert pins["torch-memory-saver"] == "0.0.9.post1"
    assert pins["flash-attn"] == "2.8.3.post1"
    assert pins["fla-core"] == "0.5.2"
    assert pins["einops"] == "0.8.2"
    assert pins["ninja"] == "1.13.0"
    assert pins["packaging"] == "26.2"
    assert pins["psutil"] == "7.2.2"
    assert pins["setuptools"] == "81.0.0"
    assert pins["wheel"] == "0.47.0"
    assert "- pip=26.1.2" in profile
    assert pins["quack-kernels"] == "0.6.5"
    assert pins["nvidia-cutlass-dsl"] == "4.8.0"
    assert "nvidia-cutlass-dsl[cu13]==4.8.0" in profile
    for library in ("base", "core", "cu12", "cu13"):
        assert pins[f"nvidia-cutlass-dsl-libs-{library}"] == "4.8.0"
    assert "cuda-compat" not in profile
    assert "git+" not in profile
    assert "torch-memory-saver==0.0.9.post1" in profile


def test_rtx5090_qualification_is_data_driven_and_exact() -> None:
    qualification = contract.load_device_qualification("rtx5090")

    assert qualification is not None
    assert qualification.name == "rtx5090"
    assert qualification.capability == (12, 0)
    assert qualification.flash_attention_arches == ("120",)
    qualification.validate_devices((_cuda_device(),))
    qualification.validate_flash_attention(("120",))

    with pytest.raises(contract.VerificationError, match="requires SM120"):
        qualification.validate_devices((_cuda_device(capability=(9, 0)),))
    with pytest.raises(contract.VerificationError, match="FlashAttention images"):
        qualification.validate_flash_attention(("90",))


def test_rtx_pro_6000_pair_qualification_is_data_driven_and_exact() -> None:
    qualification = contract.load_device_qualification("rtx-pro-6000-pair")
    pro = "NVIDIA RTX PRO 6000 Blackwell Server Edition"

    assert qualification is not None
    assert qualification.name == "rtx-pro-6000-pair"
    assert qualification.capability == (12, 0)
    assert qualification.flash_attention_arches == ("120",)
    qualification.validate_devices(
        (
            contract.CudaDevice(0, pro, (12, 0), "GPU-first"),
            contract.CudaDevice(1, pro, (12, 0), "GPU-second"),
        )
    )
    qualification.validate_flash_attention(("120",))

    with pytest.raises(contract.VerificationError, match="requires 2 visible GPU"):
        qualification.validate_devices((contract.CudaDevice(0, pro, (12, 0), "GPU-only"),))
    with pytest.raises(contract.VerificationError, match="requires a device containing"):
        qualification.validate_devices(
            (
                contract.CudaDevice(0, pro, (12, 0), "GPU-first"),
                contract.CudaDevice(1, "NVIDIA GeForce RTX 5090", (12, 0), "GPU-other"),
            )
        )


def test_dgx_spark_qualification_is_data_driven_and_exact() -> None:
    qualification = contract.load_device_qualification("dgx-spark")

    assert qualification is not None
    assert qualification.name == "dgx-spark"
    assert qualification.capability == (12, 1)
    assert qualification.flash_attention_arches == ("120",)
    qualification.validate_devices(
        (contract.CudaDevice(0, "NVIDIA GB10", (12, 1), "GPU-dgx-spark"),)
    )
    qualification.validate_flash_attention(("120",))

    with pytest.raises(contract.VerificationError, match="requires SM121"):
        qualification.validate_devices(
            (contract.CudaDevice(0, "NVIDIA GB10", (12, 0), "GPU-wrong-arch"),)
        )
    with pytest.raises(contract.VerificationError, match="requires a device containing"):
        qualification.validate_devices(
            (contract.CudaDevice(0, "NVIDIA GeForce RTX 5090", (12, 1), "GPU-other"),)
        )


def test_unknown_gpu_qualification_fails_closed() -> None:
    with pytest.raises(contract.VerificationError, match="unknown GPU qualification"):
        contract.load_device_qualification("missing", QUALIFICATIONS_PATH)


@pytest.mark.parametrize(
    ("machine", "target", "compiler_prefix"),
    (
        ("x86_64", "x86_64-linux", "x86_64-conda-linux-gnu"),
        ("aarch64", "sbsa-linux", "aarch64-conda-linux-gnu"),
    ),
)
def test_native_build_layout_discovers_host_specific_conda_tools(
    tmp_path: Path,
    machine: str,
    target: str,
    compiler_prefix: str,
) -> None:
    layout = _create_native_layout(
        tmp_path,
        target=target,
        compiler_prefix=compiler_prefix,
    )

    assert layout == contract.NativeBuildLayout.discover(tmp_path, machine=machine)
    assert layout.toolkit.name == target
    assert layout.cc.name == f"{compiler_prefix}-cc"
    assert layout.cxx.name == f"{compiler_prefix}-c++"
    assert layout.readelf.name == f"{compiler_prefix}-readelf"


@pytest.mark.parametrize(
    ("target", "compiler_prefix"),
    (("x86_64-linux", "x86_64-conda-linux-gnu"), ("sbsa-linux", "aarch64-conda-linux-gnu")),
)
def test_the_environment_activates_the_host_cuda_target_headers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str, compiler_prefix: str
) -> None:
    layout = _create_native_layout(tmp_path / "env", target=target, compiler_prefix=compiler_prefix)
    bootstrap, commands, _, _ = _bootstrap_with_recording_runner(
        monkeypatch, kernel_wheel=_kernel_wheel(tmp_path)
    )
    monkeypatch.setattr(
        contract.NativeBuildLayout, "discover", classmethod(lambda _cls, _prefix: layout)
    )

    bootstrap.create()

    configure = next(command for command in commands if command[1:4] == ("env", "config", "vars"))
    assert f"CPATH={tmp_path / 'env' / 'targets' / target / 'include'}" in configure


def test_cuda_architecture_selection_uses_same_major_binary_compatibility() -> None:
    devices = (
        _cuda_device(name="NVIDIA H100", capability=(9, 0)),
        contract.CudaDevice(1, "NVIDIA GB10", (12, 1), _cuda_device().uuid),
    )

    assert contract.select_cuda_build_arches(devices, ("80", "90", "120")) == (
        "90",
        "120",
    )
    assert contract.cuda_binary_covers((12, 1), ("120",))
    assert not contract.cuda_binary_covers((10, 0), ("90", "120"))
    with pytest.raises(contract.VerificationError, match="no compatible CUDA target"):
        contract.select_cuda_build_arches(
            (_cuda_device(name="NVIDIA B200", capability=(10, 0)),),
            ("90", "120"),
        )


def test_cuda_architecture_selection_prefers_each_native_device_target() -> None:
    devices = (
        _cuda_device(capability=(12, 0)),
        contract.CudaDevice(1, "NVIDIA GB10", (12, 1), "GPU-secondary"),
    )

    assert contract.select_cuda_build_arches(devices, ("120", "121")) == (
        "120",
        "121",
    )
    assert contract.select_cuda_build_arches(devices, ("120",)) == ("120",)


@pytest.mark.parametrize(
    ("name", "capability", "expected"),
    (
        ("NVIDIA A100", (8, 0), ("80",)),
        ("NVIDIA H100", (9, 0), ("90",)),
        ("NVIDIA B200", (10, 0), ("100",)),
        ("NVIDIA GB10", (12, 1), ("120",)),
    ),
)
def test_flash_attention_uses_provider_supported_cuda_targets(
    name: str,
    capability: tuple[int, int],
    expected: tuple[str, ...],
) -> None:
    lock = contract.FlashAttentionLock.load()
    device = _cuda_device(name=name, capability=capability)

    assert (
        lock.select_build_arches(
            (device,),
            ("80", "90", "100", "120", "121"),
        )
        == expected
    )


def test_flash_attention_fails_closed_without_a_provider_compiler_intersection() -> None:
    lock = contract.FlashAttentionLock.load()

    with pytest.raises(contract.VerificationError, match="provider architectures"):
        lock.select_build_arches(
            (_cuda_device(name="NVIDIA B200", capability=(10, 0)),),
            ("80", "90", "120", "121"),
        )


def test_flash_attention_rejects_an_unreviewed_provider_version(tmp_path: Path) -> None:
    path = tmp_path / "flash-attention-source.txt"
    path.write_text(
        "flash-attn @ https://files.pythonhosted.org/a/flash_attn-2.8.4.tar.gz#sha256="
        + "a" * 64
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(contract.VerificationError, match="no reviewed CUDA architecture"):
        contract.FlashAttentionLock.load(path)


def test_flash_attention_source_lock_is_an_exact_official_sdist() -> None:
    lock = contract.FlashAttentionLock.load()

    assert lock.version == "2.8.3.post1"
    assert lock.url.endswith("/flash_attn-2.8.3.post1.tar.gz")
    assert lock.sha256 == "55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348"
    assert lock.cuda_arches == ("80", "90", "100", "120")
    assert lock.requirement == next(
        line.strip()
        for line in FLASH_ATTENTION_SOURCE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


@pytest.mark.parametrize(
    "entry",
    (
        "flash-attn==2.8.3.post1",
        "flash-attn @ https://example.com/flash_attn-2.8.3.post1.tar.gz#sha256=" + "a" * 64,
        "flash-attn @ https://files.pythonhosted.org/a/flash_attn-2.8.3.post1.tar.gz",
    ),
)
def test_flash_attention_source_lock_rejects_mutable_or_unofficial_inputs(
    tmp_path: Path,
    entry: str,
) -> None:
    path = tmp_path / "flash-attention-source.txt"
    path.write_text(entry + "\n", encoding="utf-8")

    with pytest.raises(contract.VerificationError, match="official hashed sdist"):
        contract.FlashAttentionLock.load(path)


def test_flash_attention_source_lock_validates_installed_provenance() -> None:
    lock = contract.FlashAttentionLock.load()
    valid = {
        "archive_info": {"hashes": {"sha256": lock.sha256}},
        "url": lock.url,
    }
    lock.validate_direct_url(valid)

    valid["archive_info"]["hashes"]["sha256"] = "0" * 64
    with pytest.raises(contract.VerificationError, match="source lock"):
        lock.validate_direct_url(valid)


def _prebuilt_wheel_name(machine: str = "x86_64") -> str:
    return (
        f"flash_attn-{contract.FlashAttentionLock.load().version}-cp312-cp312-linux_{machine}.whl"
    )


def _prebuilt_flash_attention(
    tmp_path: Path, machine: str = "x86_64"
) -> tuple[Path, contract.FlashAttentionPrebuilt]:
    """A wheel file and a lock pinning its SHA256, for this machine only."""
    wheel = tmp_path / _prebuilt_wheel_name(machine)
    wheel.write_bytes(b"prebuilt flash-attn")
    sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return wheel, contract.FlashAttentionPrebuilt(("120",), ((machine, wheel.name, sha256),))


def _write_prebuilt_lock(
    tmp_path: Path,
    *,
    cuda_arches: tuple[str, ...] = ("120",),
    file: str | None = None,
    sha256: str = "a" * 64,
) -> Path:
    path = tmp_path / "flash-attention-prebuilt.toml"
    path.write_text(
        f"cuda_arches = {json.dumps(list(cuda_arches))}\n"
        f"[wheels.x86_64]\nfile = {json.dumps(file or _prebuilt_wheel_name())}\n"
        f"sha256 = {json.dumps(sha256)}\n",
        encoding="utf-8",
    )
    return path


def test_flash_attention_prebuilt_lock_pins_one_wheel_of_the_locked_source_per_machine(
    tmp_path: Path,
) -> None:
    prebuilt = contract.FlashAttentionPrebuilt.load()

    assert prebuilt == contract.load_runtime_profile().flash_attention_prebuilt
    assert prebuilt.cuda_arches == ("120",)
    assert [(machine, file) for machine, file, _ in prebuilt.wheels] == [
        (machine, _prebuilt_wheel_name(machine)) for machine in ("aarch64", "x86_64")
    ]
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha256) for _, _, sha256 in prebuilt.wheels)
    loaded = contract.FlashAttentionPrebuilt.load(_write_prebuilt_lock(tmp_path))
    assert loaded == contract.FlashAttentionPrebuilt(
        ("120",), (("x86_64", _prebuilt_wheel_name(), "a" * 64),)
    )


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"cuda_arches": ()}, "subset"),
        ({"cuda_arches": ("121",)}, "subset"),
        ({"file": "flash_attn-2.8.3.post1-cp311-cp311-linux_x86_64.whl"}, "cp312 linux_x86_64"),
        ({"file": _prebuilt_wheel_name("aarch64")}, "cp312 linux_x86_64"),
        ({"file": "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"}, "locked version"),
        ({"sha256": "A" * 64}, "invalid SHA256"),
        ({"sha256": "a" * 63}, "invalid SHA256"),
    ],
)
def test_flash_attention_prebuilt_lock_rejects_invalid_entries(
    tmp_path: Path, entry: dict, message: str
) -> None:
    with pytest.raises(contract.VerificationError, match=message):
        contract.FlashAttentionPrebuilt.load(_write_prebuilt_lock(tmp_path, **entry))


def test_flash_attention_prebuilt_lock_names_the_supported_machines() -> None:
    prebuilt = contract.FlashAttentionPrebuilt.load()

    assert prebuilt.wheel_for("aarch64")[0] == _prebuilt_wheel_name("aarch64")
    with pytest.raises(
        contract.VerificationError, match=r"supported machines are \('aarch64', 'x86_64'\)"
    ):
        prebuilt.wheel_for("ppc64le")


@pytest.mark.parametrize(
    ("capabilities", "covered"),
    [([(12, 0)], True), ([(12, 1)], True), ([(12, 0), (12, 1)], True), ([(12, 0), (9, 0)], False)],
    ids=["sm120", "sm121", "sm120-sm121", "sm120-sm90"],
)
def test_flash_attention_prebuilt_covers_only_the_sm120_family(
    capabilities: list[tuple[int, int]], covered: bool
) -> None:
    devices = [
        contract.CudaDevice(index, "GPU", capability, f"GPU-{index}")
        for index, capability in enumerate(capabilities)
    ]
    assert contract.FlashAttentionPrebuilt.load().covers(devices) is covered


def test_flash_attention_prebuilt_validates_the_wheel_name_and_sha256(tmp_path: Path) -> None:
    wheel, prebuilt = _prebuilt_flash_attention(tmp_path)
    prebuilt.validate_wheel(wheel, "x86_64")
    _, sha256 = prebuilt.wheel_for("x86_64")
    assert prebuilt.requirement(wheel, "x86_64") == f"flash-attn @ {wheel.as_uri()}#sha256={sha256}"

    renamed = tmp_path / "renamed" / "flash_attn.whl"
    renamed.parent.mkdir()
    renamed.write_bytes(wheel.read_bytes())
    for path in (renamed, tmp_path / "missing" / wheel.name):
        with pytest.raises(contract.VerificationError, match="must be an existing"):
            prebuilt.validate_wheel(path, "x86_64")
    with pytest.raises(contract.VerificationError, match="supported machines"):
        prebuilt.validate_wheel(wheel, "aarch64")
    wheel.write_bytes(b"tampered")
    with pytest.raises(contract.VerificationError, match=f"expected {sha256}"):
        prebuilt.validate_wheel(wheel, "x86_64")


def test_flash_attention_prebuilt_validates_installed_provenance(tmp_path: Path) -> None:
    wheel, prebuilt = _prebuilt_flash_attention(tmp_path)
    _, sha256 = prebuilt.wheel_for("x86_64")
    valid = {"url": wheel.as_uri(), "archive_info": {"hashes": {"sha256": sha256}}}
    prebuilt.validate_direct_url(valid, "x86_64")

    lock = contract.FlashAttentionLock.load()
    for payload in (
        {**valid, "archive_info": {"hashes": {"sha256": "0" * 64}}},
        {**valid, "archive_info": {}},
        {**valid, "url": wheel.with_name(_prebuilt_wheel_name("aarch64")).as_uri()},
        {"url": lock.url, "archive_info": {"hashes": {"sha256": lock.sha256}}},
    ):
        with pytest.raises(contract.VerificationError, match="prebuilt wheel lock"):
            prebuilt.validate_direct_url(payload, "x86_64")


def test_install_script_pins_the_prebuilt_flash_attention_lock() -> None:
    script = (REPO_ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    pins = dict(re.findall(r'^(PINNED_FLASH_ATTN_\w+)="([^"]*)"$', script, re.MULTILINE))
    prebuilt = contract.FlashAttentionPrebuilt.load()

    assert pins == {
        "PINNED_FLASH_ATTN_VERSION": contract.FlashAttentionLock.load().version,
        "PINNED_FLASH_ATTN_SHA256_X86_64": prebuilt.wheel_for("x86_64")[1],
        "PINNED_FLASH_ATTN_SHA256_AARCH64": prebuilt.wheel_for("aarch64")[1],
    }
    # The script names each machine's wheel as the lock does.
    assert 'name="flash_attn-$PINNED_FLASH_ATTN_VERSION-cp312-cp312-linux_$machine.whl"' in script


def test_torch_memory_saver_source_lock_is_an_exact_official_commit() -> None:
    lock = contract.TorchMemorySaverLock.load()

    assert lock.repository == "https://github.com/fzyzcjy/torch_memory_saver.git"
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        dependencies = tomllib.load(file)["project"]["dependencies"]
    assert lock.requirement in dependencies
    assert len(lock.commit) == 40
    assert not (REPO_ROOT / "environments" / "torch-memory-saver-source.txt").exists()


def test_project_install_requirements_exclude_the_controlled_native_build() -> None:
    requirements = contract.load_project_install_requirements()

    assert "torch==2.13.0" in requirements
    assert "pytest>=6.0" in requirements
    assert not any("torch-memory-saver" in requirement.lower() for requirement in requirements)
    assert not any("cti-kernels" in requirement.lower() for requirement in requirements)


def _project_file(tmp_path: Path, dependencies: list[str], dev: tuple[str, ...] = ()) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f"[project]\ndependencies = {json.dumps(dependencies)}\n"
        f"[project.optional-dependencies]\ndev = {json.dumps(dev)}\n"
    )
    return path


@pytest.mark.parametrize(
    "requirement",
    [
        "cti-kernels[cute,gdn,varlen]>=7.2",
        "cti-kernels[cute,gdn,varlen]==7.*",
        "cti-kernels[cute,gdn,varlen]===7.2",
        "cti-kernels[cute,gdn,varlen]==7.2,!=7.1",
        "cti-kernels[cute,gdn,varlen]==7.2; python_version > '3'",
        "cti-kernels[cute,gdn,varlen] @ https://example.com/kernel.whl",
    ],
)
def test_kernel_dependency_requires_one_exact_version(tmp_path: Path, requirement: str) -> None:
    path = _project_file(tmp_path, [requirement])
    with pytest.raises(contract.VerificationError, match="one exact"):
        contract.required_kernel_version(path)


@pytest.mark.parametrize(
    "requirement", ["cti-kernels[cute,gdn,varlen]==banana", "cti-kernels=7.2", "cti-kernels!="]
)
def test_kernel_dependency_rejects_invalid_packaging_requirements(tmp_path, requirement):
    with pytest.raises(contract.VerificationError, match="invalid project requirement"):
        contract.required_kernel_version(_project_file(tmp_path, [requirement]))


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("CTI._Kernels[cute,gdn,varlen] == v7.2RC1", "7.2rc1"),
        ("cti-kernels[cute,gdn,varlen]==5!7.2+BUILD.4", "5!7.2+build.4"),
    ],
)
def test_kernel_dependency_uses_standard_name_and_version_normalization(
    tmp_path, requirement, expected
):
    assert contract.required_kernel_version(_project_file(tmp_path, [requirement])) == expected


def test_kernel_dependency_rejects_missing_requirement(tmp_path):
    with pytest.raises(contract.VerificationError, match="exactly once"):
        contract.required_kernel_version(_project_file(tmp_path, []))


def test_kernel_dependency_rejects_duplicate_normalized_names(tmp_path: Path) -> None:
    path = _project_file(tmp_path, ["cti-kernels[cute,gdn,varlen]==7.2"], dev=("cti_kernels==7.2",))
    with pytest.raises(contract.VerificationError, match="exactly once"):
        contract.load_project_install_requirements(path)


def _kernel_wheel(
    tmp_path: Path, *, name: str = "cti-kernels", version: str = KERNEL_VERSION
) -> Path:
    """Minimal archive for metadata validation, not installed-wheel evidence."""
    wheel = tmp_path / f"cti_kernels-{version}-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"cti_kernels-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n",
        )
    return wheel


def test_kernel_target_defaults_to_the_metadata_pin() -> None:
    target = contract.KernelInstallTarget.load()
    assert target.install_args == (KERNEL_REQUIREMENT,)
    assert target.path is None and target.sha256 is None


def test_kernel_target_records_the_explicit_wheel_sha256(tmp_path: Path) -> None:
    wheel = _kernel_wheel(tmp_path)
    target = contract.KernelInstallTarget.load(wheel=wheel)
    assert target.path == wheel.resolve()
    assert target.version == KERNEL_VERSION
    assert target.sha256 == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert target.install_args == (f"cti-kernels[cute,gdn,varlen] @ {wheel.resolve().as_uri()}",)


def test_kernel_wheel_accepts_pep440_equivalent_metadata(tmp_path: Path) -> None:
    _project_file(tmp_path, ["CTI._Kernels[cute,gdn,varlen]==7.2rc1"])
    wheel = _kernel_wheel(tmp_path, name="cti_kernels", version="7.2.0RC1")
    target = contract.KernelInstallTarget.load(repo_root=tmp_path, wheel=wheel)
    assert target.version == "7.2rc1"
    assert target.path == wheel


def test_kernel_wheel_supports_an_engine_only_checkout(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text((REPO_ROOT / "pyproject.toml").read_text())
    wheel = _kernel_wheel(tmp_path)
    target = contract.KernelInstallTarget.load(repo_root=tmp_path, wheel=wheel)
    assert target.path == wheel.resolve()
    assert target.version == KERNEL_VERSION
    assert target.sha256 == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert target.install_args == (f"cti-kernels[cute,gdn,varlen] @ {wheel.resolve().as_uri()}",)
    assert contract.load_project_install_requirements(tmp_path / "pyproject.toml")
    assert contract.KernelInstallTarget.load(repo_root=tmp_path).install_args == (
        KERNEL_REQUIREMENT,
    )


@pytest.mark.parametrize(
    ("name", "version", "message"),
    [
        ("other", KERNEL_VERSION, "distribution must"),
        ("cti-kernels", f"{KERNEL_VERSION}+unexpected", "version must"),
        ("cti-kernels", "banana", "version must"),
    ],
)
def test_kernel_wheel_rejects_wrong_metadata(tmp_path, name, version, message) -> None:
    wheel = _kernel_wheel(tmp_path, name=name, version=version)
    with pytest.raises(contract.VerificationError, match=message):
        contract.KernelInstallTarget.load(wheel=wheel)


def test_kernel_wheel_rejects_ambiguous_metadata(tmp_path: Path) -> None:
    wheel = _kernel_wheel(tmp_path)
    with ZipFile(wheel, "a") as archive:
        archive.writestr(
            "other.dist-info/METADATA", f"Name: cti-kernels\nVersion: {KERNEL_VERSION}\n"
        )
    with pytest.raises(contract.VerificationError, match="one METADATA"):
        contract.KernelInstallTarget.load(wheel=wheel)


@pytest.mark.parametrize("kind", ["missing", "not-zip", "not-wheel"])
def test_kernel_wheel_rejects_invalid_artifacts(tmp_path: Path, kind: str) -> None:
    path = tmp_path / ("artifact.tar.gz" if kind == "not-wheel" else "artifact.whl")
    if kind != "missing":
        path.write_text("not a wheel archive")
    with pytest.raises(contract.VerificationError, match="kernel wheel"):
        contract.KernelInstallTarget.load(wheel=path)


def test_project_install_requirements_reject_a_second_normalized_tms_name(
    tmp_path: Path,
) -> None:
    path = _project_file(
        tmp_path,
        ["cti-kernels[cute,gdn,varlen]==7.2", contract.TorchMemorySaverLock.load().requirement],
        dev=("torch_memory_saver>=0.0.9",),
    )

    with pytest.raises(contract.VerificationError, match="exactly once"):
        contract.load_project_install_requirements(path)


@pytest.mark.parametrize(
    "entry",
    (
        "torch-memory-saver==0.0.9.post1",
        "torch-memory-saver @ git+https://example.com/torch_memory_saver.git@" + "a" * 40,
        "torch-memory-saver @ git+https://github.com/fzyzcjy/torch_memory_saver.git@main",
        "torch-memory-saver[extra] @ git+https://github.com/fzyzcjy/torch_memory_saver.git@"
        + "a" * 40,
        "torch-memory-saver @ git+https://github.com/fzyzcjy/torch_memory_saver.git@"
        + "a" * 40
        + " ; python_version > '3'",
    ),
)
def test_torch_memory_saver_source_lock_rejects_mutable_or_unofficial_inputs(
    tmp_path: Path,
    entry: str,
) -> None:
    path = _project_file(tmp_path, [entry])

    with pytest.raises(contract.VerificationError, match="official immutable Git commit"):
        contract.TorchMemorySaverLock.load(path)


def test_torch_memory_saver_source_authority_is_only_project_metadata(tmp_path) -> None:
    requirement = (
        "torch-memory-saver @ git+https://github.com/fzyzcjy/torch_memory_saver.git@" + "a" * 40
    )
    path = _project_file(tmp_path, ["cti-kernels[cute,gdn,varlen]==7.2", requirement])
    lock = contract.TorchMemorySaverLock.load(path)
    assert lock.commit == "a" * 40
    assert lock.requirement == requirement
    assert contract.load_project_install_requirements(path) == ()


def test_torch_memory_saver_rejects_missing_source_requirement(tmp_path) -> None:
    with pytest.raises(contract.VerificationError, match="exactly once"):
        contract.TorchMemorySaverLock.load(_project_file(tmp_path, []))


def test_torch_memory_saver_source_lock_validates_installed_provenance() -> None:
    lock = contract.TorchMemorySaverLock.load()
    valid = {
        "url": lock.repository,
        "vcs_info": {
            "commit_id": lock.commit,
            "requested_revision": lock.commit,
            "vcs": "git",
        },
    }
    lock.validate_direct_url(valid)

    valid["vcs_info"]["commit_id"] = "0" * 40
    with pytest.raises(contract.VerificationError, match="source lock"):
        lock.validate_direct_url(valid)


@pytest.mark.parametrize(
    ("dynamic_entry", "rpath", "accepted"),
    (
        ("0x000000000000000f (RPATH) Library rpath", "{prefix}/lib", True),
        # The prebuilt wheels' layout: prefix/lib from site-packages.
        ("0x000000000000001d (RUNPATH) Library runpath", "$ORIGIN/../..", True),
        ("0x000000000000000f (RPATH) Library rpath", "/tmp/foreign-prefix/lib", False),
        (
            "0x000000000000000f (RPATH) Library rpath",
            "$ORIGIN/../../../../foreign-prefix/lib",
            False,
        ),
    ),
)
def test_flash_attention_binary_rejects_foreign_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dynamic_entry: str,
    rpath: str,
    accepted: bool,
) -> None:
    monkeypatch.setattr(contract.platform, "machine", lambda: "x86_64")
    prefix = tmp_path / "formal-prefix"
    extension = prefix / "lib/python3.12/site-packages/flash_attn_2_cuda.so"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"native-extension")
    bin_dir = prefix / "bin"
    bin_dir.mkdir()
    (bin_dir / "cuobjdump").touch()
    _create_native_layout(
        prefix,
        target="x86_64-linux",
        compiler_prefix="x86_64-conda-linux-gnu",
    )
    verifier = verify_module.EnvironmentVerifier()
    monkeypatch.setattr(verifier, "_prefix", prefix)

    def run(command, *, env=None):
        del env
        if "--list-elf" in command:
            return "\n".join(("kernel.1.sm_120.cubin", "kernel.2.sm_120.cubin"))
        return f"{dynamic_entry}: [{rpath.format(prefix=prefix)}]"

    monkeypatch.setattr(verifier, "_run", run)
    if not accepted:
        with pytest.raises(contract.VerificationError, match="foreign prefix"):
            verifier._verify_flash_attention_binary(extension, (_cuda_device(),))
        return

    receipt = verifier._verify_flash_attention_binary(extension, (_cuda_device(),))
    assert receipt["native_arches"] == ["120"]
    assert receipt["native_image_count"] == 2
    assert receipt["rpaths"] == [rpath.format(prefix=prefix)]


def test_flash_attention_binary_rejects_an_uncovered_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(contract.platform, "machine", lambda: "x86_64")
    prefix = tmp_path / "formal-prefix"
    extension = prefix / "lib/python3.12/site-packages/flash_attn_2_cuda.so"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"native-extension")
    bin_dir = prefix / "bin"
    bin_dir.mkdir()
    (bin_dir / "cuobjdump").touch()
    _create_native_layout(
        prefix,
        target="x86_64-linux",
        compiler_prefix="x86_64-conda-linux-gnu",
    )
    verifier = verify_module.EnvironmentVerifier()
    monkeypatch.setattr(verifier, "_prefix", prefix)
    monkeypatch.setattr(
        verifier,
        "_run",
        lambda _command, env=None: "kernel.sm_90.cubin",
    )

    with pytest.raises(contract.VerificationError, match="do not cover"):
        verifier._verify_flash_attention_binary(extension, (_cuda_device(),))


def test_runtime_profile_requires_an_exact_python_patch(tmp_path: Path) -> None:
    profile = PROFILE_PATH.read_text(encoding="utf-8").replace(
        "python=3.12.13",
        "python=3.12",
    )
    path = tmp_path / "partial-python.yml"
    path.write_text(profile, encoding="utf-8")

    with pytest.raises(contract.VerificationError, match="exact Python patch"):
        contract.load_runtime_profile(path)


def test_runtime_profile_reads_its_libsanitizer_pin(tmp_path: Path) -> None:
    profile = PROFILE_PATH.read_text(encoding="utf-8")
    path = tmp_path / "libsanitizer.yml"
    path.write_text(
        profile.replace("- libsanitizer=14.4.0", "- libsanitizer=15.3.0"), encoding="utf-8"
    )
    assert contract.load_runtime_profile(path).libsanitizer == "15.3.0"

    path.write_text(profile.replace("- libsanitizer=14.4.0", ""), encoding="utf-8")
    with pytest.raises(contract.VerificationError) as caught:
        contract.load_runtime_profile(path)
    assert str(caught.value) == f"expected one '- libsanitizer=' entry in {path}"


@pytest.mark.parametrize(
    "distribution",
    ("sglang-kernel", "quack-kernels", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-cu13"),
)
def test_runtime_profile_rejects_a_mismatched_abi_artifact(distribution: str) -> None:
    runtime_profile = contract.load_runtime_profile()
    versions = dict(runtime_profile.distributions)
    runtime_profile.validate_versions(versions)

    versions[distribution] = "0.0.0"
    with pytest.raises(contract.VerificationError, match=distribution):
        runtime_profile.validate_versions(versions)


@pytest.mark.parametrize(
    "requirement",
    (
        "nvidia-cutlass-dsl[cu13]>=4.8.0",
        "nvidia-cutlass-dsl[cu13]==4.8.*",
        'nvidia-cutlass-dsl[cu13]==4.8.0; python_version >= "3.12"',
    ),
)
def test_runtime_profile_extras_still_require_an_unconditional_exact_pin(
    tmp_path: Path, requirement: str
) -> None:
    path = tmp_path / "profile.yml"
    path.write_text(
        PROFILE_PATH.read_text(encoding="utf-8").replace(
            "nvidia-cutlass-dsl[cu13]==4.8.0", requirement
        ),
        encoding="utf-8",
    )

    with pytest.raises(contract.VerificationError, match="one exact nvidia-cutlass-dsl"):
        contract.load_runtime_profile(path)


def test_runtime_profile_requires_the_cute_cuda13_libraries() -> None:
    runtime_profile = contract.load_runtime_profile()
    versions = dict(runtime_profile.distributions)
    del versions["nvidia-cutlass-dsl-libs-cu13"]

    with pytest.raises(contract.VerificationError, match="nvidia-cutlass-dsl-libs-cu13"):
        runtime_profile.validate_versions(versions)


@pytest.mark.parametrize("origin", ("source", "prebuilt"))
@pytest.mark.parametrize("available", (True, False), ids=("available", "unavailable"))
def test_flash_attention_verifier_delegates_availability_to_the_public_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    available: bool,
    origin: str,
) -> None:
    monkeypatch.setattr(verify_module.platform, "machine", lambda: "x86_64")
    verifier = verify_module.EnvironmentVerifier()
    monkeypatch.setattr(verifier, "_prefix", tmp_path)
    lock = verifier._profile.flash_attention
    if origin == "source":
        url, sha256 = lock.url, lock.sha256
        provenance = {"provenance": "source", "source_sha256": sha256, "source_url": url}
    else:
        file, sha256 = verifier._profile.flash_attention_prebuilt.wheel_for("x86_64")
        url = f"file:///wheels/{file}"
        provenance = {"provenance": "prebuilt", "wheel": file, "wheel_sha256": sha256}
    site_packages = tmp_path / "lib/python3.12/site-packages"
    metadata_path = site_packages / f"flash_attn-{lock.version}.dist-info"
    metadata_path.mkdir(parents=True)
    (metadata_path / "direct_url.json").write_text(
        json.dumps({"url": url, "archive_info": {"hashes": {"sha256": sha256}}})
    )

    def distribution(name: str) -> Distribution:
        assert name == "flash-attn"
        return Distribution.at(metadata_path)

    monkeypatch.setattr(verify_module, "distribution", distribution)
    package = ModuleType("flash_attn")
    package.__version__ = lock.version
    package.__file__ = str(site_packages / "flash_attn/__init__.py")
    extension = ModuleType("flash_attn_2_cuda")
    extension.__file__ = str(site_packages / "flash_attn_2_cuda.so")
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, extension.__name__, extension)

    calls = []
    unavailable = RuntimeError("CausalVarlenAttention requires a compatible flash-attn build")

    class PublicAttention:
        def __init__(self) -> None:
            calls.append("construct")
            if not available:
                raise unavailable

    operation_module = ModuleType("ctikernels.attention.varlen")
    operation_module.CausalVarlenAttention = PublicAttention
    monkeypatch.setitem(sys.modules, operation_module.__name__, operation_module)
    devices = (_cuda_device(),)

    def verify_binary(path, observed_devices):
        assert path == Path(extension.__file__)
        assert observed_devices is devices
        calls.append("binary")
        return {"native_arches": ["120"]}

    monkeypatch.setattr(verifier, "_verify_flash_attention_binary", verify_binary)
    if not available:
        with pytest.raises(contract.VerificationError, match=str(unavailable)) as caught:
            verifier._verify_flash_attention(devices)
        assert caught.value.__cause__ is unavailable
        assert calls == ["construct"]
        return

    receipt = verifier._verify_flash_attention(devices)

    assert calls == ["construct", "binary"]
    assert receipt == {
        "native_arches": ["120"],
        "metadata": str(site_packages),
        "package": package.__file__,
        **provenance,
        "public_operation": "ctikernels.attention.varlen.CausalVarlenAttention",
    }


@pytest.mark.parametrize(
    ("machine", "origin", "provenance"),
    [
        ("x86_64", "source", "source"),
        ("ppc64le", "source", "source"),
        ("x86_64", "x86_64", "prebuilt"),
        ("aarch64", "aarch64", "prebuilt"),
        ("aarch64", "x86_64", None),
        ("x86_64", "foreign", None),
    ],
)
def test_flash_attention_provenance_is_the_source_lock_or_this_machines_prebuilt_wheel(
    monkeypatch: pytest.MonkeyPatch, machine: str, origin: str, provenance: str | None
) -> None:
    monkeypatch.setattr(verify_module.platform, "machine", lambda: machine)
    verifier = verify_module.EnvironmentVerifier()
    lock = verifier._profile.flash_attention
    if origin == "source":
        url, sha256 = lock.url, lock.sha256
    elif origin == "foreign":
        url, sha256 = f"https://example.com/flash_attn-{lock.version}.tar.gz", lock.sha256
    else:
        file, sha256 = verifier._profile.flash_attention_prebuilt.wheel_for(origin)
        url = f"file:///wheels/{file}"
    direct_url = {"url": url, "archive_info": {"hashes": {"sha256": sha256}}}

    if provenance is None:
        with pytest.raises(contract.VerificationError, match="neither the locked source"):
            verifier._flash_attention_provenance(direct_url)
    elif provenance == "source":
        assert verifier._flash_attention_provenance(direct_url) == {
            "provenance": "source",
            "source_sha256": lock.sha256,
            "source_url": lock.url,
        }
    else:
        assert verifier._flash_attention_provenance(direct_url) == {
            "provenance": "prebuilt",
            "wheel": file,
            "wheel_sha256": sha256,
        }


@pytest.mark.parametrize("installed", ("missing", "2.8.3"))
def test_runtime_profile_rejects_missing_or_mismatched_flash_attention(
    installed: str,
) -> None:
    runtime_profile = contract.load_runtime_profile()
    versions = dict(runtime_profile.distributions)
    versions["flash-attn"] = installed

    with pytest.raises(contract.VerificationError, match="flash-attn"):
        runtime_profile.validate_versions(versions)


def test_tms_preload_path_is_explicitly_cuda_13() -> None:
    path = contract.tms_preload_path(Path("/opt/cti"))

    assert path.name == "torch_memory_saver_hook_mode_preload_cu13.abi3.so"


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", "GPU-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"),
        ("GPU-AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE", "GPU-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"),
    ),
)
def test_cuda_device_uuid_is_normalized_for_nvml_lookup(raw: str, expected: str) -> None:
    assert contract.normalize_cuda_device_uuid(raw) == expected


@pytest.mark.parametrize("raw", ("MIG-existing", "GPU-existing", ""))
def test_cuda_device_uuid_rejects_unsupported_formats(raw: str) -> None:
    with pytest.raises(contract.VerificationError):
        contract.normalize_cuda_device_uuid(raw)


def test_verifier_rejects_bypassing_conda_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_HOME", raising=False)
    verifier = verify_module.EnvironmentVerifier()
    nvcc = Path(sys.prefix) / "bin" / "nvcc"

    with pytest.raises(contract.VerificationError, match="CUDA_HOME"):
        verifier._verify_jit_environment(nvcc)


@pytest.mark.parametrize(
    ("output", "accepted"),
    (
        ("Cuda compilation tools, release 13.1, V13.1.80", True),
        ("Cuda compilation tools, release 13.0, V13.0.88", False),
    ),
    ids=("13.1", "13.0"),
)
def test_verifier_requires_the_profile_cuda_release_from_nvcc(
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    accepted: bool,
) -> None:
    verifier = verify_module.EnvironmentVerifier()
    commands = []

    def run(command, *, env=None):
        del env
        commands.append(command)
        return output

    monkeypatch.setattr(verifier, "_run", run)
    nvcc = Path("/env/bin/nvcc")
    if not accepted:
        with pytest.raises(contract.VerificationError, match="expected nvcc release 13.1"):
            verifier._verify_nvcc(nvcc)
        return

    assert verifier._verify_nvcc(nvcc) == output
    assert commands == [("/env/bin/nvcc", "--version")]


@pytest.mark.parametrize(
    ("markers", "message"),
    (
        ({}, "nvcc is missing"),
        ({"libsanitizer": None}, "libsanitizer=14.4.0 marker"),
        # The profile holds GCC at 14.4.0 through libsanitizer; GCC 15 brings 15.3.0.
        ({"libsanitizer": "15.3.0-h0_0"}, "libsanitizer=14.4.0 marker"),
        # The label's build 0 segfaults on every run.
        ({"cuda-tileiras": "13.1.80-h0_0"}, r"cuda-tileiras=13.1.80=\*_1 marker"),
        # GCC 14's default sysroot; its glibc 2.38 symbols do not load on glibc 2.35 hosts.
        ({"sysroot_linux-64": "2.39-h0_0"}, "sysroot_linux-64=2.34 marker"),
    ),
    ids=(
        "matching",
        "no-libsanitizer",
        "other-libsanitizer",
        "crashing-tileiras",
        "glibc-2.39-sysroot",
    ),
)
def test_verifier_requires_the_profile_conda_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    markers: dict[str, str | None],
    message: str,
) -> None:
    python = ".".join(map(str, sys.version_info[:3]))
    path = tmp_path / "current-python.yml"
    path.write_text(
        PROFILE_PATH.read_text(encoding="utf-8").replace("python=3.12.13", f"python={python}"),
        encoding="utf-8",
    )
    profile = contract.load_runtime_profile(path)
    installed = {"pip": profile.pip_version, **dict(profile.distributions)}

    def version(name: str) -> str:
        if name not in installed:
            raise verify_module.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(verify_module, "version", version)
    verifier = verify_module.EnvironmentVerifier(profile=profile)
    verifier._prefix = tmp_path / "env"
    conda_meta = verifier._prefix / "conda-meta"
    conda_meta.mkdir(parents=True)
    names = {
        "pip": f"{profile.pip_version}-h0_0",
        "cuda-toolkit": f"{profile.cuda_toolkit}-h0_0",
        "cuda-tileiras": "13.1.80-h0_1",
        "libsanitizer": f"{profile.libsanitizer}-h0_0",
        "sysroot_linux-64": f"{profile.sysroot}-h0_0",
        "kernel-headers_linux-64": f"{profile.kernel_headers}-h0_0",
        **markers,
    }
    for name, marker in names.items():
        if marker is not None:
            (conda_meta / f"{name}-{marker}.json").touch()

    # With every Conda marker in place, verification goes on to the nvcc this prefix lacks.
    with pytest.raises(contract.VerificationError, match=message):
        verifier.verify()


def test_verifier_requires_checkout_as_the_only_cti_source(tmp_path: Path) -> None:
    verifier = verify_module.EnvironmentVerifier(repo_root=REPO_ROOT)
    source_package = REPO_ROOT / "python" / "cti"
    package = ModuleType("cti")
    package.__file__ = str(source_package / "__init__.py")
    package.__path__ = [str(source_package)]

    assert verifier._verify_cti_source(package) == (source_package.resolve(),)

    package.__file__ = str(tmp_path / "stale-checkout" / "cti" / "__init__.py")
    with pytest.raises(contract.VerificationError, match="cti resolves to"):
        verifier._verify_cti_source(package)

    package.__file__ = str(source_package / "__init__.py")
    package.__path__.append(str(tmp_path / "stale-checkout" / "cti"))
    with pytest.raises(contract.VerificationError, match="expected only"):
        verifier._verify_cti_source(package)


@pytest.mark.parametrize("installed", [None, "0.0.1"], ids=["on-pythonpath", "installed"])
def test_verifier_takes_the_checkout_on_pythonpath_or_installed_at_its_version(
    installed: str | None,
) -> None:
    verifier = verify_module.EnvironmentVerifier(repo_root=REPO_ROOT)

    verifier._verify_engine_distribution(installed, "0.0.1")
    with pytest.raises(contract.VerificationError, match="is not this checkout's version"):
        verifier._verify_engine_distribution("0.0.0", "0.0.1")


def _installed_kernel_metadata(monkeypatch, tmp_path, direct_url, *, version=KERNEL_VERSION):
    """Install metadata fixtures under one prefix to exercise the real metadata reader."""
    prefix = tmp_path / "environment"
    purelib = prefix / "lib/site-packages"
    package_root = purelib / "ctikernels"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("")
    metadata = purelib / f"cti_kernels-{version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: cti-kernels\nVersion: {version}\n")
    (metadata / "RECORD").write_text("ctikernels/__init__.py,,\n")
    if direct_url is not None:
        (metadata / "direct_url.json").write_text(json.dumps(direct_url))
    monkeypatch.setattr(verify_module.sysconfig, "get_paths", lambda: {"purelib": str(purelib)})
    verifier = verify_module.EnvironmentVerifier()
    verifier._prefix = prefix
    package = ModuleType("ctikernels")
    package.__file__ = str(package_root / "__init__.py")
    package.__path__ = [str(package_root)]
    return verifier, package, metadata


def _wheel_provenance() -> dict[str, object]:
    return {
        "url": f"file:///removed-local-artifacts/cti_kernels-{KERNEL_VERSION}-py3-none-any.whl",
        "archive_info": {"hashes": {"sha256": "a" * 64}},
    }


def test_kernel_verifier_rejects_editable_external_checkout(monkeypatch, tmp_path) -> None:
    source = (tmp_path / "cti-kernel").resolve()
    verifier, package, _ = _installed_kernel_metadata(
        monkeypatch, tmp_path, {"url": source.as_uri(), "dir_info": {"editable": True}}
    )
    package.__file__ = str(source / "python/ctikernels/__init__.py")
    package.__path__ = [str(source / "python/ctikernels")]
    with pytest.raises(contract.VerificationError, match="noneditable installed package"):
        verifier._verify_kernel_source(package)


def test_kernel_verifier_accepts_prefix_wheel_without_original_archive(
    monkeypatch, tmp_path
) -> None:
    verifier, package, _ = _installed_kernel_metadata(monkeypatch, tmp_path, _wheel_provenance())
    receipt = verifier._verify_kernel_source(package)
    assert receipt["version"] == KERNEL_VERSION
    assert receipt["mode"] == "wheel"
    assert receipt["package"] == package.__path__
    assert receipt["wheel_sha256"] == "a" * 64
    package.__file__ = str(tmp_path / "shadowed/ctikernels/__init__.py")
    with pytest.raises(contract.VerificationError, match="ctikernels resolves to"):
        verifier._verify_kernel_source(package)


def test_kernel_verifier_checks_supplied_wheel_hash(monkeypatch, tmp_path) -> None:
    verifier, package, metadata = _installed_kernel_metadata(
        monkeypatch, tmp_path, _wheel_provenance()
    )
    wheel = _kernel_wheel(tmp_path)
    verifier._kernel_wheel = wheel
    with pytest.raises(contract.VerificationError, match="wheel hash mismatch"):
        verifier._verify_kernel_source(package)
    direct_url = {
        "url": wheel.as_uri(),
        "archive_info": {"hashes": {"sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}},
    }
    (metadata / "direct_url.json").write_text(json.dumps(direct_url))
    assert verifier._verify_kernel_source(package)["mode"] == "wheel"


@pytest.mark.parametrize(
    "direct_url",
    [
        None,
        {"url": "https://example.com/kernel.whl", "archive_info": {"hashes": {"sha256": "a" * 64}}},
        {"url": "https://example.com/kernel.whl", "archive_info": {}},
        {
            "url": "https://example.com/kernel.tar.gz",
            "archive_info": {"hashes": {"sha256": "a" * 64}},
        },
        {"url": "file:///kernel-source", "dir_info": {"editable": False}},
        {
            "url": "https://example.com/kernel.git",
            "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
        },
    ],
)
def test_kernel_verifier_accepts_installed_wheel_identity_independent_of_origin(
    monkeypatch, tmp_path, direct_url
) -> None:
    verifier, package, _ = _installed_kernel_metadata(monkeypatch, tmp_path, direct_url)
    receipt = verifier._verify_kernel_source(package)
    assert receipt["version"] == KERNEL_VERSION
    assert receipt["direct_url"] == direct_url
    assert receipt["package"] == package.__path__
    if direct_url is None:
        assert receipt["source_url"] is None
        assert receipt["wheel_sha256"] is None
    elif direct_url["url"].endswith(".tar.gz"):
        assert receipt["wheel_sha256"] is None


@pytest.mark.parametrize(
    "direct_url", [None, {"url": "https://example.com/kernel.whl", "archive_info": {}}]
)
def test_kernel_verifier_explicit_wheel_requires_artifact_sha256(monkeypatch, tmp_path, direct_url):
    verifier, package, _ = _installed_kernel_metadata(monkeypatch, tmp_path, direct_url)
    verifier._kernel_wheel = _kernel_wheel(tmp_path)
    with pytest.raises(contract.VerificationError, match="missing SHA256 provenance"):
        verifier._verify_kernel_source(package)


@pytest.mark.parametrize(
    ("direct_url", "message"),
    [
        ([], "invalid cti-kernels direct URL"),
        ({}, "no source URL"),
        ({"url": "", "archive_info": {}}, "no source URL"),
        (
            {"url": "file:///kernel.whl", "archive_info": {"hashes": {"sha256": "bad"}}},
            "invalid SHA256 provenance",
        ),
    ],
)
def test_kernel_verifier_rejects_invalid_optional_provenance(
    monkeypatch, tmp_path, direct_url, message
) -> None:
    verifier, package, _ = _installed_kernel_metadata(monkeypatch, tmp_path, direct_url)
    with pytest.raises(contract.VerificationError, match=message):
        verifier._verify_kernel_source(package)


@pytest.mark.parametrize("version", [f"{KERNEL_VERSION}+unexpected", "banana", ""])
def test_kernel_verifier_rejects_wrong_or_invalid_version(monkeypatch, tmp_path, version):
    verifier, package, _ = _installed_kernel_metadata(monkeypatch, tmp_path, None, version=version)
    with pytest.raises(contract.VerificationError, match="installed cti-kernels must be"):
        verifier._verify_kernel_source(package)


def test_kernel_verifier_accepts_pep440_equivalent_version(monkeypatch, tmp_path):
    verifier, package, _ = _installed_kernel_metadata(
        monkeypatch, tmp_path, None, version="7.2.0RC1"
    )
    _project_file(tmp_path, ["cti-kernels[cute,gdn,varlen]==7.2rc1"])
    verifier._repo_root = tmp_path
    assert verifier._verify_kernel_source(package)["version"] == "7.2rc1"


def test_kernel_verifier_rejects_invalid_provenance_json(monkeypatch, tmp_path):
    verifier, package, metadata = _installed_kernel_metadata(monkeypatch, tmp_path, None)
    (metadata / "direct_url.json").write_text("{")
    with pytest.raises(contract.VerificationError, match="direct_url.json is invalid"):
        verifier._verify_kernel_source(package)


@pytest.mark.parametrize("count", [0, 2])
def test_kernel_verifier_requires_one_installed_distribution(monkeypatch, tmp_path, count):
    verifier, package, metadata = _installed_kernel_metadata(monkeypatch, tmp_path, None)
    if count == 0:
        (metadata / "METADATA").unlink()
    else:
        duplicate = metadata.parent / "cti_kernels-duplicate.dist-info"
        duplicate.mkdir()
        (duplicate / "METADATA").write_text(f"Name: CTI._Kernels\nVersion: {KERNEL_VERSION}\n")
    with pytest.raises(contract.VerificationError, match="one installed cti-kernels"):
        verifier._verify_kernel_source(package)


def test_kernel_verifier_rejects_missing_package_ownership(monkeypatch, tmp_path) -> None:
    verifier, package, metadata = _installed_kernel_metadata(
        monkeypatch, tmp_path, _wheel_provenance()
    )
    (metadata / "RECORD").write_text("other/__init__.py,,\n")
    with pytest.raises(contract.VerificationError, match="RECORD does not own"):
        verifier._verify_kernel_source(package)


def test_verifier_requires_tms_from_the_formal_prefix(tmp_path: Path) -> None:
    verifier = verify_module.EnvironmentVerifier(repo_root=REPO_ROOT)
    expected_package = Path(sysconfig.get_paths()["purelib"]).resolve() / "torch_memory_saver"
    package = ModuleType("torch_memory_saver")
    package.__file__ = str(expected_package / "__init__.py")
    package.__path__ = [str(expected_package)]

    assert verifier._verify_tms_package(package) == expected_package / "__init__.py"

    package.__file__ = str(tmp_path / "foreign" / "torch_memory_saver" / "__init__.py")
    with pytest.raises(contract.VerificationError, match="torch-memory-saver resolves to"):
        verifier._verify_tms_package(package)

    package.__file__ = str(expected_package / "__init__.py")
    package.__path__.append(str(tmp_path / "foreign" / "torch_memory_saver"))
    with pytest.raises(contract.VerificationError, match="expected only"):
        verifier._verify_tms_package(package)


def test_tms_runtime_probe_cannot_be_disabled_by_python_optimization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verifier = verify_module.EnvironmentVerifier(repo_root=REPO_ROOT)
    observed: dict[str, object] = {}
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")

    def run(command, *, env=None):
        observed["command"] = command
        observed["env"] = env
        return ""

    monkeypatch.setattr(verifier, "_run", run)
    verifier._verify_tms_preload(tmp_path / "tms-preload.so")

    command = observed["command"]
    assert isinstance(command, tuple)
    probe = command[-1]
    assert "assert " not in probe
    assert "changed the allocation pointer" in probe
    assert "did not restore the allocation contents" in probe
    env = observed["env"]
    assert isinstance(env, dict)
    assert "PYTHONOPTIMIZE" not in env


def test_bootstrap_refuses_to_replace_an_existing_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bootstrap = bootstrap_module.EnvironmentBootstrap("conda", kernel_wheel=_kernel_wheel(tmp_path))
    existing = Path("/opt/conda/envs") / contract.load_runtime_profile().environment
    monkeypatch.setattr(bootstrap, "_existing_prefixes", lambda: (existing,))

    with pytest.raises(RuntimeError, match="already exists"):
        bootstrap.create()


def test_bootstrap_rejects_wrong_kernel_before_running_environment_commands(monkeypatch, tmp_path):
    wheel = _kernel_wheel(tmp_path, version=f"{KERNEL_VERSION}+unexpected")
    bootstrap = bootstrap_module.EnvironmentBootstrap("conda", kernel_wheel=wheel)
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("environment command before validation"),
    )
    with pytest.raises(contract.VerificationError, match="version must be"):
        bootstrap.create()


def test_bootstrap_validates_project_pin_before_running_environment_commands(monkeypatch, tmp_path):
    _project_file(tmp_path, ["cti-kernels[cute,gdn,varlen]>=7.2"])
    bootstrap = bootstrap_module.EnvironmentBootstrap("conda")
    monkeypatch.setattr(bootstrap_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("environment command before validation"),
    )
    with pytest.raises(contract.VerificationError, match="one exact"):
        bootstrap.create()


def test_bootstrap_accepts_a_caller_owned_environment_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = "research-engine"
    bootstrap = bootstrap_module.EnvironmentBootstrap(
        "conda", environment=environment, kernel_wheel=_kernel_wheel(tmp_path)
    )
    commands: list[tuple[str, ...]] = []
    prefix = Path("/opt/conda/envs") / environment

    monkeypatch.setattr(bootstrap, "_existing_prefixes", lambda: ())

    def run(command, *, capture_output=False, env=None):
        command = tuple(command)
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(bootstrap, "_run", run)
    monkeypatch.setattr(bootstrap, "_configure", lambda name: prefix)
    monkeypatch.setattr(bootstrap, "_verify", lambda name, installed_prefix: None)

    bootstrap.create()

    check, create = commands[:2]
    assert check == ("git", "rev-parse", "HEAD")
    assert create[:3] == ("conda", "env", "create")
    assert create[create.index("--name") + 1] == environment


@pytest.mark.parametrize("environment", ("", " leading", "trailing ", "path/name"))
def test_bootstrap_rejects_invalid_environment_names(environment: str) -> None:
    with pytest.raises(ValueError, match="environment name"):
        bootstrap_module.EnvironmentBootstrap("conda", environment=environment)


def test_bootstrap_rejects_a_non_string_environment_name() -> None:
    with pytest.raises(TypeError, match="environment name"):
        bootstrap_module.EnvironmentBootstrap("conda", environment=7)


def _record_digest(payload: bytes) -> tuple[str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"sha256={digest.decode()}", str(len(payload))


def _write_cusparselt_metadata(
    prefix: Path,
    tag: str,
    *,
    python_version: str = "3.12",
    distribution_version: str = "0.8.1",
) -> tuple[Path, Path]:
    dist_info = (
        prefix
        / f"lib/python{python_version}/site-packages"
        / f"nvidia_cusparselt_cu13-{distribution_version}.dist-info"
    )
    dist_info.mkdir(parents=True)
    wheel = dist_info / "WHEEL"
    wheel.write_text(f"Wheel-Version: 1.0\nTag: {tag}\n", encoding="utf-8")
    wheel_path = wheel.relative_to(dist_info.parent).as_posix()
    wheel_hash, wheel_size = _record_digest(wheel.read_bytes())
    record = dist_info / "RECORD"
    with record.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow((wheel_path, wheel_hash, wheel_size))
        writer.writerow((record.relative_to(dist_info.parent).as_posix(), "", ""))
    return wheel, record


def test_bootstrap_verification_environment_selects_the_formal_nvcc_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = Path("/opt/conda/envs/cti-cu130-torch213")
    monkeypatch.setenv("PATH", "/usr/local/cuda/bin:/opt/bin:/usr/bin")

    environment = bootstrap_module.EnvironmentBootstrap._verification_environment(prefix)

    assert environment["CUDA_HOME"] == str(prefix)
    assert environment["PATH"].split(":") == [
        str(prefix / "bin"),
        "/usr/local/cuda/bin",
        "/opt/bin",
        "/usr/bin",
    ]
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_bootstrap_reads_one_structured_target_python_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = bootstrap_module.EnvironmentBootstrap("conda")
    prefix = Path("/opt/conda/envs/cti-cu130-torch213")
    purelib = prefix / "lib/python3.12/site-packages"

    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda command, capture_output=False: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "unrelated conda notice\n"
                f'CTI_PYTHON_LAYOUT={{"prefix": "{prefix}", '
                f'"purelib": "{purelib}"}}\n'
            ),
        ),
    )

    layout = bootstrap._python_layout(prefix.name)

    assert layout.prefix == prefix
    assert layout.purelib == purelib


def test_bootstrap_rejects_target_python_purelib_outside_its_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = bootstrap_module.EnvironmentBootstrap("conda")
    prefix = Path("/opt/conda/envs/cti-cu130-torch213")
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda command, capture_output=False: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                f'CTI_PYTHON_LAYOUT={{"prefix": "{prefix}", '
                '"purelib": "/tmp/foreign/site-packages"}\n'
            ),
        ),
    )

    with pytest.raises(contract.VerificationError, match="purelib is outside"):
        bootstrap._python_layout(prefix.name)


def test_bootstrap_repairs_cusparselt_arm64_tag_and_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel, record = _write_cusparselt_metadata(
        tmp_path,
        "py3-none-manylinux2014_sbsa",
    )
    monkeypatch.setattr(wheel_metadata.platform, "machine", lambda: "aarch64")

    wheel_metadata.repair_cusparselt_arm64_tag(
        tmp_path,
        tmp_path / "lib/python3.12/site-packages",
        "0.8.1",
    )

    assert wheel.read_text(encoding="utf-8").endswith("Tag: py3-none-manylinux2014_aarch64\n")
    with record.open(encoding="utf-8", newline="") as file:
        wheel_row = next(csv.reader(file))
    assert wheel_row == [
        wheel.relative_to(record.parent.parent).as_posix(),
        *_record_digest(wheel.read_bytes()),
    ]


def test_bootstrap_repairs_only_the_target_python_purelib(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decoy_wheel, _ = _write_cusparselt_metadata(
        tmp_path,
        "py3-none-manylinux2014_sbsa",
        python_version="3.1",
    )
    target_wheel, _ = _write_cusparselt_metadata(
        tmp_path,
        "py3-none-manylinux2014_sbsa",
    )
    target_purelib = tmp_path / "lib/python3.12/site-packages"
    monkeypatch.setattr(wheel_metadata.platform, "machine", lambda: "aarch64")

    wheel_metadata.repair_cusparselt_arm64_tag(tmp_path, target_purelib, "0.8.1")

    assert target_wheel.read_text(encoding="utf-8").endswith(
        "Tag: py3-none-manylinux2014_aarch64\n"
    )
    assert decoy_wheel.read_text(encoding="utf-8").endswith("Tag: py3-none-manylinux2014_sbsa\n")


def test_bootstrap_cusparselt_repair_is_an_x86_noop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(wheel_metadata.platform, "machine", lambda: "x86_64")

    wheel_metadata.repair_cusparselt_arm64_tag(
        tmp_path,
        tmp_path / "lib/python3.12/site-packages",
        "0.8.1",
    )


@pytest.mark.parametrize(
    "tag",
    (
        "py3-none-manylinux2014_aarch64",
        "py3-none-manylinux_2_17_sbsa",
    ),
)
def test_bootstrap_rejects_unknown_cusparselt_arm64_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tag: str,
) -> None:
    _write_cusparselt_metadata(tmp_path, tag)
    monkeypatch.setattr(wheel_metadata.platform, "machine", lambda: "arm64")

    with pytest.raises(contract.VerificationError, match="unexpected cuSPARSELt"):
        wheel_metadata.repair_cusparselt_arm64_tag(
            tmp_path,
            tmp_path / "lib/python3.12/site-packages",
            "0.8.1",
        )


def _bootstrap_with_recording_runner(
    monkeypatch,
    *,
    kernel_wheel: Path | None = None,
    flash_attn_wheel: Path | None = None,
    qualification=None,
    capability: tuple[int, int] = (12, 0),
    fail_install: bool = False,
    fail_editable: bool = False,
    fail_kernel: bool = False,
    fail_verify: bool = False,
    fail_create: bool = False,
    fail_config: bool = False,
    interrupt_install: bool = False,
    fail_cleanup: bool = False,
    **options,
):
    bootstrap = bootstrap_module.EnvironmentBootstrap(
        "conda",
        qualification,
        kernel_wheel=kernel_wheel,
        flash_attn_wheel=flash_attn_wheel,
        **options,
    )
    commands: list[tuple[str, ...]] = []
    command_envs: list[dict[str, str] | None] = []
    install_error = subprocess.CalledProcessError(1, ("pip", "install"))
    cleanup_error = OSError("cleanup failed")

    monkeypatch.setattr(bootstrap, "_existing_prefixes", lambda: ())
    monkeypatch.setattr(
        bootstrap_module,
        "cuda_driver_library_dir",
        lambda: Path("/usr/lib/cuda-driver"),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "repair_cusparselt_arm64_tag",
        lambda _prefix, _purelib, _version: None,
    )
    prefix = Path("/opt/conda/envs/cti-cu130-torch213")
    layout = contract.NativeBuildLayout(
        toolkit=prefix / "targets/x86_64-linux",
        cc=prefix / "bin/x86_64-conda-linux-gnu-cc",
        cxx=prefix / "bin/x86_64-conda-linux-gnu-c++",
        readelf=prefix / "bin/x86_64-conda-linux-gnu-readelf",
    )
    monkeypatch.setattr(
        contract.NativeBuildLayout,
        "discover",
        classmethod(lambda _cls, _prefix: layout),
    )

    def run(command, *, capture_output=False, env=None):
        command = tuple(command)
        commands.append(command)
        command_envs.append(None if env is None else dict(env))
        if fail_create and command[1:3] == ("env", "create"):
            raise install_error
        if fail_config and command[1:4] == ("env", "config", "vars"):
            raise install_error
        if any("sysconfig.get_path" in part for part in command):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    'CTI_PYTHON_LAYOUT={"prefix": '
                    '"/opt/conda/envs/cti-cu130-torch213", '
                    '"purelib": "/opt/conda/envs/cti-cu130-torch213/'
                    'lib/python3.12/site-packages"}\n'
                ),
            )
        if any("torch.cuda.get_device_capability" in part for part in command):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    f'[{{"capability": {list(capability)}, "logical_index": 0, '
                    '"name": "NVIDIA GeForce RTX 5090", '
                    '"uuid": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"}]'
                ),
            )
        if command[-1] == "--list-gpu-arch":
            return subprocess.CompletedProcess(command, 0, stdout="compute_90\ncompute_120\n")
        if fail_install and "install" in command:
            raise install_error
        if fail_editable and "-e" in command and str(REPO_ROOT) in command:
            raise install_error
        if (
            fail_kernel
            and "install" in command
            and any("cti-kernels[cute,gdn,varlen] @" in part for part in command)
        ):
            raise install_error
        if fail_verify and "verify" in command:
            raise install_error
        if interrupt_install and "install" in command:
            raise KeyboardInterrupt
        if command[1:3] == ("env", "remove") and fail_cleanup:
            raise cleanup_error
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(bootstrap, "_run", run)
    return bootstrap, commands, command_envs, install_error


def test_bootstrap_uses_profile_name_and_disables_build_isolation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BUILD_TARGET", "rocm")
    monkeypatch.setenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "TRUE")
    monkeypatch.setenv("LDFLAGS", "-Wl,foreign")
    monkeypatch.setenv("PIP_TARGET", "/tmp/foreign-site-packages")
    monkeypatch.setenv("PYTHONPATH", "/tmp/foreign-pythonpath")
    monkeypatch.setenv("PYTORCH_NVCC", "/tmp/foreign-nvcc")
    monkeypatch.setenv("TMS_CUDA_MAJOR", "12")
    monkeypatch.setenv("PATH", "/usr/local/cuda/bin:/opt/bin:/usr/bin")
    # The build's MAX_JOBS follows available memory, which other processes move between the
    # build and the check below.
    monkeypatch.setattr(bootstrap_module, "_available_memory_gib", lambda: 64)
    wheel = _kernel_wheel(tmp_path)
    bootstrap, commands, command_envs, _ = _bootstrap_with_recording_runner(
        monkeypatch, kernel_wheel=wheel
    )

    bootstrap.create()

    environment = contract.load_runtime_profile().environment
    named_commands = [command for command in commands if "-n" in command]
    assert named_commands
    assert all(command[command.index("-n") + 1] == environment for command in named_commands)
    flash_attention_install = next(
        command
        for command in commands
        if contract.load_runtime_profile().flash_attention.requirement in command
    )
    tms_install = next(
        command
        for command in commands
        if contract.load_runtime_profile().torch_memory_saver.requirement in command
    )
    editable_install = next(
        command for command in commands if "-e" in command and str(REPO_ROOT) in command
    )
    project_requirements = contract.load_project_install_requirements()
    dependency_install = next(command for command in commands if project_requirements[0] in command)
    configure = next(command for command in commands if command[1:4] == ("env", "config", "vars"))
    verify = next(command for command in commands if "verify" in command)
    assert commands.index(configure) < commands.index(tms_install)
    assert commands.index(tms_install) < commands.index(flash_attention_install)
    assert commands.index(flash_attention_install) < commands.index(dependency_install)
    assert commands.index(dependency_install) < commands.index(editable_install)
    assert commands.index(editable_install) < commands.index(verify)
    assert dependency_install[-1] == f"cti-kernels[cute,gdn,varlen] @ {wheel.as_uri()}"
    assert "-e" not in dependency_install
    assert "--no-build-isolation" in dependency_install
    assert "--no-deps" not in dependency_install
    assert KERNEL_REQUIREMENT not in dependency_install
    assert "--only-binary=cti-kernels" in dependency_install
    assert not any(value.startswith("PATH=") for value in configure)
    assert verify[verify.index("python") + 1] == str(REPO_ROOT / "scripts" / "setup_env.py")
    assert verify[-2:] == ("--kernel-wheel", str(wheel))
    verify_env = command_envs[commands.index(verify)]
    assert verify_env is not None
    assert verify_env["CUDA_HOME"] == "/opt/conda/envs/cti-cu130-torch213"
    assert verify_env["PATH"].split(os.pathsep)[0] == ("/opt/conda/envs/cti-cu130-torch213/bin")
    assert "--isolated" in flash_attention_install
    assert "--no-build-isolation" in flash_attention_install
    assert "--no-cache-dir" in flash_attention_install
    assert "--no-deps" in flash_attention_install
    assert "--no-binary=flash-attn" in flash_attention_install
    assert "--isolated" in tms_install
    assert "--no-build-isolation" in tms_install
    assert "--no-cache-dir" in tms_install
    assert "--no-deps" in tms_install
    assert "--force-reinstall" in tms_install
    tms_build_env = command_envs[commands.index(tms_install)]
    assert tms_build_env is not None
    assert tms_build_env["CUDA_HOME"] == "/opt/conda/envs/cti-cu130-torch213"
    assert tms_build_env["TMS_CUDA_MAJOR"] == "13"
    assert tms_build_env["CC"].endswith("/bin/x86_64-conda-linux-gnu-cc")
    assert tms_build_env["CXX"].endswith("/bin/x86_64-conda-linux-gnu-c++")
    assert tms_build_env["CPATH"].endswith("/targets/x86_64-linux/include")
    assert "/usr/lib/cuda-driver" in tms_build_env["LIBRARY_PATH"]
    assert "PYTHONPATH" not in tms_build_env
    build_env = command_envs[commands.index(flash_attention_install)]
    assert build_env is not None
    assert build_env["BUILD_TARGET"] == "cuda"
    assert build_env["CUDA_HOME"] == "/opt/conda/envs/cti-cu130-torch213"
    assert build_env["FLASH_ATTENTION_FORCE_BUILD"] == "TRUE"
    assert build_env["FLASH_ATTN_CUDA_ARCHS"] == "120"
    assert build_env["MAX_JOBS"] == str(bootstrap_module.flash_attention_build_jobs())
    assert build_env["NVCC_THREADS"] == "2"
    for variable in (
        "FLASH_ATTENTION_SKIP_CUDA_BUILD",
        "LDFLAGS",
        "PIP_TARGET",
        "PYTHONPATH",
        "PYTORCH_NVCC",
    ):
        assert variable not in build_env
    assert "--no-build-isolation" in editable_install
    assert "--no-deps" in editable_install
    assert contract.load_runtime_profile().torch_memory_saver.requirement not in dependency_install
    for install in (dependency_install, editable_install):
        assert "--isolated" in install
        install_env = command_envs[commands.index(install)]
        assert install_env is not None
        for variable in ("PIP_TARGET", "PYTHONPATH", "PYTORCH_NVCC", "TMS_CUDA_MAJOR"):
            assert variable not in install_env
    assert not any(command[1:3] == ("env", "remove") for command in commands)


def test_bootstrap_constrains_every_pip_install_to_the_verified_pins(monkeypatch, tmp_path):
    bootstrap, commands, _, _ = _bootstrap_with_recording_runner(
        monkeypatch, kernel_wheel=_kernel_wheel(tmp_path)
    )
    record = bootstrap._run
    constraints: dict[str, str] = {}

    def run(command, **kwargs):
        if "pip" in command and "install" in command:
            path = command[command.index("-c") + 1]
            constraints[path] = Path(path).read_text(encoding="utf-8")
        return record(command, **kwargs)

    monkeypatch.setattr(bootstrap, "_run", run)
    bootstrap.create()

    profile = contract.load_runtime_profile()
    installs = [command for command in commands if "pip" in command and "install" in command]
    assert len(installs) == 4
    assert not any(
        part == "-U" or part.startswith("--upgrade") for install in installs for part in install
    )
    [(path, text)] = constraints.items()
    assert not Path(path).exists()
    # Includes packaging, which the dev group's virtualenv once pulled past its pin.
    pins = {f"{name}=={version}" for name, version in profile.distributions}
    assert set(text.splitlines()) == pins | {f"pip=={profile.pip_version}"}
    assert "nvidia-cutlass-dsl==4.8.0" in text.splitlines()
    assert "fla-core==0.5.2" in text.splitlines()
    assert "[" not in text


def _metadata_wheel(
    index: Path, name: str, version: str, *requires: str, extras: tuple[str, ...] = ()
) -> None:
    dist_info = f"{name}-{version}.dist-info"
    with ZipFile(index / f"{name}-{version}-py3-none-any.whl", "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Provides-Extra: {extra}\n" for extra in extras)
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires),
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")


@pytest.mark.parametrize("constrained", [False, True])
def test_bootstrap_constraints_keep_a_pin_through_a_dependency_install(tmp_path, constrained):
    """Resolve with the real pip offline (dry run) against a local index."""
    pinned = contract.load_runtime_profile().distribution_version("packaging")
    newer = f"{Version(pinned).major + 1}.0"
    index = tmp_path / "index"
    index.mkdir()
    _metadata_wheel(index, "packaging", pinned)
    _metadata_wheel(index, "packaging", newer)
    _metadata_wheel(index, "virtualenv", "1.0", "packaging>=23.1")
    _metadata_wheel(index, "virtualenv", "2.0", f"packaging>={newer}")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(
        bootstrap_module.EnvironmentBootstrap("conda")._pip_constraints(), encoding="utf-8"
    )
    report = tmp_path / "report.json"
    subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            "install",
            *(("-c", str(constraints)) if constrained else ()),
            "--dry-run",
            "--ignore-installed",
            "--no-cache-dir",
            "--no-index",
            "--find-links",
            str(index),
            "--quiet",
            "--report",
            str(report),
            "packaging>=24",
            "virtualenv",
        ),
        check=True,
        timeout=120,
    )

    resolved = {
        item["metadata"]["name"]: item["metadata"]["version"]
        for item in json.loads(report.read_text(encoding="utf-8"))["install"]
    }
    expected = (pinned, "1.0") if constrained else (newer, "2.0")
    assert (resolved["packaging"], resolved["virtualenv"]) == expected


def test_bootstrap_installs_the_pinned_release_wheel_by_default(monkeypatch):
    bootstrap, commands, _, _ = _bootstrap_with_recording_runner(monkeypatch)
    bootstrap.create()

    dependency_install = next(
        command for command in commands if KERNEL_REQUIREMENT in command and "install" in command
    )
    assert dependency_install[-1] == KERNEL_REQUIREMENT
    assert "--only-binary=cti-kernels" in dependency_install
    verify = next(command for command in commands if "verify" in command)
    assert "--kernel-wheel" not in verify


def test_bootstrap_installs_explicit_wheel_from_engine_only_checkout(monkeypatch, tmp_path):
    engine = tmp_path / "engine-only"
    engine.mkdir()
    (engine / "pyproject.toml").write_text((REPO_ROOT / "pyproject.toml").read_text())
    monkeypatch.setattr(bootstrap_module, "REPO_ROOT", engine)
    wheel = _kernel_wheel(tmp_path)
    bootstrap, commands, command_envs, _ = _bootstrap_with_recording_runner(
        monkeypatch, kernel_wheel=wheel
    )
    bootstrap.create()
    dependency_install = next(
        command
        for command in commands
        if f"cti-kernels[cute,gdn,varlen] @ {wheel.as_uri()}" in command and "install" in command
    )
    engine_install = next(
        command for command in commands if str(engine) in command and "-e" in command
    )
    verify = next(command for command in commands if "verify" in command)
    assert dependency_install[-1] == f"cti-kernels[cute,gdn,varlen] @ {wheel.as_uri()}"
    assert "-e" not in dependency_install and "--no-deps" not in dependency_install
    assert KERNEL_REQUIREMENT not in dependency_install
    assert "--no-deps" in engine_install
    assert (
        commands.index(dependency_install) < commands.index(engine_install) < commands.index(verify)
    )
    assert verify[-2:] == ("--kernel-wheel", str(wheel))
    assert command_envs[commands.index(dependency_install)] is not None
    assert not (engine / "packages").exists()


@pytest.mark.parametrize("failure", ["install", "verify"])
def test_bootstrap_wheel_failures_preserve_existing_rollback_contract(
    monkeypatch, tmp_path, capsys, failure
):
    wheel = _kernel_wheel(tmp_path)
    bootstrap, commands, _, error = _bootstrap_with_recording_runner(
        monkeypatch,
        kernel_wheel=wheel,
        fail_kernel=failure == "install",
        fail_verify=failure == "verify",
    )
    with pytest.raises(subprocess.CalledProcessError) as caught:
        bootstrap.create()
    assert caught.value is error
    removed = any(command[1:3] == ("env", "remove") for command in commands)
    assert removed is (failure == "install")
    if failure == "verify":
        assert "--kernel-wheel" in capsys.readouterr().err


def _prebuilt_bootstrap(monkeypatch, tmp_path, **options):
    """A recording bootstrap given a prebuilt flash-attn wheel its profile pins."""
    monkeypatch.setattr(bootstrap_module.platform, "machine", lambda: "x86_64")
    wheel, prebuilt = _prebuilt_flash_attention(tmp_path)
    bootstrap, commands, command_envs, _ = _bootstrap_with_recording_runner(
        monkeypatch, kernel_wheel=_kernel_wheel(tmp_path), flash_attn_wheel=wheel, **options
    )
    bootstrap._profile = replace(bootstrap._profile, flash_attention_prebuilt=prebuilt)
    return bootstrap, commands, command_envs, prebuilt.requirement(wheel.resolve(), "x86_64")


def test_bootstrap_installs_a_prebuilt_flash_attention_wheel_instead_of_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap, commands, command_envs, requirement = _prebuilt_bootstrap(monkeypatch, tmp_path)

    bootstrap.create()

    installs = [command for command in commands if "pip" in command and "install" in command]
    tms_install, flash_attention_install, dependency_install, _ = installs
    assert bootstrap._profile.torch_memory_saver.requirement in tms_install
    assert flash_attention_install[-1] == requirement
    assert "#sha256=" in requirement
    for flag in ("--isolated", "--no-build-isolation", "--no-cache-dir", "--no-deps"):
        assert flag in flash_attention_install
    assert "--no-binary=flash-attn" not in flash_attention_install
    flash_attention_env = command_envs[commands.index(flash_attention_install)]
    assert "FLASH_ATTN_CUDA_ARCHS" not in flash_attention_env
    assert flash_attention_env["PYTHONNOUSERSITE"] == "1"
    assert command_envs[commands.index(dependency_install)]["PYTHONNOUSERSITE"] == "1"
    # No source build: neither nvcc's target list nor the locked sdist.
    assert not any(command[-1] == "--list-gpu-arch" for command in commands)
    assert not any(
        bootstrap._profile.flash_attention.requirement in command for command in commands
    )
    assert dependency_install[-1].startswith("cti-kernels[cute,gdn,varlen] @ ")
    verify = next(command for command in commands if "verify" in command)
    assert "--flash-attn-wheel" not in verify


def test_bootstrap_refuses_a_prebuilt_flash_attention_wheel_that_misses_a_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap, commands, _, _ = _prebuilt_bootstrap(monkeypatch, tmp_path, capability=(9, 0))

    with pytest.raises(contract.VerificationError, match="omit --flash-attn-wheel") as caught:
        bootstrap.create()

    assert "SM90" in str(caught.value)
    assert not any("install" in command for command in commands)
    assert any(command[1:3] == ("env", "remove") for command in commands)


@pytest.mark.parametrize(
    ("change", "message"),
    [("tampered", "has SHA256"), ("other-machine", "supported machines")],
)
def test_bootstrap_rejects_an_invalid_flash_attention_wheel_before_creating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str, message: str
) -> None:
    bootstrap, _, _, _ = _prebuilt_bootstrap(monkeypatch, tmp_path)
    if change == "tampered":
        bootstrap._flash_attn_wheel.write_bytes(b"tampered")
    else:
        monkeypatch.setattr(bootstrap_module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("environment command before validation"),
    )

    with pytest.raises(contract.VerificationError, match=message):
        bootstrap.create()


def test_bootstrap_creates_without_a_gpu_from_the_prebuilt_wheel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    driver = Path("/usr/lib/x86_64-linux-gnu")
    bootstrap, commands, command_envs, requirement = _prebuilt_bootstrap(
        monkeypatch, tmp_path, no_gpu=True, cuda_driver_dir=driver
    )
    monkeypatch.setattr(
        bootstrap_module,
        "cuda_driver_library_dir",
        lambda: pytest.fail("a GPU-less build loaded the CUDA driver"),
    )

    bootstrap.create()

    assert not any(
        any("torch.cuda.get_device_capability" in part for part in command) for command in commands
    )
    assert not any("verify" in command for command in commands)
    installs = [command for command in commands if "pip" in command and "install" in command]
    tms_install, flash_attention_install, _, _ = installs
    assert flash_attention_install[-1] == requirement
    # The environment records the target's driver directory; the build links against the stub.
    configure = next(command for command in commands if command[1:4] == ("env", "config", "vars"))
    for variable in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        value = next(part for part in configure if part.startswith(f"{variable}="))
        assert value.split("=", 1)[1].split(os.pathsep)[-1] == str(driver)
    stubs = "/opt/conda/envs/cti-cu130-torch213/targets/x86_64-linux/lib/stubs"
    tms_library_path = command_envs[commands.index(tms_install)]["LIBRARY_PATH"].split(os.pathsep)
    assert stubs in tms_library_path
    assert str(driver) not in tms_library_path
    output = capsys.readouterr().out
    assert "without a GPU" in output
    # The GPU host verifies without the build's wheel paths.
    assert output.rstrip().endswith("scripts/setup_env.py verify")
    assert "--kernel-wheel" not in output


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"no_gpu": True, "cuda_driver_dir": Path("/usr/lib")}, "requires --flash-attn-wheel"),
        ({"no_gpu": True, "flash_attn_wheel": Path("fa.whl")}, "go together"),
        ({"cuda_driver_dir": Path("/usr/lib")}, "go together"),
    ],
)
def test_bootstrap_rejects_an_incomplete_gpu_less_build(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_module.EnvironmentBootstrap("conda", **options)


def test_bootstrap_keeps_jit_caches_in_the_requested_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    bootstrap, commands, _, _ = _bootstrap_with_recording_runner(
        monkeypatch, kernel_wheel=_kernel_wheel(tmp_path), cache_dir=cache
    )

    bootstrap.create()

    configure = next(command for command in commands if command[1:4] == ("env", "config", "vars"))
    variables = dict(part.split("=", 1) for part in configure if "=" in part)
    for variable, leaf in (
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
        ("TVM_FFI_CACHE_DIR", "tvm-ffi"),
        ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
        ("TORCHINDUCTOR_CACHE_DIR", "torch-inductor"),
        ("CUTE_DSL_CACHE_DIR", "cute-dsl"),
    ):
        assert variables[variable] == str(cache / "cu130-torch213" / leaf)


def test_bootstrap_creates_the_profile_under_flexible_channel_priority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONDA_CHANNEL_PRIORITY", "strict")
    bootstrap, commands, command_envs, _ = _bootstrap_with_recording_runner(
        monkeypatch, kernel_wheel=_kernel_wheel(tmp_path)
    )

    bootstrap.create()

    create = next(command for command in commands if command[1:3] == ("env", "create"))
    create_env = command_envs[commands.index(create)]
    assert create_env == {**os.environ, "CONDA_CHANNEL_PRIORITY": "flexible"}
    assert os.environ["CONDA_CHANNEL_PRIORITY"] == "strict"
    assert all(
        env is None or env.get("CONDA_CHANNEL_PRIORITY") == "strict"
        for command, env in zip(commands, command_envs, strict=True)
        if command is not create
    )


def test_bootstrap_maps_sm121_to_the_locked_flash_attention_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = bootstrap_module.EnvironmentBootstrap("conda")
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda _command, capture_output=False: subprocess.CompletedProcess(
            _command,
            0,
            stdout="compute_100\ncompute_120\ncompute_121\n",
        ),
    )
    gb10 = _cuda_device(name="NVIDIA GB10", capability=(12, 1))

    assert bootstrap._flash_attention_build_arches(Path("/formal-prefix"), (gb10,)) == ("120",)


def test_bootstrap_forwards_named_qualification_to_final_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    qualification = contract.load_device_qualification("rtx5090")
    bootstrap, commands, _, _ = _bootstrap_with_recording_runner(
        monkeypatch,
        kernel_wheel=_kernel_wheel(tmp_path),
        qualification=qualification,
    )

    bootstrap.create()

    verify = next(command for command in commands if "verify" in command)
    assert verify[-2:] == ("--qualification", "rtx5090")


def test_bootstrap_rolls_back_a_failed_fresh_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bootstrap, commands, _, install_error = _bootstrap_with_recording_runner(
        monkeypatch,
        kernel_wheel=_kernel_wheel(tmp_path),
        fail_install=True,
    )

    with pytest.raises(subprocess.CalledProcessError) as caught:
        bootstrap.create()

    assert caught.value is install_error
    assert any(command[1:3] == ("env", "remove") for command in commands)


@pytest.mark.parametrize("failure", ("config", "kernel", "editable"))
def test_bootstrap_rolls_back_after_later_configuration_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    bootstrap, commands, _, install_error = _bootstrap_with_recording_runner(
        monkeypatch,
        kernel_wheel=_kernel_wheel(tmp_path),
        fail_config=failure == "config",
        fail_kernel=failure == "kernel",
        fail_editable=failure == "editable",
    )

    with pytest.raises(subprocess.CalledProcessError) as caught:
        bootstrap.create()

    assert caught.value is install_error
    assert any(command[1:3] == ("env", "remove") for command in commands)


def test_bootstrap_preserves_a_configured_environment_after_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bootstrap, commands, _, install_error = _bootstrap_with_recording_runner(
        monkeypatch,
        kernel_wheel=_kernel_wheel(tmp_path),
        fail_verify=True,
    )

    with pytest.raises(subprocess.CalledProcessError) as caught:
        bootstrap.create()

    assert caught.value is install_error
    assert not any(command[1:3] == ("env", "remove") for command in commands)
    stderr = capsys.readouterr().err
    assert "installed but did not pass verification" in stderr
    assert "conda run --no-capture-output -n cti-cu130-torch213" in stderr
    assert "scripts/setup_env.py verify" in stderr


def test_bootstrap_cleanup_failure_preserves_the_original_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bootstrap, commands, _, install_error = _bootstrap_with_recording_runner(
        monkeypatch,
        kernel_wheel=_kernel_wheel(tmp_path),
        fail_install=True,
        fail_cleanup=True,
    )

    with pytest.raises(subprocess.CalledProcessError) as caught:
        bootstrap.create()

    assert caught.value is install_error
    assert any(command[1:3] == ("env", "remove") for command in commands)


def test_bootstrap_does_not_remove_an_unowned_environment_after_create_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bootstrap, commands, _, create_error = _bootstrap_with_recording_runner(
        monkeypatch,
        kernel_wheel=_kernel_wheel(tmp_path),
        fail_create=True,
    )

    with pytest.raises(subprocess.CalledProcessError) as caught:
        bootstrap.create()

    assert caught.value is create_error
    assert not any(command[1:3] == ("env", "remove") for command in commands)


def test_bootstrap_removes_a_fresh_environment_when_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bootstrap, commands, _, _ = _bootstrap_with_recording_runner(
        monkeypatch,
        kernel_wheel=_kernel_wheel(tmp_path),
        interrupt_install=True,
    )

    with pytest.raises(KeyboardInterrupt):
        bootstrap.create()

    assert any(command[1:3] == ("env", "remove") for command in commands)


def test_verifier_timeout_preserves_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout = subprocess.TimeoutExpired(
        cmd=("child",),
        timeout=180,
        output="partial stdout",
        stderr="partial stderr",
    )

    def raise_timeout(*_args, **_kwargs):
        raise timeout

    monkeypatch.setattr(verify_module.subprocess, "run", raise_timeout)

    with pytest.raises(contract.VerificationError, match="timed out") as caught:
        verify_module.EnvironmentVerifier._run(("child",))

    assert "partial stdout" in str(caught.value)
    assert "partial stderr" in str(caught.value)


@pytest.mark.parametrize(
    "extras",
    ["", "[gdn]", "[cute]", "[varlen]", "[cute,gdn]", "[cute,gdn,varlen,dev]", "[extra]"],
)
def test_kernel_dependency_requires_exactly_the_cute_gdn_and_varlen_extras(tmp_path, extras):
    path = _project_file(tmp_path, [f"cti-kernels{extras}==7.2"])
    with pytest.raises(contract.VerificationError, match=r"\[cute,gdn,varlen\] extras"):
        contract.required_kernel_version(path)


@pytest.mark.parametrize("distribution", ["fla-core", "einops"])
def test_runtime_profile_requires_gdn_closure_pins(tmp_path, distribution):
    profile = contract.load_runtime_profile()
    pin = f"      - {distribution}=={profile.distribution_version(distribution)}\n"
    path = tmp_path / "profile.yml"
    path.write_text(PROFILE_PATH.read_text().replace(pin, ""))
    with pytest.raises(contract.VerificationError, match=distribution):
        contract.load_runtime_profile(path)


@pytest.mark.parametrize("distribution", ["fla-core", "einops"])
@pytest.mark.parametrize("installed", [None, "0.0.0", "pinned"])
def test_verifier_checks_gdn_closure_before_native_checks(monkeypatch, distribution, installed):
    profile = contract.load_runtime_profile()
    versions = dict(profile.distributions, pip=profile.pip_version)
    if installed is None:
        del versions[distribution]
    elif installed != "pinned":
        versions[distribution] = installed

    def version(name):
        if name not in versions:
            raise verify_module.PackageNotFoundError(name)
        return versions[name]

    class NativeChecksReached(Exception):
        pass

    def native_check(*args):
        raise NativeChecksReached

    verifier = verify_module.EnvironmentVerifier(profile=profile)
    monkeypatch.setattr(verify_module, "version", version)
    monkeypatch.setattr(verify_module.sys, "version_info", profile.python)
    monkeypatch.setattr(verifier, "_verify_conda_package", native_check)
    if installed == "pinned":
        with pytest.raises(NativeChecksReached):
            verifier.verify()
    else:
        with pytest.raises(contract.VerificationError, match=distribution):
            verifier.verify()


def _kernel_index(tmp_path: Path) -> Path:
    """An offline index with the kernel, its extras' dependencies and the pip constraints."""
    index = tmp_path / "wheel index #1"
    index.mkdir()
    _metadata_wheel(
        index,
        "cti_kernels",
        KERNEL_VERSION,
        'nvidia-cutlass-dsl[cu13]==4.8.0; extra == "cute"',
        'fla-core==0.5.2; extra == "gdn"',
        'flash-attn==2.8.3.post1; extra == "varlen"',
        extras=("cute", "gdn", "varlen"),
    )
    _metadata_wheel(
        index,
        "nvidia_cutlass_dsl",
        "4.8.0",
        "nvidia-cutlass-dsl-libs-base==4.8.0",
        'nvidia-cutlass-dsl-libs-cu13==4.8.0; extra == "cu13"',
        extras=("cu13",),
    )
    for library in ("base", "cu13"):
        _metadata_wheel(index, f"nvidia_cutlass_dsl_libs_{library}", "4.8.0")
    _metadata_wheel(index, "fla_core", "0.5.2", "einops")
    _metadata_wheel(index, "einops", "0.8.2")
    _metadata_wheel(index, "einops", "99.0")
    # The varlen extra's flash-attn, held at the locked version by the profile constraints.
    _metadata_wheel(index, "flash_attn", contract.FlashAttentionLock.load().version)
    (tmp_path / "constraints.txt").write_text(
        bootstrap_module.EnvironmentBootstrap("conda")._pip_constraints()
    )
    return index


@pytest.mark.parametrize("local_wheel", [False, True], ids=["index", "local-wheel"])
def test_kernel_install_resolves_its_extras_and_offline_reinstall_preserves_them(
    tmp_path, local_wheel
):
    """Real offline pip exercises extra selection and an offline reinstall of the wheel."""
    index = _kernel_index(tmp_path)
    wheel = index / f"cti_kernels-{KERNEL_VERSION}-py3-none-any.whl"
    target = contract.KernelInstallTarget.load(wheel=wheel if local_wheel else None)
    constraints = tmp_path / "constraints.txt"
    installed = tmp_path / "installed"
    pip = (sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "install")

    def install(*args):
        subprocess.run(
            (
                *pip,
                "--no-index",
                "--no-cache-dir",
                "--no-compile",
                "--target",
                str(installed),
                *args,
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

    install(
        "--ignore-installed",
        "--find-links",
        str(index),
        "-c",
        str(constraints),
        *target.install_args,
    )

    def versions():
        from packaging.utils import canonicalize_name

        return {
            canonicalize_name(dist.metadata["Name"]): dist.version
            for dist in Distribution.discover(path=[str(installed)])
        }

    expected = {
        "cti-kernels": KERNEL_VERSION,
        "nvidia-cutlass-dsl": "4.8.0",
        "nvidia-cutlass-dsl-libs-base": "4.8.0",
        "nvidia-cutlass-dsl-libs-cu13": "4.8.0",
        "fla-core": "0.5.2",
        "einops": "0.8.2",
        "flash-attn": contract.FlashAttentionLock.load().version,
    }
    assert versions() == expected
    # A provisioned environment may reinstall just the artifact.
    install("--no-deps", "--force-reinstall", "--upgrade", str(wheel))
    assert versions() == expected


def test_kernel_install_keeps_the_flash_attention_installed_before_it(tmp_path):
    """Real offline pip: the varlen extra keeps setup's hash-recorded flash-attn."""
    index = _kernel_index(tmp_path)
    version = contract.FlashAttentionLock.load().version
    flash_attention = tmp_path / f"flash_attn-{version}-py3-none-any.whl"
    flash_attention.write_bytes((index / flash_attention.name).read_bytes())
    with flash_attention.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    environment = tmp_path / "venv"
    venv.create(environment)
    pip = (
        sys.executable,
        "-m",
        "pip",
        "--isolated",
        "--disable-pip-version-check",
        "--python",
        str(environment / "bin" / "python"),
        "install",
        "--no-index",
        "--no-cache-dir",
        "--no-compile",
    )

    def install(*args):
        subprocess.run((*pip, *args), check=True, capture_output=True, text=True, timeout=120)

    install("--no-deps", f"flash-attn @ {flash_attention.as_uri()}#sha256={digest}")
    (record,) = environment.glob(f"lib/python*/site-packages/flash_attn-{version}.dist-info")
    direct_url = (record / "direct_url.json").read_text()
    assert json.loads(direct_url)["archive_info"]["hashes"] == {"sha256": digest}

    wheel = index / f"cti_kernels-{KERNEL_VERSION}-py3-none-any.whl"
    install(
        "--find-links",
        str(index),
        "-c",
        str(tmp_path / "constraints.txt"),
        *contract.KernelInstallTarget.load(wheel=wheel).install_args,
    )

    assert (record / "direct_url.json").read_text() == direct_url


def test_flash_attention_build_jobs_follow_cpus_memory_and_the_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(224)))
    monkeypatch.setattr(bootstrap_module, "_available_memory_gib", lambda: 2000)
    assert bootstrap_module.flash_attention_build_jobs() == 112
    monkeypatch.setattr(bootstrap_module, "_available_memory_gib", lambda: 64)
    assert bootstrap_module.flash_attention_build_jobs() == 8
    monkeypatch.setattr(bootstrap_module, "_available_memory_gib", lambda: 4)
    assert bootstrap_module.flash_attention_build_jobs() == 1
    monkeypatch.setenv("MAX_JOBS", "3")
    assert bootstrap_module.flash_attention_build_jobs() == 3


def test_create_refuses_a_source_tree_without_git_before_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = bootstrap_module.EnvironmentBootstrap("conda")
    commands: list[tuple[str, ...]] = []

    def run(command, **_kwargs):
        commands.append(tuple(command))
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(bootstrap, "_run", run)
    with pytest.raises(RuntimeError, match="not a git checkout and has no REVISION file"):
        bootstrap.create()
    assert commands == [("git", "rev-parse", "HEAD")]


_REVISION = "0123456789abcdef0123456789abcdef01234567"


def test_create_takes_the_revision_file_of_a_tree_without_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap = bootstrap_module.EnvironmentBootstrap("conda")
    monkeypatch.setattr(bootstrap_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        bootstrap, "_run", lambda *_args, **_kwargs: pytest.fail("ran git without .git")
    )
    (tmp_path / "REVISION").write_text(f"{_REVISION}\n")

    bootstrap._require_source_checkout()

    (tmp_path / "REVISION").write_text("main\n")
    with pytest.raises(contract.VerificationError, match="one full commit SHA"):
        bootstrap._require_source_checkout()


def test_verifier_records_the_revision_file_of_a_tree_without_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = verify_module.EnvironmentVerifier(repo_root=tmp_path)
    (tmp_path / "REVISION").write_text(f"{_REVISION}\n")
    monkeypatch.setattr(verifier, "_run", lambda _command: pytest.fail("ran git without .git"))

    assert verifier._source_state() == (_REVISION, None)


@pytest.mark.parametrize("revision_file", (False, True), ids=("no-revision-file", "revision-file"))
def test_verifier_reads_a_git_checkout_from_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, revision_file: bool
) -> None:
    (tmp_path / ".git").mkdir()
    if revision_file:
        (tmp_path / "REVISION").write_text("not a revision\n")
    verifier = verify_module.EnvironmentVerifier(repo_root=tmp_path)
    outputs = {"rev-parse": "f" * 40, "status": " M README.md"}
    monkeypatch.setattr(verifier, "_run", lambda command: outputs[command[3]])

    assert verifier._source_state() == ("f" * 40, True)


def test_verifier_still_requires_git_for_a_tree_without_a_revision_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verifier = verify_module.EnvironmentVerifier(repo_root=tmp_path)
    commands = []

    def fail(command):
        commands.append(tuple(command))
        raise contract.VerificationError("not a git repository")

    monkeypatch.setattr(verifier, "_run", fail)

    with pytest.raises(contract.VerificationError):
        verifier._source_state()
    assert commands == [("git", "-C", str(tmp_path.resolve()), "rev-parse", "HEAD")]
