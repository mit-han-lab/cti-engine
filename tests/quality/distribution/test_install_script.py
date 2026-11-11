"""scripts/install.sh against stub tools: a copy whose pins name stand-in wheels."""

from __future__ import annotations

import hashlib
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.environment import contract
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"),
    reason="install.sh supports Linux on x86_64 and aarch64",
)

KERNEL_VERSION = contract.required_kernel_version()
KERNEL_WHEEL = f"cti_kernels-{KERNEL_VERSION}-py3-none-any.whl"
FLASH_ATTN_WHEEL = (
    f"flash_attn-{contract.FlashAttentionLock.load().version}"
    f"-cp312-cp312-linux_{platform.machine()}.whl"
)
WHEELS = {KERNEL_WHEEL: b"stand-in kernel wheel", FLASH_ATTN_WHEEL: b"stand-in flash-attn wheel"}
RELEASE_URL = f"https://github.com/mit-han-lab/cti-kernel/releases/download/v{KERNEL_VERSION}"

# Every stub logs its command line. curl and `gh release download` copy the named release asset
# unless the test makes them fail; any other gh command fails, as `gh auth status` does for a
# token that lacks an optional scope.
_STUBS = {
    "curl": """
[[ -z "${CURL_FAILS:-}" ]] || exit 22
while (($#)); do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    *) url="$1"; shift ;;
  esac
done
cp "$ASSETS/${url##*/}" "$out"
""",
    "gh": """
[[ -z "${GH_FAILS:-}" && "$1 $2" == "release download" ]] || exit 1
while (($#)); do
  case "$1" in
    --pattern) pattern="$2"; shift 2 ;;
    --dir) dir="$2"; shift 2 ;;
    *) shift ;;
  esac
done
cp "$ASSETS/$pattern" "$dir/"
""",
    "conda": "exit 1\n",
    "git": "exit 1\n",
    "nvidia-smi": "exit 1\n",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def sandbox(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "pyproject.toml", root / "pyproject.toml")
    script = (REPO_ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    script, kernel_pins = re.subn(
        r'^(PINNED_KERNEL_SHA256)=".*"$',
        rf'\1="{_sha256(WHEELS[KERNEL_WHEEL])}"',
        script,
        flags=re.MULTILINE,
    )
    script, flash_attention_pins = re.subn(
        r'^(PINNED_FLASH_ATTN_SHA256_\w+)=".*"$',
        rf'\1="{_sha256(WHEELS[FLASH_ATTN_WHEEL])}"',
        script,
        flags=re.MULTILINE,
    )
    assert (kernel_pins, flash_attention_pins) == (1, 2)
    (root / "scripts" / "install.sh").write_text(script, encoding="utf-8")

    assets = tmp_path / "release"
    assets.mkdir()
    for name, data in WHEELS.items():
        (assets / name).write_bytes(data)
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in _STUBS.items():
        stub = stubs / name
        stub.write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> "$STUB_LOG"\n{body}')
        stub.chmod(0o755)
    return {
        "script": root / "scripts" / "install.sh",
        "assets": assets,
        "stubs": stubs,
        "cache": tmp_path / "cache",
        "out": tmp_path / "out",
        "log": tmp_path / "commands.log",
        "home": tmp_path / "home",
    }


def _install(sandbox: dict[str, Path], *args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ("bash", str(sandbox["script"]), *args),
        env={
            "PATH": f"{sandbox['stubs']}:/usr/bin:/bin",
            "HOME": str(sandbox["home"]),
            "CTI_CACHE_DIR": str(sandbox["cache"]),
            "ASSETS": str(sandbox["assets"]),
            "STUB_LOG": str(sandbox["log"]),
            **env,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )


def _commands(sandbox: dict[str, Path]) -> list[str]:
    log = sandbox["log"]
    return log.read_text().splitlines() if log.exists() else []


def _fetched(sandbox: dict[str, Path]) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sandbox["out"].iterdir()}


def test_fetch_only_downloads_both_pinned_wheels_without_host_checks(sandbox) -> None:
    completed = _install(sandbox, "--fetch-only", str(sandbox["out"]))

    assert completed.returncode == 0, completed.stderr
    assert _fetched(sandbox) == WHEELS
    commands = _commands(sandbox)
    assert [command.split()[-1] for command in commands] == [
        f"{RELEASE_URL}/{KERNEL_WHEEL}",
        f"{RELEASE_URL}/{FLASH_ATTN_WHEEL}",
    ]
    assert all(command.startswith("curl ") for command in commands)


@pytest.mark.parametrize("tampered", (False, True), ids=("verified", "tampered"))
def test_fetch_only_takes_cached_wheels_only_when_their_sha256_matches(sandbox, tampered) -> None:
    # The container build points the cache at docker/wheels/, where an offline build places them.
    cached = sandbox["cache"] / "wheels"
    cached.mkdir(parents=True)
    for name, data in WHEELS.items():
        (cached / name).write_bytes(b"tampered" if tampered else data)

    completed = _install(sandbox, "--fetch-only", str(sandbox["out"]))

    assert completed.returncode == 0, completed.stderr
    assert _fetched(sandbox) == WHEELS
    assert len(_commands(sandbox)) == (2 if tampered else 0)


def test_gh_fallback_does_not_depend_on_gh_auth_status(sandbox) -> None:
    completed = _install(sandbox, "--fetch-only", str(sandbox["out"]), CURL_FAILS="1")

    assert completed.returncode == 0, completed.stderr
    assert _fetched(sandbox) == WHEELS
    gh = [command for command in _commands(sandbox) if command.startswith("gh ")]
    assert [command.split()[:3] for command in gh] == [["gh", "release", "download"]] * 2
    assert f"--pattern {KERNEL_WHEEL}" in gh[0]


def test_an_install_checks_the_gpu_before_fetching(sandbox) -> None:
    completed = _install(sandbox)

    assert completed.returncode == 1
    assert "nvidia-smi could not query a GPU" in completed.stderr
    assert not any(command.startswith("curl ") for command in _commands(sandbox))


def test_fetch_only_fails_without_the_prebuilt_flash_attention_wheel(sandbox) -> None:
    (sandbox["assets"] / FLASH_ATTN_WHEEL).unlink()

    completed = _install(sandbox, "--fetch-only", str(sandbox["out"]))

    assert completed.returncode == 1
    assert "could not fetch a verified prebuilt flash-attn wheel" in completed.stderr
    assert not sandbox["out"].exists()


def test_fetch_only_fails_when_no_source_has_a_wheel(sandbox) -> None:
    completed = _install(sandbox, "--fetch-only", str(sandbox["out"]), CURL_FAILS="1", GH_FAILS="1")

    assert completed.returncode == 1
    assert f"could not download {KERNEL_WHEEL}" in completed.stderr
    assert not sandbox["out"].exists()


@pytest.mark.parametrize("option", (("--build-flash-attn",), ("--flash-attn-wheel", "fa.whl")))
def test_fetch_only_takes_only_the_prebuilt_flash_attention_wheel(sandbox, option) -> None:
    completed = _install(sandbox, "--fetch-only", str(sandbox["out"]), *option)

    assert completed.returncode == 1
    assert "drop --flash-attn-wheel and --build-flash-attn" in completed.stderr
    assert _commands(sandbox) == []
