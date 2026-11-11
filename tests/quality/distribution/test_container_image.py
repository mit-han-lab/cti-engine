"""The container image's build: docker/Dockerfile and the build context it copies."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

from scripts.environment import contract
from tests.support.paths import REPO_ROOT

DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"


def _patterns(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def test_the_image_takes_its_wheel_pins_from_the_install_script() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    install = (REPO_ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    pins = re.findall(r'^PINNED_\w+="([^"]+)"$', install, re.MULTILINE)
    prebuilt = contract.FlashAttentionPrebuilt.load()

    assert len(pins) == 5
    assert "scripts/install.sh --fetch-only" in dockerfile
    assert "scripts/setup_env.py create --no-gpu" in dockerfile
    for literal in (
        contract.required_kernel_version(),
        contract.FlashAttentionLock.load().version,
        *pins,
        *(sha256 for _, _, sha256 in prebuilt.wheels),
    ):
        assert literal not in dockerfile


def test_the_image_references_only_paths_of_the_repository() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    # Not the `syntax=docker/dockerfile` frontend image, nor paths under the image's root.
    paths = {
        path.rstrip(".")
        for path in re.findall(
            r"(?<![\w./=-])((?:docker|environments|scripts)/[\w./-]+)", dockerfile
        )
    }

    assert {"scripts/install.sh", "scripts/setup_env.py", "scripts/smoke_check.py"} <= paths
    # An offline build places the release wheels in docker/wheels/, which Git ignores.
    assert not [
        path
        for path in sorted(paths - {"docker/wheels", "docker/wheels/"})
        if not (REPO_ROOT / path).exists()
    ]
    # Its default command is the install script's smoke check.
    assert 'CMD ["python", "scripts/smoke_check.py"]' in dockerfile
    install = (REPO_ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    assert '"$REPO_ROOT/scripts/smoke_check.py"' in install


def test_the_image_records_its_revision_instead_of_git_metadata() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG REVISION" in dockerfile
    assert '"$REVISION" > REVISION' in dockerfile
    assert {"/REVISION", "/docker/wheels/"} <= set(_patterns(REPO_ROOT / ".gitignore"))


def test_the_build_context_admits_only_the_sources_the_image_needs() -> None:
    dockerignore = _patterns(REPO_ROOT / ".dockerignore")
    admitted = {pattern[1:] for pattern in dockerignore if pattern.startswith("!")}

    # Everything else stays out, .git and REVISION included.
    assert dockerignore[0] == "*"
    assert {"pyproject.toml", "README.md", "LICENSE", "python", "scripts", "environments"} <= (
        admitted
    )
    assert all((REPO_ROOT / path).exists() for path in admitted)
    assert not {".git", "REVISION"} & admitted
    # Within the admitted sources, nothing .gitignore excludes: hidden files, caches, output.
    assert "**/.*" in dockerignore
    for line in _patterns(REPO_ROOT / ".gitignore"):
        pattern = line.rstrip("/")
        if pattern.startswith(("!", ".")):
            continue
        if pattern == "/docker/wheels":
            # The build mounts them; no layer copies them.
            assert "COPY --exclude=docker/wheels . ." in DOCKERFILE.read_text(encoding="utf-8")
        elif pattern.startswith("/"):
            assert pattern[1:].split("/")[0] not in admitted, pattern
        else:
            assert f"**/{pattern}" in dockerignore, pattern


def test_the_readme_pulls_the_image_of_this_version() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    tags = re.findall(r"ghcr\.io/mit-han-lab/cti-engine:([\w.+-]+)", readme)
    assert tags
    assert set(tags) == {pyproject["project"]["version"]}
