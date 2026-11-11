from __future__ import annotations

import os
import re
import subprocess
import sys
from email.parser import Parser
from pathlib import Path
from shutil import copy2
from tarfile import open as open_tar
from zipfile import ZipFile

import pytest
import tomllib
from packaging.requirements import Requirement
from setuptools import build_meta

from tests.support.imports import imports_from_source
from tests.support.paths import REPO_ROOT
from tests.support.repository import git_visible_paths, read_repository_entry

_FORBIDDEN_ARTIFACT_PARTS = (
    "benchmark/",
    "cookbook/",
    "docs/",
    "cti/benchmark/",
    "cti/shell.py",
    "environments/",
    "scripts/",
    "tests/",
)
_PRODUCTION_SOURCE_SUFFIXES = frozenset({".cu", ".cuh", ".cpp", ".h", ".py"})


def _contains_forbidden_release_path(path: str) -> bool:
    normalized = path.removeprefix("./")
    # Hidden files and folders hold repository tooling.
    return any(part.startswith(".") for part in normalized.split("/")) or any(
        normalized.startswith(part) or f"/{part}" in normalized
        for part in _FORBIDDEN_ARTIFACT_PARTS
    )


def _copy_git_visible_source_tree(destination: Path) -> None:
    for source in git_visible_paths():
        relative_path = source.relative_to(REPO_ROOT)
        if not source.exists() and not source.is_symlink():
            # The release surface follows the reviewable worktree, where an
            # unstaged deletion is absent even though it remains in the index.
            continue
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            link = Path(os.readlink(source))
            if link.is_absolute() or ".." in link.parts:
                raise RuntimeError(f"unsafe source symlink: {relative_path} -> {link}")
            target.symlink_to(link)
            continue
        if not source.is_file():
            raise RuntimeError(f"unsupported source entry: {relative_path}")
        copy2(source, target)


# Byte patterns, applied to every Git-visible regular file regardless of type:
# a developer home path in any form (bare, or inside a file:// URI).
_FORBIDDEN_CONTENT = {
    "developer home path": re.compile(rb"/(?:home|Users)/[A-Za-z0-9._-]+/"),
}
_SCAN_SENTINELS = (
    "README.md",
    "pyproject.toml",
    "THIRD_PARTY_NOTICES",
    ".pre-commit-config.yaml",
    "python/cti/__init__.py",
)


@pytest.mark.parametrize(
    "sample",
    [
        b"see /home/alice/checkout/x.py",
        b"file:///home/alice/checkout/x.py",
        b"file:///Users/alice/checkout/x.py",
        b"path=/Users/alice/x",
    ],
)
def test_forbidden_content_patterns_match_every_form(sample: bytes) -> None:
    assert any(pattern.search(sample) for pattern in _FORBIDDEN_CONTENT.values())


def test_forbidden_content_patterns_ignore_public_paths() -> None:
    for sample in (
        b"https://github.com/mit-han-lab/cti-engine",
        b"/home/",
        b"$HOME/.cache",
        b"~/x",
    ):
        assert not any(pattern.search(sample) for pattern in _FORBIDDEN_CONTENT.values())


def test_repository_entry_reader_reads_regular_files(tmp_path: Path) -> None:
    regular = tmp_path / "regular.txt"
    regular.write_bytes(b"plain content")
    assert read_repository_entry(regular) == b"plain content"


def test_repository_entry_reader_scans_link_targets_not_linked_content(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_bytes(b"/home/alice/never-read")
    repo = tmp_path / "repo"
    repo.mkdir()
    link = repo / "link"
    link.symlink_to(secret)

    payload = read_repository_entry(link)

    assert payload == os.fsencode(str(secret))
    assert b"never-read" not in payload


def test_repository_entry_reader_still_scans_broken_links(tmp_path: Path) -> None:
    link = tmp_path / "dangling"
    link.symlink_to("/home/alice/gone")
    assert not link.exists()
    assert read_repository_entry(link) == b"/home/alice/gone"
    assert read_repository_entry(tmp_path) is None  # a directory contributes nothing


def test_release_repository_has_no_local_checkout_dependency() -> None:
    """Every Git-visible entry, whatever its type, is free of developer machine
    paths."""
    # This module names the forbidden patterns and is the only file allowed to.
    this_test = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    scanned: set[str] = set()
    violations: dict[str, list[str]] = {}
    for path in git_visible_paths():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative == this_test:
            continue
        data = read_repository_entry(path)
        if data is None:
            continue
        scanned.add(relative)
        hits = [label for label, pattern in _FORBIDDEN_CONTENT.items() if pattern.search(data)]
        if hits:
            violations[relative] = hits
    assert set(_SCAN_SENTINELS) <= scanned
    assert not violations


def test_no_tracked_file_is_ignored() -> None:
    # `git add -A` on a copy of the tree leaves out ignored files, tracked or not.
    completed = subprocess.run(
        ("git", "ls-files", "--cached", "--ignored", "--exclude-standard"),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.split() == []


def test_embedded_shell_is_absent_from_the_release_surface() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    cpu_dependencies = metadata["dependency-groups"]["cpu-ci"]

    assert not (REPO_ROOT / "python/cti/shell.py").exists()
    assert all(not dependency.lower().startswith("prompt-toolkit") for dependency in dependencies)
    assert all(
        not dependency.lower().startswith("prompt-toolkit") for dependency in cpu_dependencies
    )
    assert not any(
        "prompt_toolkit" in source.read_text(encoding="utf-8")
        for source in (REPO_ROOT / "python/cti").rglob("*.py")
    )


def test_development_sources_are_outside_release_package_discovery() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools_config = metadata["tool"]["setuptools"]

    assert setuptools_config["package-dir"] == {"": "python"}
    assert setuptools_config["include-package-data"] is False
    assert setuptools_config["packages"]["find"] == {
        "where": ["python"],
        "include": ["cti", "cti.*"],
        "namespaces": False,
    }
    assert not setuptools_config.get("package-data")
    assert not (REPO_ROOT / "python/cti/kernel").exists()
    assert not any((REPO_ROOT / "python/cti/benchmark").rglob("*.py"))
    assert (REPO_ROOT / "benchmark").is_dir()
    assert (REPO_ROOT / "cookbook").is_dir()


def test_chat_recipe_uses_only_the_public_http_client() -> None:
    recipe = REPO_ROOT / "cookbook/chat.py"
    imported_modules = imports_from_source(
        recipe.read_text(encoding="utf-8"),
        package="cookbook",
    )

    assert not any(module == "cti" or module.startswith("cti.") for module in imported_modules)
    assert recipe.parent == REPO_ROOT / "cookbook"


@pytest.mark.parametrize("name", ("sft", "grpo"))
def test_training_recipes_use_only_public_cti_modules(name: str) -> None:
    recipe = REPO_ROOT / "cookbook" / f"{name}.py"
    imported_modules = imports_from_source(
        recipe.read_text(encoding="utf-8"),
        package="cookbook",
    )
    cti_modules = {
        module for module in imported_modules if module == "cti" or module.startswith("cti.")
    }

    public_modules = {
        "cti.config",
        "cti.llm",
        "cti.models",
        "cti.protocol",
        "cti.training",
    }
    assert all(
        any(module == public or module.startswith(f"{public}.") for public in public_modules)
        for module in cti_modules
    )


def test_built_distributions_exclude_development_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the configured PEP 517 backend without isolation or dependency resolution."""

    source_root = tmp_path / "source"
    source_root.mkdir()
    _copy_git_visible_source_tree(source_root)

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.chdir(source_root)
    wheel_name = build_meta.build_wheel(str(artifact_root))
    sdist_name = build_meta.build_sdist(str(artifact_root))

    with ZipFile(artifact_root / wheel_name) as wheel:
        wheel_members = tuple(wheel.namelist())
        metadata_name = next(name for name in wheel_members if name.endswith(".dist-info/METADATA"))
        package_metadata = Parser().parsestr(wheel.read(metadata_name).decode())
    with open_tar(artifact_root / sdist_name, mode="r:gz") as sdist:
        sdist_members = tuple(sdist.getnames())

    expected_package_sources = {
        path.relative_to(REPO_ROOT / "python").as_posix()
        for path in (REPO_ROOT / "python/cti").rglob("*")
        if path.is_file() and path.suffix in _PRODUCTION_SOURCE_SUFFIXES
    }
    assert expected_package_sources <= set(wheel_members)
    assert not any(member.startswith("ctikernels/") for member in wheel_members)
    for license_name in ("LICENSE", "THIRD_PARTY_NOTICES"):
        assert any(
            member.endswith(f".dist-info/licenses/{license_name}") for member in wheel_members
        )
        assert any(member.endswith(f"/{license_name}") for member in sdist_members)
    assert not any(map(_contains_forbidden_release_path, wheel_members))
    assert not any(map(_contains_forbidden_release_path, sdist_members))
    assert package_metadata["License-Expression"] == "MIT"
    requirements = package_metadata.get_all("Requires-Dist", failobj=[])
    assert not any(requirement.lower().startswith("prompt-toolkit") for requirement in requirements)
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    kernel_requirement = next(
        Requirement(value)
        for value in project["dependencies"]
        if Requirement(value).name == "cti-kernels"
    )
    assert kernel_requirement in [Requirement(value) for value in requirements]
    numerical_providers = (
        "flashinfer-python",
        "sglang-kernel",
        "liger-kernel",
        "apache-tvm-ffi",
        "quack-kernels",
    )
    assert not any(requirement.startswith(numerical_providers) for requirement in requirements)


def test_release_frontend_builds_wheel_from_sdist_not_workspace_build_tree(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _copy_git_visible_source_tree(source_root)

    stale_module = source_root / "build/lib/cti/stale_deleted_module.py"
    stale_module.parent.mkdir(parents=True)
    stale_module.write_text("raise RuntimeError('stale workspace module')\n", encoding="utf-8")

    artifact_root = tmp_path / "artifacts"
    subprocess.run(
        (
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(artifact_root),
        ),
        cwd=source_root,
        capture_output=True,
        text=True,
        check=True,
    )

    wheel_path = next(artifact_root.glob("*.whl"))
    with ZipFile(wheel_path) as wheel:
        wheel_members = tuple(wheel.namelist())

    assert "cti/stale_deleted_module.py" not in wheel_members
    assert not any(map(_contains_forbidden_release_path, wheel_members))
