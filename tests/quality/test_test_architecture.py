from __future__ import annotations

import ast
from pathlib import Path

import pytest
import tomllib

from tests.support.paths import REPO_ROOT

_OWNERSHIP_ROOTS = (
    "modules",
    "integration",
    "benchmarks",
    "quality",
    "support",
    "gpu",
    "hf_parity",
)
# Checks run by hand on hardware beyond the GPU tiers'; each one's README.md says when and how.
_BY_HAND_ROOTS = ("hf_parity",)


def _imported_modules(path: Path) -> tuple[str, ...]:
    modules: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parts = list(path.parent.relative_to(REPO_ROOT).parts) if node.level else []
            if node.level > 1:
                parts = parts[: 1 - node.level]
            if node.module:
                parts.extend(node.module.split("."))
            modules.append(".".join(parts))
            modules.extend(".".join([*parts, alias.name]) for alias in node.names)
    # Imported attributes are not modules unless a corresponding file exists.
    return tuple(
        module
        for module in modules
        if (REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")).is_file()
        or (REPO_ROOT.joinpath(*module.split(".")) / "__init__.py").is_file()
    )


def test_test_modules_do_not_import_other_test_modules() -> None:
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        for module in _imported_modules(path):
            if any(part.startswith("test_") for part in module.split(".")):
                violations.append(f"{path.relative_to(REPO_ROOT)} imports {module}")

    assert not violations, "cross-test imports:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    [
        "import tests.gpu.models.test_example",
        "import tests.gpu.models.test_example as gpu",
        "from tests.gpu.models.test_example import run",
        "from tests.gpu.models import test_example as gpu",
        "from . import test_example",
        "from .test_example import run",
        "from .gpu.models import test_example",
    ],
)
@pytest.mark.parametrize("module_file", ["test_example.py", "test_example/__init__.py"])
def test_cross_test_import_checker_rejects_test_modules(source, module_file, monkeypatch, tmp_path):
    tests = tmp_path / "tests"
    for parent in (tests, tests / "gpu" / "models"):
        target = parent / module_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def run(): pass\n")
    (tests / "test_caller.py").write_text(source)
    monkeypatch.setattr("tests.quality.test_test_architecture.REPO_ROOT", tmp_path)
    with pytest.raises(AssertionError, match="cross-test imports:"):
        test_test_modules_do_not_import_other_test_modules()


@pytest.mark.parametrize(
    "source",
    [
        "import tests.support.helpers",
        "from tests.support import helpers",
        "from tests.support.helpers import test_data",
        "from tests.support.helpers import test_function",
        "from . import helpers",
        "from .helpers import test_data",
        "from ..support.helpers import test_function",
    ],
)
def test_cross_test_import_checker_accepts_support(source, monkeypatch, tmp_path):
    tests = tmp_path / "tests"
    support = tests / "support"
    support.mkdir(parents=True)
    (support / "helpers.py").write_text("test_data = [1, 2, 3]\ndef test_function(): pass\n")
    (support / "__init__.py").write_text(source)
    monkeypatch.setattr("tests.quality.test_test_architecture.REPO_ROOT", tmp_path)
    test_test_modules_do_not_import_other_test_modules()


def test_test_tree_uses_behavioral_ownership_roots() -> None:
    tests_root = REPO_ROOT / "tests"
    assert all((tests_root / name).is_dir() for name in _OWNERSHIP_ROOTS)
    for legacy_name in ("core", "misc", "kernel"):
        assert not tuple((tests_root / legacy_name).glob("*.py"))


def test_by_hand_checks_are_not_collected_by_default() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    collected = pyproject["tool"]["pytest"]["ini_options"]["testpaths"]
    for name in _BY_HAND_ROOTS:
        root = Path("tests", name)
        assert (REPO_ROOT / root / "README.md").is_file(), root
        assert not [path for path in collected if root.is_relative_to(path)], (root, collected)
