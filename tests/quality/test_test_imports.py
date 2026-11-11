"""Every ``cti`` or ``tests`` name a test imports resolves, including imports inside
functions.

GPU tests import the engine inside their child processes, where collection never reaches;
this catches a renamed or moved name, in the engine or in the test support, on CPU instead of on
the first GPU run.
"""

from __future__ import annotations

import ast
import importlib

from tests.support.paths import REPO_ROOT

_ROOTS = ("cti", "tests")


def _checked_imports() -> set[tuple[str, str, str | None]]:
    imports = set()
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        where = str(path.relative_to(REPO_ROOT))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=where)):
            if isinstance(node, ast.Import):
                imports.update(
                    (where, alias.name, None)
                    for alias in node.names
                    if alias.name.split(".")[0] in _ROOTS
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module.split(".")[0] in _ROOTS
            ):
                imports.update((where, node.module, alias.name) for alias in node.names)
    return imports


def _resolves(module_name: str, name: str | None) -> bool:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False
    if name is None or name == "*" or hasattr(module, name):
        return True
    try:
        importlib.import_module(f"{module_name}.{name}")
    except ImportError:
        return False
    return True


def test_every_engine_and_support_import_in_the_tests_resolves() -> None:
    missing = sorted(
        f"{where}: {module}" + (f".{name}" if name else "")
        for where, module, name in _checked_imports()
        if not _resolves(module, name)
    )
    assert not missing, "\n".join(missing)
