"""Layers and models never address the KV pool, which attention backends write and read."""

from __future__ import annotations

import ast

import pytest

from tests.support.paths import PYTHON_ROOT

_KV_POOL_NAMES = frozenset({"PageGroup", "kv_pages", "store_kv_pair", "cache_write_indices"})


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.arg | ast.keyword):
        return node.arg
    if isinstance(node, ast.alias):
        return node.name.rpartition(".")[2]
    return None


@pytest.mark.parametrize("package", ["layers", "models"])
def test_package_never_names_the_kv_pool_or_its_write_indices(package: str) -> None:
    found = [
        f"{path.relative_to(PYTHON_ROOT)}:{node.lineno}: {name}"
        for path in sorted((PYTHON_ROOT / "cti" / package).rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if (name := _name(node)) in _KV_POOL_NAMES
    ]

    assert not found, "KV pool outside the attention backends:\n" + "\n".join(found)
