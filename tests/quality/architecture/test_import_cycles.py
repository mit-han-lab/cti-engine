"""Runtime imports among cti subpackages and root modules form no cycle.

Function-level imports count; ``if TYPE_CHECKING:`` imports do not.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tests.support.imports import runtime_imports_from_source
from tests.support.paths import PYTHON_ROOT, REPO_ROOT

Edges = dict[tuple[str, str], set[str]]


def _runtime_edges(root: Path) -> Edges:
    """``(source, target)`` units -> ``path:line``; a unit is a subpackage or root module."""

    edges: Edges = defaultdict(set)
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent)
        parts = relative.with_suffix("").parts
        if len(parts) < 2 or parts[1] == "__init__":
            continue
        package = ".".join(relative.parent.parts)
        source = path.read_text(encoding="utf-8")
        for module, line in runtime_imports_from_source(source, package=package):
            target = module.split(".")
            if len(target) > 1 and target[0] == root.name and target[1] != parts[1]:
                edges[(parts[1], target[1])].add(f"{path.relative_to(REPO_ROOT)}:{line}")
    return edges


def _cycles(edges: Edges) -> list[list[str]]:
    """Units of each strongly connected component larger than one unit."""

    graph: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        graph[source].add(target)

    def reachable(start: str) -> set[str]:
        seen: set[str] = set()
        pending = [start]
        while pending:
            for unit in graph[pending.pop()] - seen:
                seen.add(unit)
                pending.append(unit)
        return seen

    reach = {unit: reachable(unit) for unit in tuple(graph)}
    components = {
        frozenset(other for other in reach[unit] if unit in reach.get(other, ()))
        for unit in reach
        if unit in reach[unit]
    }
    return sorted(sorted(component) for component in components)


def test_cycle_finder_reports_each_component_once() -> None:
    edges: Edges = {("a", "b"): set(), ("b", "a"): set(), ("b", "c"): set(), ("c", "d"): set()}

    assert _cycles(edges) == [["a", "b"]]


def test_runtime_import_graph_is_acyclic() -> None:
    edges = _runtime_edges(PYTHON_ROOT / "cti")
    cycles = _cycles(edges)
    sites = [
        f"{source} -> {target}: {', '.join(sorted(edges[(source, target)]))}"
        for cycle in cycles
        for source, target in sorted(edges)
        if source in cycle and target in cycle
    ]

    assert not cycles, f"runtime import cycles {cycles}:\n" + "\n".join(sites)
