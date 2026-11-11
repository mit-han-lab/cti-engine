"""The capture report an engine logs once per graph group it captures
(``cti.inference._execution._graphs.capture_graphs``): "Captured <kind> graphs of sizes
[...]: ...", one line per decode, breakable or round group."""

from __future__ import annotations

from collections.abc import Sequence


def capture_line(kind: str, sizes: Sequence[int] | None = None) -> str:
    """The report's prefix for a group of ``kind``, of exactly ``sizes`` when given."""

    prefix = f"Captured {kind} graphs of sizes"
    return prefix if sizes is None else f"{prefix} {list(sizes)}:"


def graph_captures(log: str, kind: str, sizes: Sequence[int] | None = None) -> int:
    """How many groups of ``kind`` (of exactly ``sizes`` when given) ``log`` reports captured."""

    return log.count(capture_line(kind, sizes))


__all__ = ["capture_line", "graph_captures"]
