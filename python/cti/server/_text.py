"""Text rules the API server's renderers share."""

from __future__ import annotations

from collections.abc import Sequence


def held_tail(text: str, needles: Sequence[str]) -> int:
    """The length of the longest tail of ``text`` that may begin one of ``needles``: text a
    renderer holds back until it knows whether it does."""

    for length in range(min(len(text), max(map(len, needles), default=0) - 1), 0, -1):
        if any(needle.startswith(text[-length:]) for needle in needles):
            return length
    return 0


__all__ = ["held_tail"]
