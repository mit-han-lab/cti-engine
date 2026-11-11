"""Relative links of the repository's Markdown documents."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+]\(([^)]+)\)")


def unresolved_links(document: Path) -> list[str]:
    """The link destinations of ``document`` that name a missing file; anchors and URLs are
    not followed."""

    unresolved: list[str] = []
    for destination in _MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
        if destination.startswith(("#", "https://", "http://", "mailto:")):
            continue
        relative_path = unquote(destination.split("#", 1)[0])
        if not (document.parent / relative_path).exists():
            unresolved.append(destination)
    return unresolved
