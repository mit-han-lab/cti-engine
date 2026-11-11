from __future__ import annotations

from tests.support.markdown import unresolved_links
from tests.support.paths import REPO_ROOT

_DOCS_ROOT = REPO_ROOT / "docs"
_ACTIVE_DOCUMENTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "benchmark/README.md",
    REPO_ROOT / "cookbook/README.md",
    REPO_ROOT / "docker/README.md",
    *sorted(_DOCS_ROOT.rglob("*.md")),
)


def test_active_document_links_resolve() -> None:
    unresolved = [
        f"{document.relative_to(REPO_ROOT)} -> {destination}"
        for document in _ACTIVE_DOCUMENTS
        for destination in unresolved_links(document)
    ]

    assert not unresolved, "unresolved active-document links:\n" + "\n".join(unresolved)
