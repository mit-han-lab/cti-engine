"""Stable repository paths shared by test modules."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SOURCE_ROOTS = (PYTHON_ROOT,)


def source_pythonpath(*additional_roots: Path) -> str:
    return os.pathsep.join(str(path) for path in (*SOURCE_ROOTS, *additional_roots))
