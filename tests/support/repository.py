"""The repository's Git-visible entries, read the way Git stores them."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tests.support.paths import REPO_ROOT


def git_visible_paths() -> list[Path]:
    """Tracked and untracked, non-ignored entries of the worktree."""

    completed = subprocess.run(
        ("git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"),
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    return [REPO_ROOT / os.fsdecode(entry) for entry in completed.stdout.split(b"\0") if entry]


def read_repository_entry(path: Path) -> bytes | None:
    """Bytes that Git stores for a repository entry, without following links.

    A symlink contributes its link target string (a broken link included); a
    regular file contributes its content; anything else contributes nothing.
    Nothing outside the repository is ever read.
    """
    if path.is_symlink():
        return os.fsencode(os.readlink(path))
    if path.is_file():
        return path.read_bytes()
    return None
