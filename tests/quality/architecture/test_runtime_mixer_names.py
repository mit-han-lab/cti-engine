"""Runtime files name no mixer family, in code, comments or docstrings.

They see recurrent layers only through ``cti.recurrent``, the recurrent-state
pool in ``cti.kvcache`` and the model hooks.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.support.paths import PYTHON_ROOT, REPO_ROOT

_MIXER_NAME = re.compile(r"gated.?delta|gdn", re.IGNORECASE)
_RUNTIME_PATHS = (
    "forward.py",
    "plan.py",
    "resolution.py",
    "memory.py",
    "worker",
    "kvcache",
    "inference",
    "training",
    "recurrent.py",
    "models/base.py",
    "layers/__init__.py",
)


def _runtime_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for name in _RUNTIME_PATHS:
        path = PYTHON_ROOT / "cti" / name
        assert path.exists(), path
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else (path,))
    return tuple(files)


def test_mixer_name_pattern() -> None:
    for name in ("GatedDeltaNet", "gated_delta_net", "Gated DeltaNet", "gdn_backend", "GDN"):
        assert _MIXER_NAME.search(name), name
    assert not _MIXER_NAME.search("recurrent delta-rule state")


def test_runtime_files_name_no_mixer_family() -> None:
    violations = [
        f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}"
        for path in _runtime_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _MIXER_NAME.search(line)
    ]

    assert not violations, "mixer names in runtime files:\n" + "\n".join(violations)
