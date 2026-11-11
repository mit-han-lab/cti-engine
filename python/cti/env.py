# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/env.py`); see THIRD_PARTY_NOTICES.

"""The engine's environment switches, read once when the rank entry imports this module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _flag(name: str) -> bool:
    """Whether ``CTI_<name>`` is set to 1, true or yes."""

    return os.environ.get(f"CTI_{name}", "").lower() in ("1", "true", "yes")


@dataclass(frozen=True, slots=True)
class Env:
    # Run the scheduler without overlapping CPU planning and GPU execution.
    disable_overlap_scheduling: bool
    # Internal and expensive development checks, off unless CTI_CHECKS=1; tests turn
    # them on.
    checks: bool
    # The secret every node of a multi-node engine shares (CTI_DIST_AUTHKEY): its links
    # carry pickles, so a node accepts only peers that hold it. Never printed.
    dist_authkey: bytes | None = field(default=None, repr=False)


ENV = Env(
    disable_overlap_scheduling=_flag("DISABLE_OVERLAP_SCHEDULING"),
    checks=_flag("CHECKS"),
    dist_authkey=os.environ.get("CTI_DIST_AUTHKEY", "").encode() or None,
)
