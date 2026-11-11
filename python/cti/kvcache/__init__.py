# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/kvcache/__init__.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

from .base import BaseCacheHandle, BasePrefixCache, RestorePlan
from .budget import CapacityError, SnapshotPolicy, StateBudget, plan_capacity
from .naive_cache import NaivePrefixCache
from .pages import PageTable, choose_page_size
from .radix_cache import StateCoordinator, StateRowIndex
from .recurrent_pool import (
    RecurrentStatePool,
    SlotStateSpec,
    next_snapshot_position,
)
from .state import (
    BoundSlot,
    GroupLayout,
    GroupRole,
    InferenceState,
    KVLayout,
    PagedSlot,
    PageGroup,
    RecurrentRows,
    ResolvedSlot,
    RowLayout,
    SlotKey,
    Spec,
    StateCapacity,
    StateData,
    StateLayer,
    StateSlot,
    TableGeometry,
    group_layers,
)


def create_prefix_cache(
    capacity: StateCapacity, *, kind: Literal["naive", "radix"], device: torch.device
) -> BasePrefixCache:
    """A radix ``StateCoordinator`` over ``capacity``'s pages (and its snapshot rows for
    recurrent state), else a naive cache that reuses nothing."""

    if kind == "naive":
        return NaivePrefixCache(device)
    snapshot_rows = capacity.snapshot_rows
    rows = StateRowIndex(snapshot_rows, capacity.snapshot_grid) if snapshot_rows else None
    return StateCoordinator(device, capacity.page_size, rows)


__all__ = [
    "CapacityError",
    "choose_page_size",
    "create_prefix_cache",
    "BaseCacheHandle",
    "RestorePlan",
    "StateRowIndex",
    "BoundSlot",
    "GroupLayout",
    "GroupRole",
    "InferenceState",
    "KVLayout",
    "PagedSlot",
    "PageGroup",
    "PageTable",
    "RecurrentRows",
    "RecurrentStatePool",
    "ResolvedSlot",
    "RowLayout",
    "SlotKey",
    "SlotStateSpec",
    "Spec",
    "StateCapacity",
    "StateData",
    "StateLayer",
    "StateSlot",
    "TableGeometry",
    "group_layers",
    "next_snapshot_position",
    "SnapshotPolicy",
    "StateBudget",
    "plan_capacity",
]
