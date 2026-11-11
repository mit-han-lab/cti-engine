# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/engine/engine.py`); see THIRD_PARTY_NOTICES.

"""Capacity planning: the pages and snapshot rows of an inference state that a measured
budget holds."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from numbers import Real
from typing import TYPE_CHECKING, ClassVar

from cti.utils import init_logger, mem_GB

from .state import PAYLOAD

if TYPE_CHECKING:
    from .state import AllocatorModel, KVLayout, StateCapacity, TableGeometry

logger = init_logger(__name__)

# The bytes of each device allocation beside the pools, by request slots and snapshot rows.
Allocations = Callable[[int, int], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class SnapshotPolicy:
    """Snapshot rows that prefix reuse keeps of recurrent state: ``rows``, or a ``ratio``
    of the recurrent share of the budget (by default ``MEMORY_RATIO``)."""

    MEMORY_RATIO: ClassVar[float] = 0.25

    rows: int | None = None
    ratio: float | None = None

    def __post_init__(self) -> None:
        rows, ratio = self.rows, self.ratio
        if rows is not None and ratio is not None:
            raise ValueError("set only one of snapshot_rows and snapshot_memory_ratio")
        if rows is not None and (type(rows) is not int or rows < 2):
            raise ValueError("snapshot_rows must be an integer >= 2")
        if ratio is not None and (
            isinstance(ratio, bool)
            or not isinstance(ratio, Real)
            or not math.isfinite(ratio)
            or not 0 < ratio < 1
        ):
            raise ValueError("snapshot_memory_ratio must be in (0, 1)")


class CapacityError(ValueError):
    """The budget holds no inference state; ``beside`` when it would without what runs
    beside inference."""

    def __init__(self, reason: str, *, beside: bool) -> None:
        super().__init__(reason)
        self.beside = beside


@dataclass(frozen=True, slots=True)
class StateBudget:
    """Device bytes the inference state may take, ``available``, less ``beside``, what runs
    beside inference keeps, ``image_encoder``, the image activations the peak reserve cannot
    hold, and ``image_rows``, the encoder-output rows' bytes. ``backing`` sizes every pool,
    and the graph sink page, by the allocator's backing; ``None`` sizes them by payload.
    Recurrent rows, live and snapshot, take the share ``recurrent_to_kv_ratio`` gives them:
    ratio / (1 + ratio) of what is left. With ``fit_slots``, the geometry's request slots
    only bound the slots: the plan takes as many as the share holds past its snapshot rows."""

    available: int
    backing: AllocatorModel | None
    recurrent_to_kv_ratio: float
    beside: int = 0
    image_encoder: int = 0
    image_rows: int = 0
    fit_slots: bool = False


def plan_capacity(
    layout: KVLayout,
    geometry: TableGeometry,
    budget: StateBudget,
    *,
    snapshots: SnapshotPolicy | None,
    allocations: Allocations | None = None,
) -> StateCapacity:
    """The capacity ``budget`` holds, without changing any state; ``snapshots`` is ``None``
    when no prefix reuse restores recurrent state. ``allocations`` maps the request slots and
    snapshot rows to the bytes of each device allocation beside the pools (speculative
    rounds' slot rows and recurrent blocks). ``geometry``'s reserved pages come off the
    usable pages. ``CapacityError`` says why none fits."""

    plan = partial(_plan, layout, geometry, snapshots=snapshots, allocations=allocations)
    planned = plan(budget)
    if isinstance(planned, str):
        beside = bool(budget.beside) and not isinstance(plan(replace(budget, beside=0)), str)
        raise CapacityError(planned, beside=beside)
    if budget.fit_slots:
        rows = layout.rows
        logger.info_rank0(
            "max_running_requests=%d (auto: recurrent state %s/request and %d snapshot rows "
            "in %s of the %s budget after weights, recurrent_to_kv_ratio %s; cap %d)",
            planned.num_slots,
            _mem_MiB(rows.row_bytes),
            len(planned.snapshot_rows),
            mem_GB(rows.row_bytes * (planned.num_slots + len(planned.snapshot_rows))),
            mem_GB(budget.available),
            budget.recurrent_to_kv_ratio,
            geometry.num_slots,
        )
    return planned


def _plan(
    layout: KVLayout,
    geometry: TableGeometry,
    budget: StateBudget,
    *,
    snapshots: SnapshotPolicy | None,
    allocations: Allocations | None,
) -> StateCapacity | str:
    """The capacity ``budget`` holds, or why it holds none."""

    rows, page_size = layout.rows, geometry.page_size
    # Without an allocator model, every allocation backs its payload.
    backing = PAYLOAD if budget.backing is None else budget.backing
    available = (
        budget.available - budget.beside - budget.image_encoder - backing.segment(budget.image_rows)
    )
    num_slots, num_snapshot_rows = geometry.num_slots, 0
    if rows.specs:
        ratio = budget.recurrent_to_kv_ratio
        # SGLang 0.5.19 splits the same way by mamba_full_memory_ratio, default 0.9
        # (srt/mem_cache/kv_cache_configurator.py: the mamba budget is
        # total_rest_memory * r / (1 + r)); its one state pool holds running and cached
        # states, as the share here holds live and snapshot rows.
        share = int(max(available, 0) * ratio / (1 + ratio))
        if snapshots is not None:
            num_snapshot_rows = rows.snapshot_count(share, snapshots)
            if num_snapshot_rows < 2:
                return (
                    f"{num_snapshot_rows} snapshot rows of {_mem_MiB(rows.row_bytes)} fit the "
                    f"{mem_GB(share)} recurrent share; prefix reuse needs at least 2: raise "
                    "memory.inference_ratio or memory.recurrent_to_kv_ratio"
                )
        if budget.fit_slots:
            fitted = (share - num_snapshot_rows * rows.row_bytes) // rows.row_bytes
            num_slots = min(num_slots, fitted)
            if num_slots < 1:
                return (
                    "Not enough memory for one running request: the recurrent share "
                    f"({mem_GB(share)} at recurrent_to_kv_ratio {ratio}) holds "
                    f"{num_snapshot_rows} snapshot rows and no request's recurrent state "
                    f"({_mem_MiB(rows.row_bytes)}); raise memory.inference_ratio or "
                    "memory.recurrent_to_kv_ratio"
                )
    state = rows.allocation(num_slots, num_snapshot_rows, backing)
    if rows.specs and state.backing_bytes >= available:
        return (
            f"Recurrent state does not fit the inference memory budget: "
            f"max_running_requests={num_slots} needs {num_slots} x "
            f"{_mem_MiB(rows.row_bytes)} and {num_snapshot_rows} snapshot rows = "
            f"{mem_GB(state.backing_bytes)}, but the budget left after weights and workspaces "
            f"is {mem_GB(max(available, 0))}; lower scheduling.max_running_requests (unset, "
            "it is sized to the budget) or raise memory.inference_ratio"
        )
    beside = () if allocations is None else allocations(num_slots, num_snapshot_rows)
    kv_budget = available - state.backing_bytes - sum(map(backing.segment, beside))
    # Every pool at its allocation size, and the graph sink page.
    num_pages = layout.fit(kv_budget, page_size, backing) - 1 - geometry.guard_pages
    if num_pages <= 1:
        return (
            "Not enough memory for KV cache: at least 2 pages are required, but the "
            f"inference memory budget allows {num_pages}; reduce the inference memory load "
            "(scheduling.max_running_requests, snapshot_rows or snapshot_memory_ratio) or "
            "raise memory.inference_ratio"
        )
    return replace(geometry, num_slots=num_slots).capacity(layout, num_pages, num_snapshot_rows)


def _mem_MiB(size: int) -> str:
    return f"{size / (1 << 20):.1f} MiB"
