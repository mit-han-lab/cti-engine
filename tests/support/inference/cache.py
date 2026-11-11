"""``CacheManager`` inputs and steps shared by the cache tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from typing import Literal

import torch

from cti.inference._control._cache import CacheManager, CacheUse
from cti.kvcache import BasePrefixCache, StateCapacity, create_prefix_cache
from cti.protocol import SamplingParams


@dataclass(frozen=True)
class RequestInput:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_new_tokens


def state_capacity(
    num_pages: int,
    page_size: int,
    num_slots: int,
    width: int,
    *,
    snapshot_rows: range = range(0),
) -> StateCapacity:
    """``num_pages`` pages for ``num_slots`` request slots of ``width`` page columns, with
    ``snapshot_rows`` on the snapshot grid of a Gated DeltaNet model."""

    return StateCapacity(
        page_size,
        num_slots,
        width * page_size,
        width,
        num_pages,
        snapshot_rows,
        num_pages,
        range(num_pages + 1, num_pages + 1),
        lcm(page_size, 64),
        64,
    )


def prefix_cache(
    page_size: int = 1,
    *,
    snapshot_rows: range = range(0),
    kind: Literal["naive", "radix"] = "radix",
) -> BasePrefixCache:
    """A CPU prefix cache of ``kind``, a radix one holding ``snapshot_rows``."""

    capacity = state_capacity(1, page_size, 1, 1, snapshot_rows=snapshot_rows)
    return create_prefix_cache(capacity, kind=kind, device=torch.device("cpu"))


def new_cache_manager(
    num_pages: int,
    page_size: int,
    page_table: torch.Tensor,
    *,
    reuse: bool,
    checks: bool,
    snapshot_rows: range = range(0),
) -> CacheManager:
    """A ``CacheManager`` of ``num_pages`` pages for ``page_table``'s request slots and page
    columns (only its shape is read), with ``snapshot_rows`` on the snapshot grid of a Gated
    DeltaNet model."""

    num_slots, width = page_table.shape
    capacity = state_capacity(num_pages, page_size, num_slots, width, snapshot_rows=snapshot_rows)
    return CacheManager(capacity, prefix="radix" if reuse else "naive", checks=checks)


def commit_reservation(
    cache: CacheManager, use: CacheUse, *, prefix_ids: torch.Tensor | None = None
) -> None:
    """Settle a forward's ``use`` as committed, publishing ``prefix_ids`` when given."""

    cache.settle(use, use.end - use.start, prefix_ids=prefix_ids)
