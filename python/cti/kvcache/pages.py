"""The device page table, each request slot's page IDs, and the page size choice."""

from __future__ import annotations

from collections.abc import Sequence
from math import lcm
from typing import TYPE_CHECKING, Protocol

import torch

from cti.utils import align_ceil

if TYPE_CHECKING:
    from .state import StateCapacity


class PageTable:
    """Each row's page IDs, ``[rows, capacity.table_width]`` int32, rows past the request
    slots padding graph batches: the one device page table, which the runner alone writes.
    An unmapped entry names the sink page, which no request owns, never a usable page.

    With ``token_view``, ``tokens`` also holds each mapped position's flat token index
    (page ID x page size + offset), ``capacity.max_seq_len`` positions at least, for
    readers that address tokens; at page size 1 it is ``pages`` itself."""

    def __init__(
        self,
        capacity: StateCapacity,
        rows: int,
        *,
        device: torch.device,
        token_view: bool = False,
    ) -> None:
        self.page_size = page_size = capacity.page_size
        self.pages = torch.full(
            (rows, capacity.table_width), capacity.sink_slot, dtype=torch.int32, device=device
        )
        self.tokens: torch.Tensor | None = None
        if token_view:
            self.tokens = self.pages
            if page_size > 1:
                self.tokens = torch.full(
                    (rows, align_ceil(capacity.max_seq_len, lcm(32, page_size))),
                    capacity.sink_slot * page_size,
                    dtype=torch.int32,
                    device=device,
                )
                self._offsets = torch.arange(page_size, dtype=torch.int32, device=device)
                self._positions = self._offsets.long()

    def apply(self, entries: torch.Tensor) -> None:
        """Copy host int32 ``(row, page index, page ID)`` triples, ``[n, 3]``, into the table
        on the current stream; ``entries`` must live until the copy completes."""

        device_entries = entries.to(self.pages.device, non_blocking=True)
        rows, columns = device_entries[:, 0].long(), device_entries[:, 1].long()
        page_ids = device_entries[:, 2]
        self.pages[rows, columns] = page_ids
        if self.tokens is None or self.tokens is self.pages:
            return
        page_size = self.page_size
        positions = columns.unsqueeze(1) * page_size + self._positions
        self.tokens[rows.unsqueeze(1), positions] = (
            page_ids.unsqueeze(1) * page_size + self._offsets
        )


class PageRule(Protocol):
    """The page sizes one attention backend takes (``attention.PageConstraint``): those in
    ``allowed`` (None: any), multiples of ``multiple``."""

    @property
    def allowed(self) -> frozenset[int] | None: ...
    @property
    def multiple(self) -> int: ...


def choose_page_size(
    requested: int | None, constraints: Sequence[PageRule], ratios: Sequence[int] = ()
) -> int:
    """The page size every constraint takes and every ratio (tokens per cache entry)
    divides: ``requested`` when given, which must be one, else the smallest."""

    multiple = lcm(*(rule.multiple for rule in constraints), *ratios)
    allowed: frozenset[int] | None = None
    for rule in constraints:
        if rule.allowed is not None:
            allowed = rule.allowed if allowed is None else allowed & rule.allowed
    sizes = None if allowed is None else sorted(p for p in allowed if p % multiple == 0)
    if requested is None:
        if sizes is None:
            return multiple
        if not sizes:
            raise ValueError(f"no page size suits the attention backends: {list(constraints)}")
        return sizes[0]
    if requested <= 0 or requested % multiple or (sizes is not None and requested not in sizes):
        suits = f"multiples of {multiple}" if sizes is None else str(sizes)
        raise ValueError(f"page size {requested} does not suit the attention backends ({suits})")
    return requested
