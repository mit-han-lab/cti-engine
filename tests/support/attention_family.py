"""A second attention family for the registry's tests: slots of any tokens per entry in
byte rows of their requirement's ``row_bytes``, on pages of 256 tokens, served by one
backend that records the KV groups it was built over."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from cti.attention.base import PageConstraint

if TYPE_CHECKING:
    from cti.attention import AttentionRequest, AttentionRequirement
    from cti.kvcache import GroupRole, PageGroup, PageTable, SlotKey
    from cti.utils import DeviceCapability


@dataclass(frozen=True)
class FakeRows:
    """A row layout of ``row_bytes`` per entry."""

    row_bytes: int

    def page_bytes(self, entries: int) -> int:
        return entries * self.row_bytes

    def allocate(
        self,
        num_pages: int,
        entries: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.zeros(num_pages, self.page_bytes(entries), dtype=torch.uint8, device=device)


@dataclass(frozen=True)
class FakeBackend:
    """The backend built over the target's ``groups`` on ``page_table``."""

    groups: tuple[PageGroup, ...]
    page_table: PageTable


@dataclass(frozen=True)
class ResolvedFake:
    """The fake family resolved: the ``requirements`` and ``request`` it received."""

    requirements: Mapping[GroupRole, Sequence[AttentionRequirement]]
    request: AttentionRequest
    selected: str = "fake"
    reason: str = "the test family"
    pages: tuple[PageConstraint, ...] = (PageConstraint(frozenset({256})),)

    @property
    def layouts(self) -> Mapping[SlotKey, FakeRows]:
        return {
            (role, r.slot): FakeRows(dict(r.geometry).get("row_bytes", 1))
            for role, role_requirements in self.requirements.items()
            for r in role_requirements
        }

    def workspace_bytes(self, *, device: torch.device, max_running_req: int) -> int:
        return 0

    def build(
        self,
        groups: Sequence[PageGroup],
        page_table: PageTable,
        *,
        max_running_req: int,
        capability: DeviceCapability | None,
    ) -> FakeBackend:
        return FakeBackend(tuple(groups), page_table)


def resolve(
    requirements: Mapping[GroupRole, Sequence[AttentionRequirement]], request: AttentionRequest
) -> ResolvedFake:
    return ResolvedFake(requirements, request)


__all__ = ["FakeBackend", "FakeRows", "ResolvedFake", "resolve"]
