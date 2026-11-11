"""Attention families by the name a model's attention requirements give: each resolves the
requirements into what the engine plans and builds with, loaded on first use."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from cti.forward import RoundShape
    from cti.kvcache import GroupRole, PageGroup, PageTable, RowLayout, SlotKey
    from cti.utils import DeviceCapability

    from .base import BaseAttnBackend, PageConstraint
    from .requirement import AttentionRequirement

# Each family's module and resolver, by family name.
FAMILIES: Mapping[str, tuple[str, str]] = MappingProxyType({"mha": (".mha", "resolve")})


@dataclass(frozen=True, kw_only=True)
class AttentionRequest:
    """What the engine asks of the family: the ``target`` role, whose layers the built
    backend serves, the ``backend`` requested by name ("auto" lets the family choose), an
    explicit ``page_size`` (None: the family's choice), the device, the ranks that split
    the heads, the fixed-query ``rounds`` beside it, the model's ``dtype`` and a
    tensor-core override (None: the backend's choice)."""

    target: GroupRole
    backend: str
    page_size: int | None
    capability: DeviceCapability | None
    tensor_parallel_size: int
    dtype: torch.dtype
    rounds: RoundShape | None = None
    tensor_cores: bool | None = None


class ResolvedFamily(Protocol):
    """An attention family resolved for one engine: what it selected and why, the page
    sizes it takes, each paged slot's row layout on this rank, and how it sizes and builds
    the target's backend."""

    @property
    def selected(self) -> str: ...

    @property
    def reason(self) -> str: ...

    @property
    def pages(self) -> tuple[PageConstraint, ...]: ...

    @property
    def layouts(self) -> Mapping[SlotKey, RowLayout]: ...

    def workspace_bytes(self, *, device: torch.device, max_running_req: int) -> int:
        """The device scratch ``build`` allocates."""
        ...

    def build(
        self,
        groups: Sequence[PageGroup],
        page_table: PageTable,
        *,
        max_running_req: int,
        capability: DeviceCapability | None,
    ) -> BaseAttnBackend:
        """The one backend serving the target's KV ``groups`` on ``page_table``."""
        ...


def resolve_attention_config(
    requirements: Mapping[GroupRole, Sequence[AttentionRequirement]], request: AttentionRequest
) -> ResolvedFamily:
    """Resolve the one family every role's requirements name, the model's and its
    speculative method's."""

    families = list(dict.fromkeys(r.family for role in requirements.values() for r in role))
    if not families:
        raise ValueError("attention resolves for at least one requirement")
    if len(families) > 1:
        raise ValueError(f"one attention family serves an engine, got {', '.join(families)}")
    if families[0] not in FAMILIES:
        raise ValueError(f"no attention family {families[0]!r}")
    if request.page_size is not None and request.page_size <= 0:
        raise ValueError(f"page_size must be positive, got {request.page_size}")
    module, resolver = FAMILIES[families[0]]
    return getattr(importlib.import_module(module, package=__package__), resolver)(
        requirements, request
    )


__all__ = ["FAMILIES", "AttentionRequest", "ResolvedFamily", "resolve_attention_config"]
