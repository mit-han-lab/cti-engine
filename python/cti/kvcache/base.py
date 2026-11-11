# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/kvcache/base.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from .radix_cache import StateRowIndex


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """A prefix-provider handle whose nonzero frontier covers complete cache pages."""

    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor:
        """Return one page ID per cached page on the table device and dtype."""


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int


class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix


class RestorePlan(NamedTuple):
    """A prompt's hit: ``cached_len`` page-aligned tokens, short of the final prompt token,
    restoring snapshot ``row`` there when the cache holds rows. ``kv_match_len`` is the
    deepest page-aligned KV match, where the prompt leaves the cache; acquiring the hit
    moves its ``unprotected_len`` tokens out of the evictable size. ``path`` is the
    cache's own, valid until the cache next changes."""

    cached_len: int
    kv_match_len: int = 0
    row: int | None = None
    unprotected_len: int = 0
    path: object = None


class BasePrefixCache(ABC):
    rows: StateRowIndex | None = None  # the recurrent-state snapshot rows, if the cache has any

    def hit_limit(self, prompt_len: int) -> int:
        """The most of a prompt of ``prompt_len`` tokens a hit restores: none without reuse."""

        return 0

    def restorable(self, end: int, exports: tuple[int, ...]) -> tuple[int, ...]:
        """The prefix lengths a settled prompt forward up to ``end``, exporting recurrent
        state at ``exports``, makes restorable: none without reuse."""

        return ()

    def restored_len(self, prompt: torch.Tensor, leader: torch.Tensor, position: int) -> int:
        """The tokens of ``prompt`` a hit would restore once ``leader``'s prefix of
        ``position`` tokens is published: none without reuse."""

        return 0

    @abstractmethod
    def plan_restore(self, prompt: torch.Tensor) -> RestorePlan:
        """The hit on ``prompt`` if it were admitted now; acquire it before the cache
        changes."""

    @abstractmethod
    def acquire(self, plan: RestorePlan) -> BaseCacheHandle:
        """Protect ``plan``'s prefix, with the row it restores, and return its handle."""

    @abstractmethod
    def lock(self, handle: BaseCacheHandle) -> None:
        """Protect a handle's prefix from eviction before using its indices."""

    @abstractmethod
    def unlock(self, handle: BaseCacheHandle) -> None:
        """Release a handle's prefix protection."""

    @abstractmethod
    def publish(
        self, ids: torch.Tensor, pages: torch.Tensor, row: int | None = None
    ) -> InsertResult:
        """Publish the KV of ``ids`` stored in ``pages`` (a page ID per whole page) and, with ``row``, a leased
        snapshot row holding the recurrent state after them."""

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """Free at least ``size`` evictable tokens (0 is a no-op); return their page IDs."""

    @abstractmethod
    def reset(self) -> None:
        """Invalidate this provider's prefix visibility after handles are unprotected.

        This does not scrub physical KV storage or request-table mappings.
        """

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""

    def check_integrity(self) -> None:  # noqa: B027 - optional hook
        """Raise if the cache is corrupted."""

    def warm_up(self) -> None:  # noqa: B027 - optional hook
        """Load what matching needs before the first request, so that none builds it."""
