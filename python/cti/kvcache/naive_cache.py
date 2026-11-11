# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/kvcache/naive_cache.py`); see THIRD_PARTY_NOTICES.

from dataclasses import dataclass

import torch

from .base import (
    BaseCacheHandle,
    BasePrefixCache,
    InsertResult,
    RestorePlan,
    SizeInfo,
)


@dataclass(frozen=True)
class NaiveCacheHandle(BaseCacheHandle):
    empty_tensor: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.handle = NaiveCacheHandle(cached_len=0, empty_tensor=self.empty_tensor)
        super().__init__()

    def plan_restore(self, prompt: torch.Tensor) -> RestorePlan:
        return RestorePlan(0)

    def acquire(self, plan: RestorePlan) -> BaseCacheHandle:
        return self.handle

    def lock(self, handle: BaseCacheHandle) -> None:
        pass

    def unlock(self, handle: BaseCacheHandle) -> None:
        pass

    def publish(
        self, ids: torch.Tensor, pages: torch.Tensor, row: int | None = None
    ) -> InsertResult:
        return InsertResult(0, self.handle)

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)
