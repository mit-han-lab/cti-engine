# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/attention/fa.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ctikernels.attention.inference import (
    check_flash_attention_available,
    flash_attention_with_kv_cache,
)

from .paged import PageTableBackend, PageTableMetadata, store_kv_pair

if TYPE_CHECKING:
    from cti.forward import ForwardInputs
    from cti.kvcache import PageTable
    from cti.utils import DeviceCapability

    from .base import BackendSetup
    from .paged import KVPages


class FlashAttentionBackend(PageTableBackend):
    UNQUALIFIED_HEAD_DIMS = frozenset({256})

    @classmethod
    def validate_environment(
        cls,
        capability: DeviceCapability | None,
    ) -> tuple[str, ...]:
        if reasons := super().validate_environment(capability):
            return reasons
        assert capability is not None
        if (capability.major, capability.minor) not in {(9, 0), (10, 0)}:
            return (
                f"SGLang FlashAttention does not support {capability}; "
                "supported capabilities are SM90 and SM100",
            )
        return cls._check_provider(
            check_flash_attention_available, "SGLang FlashAttention extension"
        )

    def __init__(self, setup: BackendSetup, *, kv: tuple[KVPages, ...], table: PageTable) -> None:
        super().__init__(setup, kv=kv, table=table)
        # validate_environment admits only SM90 and SM100.
        assert setup.capability is not None
        self.version = 4 if setup.capability.is_family(10) else 3

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardInputs,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, PageTableMetadata)
        pages = self.kv[layer_id]
        store_kv_pair(pages, k, v, metadata.cache_write_indices)
        return flash_attention_with_kv_cache(
            q=q,
            k_cache=pages[0],
            v_cache=pages[1],
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
        )
