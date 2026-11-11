# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/attention/trtllm.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ctikernels.attention.inference import (
    check_trtllm_available,
    trtllm_context_attention,
    trtllm_decode_attention,
)

from .base import BackendSetup, PageConstraint
from .paged import KVPages, PageTableBackend, PageTableMetadata, store_kv_pair

if TYPE_CHECKING:
    from cti.forward import ForwardInputs
    from cti.kvcache import PageTable
    from cti.utils import DeviceCapability

_WORKSPACE_BYTES = 128 * 1024 * 1024


class TensorRTLLMBackend(PageTableBackend):
    PAGES = PageConstraint(frozenset({16, 32, 64}))
    UNQUALIFIED_HEAD_DIMS = frozenset({256})

    @classmethod
    def validate_environment(
        cls,
        capability: DeviceCapability | None,
    ) -> tuple[str, ...]:
        if reasons := super().validate_environment(capability):
            return reasons
        assert capability is not None
        if (capability.major, capability.minor) not in {(10, 0), (10, 3)}:
            return (
                "FlashInfer TRTLLM attention does not support "
                f"{capability}; supported capabilities are SM100 and SM103",
            )
        return cls._check_provider(check_trtllm_available, "FlashInfer TRTLLM attention functions")

    @classmethod
    def workspace_bytes(cls, setup: BackendSetup) -> int:
        return _WORKSPACE_BYTES

    def __init__(self, setup: BackendSetup, *, kv: tuple[KVPages, ...], table: PageTable) -> None:
        super().__init__(setup, kv=kv, table=table)
        self.workspace_buffer = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)

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
        kv_cache = self.kv[layer_id]
        store_kv_pair(kv_cache, k, v, metadata.cache_write_indices)

        if batch.ragged:
            return trtllm_context_attention(
                q=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens,
                max_seqlen_q=metadata.max_seqlen_q,
                max_seqlen_k=metadata.max_seqlen_k,
                softmax_scale=self.scale,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k=metadata.cu_seqlens_k,
                batch_size=batch.size,
            )
        return trtllm_decode_attention(
            q=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            max_seqlen_k=metadata.max_seqlen_k,
            softmax_scale=self.scale,
        )
