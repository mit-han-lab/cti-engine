# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/attention/fa.py`, `python/minisgl/attention/trtllm.py`,
# `python/minisgl/attention/utils.py`, `python/minisgl/kvcache/mha_pool.py`);
# see THIRD_PARTY_NOTICES.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from cti.utils import div_ceil

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from cti.forward import ForwardInputs, ForwardRows
    from cti.kvcache import PageGroup, PageTable

    from .base import BackendSetup

# One layer's key and value pages, each [pages, entries, kv heads, head_dim].
KVPages = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class KVPair:
    """MHA's row layout: an entry holds one token's key and value over this rank's KV heads,
    one ``[2, pages, entries, num_kv_heads, head_dim]`` tensor per layer."""

    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    def page_bytes(self, entries: int) -> int:
        return 2 * entries * self.num_kv_heads * self.head_dim * self.dtype.itemsize

    def allocate(
        self,
        num_pages: int,
        entries: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        shape = (2, num_pages, entries, self.num_kv_heads, self.head_dim)
        return torch.empty(shape, device=device, dtype=self.dtype)


def kv_pages(group: PageGroup) -> tuple[KVPages, ...]:
    """Each layer's key and value pages of a ``KVPair`` group."""
    return tuple((layer[0], layer[1]) for layer in group.layers)


def store_kv_pair(pages: KVPages, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> None:
    """Write ``k`` and ``v`` at token ``slots`` of one layer's pages."""
    from ctikernels import store_cache

    k_cache, v_cache = pages
    store_cache(
        k_cache=k_cache.view(-1, *k_cache.shape[2:]),
        v_cache=v_cache.view(-1, *v_cache.shape[2:]),
        indices=slots,
        k=k,
        v=v,
    )


def token_slots(page_table: PageTable, slot: int, start: int, end: int) -> torch.Tensor:
    """The flat token indices (page ID x page size + offset) of ``slot``'s positions
    ``[start, end)``; at page size 1 a slice of the page table."""

    page_size, pages = page_table.page_size, page_table.pages
    if page_size == 1:
        return pages[slot, start:end]
    first = start // page_size
    tokens = pages[slot, first : div_ceil(end, page_size), None] * page_size + torch.arange(
        page_size, dtype=pages.dtype, device=pages.device
    )
    offset = start - first * page_size
    return tokens.flatten()[offset : offset + end - start]


def write_slots(page_table: PageTable, rows: ForwardRows) -> torch.Tensor:
    """The cache slots of ``rows``' query rows, padding included: each row's, past its
    cached prefix."""
    return torch.cat(
        [
            token_slots(page_table, slot, start, end)
            for slot, start, end in zip(
                rows.slots, rows.kv_prefix_lens, rows.sequence_lens, strict=True
            )
        ]
    )


def ragged_lengths(rows: ForwardRows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pinned host int32 ``(seq_lens, cu_seqlens_k, cu_seqlens_q)`` of ``rows``' padded rows."""

    host = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
    seqlens_q = [rows.extend_len(index) for index in range(rows.padded_size)]
    seqlens_k = list(rows.sequence_lens)
    cu_seqlens_k = torch.tensor([0] + seqlens_k, **host).cumsum_(dim=0)
    if max(seqlens_q) == 1:  # decode
        cu_seqlens_q = torch.arange(0, rows.padded_size + 1, **host)
    elif not any(rows.kv_prefix_lens):  # prefill with no cache hit
        cu_seqlens_q = cu_seqlens_k
    else:  # extend prefill after a partial cache hit
        cu_seqlens_q = torch.tensor([0] + seqlens_q, **host).cumsum_(dim=0)
    return torch.tensor(seqlens_k, **host), cu_seqlens_k, cu_seqlens_q


@dataclass
class CaptureBuffers:
    """The static decode-graph inputs of up to ``max_bs`` rows, which replays rewrite."""

    seq_lens: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
    cache_write_indices: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device) -> CaptureBuffers:
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            cache_write_indices=torch.zeros((max_bs,), dtype=torch.int32, device=device),
        )


@dataclass
class PageTableMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor


class PageTableBackend(BaseAttnBackend):
    """Paged attention fed one physical page ID per logical page position; subclasses
    supply ``forward``."""

    def __init__(self, setup: BackendSetup, *, kv: tuple[KVPages, ...], table: PageTable) -> None:
        self.kv = kv
        self.device = setup.device
        self.page_table = table
        self.page_size = table.page_size
        self.capture: CaptureBuffers | None = None
        self.capture_bs: List[int] = []
        self.scale = setup.head_dim**-0.5

    def prepare_metadata(self, rows: ForwardRows) -> PageTableMetadata:
        max_seqlen_k = max(rows.sequence_lens)
        device = self.device
        cache_seqlens, cu_seqlens_k, cu_seqlens_q = (
            lengths.to(device, non_blocking=True) for lengths in ragged_lengths(rows)
        )

        # These providers consume the page table's page IDs as they are.
        num_pages = div_ceil(max_seqlen_k, self.page_size)
        pages = self.page_table.pages
        new_page_table = torch.stack([pages[slot, :num_pages] for slot in rows.slots])
        return PageTableMetadata(
            cache_write_indices=write_slots(self.page_table, rows),
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max(rows.extend_len(index) for index in range(rows.padded_size)),
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        self.capture = CaptureBuffers.create(
            max(bs_list),
            div_ceil(max_seq_len, self.page_size),
            self.device,
        )
        self.capture_bs = sorted(bs_list)

    def capture_metadata(self, rows: ForwardRows) -> PageTableMetadata:
        assert (bs := rows.size) in self.capture_bs and self.capture
        capture = self.capture
        return PageTableMetadata(
            cache_write_indices=capture.cache_write_indices[:bs],
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )

    def prepare_for_replay(self, batch: ForwardInputs, bs: int) -> None:
        metadata = batch.attn_metadata
        assert isinstance(metadata, PageTableMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)
        self.capture.cache_write_indices[:bs].copy_(metadata.cache_write_indices)
