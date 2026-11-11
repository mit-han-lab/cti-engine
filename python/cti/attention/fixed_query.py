"""Fixed-query attention of speculative rounds on cti-kernels' ``DraftAttention``.

A round attends ``B`` requests of ``Q`` queries each over device-resident KV lengths: the
target's verify, causally, and each speculative layer under its ``(causal, window)`` mask.
``DraftAttention`` picks its provider per device (its CuTe kernels for head dimension 128 on
RTX 5090, RTX PRO 6000 and GB10, FlashInfer elsewhere).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Protocol

import torch

from ctikernels.attention.inference import DraftAttention
from ctikernels.speculative import fixed_query_addresses

from .base import BaseAttnBackend, BaseAttnMetadata
from .paged import kv_pages, store_kv_pair

if TYPE_CHECKING:
    from cti.forward import ForwardInputs, ForwardRows
    from cti.kvcache import InferenceState, PageGroup, PageTable

    from .flashinfer import FlashInferBackend

# A ``(causal, window)`` mask: a ``window`` W admits each query's last W keys, and a
# non-causal one also every key of its block.
Mask = tuple[bool, int | None]


@dataclass(frozen=True, slots=True)
class FixedQueryShape:
    """The largest fixed-query operations over one KV pool: up to ``max_batch_size``
    requests of ``query_length`` queries over at most ``max_kv_tokens`` keys each, in
    ``dtype``, under each of ``masks``."""

    max_batch_size: int
    query_length: int
    max_kv_tokens: int
    masks: tuple[Mask, ...]
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class FixedQueryGeometry:
    """One KV pool's fixed-query attention on this rank: its heads and largest operations."""

    head_dim: int
    num_qo_heads: int
    num_kv_heads: int
    shape: FixedQueryShape

    @property
    def operations(self) -> tuple[tuple[int, Mask], ...]:
        """Every ``(batch size, mask)`` the pool's rounds run."""

        sizes = range(1, self.shape.max_batch_size + 1)
        return tuple((size, mask) for size in sizes for mask in self.shape.masks)

    def operation_bytes(self, batch_size: int, mask: Mask, device: torch.device) -> tuple[int, int]:
        """The float scratch and integer workspace of one operation."""

        return DraftAttention.workspace_size(
            batch_size=batch_size,
            max_kv_tokens=self.shape.max_kv_tokens,
            device=device,
            **self.arguments(mask),
        )

    @property
    def csr_words(self) -> int:
        """Query and page CSR (page size 1) and last-page lengths of the largest batch."""

        batch = self.shape.max_batch_size
        return 2 * (batch + 1) + batch * self.shape.max_kv_tokens + batch

    def arguments(self, mask: Mask) -> dict:
        """``DraftAttention``'s geometry arguments under ``mask``."""

        causal, window = mask
        return {
            "query_length": self.shape.query_length,
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": 1,
            "dtype": self.shape.dtype,
            "causal": causal,
            "window": window,
        }


def fixed_query_bytes(geometries: Sequence[FixedQueryGeometry], device: torch.device) -> int:
    """Device bytes ``create_fixed_query_backends`` allocates for ``geometries``: one float scratch
    that every operation borrows in turn, as rounds run on one stream, and per pool each
    operation's integer workspace and each mask's CSR."""

    floats, ints = 0, 0
    for geometry in geometries:
        for size, mask in geometry.operations:
            float_bytes, int_bytes = geometry.operation_bytes(size, mask, device)
            floats, ints = max(floats, float_bytes), ints + int_bytes
        ints += 4 * geometry.csr_words * len(geometry.shape.masks)
    return floats + ints


class RoundAddresses:
    """Where a round's rows write (``write``, one query length per row) and the cache rows
    they read (``indptr``/``indices``, a CSR of up to ``max_kv_tokens`` per row) in every
    pool behind one page table, for rounds of ``shape``, which ``prepare`` rewrites on the
    device each round from the rows' frontiers: a row's block at its frontier, a done row's
    on the guard rows from ``guard_start``."""

    def __init__(
        self,
        page_table: torch.Tensor,
        shape: FixedQueryShape,
        *,
        max_seq_len: int,
        guard_start: int,
    ) -> None:
        max_rows, self.width = shape.max_batch_size, shape.query_length
        self.max_kv_tokens = max_kv_tokens = shape.max_kv_tokens
        if max_rows * max_kv_tokens > torch.iinfo(torch.int32).max:
            raise ValueError("a round's KV indices exceed int32")
        if page_table.shape[1] < max_seq_len:
            raise ValueError("the page table is narrower than rounds address")
        self._page_table, self._max_seq_len, self._guard_start = (
            page_table,
            max_seq_len,
            guard_start,
        )
        device = page_table.device

        def zeros(count: int) -> torch.Tensor:
            return torch.zeros(count, dtype=torch.int32, device=device)

        self.write, self.indptr = zeros(max_rows * self.width), zeros(max_rows + 1)
        self.indices = zeros(max_rows * max_kv_tokens)

    def prepare(
        self, request_indices: torch.Tensor, frontiers: torch.Tensor, done: torch.Tensor
    ) -> None:
        """Address a round of ``request_indices`` (int32 ``[B]``) from their frontiers and
        done flags at its start; one launch on the current stream, which a graph captures."""

        rows = request_indices.numel()
        fixed_query_addresses(
            request_indices,
            frontiers,
            done,
            self._page_table,
            block_size=self.width,
            max_seq_len=self._max_seq_len,
            guard_start=self._guard_start,
            cache_write_indices=self.write[: rows * self.width],
            kv_indptr=self.indptr[: rows + 1],
            kv_indices=self.indices[: rows * self.max_kv_tokens],
        )

    def map_warm_up(self, rows: int) -> None:
        """Map the first ``rows`` request rows' first block onto distinct cache rows, which
        warm-up rounds write before any request holds them."""

        table = self._page_table
        blocks = torch.arange(rows, dtype=table.dtype, device=table.device)[:, None] * self.width
        table[:rows, : self.width] = blocks + torch.arange(
            self.width, dtype=table.dtype, device=table.device
        )


def create_fixed_query_backends(
    geometries: Sequence[FixedQueryGeometry],
    state: InferenceState,
    paged: Sequence[FlashInferBackend | None],
    page_table: PageTable,
) -> tuple[FixedQueryBackend, ...]:
    """One backend per KV group of ``state``, all borrowing one float scratch (see
    ``fixed_query_bytes``) and one round's addresses over ``page_table``'s token view and
    the state's guard rows; ``paged[i]`` serves group ``i``'s other batches."""

    groups, capacity = state.groups, state.capacity
    if page_table.tokens is None:
        raise ValueError("rounds address tokens: build the page table with its token view")
    device = groups[0].layers[0].device
    addresses = RoundAddresses(
        page_table.tokens,
        geometries[0].shape,
        max_seq_len=capacity.max_seq_len,
        guard_start=capacity.guard_start,
    )
    float_bytes = max(
        geometry.operation_bytes(size, mask, device)[0]
        for geometry in geometries
        for size, mask in geometry.operations
    )
    # The operations need zeroed float scratch on first use; it is never cleared again.
    scratch = torch.zeros(float_bytes, dtype=torch.uint8, device=device)
    return tuple(
        FixedQueryBackend(
            geometry, group=group, float_workspace=scratch, paged=backend, addresses=addresses
        )
        for geometry, group, backend in zip(geometries, groups, paged, strict=True)
    )


@dataclass
class FixedQueryMetadata(BaseAttnMetadata):
    """Attention of ``B`` requests with ``Q`` queries each over device-resident KV lengths.

    ``prepare()`` runs once per KV metadata update, before every layer that shares it, on
    the execution stream. It neither allocates nor reads lengths back to the host, so a
    CUDA graph may capture it.
    """

    operation: DraftAttention
    query_indptr: torch.Tensor
    _int_workspace: torch.Tensor

    def prepare(self) -> None:
        self.operation.prepare()


@dataclass
class MixedQuery(BaseAttnMetadata):
    """Attention of one forward over two row groups: its first ``paged_tokens`` rows attend
    as the pool's ordinary backend planned them (``paged``), the fixed-query rows after
    them under ``fixed``. Every row's keys and values are written once, before either."""

    paged: BaseAttnMetadata
    fixed: FixedQueryMetadata
    paged_tokens: int
    cache_write_indices: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.cache_write_indices = torch.cat(
            (self.paged.cache_write_indices, self.fixed.cache_write_indices)
        )


class SupportsFixedQuery(Protocol):
    """A backend that attends speculative rounds: it creates their fixed-query metadata,
    addresses each round in its pools from the rows' frontiers, and writes keys and values
    no query of the step attends."""

    def create_fixed_query(
        self,
        request_indices: torch.Tensor,
        positions: torch.Tensor,
        *,
        causal: bool,
        window: int | None = None,
    ) -> FixedQueryMetadata: ...

    def prepare_round(
        self, request_indices: torch.Tensor, frontiers: torch.Tensor, done: torch.Tensor
    ) -> None: ...

    def round_write_rows(self, tokens: int) -> torch.Tensor: ...

    def map_warm_up(self, rows: int) -> None: ...

    def write_kv(
        self, key: torch.Tensor, value: torch.Tensor, layer_id: int, batch: ForwardInputs
    ) -> None: ...


class FixedQueryBackend(BaseAttnBackend):
    """A KV pool's attention in speculative rounds, on cti-kernels' ``DraftAttention``.

    Batches whose metadata ``create_fixed_query`` made attend here, and the fixed-query rows
    of a ``MixedQuery``; every other batch, and a mixed query's leading rows, go to
    ``paged``, the pool's ordinary backend (the backends the rounds are qualified beside). A
    speculative pool has none: it attends only in rounds.
    """

    SUPPORTS_KV_PROLOGUE = True

    def __init__(
        self,
        geometry: FixedQueryGeometry,
        *,
        group: PageGroup,
        float_workspace: torch.Tensor,
        paged: FlashInferBackend | None,
        addresses: RoundAddresses,
    ) -> None:
        self.geometry = geometry
        self._addresses = addresses
        self.kv = kv_pages(group)
        self.device = group.layers[0].device
        self._float_workspace = float_workspace
        self._paged = paged
        # Per mask, the query/page CSR and last-page lengths that its operations' prepare
        # rewrites each round; batch sizes share them, as rounds run one at a time.
        self._csr = {
            mask: torch.zeros(geometry.csr_words, dtype=torch.int32, device=self.device)
            for mask in geometry.shape.masks
        }
        # The provider DraftAttention chose for each mask's operations.
        self.providers: dict[Mask, set[str]] = {mask: set() for mask in geometry.shape.masks}

    def create_fixed_query(
        self,
        request_indices: torch.Tensor,
        positions: torch.Tensor,
        *,
        causal: bool,
        window: int | None = None,
    ) -> FixedQueryMetadata:
        """Create, outside capture, one fixed-query operation over borrowed metadata.

        Row b of ``request_indices`` (int32 ``[B]``) attends with its Q queries, at the
        int64 ``positions[b * Q:(b + 1) * Q]``, to the cache rows the round's addresses give
        it, its queries' keys last, which go to their ``write`` rows. A ``window`` W admits each
        query's last W keys, and a non-causal one also every key of its block. The caller
        rewrites the metadata and the addresses and calls ``prepare()`` per round.
        """

        geometry, batch, mask = self.geometry, request_indices.numel(), (causal, window)
        addresses = self._addresses
        kv_indptr = addresses.indptr[: batch + 1]
        kv_indices = addresses.indices[: batch * addresses.max_kv_tokens]
        shape = geometry.shape
        if (
            mask not in self._csr
            or batch > shape.max_batch_size
            or kv_indices.numel() != batch * shape.max_kv_tokens
        ):
            raise ValueError(f"fixed-query attention outside this pool's shape {shape}")
        _, int_bytes = geometry.operation_bytes(batch, mask, self.device)
        int_workspace = torch.empty(int_bytes, dtype=torch.uint8, device=self.device)
        most = shape.max_batch_size
        query_indptr, page_indptr, last_page_lengths, page_indices = self._csr[mask].split(
            [most + 1, most + 1, most, most * shape.max_kv_tokens]
        )
        operation = DraftAttention(
            self._float_workspace,
            int_workspace,
            request_indices=request_indices,
            positions=positions,
            token_indptr=kv_indptr,
            token_indices=kv_indices,
            query_indptr=query_indptr[: batch + 1],
            page_indptr=page_indptr[: batch + 1],
            page_indices=page_indices[: kv_indices.numel()],
            last_page_lengths=last_page_lengths[:batch],
            **geometry.arguments(mask),
        )
        self.providers[mask].add(operation.provider)
        write = addresses.write[: batch * addresses.width]
        return FixedQueryMetadata(write, operation, query_indptr[: batch + 1], int_workspace)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardInputs,
    ) -> torch.Tensor:
        out = q.new_empty(q.shape)
        self.forward_into(q, k, v, layer_id, batch, out)
        return out

    def write_kv(
        self, key: torch.Tensor, value: torch.Tensor, layer_id: int, batch: ForwardInputs
    ) -> None:
        """Write keys and values that no query of this step attends at the first of
        ``batch``'s write slots: the page table this group shares with the target's maps
        them alike."""

        slots = batch.attn_metadata.cache_write_indices[: key.shape[0]]
        store_kv_pair(self.kv[layer_id], key, value, slots)

    def prepare_round(
        self, request_indices: torch.Tensor, frontiers: torch.Tensor, done: torch.Tensor
    ) -> None:
        """Address a round of ``request_indices`` in every pool behind this backend's page
        table, from their frontiers (int64) and done flags at its start."""

        self._addresses.prepare(request_indices, frontiers, done)

    def round_write_rows(self, tokens: int) -> torch.Tensor:
        """The pool rows a round's first ``tokens`` block tokens write."""

        return self._addresses.write[:tokens]

    def map_warm_up(self, rows: int) -> None:
        """Map the first ``rows`` request rows' first block for warm-up rounds."""

        self._addresses.map_warm_up(rows)

    def forward_into(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardInputs,
        out: torch.Tensor,
    ) -> None:
        metadata = batch.attn_metadata
        if not isinstance(metadata, FixedQueryMetadata | MixedQuery):
            self._require_paged().forward_into(q, k, v, layer_id, batch, out)
            return
        store_kv_pair(self.kv[layer_id], k, v, metadata.cache_write_indices)
        if isinstance(metadata, MixedQuery):
            self._attend_mixed(metadata, q, layer_id, out)
        else:
            self._attend(metadata, q, self._pages(layer_id), out)

    def forward_prologue(
        self,
        prologue: Callable[..., torch.Tensor],
        layer_id: int,
        batch: ForwardInputs,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        if not isinstance(metadata, FixedQueryMetadata | MixedQuery):
            return self._require_paged().forward_prologue(prologue, layer_id, batch, out)
        k_cache, v_cache = self._pages(layer_id)
        # Without their page axis, the page-size-1 views are the prologue's token rows.
        q = prologue(k_cache.squeeze(1), v_cache.squeeze(1), metadata.cache_write_indices)
        out = q.new_empty(q.shape) if out is None else out
        if isinstance(metadata, MixedQuery):
            self._attend_mixed(metadata, q, layer_id, out)
        else:
            self._attend(metadata, q, (k_cache, v_cache), out)
        return out

    def prepare_metadata(self, rows: ForwardRows) -> BaseAttnMetadata:
        return self._require_paged().prepare_metadata(rows)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self._require_paged().init_capture_graph(max_seq_len, bs_list)

    def capture_metadata(self, rows: ForwardRows) -> BaseAttnMetadata:
        return self._require_paged().capture_metadata(rows)

    def prepare_for_replay(self, batch: ForwardInputs, bs: int) -> None:
        self._require_paged().prepare_for_replay(batch, bs)

    def _attend_mixed(
        self, metadata: MixedQuery, q: torch.Tensor, layer_id: int, out: torch.Tensor
    ) -> None:
        """Attend a mixed query's two row groups; their keys and values are written."""

        split = metadata.paged_tokens
        self._require_paged().attend_into(q[:split], layer_id, metadata.paged, out[:split])
        self._attend(metadata.fixed, q[split:], self._pages(layer_id), out[split:])

    @staticmethod
    def _attend(
        metadata: FixedQueryMetadata,
        q: torch.Tensor,
        pages: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor,
    ) -> None:
        # DraftAttention reads dense query rows, while a fused QKV projection's (or fused
        # prologue's) Q is a strided view: the one copy, until the operation takes a row stride.
        metadata.operation.run(q=q.contiguous(), paged_kv_cache=pages, out=out)

    def _pages(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer ``layer_id``'s K and V caches as pages of one token."""

        k_cache, v_cache = self.kv[layer_id]
        return (
            k_cache.view(-1, 1, k_cache.shape[2], k_cache.shape[3]),
            v_cache.view(-1, 1, v_cache.shape[2], v_cache.shape[3]),
        )

    def _require_paged(self) -> FlashInferBackend:
        if self._paged is None:
            raise RuntimeError("a speculative pool attends only in rounds")
        return self._paged


__all__ = [
    "SupportsFixedQuery",
    "FixedQueryBackend",
    "FixedQueryGeometry",
    "FixedQueryMetadata",
    "FixedQueryShape",
    "MixedQuery",
    "create_fixed_query_backends",
    "fixed_query_bytes",
]
