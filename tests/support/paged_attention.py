"""A CPU paged-attention backend for model tests, and the KV groups it serves.

It writes K and V into a ``KVPair`` group (``store_kv_pair``), or runs a KV prologue on the
group's token rows, and reads each request's rows back through the runner's page table at
page size 1, the addressing the FlashInfer backend hands its kernels, then attends causally
per request with ``scaled_dot_product_attention`` at the default ``head_dim ** -0.5`` scale.
It serves eager EXTEND, MIXED and DECODE batches; it does not capture graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

from cti.attention import BaseAttnBackend, BaseAttnMetadata
from cti.attention.paged import KVPair, kv_pages, store_kv_pair, write_slots
from cti.kvcache import GroupRole, PageGroup, ResolvedSlot, Spec, group_layers

if TYPE_CHECKING:
    from cti.forward import ForwardInputs, ForwardRows
    from cti.kvcache import RecurrentStatePool


def kv_group(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_pages: int,
    *,
    dtype: torch.dtype = torch.float32,
    page_size: int = 1,
    device: torch.device | str = "cpu",
    role: GroupRole = GroupRole.TARGET,
    index: int = 0,
) -> PageGroup:
    """An MHA KV group of ``num_layers`` uninitialized layers, as ``InferenceState``
    allocates one."""
    spec = Spec(KVPair(num_kv_heads, head_dim, dtype))
    device = torch.device(device)
    layers = tuple(
        spec.layout.allocate(num_pages, page_size, device=device) for _ in range(num_layers)
    )
    return PageGroup(role, index, spec, layers)


def bind_model_state(
    model: object,
    backend: object,
    recurrent: RecurrentStatePool | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Bind ``model``'s state layers as the runner does: its attention layers' KV group to
    ``backend``, its recurrent layers to their rows of ``recurrent``."""
    attention = next(model.iter_attention_layers())
    pair = KVPair(attention.num_kv_heads, attention.head_dim, dtype)
    slots = {(GroupRole.TARGET, "kv"): ResolvedSlot("mha", Spec(pair))}
    layers = [(GroupRole.TARGET, layer) for layer in model.iter_state_layers()]
    group_layers(layers, slots).bind((backend,), recurrent)


@dataclass
class _ReferenceAttentionMetadata(BaseAttnMetadata):
    rows: ForwardRows  # attention walks the padded rows per request


class ReferencePagedAttentionBackend(BaseAttnBackend):
    SUPPORTS_KV_PROLOGUE = True

    def __init__(self, group: PageGroup, pages: torch.Tensor) -> None:
        """``pages``: each slot's page IDs on page-size-1 pools, which are token rows."""
        self._kv = kv_pages(group)
        self._page_table = SimpleNamespace(page_size=1, pages=pages)

    def prepare_metadata(self, rows: ForwardRows) -> _ReferenceAttentionMetadata:
        return _ReferenceAttentionMetadata(
            cache_write_indices=write_slots(self._page_table, rows), rows=rows
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardInputs,
    ) -> torch.Tensor:
        store_kv_pair(self._kv[layer_id], k, v, batch.attn_metadata.cache_write_indices)
        return self._attend(q, layer_id, batch)

    def forward_prologue(
        self,
        prologue: Callable[..., torch.Tensor],
        layer_id: int,
        batch: ForwardInputs,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k_cache, v_cache = self._token_rows(layer_id)
        q = prologue(k_cache, v_cache, batch.attn_metadata.cache_write_indices)
        o = self._attend(q, layer_id, batch)
        return o if out is None else out.copy_(o)

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        raise NotImplementedError("the reference attention backend runs eagerly")

    def capture_metadata(self, rows: ForwardRows) -> BaseAttnMetadata:
        raise NotImplementedError("the reference attention backend runs eagerly")

    def prepare_for_replay(self, batch: ForwardInputs, bs: int) -> None:
        raise NotImplementedError("the reference attention backend runs eagerly")

    def _token_rows(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        k_cache, v_cache = self._kv[layer_id]
        return k_cache.flatten(0, 1), v_cache.flatten(0, 1)

    def _attend(self, q: torch.Tensor, layer_id: int, batch: ForwardInputs) -> torch.Tensor:
        k_cache, v_cache = self._token_rows(layer_id)
        outputs = []
        start = 0
        forward = batch.attn_metadata.rows
        for slot, prefix, length in zip(
            forward.slots, forward.kv_prefix_lens, forward.sequence_lens, strict=True
        ):
            end = start + length - prefix
            rows = self._page_table.pages[slot, :length]
            key_positions = torch.arange(length, device=q.device)
            visible = key_positions <= key_positions[prefix:, None]
            output = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1),
                k_cache[rows].transpose(0, 1),
                v_cache[rows].transpose(0, 1),
                attn_mask=visible,
                enable_gqa=True,
            )
            outputs.append(output.transpose(0, 1))
            start = end
        return torch.cat(outputs)


__all__ = ["ReferencePagedAttentionBackend", "bind_model_state", "kv_group"]
