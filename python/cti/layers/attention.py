# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/layers/attention.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from numbers import Real
from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.nn.functional as F

from cti.kvcache import StateSlot

from .base import StateLessOP
from .services import PrologueGeometry

if TYPE_CHECKING:
    from cti.attention import BaseAttnBackend
    from cti.forward import ForwardInputs
    from cti.kvcache import BoundSlot, PagedSlot
    from ctikernels.attention.inference import Prologue

    from .linear import LinearQKVMerged
    from .norm import RMSNorm
    from .rotary import RotaryEmbedding
    from .site import Site

# The paged slot of ``AttentionRequirement.mha``, which names it so.
_KV_SLOT = "kv"


class PackedSequences(Protocol):
    """Packed rows' sequence boundaries and the longest sequence, as
    ``models.training_contract.Packing`` holds them."""

    @property
    def cu_seqlens(self) -> torch.Tensor: ...

    @property
    def max_sequence_length(self) -> int: ...


@dataclass(frozen=True, slots=True)
class PrologueFactory:
    """The fused prologue (``ctikernels.attention.inference.Prologue``: Q/K RMSNorm, NeoX
    rotary and the K/V write as one operation on the backend's cache) an ``AttentionLayer``
    takes at construction, chosen by the layer's geometry (head counts and width, rotary
    width, norm kind and epsilon) on its site's device.

    ``required``: the layer always runs the operation, which cti-kernels composes from
    separate kernels where it fuses none; a geometry it cannot run fails construction, and a
    backend without ``SUPPORTS_KV_PROLOGUE`` fails the binding. Otherwise the layer takes it
    only where its services fuse it (``LayerServices.attention_prologue``) and runs it only
    on a backend that runs KV prologues; elsewhere it keeps its separate operations."""

    required: bool = False

    def build(self, layer: AttentionLayer, site: Site) -> Prologue | None:
        geometry = _prologue_geometry(layer)
        if isinstance(geometry, str):
            if self.required:
                raise ValueError(f"the attention layer's prologue {geometry}")
            return None
        return site.services.attention_prologue(
            site.device,
            geometry,
            required=self.required,
            norm_dtype=layer.q_norm.weight.dtype,
        )


class AttentionLayer(StateLessOP):
    """Paged attention of one ``[Q | K | V]`` projection's rows after optional Q/K RMSNorm and
    rotary, on the backend of the KV group its slot is bound to (``bind_state``). With the
    fused prologue that ``prologue`` chooses, inference runs the norms, the rotary and the
    K/V write as that one operation on the backend's cache; training keeps them separate."""

    def __init__(
        self,
        *,
        qkv: LinearQKVMerged,
        rotary: RotaryEmbedding,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
        prologue: PrologueFactory | None = None,
    ):
        """``qkv`` is the projection that feeds this layer; its heads are this rank's, and the
        layer runs on its site."""
        site = qkv.site
        self.head_dim = qkv.head_dim
        self.num_qo_heads = qkv.num_qo_heads
        self.num_kv_heads = qkv.num_kv_heads
        self.qo_attn_dim = self.num_qo_heads * self.head_dim
        self.kv_attn_dim = self.num_kv_heads * self.head_dim
        self.rotary = rotary
        self.q_norm = q_norm
        self.k_norm = k_norm
        self._services = site.services
        self._prologue_op = None if prologue is None else prologue.build(self, site)
        self._prologue_required = prologue is not None and prologue.required
        # Bound once the inference state exists: the group's backend and this layer's index
        # in the group, and whether the prologue runs fused on it.
        self._backend: BaseAttnBackend | None = None
        self._index = -1
        self._fused = False

    @property
    def prologue(self) -> Prologue | None:
        """The fused prologue, which the layers of one geometry share, or None."""
        return self._prologue_op

    def state_slots(self) -> tuple[StateSlot, ...]:
        """One paged slot, of the MHA requirement's name."""
        return (StateSlot(_KV_SLOT, requirement=_KV_SLOT),)

    def bind_state(self, bound: Mapping[str, BoundSlot]) -> None:
        """Attend on the paged slot's backend at the slot's index; a required prologue needs
        a backend that runs KV prologues."""
        slot = cast("PagedSlot", bound[_KV_SLOT])
        backend = slot.backend
        if self._prologue_required and not backend.SUPPORTS_KV_PROLOGUE:
            raise ValueError(f"{type(backend).__name__} does not run the required KV prologue")
        self._backend, self._index = backend, slot.index
        self._fused = self._prologue_op is not None and backend.SUPPORTS_KV_PROLOGUE

    def forward(self, qkv: torch.Tensor, batch: ForwardInputs) -> torch.Tensor:
        """``[rows, qo_attn_dim]`` attention of ``qkv``'s rows, a break of ``batch``'s graph:
        the prologue, the K/V write and the attend all run eagerly between segments."""

        return batch.graph_breaks.eager(self._attend_into, qkv, batch, width=self.qo_attn_dim)

    def store_context(
        self, kv: tuple[torch.Tensor, torch.Tensor], positions: torch.Tensor, batch: ForwardInputs
    ) -> None:
        """Write context rows' K (K-normed and rotated at ``positions``) and V, which no query
        of this step attends, at the first of ``batch``'s write slots."""

        k, v = kv
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        backend = self._bound()
        backend.write_kv(self.rotary.forward_key(positions, k), v, self._index, batch)

    def _bound(self) -> BaseAttnBackend:
        if self._backend is None:
            raise RuntimeError("the attention layer is not bound to its KV group")
        return self._backend

    def _attend_into(
        self,
        qkv: torch.Tensor,
        batch: ForwardInputs,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        heads = None if out is None else out.view(-1, self.num_qo_heads, self.head_dim)
        backend, index = self._bound(), self._index
        if self._fused:
            o = backend.forward_prologue(self._prologue(qkv, batch), index, batch, heads)
        else:
            q, k, v = self._prepare_qkv(qkv, batch)
            if heads is None:
                o = backend.forward(q, k, v, index, batch)
            else:
                backend.forward_into(q, k, v, index, batch, heads)
                o = heads
        return o.view(-1, self.qo_attn_dim)

    def _prologue(self, qkv: torch.Tensor, batch: ForwardInputs) -> Callable[..., torch.Tensor]:
        norms = (self.q_norm.weight, self.k_norm.weight)
        return partial(
            self._prologue_op, qkv, *norms, self.rotary.cos_sin_cache, batch.rope_positions
        )

    def _prepare_qkv(
        self,
        qkv: torch.Tensor,
        batch: ForwardInputs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(batch.rope_positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        return q, k, v

    def forward_training(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        packing: PackedSequences | None,
    ) -> torch.Tensor:
        """Causal attention of padded ``[batch, sequence, q | k | v]`` rows (torch SDPA)
        without ``packing``, or of packed ``[tokens, q | k | v]`` rows whose sequences
        ``packing`` bounds (cti-kernels' causal varlen attention)."""
        if packing is None:
            q, k, v = self._training_qkv(qkv, positions)
            output = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=self.num_qo_heads != self.num_kv_heads,
            )
            return output.transpose(1, 2).reshape(*qkv.shape[:-1], self.qo_attn_dim)
        lengths = {
            "cu_seqlens": packing.cu_seqlens,
            "max_sequence_length": packing.max_sequence_length,
        }
        q, k, v = self._training_qkv(qkv, positions, **lengths)
        output = self._services.packed_training_attention().forward(q, k, v, **lengths)
        return output.view(qkv.shape[0], self.qo_attn_dim)

    def _training_qkv(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        **lengths: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        lead = qkv.shape[:-1]
        q = q.view(*lead, self.num_qo_heads, self.head_dim)
        k = k.view(*lead, self.num_kv_heads, self.head_dim)
        v = v.view(*lead, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm.forward_training(q)
        if self.k_norm is not None:
            k = self.k_norm.forward_training(k)
        q, k = self.rotary.forward_training(positions, q, k, **lengths)
        return q, k, v


class BidirectionalAttention(StateLessOP):
    """Noncausal attention within each sequence of a packed batch, with no KV cache: every
    token attends to all tokens of its own sequence (``cu_seqlens`` bounds them), and one
    cti-kernels varlen call covers the batch. Vision towers run it with one sequence per
    image."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        from ctikernels.attention.varlen import NoncausalVarlenAttention

        self.num_heads = num_heads
        self.head_dim = head_dim
        self._attention = NoncausalVarlenAttention()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_sequence_length: int,
    ) -> torch.Tensor:
        """Attend q/k/v given as ``[tokens, heads * head_dim]`` or per head.

        Returns ``[tokens, heads * head_dim]``.
        """
        head_shape = (-1, self.num_heads, self.head_dim)
        o = self._attention.forward(
            q.view(head_shape),
            k.view(head_shape),
            v.view(head_shape),
            cu_seqlens=cu_seqlens,
            max_sequence_length=max_sequence_length,
        )
        return o.view(-1, self.num_heads * self.head_dim)


def _prologue_geometry(layer: AttentionLayer) -> PrologueGeometry | str:
    """What a fused prologue of ``layer`` depends on, or why one cannot run it."""

    norms = {"Q": layer.q_norm, "K": layer.k_norm}
    for name, norm in norms.items():
        if norm is None:
            return f"requires a {name} RMSNorm"
        if norm.weight.shape != (layer.head_dim,):
            return f"requires {name} norm weight shape ({layer.head_dim},)"
        eps = norm.eps
        if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
            return f"requires positive finite {name} norm epsilon; got {eps}"
    q_norm, k_norm = norms.values()
    if layer.rotary.head_dim != layer.head_dim:
        return f"requires rotary head_dim={layer.head_dim}; got {layer.rotary.head_dim}"
    if q_norm.zero_centered != k_norm.zero_centered:
        return "requires Q and K RMSNorms of one kind (zero-centered or not)"
    if q_norm.eps != k_norm.eps:
        return f"requires the same Q/K norm epsilon; got Q={q_norm.eps}, K={k_norm.eps}"
    if q_norm.weight.dtype != k_norm.weight.dtype:
        return "requires Q and K norm weights of one dtype"
    return PrologueGeometry(
        num_q_heads=layer.num_qo_heads,
        num_kv_heads=layer.num_kv_heads,
        eps=q_norm.eps,
        head_dim=layer.head_dim,
        rotary_dim=layer.rotary.rotary_dim,
        zero_centered=q_norm.zero_centered,
        position_axes=layer.rotary.position_axes,
    )
