"""CPU torch references for cti-kernels inference operations.

Each function follows the documented contract of the kernel leaf it names, so CPU
tests bind it through the same seam the layer uses for the CUDA operation;
``bind_norm_references`` binds the norms and ``bind_vision_references`` the vision
tower's operations.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ctikernels.norm import inference as inference_norm

if TYPE_CHECKING:
    import pytest


def _require_no_autograd(operation: str, *tensors: torch.Tensor) -> None:
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError(f"{operation} does not support autograd")


def _require_mutable(operation: str, *tensors: torch.Tensor) -> None:
    if any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError(f"{operation} cannot modify a tensor that requires gradients")


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
) -> torch.Tensor:
    """``ctikernels.norm.inference.rms_norm``: normalize and scale in FP32, round once."""
    return normalized(x, weight, eps, zero_centered=zero_centered).to(x.dtype)


def rms_norm_(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
) -> torch.Tensor:
    """``ctikernels.norm.inference.rms_norm_``: overwrite ``x`` with its normalization."""
    return x.copy_(rms_norm(x, weight, eps, zero_centered=zero_centered))


def add_rms_norm_(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
    store_residual: bool = True,
    normalize: str = "fp32_sum",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``ctikernels.norm.inference.add_rms_norm_``: ``residual += x``, then ``x = norm``.

    Neither ``x`` nor ``residual`` may require grad, nor ``weight`` under autograd. Both
    providers normalize the FP32 sum by default. FlashInfer requires ``residual`` in
    ``x``'s dtype. The zero-centered residual-stream kernel stores the sum in
    ``residual``'s dtype, ``x``'s or FP32, and takes the operands of
    :func:`require_stream_operands` with a ``residual`` distinct from ``x``, laid out as
    :func:`require_stream_layout` says. Only it has ``normalize="stored_sum"``, which
    normalizes the sum as stored, and ``store_residual=False``, which leaves ``residual``
    unwritten.
    """
    _require_mutable("add_rms_norm_", x, residual)
    _require_no_autograd("add_rms_norm_", weight)
    if normalize not in ("fp32_sum", "stored_sum"):
        raise ValueError(f"normalize must be 'fp32_sum' or 'stored_sum', not {normalize!r}")
    if not zero_centered:
        if not store_residual:
            raise ValueError("add_rms_norm_ needs zero_centered=True to leave residual unwritten")
        if normalize != "fp32_sum":
            raise ValueError("add_rms_norm_ needs zero_centered=True to normalize the stored sum")
        if residual.dtype != x.dtype:
            raise TypeError("add_rms_norm_ needs zero_centered=True for another residual dtype")
        total = residual.float() + x.float()
        residual.copy_(total)
        return x.copy_(normalized(total, weight, eps, zero_centered=False)), residual
    require_stream_operands(x, residual, weight)
    # Only the in-place kernel refuses an aliased pair; empty ones may share a null pointer.
    if x.numel() and residual.data_ptr() == x.data_ptr():
        raise ValueError("residual must be distinct from the input")
    require_stream_layout(x, residual, weight)
    total = residual.float() + x.float()
    stored = total.to(residual.dtype)
    if store_residual:
        residual.copy_(stored)
    summed = stored if normalize == "stored_sum" else total
    return x.copy_(normalized(summed, weight, eps, zero_centered=True)), residual


def residual_rms_norm(
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``ctikernels.norm.inference.residual_rms_norm``: the zero-centered norm of a residual
    that no add precedes, rounded once to ``out_dtype`` in new storage.

    Under autograd neither ``residual`` nor ``weight`` may require grad. The residual is in
    ``out_dtype`` or FP32, on the operands of :func:`require_stream_operands` laid out as
    :func:`require_stream_layout` says.
    """
    _require_no_autograd("residual_rms_norm", residual, weight)
    if not zero_centered:
        raise ValueError("residual_rms_norm is zero-centered only and needs zero_centered=True")
    require_stream_operands(residual, None, weight)
    if residual.dtype not in (out_dtype, torch.float32):
        raise TypeError("residual must use out_dtype or FP32")
    require_stream_layout(residual, weight)
    return normalized(residual, weight, eps, zero_centered=True).to(out_dtype)


def normalized(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool,
) -> torch.Tensor:
    """The FP32 norm of ``x`` scaled by ``weight`` or ``1 + weight``, before any rounding."""
    hidden = x.float()
    scale = 1.0 + weight.float() if zero_centered else weight.float()
    return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps) * scale


def require_stream_operands(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
) -> None:
    """The residual-stream kernel's operands: ``[..., hidden]`` rows, a ``[hidden]`` weight
    and a ``residual`` of ``x``'s shape in its dtype or FP32 that, in FP32, shares no
    storage with ``x``."""
    if weight.shape != (x.shape[-1],):
        raise ValueError("weight must be a contiguous [hidden] tensor on the input device")
    if residual is not None:
        if residual.shape != x.shape:
            raise ValueError("residual must match the input shape and device")
        if residual.dtype not in (x.dtype, torch.float32):
            raise TypeError("residual must use the input dtype or FP32")
        shared = residual.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        if x.numel() and residual.dtype != x.dtype and shared:
            raise ValueError("an FP32 residual must not share storage with the input")


def require_stream_layout(*tensors: torch.Tensor) -> None:
    """The residual-stream kernel's layout: every operand contiguous at a 16-byte aligned
    address."""
    for tensor in tensors:
        if not tensor.is_contiguous() or tensor.data_ptr() % 16:
            raise ValueError("residual-stream norms need contiguous, 16-byte aligned tensors")


def apply_rotary_(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    position_axes: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``ctikernels.rotary.inference.apply_rotary_``: in-place NeoX rotation in FP32.

    The cache rows are ``cos | sin`` of width ``rotary_dim``; only the leading
    ``rotary_dim`` channels of each head are rotated. The provider requires a contiguous
    cache. ``positions`` is ``[tokens]``, or ``[3, tokens]`` with ``position_axes``: then
    frequency ``j`` reads the cache at axis ``position_axes[j]``'s position.
    """
    if not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache must be contiguous")
    half = cos_sin_cache.shape[-1] // 2
    if positions.ndim == 1:
        rows = cos_sin_cache[positions]
    elif position_axes is None:
        raise ValueError("[3, tokens] positions need position_axes")
    else:
        axes = torch.tensor(position_axes * 2, device=positions.device)
        columns = torch.arange(2 * half, device=positions.device)
        rows = cos_sin_cache[positions[axes].T, columns]
    cos, sin = rows.unsqueeze(1).split(half, dim=-1)
    for tensor in (query, key):
        heads = tensor.view(tensor.shape[0], -1, head_dim)
        first = heads[..., :half].float()
        second = heads[..., half : 2 * half].float()
        heads[..., : 2 * half] = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), -1
        )
    return query, key


def vision_apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``ctikernels.rotary.vision.apply_rotary``: per-token NeoX rotation, out of place."""

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        values = tensor.float()
        first, second = values.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (values * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)).to(tensor.dtype)

    return rotate(query), rotate(key)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``ctikernels.activation.inference.silu_and_mul``: ``silu(gate) * up`` of ``gate | up``."""
    gate, up = x.chunk(2, dim=-1)
    y = F.silu(gate) * up
    return y if out is None else out.copy_(y)


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    """``ctikernels.indexing``: gather embedding rows; ``vocab_range`` is ``(start, length)``.

    With a vocabulary shard, a token outside ``[start, start + length)`` reads a zero row
    and a token inside it reads local row ``token - start``.
    """
    if vocab_range is None:
        rows = weights[indices]
    else:
        start, length = vocab_range
        local = indices.long() - start
        owned = (local >= 0) & (local < length)
        rows = weights.new_zeros(len(indices), weights.shape[1])
        rows[owned] = weights[local[owned]]
    return rows if output is None else output.copy_(rows)


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """``ctikernels.store_cache``: write token ``i``'s K and V rows at cache row ``indices[i]``."""
    for cache, rows in ((k_cache, k), (v_cache, v)):
        cache.view(cache.shape[0], -1)[indices] = rows.reshape(len(indices), -1)


def paged_kv_metadata(
    page_table: torch.Tensor,
    rows: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    page_size: int,
    granularity: str,
    max_kv_len: int,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> None:
    """``ctikernels.attention.inference.paged_kv_metadata``: request b's first ``kv_lens[b]``
    tokens of table row ``rows[b]`` as a CSR of token slots (``"token"``) or page IDs
    (``"page"``); entries past ``kv_indices``' length are dropped."""
    del max_kv_len  # sizes the kernel's grid only
    lengths = kv_lens.long()
    tokens = granularity == "token"
    entries = []
    for row, length in zip(rows.tolist(), lengths.tolist(), strict=True):
        positions = torch.arange(length if tokens else -(-length // page_size))
        if tokens:
            entries.append(
                page_table[row, positions // page_size] * page_size + positions % page_size
            )
        else:
            entries.append(page_table[row, positions])
    counts = torch.tensor([len(entry) for entry in entries], dtype=torch.long)
    kv_indptr.copy_(F.pad(counts.cumsum(0), (1, 0)))
    values = torch.cat(entries)[: kv_indices.numel()]
    kv_indices[: len(values)] = values
    last = lengths.clamp(max=1) if tokens else (lengths - 1) % page_size + 1
    kv_last_page_len.copy_(torch.where(lengths > 0, last, 0))


class Prologue:
    """``ctikernels.attention.inference.Prologue`` from the norm, rotary and store references,
    in the composition's order; any dtype and device."""

    provider = "reference"

    def __init__(
        self,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        device,
        head_dim: int = 256,
        rotary_dim: int = 64,
        zero_centered: bool = True,
        position_axes: tuple[int, ...] | None = None,
    ) -> None:
        self.num_q_heads, self.num_kv_heads, self.eps = num_q_heads, num_kv_heads, eps
        self.head_dim, self.rotary_dim, self.zero_centered = head_dim, rotary_dim, zero_centered
        self.position_axes = position_axes

    def __call__(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        head_dim = k_cache.shape[-1]
        q_size, kv_size = self.num_q_heads * head_dim, self.num_kv_heads * head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        for x, weight, heads in ((q, q_weight, self.num_q_heads), (k, k_weight, self.num_kv_heads)):
            rms_norm_(
                x.view(-1, heads, head_dim), weight, self.eps, zero_centered=self.zero_centered
            )
        apply_rotary_(
            q, k, cos_sin_cache, positions, head_dim=head_dim, position_axes=self.position_axes
        )
        store_cache(k_cache, v_cache, cache_indices, k, v)
        return q.view(-1, self.num_q_heads, head_dim)


class NoncausalVarlenAttention:
    """``ctikernels.attention.varlen.NoncausalVarlenAttention`` by per-sequence SDPA."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_sequence_length: int,
    ) -> torch.Tensor:
        bounds = cu_seqlens.tolist()
        if max(end - start for start, end in pairwise(bounds)) > max_sequence_length:
            raise ValueError("a sequence exceeds max_sequence_length")
        outputs = [
            F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1),
                k[start:end].transpose(0, 1),
                v[start:end].transpose(0, 1),
                enable_gqa=q.shape[1] != k.shape[1],
            ).transpose(0, 1)
            for start, end in pairwise(bounds)
        ]
        return torch.cat(outputs)


def bind_vision_references(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the vision rotary and varlen attention references where the vision layers
    resolve them, at construction; the varlen provider needs flash-attn even to build."""

    monkeypatch.setattr(
        "ctikernels.attention.varlen.NoncausalVarlenAttention", NoncausalVarlenAttention
    )
    monkeypatch.setattr("ctikernels.rotary.vision.apply_rotary", vision_apply_rotary)


def bind_norm_references(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the norm references in ``ctikernels.norm.inference``, the module the layers call."""

    monkeypatch.setattr(inference_norm, "rms_norm", rms_norm)
    monkeypatch.setattr(inference_norm, "rms_norm_", rms_norm_)
    monkeypatch.setattr(inference_norm, "add_rms_norm_", add_rms_norm_)
    monkeypatch.setattr(inference_norm, "residual_rms_norm", residual_rms_norm)
