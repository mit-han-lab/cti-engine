"""CPU torch references for the cti-kernels training operations a model binds.

Each follows the documented contract of the kernel leaf it names and is differentiable
through autograd in FP32, so CPU tests bind it through the seam the layer uses for the
CUDA operation:

- ``ctikernels.norm.training``: ``rms_norm``, out of place, with its ``zero_centered`` flag,
  which scales by ``1 + weight``, accepts only a frozen weight and takes an ``out_dtype``;
- ``ctikernels.rotary.training.apply_rotary``: NeoX rotary of the leading ``rotary_dim``
  channels from ``[max_position, rotary_dim / 2]`` caches (partial rotary), the remaining
  channels passed through;
- ``ctikernels.activation.training.silu_and_mul``;
- ``ctikernels.attention.varlen.CausalVarlenAttention``, ``ctikernels.cross_entropy``'s
  linear cross entropy and ``ctikernels.logprob``'s selected-token log-probabilities.
"""

from __future__ import annotations

import importlib
import sys
from itertools import pairwise
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from tests.support import reference_ops


def _require_frozen(operation: str, weight: torch.Tensor) -> None:
    if weight.requires_grad:
        raise ValueError(f"zero-centered {operation} requires a frozen weight")


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``ctikernels.norm.training.rms_norm``: normalize and scale in FP32, round once to
    ``out_dtype`` (zero-centered only; ``x``'s dtype otherwise)."""

    if not zero_centered:
        if out_dtype not in (None, x.dtype):
            raise TypeError("rms_norm needs zero_centered=True for another out_dtype")
        return reference_ops.rms_norm(x, weight, eps)
    _require_frozen("rms_norm", weight)
    return reference_ops.normalized(x, weight, eps, zero_centered=True).to(out_dtype or x.dtype)


def apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    max_sequence_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``ctikernels.rotary.training.apply_rotary``: out-of-place NeoX rotary in FP32.

    Padded Q/K are ``[batch, sequence, heads, head_dim]`` with one ``positions`` vector
    of the sequence length; packed Q/K are ``[tokens, heads, head_dim]`` with one
    position per token. ``cos`` and ``sin`` are ``[max_position, rotary_dim / 2]``.
    """

    if cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError("cos and sin must share shape [max_position, rotary_dim / 2]")
    if positions.dtype != torch.int64:
        raise TypeError(f"positions must be int64, not {positions.dtype}")
    if cu_seqlens is None:
        if query.ndim != 4 or positions.shape != (query.shape[1],):
            raise ValueError("padded query and positions must share the sequence length")
        cache_shape: tuple[int, ...] = (1, query.shape[1], 1, cos.shape[1])
    else:
        if query.ndim != 3 or positions.shape != (query.shape[0],):
            raise ValueError("packed query and positions must share the token count")
        if max_sequence_length is None:
            raise ValueError("packed rotary requires max_sequence_length")
        cache_shape = (query.shape[0], 1, cos.shape[1])
    rotary_dim = 2 * cos.shape[1]
    if not 0 < rotary_dim <= query.shape[-1] or key.shape[-1] != query.shape[-1]:
        raise ValueError("rotary_dim must be positive and at most the shared head_dim")
    selected_cos = cos.index_select(0, positions).view(cache_shape)
    selected_sin = sin.index_select(0, positions).view(cache_shape)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        rotated, passed = x.float().split((rotary_dim, x.shape[-1] - rotary_dim), dim=-1)
        first, second = rotated.chunk(2, dim=-1)
        return torch.cat(
            (
                first * selected_cos - second * selected_sin,
                second * selected_cos + first * selected_sin,
                passed,
            ),
            dim=-1,
        ).to(x.dtype)

    return rotate(query), rotate(key)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``ctikernels.activation.training.silu_and_mul``: ``silu(gate) * up`` of ``gate | up``."""

    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


class CausalVarlenAttention:
    """``ctikernels.attention.varlen.CausalVarlenAttention`` by per-sequence causal SDPA."""

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
        return torch.cat(
            [
                F.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1),
                    k[start:end].transpose(0, 1),
                    v[start:end].transpose(0, 1),
                    is_causal=True,
                    enable_gqa=q.shape[1] != k.shape[1],
                ).transpose(0, 1)
                for start, end in pairwise(bounds)
            ]
        )


class LinearCrossEntropyLoss:
    """``ctikernels.cross_entropy.LinearCrossEntropyLoss``: the summed FP32 loss."""

    # The fused loss's token rows per logits chunk, which size the head's working set.
    chunk_size = 256

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = F.linear(hidden_states, weight, bias).float()
        return F.cross_entropy(logits, target_ids, reduction="sum")


class LinearSelectedTokenLogprobs:
    """``ctikernels.logprob.LinearSelectedTokenLogprobs``: FP32 ``[N]`` log-probabilities."""

    # The chunked operation's token rows per logits chunk.
    chunk_size = 64

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        logits = F.linear(hidden_states, weight, bias).float() / temperature
        return torch.log_softmax(logits, dim=-1).gather(1, target_ids[:, None]).squeeze(1)


def bind_kernel_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    **members: object,
) -> ModuleType:
    """Stand in a module named ``name`` holding ``members`` for a cti-kernels leaf.

    Layers import a leaf either by module or by package attribute, so the stand-in
    replaces both the module and the package's attribute.
    """

    module = ModuleType(name)
    for member, value in members.items():
        setattr(module, member, value)
    monkeypatch.setitem(sys.modules, name, module)
    package, _, leaf = name.rpartition(".")
    monkeypatch.setattr(importlib.import_module(package), leaf, module, raising=False)
    return module


def bind_training_references(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the norm, rotary and activation references at their training seams."""

    bind_kernel_module(
        monkeypatch,
        "ctikernels.norm.training",
        rms_norm=rms_norm,
    )
    bind_kernel_module(monkeypatch, "ctikernels.rotary.training", apply_rotary=apply_rotary)
    bind_kernel_module(
        monkeypatch,
        "ctikernels.activation.training",
        silu_and_mul=silu_and_mul,
    )


__all__ = [
    "CausalVarlenAttention",
    "LinearCrossEntropyLoss",
    "LinearSelectedTokenLogprobs",
    "apply_rotary",
    "bind_kernel_module",
    "bind_training_references",
    "rms_norm",
    "silu_and_mul",
]
