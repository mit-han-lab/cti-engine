"""``AttentionLayer.forward_training`` runs padded rows through SDPA and packed rows through the
causal varlen attention, bit for bit as the per-layout methods it replaced."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from cti.layers import AttentionLayer, LinearQKVMerged, RMSNorm, RotaryEmbedding
from cti.models.training_contract import Packing
from tests.support import reference_ops
from tests.support.tp_shards import default_dtype, site
from tests.support.training_ops import bind_training_references

HEAD_DIM = 64
QO_HEADS, KV_HEADS = 2, 1


@pytest.fixture(autouse=True)
def _references(monkeypatch: pytest.MonkeyPatch) -> None:
    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)
    bind_training_references(monkeypatch)


def _layer() -> AttentionLayer:
    generator = torch.Generator().manual_seed(0)
    layer_site = site(training=True)
    norms = (RMSNorm(HEAD_DIM, 1e-6), RMSNorm(HEAD_DIM, 1e-6))
    for norm in norms:
        norm.weight = 1 + 0.1 * torch.randn(HEAD_DIM, generator=generator)
    return AttentionLayer(
        qkv=LinearQKVMerged(
            hidden_size=8,
            head_dim=HEAD_DIM,
            num_qo_heads=QO_HEADS,
            num_kv_heads=KV_HEADS,
            has_bias=False,
            site=layer_site,
        ),
        rotary=RotaryEmbedding.create(HEAD_DIM, HEAD_DIM, 64, 1e6, site=layer_site),
        q_norm=norms[0],
        k_norm=norms[1],
    )


def _split(
    layer: AttentionLayer, qkv: torch.Tensor, positions: torch.Tensor, **lengths: object
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The replaced methods' shared Q/K norm and rotary, step for step."""

    q, k, v = qkv.split([QO_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
    lead = qkv.shape[:-1]
    q = layer.q_norm.forward_training(q.view(*lead, QO_HEADS, HEAD_DIM))
    k = layer.k_norm.forward_training(k.view(*lead, KV_HEADS, HEAD_DIM))
    q, k = layer.rotary.forward_training(positions, q, k, **lengths)
    return q, k, v.view(*lead, KV_HEADS, HEAD_DIM)


def _padded(layer: AttentionLayer, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """The replaced ``forward_training``."""

    q, k, v = _split(layer, qkv, positions)
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=True,
    )
    return output.transpose(1, 2).reshape(*qkv.shape[:-1], QO_HEADS * HEAD_DIM)


def _packed(layer: AttentionLayer, qkv: torch.Tensor, packing: Packing) -> torch.Tensor:
    """The replaced ``forward_packed_training``."""

    lengths = {
        "cu_seqlens": packing.cu_seqlens,
        "max_sequence_length": packing.max_sequence_length,
    }
    q, k, v = _split(layer, qkv, packing.positions, **lengths)
    output = site().services.packed_training_attention().forward(q, k, v, **lengths)
    return output.view(qkv.shape[0], QO_HEADS * HEAD_DIM)


def _outputs_and_gradient(run, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qkv = qkv.clone().requires_grad_()
    output = run(qkv)
    output.backward(torch.ones_like(output).mul_(torch.arange(output.shape[-1])))
    return output.detach(), qkv.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("layout", ["padded", "packed"])
def test_one_training_forward_matches_each_replaced_layout_bit_for_bit(
    layout: str, dtype: torch.dtype
) -> None:
    with default_dtype(dtype):
        layer = _layer()
    width = (QO_HEADS + 2 * KV_HEADS) * HEAD_DIM
    generator = torch.Generator().manual_seed(1)
    if layout == "padded":
        qkv = torch.randn(2, 7, width, generator=generator).to(dtype)
        positions = torch.arange(7)
        new = _outputs_and_gradient(lambda x: layer.forward_training(x, positions, None), qkv)
        old = _outputs_and_gradient(lambda x: _padded(layer, x, positions), qkv)
    else:
        lengths = (3, 5, 1)
        qkv = torch.randn(sum(lengths), width, generator=generator).to(dtype)
        positions = torch.cat([torch.arange(n) for n in lengths])
        cu_seqlens = torch.tensor([0, 3, 8, 9], dtype=torch.int32)
        packing = Packing(positions, cu_seqlens, max(lengths))
        new = _outputs_and_gradient(lambda x: layer.forward_training(x, positions, packing), qkv)
        old = _outputs_and_gradient(lambda x: _packed(layer, x, packing), qkv)

    assert all(torch.equal(a, b) for a, b in zip(new, old, strict=True))
    assert new[0].dtype == dtype and new[1].abs().sum() > 0
