"""The required fused attention prologue of Qwen3.5's geometry as the layer takes it and the
backend runs it, against the composed layer."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cti.attention.paged import kv_pages, store_kv_pair
from cti.forward import InlineBreaks
from cti.kvcache import PagedSlot, PageGroup
from cti.layers import (
    AttentionLayer,
    LayerServices,
    LinearQKVMerged,
    PrologueFactory,
    RMSNorm,
    RotaryEmbedding,
)
from cti.layers.services import PrologueGeometry
from tests.support import reference_ops
from tests.support.paged_attention import kv_group
from tests.support.tp_shards import site

CPU = torch.device("cpu")
HEAD_DIM = 256
QKV_WIDTH = 4 * HEAD_DIM  # two query heads over one KV head


@pytest.fixture(autouse=True)
def _references(monkeypatch: pytest.MonkeyPatch) -> None:
    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)
    reference_ops.bind_norm_references(monkeypatch)
    monkeypatch.setattr("ctikernels.store_cache", reference_ops.store_cache)
    monkeypatch.setattr("ctikernels.attention.inference.Prologue", _RecordingPrologue)
    _RecordingPrologue.calls.clear()


class _RecordingPrologue(reference_ops.Prologue):
    """The reference, recording each call's arguments and Q."""

    calls: list[tuple[tuple[torch.Tensor, ...], torch.Tensor]] = []

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        self.calls.append((args, super().__call__(*args)))
        return self.calls[-1][1]


class _PoolBackend:
    """Writes K and V to ``pool`` either way, then returns Q scaled by the layer index + 1."""

    SUPPORTS_KV_PROLOGUE = True

    def __init__(self, pool: PageGroup) -> None:
        self.kv = kv_pages(pool)

    def forward(self, q, k, v, layer_id, batch):
        store_kv_pair(self.kv[layer_id], k, v, batch.attn_metadata.cache_write_indices)
        return q * (layer_id + 1)

    def forward_prologue(self, prologue, layer_id, batch, out=None):
        k_cache, v_cache = (cache.flatten(0, 1) for cache in self.kv[layer_id])
        q = prologue(k_cache, v_cache, batch.attn_metadata.cache_write_indices) * (layer_id + 1)
        return q if out is None else out.copy_(q)


def _pool() -> PageGroup:
    pool = kv_group(2, 1, HEAD_DIM, 4, dtype=torch.bfloat16, page_size=4)
    for layer in pool.layers:
        layer.zero_()
    return pool


def _layer(
    fused: bool = True,
    attribute: str = "",
    value: object = None,
    *,
    backend: _PoolBackend | None = None,
) -> AttentionLayer:
    """Layer 1 of a Qwen3.5-like geometry with M-RoPE and BF16 norm weights, with the
    required prologue if ``fused``; ``attribute`` (a constructor argument, or an attribute
    path below one) is set to ``value`` first."""
    generator = torch.Generator().manual_seed(0)
    layer_site = site(services=LayerServices())
    norms = (
        RMSNorm(HEAD_DIM, 1e-6, zero_centered=True),
        RMSNorm(HEAD_DIM, 1e-6, zero_centered=True),
    )
    for norm in norms:
        norm.weight = (0.1 * torch.randn(HEAD_DIM, generator=generator)).bfloat16()
    parts = {
        "qkv": LinearQKVMerged(
            hidden_size=8,
            head_dim=HEAD_DIM,
            num_qo_heads=2,
            num_kv_heads=1,
            has_bias=False,
            site=layer_site,
        ),
        "rotary": RotaryEmbedding.create(
            HEAD_DIM, 64, 64, 1e7, mrope_sections=(11, 11, 10), site=layer_site
        ),
        "q_norm": norms[0],
        "k_norm": norms[1],
    }
    if attribute == "rotary_dim":
        parts["rotary"] = RotaryEmbedding.create(HEAD_DIM, value, 64, 1e7, site=layer_site)
    elif attribute:
        part, _, name = attribute.partition(".")
        if name:
            owner, _, name = name.rpartition(".")
            target = parts[part]
            setattr(getattr(target, owner) if owner else target, name, value)
        else:
            parts[part] = value
    prologue = PrologueFactory(required=True) if fused else None
    layer = AttentionLayer(prologue=prologue, **parts)
    if backend is not None:
        layer.bind_state({"kv": PagedSlot(backend, 1)})
    return layer


@pytest.mark.parametrize("image_rows", [False, True], ids=["text", "image"])
def test_fused_prologue_layer_hands_the_operation_its_contract_and_matches_composition(
    image_rows: bool,
) -> None:
    positions = torch.tensor([3, 4, 9])
    rope_positions = (
        torch.stack((positions, positions + 7, positions * 2)) if image_rows else positions
    )
    batch = SimpleNamespace(
        graph_breaks=InlineBreaks(),
        rope_positions=rope_positions,
        attn_metadata=SimpleNamespace(
            cache_write_indices=torch.tensor([5, 2, 9], dtype=torch.int32)
        ),
    )
    # q | k | v | gate rows, as the gated projection produces them.
    generator = torch.Generator().manual_seed(1)
    projection = torch.randn(3, QKV_WIDTH + 2 * HEAD_DIM, generator=generator).bfloat16()
    outputs, pools, rows_after = [], [], []
    for fused in (True, False):
        pool = _pool()
        layer = _layer(fused, backend=_PoolBackend(pool))
        if fused:
            fused_layer, fused_pool = layer, pool
        rows = projection.clone()
        outputs.append(layer.forward(rows[:, :QKV_WIDTH], batch))
        pools.append(pool)
        rows_after.append(rows)

    (((qkv, q_weight, k_weight, table, table_rows, k_cache, v_cache, indices), q),) = (
        _RecordingPrologue.calls
    )
    # A view of the q | k | v | gate rows in, Q back as a view of them.
    assert qkv.data_ptr() == q.data_ptr() == rows_after[0].data_ptr()
    assert qkv.stride() == projection.stride() and q.shape == (3, 2, HEAD_DIM)
    assert q_weight is fused_layer.q_norm.weight and k_weight is fused_layer.k_norm.weight
    assert indices is batch.attn_metadata.cache_write_indices
    for cache, layer_view in zip((k_cache, v_cache), fused_pool.layers[1], strict=True):
        assert cache.shape == (16, 1, HEAD_DIM) and cache.data_ptr() == layer_view.data_ptr()
    assert k_cache.stride() == v_cache.stride()
    # The static FP32 table and the batch's positions as given, [3, tokens] for an image row:
    # the operation picks each frequency's axis.
    assert table_rows.dtype == torch.int64 and table.dtype == torch.float32
    assert table is fused_layer.rotary.cos_sin_cache and table_rows is rope_positions
    assert table_rows.shape == ((3, 3) if image_rows else (3,))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert torch.equal(rows_after[0][:, QKV_WIDTH:], projection[:, QKV_WIDTH:])
    fused, composed = pools
    for index in range(2):
        torch.testing.assert_close(fused.layers[index], composed.layers[index], rtol=0, atol=0)
    assert fused.layers[1][0].flatten(0, 1)[[5, 2, 9]].any()


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("qkv.head_dim", 128, r"Q norm weight shape \(128,\)"),
        ("rotary.head_dim", 128, "rotary head_dim=256; got 128"),
        ("q_norm", None, "a Q RMSNorm"),
        ("k_norm", None, "a K RMSNorm"),
        ("q_norm", RMSNorm(HEAD_DIM, 1e-6), "Q and K RMSNorms of one kind"),
        ("k_norm", RMSNorm(HEAD_DIM, 1e-6), "Q and K RMSNorms of one kind"),
        ("q_norm.weight", torch.zeros(128), r"Q norm weight shape \(256,\)"),
        ("k_norm.weight", torch.zeros(128), r"K norm weight shape \(256,\)"),
        *[
            (f"{norm}.eps", eps, "positive finite .* norm epsilon")
            for norm in ("q_norm", "k_norm")
            for eps in (0, -1e-6, float("nan"), float("inf"), True, "1e-6")
        ],
        ("q_norm.eps", 1e-5, "the same Q/K norm epsilon"),
    ],
)
def test_invalid_layer_is_rejected_before_constructing_the_operation(
    monkeypatch: pytest.MonkeyPatch, attribute: str, value: object, message: str
) -> None:
    operation = Mock()
    monkeypatch.setattr("ctikernels.attention.inference.Prologue", operation)

    with pytest.raises(ValueError, match=f"the attention layer's prologue requires {message}"):
        _layer(True, attribute, value)

    operation.assert_not_called()


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("", None, "needs a CUDA device"),
        ("rotary_dim", 128, r"unsupported .*\(256, 128, True\)"),
    ],
)
def test_a_prologue_cti_kernels_refuses_fails_the_layer(
    monkeypatch: pytest.MonkeyPatch, attribute: str, value: object, message: str
) -> None:
    from ctikernels.attention import _prologue

    # The real operation, which checks its variant before its device.
    monkeypatch.setattr("ctikernels.attention.inference.Prologue", _prologue.Prologue)

    with pytest.raises(ValueError, match=message):
        _layer(True, attribute, value)


def test_a_device_outside_the_fused_list_composes_the_prologue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ctikernels.attention import _prologue

    # The real operation, not this module's recording reference.
    monkeypatch.setattr("ctikernels.attention.inference.Prologue", _prologue.Prologue)
    monkeypatch.setattr(_prologue, "_launchers", {})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))

    geometry = PrologueGeometry(
        num_q_heads=8, num_kv_heads=2, eps=1e-6, head_dim=256, rotary_dim=64, zero_centered=True
    )
    operation = LayerServices().attention_prologue(
        torch.device("cuda", 0), geometry, required=True, norm_dtype=torch.bfloat16
    )

    assert operation.provider == "compose"
