"""The optional fused attention prologue of 128-wide heads (Qwen3, the DFlash draft): when a
layer takes it, and that the layer runs it through the backend like the composed path."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

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
from tests.support import reference_ops
from tests.support.graph_breaks import PaddedBreaks
from tests.support.paged_attention import kv_group
from tests.support.tp_shards import site

CPU = torch.device("cpu")
HEAD_DIM = 128
QKV_WIDTH = 4 * HEAD_DIM  # two query heads over one KV head


class _CuTePrologue(reference_ops.Prologue):
    provider = "cute"
    constructed: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.constructed.append(kwargs)


@pytest.fixture(autouse=True)
def _references(monkeypatch: pytest.MonkeyPatch) -> None:
    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)
    reference_ops.bind_norm_references(monkeypatch)
    monkeypatch.setattr("ctikernels.store_cache", reference_ops.store_cache)
    monkeypatch.setattr("ctikernels.attention.inference.Prologue", _CuTePrologue)
    _CuTePrologue.constructed.clear()


class _PoolBackend:
    """Writes K and V to ``pool`` either way and returns Q."""

    def __init__(self, pool: PageGroup, *, prologues: bool = True) -> None:
        self.SUPPORTS_KV_PROLOGUE = prologues
        self.kv = kv_pages(pool)
        self.calls: list[str] = []

    def forward(self, q, k, v, layer_id, batch):
        self.calls.append("forward")
        store_kv_pair(self.kv[layer_id], k, v, batch.attn_metadata.cache_write_indices)
        return q

    def forward_into(self, q, k, v, layer_id, batch, out):
        out.copy_(self.forward(q, k, v, layer_id, batch))

    def forward_prologue(self, prologue, layer_id, batch, out=None):
        self.calls.append("forward_prologue")
        k_cache, v_cache = (cache.flatten(0, 1) for cache in self.kv[layer_id])
        q = prologue(k_cache, v_cache, batch.attn_metadata.cache_write_indices)
        return q if out is None else out.copy_(q)


def _pool() -> PageGroup:
    pool = kv_group(1, 1, HEAD_DIM, 4, dtype=torch.bfloat16, page_size=4)
    pool.layers[0].zero_()
    return pool


def _layer(
    fused: bool = True,
    *,
    backend: _PoolBackend | None = None,
    services: LayerServices | None = None,
    k_norm: bool = True,
    eps: tuple[float, float] = (1e-6, 1e-6),
    norm_width: int = HEAD_DIM,
    norm_dtype: torch.dtype = torch.bfloat16,
) -> AttentionLayer:
    """A Qwen3-like layer: standard Q/K RMSNorm and full NeoX rotary of 128-wide heads; the
    composed base layer unless ``fused``."""
    generator = torch.Generator().manual_seed(0)
    norms = (RMSNorm(HEAD_DIM, eps[0]), RMSNorm(norm_width, eps[1]))
    for norm in norms:
        weight = 1 + 0.1 * torch.randn(norm.weight.shape[0], generator=generator)
        norm.weight = weight.to(norm_dtype)
    layer_site = site(services=services or LayerServices())
    layer = AttentionLayer(
        qkv=LinearQKVMerged(
            hidden_size=8,
            head_dim=HEAD_DIM,
            num_qo_heads=2,
            num_kv_heads=1,
            has_bias=False,
            site=layer_site,
        ),
        rotary=RotaryEmbedding.create(HEAD_DIM, HEAD_DIM, 64, 1e6, site=layer_site),
        q_norm=norms[0],
        k_norm=norms[1] if k_norm else None,
        prologue=PrologueFactory() if fused else None,
    )
    if backend is not None:
        layer.bind_state({"kv": PagedSlot(backend, 0)})
    return layer


@pytest.mark.parametrize("prologues", [True, False], ids=["fused", "backend-without"])
def test_fused_prologue_matches_the_composed_layer(prologues: bool) -> None:
    batch = SimpleNamespace(
        graph_breaks=InlineBreaks(),
        rope_positions=torch.tensor([3, 4, 9]),
        attn_metadata=SimpleNamespace(
            cache_write_indices=torch.tensor([5, 2, 9], dtype=torch.int32)
        ),
    )
    projection = torch.randn(3, QKV_WIDTH, generator=torch.Generator().manual_seed(1)).bfloat16()
    outputs, pools, backends = [], [], []
    for fused in (True, False):
        pool = _pool()
        backend = _PoolBackend(pool, prologues=prologues)
        layer = _layer(fused, backend=backend)
        outputs.append(layer.forward(projection.clone(), batch))
        pools.append(pool)
        backends.append(backend)

    (constructed,) = _CuTePrologue.constructed
    assert constructed == {
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "eps": 1e-6,
        "device": CPU,
        "head_dim": HEAD_DIM,
        "rotary_dim": HEAD_DIM,
        "zero_centered": False,
        "position_axes": None,
    }
    assert backends[0].calls == ["forward_prologue" if prologues else "forward"]
    assert backends[1].calls == ["forward"]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(pools[0].layers[0], pools[1].layers[0], rtol=0, atol=0)
    assert pools[0].layers[0][0].flatten(0, 1)[[5, 2, 9]].any()


def _compose_provider(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(_CuTePrologue, "provider", "compose")
    return {}


def _raise(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    def refuse(**_kwargs: object) -> None:
        raise ValueError("unsupported (head_dim, rotary_dim, zero_centered)")

    monkeypatch.setattr("ctikernels.attention.inference.Prologue", refuse)
    return {}


@pytest.mark.parametrize(
    "unfusable",
    [
        _compose_provider,
        _raise,
        lambda monkeypatch: {"k_norm": False},
        lambda monkeypatch: {"eps": (1e-6, 1e-5)},
        lambda monkeypatch: {"norm_width": 64},
        lambda monkeypatch: {"norm_dtype": torch.float16},
    ],
    ids=["compose", "refused", "no-k-norm", "q-k-eps", "norm-width", "fp16"],
)
def test_a_layer_keeps_the_composition_unless_a_cute_prologue_runs_it(
    monkeypatch: pytest.MonkeyPatch, unfusable
) -> None:
    assert _layer(**unfusable(monkeypatch)).prologue is None


def test_qwen3_layers_share_one_prologue(tmp_path) -> None:
    from tests.support.tiny_configs import tiny_hf_config
    from tests.support.tp_shards import build_model

    model = build_model(
        tiny_hf_config("Qwen3ForCausalLM", tmp_path),
        dtype=torch.bfloat16,
        services=LayerServices(),
    )

    prologues = {id(layer.prologue) for layer in model.iter_attention_layers()}
    assert len(prologues) == 1 and len(_CuTePrologue.constructed) == 1


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "composed"])
def test_padded_breaks_attend_what_inline_breaks_attend(fused: bool) -> None:
    projection = torch.randn(3, QKV_WIDTH, generator=torch.Generator().manual_seed(1)).bfloat16()
    outputs, pools, calls = [], [], []
    for breaks in (InlineBreaks(), PaddedBreaks(3)):
        batch = SimpleNamespace(
            rope_positions=torch.tensor([3, 4, 9]),
            attn_metadata=SimpleNamespace(
                cache_write_indices=torch.tensor([5, 2, 9], dtype=torch.int32)
            ),
            graph_breaks=breaks,
        )
        pool = _pool()
        backend = _PoolBackend(pool)
        layer = _layer(fused, backend=backend)
        outputs.append(layer.forward(projection.clone(), batch))
        pools.append(pool)
        calls.append(backend.calls)

    # Breakable graphs' eager attention matches eager forwards bit for bit.
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(pools[0].layers[0], pools[1].layers[0])
    assert calls[0] == calls[1] == ["forward_prologue" if fused else "forward"]


# Fusion is pinned per device, geometry and norm dtype: the CuTe kernel's devices.
_FUSED_DEVICES = {
    ("NVIDIA GeForce RTX 5090", (12, 0)): True,
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0)): True,
    ("NVIDIA GB10", (12, 1)): True,
    ("NVIDIA H100 80GB HBM3", (9, 0)): False,
    ("NVIDIA B200", (10, 0)): False,
}
# (head_dim, rotary_dim, zero_centered): Qwen3 and the DFlash draft, Qwen3.5.
_GEOMETRIES = {"qwen3": (128, 128, False), "qwen3_5": (256, 64, True)}


@pytest.mark.parametrize("required", [False, True], ids=["optional", "required"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=str)
@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=str)
@pytest.mark.parametrize("hardware", list(_FUSED_DEVICES), ids=lambda h: f"{h[0]}-sm{h[1]}")
def test_the_prologue_fuses_only_on_the_cute_devices_with_bf16_norms(
    monkeypatch: pytest.MonkeyPatch,
    hardware: tuple[str, tuple[int, int]],
    geometry: str,
    dtype: torch.dtype,
    required: bool,
) -> None:
    from ctikernels.attention import _prologue

    name, capability = hardware
    monkeypatch.setattr("ctikernels.attention.inference.Prologue", _prologue.Prologue)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    # The kernel's own device table, without compiling: a launcher wherever it lists the device.
    monkeypatch.setattr(
        _prologue,
        "load",
        lambda device, variant: object() if hardware in _prologue.DEVICES[variant] else None,
    )
    head_dim, rotary_dim, zero_centered = _GEOMETRIES[geometry]
    norm = SimpleNamespace(
        weight=torch.ones(head_dim, dtype=dtype), eps=1e-6, zero_centered=zero_centered
    )
    layer = SimpleNamespace(
        q_norm=norm,
        k_norm=norm,
        head_dim=head_dim,
        num_qo_heads=8,
        num_kv_heads=2,
        rotary=SimpleNamespace(head_dim=head_dim, rotary_dim=rotary_dim, position_axes=None),
    )
    cuda = SimpleNamespace(device=torch.device("cuda", 0), services=LayerServices())
    # A fused Prologue allocates its unread scale arguments on the device; this CPU test
    # judges only the device table, so they stay on the host.
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, device=None, **kwargs: empty(*args, **kwargs))

    operation = PrologueFactory(required=required).build(layer, cuda)

    fused = _FUSED_DEVICES[hardware]
    if required:
        assert operation.provider == ("cute" if fused else "compose")
    else:
        assert (operation is not None) is (fused and dtype == torch.bfloat16)
        assert operation is None or operation.provider == "cute"


@pytest.mark.parametrize("required", [False, True], ids=["optional", "required"])
def test_a_cpu_layer_takes_no_optional_prologue_and_fails_a_required_one(
    monkeypatch: pytest.MonkeyPatch, required: bool
) -> None:
    from ctikernels.attention import _prologue

    monkeypatch.setattr("ctikernels.attention.inference.Prologue", _prologue.Prologue)
    layer = _layer(False)
    cpu = SimpleNamespace(device=CPU, services=LayerServices())
    factory = PrologueFactory(required=required)

    if required:
        with pytest.raises(ValueError, match="CUDA device"):
            factory.build(layer, cpu)
    else:
        assert factory.build(layer, cpu) is None
