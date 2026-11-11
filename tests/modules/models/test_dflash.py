from __future__ import annotations

import copy

import pytest
import torch

from cti.attention.paged import kv_pages, store_kv_pair
from cti.forward import InlineBreaks
from cti.layers import FeatureTaps
from cti.models import dflash
from cti.models.dflash import DFlash2DraftModel, DFlashConfig, DFlashDraftModel
from cti.models.dflash.config import DFlash2Extension
from tests.support import dflash2_reference
from tests.support.dflash import TINY_DRAFT_CONFIG as _DRAFT
from tests.support.dflash import qwen3_hf_config, target_engine_config
from tests.support.paged_attention import bind_model_state, kv_group
from tests.support.tp_shards import site

# The pinned Qwen3.8-27B DFlash2 draft's schema at the tiny geometry: Q8, non-causal
# sliding layers, four 16-channel conv segments and width-8 selector codebooks.
_DRAFT2 = {
    **{key: value for key, value in _DRAFT.items() if key != "block_size"},
    "architectures": ["DFlash2DraftModel"],
    "is_causal": False,
    "layer_types": ["sliding_attention"] * 5,
    "sliding_window": 2048,
    "dflash_config": {
        "block_size": 8,
        "mask_token_id": 290,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 8,
        "selector_top_k": 16,
    },
}


def _from_hf(document: dict) -> DFlashConfig:
    return DFlashConfig.from_hf(document, qwen3_hf_config(_DRAFT), target_engine_config())


def test_the_pinned_v1_draft_schema_resolves_against_its_target() -> None:
    config = _from_hf(_DRAFT)

    assert config.block_size == 16
    assert config.mask_token_id == 290
    assert config.target_layer_ids == (1, 9, 17, 25, 33)
    assert config.model.num_layers == 5
    assert DFlashDraftModel.CONFIG is DFlashConfig
    # Qwen3.5-style drafts declare Q inside dflash_config.
    nested = copy.deepcopy(_DRAFT)
    del nested["block_size"]
    nested["dflash_config"]["block_size"] = 16
    assert _from_hf(nested).block_size == 16
    assert config.layer_windows == (None,) * 5
    # Qwen3.5 drafts: causal sliding layers, then a full layer.
    sliding = {
        **_DRAFT,
        "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
        "sliding_window": 4096,
    }
    assert _from_hf(sliding).layer_windows == (4096,) * 4 + (None,)
    assert _from_hf(sliding).layer_masks == ((True, 4096),) * 4 + ((False, None),)
    # Any positive window, per layer.
    assert _from_hf({**sliding, "sliding_window": 1000}).layer_windows == (1000,) * 4 + (None,)


def test_the_pinned_dflash2_draft_schema_resolves_against_its_target() -> None:
    config = _from_hf(_DRAFT2)

    assert config.block_size == 8
    assert config.dflash2 == DFlash2Extension(2, 16, 8, 16)
    # Non-causal sliding layers: every block key, and context within the window.
    assert config.layer_masks == ((False, 2048),) * 5
    assert DFlash2DraftModel.CONFIG is DFlashConfig
    full = {**_DRAFT2, "layer_types": ["sliding_attention"] * 4 + ["full_attention"]}
    assert _from_hf(full).layer_masks == ((False, 2048),) * 4 + ((False, None),)


def _with(path: tuple[str, ...], value: object, draft: dict = _DRAFT) -> dict:
    document = copy.deepcopy(draft)
    fields = document
    for key in path[:-1]:
        fields = fields[key]
    if value is _MISSING:
        fields.pop(path[-1], None)
    else:
        fields[path[-1]] = value
    return document


_MISSING = object()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("architectures",), ["DFlash2DraftModel"]),
        *((("is_causal",), value) for value in (False, None, True)),
        # DFlash2 fields are invalid for v1 even when null.
        *((("dflash_config", "conv_kernel_size"), value) for value in (2, None)),
        *((("selector_top_k",), value) for value in (16, None)),
        (("dflash_config", "input_embedding_scale"), 2.0),
        (("final_logit_softcapping",), 30.0),
        (("layer_types",), ["sliding_attention"] * 5),
        (("layer_types",), ["full_attention"] * 4),
        (("layer_types",), ["linear_attention"] * 5),
        (("attention_bias",), True),
        (("num_target_layers",), 32),
        (("num_target_layers",), _MISSING),
        *((("block_size",), value) for value in (8, 32, "16", 16.0, _MISSING)),
        *((("dflash_config", "mask_token_id"), value) for value in (300, -1, _MISSING)),
        # Taps read the stream before the final norm: the last layer is not a tap.
        *(
            (("dflash_config", "target_layer_ids"), value)
            for value in ([1, 35], [36], [1, 1], [1.0], [], _MISSING)
        ),
        (("hidden_size",), 32),
    ],
)
def test_unimplemented_or_mismatched_drafts_are_rejected(
    path: tuple[str, ...], value: object
) -> None:
    document = _with(path, value)
    draft = qwen3_hf_config({**_DRAFT, "hidden_size": document["hidden_size"]})
    with pytest.raises(ValueError):
        DFlashConfig.from_hf(document, draft, target_engine_config())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        *((("is_causal",), value) for value in (True, None, 0, _MISSING)),
        *((("dflash_config", "block_size"), value) for value in (16, 8.0, _MISSING)),
        *(
            (("dflash_config", name), value)
            for name in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k")
            for value in (0, None, True, _MISSING)
        ),
        # Groups must divide the hidden size; K1 samples over 16 candidates.
        (("dflash_config", "conv_group_size"), 24),
        (("dflash_config", "selector_top_k"), 8),
        (("architectures",), ["DFlash3DraftModel"]),
    ],
)
def test_unimplemented_or_mismatched_dflash2_drafts_are_rejected(
    path: tuple[str, ...], value: object
) -> None:
    with pytest.raises(ValueError):
        _from_hf(_with(path, value, _DRAFT2))


@pytest.mark.parametrize("window", [0, -1, True, 4096.0, "4096", None])
def test_sliding_layers_need_a_positive_integer_window(window: object) -> None:
    layer_types = ["sliding_attention"] * 4 + ["full_attention"]
    with pytest.raises(ValueError, match="sliding_window"):
        _from_hf({**_DRAFT, "layer_types": layer_types, "sliding_window": window})


def test_taps_write_the_stream_after_selected_layers() -> None:
    buffer = torch.full((4, 6), -1.0, dtype=torch.bfloat16)
    taps = FeatureTaps((2, 0), buffer)
    stream = torch.tensor([[1.0, 2.0, 3.0]] * 3, dtype=torch.bfloat16)

    taps.store(1, stream)
    assert torch.equal(buffer, torch.full((4, 6), -1.0, dtype=torch.bfloat16))
    taps.store(0, stream)
    taps.store(2, stream * 2)

    assert torch.equal(buffer[:3, 3:], stream)
    assert torch.equal(buffer[:3, :3], stream * 2)
    # Rows past the forward's tokens keep their values.
    assert torch.equal(buffer[3], torch.full((6,), -1.0, dtype=torch.bfloat16))


def _rotate(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    half = x.shape[-1] // 2
    inv_freq = 1.0 / base ** (torch.arange(0, half, dtype=torch.float32) / half)
    angles = positions.float()[:, None, None] * inv_freq
    first, second = x[..., :half], x[..., half:]
    return torch.cat(
        (
            first * angles.cos() - second * angles.sin(),
            second * angles.cos() + first * angles.sin(),
        ),
        -1,
    )


def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight


def _reference_block(
    weights: dict, taps: torch.Tensor, embeddings: torch.Tensor, causal: tuple[bool, ...]
) -> torch.Tensor:
    """The v1 draft written out: block rows attend to context and block keys, all visible
    in a full layer, up to their own position in a sliding one (windows cover all keys)."""

    heads, kv_heads, head_dim = 2, 1, 64
    num_context, num_block = taps.shape[0], embeddings.shape[0]
    context = _rms(taps @ weights["fc.weight"].T, weights["hidden_norm.weight"])
    positions = torch.arange(num_context + num_block)
    h = embeddings
    for layer in range(5):
        w = {
            name.split(f"layers.{layer}.", 1)[1]: t
            for name, t in weights.items()
            if name.startswith(f"layers.{layer}.")
        }
        q_w, k_w, v_w = w["self_attn.qkv_proj.weight"].split([128, 64, 64])
        x = _rms(h, w["input_layernorm.weight"])
        q = _rms((x @ q_w.T).view(-1, heads, head_dim), w["self_attn.q_norm.weight"])
        keys = torch.cat((context, x)) @ k_w.T
        k = _rms(keys.view(-1, kv_heads, head_dim), w["self_attn.k_norm.weight"])
        v = (torch.cat((context, x)) @ v_w.T).view(-1, kv_heads, head_dim)
        q = _rotate(q, positions[num_context:], 1_000_000.0)
        k = _rotate(k, positions, 1_000_000.0)
        mask = positions[None] <= positions[num_context:, None] if causal[layer] else None
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), mask, enable_gqa=True
        )
        h = h + out.transpose(0, 1).reshape(num_block, -1) @ w["self_attn.o_proj.weight"].T
        x = _rms(h, w["post_attention_layernorm.weight"])
        gate, up = (x @ w["mlp.gate_up_proj.weight"].T).chunk(2, -1)
        h = h + (torch.nn.functional.silu(gate) * up) @ w["mlp.down_proj.weight"].T
    return _rms(h, weights["norm.weight"])


class _PagedAttention:
    """Block rows attend to all of their request's context and block keys, or, for a
    ``causal`` batch, to those at or before their own cache row."""

    SUPPORTS_KV_PROLOGUE = False

    def __init__(self, group: object) -> None:
        self.kv = kv_pages(group)

    def write_kv(self, k, v, layer_id, batch):
        slots = batch.attn_metadata.cache_write_indices[: k.shape[0]]
        store_kv_pair(self.kv[layer_id], k, v, slots)

    def forward(self, q, k, v, layer_id, batch):
        written = batch.attn_metadata.cache_write_indices
        store_kv_pair(self.kv[layer_id], k, v, written)
        rows = torch.arange(int(written.max()) + 1)
        keys, values = (cache.flatten(0, 1)[rows] for cache in self.kv[layer_id])
        mask = rows[None] <= written[:, None] if batch.causal else None
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1), mask, enable_gqa=True
        )
        return out.transpose(0, 1).contiguous()


def _bind_reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The draft's rotary, norm, activation and cache-store kernels on their references."""

    import sys
    from types import ModuleType

    from cti.layers import activation as activation_module
    from tests.support import reference_ops

    def rotate_key(key, cos_sin_cache, positions, *, head_dim):
        return reference_ops.apply_rotary_(
            key.clone(), key, cos_sin_cache, positions, head_dim=head_dim
        )[1]

    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_
    rotary.apply_rotary_key_ = rotate_key
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)
    reference_ops.bind_norm_references(monkeypatch)
    monkeypatch.setattr(
        activation_module.inference_activation, "silu_and_mul", reference_ops.silu_and_mul
    )
    monkeypatch.setattr("ctikernels.store_cache", reference_ops.store_cache)


@pytest.mark.parametrize("sliding", [False, True])
def test_the_draft_block_matches_the_v1_formula(
    monkeypatch: pytest.MonkeyPatch, sliding: bool
) -> None:
    from types import SimpleNamespace

    from cti.models.dflash.model import DFlashDraftModel
    from tests.support.tp_shards import default_dtype

    _bind_reference_kernels(monkeypatch)
    torch.manual_seed(0)
    layer_types = ["sliding_attention"] * 4 + ["full_attention"] if sliding else None
    config = _from_hf({**_DRAFT, "layer_types": layer_types, "sliding_window": 4096})
    draft_site = site()
    with torch.device("meta"), default_dtype(torch.float32):
        model = DFlashDraftModel(config, round_width=config.block_size, site=draft_site)
    weights = {
        name: torch.randn(tensor.shape) * (1.0 if name.endswith("norm.weight") else 0.1)
        for name, tensor in model.state_dict().items()
    }
    model.load_state_dict(dict(weights))
    bind_model_state(model, _PagedAttention(kv_group(5, 1, 64, 16)))
    taps = torch.randn(6, 5 * 64)
    embeddings = torch.randn(4, 64)

    positions = torch.arange(10)
    context = SimpleNamespace(
        attn_metadata=SimpleNamespace(cache_write_indices=torch.arange(6, dtype=torch.int32))
    )
    model.store_context(taps, positions[:6], context)
    rows = {
        "rope_positions": positions[6:],
        "attn_metadata": SimpleNamespace(
            cache_write_indices=torch.arange(6, 10, dtype=torch.int32)
        ),
    }
    causal = tuple(causal for causal, _ in config.layer_masks)
    assert causal == ((True,) * 4 + (False,) if sliding else (False,) * 5)
    blocks = tuple(
        SimpleNamespace(**rows, causal=layer_causal, graph_breaks=InlineBreaks())
        for layer_causal in causal
    )
    hidden = model.forward(blocks, embeddings.clone())
    torch.testing.assert_close(
        hidden, _reference_block(weights, taps, embeddings, causal), rtol=1e-4, atol=1e-4
    )


def test_the_draft_layers_share_one_prologue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import ModuleType

    from cti.layers import LayerServices
    from cti.models.dflash.model import DFlashDraftModel
    from tests.support import reference_ops
    from tests.support.tp_shards import default_dtype

    class Prologue(reference_ops.Prologue):
        provider = "cute"

    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)
    monkeypatch.setattr("ctikernels.attention.inference.Prologue", Prologue)
    with torch.device("meta"), default_dtype(torch.bfloat16):
        config = _from_hf(_DRAFT)
        model = DFlashDraftModel(
            config, round_width=config.block_size, site=site(services=LayerServices())
        )

    prologues = {layer.prologue for layer in model.iter_attention_layers()}
    assert len(prologues) == 1 and isinstance(prologues.pop(), Prologue)


@pytest.mark.parametrize("document", [_DRAFT, _DRAFT2], ids=["v1", "dflash2"])
def test_every_draft_gemm_call_site_is_warmed_up(
    monkeypatch: pytest.MonkeyPatch, document: dict
) -> None:
    from cti.layers import prepare_inference
    from tests.support.gemm import GemmRecorder
    from tests.support.tp_shards import default_dtype

    _bind_reference_kernels(monkeypatch)
    recorder = GemmRecorder(monkeypatch)
    config = _from_hf(document)
    with torch.device("meta"), default_dtype(torch.float32):
        model = getattr(dflash, document["architectures"][0])(
            config, round_width=config.block_size, site=site()
        )
    model.load_state_dict(
        {name: torch.randn(tensor.shape) for name, tensor in model.state_dict().items()}
    )

    prepare_inference(
        (model,),
        max_rows=1,
        max_sequences=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # Each layer's K/V-only projection of context rows is a call site too.
    assert len(recorder.operations) == len(list(model.inference_gemms()))
    assert all(operation.calls for operation in recorder.operations)


class _BlockAttention:
    """DFlash2's mask: a block query at position p sees every key from ``block_start`` on,
    and earlier keys k with p - k < its batch's ``window``; cache rows are positions."""

    SUPPORTS_KV_PROLOGUE = False

    def __init__(self, group: object, block_start: int) -> None:
        self.kv = kv_pages(group)
        self.block_start = block_start

    def write_kv(self, k, v, layer_id, batch):
        store_kv_pair(self.kv[layer_id], k, v, batch.cache_write_indices[: k.shape[0]])

    def forward(self, q, k, v, layer_id, batch):
        assert not batch.causal
        store_kv_pair(self.kv[layer_id], k, v, batch.cache_write_indices)
        rows = torch.arange(int(batch.cache_write_indices.max()) + 1)
        keys, values = (cache.flatten(0, 1)[rows] for cache in self.kv[layer_id])
        queries = batch.cache_write_indices.long()[:, None]
        mask = (rows[None] >= self.block_start) | (queries - rows[None] < batch.window)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1), mask, enable_gqa=True
        )
        return out.transpose(0, 1).contiguous()


def _dflash2_weights(model) -> dict:
    generator = torch.Generator().manual_seed(1)
    weights = {}
    for name, tensor in model.state_dict().items():
        scale = 1.0 if name.endswith("norm.weight") else 0.1
        weights[name] = torch.randn(tensor.shape, generator=generator) * scale
        if name.endswith("base_kernel"):
            weights[name] += 1.0
    return weights


# A round's rows: fewer than, as many as and more than the trained block of 8, whose
# convolution restarts at each runtime block, as SGLang and vLLM run it.
@pytest.mark.parametrize("width", [4, 8, 17])
def test_the_dflash2_block_matches_the_reference(
    monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    from types import SimpleNamespace

    from cti.models.dflash.dflash2 import DFlash2DraftModel
    from tests.support.tp_shards import default_dtype

    _bind_reference_kernels(monkeypatch)
    window, num_context = 3, 6
    config = _from_hf({**_DRAFT2, "sliding_window": window})
    draft_site = site()
    with torch.device("meta"), default_dtype(torch.float32):
        model = DFlash2DraftModel(config, round_width=width, site=draft_site)
    weights = _dflash2_weights(model)
    model.load_state_dict(dict(weights))
    bind_model_state(model, _BlockAttention(kv_group(5, 1, 64, 32), num_context))
    generator = torch.Generator().manual_seed(2)
    taps = torch.randn(num_context, 5 * 64, generator=generator)
    embeddings = torch.randn(width, 64, generator=generator)

    positions = torch.arange(num_context + width)
    context = SimpleNamespace(cache_write_indices=torch.arange(num_context, dtype=torch.int32))
    model.store_context(taps, positions[:num_context], context)
    block = SimpleNamespace(
        rope_positions=positions[num_context:],
        cache_write_indices=torch.arange(num_context, num_context + width, dtype=torch.int32),
        causal=False,
        window=window,
        graph_breaks=InlineBreaks(),
    )
    hidden = model.forward((block,) * 5, embeddings.clone())

    expected = dflash2_reference.draft_block(
        weights,
        taps,
        embeddings,
        heads=2,
        kv_heads=1,
        head_dim=64,
        window=window,
        rope_base=1_000_000.0,
        conv_taps=2,
    )
    torch.testing.assert_close(hidden.double(), expected, rtol=1e-4, atol=1e-4)
