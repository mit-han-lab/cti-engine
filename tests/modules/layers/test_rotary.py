# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb as qwen3_5_apply_rotary_pos_emb,
)

import cti.layers.rotary as rotary_module
from cti.models.qwen3 import config as qwen3_config
from tests.support import reference_ops
from tests.support.tp_shards import site
from tests.support.training_ops import bind_training_references

# Qwen3.5 text attention: 64 of 256 head channels rotate, M-RoPE sections (T, H, W).
QWEN3_5_HEAD_DIM = 256
QWEN3_5_ROTARY_DIM = 64
QWEN3_5_ROPE_THETA = 10_000_000
QWEN3_5_MROPE_SECTIONS = (11, 11, 10)
MAX_POSITION = 512


@pytest.fixture
def rotary_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_forward(*_args, **_kwargs):
        pytest.fail("constructing a rotary cache must not execute the operation")

    operation = ModuleType("ctikernels.rotary.inference")
    operation.apply_rotary_ = unexpected_forward
    monkeypatch.setitem(sys.modules, operation.__name__, operation)


@pytest.fixture
def reference_rotary(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    """Bind the CPU reference and record each call's cache, positions and axis map."""
    calls: list[tuple[object, ...]] = []

    def apply_rotary_(query, key, cos_sin_cache, positions, *, head_dim, position_axes=None):
        calls.append((cos_sin_cache, positions, position_axes))
        return reference_ops.apply_rotary_(
            query, key, cos_sin_cache, positions, head_dim=head_dim, position_axes=position_axes
        )

    operation = ModuleType("ctikernels.rotary.inference")
    operation.apply_rotary_ = apply_rotary_
    monkeypatch.setitem(sys.modules, operation.__name__, operation)
    return calls


def test_rotary_tables_live_on_the_site_device(rotary_operation: None) -> None:
    with torch.device("meta"):
        rotary = rotary_module.RotaryEmbedding.create(
            64, 64, 128, 10_000, {"rope_type": "default"}, site=site()
        )

    assert rotary.cos_sin_cache.device == torch.device("cpu")


def _yarn(**parameters: object) -> dict[str, object]:
    return {"rope_type": "yarn", **parameters}


def _rotate_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply half-width cos/sin the way the inference and training RoPE kernels do."""

    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


@pytest.mark.parametrize(
    ("rope_parameters", "max_position_embeddings", "tolerance"),
    [
        pytest.param({"rope_type": "default"}, 16_384, 0.0, id="default"),
        # The same llama3 formula, evaluated in a different order, differs from
        # Hugging Face only by fp32 rounding (measured below 2e-6 here).
        pytest.param(
            {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8_192,
            },
            16_384,
            1e-5,
            id="llama3",
        ),
        pytest.param(
            {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 2.0,
                "high_freq_factor": 2.0,
                "original_max_position_embeddings": 8_192,
            },
            16_384,
            1e-5,
            id="llama3-step",
        ),
        # Qwen3's long-context recipe, with max_position_embeddings at the extended
        # length Hugging Face's YaRN validation expects.
        pytest.param(
            _yarn(factor=4.0, original_max_position_embeddings=32_768),
            131_072,
            0.0,
            id="yarn-qwen3",
        ),
        pytest.param(
            _yarn(
                factor=8.0,
                original_max_position_embeddings=4_096,
                mscale=1.0,
                mscale_all_dim=0.8,
            ),
            16_384,
            0.0,
            id="yarn-mscale-pair",
        ),
        pytest.param(
            _yarn(factor=4.0, original_max_position_embeddings=4_096, mscale=0.5),
            16_384,
            0.0,
            id="yarn-lone-mscale",
        ),
        # A zero mscale counts as unset, so Hugging Face ignores the pair.
        pytest.param(
            _yarn(
                factor=4.0,
                original_max_position_embeddings=4_096,
                mscale=0.0,
                mscale_all_dim=1.0,
            ),
            16_384,
            0.0,
            id="yarn-zero-mscale",
        ),
        pytest.param(
            _yarn(factor=4.0, original_max_position_embeddings=4_096, attention_factor=1.5),
            16_384,
            0.0,
            id="yarn-attention-factor",
        ),
        pytest.param(
            _yarn(
                rope_theta=10_000.0,
                factor=2.5,
                original_max_position_embeddings=65_536,
                beta_fast=16.0,
                beta_slow=2.0,
                truncate=False,
            ),
            16_384,
            0.0,
            id="yarn-untruncated-betas",
        ),
        # Zero betas count as unset and fall back to 32 and 1.
        pytest.param(
            _yarn(
                factor=4.0,
                original_max_position_embeddings=32_768,
                beta_fast=0.0,
                beta_slow=0.0,
            ),
            16_384,
            0.0,
            id="yarn-zero-betas",
        ),
        # The slow correction dimension lies past head_dim / 2.
        pytest.param(
            _yarn(rope_theta=10_000.0, factor=2.0, original_max_position_embeddings=131_072),
            16_384,
            0.0,
            id="yarn-wide-ramp",
        ),
        # Both betas put the correction dimension at exactly 20, so the ramp has
        # zero width and Hugging Face widens it to keep it finite.
        pytest.param(
            _yarn(
                factor=3.0,
                original_max_position_embeddings=8_192,
                beta_fast=17.386416344905864,
                beta_slow=17.386416344905864,
            ),
            16_384,
            0.0,
            id="yarn-zero-width-ramp",
        ),
        pytest.param(
            _yarn(factor=None, original_max_position_embeddings=4_096),
            16_384,
            0.0,
            id="yarn-factor-from-lengths",
        ),
    ],
)
def test_rotary_matches_hugging_face_qwen3_rope(
    rotary_operation: None,
    rope_parameters: dict[str, object],
    max_position_embeddings: int,
    tolerance: float,
) -> None:
    head_dim = 128
    half = head_dim // 2
    hf_config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=2 * head_dim,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_parameters={"rope_theta": 1_000_000.0, **rope_parameters},
    )
    rotary = qwen3_config.Qwen3Config.from_hf(hf_config).rotary_config
    embedding = rotary_module.RotaryEmbedding.create(
        rotary.head_dim,
        rotary.rotary_dim,
        rotary.max_position,
        rotary.base,
        rotary.scaling,
        site=site(training=True),
    )

    # Hugging Face returns cos/sin over both rotate_half halves, including any
    # attention scaling of the RoPE type; the engine caches one half of each.
    cos, sin = Qwen3RotaryEmbedding(hf_config)(
        torch.empty(0),
        torch.arange(rotary.max_position)[None],
    )
    expected_cos, expected_sin = cos[0, :, :half], sin[0, :, :half]
    exact = {"rtol": 0, "atol": tolerance}
    torch.testing.assert_close(
        embedding.cos_sin_cache,
        torch.cat((expected_cos, expected_sin), dim=-1),
        **exact,
    )
    torch.testing.assert_close(embedding._training_cos_cache, expected_cos, **exact)
    torch.testing.assert_close(embedding._training_sin_cache, expected_sin, **exact)

    generator = torch.Generator().manual_seed(0)
    positions = torch.randint(rotary.max_position, (256,), generator=generator)
    # Values in [-1, 1) bound each rotated element's error by 2 * tolerance.
    query, key = torch.rand(2, 1, 2, positions.numel(), head_dim, generator=generator) * 2 - 1
    expected_query, expected_key = apply_rotary_pos_emb(
        query,
        key,
        cos[:, positions],
        sin[:, positions],
    )
    for cache_cos, cache_sin in (
        embedding.cos_sin_cache[positions].chunk(2, dim=-1),
        (embedding._training_cos_cache[positions], embedding._training_sin_cache[positions]),
    ):
        applied = {"rtol": 0, "atol": 2 * tolerance}
        torch.testing.assert_close(
            _rotate_neox(query, cache_cos, cache_sin),
            expected_query,
            **applied,
        )
        torch.testing.assert_close(_rotate_neox(key, cache_cos, cache_sin), expected_key, **applied)


@pytest.mark.parametrize("rope_type", ["linear", "dynamic", "longrope", "unknown"])
def test_rotary_rejects_unimplemented_rope_types(rope_type: str) -> None:
    with pytest.raises(ValueError, match=f"unsupported rope_type '{rope_type}'"):
        rotary_module.RotaryEmbedding.create(
            64, 64, 128, 10_000, {"rope_type": rope_type, "factor": 4.0}, site=site()
        )


def test_rotary_only_keeps_training_layout_for_training_models(
    rotary_operation: None,
) -> None:
    serving = rotary_module.RotaryEmbedding.create(64, 64, 128, 10_000, site=site())
    training = rotary_module.RotaryEmbedding.create(64, 64, 128, 10_000, site=site(training=True))

    assert serving._training_cos_cache is None
    assert serving._training_sin_cache is None
    assert serving.cos_sin_cache.is_contiguous()
    assert training._training_cos_cache is not None
    assert training._training_sin_cache is not None
    assert training._training_cos_cache.is_contiguous()
    assert training._training_sin_cache.is_contiguous()
    assert training.cos_sin_cache.is_contiguous()
    torch.testing.assert_close(training.cos_sin_cache, serving.cos_sin_cache, rtol=0, atol=0)
    assert training._training_cos_cache.untyped_storage().data_ptr() != (
        training.cos_sin_cache.untyped_storage().data_ptr()
    )
    assert training._training_sin_cache.untyped_storage().data_ptr() != (
        training.cos_sin_cache.untyped_storage().data_ptr()
    )


def _qwen3_5_rotary(
    *,
    mrope: bool,
    rope_scaling: dict[str, object] | None = None,
) -> rotary_module.RotaryEmbedding:
    return rotary_module.RotaryEmbedding.create(
        QWEN3_5_HEAD_DIM,
        QWEN3_5_ROTARY_DIM,
        MAX_POSITION,
        QWEN3_5_ROPE_THETA,
        rope_scaling,
        mrope_sections=QWEN3_5_MROPE_SECTIONS if mrope else None,
        site=site(),
    )


def _hf_qwen3_5_rotary(**rope_parameters: object) -> Qwen3_5TextRotaryEmbedding:
    config = Qwen3_5TextConfig(
        hidden_size=QWEN3_5_HEAD_DIM,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=QWEN3_5_HEAD_DIM,
        max_position_embeddings=MAX_POSITION,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": QWEN3_5_ROPE_THETA,
            "partial_rotary_factor": QWEN3_5_ROTARY_DIM / QWEN3_5_HEAD_DIM,
            "mrope_section": list(QWEN3_5_MROPE_SECTIONS),
            "mrope_interleaved": True,
            **rope_parameters,
        },
    )
    return Qwen3_5TextRotaryEmbedding(config)


def _merged_qkv_gate(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the projection and its strided Q (4 heads) and K (2 heads) slices."""
    torch.manual_seed(3)
    head_dim = QWEN3_5_HEAD_DIM
    projection = torch.randn(num_tokens, (4 + 2 + 2 + 4) * head_dim)
    return projection, projection[:, : 4 * head_dim], projection[:, 4 * head_dim : 6 * head_dim]


def _hf_rotated(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    **rope_parameters: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF rotation of ``[tokens, heads * head_dim]`` Q/K; ``position_ids`` as HF takes them."""
    cos, sin = _hf_qwen3_5_rotary(**rope_parameters)(query, position_ids)
    tokens = query.shape[0]

    def heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(1, tokens, -1, QWEN3_5_HEAD_DIM).transpose(1, 2)

    rotated_query, rotated_key = qwen3_5_apply_rotary_pos_emb(heads(query), heads(key), cos, sin)
    return (
        rotated_query.transpose(1, 2).reshape(tokens, -1),
        rotated_key.transpose(1, 2).reshape(tokens, -1),
    )


def _image_prompt_positions() -> tuple[torch.Tensor, int]:
    """Three-axis positions of text, a 2x3 merged image grid and text, plus the decode offset.

    Rewritten from the Qwen3.5 rule (transformers 5.16.1 ``Qwen3_5Model.get_rope_index``):
    image tokens sit at ``start + (t, h, w)`` and the next text resumes at
    ``start + max(h, w)``.
    """
    rows = [[position] * 3 for position in range(4)]
    image_start = 4
    rows += [[image_start, image_start + h, image_start + w] for h in range(2) for w in range(3)]
    next_text = image_start + 3
    rows += [[position] * 3 for position in range(next_text, next_text + 2)]
    prompt_length = len(rows)
    return torch.tensor(rows).T.contiguous(), next_text + 2 - prompt_length


def test_partial_rotary_matches_hf_for_text_positions(
    reference_rotary: list[tuple[object, ...]],
) -> None:
    rotary = _qwen3_5_rotary(mrope=False)
    projection, query, key = _merged_qkv_gate(6)
    positions = torch.arange(37, 43)
    expected_query, expected_key = _hf_rotated(query.clone(), key.clone(), positions[None])
    untouched = projection[:, 6 * QWEN3_5_HEAD_DIM :].clone()

    rotary.forward(positions, query, key)

    assert type(rotary) is rotary_module.RotaryEmbedding
    assert rotary.cos_sin_cache.shape == (MAX_POSITION, QWEN3_5_ROTARY_DIM)
    torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(projection[:, 6 * QWEN3_5_HEAD_DIM :], untouched, rtol=0, atol=0)
    pass_through = query.view(6, 4, QWEN3_5_HEAD_DIM)[..., QWEN3_5_ROTARY_DIM:]
    original = _merged_qkv_gate(6)[1].view(6, 4, QWEN3_5_HEAD_DIM)[..., QWEN3_5_ROTARY_DIM:]
    torch.testing.assert_close(pass_through, original, rtol=0, atol=0)


@pytest.mark.parametrize("mrope", [False, True], ids=("text", "multi-axis"))
def test_partial_rotary_applies_yarn_over_the_rotated_width(
    reference_rotary: list[tuple[object, ...]],
    mrope: bool,
) -> None:
    # Hugging Face computes the YaRN ramp over head_dim * partial_rotary_factor and
    # scales cos and sin, gathered per axis for M-RoPE, by the attention factor.
    yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 128}
    rotary = _qwen3_5_rotary(mrope=mrope, rope_scaling=yarn)
    positions = _image_prompt_positions()[0] if mrope else torch.arange(37, 43)
    hf_positions = positions[:, None] if mrope else positions[None]
    _, query, key = _merged_qkv_gate(positions.shape[-1])
    expected_query, expected_key = _hf_rotated(query.clone(), key.clone(), hf_positions, **yarn)

    rotary.forward(positions, query, key)

    torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(key, expected_key, rtol=0, atol=0)


def test_multi_axis_rotary_matches_hf_interleaved_mrope_with_decode_offsets(
    reference_rotary: list[tuple[object, ...]],
) -> None:
    rotary = _qwen3_5_rotary(mrope=True)
    prompt, decode_offset = _image_prompt_positions()
    # Two decode steps continue the prompt: sequence index + offset on every axis.
    sequence_indices = torch.arange(prompt.shape[1], prompt.shape[1] + 2)
    positions = torch.cat((prompt, (sequence_indices + decode_offset).expand(3, -1)), dim=1)
    _, query, key = _merged_qkv_gate(positions.shape[1])
    expected_query, expected_key = _hf_rotated(query.clone(), key.clone(), positions[:, None])

    rotary.forward(positions, query, key)

    assert isinstance(rotary, rotary_module.MultiAxisRotaryEmbedding)
    assert decode_offset == -3
    torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(key, expected_key, rtol=0, atol=0)


def test_multi_axis_rotary_reduces_to_one_dimensional_rotary(
    reference_rotary: list[tuple[object, ...]],
) -> None:
    single_axis = _qwen3_5_rotary(mrope=False)
    multi_axis = _qwen3_5_rotary(mrope=True)
    positions = torch.tensor([0, 5, 9, 200, 511])
    _, base_query, base_key = _merged_qkv_gate(5)
    single_axis.forward(positions, base_query, base_key)

    _, query, key = _merged_qkv_gate(5)
    multi_axis.forward(positions, query, key)
    one_dimensional_call = reference_rotary[-1]
    _, equal_rows_query, equal_rows_key = _merged_qkv_gate(5)
    multi_axis.forward(positions.expand(3, -1), equal_rows_query, equal_rows_key)

    # A 1-D input is today's operation call: the shared cache and the positions as given;
    # the axis map rides along for the operation to pick each frequency's axis.
    assert one_dimensional_call[0] is multi_axis.cos_sin_cache
    assert one_dimensional_call[1] is positions
    assert one_dimensional_call[2] == multi_axis.position_axes
    assert reference_rotary[0][2] is None and single_axis.position_axes is None
    torch.testing.assert_close(multi_axis.cos_sin_cache, single_axis.cos_sin_cache, rtol=0, atol=0)
    for actual in (query, equal_rows_query):
        torch.testing.assert_close(actual, base_query, rtol=0, atol=0)
    for actual in (key, equal_rows_key):
        torch.testing.assert_close(actual, base_key, rtol=0, atol=0)


def test_multi_axis_rotary_interleaves_qwen3_5_sections(rotary_operation: None) -> None:
    rotary = _qwen3_5_rotary(mrope=True)
    expected_axis = [0] * 32
    expected_axis[1:33:3] = [1] * 11
    expected_axis[2:30:3] = [2] * 10

    assert rotary.position_axes == tuple(expected_axis)


@pytest.mark.parametrize(
    "sections",
    [(11, 11, 11), (16, 16), (11, 11, 10, 0), (2, 15, 15)],
    ids=("oversized", "two-axes", "four-axes", "height-overflows"),
)
def test_multi_axis_rotary_rejects_sections_that_do_not_interleave(
    rotary_operation: None,
    sections: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="do not interleave"):
        rotary_module.MultiAxisRotaryEmbedding(
            QWEN3_5_HEAD_DIM,
            QWEN3_5_ROTARY_DIM,
            MAX_POSITION,
            torch.ones(QWEN3_5_ROTARY_DIM // 2),
            sections=sections,
        )


@pytest.mark.parametrize("rotary_dim", [0, 63, 258])
def test_rotary_rejects_invalid_rotary_width(rotary_operation: None, rotary_dim: int) -> None:
    with pytest.raises(ValueError, match="rotary_dim"):
        rotary_module.RotaryEmbedding(
            QWEN3_5_HEAD_DIM, rotary_dim, MAX_POSITION, torch.ones(QWEN3_5_ROTARY_DIM // 2)
        )


def test_rotary_rejects_unsupported_head_dim(rotary_operation: None) -> None:
    with pytest.raises(ValueError, match="head_dim 72"):
        rotary_module.RotaryEmbedding(72, 64, MAX_POSITION, torch.ones(32))


@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
def test_partial_multi_axis_rotary_trains_like_hf_text_rotary(
    rotary_operation: None,
    monkeypatch: pytest.MonkeyPatch,
    packed: bool,
) -> None:
    bind_training_references(monkeypatch)
    rotary = rotary_module.RotaryEmbedding.create(
        QWEN3_5_HEAD_DIM,
        QWEN3_5_ROTARY_DIM,
        MAX_POSITION,
        QWEN3_5_ROPE_THETA,
        mrope_sections=QWEN3_5_MROPE_SECTIONS,
        site=site(training=True),
    )
    # Training reads one position row: text-only M-RoPE, whose three rows are equal.
    positions = torch.arange(37, 43)
    _, query, key = _merged_qkv_gate(6)
    query = query.clone().requires_grad_(True)
    key = key.clone().requires_grad_(True)
    expected_query, expected_key = _hf_rotated(query, key, positions.expand(3, 1, -1))

    if packed:
        heads = (6, -1, QWEN3_5_HEAD_DIM)
        rotated = rotary.forward_training(
            positions,
            query.view(heads),
            key.view(heads),
            torch.tensor([0, 2, 6], dtype=torch.int32),
            4,
        )
    else:
        heads = (1, 6, -1, QWEN3_5_HEAD_DIM)
        rotated = rotary.forward_training(positions, query.view(heads), key.view(heads))
    upstream = torch.randn(2, 6, 4 * QWEN3_5_HEAD_DIM)
    gradients = torch.autograd.grad(
        [tensor.reshape(6, -1) for tensor in rotated],
        (query, key),
        (upstream[0], upstream[1, :, : 2 * QWEN3_5_HEAD_DIM]),
    )
    expected_gradients = torch.autograd.grad(
        (expected_query, expected_key),
        (query, key),
        (upstream[0], upstream[1, :, : 2 * QWEN3_5_HEAD_DIM]),
    )

    assert rotary._training_cos_cache.shape == (MAX_POSITION, QWEN3_5_ROTARY_DIM // 2)
    for actual, expected in zip(rotated, (expected_query, expected_key), strict=True):
        torch.testing.assert_close(actual.reshape(6, -1), expected.detach())
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    # The head channels past rotary_dim and their gradients pass through unchanged.
    pass_through = rotated[0].reshape(6, 4, QWEN3_5_HEAD_DIM)[..., QWEN3_5_ROTARY_DIM:]
    original = query.detach().view(6, 4, QWEN3_5_HEAD_DIM)[..., QWEN3_5_ROTARY_DIM:]
    torch.testing.assert_close(pass_through, original, rtol=0, atol=0)


def test_mrope_sections_select_the_multi_axis_rotary(rotary_operation: None) -> None:
    geometry = (QWEN3_5_HEAD_DIM, QWEN3_5_ROTARY_DIM, MAX_POSITION, QWEN3_5_ROPE_THETA)

    multi_axis = rotary_module.RotaryEmbedding.create(
        *geometry, mrope_sections=QWEN3_5_MROPE_SECTIONS, site=site()
    )
    single_axis = rotary_module.RotaryEmbedding.create(*geometry, site=site())

    assert type(multi_axis) is rotary_module.MultiAxisRotaryEmbedding
    assert type(single_axis) is rotary_module.RotaryEmbedding
