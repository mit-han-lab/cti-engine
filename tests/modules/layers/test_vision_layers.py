"""The vision tower layers (LayerNorm, the 2-D patch rotary and the bidirectional
attention) on CPU, against transformers 5.16.1's Qwen3.5 vision tower.

The CPU references of the vision rotary and the noncausal varlen attention are bound where
those layers resolve their operations.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionAttention,
    Qwen3_5VisionBlock,
    Qwen3_5VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)
from transformers.vision_utils import get_vision_position_ids

from cti.layers import BidirectionalAttention, LayerNorm, VisionRotaryEmbedding
from tests.support import reference_ops
from tests.support.tp_shards import site

HIDDEN = 64
# Qwen3.5 vision geometry per head; two heads keep the CPU reference small.
HEAD_DIM = 72
NUM_HEADS = 2
MERGE_SIZE = 2


@pytest.fixture(autouse=True)
def reference_vision_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    reference_ops.bind_vision_references(monkeypatch)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layer_norm_matches_hf_vision_layer_norm(dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    config = Qwen3_5VisionConfig(hidden_size=HIDDEN, num_heads=4, intermediate_size=2 * HIDDEN)
    reference = Qwen3_5VisionBlock(config).norm1.to(dtype)
    with torch.no_grad():
        reference.weight.normal_()
        reference.bias.normal_()
    norm = LayerNorm(HIDDEN, reference.eps)
    norm.weight = torch.empty(HIDDEN, dtype=dtype)
    norm.bias = torch.empty(HIDDEN, dtype=dtype)
    norm.load_state_dict(
        {"weight": reference.weight.detach().clone(), "bias": reference.bias.detach().clone()}
    )
    x = torch.randn(3, 7, HIDDEN, dtype=dtype)

    torch.testing.assert_close(norm.forward(x), reference(x).detach(), rtol=0, atol=0)


def test_vision_rotary_matches_hf_patch_rotation() -> None:
    head_dim = 72
    num_heads = 3
    merge_size = 2
    image_grids = torch.tensor([[1, 4, 6], [1, 2, 2]])
    patch_positions = get_vision_position_ids(image_grids, merge_size)
    rotary = VisionRotaryEmbedding(head_dim, 10_000.0, site=site())
    hf_frequencies = Qwen3_5VisionRotaryEmbedding(head_dim // 2)(patch_positions)
    hf_angles = torch.cat((hf_frequencies, hf_frequencies), dim=-1)
    torch.manual_seed(13)
    qkv = torch.randn(patch_positions.shape[0], 3 * num_heads * head_dim, dtype=torch.bfloat16)
    query, key, _ = qkv.view(-1, 3, num_heads, head_dim).unbind(1)
    expected = apply_rotary_pos_emb_vision(query, key, hf_angles.cos(), hf_angles.sin())

    cos, sin = rotary.cos_sin(patch_positions)
    rotated = rotary.forward(query, key, cos, sin)

    torch.testing.assert_close(cos, hf_angles.cos(), rtol=0, atol=0)
    torch.testing.assert_close(sin, hf_angles.sin(), rtol=0, atol=0)
    for actual, reference in zip(rotated, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_vision_rotary_rejects_head_dim_without_two_axis_halves() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        VisionRotaryEmbedding(70, 10_000.0, site=site())


def test_vision_attention_layers_match_hf_vision_attention() -> None:
    torch.manual_seed(19)
    config = Qwen3_5VisionConfig(
        hidden_size=NUM_HEADS * HEAD_DIM,
        num_heads=NUM_HEADS,
        intermediate_size=4 * HEAD_DIM,
    )
    config._attn_implementation = "sdpa"
    reference = Qwen3_5VisionAttention(config).eval()
    # Three images packed in one step; each one is a sequence.
    image_grids = torch.tensor([[1, 4, 4], [1, 2, 6], [1, 2, 2]])
    cu_seqlens = torch.tensor([0, 16, 28, 32], dtype=torch.int32)
    patch_positions = get_vision_position_ids(image_grids, MERGE_SIZE)
    hidden = torch.randn(32, NUM_HEADS * HEAD_DIM)
    rotary = VisionRotaryEmbedding(HEAD_DIM, 10_000.0, site=site())
    attention = BidirectionalAttention(NUM_HEADS, HEAD_DIM)

    cos, sin = rotary.cos_sin(patch_positions)
    with torch.no_grad():
        qkv = F.linear(hidden, reference.qkv.weight, reference.qkv.bias)
        query, key, value = qkv.view(-1, 3, NUM_HEADS, HEAD_DIM).unbind(1)
        query, key = rotary.forward(query, key, cos, sin)
        attended = attention.forward(query, key, value, cu_seqlens, max_sequence_length=16)
        actual = F.linear(attended, reference.proj.weight, reference.proj.bias)
        expected = reference(hidden, cu_seqlens=cu_seqlens, position_embeddings=(cos, sin))

    assert attended.shape == (32, NUM_HEADS * HEAD_DIM)
    torch.testing.assert_close(actual, expected)


def test_bidirectional_attention_binds_once_and_forwards_the_length_bound() -> None:
    attention = BidirectionalAttention(NUM_HEADS, HEAD_DIM)
    tokens = torch.randn(6, NUM_HEADS * HEAD_DIM)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)

    assert isinstance(attention._attention, reference_ops.NoncausalVarlenAttention)
    assert attention.state_dict() == {}
    with pytest.raises(ValueError, match="max_sequence_length"):
        attention.forward(tokens, tokens, tokens, cu_seqlens, max_sequence_length=3)
