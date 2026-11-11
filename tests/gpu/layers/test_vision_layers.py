"""The vision tower layers on their CUDA operations, against transformers 5.16.1's Qwen3.5
vision tower."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)
from transformers.vision_utils import get_vision_position_ids

from cti.layers import BidirectionalAttention, VisionRotaryEmbedding
from tests.support.tp_shards import site

pytestmark = pytest.mark.checkpoints()

DEVICE = torch.device("cuda")
# Qwen3.5 vision tower: 16 heads of 1152 / 16 = 72 channels, 2x2 patch merge.
VISION_HEAD_DIM = 72
VISION_NUM_HEADS = 16
MERGE_SIZE = 2


def test_vision_rotary_head_dim_72_matches_hf() -> None:
    torch.manual_seed(19)
    image_grids = torch.tensor([[1, 8, 12], [1, 4, 4]], device=DEVICE)
    patch_positions = get_vision_position_ids(image_grids, MERGE_SIZE)
    rotary = VisionRotaryEmbedding(VISION_HEAD_DIM, 10_000.0, site=site(device=DEVICE))
    qkv = torch.randn(
        patch_positions.shape[0],
        3 * VISION_NUM_HEADS * VISION_HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    query, key, _ = qkv.view(-1, 3, VISION_NUM_HEADS, VISION_HEAD_DIM).unbind(1)
    frequencies = Qwen3_5VisionRotaryEmbedding(VISION_HEAD_DIM // 2).to(DEVICE)(patch_positions)
    hf_angles = torch.cat((frequencies, frequencies), dim=-1)

    with torch.no_grad():
        cos, sin = rotary.cos_sin(patch_positions)
        rotated = rotary.forward(query, key, cos, sin)
    hf_rotated = apply_rotary_pos_emb_vision(query, key, hf_angles.cos(), hf_angles.sin())
    eager_rotated = apply_rotary_pos_emb_vision(query, key, cos, sin)

    torch.testing.assert_close(cos, hf_angles.cos())
    torch.testing.assert_close(sin, hf_angles.sin())
    for actual, reference, eager in zip(rotated, hf_rotated, eager_rotated, strict=True):
        torch.testing.assert_close(actual, reference)
        # The operation reproduces the FP32 eager rotation before its single rounding.
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)


def test_bidirectional_attention_head_dim_72_matches_fp32_sdpa() -> None:
    torch.manual_seed(23)
    # Three images of 64, 40 and 4 patches in one packed step.
    lengths = (64, 40, 4)
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device=DEVICE)
    qkv = torch.randn(
        bounds[-1], 3 * VISION_NUM_HEADS * VISION_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    query, key, value = qkv.view(-1, 3, VISION_NUM_HEADS, VISION_HEAD_DIM).unbind(1)
    attention = BidirectionalAttention(VISION_NUM_HEADS, VISION_HEAD_DIM)

    with torch.no_grad():
        output = attention.forward(query, key, value, cu_seqlens, max_sequence_length=max(lengths))

    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.cat(
            [
                F.scaled_dot_product_attention(
                    *(tensor[start:end].float().transpose(0, 1) for tensor in (query, key, value))
                )
                .transpose(0, 1)
                .reshape(end - start, -1)
                for start, end in pairwise(bounds)
            ]
        )
    assert output.shape == (bounds[-1], VISION_NUM_HEADS * VISION_HEAD_DIM)
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.02)
