# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`); see THIRD_PARTY_NOTICES.

"""Qwen3.5 layer paths on their CUDA operations, against transformers 5.16.1."""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from cti.layers import ResidualStreamRMSNorm, RMSNorm, RotaryEmbedding
from ctikernels.norm import inference as inference_norm
from tests.support.tp_shards import site

pytestmark = pytest.mark.checkpoints()

DEVICE = torch.device("cuda")
EPS = 1e-6
# Qwen3.5 text attention (9B): 16 query and 4 KV heads of 256 channels, 64 rotated.
HEAD_DIM = 256
ROTARY_DIM = 64
NUM_QO_HEADS = 16
NUM_KV_HEADS = 4
ROPE_THETA = 10_000_000
MROPE_SECTIONS = (11, 11, 10)
MAX_POSITION = 4096


def _gated_projection(num_tokens: int) -> torch.Tensor:
    """A BF16 ``q | k | v | gate`` projection, as the gated attention layer produces it."""
    width = (2 * NUM_QO_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    return torch.randn(num_tokens, width, device=DEVICE, dtype=torch.bfloat16)


def _query_key(projection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    query_size = NUM_QO_HEADS * HEAD_DIM
    key_size = NUM_KV_HEADS * HEAD_DIM
    return projection[:, :query_size], projection[:, query_size : query_size + key_size]


def _image_prompt_positions() -> torch.Tensor:
    """Text, a 4x6 merged image grid, text, then three decode steps on every axis.

    Rewritten from the Qwen3.5 rule (transformers 5.16.1 ``Qwen3_5Model.get_rope_index``):
    image tokens sit at ``start + (t, h, w)``; the following text resumes at
    ``start + max(h, w)``; decode positions are the sequence index plus that offset.
    """
    rows = [[position] * 3 for position in range(5)]
    start = 5
    rows += [[start, start + h, start + w] for h in range(4) for w in range(6)]
    next_text = start + 6
    rows += [[position] * 3 for position in range(next_text, next_text + 3)]
    decode_offset = next_text + 3 - len(rows)
    rows += [[index + decode_offset] * 3 for index in range(len(rows), len(rows) + 3)]
    return torch.tensor(rows, device=DEVICE).T.contiguous()


def _hf_rotated(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF's FP32 rotation of ``[tokens, heads * head_dim]`` Q/K, rounded once to BF16."""
    config = Qwen3_5TextConfig(
        hidden_size=HEAD_DIM,
        num_attention_heads=NUM_QO_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=MAX_POSITION,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": ROPE_THETA,
            "partial_rotary_factor": ROTARY_DIM / HEAD_DIM,
            "mrope_section": list(MROPE_SECTIONS),
            "mrope_interleaved": True,
        },
    )
    tokens = query.shape[0]
    cos, sin = Qwen3_5TextRotaryEmbedding(config).to(DEVICE)(query.float(), positions[:, None])

    def heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().reshape(1, tokens, -1, HEAD_DIM).transpose(1, 2)

    def rows(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(tokens, -1).to(torch.bfloat16)

    rotated_query, rotated_key = apply_rotary_pos_emb(heads(query), heads(key), cos, sin)
    return rows(rotated_query), rows(rotated_key)


def test_zero_centered_rms_norm_matches_hf() -> None:
    torch.manual_seed(20261001)
    hidden = 4096
    weight = (0.1 * torch.randn(hidden, device=DEVICE)).to(torch.bfloat16)
    reference = Qwen3_5RMSNorm(hidden, eps=EPS).to(DEVICE, torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(weight)
    norm = RMSNorm(hidden, EPS, zero_centered=True)
    fused = ResidualStreamRMSNorm(hidden, EPS)
    norm.weight = fused.weight = weight
    x = torch.randn(37, hidden, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    expected_residual = residual + x

    with torch.no_grad():
        torch.testing.assert_close(norm.forward(x), reference(x))
        normalized, summed = fused.forward(x.clone(), residual)
        torch.testing.assert_close(summed, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(normalized, reference(expected_residual))


def _zero_centered_norm(stream: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The zero-centered RMS norm of ``stream`` in FP64."""

    values = stream.double()
    scale = torch.rsqrt(values.square().mean(-1, keepdim=True) + EPS)
    return values * scale * (1 + weight.double())


def _assert_one_rounding(actual: torch.Tensor, exact: torch.Tensor, message: str) -> None:
    """``actual`` is ``exact`` rounded once to BF16: its nearest value or the next one
    toward ``exact`` (BF16 bit patterns step by one in magnitude)."""

    nearest = exact.to(torch.bfloat16)
    step = torch.sign(exact.abs() - nearest.double().abs()).to(torch.int16)
    other = (nearest.view(torch.int16) + step).view(torch.bfloat16)
    wrong = (actual != nearest) & (actual != other)
    assert not wrong.any(), f"{message}: {int(wrong.sum())} values beyond one rounding"


@pytest.mark.parametrize("hidden", [4096, 5120], ids=["9b", "27b"])
def test_residual_stream_norm_matches_an_add_then_the_grouped_norm(hidden: int) -> None:
    """Every residual-stream norm equals a torch add, then the grouped zero-centered norm,
    rounded once to BF16: of the FP32 sum in inference's fused norm, of the sum as stored
    in training, which adds in a torch add, or of the stream alone. Training's and the
    stream-only norm's have their bits. Inference's fused norm picks its kernel by row count,
    whose FP32 reduction can round a near tie the other way: it is one rounding of the FP64
    norm of the FP32 sum (either BF16 neighbour), as is the grouped norm. Backward, as
    Unsloth trains: the grouped backward of the stored value as an FP32 input, rounded to
    the stream's BF16, plus the stream's own gradient, added in BF16. One token is a decode
    step, 512 a prefill."""

    torch.manual_seed(hidden)
    weight = (0.1 * torch.randn(hidden, device=DEVICE)).to(torch.bfloat16)
    norm = ResidualStreamRMSNorm(hidden, EPS)
    norm.weight = weight

    def stream_gradient_of(
        stored: torch.Tensor, upstream: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        values = stored.float().requires_grad_(True)
        from ctikernels.norm import training as training_norm

        grouped = training_norm.rms_norm(values, weight, EPS, zero_centered=True)
        (gradient,) = torch.autograd.grad(grouped, values, upstream[0].float())
        return gradient.to(stored.dtype) + upstream[1]

    def grouped_norm(stored: torch.Tensor) -> torch.Tensor:
        return inference_norm.rms_norm(
            stored, weight, EPS, zero_centered=True, out_dtype=torch.bfloat16
        )

    for tokens in (1, 7, 512):
        x = torch.randn(tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
        # Eight times x's scale, so that a BF16 sum drops low bits of x.
        residual = 8 * torch.randn(tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
        # Rounded once in BF16.
        summed = residual + x
        upstream = (torch.randn_like(x), torch.randn_like(residual))
        with torch.no_grad():
            expected = grouped_norm(residual.float() + x.float())
            expected_training = grouped_norm(summed)
            expected_stream = grouped_norm(residual)
            inferred = norm.forward(x.clone(), residual.clone())
            last_residual = residual.clone()
            last = norm.forward_last(x.clone(), last_residual)
            inferred_stream, _ = norm.forward(residual)
        trained_x = x.clone().requires_grad_(True)
        trained_residual = residual.clone().requires_grad_(True)
        trained_stream, stream = norm.forward_training(trained_residual)
        (stream_gradient,) = torch.autograd.grad(
            (trained_stream, stream), trained_residual, upstream
        )
        # A training segment's add, then the norm of the sum as stored: the add hands both
        # summands the sum's gradient.
        trained = norm.forward_training(trained_residual + trained_x)
        gradients = torch.autograd.grad(trained, (trained_x, trained_residual), upstream)
        expected_gradient = stream_gradient_of(summed, upstream)
        expected_stream_gradient = stream_gradient_of(residual, upstream)

        message = f"{tokens} tokens"
        # The BF16 stream rounds the sum before training's norm reads it.
        assert not torch.equal(expected_training, expected)
        for actual in (inferred, trained):
            torch.testing.assert_close(actual[1], summed, rtol=0, atol=0, msg=message)
        torch.testing.assert_close(trained[0], expected_training, rtol=0, atol=0, msg=message)
        exact = _zero_centered_norm(residual.float() + x.float(), weight)
        for actual in (expected, inferred[0], last):
            _assert_one_rounding(actual, exact, message)
        torch.testing.assert_close(last_residual, residual, rtol=0, atol=0, msg=message)
        for actual in (inferred_stream, trained_stream):
            torch.testing.assert_close(actual, expected_stream, rtol=0, atol=0, msg=message)
        assert stream.data_ptr() == trained_residual.data_ptr()
        for gradient in gradients:
            torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0, msg=message)
        torch.testing.assert_close(
            stream_gradient, expected_stream_gradient, rtol=0, atol=0, msg=message
        )


def test_zero_centered_head_norm_writes_through_the_merged_projection() -> None:
    torch.manual_seed(7)
    weight = (0.1 * torch.randn(HEAD_DIM, device=DEVICE)).to(torch.bfloat16)
    reference = Qwen3_5RMSNorm(HEAD_DIM, eps=EPS).to(DEVICE, torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(weight)
    norm = RMSNorm(HEAD_DIM, EPS, zero_centered=True)
    norm.weight = weight
    projection = _gated_projection(29)
    query, key = _query_key(projection)
    expected_query = reference(query.reshape(29, NUM_QO_HEADS, HEAD_DIM))
    expected_key = reference(key.reshape(29, NUM_KV_HEADS, HEAD_DIM))
    rest = projection[:, (NUM_QO_HEADS + NUM_KV_HEADS) * HEAD_DIM :].clone()

    with torch.no_grad():
        norm.forward_inplace(query.view(29, NUM_QO_HEADS, HEAD_DIM))
        norm.forward_inplace(key.view(29, NUM_KV_HEADS, HEAD_DIM))

    torch.testing.assert_close(query.view(29, NUM_QO_HEADS, HEAD_DIM), expected_query)
    torch.testing.assert_close(key.view(29, NUM_KV_HEADS, HEAD_DIM), expected_key)
    torch.testing.assert_close(
        projection[:, (NUM_QO_HEADS + NUM_KV_HEADS) * HEAD_DIM :], rest, rtol=0, atol=0
    )


def test_partial_multi_axis_rope_matches_hf_with_decode_offsets() -> None:
    torch.manual_seed(11)
    rotary = RotaryEmbedding.create(
        HEAD_DIM,
        ROTARY_DIM,
        MAX_POSITION,
        ROPE_THETA,
        mrope_sections=MROPE_SECTIONS,
        site=site(device=DEVICE),
    )
    positions = _image_prompt_positions()
    projection = _gated_projection(positions.shape[1])
    query, key = _query_key(projection)
    expected_query, expected_key = _hf_rotated(query, key, positions)
    original = projection.clone()

    with torch.no_grad():
        rotary.forward(positions, query, key)

    torch.testing.assert_close(query, expected_query)
    torch.testing.assert_close(key, expected_key)
    # Only the leading 64 channels of each Q/K head move; V and the gate stay put.
    num_rotated_heads = NUM_QO_HEADS + NUM_KV_HEADS
    rotated_width = num_rotated_heads * HEAD_DIM
    heads = projection[:, :rotated_width].view(-1, num_rotated_heads, HEAD_DIM)
    original_heads = original[:, :rotated_width].view(-1, num_rotated_heads, HEAD_DIM)
    torch.testing.assert_close(
        heads[..., ROTARY_DIM:], original_heads[..., ROTARY_DIM:], rtol=0, atol=0
    )
    torch.testing.assert_close(
        projection[:, rotated_width:], original[:, rotated_width:], rtol=0, atol=0
    )


def test_multi_axis_rope_with_equal_rows_is_one_dimensional_rope() -> None:
    torch.manual_seed(13)
    at = site(device=DEVICE)
    single_axis = RotaryEmbedding.create(HEAD_DIM, ROTARY_DIM, MAX_POSITION, ROPE_THETA, site=at)
    multi_axis = RotaryEmbedding.create(
        HEAD_DIM, ROTARY_DIM, MAX_POSITION, ROPE_THETA, mrope_sections=MROPE_SECTIONS, site=at
    )
    positions = torch.tensor([0, 3, 17, 1000, 4095], device=DEVICE)
    projection = _gated_projection(5)
    results = []
    with torch.no_grad():
        for rotary, rotary_positions in (
            (single_axis, positions),
            (multi_axis, positions),
            (multi_axis, positions.expand(3, -1)),
        ):
            rotated = projection.clone()
            rotary.forward(rotary_positions, *_query_key(rotated))
            results.append(rotated)

    expected_query, expected_key = _hf_rotated(*_query_key(projection), positions.expand(3, -1))
    torch.testing.assert_close(_query_key(results[0])[0], expected_query)
    torch.testing.assert_close(_query_key(results[0])[1], expected_key)
    for result in results[1:]:
        torch.testing.assert_close(result, results[0], rtol=0, atol=0)


def test_multi_axis_rope_replays_in_a_cuda_graph() -> None:
    torch.manual_seed(17)
    rotary = RotaryEmbedding.create(
        HEAD_DIM,
        ROTARY_DIM,
        MAX_POSITION,
        ROPE_THETA,
        mrope_sections=MROPE_SECTIONS,
        site=site(device=DEVICE),
    )
    batch_size = 8
    static_positions = torch.zeros(3, batch_size, dtype=torch.int64, device=DEVICE)
    static_projection = _gated_projection(batch_size)
    with torch.no_grad():
        # Warm the provider outside capture, then capture on a side stream.
        rotary.forward(static_positions, *_query_key(static_projection.clone()))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rotary.forward(static_positions, *_query_key(static_projection))

        for step in range(3):
            positions = torch.randint(0, MAX_POSITION, (3, batch_size), device=DEVICE)
            projection = _gated_projection(batch_size)
            static_positions.copy_(positions)
            static_projection.copy_(projection)
            graph.replay()
            rotary.forward(positions, *_query_key(projection))
            torch.testing.assert_close(static_projection, projection, rtol=0, atol=0, msg=str(step))
