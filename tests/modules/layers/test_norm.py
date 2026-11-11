from __future__ import annotations

import sys

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from cti.layers import ResidualStreamRMSNorm, RMSNorm, RMSNormFused
from tests.support import reference_ops, training_ops
from tests.support.training_ops import bind_training_references

HIDDEN = 64
EPS = 1e-6


@pytest.fixture(autouse=True)
def reference_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    reference_ops.bind_norm_references(monkeypatch)


def _hf_norm(weight: torch.Tensor) -> Qwen3_5RMSNorm:
    norm = Qwen3_5RMSNorm(weight.shape[0], eps=EPS)
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


def _zero_centered_weight(dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(5)
    # Zero-centered checkpoints store small offsets from one.
    return (0.1 * torch.randn(HIDDEN)).to(dtype)


def _stream_gradient(
    stored: torch.Tensor,
    weight: torch.Tensor,
    upstream: torch.Tensor,
    stream_upstream: torch.Tensor,
) -> torch.Tensor:
    """The gradient of a stored stream value: the FP32 input gradient of Transformers' norm
    of it, rounded to its dtype, plus the stream's own, added in its dtype."""

    values = stored.detach().float().requires_grad_(True)
    hf_norm = _hf_norm(weight.float()).requires_grad_(False)
    (gradient,) = torch.autograd.grad(hf_norm(values), values, upstream.float())
    return gradient.to(stored.dtype) + stream_upstream


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_centered_norm_matches_hf_rms_norm(dtype: torch.dtype) -> None:
    weight = _zero_centered_weight(dtype)
    norm = RMSNorm(HIDDEN, EPS, zero_centered=True)
    norm.weight = weight
    hf_norm = _hf_norm(weight).to(dtype)
    x = torch.randn(5, HIDDEN, dtype=dtype)
    expected = hf_norm(x)

    torch.testing.assert_close(norm.forward(x), expected, rtol=0, atol=0)

    # Q/K heads normalize in place inside the merged projection, which stays strided.
    num_heads = 3
    projection = torch.randn(5, (num_heads + 2) * HIDDEN, dtype=dtype)
    heads = projection[:, : num_heads * HIDDEN].view(5, num_heads, HIDDEN)
    expected_heads = hf_norm(heads.clone())
    untouched = projection[:, num_heads * HIDDEN :].clone()
    norm.forward_inplace(heads)
    torch.testing.assert_close(heads, expected_heads, rtol=0, atol=0)
    torch.testing.assert_close(projection[:, num_heads * HIDDEN :], untouched, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_residual_stream_norm_matches_hf_norm_of_the_fp32_sum(dtype: torch.dtype) -> None:
    """Inference stores ``residual + x`` rounded to the stream's dtype and normalizes the
    FP32 sum before that rounding, which only a BF16 stream tells from the stored one."""

    weight = _zero_centered_weight(dtype)
    norm = ResidualStreamRMSNorm(HIDDEN, EPS)
    norm.weight = weight
    hf_norm = _hf_norm(weight).to(dtype)
    x = torch.randn(5, HIDDEN, dtype=dtype)
    # Thirty times x's scale, so that a BF16 sum drops low bits of x.
    residual = (30 * torch.randn(5, HIDDEN)).to(dtype)
    expected_residual = residual + x
    # Transformers' norm of an FP32 input is FP32, rounded here once to the dtype.
    expected = hf_norm(residual.float() + x.float()).to(dtype)
    assert torch.equal(hf_norm(expected_residual), expected) is (dtype == torch.float32)

    # The stream's last norm normalizes the same sum and leaves the residual unwritten.
    last_x, last_residual = x.clone(), residual.clone()
    last = norm.forward_last(last_x, last_residual)
    assert last is last_x
    torch.testing.assert_close(last, expected, rtol=0, atol=0)
    torch.testing.assert_close(last_residual, residual, rtol=0, atol=0)

    normalized, summed = norm.forward(x.clone(), residual)

    assert summed is residual
    torch.testing.assert_close(summed, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(normalized, expected, rtol=0, atol=0)

    first, passthrough = norm.forward(x)
    assert passthrough is x
    torch.testing.assert_close(first, hf_norm(x), rtol=0, atol=0)


def test_default_norm_keeps_stored_weight_scaling() -> None:
    weight = torch.randn(HIDDEN)
    x = torch.randn(4, HIDDEN)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * weight
    norm = RMSNorm(HIDDEN, EPS)
    fused = RMSNormFused(HIDDEN, EPS)
    norm.weight = fused.weight = weight

    torch.testing.assert_close(norm.forward(x), expected)
    torch.testing.assert_close(fused.forward(x)[0], expected)


class _RecordingTrainingNorm:
    """The training norm's leaves, recording the name and options of each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float, **options: object):
        self.calls.append(("rms_norm", options))
        return training_ops.rms_norm(x, weight, eps, **options)


@pytest.fixture
def training_norm(monkeypatch: pytest.MonkeyPatch) -> _RecordingTrainingNorm:
    bind_training_references(monkeypatch)
    recording = _RecordingTrainingNorm()
    monkeypatch.setattr(sys.modules["ctikernels.norm.training"], "rms_norm", recording.rms_norm)
    return recording


def test_zero_centered_training_norm_matches_hf_autograd(
    training_norm: _RecordingTrainingNorm,
) -> None:
    weight = _zero_centered_weight(torch.float32)
    norm = RMSNorm(HIDDEN, EPS, zero_centered=True)
    norm.weight = weight
    hf_norm = _hf_norm(weight).requires_grad_(False)
    x = torch.randn(5, HIDDEN, requires_grad=True)
    upstream = torch.randn(5, HIDDEN)

    normalized = norm.forward_training(x)
    (gradient,) = torch.autograd.grad(normalized, x, upstream)
    (expected_gradient,) = torch.autograd.grad(hf_norm(x), x, upstream)

    with torch.no_grad():
        torch.testing.assert_close(normalized, hf_norm(x))
    torch.testing.assert_close(gradient, expected_gradient)
    assert training_norm.calls == [("rms_norm", {"zero_centered": True})]
    assert norm.weight.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_residual_stream_training_norm_matches_hf_autograd(
    training_norm: _RecordingTrainingNorm,
    dtype: torch.dtype,
) -> None:
    """The stream comes back as the input itself: autograd adds its gradient to the norm's,
    each in the input dtype, as Unsloth trains."""

    weight = _zero_centered_weight(dtype)
    norm = ResidualStreamRMSNorm(HIDDEN, EPS)
    norm.weight = weight
    hf_norm = _hf_norm(weight).to(dtype).requires_grad_(False)
    x = torch.randn(5, HIDDEN, dtype=dtype, requires_grad=True)
    upstream = torch.randn(5, HIDDEN, dtype=dtype)
    stream_upstream = torch.randn(5, HIDDEN, dtype=dtype)

    normalized, stream = norm.forward_training(x)
    (gradient,) = torch.autograd.grad((normalized, stream), x, (upstream, stream_upstream))

    with torch.no_grad():
        torch.testing.assert_close(normalized, hf_norm(x), rtol=0, atol=0)
        assert stream is x
    # The FP32 reference graph sums the stream's gradient among the norm's partial ones.
    exact = {"rtol": 0, "atol": 0} if dtype is torch.bfloat16 else {}
    torch.testing.assert_close(
        gradient, _stream_gradient(x, weight, upstream, stream_upstream), **exact
    )
    assert training_norm.calls == [
        ("rms_norm", {"zero_centered": True, "out_dtype": dtype}),
    ]


def test_residual_stream_references_take_the_kernels_operands() -> None:
    """Only the in-place add needs a residual distinct from ``x``; the inference leaves
    have no autograd and write no tensor that requires grad."""

    weight = _zero_centered_weight(torch.float32)
    x = torch.randn(2, HIDDEN)

    with pytest.raises(ValueError, match="distinct from the input"):
        reference_ops.add_rms_norm_(x, x, weight, EPS, zero_centered=True)
    with pytest.raises(RuntimeError, match="cannot modify a tensor that requires gradients"):
        reference_ops.add_rms_norm_(
            x.clone().requires_grad_(True), x.clone(), weight, EPS, zero_centered=True
        )
    with pytest.raises(RuntimeError, match="does not support autograd"):
        reference_ops.residual_rms_norm(
            x.requires_grad_(True), weight, EPS, zero_centered=True, out_dtype=torch.float32
        )


def test_default_training_norm_keeps_stored_weight_scaling(
    training_norm: _RecordingTrainingNorm,
) -> None:
    weight = torch.randn(HIDDEN)
    x = torch.randn(4, HIDDEN)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * weight
    norm = RMSNorm(HIDDEN, EPS)
    fused = RMSNormFused(HIDDEN, EPS)
    norm.weight = fused.weight = weight

    torch.testing.assert_close(norm.forward_training(x), expected)
    torch.testing.assert_close(fused.forward_training(x)[0], expected)


def test_residual_stream_training_norm_refuses_a_weight_that_requires_grad(
    training_norm: _RecordingTrainingNorm,
) -> None:
    norm = ResidualStreamRMSNorm(HIDDEN, EPS)
    norm.weight = torch.zeros(HIDDEN, requires_grad=True)

    with pytest.raises(ValueError, match="frozen weight"):
        norm.forward_training(torch.ones(2, HIDDEN))
    assert set(norm.state_dict()) == {"weight"}
