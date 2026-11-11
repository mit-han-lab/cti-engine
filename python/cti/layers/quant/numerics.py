"""The numerics contract of every quantized format: the one source the reference providers,
the test oracles, the truth scripts and the kernel requests follow.

Fp8Block128 (Qwen's FP8 checkpoints):
- weight: ``W_hat[n, k] = f32(W[n, k]) * S[n // 128, k // 128]``; ``weight_scale_inv`` is the
  multiplier ``S`` despite its name, stored on disk in BF16 and held losslessly in FP32;
- activation: per token and 128 columns, with ``a = max(amax, EPS)`` in FP32 (no power-of-two
  rounding), ``s = a * f32(1 / 448)`` and ``q = e4m3_rne(clamp(x * f32(448 / a), -448, 448))``:
  multiplications, as the pinned kernels compute them;
- product: ``y = sum_g (q W)_g * s_g * S``, accumulated in FP32, returned in the input dtype.

Fp8Static (ModelOpt FP8, per-tensor static scales):
- weight: ``W_hat = f32(W) * weight_scale[slot]``, one FP32 scale per merged module;
- activation: ``q = e4m3_rne(clamp(x / input_scale, -448, 448))``, ``input_scale`` a calibrated
  FP32 scalar on the dequantization side;
- product: ``y = (q W) * (input_scale * weight_scale[slot])``, accumulated in FP32.

Nvfp4 (ModelOpt NVFP4):
- weight: ``W_hat[n, k] = e2m1(W)[n, k] * f32(weight_scale[n, k // 16]) * weight_scale_2[slot]``;
  ``W`` packs two E2M1 codes per byte, the even column in the low nibble; a code's bit 3 is its
  sign and its magnitude one of {0, 0.5, 1, 1.5, 2, 3, 4, 6};
- activation, per token and 16 columns, in FP32 operations in this order (cti-kernels'
  quantizer, FlashInfer's recipe): ``inverse = 1 / input_scale`` (IEEE),
  ``s = e4m3_rn_satfinite(inverse * (amax * rcp(6)))``,
  ``factor = rcp(f32(s) * rcp(inverse))``, 0 for an all-zero block, and
  ``q = e2m1_rn_satfinite(x * factor)`` saturating at 6. The kernel's ``rcp`` is PTX
  ``rcp.approx.ftz`` and its products flush subnormals; this module's reference uses IEEE
  reciprocals in the same order, so a block scale or code can differ from the kernel's by one
  ulp, and the bitwise reference for activation codes is the kernel's own quantize entry;
- product: ``y = sum_blocks (q W) * f32(s) * f32(weight_scale) * (input_scale *
  weight_scale_2[slot])``, accumulated in FP32.

``EPS`` keeps an all-zero (padding) row at zero instead of NaN; values are clamped before the
E4M3 conversion, which would turn anything past 448 into NaN.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

E4M3 = torch.float8_e4m3fn
E4M3_MAX = torch.finfo(E4M3).max
EPS = 1e-10
# Columns per activation scale and rows and columns per weight scale.
BLOCK = 128


def dequantize_block_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``W_hat`` in FP32 from an E4M3 weight ``[N, K]`` and its scale grid
    ``[ceil(N / 128), ceil(K / 128)]``."""

    rows, cols = weight.shape
    grid = scale.float().repeat_interleave(BLOCK, 0)[:rows].repeat_interleave(BLOCK, 1)[:, :cols]
    return weight.float() * grid


def quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q`` (E4M3, shaped like ``x``) and ``s`` (FP32 ``[..., ceil(K / 128)]``) of ``x``."""

    k = x.shape[-1]
    groups = -(-k // BLOCK)
    blocks = F.pad(x.float(), (0, groups * BLOCK - k)).unflatten(-1, (groups, BLOCK))
    amax = blocks.abs().amax(-1).clamp_min(EPS)
    scale = amax * (1.0 / E4M3_MAX)
    # One IEEE division, as the kernel's div_rn: ``E4M3_MAX / amax`` would be torch's
    # reciprocal-then-multiply, rounded twice.
    factor = torch.full_like(amax, E4M3_MAX) / amax
    q = (blocks * factor.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)
    return q.flatten(-2)[..., :k], scale


def dequantize_activation(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``q * s`` in FP32, each 128-column group by its scale."""

    k = q.shape[-1]
    return q.float() * scale.repeat_interleave(BLOCK, -1)[..., :k]


def quantize_static_activation(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """``q`` (E4M3, shaped like ``x``) of ``x`` under the static scalar ``input_scale``."""

    return (x.float() / input_scale.float()).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)


def dequantize_static(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``q * scale`` in FP32, for an E4M3 weight or activation and its scalar scale."""

    return q.float() * scale.float()


# Columns per NVFP4 block scale, the largest E2M1 magnitude and the E2M1 magnitudes by code.
FP4_BLOCK = 16
E2M1_MAX = 6.0
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Magnitudes past which a code rounds up, and whether a tie at one rounds up (to the even code).
_E2M1_BOUNDS = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False))
_E2M1_BOUNDS += ((3.5, True), (5.0, False))


def e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """E2M1 codes (uint8, sign in bit 3) of FP32 ``v``: round to nearest, ties to even,
    saturating at 6."""

    a = v.abs()
    code = torch.zeros(v.shape, dtype=torch.uint8, device=v.device)
    for bound, tie_up in _E2M1_BOUNDS:
        code += (a >= bound if tie_up else a > bound).to(torch.uint8)
    return code | ((v < 0).to(torch.uint8) << 3)


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    """FP32 values of E2M1 codes."""

    magnitudes = torch.tensor(E2M1_VALUES, device=codes.device)
    values = magnitudes[(codes & 7).long()]
    return torch.where(codes >= 8, -values, values)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Codes ``[..., K]`` packed two per byte ``[..., K / 2]``, the even column low."""

    pairs = codes.unflatten(-1, (-1, 2))
    return pairs[..., 0] | (pairs[..., 1] << 4)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Codes ``[..., 2 K]`` of packed bytes ``[..., K]``."""

    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)


def dequantize_nvfp4_weight(
    weight: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """``W_hat`` in FP32 of a packed NVFP4 weight ``[N, K / 2]``, its E4M3 block scales
    ``[N, K / 16]`` and its scalar ``weight_scale_2``."""

    values = e2m1_values(unpack_e2m1(weight))
    return values * scale.float().repeat_interleave(FP4_BLOCK, -1) * global_scale.float()


def quantize_nvfp4_activation(
    x: torch.Tensor, input_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes, s)`` of ``x [..., K]`` (``K`` a multiple of 16): E2M1 codes (uint8, unpacked)
    and the E4M3 block scales ``[..., K / 16]``, in the order the module docstring gives."""

    blocks = x.float().unflatten(-1, (-1, FP4_BLOCK))
    inverse = 1.0 / input_scale.float()
    amax = blocks.abs().amax(-1)
    scale = (inverse * (amax * (1.0 / 6.0))).clamp_max(E4M3_MAX).to(E4M3)
    factor = 1.0 / (scale.float() * (1.0 / inverse))
    factor = torch.where(amax > 0, factor, torch.zeros_like(factor))
    return e2m1_codes(blocks * factor.unsqueeze(-1)).flatten(-2), scale


def dequantize_nvfp4_activation(
    codes: torch.Tensor, scale: torch.Tensor, input_scale: torch.Tensor
) -> torch.Tensor:
    """``q * s * input_scale`` in FP32, each 16-column block by its scale."""

    blocks = e2m1_values(codes).unflatten(-1, (-1, FP4_BLOCK))
    return (blocks * scale.float().unsqueeze(-1) * input_scale.float()).flatten(-2)


__all__ = [
    "BLOCK",
    "E2M1_MAX",
    "E2M1_VALUES",
    "E4M3",
    "E4M3_MAX",
    "EPS",
    "FP4_BLOCK",
    "dequantize_activation",
    "dequantize_block_weight",
    "dequantize_nvfp4_activation",
    "dequantize_nvfp4_weight",
    "dequantize_static",
    "e2m1_codes",
    "e2m1_values",
    "pack_e2m1",
    "quantize_activation",
    "quantize_nvfp4_activation",
    "quantize_static_activation",
    "unpack_e2m1",
]
