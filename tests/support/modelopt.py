"""ModelOpt test weights (static FP8 and NVFP4) and the contract's products in FP64
(``cti.layers.quant.numerics``)."""

from __future__ import annotations

import torch

from cti.layers.quant import numerics


def static_fp8_weight(
    rows: int, cols: int, *, seed: int = 0, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """An E4M3 weight and its FP32 scalar scale, as a ModelOpt FP8 checkpoint stores them."""

    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator) * 0.05
    scale = weight.abs().amax() / numerics.E4M3_MAX
    return (weight / scale).to(numerics.E4M3).to(device), scale.reshape(()).to(device)


def nvfp4_weight(
    rows: int, cols: int, *, seed: int = 0, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A packed NVFP4 weight ``[rows, cols / 2]``, its E4M3 block scales ``[rows, cols / 16]``
    and FP32 scalar global scale, as a ModelOpt NVFP4 checkpoint stores them."""

    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (rows, cols), generator=generator, dtype=torch.uint8)
    scale = (torch.rand(rows, cols // numerics.FP4_BLOCK, generator=generator) * 400 + 8).to(
        numerics.E4M3
    )
    global_scale = torch.rand((), generator=generator) * 1e-4 + 1e-5
    return numerics.pack_e2m1(codes).to(device), scale.to(device), global_scale.to(device)


def input_scale(x: torch.Tensor, maximum: float) -> torch.Tensor:
    """A calibrated input scale for ``x``: its amax over the format's largest value."""

    return (x.float().abs().amax() / maximum).reshape(())


def static_fp8_product(
    x: torch.Tensor, weight: torch.Tensor, alpha: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``y = (q W) * (input_scale * alpha)`` in FP64, each column by its slot's ``alpha``, and
    the bound of a BF16 computation of it (operands and output rounded once to BF16)."""

    q = numerics.quantize_static_activation(x, scale).double()
    y = (q @ weight.double().t()) * (scale.double() * alpha.double())
    magnitude = (q.abs() @ weight.double().abs().t()) * (scale.double() * alpha.double()).abs()
    return y, 2**-8 * magnitude + 2**-8 * y.abs() + 1e-6


def nvfp4_product(
    x: torch.Tensor,
    weights: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The contract's product of ``x`` with each slot's NVFP4 weight, concatenated, in FP64,
    and the bound of a BF16 computation of it (operands and output rounded once to BF16)."""

    codes, block = numerics.quantize_nvfp4_activation(x, scale)
    x_hat = numerics.dequantize_nvfp4_activation(codes, block, scale).double()
    w_hat = torch.cat([numerics.dequantize_nvfp4_weight(*weight) for weight in weights]).double()
    y = x_hat @ w_hat.t()
    return y, 2**-8 * (x_hat.abs() @ w_hat.abs().t()) + 2**-8 * y.abs() + 1e-6
