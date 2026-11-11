"""Block-FP8 test weights and the contract's product (``cti.layers.quant.numerics``)."""

from __future__ import annotations

import torch

from cti.layers.quant import numerics


def quantized_weight(
    rows: int, cols: int, *, seed: int = 0, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """An E4M3 weight and its FP32 128x128 scale grid, as a block-FP8 checkpoint stores them
    (scales BF16-exact)."""

    generator = torch.Generator().manual_seed(seed)
    grid = (-(-rows // 128), -(-cols // 128))
    scale = (torch.rand(grid, generator=generator) + 0.5).bfloat16().float() / 448
    full = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    weight = (torch.randn(rows, cols, generator=generator) * full * 448 / 4).clamp(-448, 448)
    return (weight / full).to(numerics.E4M3).to(device), scale.to(device)


def contract_product(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``y = sum_g (q W)_g s_g S (+ bias)`` in FP64, the contract's product, and the bound of
    a BF16 computation of it: operands and output each rounded once to BF16 (2^-9 relative),
    the sum accumulated in FP32."""

    q, s = numerics.quantize_activation(x)
    x_hat = q.double() * s.double().repeat_interleave(128, -1)[..., : x.shape[-1]]
    w_hat = numerics.dequantize_block_weight(weight, scale).double()
    y = x_hat @ w_hat.t()
    if bias is not None:
        y = y + bias.double()
    bound = 2**-8 * (x_hat.abs() @ w_hat.abs().t()) + 2**-8 * y.abs() + 1e-6
    return y, bound
