from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ctikernels.activation import inference as inference_activation

from .base import StateLessOP

if TYPE_CHECKING:
    from .linear import LinearColParallelMerged, LinearRowParallel
    from .linear_method import FrozenBase


class SiluAndMul(StateLessOP):
    """SwiGLU of a ``[..., 2 * width]`` gate | up input; ``width`` is this rank's."""

    def __init__(self, width: int) -> None:
        self.width = width

    def forward(
        self,
        x: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return inference_activation.silu_and_mul(x, out=out)

    def warm_up_rows(
        self, max_rows: int, *, dtype: torch.dtype, device: torch.device, warmed: set
    ) -> None:
        """Run every row count up to ``max_rows``: cti-kernels' SiLU compiles a kernel per
        row band and has no warm-up of its own."""

        key = (type(self), self.width, dtype)
        if key in warmed:
            return
        warmed.add(key)
        x = torch.zeros(max_rows, 2 * self.width, dtype=dtype, device=device)
        out = torch.empty(max_rows, self.width, dtype=dtype, device=device)
        for rows in range(1, max_rows + 1):
            self.forward(x[:rows], out=out[:rows])

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        from ctikernels.activation import training as training_activation

        return training_activation.silu_and_mul(x)

    def mlp_training(
        self, gate_up: LinearColParallelMerged, down: LinearRowParallel, x: torch.Tensor
    ) -> torch.Tensor:
        """The SwiGLU MLP's training product ``down(self(gate_up(x)))``, differentiable in ``x``
        and the factors; ``down``'s exit applies as in ``forward_training``. With LoRA on both
        projections of one rank, whose frozen bases the kernel can multiply, one cti-kernels op
        runs it and saves only ``x``, gate | up and the three low-rank inputs, recomputing the
        SwiGLU in its backward; otherwise the three products run as autograd nodes. A layer
        split over several ranks keeps the nodes until the op's partial input gradient has an
        exchange."""

        products = _fused_products(gate_up, down)
        if products is None:
            return down.forward_training(self.forward_training(gate_up.forward_training(x)))
        from ctikernels.mlp.training import gated_mlp_lora

        gate_up_lora, down_lora = gate_up.lora, down.lora
        return gated_mlp_lora(
            x,
            *products,
            gate_up_lora.training_factors,
            down_lora.training_factors,
            scaling=gate_up_lora.scaling,
        )

    @staticmethod
    def mlp_training_bytes(
        gate_up: LinearColParallelMerged, down: LinearRowParallel, tokens: int, *, rank: int
    ) -> tuple[int, int]:
        """``mlp_training``'s ``(saved, backward)`` over ``tokens`` rows at LoRA ``rank``. The
        nodes save the input, gate | up, the SwiGLU output and three low-rank inputs, and their
        SwiGLU backward holds gate | up's gradient; the fused op saves no SwiGLU output and its
        backward holds only the SwiGLU output's gradient, writing gate | up's over the saved
        gate | up."""

        hidden, width = down.local_output_size, gate_up.local_output_size // 2
        low_rank = 3 * 2 * rank
        if _fused_products(gate_up, down) is None:
            return tokens * (2 * hidden + 6 * width + low_rank), tokens * 4 * width
        return tokens * (2 * hidden + 4 * width + low_rank), tokens * 2 * width


def _fused_products(
    gate_up: LinearColParallelMerged, down: LinearRowParallel
) -> tuple[FrozenBase, FrozenBase] | None:
    """The two frozen bases the fused op multiplies, or None where it does not apply."""

    gate_up_lora, down_lora = gate_up.lora, down.lora
    if gate_up_lora is None or down_lora is None or gate_up.ranks > 1:
        return None
    # gate and up own one A each; the op reads (A_gate, B_gate, A_up, B_up) and (A_down, B_down).
    if len(gate_up_lora.output_slices) != 2 or len(down_lora.output_slices) != 1:
        return None
    if gate_up_lora.scaling != down_lora.scaling:
        return None
    gate_up_product, down_product = gate_up.frozen_product(), down.frozen_product()
    if gate_up_product is None or down_product is None:
        return None
    return gate_up_product, down_product


__all__ = ["SiluAndMul"]
