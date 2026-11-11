# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Semantics follow SGLang 0.5.22 (`python/sglang/srt/models/dflash.py`) and vLLM 0.31.0
# (`vllm/model_executor/models/qwen3_dflash2.py`); written for cti, no code reused.

"""DFlash2's dynamic block convolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import BaseOP
from .linear import LinearReplicated

if TYPE_CHECKING:
    from .site import Site


class BlockConv(BaseOP):
    """A two-sided dynamic depthwise convolution along each round block of ``block_size``
    rows, the round's width, which may differ from the block the draft was trained on (as
    SGLang and vLLM run it: the convolution restarts at each runtime block).

    Row n of a side's output is ``sum_t (base[t] + delta[n, t, s]) * x[n - t]`` per channel
    of segment s (``segment_size`` channels), with taps that reach before the block start reading zero. ``prepare``
    convolves a branch's input and returns the output side's deltas, which ``finish``
    applies to the branch's output; one projection of the input yields both sides'.
    """

    def __init__(
        self, hidden_size: int, taps: int, segment_size: int, block_size: int, *, site: Site
    ):
        self.taps = taps
        self.num_segments = hidden_size // segment_size
        self.block_size = block_size
        # [input/output side, tap, channel].
        self.base_kernel = torch.empty(2, self.taps, hidden_size)
        self.kernel_projection = LinearReplicated(
            hidden_size, 2 * self.taps * self.num_segments, has_bias=False, site=site
        )

    def prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        deltas = self.kernel_projection.forward(x).view(-1, 2, self.taps, self.num_segments)
        return self._convolve(x, deltas[:, 0], 0), deltas[:, 1]

    def finish(self, y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        return self._convolve(y, deltas, 1)

    def _convolve(self, x: torch.Tensor, deltas: torch.Tensor, side: int) -> torch.Tensor:
        """One side's convolution: cti-kernels' ``block_conv`` on CUDA (one launch, the same
        bits), the composed FP32 form elsewhere."""

        if x.is_cuda:
            from ctikernels.speculative import block_conv

            return block_conv(x, deltas, self.base_kernel[side], self.block_size)
        q, taps, segments = self.block_size, self.taps, self.num_segments
        # FP32 [blocks, Q, taps, segments, segment size] coefficients.
        kernel = self.base_kernel[side].float().view(1, 1, taps, segments, -1)
        kernel = kernel + deltas.float().view(-1, q, taps, segments, 1)
        rows = x.float().view(-1, q, segments, kernel.shape[-1])
        out = kernel[:, :, 0] * rows
        for tap in range(1, taps):
            # Row i reads row i - tap of its own block; the first ``tap`` rows read zero.
            out[:, tap:] += kernel[:, tap:, tap] * rows[:, :-tap]
        return out.view_as(x).to(x.dtype)


__all__ = ["BlockConv"]
