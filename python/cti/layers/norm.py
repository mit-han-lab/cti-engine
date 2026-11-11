# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/layers/norm.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Protocol

import torch
import torch.nn.functional as F

from ctikernels.norm import inference as inference_norm

from .base import BaseOP


class LayerNorm(BaseOP):
    """Affine layer normalization over the last dimension, with a weight and a bias."""

    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class ResidualReduction(Protocol):
    """Sums ``x``'s shares into ``residual`` and normalizes by ``weight``, in place, returning
    ``(x, residual)`` as the norm and the stream."""

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def prepare(self, token_counts: Iterable[int]) -> None: ...


class _RMSNormBase(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def _training_norm(self) -> ModuleType:
        from ctikernels.norm import training as training_norm

        return training_norm

    def _warm_up_adds(
        self,
        max_rows: int,
        warmed: set,
        *,
        dtype: torch.dtype,
        device: torch.device,
        add: Callable[[torch.Tensor, torch.Tensor], object],
    ) -> None:
        """Run ``add`` on every row count up to ``max_rows``: cti-kernels' fused add + norm
        compiles a kernel per row band and has no warm-up of its own."""

        key = (type(self), self.weight.shape, self.weight.dtype, dtype)
        if key in warmed:
            return
        warmed.add(key)
        size = self.weight.shape[0]
        x = torch.zeros(max_rows, size, dtype=dtype, device=device)
        residual = torch.zeros(max_rows, size, dtype=dtype, device=device)
        for rows in range(1, max_rows + 1):
            add(x[:rows], residual[:rows])

    @staticmethod
    def _warm_up_reduction(reduction: ResidualReduction | None, max_rows: int, warmed: set) -> None:
        """Compile ``reduction`` for every row count up to ``max_rows``, without
        communicating."""

        if reduction is not None and ("reduction", id(reduction)) not in warmed:
            warmed.add(("reduction", id(reduction)))
            reduction.prepare(range(1, max_rows + 1))


class RMSNorm(_RMSNormBase):
    """RMS norm over the last dimension, scaled by the stored weight.

    The scale is ``weight``, or ``1 + weight`` with ``zero_centered=True``, the
    ``zero_centered`` of ``ctikernels.norm``; the stored weight is never
    pre-incremented. ``zero_centered=True`` computes what HF ``Qwen3_5RMSNorm`` and
    vLLM/SGLang ``GemmaRMSNorm`` compute.
    """

    def __init__(self, size: int, eps: float, *, zero_centered: bool = False) -> None:
        super().__init__(size, eps)
        self.zero_centered = zero_centered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return inference_norm.rms_norm(x, self.weight, self.eps, zero_centered=self.zero_centered)

    def forward_inplace(self, x: torch.Tensor) -> None:
        inference_norm.rms_norm_(x, self.weight, self.eps, zero_centered=self.zero_centered)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        return self._training_norm().rms_norm(
            x, self.weight, self.eps, zero_centered=self.zero_centered
        )


class RMSNormFused(_RMSNormBase):
    """The RMSNorms of a residual stream scaled by the stored weight (Qwen3 and Qwen3-MoE),
    on cti-kernels' default path.

    Without ``residual``, ``x`` is the stream: its norm comes back beside ``x`` itself. With
    ``reduction``, inference's ``x`` beside a residual is this rank's share of the branch
    output, which the reduction sums over the ranks with the add.
    """

    # Training saves the FP32 inverse RMS per row.
    TRAINING_SAVED_BYTES = 4

    def __init__(
        self, size: int, eps: float, *, reduction: ResidualReduction | None = None
    ) -> None:
        super().__init__(size, eps)
        self._reduction = reduction

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return inference_norm.rms_norm(x, self.weight, self.eps), x
        if self._reduction is not None:
            return self._reduction(x, residual, self.weight)
        return inference_norm.add_rms_norm_(x, residual, self.weight, self.eps)

    def warm_up_rows(
        self, max_rows: int, *, dtype: torch.dtype, device: torch.device, warmed: set
    ) -> None:
        def add(x: torch.Tensor, residual: torch.Tensor) -> None:
            inference_norm.add_rms_norm_(x, residual, self.weight, self.eps)

        self._warm_up_adds(max_rows, warmed, dtype=dtype, device=device, add=add)
        self._warm_up_reduction(self._reduction, max_rows, warmed)

    def forward_training(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The norm of the stream ``x``, and ``x``."""

        return self._training_norm().rms_norm(x, self.weight, self.eps), x


class ResidualStreamRMSNorm(_RMSNormBase):
    """Zero-centered RMSNorm of a residual stream, scaled by ``1 + weight``.

    Inference adds in FP32, stores the sum in the stream dtype and normalizes the FP32
    sum; training normalizes the stream it is given. Operands must be contiguous
    and 16-byte aligned. With ``reduction``, inference's ``x`` beside a residual is this
    rank's share of the branch output, which the reduction sums over the ranks with the add.
    """

    # Training saves nothing per row beside its input.
    TRAINING_SAVED_BYTES = 0

    def __init__(
        self, size: int, eps: float, *, reduction: ResidualReduction | None = None
    ) -> None:
        super().__init__(size, eps)
        self._reduction = reduction

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Overwrite ``residual`` with the sum and ``x`` with its FP32-sum norm.

        Without ``residual``, return a new norm output and ``x`` unchanged.
        """

        if residual is None:
            normalized = inference_norm.residual_rms_norm(
                x, self.weight, self.eps, zero_centered=True, out_dtype=self.weight.dtype
            )
            return normalized, x
        if self._reduction is not None:
            return self._reduction(x, residual, self.weight)
        return inference_norm.add_rms_norm_(x, residual, self.weight, self.eps, zero_centered=True)

    def forward_last(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Overwrite ``x`` with the FP32-sum norm; leave ``residual`` unchanged, except that
        a reduction stores the sum there."""

        if self._reduction is not None:
            return self._reduction(x, residual, self.weight)[0]
        normalized, _ = inference_norm.add_rms_norm_(
            x, residual, self.weight, self.eps, zero_centered=True, store_residual=False
        )
        return normalized

    def warm_up_rows(
        self, max_rows: int, *, dtype: torch.dtype, device: torch.device, warmed: set
    ) -> None:
        def add(x: torch.Tensor, residual: torch.Tensor) -> None:
            for store in (True, False):
                inference_norm.add_rms_norm_(
                    x, residual, self.weight, self.eps, zero_centered=True, store_residual=store
                )

        self._warm_up_adds(max_rows, warmed, dtype=dtype, device=device, add=add)
        self._warm_up_reduction(self._reduction, max_rows, warmed)

    def forward_training(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The norm of the stream ``x``, and ``x``."""

        normalized = self._training_norm().rms_norm(
            x, self.weight, self.eps, zero_centered=True, out_dtype=self.weight.dtype
        )
        return normalized, x
