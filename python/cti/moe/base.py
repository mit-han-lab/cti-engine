# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/moe/base.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch
import torch.nn.functional as F

from .training import route


class ExpertLoRA(Protocol):
    """Unmerged LoRA on each routed expert one rank holds, its factors stacked by local
    expert as ``w1`` and ``w2`` are: A ``[E, r, in]`` and B ``[E, out, r]`` per projection.
    A row ``x`` that a route sends to local expert ``e`` gains ``scaling * B[e] @ A[e] @ x``
    in each projection."""

    @property
    def scaling(self) -> float: ...

    @property
    def gate_up(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(gate A, gate B, up A, up B)``: the gate half of ``w1``'s rows, then the up."""
        ...

    @property
    def gate_up_shared(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """In the shared-A layout, ``(A [E, r, H], B [2, E, I, r])``: one A for the gate and
        up halves and their B stacked slice-major (``gate_up`` views them); else None."""
        ...

    @property
    def down(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(A, B)`` of ``w2``."""
        ...

    def training(self) -> ExpertLoRA:
        """These factors as a training forward reads them: their bound FP32 masters
        (``layers.lora.Adapter.bind_training``), which the kernels cast at each use, or the
        factors themselves."""
        ...


class SparseBlock(Protocol):
    """One MoE layer's whole inference block: router, top-k, the routed SiLU experts and the
    sigmoid-gated shared expert, giving this rank's partial sum (the layer reduces it). The
    op picks its implementation per call, e.g. one fused kernel for few rows."""

    def __call__(
        self,
        hidden_states: torch.Tensor,  # [T, H]
        router: torch.Tensor,  # [E, H]
        gate_up: torch.Tensor,  # [E_local, 2I, H]
        down: torch.Tensor,  # [E_local, H, I]
        shared_gate_up: torch.Tensor,  # [2 SI_local, H]
        shared_down: torch.Tensor,  # [H, SI_local]
        shared_gate: torch.Tensor,  # [1, H]
        /,
        *,
        expert_map: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class TrainingBlock(Protocol):
    """One MoE layer's whole training block over frozen weights: ``SparseBlock``'s routes,
    experts and gated shared expert out of place, differentiable in ``hidden_states`` alone,
    giving this rank's partial sum and its partial input gradient. Without a shared expert,
    its three weights are None."""

    def __call__(
        self,
        hidden_states: torch.Tensor,  # [T, H]
        router: torch.Tensor,  # [E, H]
        gate_up: torch.Tensor,  # [E_local, 2I, H]
        down: torch.Tensor,  # [E_local, H, I]
        shared_gate_up: torch.Tensor | None,  # [2 SI_local, H]
        shared_down: torch.Tensor | None,  # [H, SI_local]
        shared_gate: torch.Tensor | None,  # [1, H]
        /,
        *,
        expert_map: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def training_bytes(
        self,
        tokens: int,
        *,
        hidden: int,
        intermediate: int,
        experts: int,
        shared_intermediate: int,
        device: torch.device,
    ) -> tuple[int, int]:
        """``(saved, backward)`` bytes of ``tokens`` rows through the op on ``device``, with
        ``experts`` local experts of width ``intermediate`` and a shared expert of width
        ``shared_intermediate`` (0: none), as the caching allocator holds them."""
        ...


class BaseMoeBackend(ABC):
    """The routed experts of a MoE block, in inference and in training."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router: torch.Tensor,
        topk: int,
        renormalize: bool,
        expert_map: torch.Tensor | None = None,
        lora: ExpertLoRA | None = None,
        scales: tuple[torch.Tensor, torch.Tensor] | None = None,
        shared: torch.Tensor | None = None,
        shared_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route each row to its ``topk`` experts by its logits under the frozen ``router``
        ``[E, H]`` over all experts, through SiLU-gated experts, the router weights applied
        to their outputs. A row's routes must not depend on the rows beside it.

        ``expert_map`` (int32, one entry per expert) gives the index in ``w1`` and ``w2`` of
        each expert they hold, or -1: routes to the others add nothing, and the weights stay
        normalized over all ``topk`` routes. ``None`` means they hold every expert.
        ``lora`` adds its deltas to the experts' projections, indexed as ``w1`` and ``w2``.
        ``scales`` marks ``w1`` and ``w2`` as E4M3 stacks with these 128x128 block
        multipliers ``[E, rows / 128, columns / 128]``, run as stored; with ``lora`` only.
        ``shared`` ``[T, H]`` and ``shared_gate`` ``[T, 1]``, the gated shared expert's output
        and gate logits, join the routed sum in vLLM's order,
        ``r(r(r(sigmoid(gate)) * shared) + routed)``; an adapted call takes them
        (``ExpertsMethod.joins_shared``).
        """

    def sparse_block(self, top_k: int, renormalize: bool, dtype: torch.dtype) -> SparseBlock | None:
        """A whole-block op for one layer with a gated shared expert, these routes and
        ``dtype`` weights, or None: the layer composes its block around ``forward``."""

        return None

    def training_block(
        self, top_k: int, renormalize: bool, dtype: torch.dtype
    ) -> TrainingBlock | None:
        """A whole-block training op for one layer whose weights are all frozen, with these
        routes and ``dtype`` weights, or None: the layer composes its block around
        ``forward_training``. It routes as ``route`` does."""

        return None

    @abstractmethod
    def route(
        self, hidden_states: torch.Tensor, router: torch.Tensor, topk: int, renormalize: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(weights, ids)``, each ``[T, topk]``: the experts ``forward`` routes each row to
        under the frozen ``router`` ``[E, H]``, and their FP32 weights."""

    def forward_training(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router: torch.Tensor,
        topk: int,
        renormalize: bool,
        expert_map: torch.Tensor | None = None,
        lora: ExpertLoRA | None = None,
        scales: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``forward``'s SiLU experts out of place, through the experts ``route`` selects
        weighed by ``training_route``, differentiable in ``hidden_states`` (through the
        experts and their router weights) and in ``lora``'s factors; ``scales`` as in
        ``forward``."""

        raise NotImplementedError(f"{type(self).__name__} does not train")

    def training_bytes(
        self,
        tokens: int,
        *,
        hidden: int,
        intermediate: int,
        experts: int,
        top_k: int,
        rank: int,
        local_only: bool,
        device: torch.device,
    ) -> tuple[int, int]:
        """``(saved, backward)`` bytes of ``tokens`` rows through ``forward_training`` on
        ``device`` over a model-dtype base of ``experts`` local experts of width
        ``intermediate``, with LoRA ``rank`` factors, as the caching allocator holds them;
        ``local_only`` when the stacks hold every expert."""

        raise NotImplementedError(f"{type(self).__name__} does not train")

    def training_route(
        self, hidden_states: torch.Tensor, router: torch.Tensor, topk: int, renormalize: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(weights, selected)`` for training: the experts ``route`` selects (int64) and
        their FP32 weights, differentiable in ``hidden_states`` through the router logits."""

        with torch.no_grad():
            _, selected = self.route(hidden_states, router, topk, renormalize)
        logits = F.linear(hidden_states, router)
        weights, _ = route(logits, topk, renormalize=renormalize, selected=selected.long())
        return weights, selected.long()
