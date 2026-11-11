# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors

"""The training router: the experts the backend's inference router selects, weighed
differentiably.

Routing selects as transformers 5.16.1's routers do (``Qwen3MoeTopKRouter``,
``Qwen3_5MoeTopKRouter``): softmax, top-k and renormalization in FP32. The weights stay FP32,
as the inference router (``ctikernels.moe.route``) gives them, so a training forward weighs
the routes as the rollout that sampled its tokens did; Transformers rounds them to the router
logits' dtype. FP64 logits stay FP64 for gradcheck. Autograd reaches the input through the
selected weights' softmax, whose denominator couples every router logit unless the weights
are renormalized. Written from that reference; no code is copied.
"""

from __future__ import annotations

import torch


def route(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool,
    selected: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weights, selected)``, each ``[tokens, top_k]``: the most probable experts, or the
    given ``selected`` ones, and their differentiable FP32 probabilities (FP64 for FP64
    logits)."""

    accumulate = torch.float64 if router_logits.dtype == torch.float64 else torch.float32
    probabilities = router_logits.softmax(-1, dtype=accumulate)
    if selected is None:
        selected = probabilities.detach().topk(top_k, dim=-1).indices
    weights = probabilities.gather(-1, selected)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return weights, selected


__all__ = ["route"]
