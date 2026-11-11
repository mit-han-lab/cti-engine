# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/moe/fused.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cti.moe import BaseMoeBackend, SparseBlock, TrainingBlock
from ctikernels import moe_align_block_size
from ctikernels.moe import MoEBlock, fused_experts, route

if TYPE_CHECKING:
    from cti.moe import ExpertLoRA
    from ctikernels.moe.lora import MoELoRAWeights


def _tile(num_tokens: int, num_experts: int) -> dict[str, int]:
    """The grouped GEMMs' Triton tile: a smaller one when the experts outnumber the rows."""

    if num_tokens <= num_experts:
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """The SiLU experts' grouped GEMMs, the router weights applied to the second's outputs.
    Writes over ``hidden_states``."""

    from ctikernels import fused_moe_kernel_triton, moe_sum_reduce_triton
    from ctikernels.activation import inference as inference_activation

    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, top_k = topk_ids.shape
    num_experts, width, _ = w1.shape
    config = _tile(num_tokens, num_experts)
    device, dtype = hidden_states.device, hidden_states.dtype
    # The gate/up output and the down output share one buffer: the first is dead once
    # activated.
    cache = torch.empty(num_tokens * top_k * max(width, w2.shape[1]), device=device, dtype=dtype)
    gate_up = cache[: num_tokens * top_k * width].view(num_tokens, top_k, width)
    activated = torch.empty((num_tokens * top_k, width // 2), device=device, dtype=dtype)
    down = cache[: num_tokens * top_k * w2.shape[1]].view(num_tokens, top_k, w2.shape[1])

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts
    )
    fused_moe_kernel_triton(
        hidden_states,
        w1,
        gate_up,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
        config=config,
        compute_type=dtype,
    )
    inference_activation.silu_and_mul(gate_up.view(-1, width), activated)
    fused_moe_kernel_triton(
        activated,
        w2,
        down,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=config,
        compute_type=dtype,
    )
    moe_sum_reduce_triton(down, hidden_states)
    return hidden_states


def compose(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """``fused_experts_impl`` over global routes: a route to another rank's expert runs on
    local expert 0 at weight exactly 0."""

    if expert_map is not None:
        topk_weights, topk_ids = local_routes(topk_weights, topk_ids, expert_map)
    return fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids)


def local_routes(
    topk_weights: torch.Tensor, topk_ids: torch.Tensor, expert_map: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local ids, and the weights of the other ranks' routes set to 0 (the rest keep theirs)."""

    local_ids = expert_map[topk_ids.long()]
    held = local_ids >= 0
    return topk_weights.where(held, 0.0), local_ids.where(held, 0)


class FusedMoe(BaseMoeBackend):
    """cti-kernels' experts (``fused_experts``) and whole-block op (``MoEBlock``).

    ``fused_experts`` runs grouped GEMMs on tiles tuned per GPU and expert shape, where a
    route the map marks -1 takes no slot or block; it takes BF16 alone, so other dtypes run
    the composition. Each call takes its intermediates from the caching allocator, so a
    captured graph keeps its own: a persistent ``ctikernels.moe.Experts`` would drop the
    buffers a graph captured once a larger eager batch grows them."""

    def __init__(self) -> None:
        # One op serves every layer, which run one after another: its workspaces and
        # intermediate buffers persist, so they are held once.
        self._block = MoEBlock()

    def sparse_block(self, top_k: int, renormalize: bool, dtype: torch.dtype) -> SparseBlock | None:
        """cti-kernels' ``MoEBlock``, which routes top-8 with renormalized weights over BF16
        weights: a fused kernel for few rows on the GPUs it dispatches to, otherwise a
        composition that runs the shared expert on a second stream beside the routed experts
        and adds it into their FP32 sum."""

        if (top_k, renormalize, dtype) != (8, True, torch.bfloat16):
            return None
        return self._block

    def training_block(
        self, top_k: int, renormalize: bool, dtype: torch.dtype
    ) -> TrainingBlock | None:
        """cti-kernels' ``moe_block`` over BF16 weights, replaying the experts ``route``
        selects."""

        if dtype != torch.bfloat16:
            return None
        return _TrainingBlock(self, top_k, renormalize)

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
        """cti-kernels' ``fused_experts_lora``, whose autograd gives the input, router-weight
        and factor gradients (straight-through over E4M3 stacks, which need ``lora``)."""

        if scales is not None and lora is None:
            raise ValueError("E4M3 expert stacks run an adapted call only")
        weights, selected = self.training_route(hidden_states, router, topk, renormalize)
        return _kernel_experts(
            hidden_states, w1, w2, weights, selected, expert_map, lora, scales, detach=False
        )

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
        """cti-kernels' ``fused_experts_lora_training_bytes``: every factor and the input
        trainable."""

        from ctikernels.moe.lora import fused_experts_lora_training_bytes

        return fused_experts_lora_training_bytes(
            tokens,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            top_k=top_k,
            rank=rank,
            local_only=local_only,
            device=device,
        )

    def route(
        self, hidden_states: torch.Tensor, router: torch.Tensor, topk: int, renormalize: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The whole-block op's router arithmetic, which no row count changes: a composed block
        # and training route each token as the op and any other batch would.
        return route(hidden_states, router, topk, renormalize=renormalize)

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
        topk_weights, topk_ids = self.route(hidden_states, router, topk, renormalize)
        if lora is not None:
            # cti-kernels' fused_experts_lora over BF16 or E4M3 stacks; inference reads the
            # trainable factors as frozen views; it joins the shared expert too.
            return _kernel_experts(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                expert_map,
                lora,
                scales,
                detach=True,
                shared=shared,
                shared_gate=shared_gate,
            )
        if shared is not None:
            raise ValueError("only an adapted call joins the shared expert")
        if scales is not None:
            raise ValueError("E4M3 expert stacks run an adapted call only")
        if hidden_states.dtype != torch.bfloat16:
            return compose(hidden_states, w1, w2, topk_weights, topk_ids, expert_map)
        return fused_experts(
            hidden_states, w1, w2, topk_weights, topk_ids, expert_map=expert_map, out=hidden_states
        )


def _kernel_factors(lora: ExpertLoRA, *, detach: bool) -> MoELoRAWeights:
    """``lora``'s routed factors as cti-kernels takes them, frozen views when ``detach``: the
    shared-A layout's four factors (which train), or a per-projection adapter's six
    (``MoELoRAWeights.separate``, served only)."""

    from ctikernels.moe.lora import MoELoRAWeights

    def view(factor: torch.Tensor) -> torch.Tensor:
        return factor.detach() if detach else factor

    down = tuple(view(factor) for factor in lora.down)
    if (shared := lora.gate_up_shared) is not None:
        return MoELoRAWeights(*(view(factor) for factor in shared), *down)
    gate_a, gate_b, up_a, up_b = (view(factor) for factor in lora.gate_up)
    return MoELoRAWeights.separate(gate_a, up_a, gate_b, up_b, *down)


def _kernel_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    lora: ExpertLoRA | None,
    scales: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    detach: bool,
    shared: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """The routed experts over global routes ``topk_ids`` with ``lora``'s deltas (frozen
    views when ``detach``) by cti-kernels' ``fused_experts_lora``, over model-dtype stacks or,
    with ``scales``, E4M3 stacks and their 128x128 block multipliers (the rows quantized 1x128
    per token, the LoRA branch in BF16). Differentiable in the rows, the router weights and
    the factors when they require gradients, straight-through the activation quantizers.
    CUDA only."""

    from ctikernels.moe import to_local_topk_ids
    from ctikernels.moe.lora import fused_experts_lora

    local = expert_map is None
    return fused_experts_lora(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids if local else to_local_topk_ids(topk_ids, expert_map),
        None if lora is None else _kernel_factors(lora, detach=detach),
        scaling=1.0 if lora is None else lora.scaling,
        local_only=local,
        num_global_experts=None if local else len(expert_map),
        w13_scale=None if scales is None else scales[0],
        w2_scale=None if scales is None else scales[1],
        shared=shared,
        shared_gate=shared_gate,
    )


class _TrainingBlock:
    """``FusedMoe``'s whole training block: cti-kernels' ``moe_block``, one autograd node
    differentiable in its input, replaying the experts the backend's ``route`` selects, so
    a training forward routes as inference does; the op weighs them from the current logits."""

    def __init__(self, backend: FusedMoe, top_k: int, renormalize: bool) -> None:
        self._backend = backend
        self._top_k = top_k
        self._renormalize = renormalize

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router: torch.Tensor,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        shared_gate_up: torch.Tensor | None,
        shared_down: torch.Tensor | None,
        shared_gate: torch.Tensor | None,
        /,
        *,
        expert_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from ctikernels.moe import moe_block

        with torch.no_grad():
            _, selected = self._backend.route(hidden_states, router, self._top_k, self._renormalize)
        output, _ = moe_block(
            hidden_states,
            router,
            shared_gate,
            gate_up,
            down,
            shared_gate_up,
            shared_down,
            top_k=self._top_k,
            expert_map=expert_map,
            routing=selected.to(torch.uint8),
            norm_topk_prob=self._renormalize,
        )
        return output

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
        from ctikernels.moe import moe_block_training_bytes

        return moe_block_training_bytes(
            tokens,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            top_k=self._top_k,
            shared_intermediate=shared_intermediate,
            device=device,
            norm_topk_prob=self._renormalize,
        )
