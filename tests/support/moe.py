"""MoE test backends: ``ReferenceMoe``, the routed experts on CPU with their per-expert LoRA,
training through the torch oracle ``routed_experts``, and ``BlockReferenceMoe``, the same with a whole-block op; ``rank_state``, a rank's share of a
whole layer's state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from cti.moe import BaseMoeBackend, ExpertLoRA, SparseBlock
from cti.moe.fused import FusedMoe
from tests.support.expert_lora import expert_lora_deltas

if TYPE_CHECKING:
    from cti.layers import BaseOP


class _ExpertDeltas:
    """``lora``'s deltas expert by expert. Each stack is split into per-expert views once,
    so its gradient gathers into one tensor rather than one full-size tensor per expert."""

    def __init__(self, lora: ExpertLoRA) -> None:
        self._scaling = lora.scaling
        self._gate_up = tuple(factor.unbind(0) for factor in lora.gate_up)
        self._down = tuple(factor.unbind(0) for factor in lora.down)

    def gate_up(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        return self._delta(x, self._gate_up, expert)

    def down(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        return self._delta(x, self._down, expert)

    def _delta(
        self, x: torch.Tensor, factors: tuple[tuple[torch.Tensor, ...], ...], expert: int
    ) -> torch.Tensor:
        """``scaling * B @ A @ x`` for each (A, B) pair, concatenated in pair order."""

        pairs = zip(factors[0::2], factors[1::2], strict=True)
        deltas = [F.linear(F.linear(x, a[expert]), b[expert]) for a, b in pairs]
        return torch.cat(deltas, -1) * self._scaling


def routed_experts(
    x: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    weights: torch.Tensor,
    selected: torch.Tensor,
    lora: ExpertLoRA | None = None,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """``sum_k weights[t, k] * expert_{selected[t, k]}(x[t])`` with SiLU-gated experts
    ``gate_up [E, 2I, H]`` and ``down [E, H, I]``, each projection with its ``lora`` delta
    if given, summed over ``k`` in FP32 and rounded once to ``x.dtype``, as the inference
    reduction does.

    ``expert_map`` maps each global expert to its index in the stacks, -1 for experts held
    elsewhere: their routes add 0, as in inference. Stacks that hold a window of every
    expert's intermediate width give that window's partial sum.

    The training oracle: assignments run grouped by expert, one pair of GEMMs per expert
    hit, in torch autograd. Every gather is a permutation, so neither pass accumulates with
    atomics: repeated runs agree bit for bit.
    """

    if gate_up.requires_grad or down.requires_grad:
        raise ValueError("routed expert weights must be frozen")
    tokens, top_k = selected.shape
    hidden_size = x.shape[-1]
    accumulate = torch.float64 if x.dtype == torch.float64 else torch.float32
    flat = selected.flatten()
    if expert_map is not None:
        flat = expert_map[flat].long()
    # Routes to other ranks' experts (-1) sort first and are dropped.
    order = flat.argsort(stable=True)
    counts = torch.bincount(flat + 1, minlength=gate_up.shape[0] + 1).tolist()
    order = order[counts.pop(0) :]
    # Row t * top_k + k of the expanded input is token t's k-th assignment.
    assigned = x.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size).index_select(0, order)
    deltas = None if lora is None else _ExpertDeltas(lora)
    outputs = []
    for expert, rows in enumerate(assigned.split(counts)):
        if not len(rows):
            continue
        projected = F.linear(rows, gate_up[expert])
        if deltas is not None:
            projected = projected + deltas.gate_up(rows, expert)
        gate, up = projected.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        routed = F.linear(activated, down[expert])
        if deltas is not None:
            routed = routed + deltas.down(activated, expert)
        outputs.append(routed)
    if not outputs:
        # No route stays on this rank: zeros, still connected to the input's graph.
        return x * 0
    weighted = torch.cat(outputs).to(accumulate) * weights.flatten()[order, None].to(accumulate)
    by_route = weighted.new_zeros(tokens * top_k, hidden_size).index_copy(0, order, weighted)
    return by_route.view(tokens, top_k, hidden_size).sum(1).to(x.dtype)


class ReferenceMoe(BaseMoeBackend):
    """``FusedMoe``'s contract on CPU: the router's logits in FP32, the top ``topk`` experts by
    softmax probability, their weights renormalized on request, SiLU-gated experts with their
    per-expert LoRA deltas, and the weighted sum of the routes to held experts written over
    ``hidden_states``, joined with a given gated shared expert in vLLM's order."""

    def route(
        self, hidden_states: torch.Tensor, router: torch.Tensor, topk: int, renormalize: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states.float(), router.float())
        weights, ids = torch.softmax(logits, dim=-1).topk(topk, dim=-1)
        if renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, ids

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
        shared: torch.Tensor | None = None,
        shared_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights, ids = self.route(hidden_states, router, topk, renormalize)
        if expert_map is not None:
            assert expert_map.dtype == torch.int32 and len(expert_map) == router.shape[0]
            ids = expert_map[ids]
        output = hidden_states.new_zeros(hidden_states.shape, dtype=torch.float32)
        for expert in (ids[ids >= 0]).unique().tolist():
            rows, slots = (ids == expert).nonzero(as_tuple=True)
            x = hidden_states[rows]
            gate_up = F.linear(x, w1[expert])
            if lora is not None:
                gate_up = gate_up + expert_lora_deltas(lora, x, expert, "gate_up")
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            routed = F.linear(activated, w2[expert])
            if lora is not None:
                routed = routed + expert_lora_deltas(lora, activated, expert, "down")
            output.index_add_(0, rows, routed.float() * weights[rows, slots, None])
        routed = hidden_states.copy_(output)
        if shared is not None:
            # r(r(r(sigmoid(gate)) * shared) + routed).
            routed.add_(shared * torch.sigmoid(shared_gate))
        return routed

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
        if scales is not None:
            raise ValueError("ReferenceMoe trains no E4M3 expert stacks")
        weights, selected = self.training_route(hidden_states, router, topk, renormalize)
        return routed_experts(hidden_states, w1, w2, weights, selected, lora, expert_map)


class ComposedFusedMoe(FusedMoe):
    """The fused backend as it runs an adapted block, without the adapter: no whole-block op,
    the routed experts on cti-kernels' ``fused_experts``, whose bits ``fused_experts_lora``
    keeps before it adds a delta."""

    def sparse_block(self, top_k: int, renormalize: bool, dtype: torch.dtype) -> None:
        return None


def rank_state(state: Mapping[str, torch.Tensor], layer: BaseOP) -> dict[str, torch.Tensor]:
    """``layer``'s share of a whole layer's ``state``, as the loader reads it: its experts'
    stacks and its checkpoint windows."""

    local = {}
    for param in layer.params():
        tensor = state[param.path]
        if param.experts is not None:
            tensor = tensor[param.experts.start : param.experts.stop]
        if windows := param.windows:
            axis = windows[0].axis
            parts = tensor.chunk(len(windows), dim=axis)
            tensor = torch.cat([w.select(p) for w, p in zip(windows, parts, strict=True)], axis)
        local[param.path] = tensor
    return local


class BlockReferenceMoe(ReferenceMoe):
    """``ReferenceMoe`` offering a whole-block op that composes the same reference from the
    weights a layer hands it, and counting the requests for it and its calls."""

    def __init__(self) -> None:
        self.requests: list[tuple[int, bool, torch.dtype]] = []
        self.calls = 0

    def sparse_block(self, top_k: int, renormalize: bool, dtype: torch.dtype) -> SparseBlock:
        self.requests.append((top_k, renormalize, dtype))

        def block(
            x, router, gate_up, down, shared_gate_up, shared_down, shared_gate, *, expert_map
        ):
            self.calls += 1
            routed = self.forward(
                x.clone(), gate_up, down, router, top_k, renormalize, expert_map=expert_map
            )
            gate, up = F.linear(x, shared_gate_up).chunk(2, dim=-1)
            shared = F.linear(F.silu(gate) * up, shared_down)
            return routed + shared * torch.sigmoid(F.linear(x, shared_gate))

        return block
