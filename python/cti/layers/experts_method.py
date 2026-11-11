"""How the routed experts store their stacks and run them: ``ExpertsMethod`` and the model
dtype's ``Bf16Experts``; the format methods in ``layers/quant`` replace it."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar

import torch

from .linear_method import GemmCallSite
from .placement import WeightShard

if TYPE_CHECKING:
    from .moe import Experts


class ExpertsMethod:
    """One MoE layer's stacked expert weights and their routed product.

    ``declare`` returns the stacks the op holds for its local experts (``gate_up_proj``
    ``[E, 2I, H]``, ``down_proj`` ``[E, H, I]`` and any scales, each stacked by expert so the
    placement's expert window loads them); ``call_sites`` are what ``prepare_inference`` prepares;
    ``apply`` routes and runs the experts.
    """

    # Whether the stacks hold the model dtype's weights as multiplied (``LinearMethod.dense``).
    dense: ClassVar[bool] = False

    def __init__(self, experts: Experts) -> None:
        self._experts = experts

    @classmethod
    def computed_by(cls, device: torch.device) -> str:
        """The row of the method's provider table that computes it on ``device``, by name;
        a device without one raises ``ValueError``."""
        raise NotImplementedError

    def declare(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def call_sites(self) -> Iterator[GemmCallSite]:
        return iter(())

    def windows(self, leaf: str, windows: tuple[WeightShard, ...]) -> tuple[WeightShard, ...]:
        """The checkpoint windows of stack ``leaf`` for the experts' intermediate windows;
        the same by default. A window the stack cannot take raises ``ValueError``."""

        return windows

    def joins_shared(self) -> bool:
        """Whether ``apply`` joins the block's gated shared expert into the routed sum, given
        its output and gate logits; none does by default."""

        return False

    def apply(
        self,
        hidden_states: torch.Tensor,
        router: torch.Tensor,
        shared: torch.Tensor | None = None,
        shared_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This rank's share of the weighted routed experts of ``hidden_states`` ``[T, H]``,
        each row routed by its logits under the frozen ``router`` ``[num_experts, H]``, with
        the experts' LoRA; a route to another rank's expert adds 0. ``shared`` and
        ``shared_gate`` (where ``joins_shared``) join in vLLM's order. It may write over
        ``hidden_states``."""

        raise NotImplementedError

    def training(self, hidden_states: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        """``apply`` differentiable in ``hidden_states`` and the LoRA factors, out of place."""

        raise NotImplementedError

    def training_bytes(self, tokens: int, *, rank: int) -> tuple[int, int]:
        """``(saved, backward)`` of ``tokens`` rows through ``training`` at LoRA ``rank``, as
        the MoE backend sizes its training (``BaseMoeBackend.training_bytes``)."""

        experts = self._experts
        return experts.moe_backend.training_bytes(
            tokens,
            hidden=experts.hidden_size,
            intermediate=len(experts.placement.intermediate),
            experts=len(experts.placement.experts),
            top_k=experts.top_k,
            rank=rank,
            local_only=experts.expert_map is None,
            device=experts.gate_up_proj.device,
        )


class Bf16Experts(ExpertsMethod):
    """Stacks of the model dtype, run by the MoE backend."""

    dense = True

    def declare(self) -> dict[str, torch.Tensor]:
        experts = self._experts
        count, width = len(experts.placement.experts), len(experts.placement.intermediate)
        return {
            "gate_up_proj": torch.empty(count, 2 * width, experts.hidden_size),
            "down_proj": torch.empty(count, experts.hidden_size, width),
        }

    def joins_shared(self) -> bool:
        """An adapted call on CUDA: cti-kernels' fused_experts_lora joins the shared expert in
        its combine."""

        experts = self._experts
        return experts.lora is not None and experts.gate_up_proj.device.type == "cuda"

    def apply(
        self,
        hidden_states: torch.Tensor,
        router: torch.Tensor,
        shared: torch.Tensor | None = None,
        shared_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        experts = self._experts
        return experts.moe_backend.forward(
            hidden_states=hidden_states,
            w1=experts.gate_up_proj,
            w2=experts.down_proj,
            router=router,
            topk=experts.top_k,
            renormalize=experts.renormalize,
            expert_map=experts.expert_map,
            lora=experts.lora,
            shared=shared,
            shared_gate=shared_gate,
        )

    def training(self, hidden_states: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        experts = self._experts
        return experts.moe_backend.forward_training(
            hidden_states=hidden_states,
            w1=experts.gate_up_proj,
            w2=experts.down_proj,
            router=router,
            topk=experts.top_k,
            renormalize=experts.renormalize,
            expert_map=experts.expert_map,
            lora=None if experts.lora is None else experts.lora.training(),
        )


__all__ = ["Bf16Experts", "ExpertsMethod"]
