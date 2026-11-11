# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/layers/moe.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from cti.distributed import Topology
from cti.utils import init_logger

from .activation import SiluAndMul
from .base import BaseOP
from .experts_method import Bf16Experts, ExpertsMethod
from .linear import (
    AllReduceExit,
    DeferredExit,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    LocalEntry,
    LocalExit,
    ParallelEntry,
)
from .linear_method import GemmCallSite, declare_method, require_meta
from .lora import ExpertLoRAFactors
from .placement import Param, WeightShard

logger = init_logger(__name__)

if TYPE_CHECKING:
    from cti.config import LoRAConfig
    from cti.moe import BaseMoeBackend, SparseBlock, TrainingBlock

    from .site import Site


class Experts(BaseOP):
    """The routed experts the site's placement gives this rank, stacked for the MoE backend,
    with the per-expert LoRA its block attaches."""

    # The per-expert LoRA its block attaches (``adapt``); none until then.
    lora: ExpertLoRAFactors | None = None

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        renormalize: bool,
        site: Site,
    ):
        self.placement = placement = site.experts.shard(
            num_experts, intermediate_size, Topology(site.group.rank, site.group.size)
        )
        self.top_k = top_k
        self.renormalize = renormalize
        self.services = site.services
        self.hidden_size = hidden_size
        self._owns_whole = site.group.rank == 0
        # None on a dry build, which only reads shapes.
        self.moe_backend: BaseMoeBackend | None = site.services.moe(site.device)
        # Routes still name global experts; a partial placement maps them to its own.
        self._expert_map = (
            None if placement.holds_every_expert else placement.expert_map().to(site.device)
        )
        self._declared: tuple[str, ...] = ()
        self.method, self._declared = declare_method(self, Bf16Experts, ())

    def use_method(self, method: type[ExpertsMethod]) -> None:
        """Store and run the stacks as ``method`` does: its tensors replace the ones declared
        before, on the device and in the dtype in effect, before weights load."""

        require_meta(tensor for _, tensor in self._tensors())
        self.method, self._declared = declare_method(self, method, self._declared)

    def inference_gemms(self) -> Iterator[GemmCallSite]:
        yield from self.method.call_sites()

    def adapt(self, config: LoRAConfig) -> ExpertLoRAFactors:
        """Add ``config``'s unmerged per-expert factors in its expert layout, placed as the
        expert stacks are, to every held expert's projections."""

        self.lora = ExpertLoRAFactors(
            self.placement,
            self.hidden_size,
            rank=config.rank,
            scaling=config.alpha / config.rank,
            owns_whole=self._owns_whole,
            layout=config.expert_layout,
        )
        return self.lora

    @property
    def expert_map(self) -> torch.Tensor | None:
        """int32 ``[num_experts]`` on the site's device: each held expert's index in the
        stacks, -1 for the others; None when this rank holds every expert."""

        return self._expert_map

    def _param(self, path: str, name: str, tensor: torch.Tensor) -> Param:
        placement = self.placement
        if placement.is_whole:
            return Param(path, tuple(tensor.shape), tensor.dtype)
        windows: tuple[WeightShard, ...] = ()
        if len(rows := placement.intermediate) < placement.intermediate_size:
            # Each expert's gate and up rows, or its down columns (and their scales').
            window = WeightShard(int(name.startswith("down_proj")), rows.start, rows.stop)
            windows = (window,) * (2 if name.startswith("gate_up_proj") else 1)
            try:
                windows = self.method.windows(name, windows)
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from None
        return Param(
            path, tuple(tensor.shape), tensor.dtype, windows=windows, experts=placement.experts
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router: torch.Tensor,
        shared: torch.Tensor | None = None,
        shared_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The weighted sum of this rank's routes by the frozen ``router`` ``[E, H]``; routes
        to other ranks' experts add 0. ``shared`` and ``shared_gate``, where the method
        ``joins_shared``, join it in vLLM's order."""

        return self.method.apply(hidden_states, router, shared, shared_gate)

    def forward_training(self, hidden_states: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        """This rank's partial sum through the experts inference routes to, differentiable in
        ``hidden_states`` and the LoRA factors, out of place."""

        return self.method.training(hidden_states, router)


class SharedExpert(BaseOP):
    """The SiLU-gated MLP that every token takes beside its routed experts, split over the
    ranks by its intermediate width (as SGLang 0.5.22 and vLLM 0.31.0 split it on CUDA): the
    down projection keeps its partial sum for the MoE block's single reduction."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, *, site: Site, lora: str | None = None
    ) -> None:
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size,
            [intermediate_size, intermediate_size],
            has_bias=False,
            site=site,
            output_names=("gate", "up"),
            lora=lora,
        )
        self.activation = SiluAndMul(self.gate_up_proj.local_output_size // 2)
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False, site=site, reduce=False, lora=lora
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(self.activation.forward(self.gate_up_proj.forward(x)))

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation.mlp_training(self.gate_up_proj, self.down_proj, x)


class MoELayer(BaseOP):
    """A sparse MoE block: a replicated router over the routed experts the site gives this
    rank and, with ``shared_expert_intermediate_size``, a shared expert scaled by a sigmoid
    gate. Every rank routes every token; the ranks' partial sums of the routed and shared
    experts meet in one all-reduce at the exit.

    LoRA target ``lora`` adapts every expert, unmerged: the block holds the routed experts'
    stacked factors, and the shared expert's projections hold their own; the router and the
    shared gate stay frozen.

    With a shared expert, inference runs the backend's whole-block op when it offers one
    for these routes and weights and no adapter is attached (the op reads the base weights
    alone), as ``settle_inference`` fixes; otherwise the layer composes the block around the
    routed experts. Training likewise runs the backend's whole training op over a frozen
    block, with or without a shared expert, and composes an adapted one. Every path routes
    by the backend's router, so every path routes the same hidden state alike.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        renormalize: bool,
        site: Site,
        shared_expert_intermediate_size: int | None = None,
        lora: str | None = None,
        deferred: bool = False,
    ):
        """``deferred``: the residual norm after the block sums inference's partial sums
        (``DeferredExit``)."""

        self.experts = Experts(
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            renormalize=renormalize,
            site=site,
        )
        self.lora_target = lora
        self.gate = LinearReplicated(hidden_size, num_experts, has_bias=False, site=site)
        self.shared_expert = self.shared_expert_gate = None
        # The backend's whole-block ops: inference's, the one it runs once settled, and
        # training's.
        self._block: SparseBlock | None = None
        self._training_block: TrainingBlock | None = None
        self._whole: SparseBlock | None = None
        self._settled = False
        if shared_expert_intermediate_size is not None:
            self.shared_expert = SharedExpert(
                hidden_size, shared_expert_intermediate_size, site=site, lora=lora
            )
            self.shared_expert_gate = LinearReplicated(hidden_size, 1, has_bias=False, site=site)
        # A dry build, which only reads shapes, has no backend.
        backend = site.services.moe(site.device)
        if backend is not None:
            dtype = self.experts.gate_up_proj.dtype
            if self.shared_expert is not None:
                self._block = backend.sparse_block(top_k, renormalize, dtype)
            self._training_block = backend.training_block(top_k, renormalize, dtype)
        local = site.group.size == 1
        # Training sums the routed path's and the gates' input gradients at the entry; the
        # shared expert's projections enter on their own.
        self._entry = LocalEntry() if local else ParallelEntry(site.group)
        self._exit: LocalExit | AllReduceExit
        if local:
            self._exit = LocalExit()
        else:
            self._exit = DeferredExit(site.group) if deferred else AllReduceExit(site.group)

    def attach_lora(self, config: LoRAConfig) -> ExpertLoRAFactors:
        """Adapt the routed experts, before inference settles."""

        return self.experts.adapt(config)

    def settle_inference(self) -> None:
        """Run the whole-block op from now on where the backend offers one, no adapter is
        attached and every weight it reads is stored as multiplied; compose elsewhere."""

        self._settled = True
        self._whole = None
        if self._block is not None and self.shared_expert is not None and self._frozen_dense():
            self._whole = self._block

    def _frozen_dense(self) -> bool:
        """Whether no adapter is attached (the routed and shared experts' share one target)
        and every weight a whole-block op reads is stored as multiplied."""

        if self.experts.lora is not None:
            return False
        methods = [self.experts.method]
        if (shared := self.shared_expert) is not None:
            methods += [
                shared.gate_up_proj.method,
                shared.down_proj.method,
                self.shared_expert_gate.method,
            ]
        return all(method.dense for method in methods)

    def _whole_training(self) -> TrainingBlock | None:
        """The backend's whole training op over a frozen block, or None to compose."""

        return self._training_block if self._frozen_dense() else None

    def warm_up_rows(
        self, max_rows: int, *, dtype: torch.dtype, device: torch.device, warmed: set
    ) -> None:
        """Run the whole-block op once at ``max_rows`` rows, before any graph is captured.

        cti-kernels' ``MoEBlock`` keeps its composition's intermediates per (device, current
        stream) and replaces them when a larger eager batch comes
        (``moe/inference.py`` ``Experts._intermediates``); a captured graph reuses them
        without holding them, so a later larger eager batch would leave it replaying over
        freed storage. The op offers no ``warm_up``, but an eager call allocates the
        intermediates at its size and keeps them: one call at the most rows any forward runs,
        on the stream the runner captures and serves on (current through its warm-up), sizes
        them for good. One op serves every layer.
        """

        super().warm_up_rows(max_rows, dtype=dtype, device=device, warmed=warmed)
        if self._whole is None:
            return
        key = (id(self._whole), max_rows)
        if key in warmed:
            return
        warmed.add(key)

        def allocated() -> int:
            return torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

        before = allocated()
        x = torch.zeros(max_rows, self.gate.weight.shape[1], dtype=dtype, device=device)
        with torch.inference_mode():
            self.forward(x)
        logger.info_rank0(
            "MoE whole-block op warmed at %d rows; its intermediates keep %.1f MiB",
            max_rows,
            (allocated() - before) / (1 << 20),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The block's output for every row it gets. A breakable graph runs it inside its
        segment over the whole row bucket: padding rows route and run the experts like real
        ones and their outputs are dropped, as vLLM 0.31 (MoE is no piecewise splitting op,
        config/compilation.py:764-782) and SGLang 0.5.22 (no MoE split op,
        compilation/compilation_config.py:10-16) capture MoE. With no token dispatch (the
        exit is one all-reduce), neither masks padding rows: they mask routes only before an
        all-to-all dispatch (SGLang layers/moe/topk.py:1736-1752; vLLM
        router/grouped_topk_router.py:351-360)."""

        if not self._settled:
            raise RuntimeError("prepare_inference settles the MoE block before its inference")
        experts, router = self.experts, self.gate.weight
        if self._whole is not None:
            shared = self.shared_expert
            return self._exit.reduce_inference(
                self._whole(
                    hidden_states,
                    router,
                    experts.gate_up_proj,
                    experts.down_proj,
                    shared.gate_up_proj.weight,
                    shared.down_proj.weight,
                    self.shared_expert_gate.weight,
                    expert_map=experts.expert_map,
                )
            )
        if self.shared_expert is None:
            return self._exit.reduce_inference(experts.forward(hidden_states, router))
        # The backend may write its output over its input, so the shared expert reads it first.
        shared = self.shared_expert.forward(hidden_states)
        gate = self.shared_expert_gate.forward(hidden_states)
        if experts.method.joins_shared():
            routed = experts.forward(hidden_states, router, shared, gate)
            return self._exit.reduce_inference(routed)
        # vLLM's order: r(r(r(sigmoid(gate)) * shared) + routed).
        shared *= torch.sigmoid(gate)
        return self._exit.reduce_inference(experts.forward(hidden_states, router).add_(shared))

    def training_bytes(self, tokens: int, *, rank: int) -> tuple[int, int]:
        """``(saved, backward)`` of ``tokens`` rows in training, at LoRA ``rank``: the whole
        training op's over a frozen block (``TrainingBlock.training_bytes``); else the
        FP32 router probabilities, the routed experts' as their method trains them
        (``ExpertsMethod.training_bytes``) and the shared expert's as a dense MLP's with its
        sigmoid gate."""

        shared = self.shared_expert
        if (whole := self._whole_training()) is not None:
            experts = self.experts
            width = 0 if shared is None else shared.gate_up_proj.local_output_size // 2
            return whole.training_bytes(
                tokens,
                hidden=experts.hidden_size,
                intermediate=len(experts.placement.intermediate),
                experts=len(experts.placement.experts),
                shared_intermediate=width,
                device=self.gate.weight.device,
            )
        routed_saved, backward = self.experts.method.training_bytes(tokens, rank=rank)
        saved = tokens * 4 * self.gate.local_output_size + routed_saved
        if (shared := self.shared_expert) is not None:
            # Its MLP's, and the FP32 sigmoid gate.
            mlp_saved, mlp_backward = shared.activation.mlp_training_bytes(
                shared.gate_up_proj, shared.down_proj, tokens, rank=rank
            )
            saved += mlp_saved + tokens * 4
            backward = max(backward, mlp_backward)
        return saved, backward

    def forward_training(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The block of frozen weights, differentiable in ``hidden_states`` through the
        routed experts, their router weights and the gated shared expert, and in their LoRA
        factors. Each rank backpropagates its partial sum."""

        x = hidden_states.flatten(0, -2)
        entered = self._entry.enter(x)
        if (whole := self._whole_training()) is not None:
            experts, shared = self.experts, self.shared_expert
            output = whole(
                entered.contiguous(),
                self.gate.weight,
                experts.gate_up_proj,
                experts.down_proj,
                None if shared is None else shared.gate_up_proj.weight,
                None if shared is None else shared.down_proj.weight,
                None if shared is None else self.shared_expert_gate.weight,
                expert_map=experts.expert_map,
            )
            return self._exit.reduce_training(output).view_as(hidden_states)
        output = self.experts.forward_training(entered, self.gate.weight)
        if self.shared_expert is not None:
            gate = torch.sigmoid(self.shared_expert_gate.forward_training(entered))
            output = output + self.shared_expert.forward_training(x) * gate
        return self._exit.reduce_training(output).view_as(hidden_states)
