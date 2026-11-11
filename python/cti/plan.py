# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/engine/config.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cti.attention import AttentionRequirement, ResolvedFamily
    from cti.config import Checkpoint, LoRAConfig, RandomWeights, TrainingConfig
    from cti.forward import RoundShape
    from cti.kvcache import ResolvedSlot, SlotKey, SnapshotPolicy
    from cti.layers import ExpertPlacement
    from cti.models import CheckpointSource, ModelCapability
    from cti.multimodal import ImageLimits
    from cti.utils import DeviceCapability


class SpeculativeMethod(Protocol):
    """A speculative method, loaded and checked against its target before any device state:
    the engine features its rounds run beside (a refusal for the rest), what its model's
    attention needs, which resolution keys by the draft's role beside the target's, its
    rounds' shape, and its own model, which the model host builds as
    ``model_class(config, site=...)`` and loads from ``source``."""

    source: CheckpointSource
    model_class: type
    config: object
    capabilities: ModelCapability

    def attention_requirements(self) -> tuple[AttentionRequirement, ...]: ...

    def check(self, *, max_seq_len: int) -> None: ...

    def round_shape(self, *, max_running_req: int, max_forward_len: int) -> RoundShape: ...


@dataclass(frozen=True)
class PromptBudget:
    """Prompt tokens one step may carry after its rounds or decodes, and why."""

    tokens: int
    reason: str


@dataclass(frozen=True, kw_only=True)
class Plan:
    """What one rank runs: every value resolved for its checkpoint and device. Only
    ``ResolvedModel.plan`` builds one, from the options and the tuning; the engine derives
    no value elsewhere."""

    # The most tokens one forward carries (``SchedulingOptions.max_batch_tokens``).
    max_batch_tokens: int
    # The option, else the default for the entry point.
    prompt_budget: PromptBudget
    # Shares of the free device memory (``MemoryOptions``): inference's budget, None for the
    # device's default resolved at boot, and the peak-planning budget, None for the former.
    inference_ratio: float | None
    peak_ratio: float | None
    # The recurrent rows' share of the inference state's budget
    # (``MemoryOptions.recurrent_to_kv_ratio``).
    recurrent_to_kv_ratio: float
    lora: LoRAConfig | None
    training: TrainingConfig | None
    # The checkpoint, a local directory made absolute.
    model_path: str
    # Where the weights come from (``ModelOptions.weights``); a rank opens them with
    # ``models.open_weights``.
    weights: Checkpoint | RandomWeights
    initialization_seed: int
    dtype: torch.dtype
    # The option, else the checkpoint's RoPE positions.
    max_seq_len: int
    # One image per sequence token, each up to the largest image the checkpoint's
    # preprocessing makes, within max_seq_len (0 tokens without image preprocessing).
    image_limits: ImageLimits
    # The most image-encoder rows one step encodes and the runner keeps: a step's tokens
    # and the largest image, as vLLM's ``compute_mm_encoder_budget`` sizes them.
    encoder_rows: int
    # The option, else the most the memory budget may choose (``auto_running_requests``);
    # the inference state's ``num_slots`` is the count requests run with.
    max_running_requests: int
    auto_running_requests: bool
    num_ranks: int
    # The CUDA device on its host the rank runs on (ParallelOptions.local_device).
    local_device: int
    # How MoE layers split their routed experts over the ranks.
    experts: ExpertPlacement
    # The prefix cache CacheManager keeps: radix reuses prefixes.
    prefix_cache: Literal["naive", "radix"]
    # The recurrent snapshot rows prefix reuse keeps; None without prefix reuse.
    snapshots: SnapshotPolicy | None
    # The resolved attention family, and the page size kvcache chose for it.
    attention: ResolvedFamily
    page_size: int
    # Each paged slot's family and spec, by role and requirement name.
    state_slots: Mapping[SlotKey, ResolvedSlot]
    # The speculative method the composition root loaded, and its rounds; None without one.
    speculative: SpeculativeMethod | None
    rounds: RoundShape | None
    # Rounds seed every request where it enters; elsewhere a batch's rows can be seeded or
    # not.
    seeds_every_row: bool
    # Sorted breakable-graph row buckets, or None for RowBuckets' default ladder.
    graph_row_buckets: tuple[int, ...] | None
    # Decode graph batch sizes, and the largest (None: what the bounds and memory allow).
    graph_batch_sizes: tuple[int, ...] | None
    graph_max_batch: int | None
    # Token budgets vision graphs are captured at: the smallest image's tokens doubling below
    # max_batch_tokens, then max_batch_tokens; () for none (EngineTuning.vision_graphs off, or
    # no image).
    vision_graph_budgets: tuple[int, ...]
    # The process group's collective timeout and whether it uses PyNCCL.
    distributed_timeout: float
    use_pynccl: bool
    capability: DeviceCapability | None
