# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/utils/hf.py`); see THIRD_PARTY_NOTICES.

"""Resolved phase: one rank's choices from the request, its checkpoint's model and its device."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING, Literal

import torch

from cti.attention import AttentionRequest, resolve_attention_config
from cti.config import ExpertLoRALayout
from cti.kvcache import (
    GroupRole,
    ResolvedSlot,
    SlotKey,
    SnapshotPolicy,
    Spec,
    choose_page_size,
)
from cti.layers import ExpertParallel, ExpertTensorParallel
from cti.models import (
    CheckpointFacts,
    ModelCapability,
    ModelConfig,
    get_model_class,
    peft_lora,
    read_checkpoint,
)
from cti.multimodal import ImageLimits
from cti.plan import Plan, PromptBudget
from cti.utils import DeviceCapability, init_logger

if TYPE_CHECKING:
    from cti.attention import AttentionRequirement
    from cti.config import EngineOptions, EngineTuning, LoRAConfig, ModelHub
    from cti.kvcache import RowLayout
    from cti.models import BaseLLMModel
    from cti.plan import SpeculativeMethod


logger = init_logger(__name__)

# The most running requests the memory budget may choose when the option is unset.
_AUTO_MAX_RUNNING_REQUESTS = 256
# An online server's prompt tokens per step by the device's total memory, below each GiB
# bound: SGLang 0.5.22's chunked_prefill_size tiers (srt/arg_groups/memory_hook.py:88-170).
_ONLINE_PROMPT_TOKENS = ((20, 2048), (60, 4096), (160, 8192), (float("inf"), 16384))


def _state_slots(
    requirements: Mapping[GroupRole, Sequence[AttentionRequirement]],
    layouts: Mapping[SlotKey, RowLayout],
) -> dict[SlotKey, ResolvedSlot]:
    """Each role's paged slots in the row layouts attention chose. A slot's requirements
    share one KV group until windowed groups: it reads up to their widest window."""

    slots: dict[SlotKey, ResolvedSlot] = {}
    for role, role_requirements in requirements.items():
        for requirement in role_requirements:
            key = (role, requirement.slot)
            window = requirement.window
            if (known := slots.get(key)) is not None:
                if (known.family, known.spec.tokens_per_entry) != (
                    requirement.family,
                    requirement.tokens_per_entry,
                ):
                    raise ValueError(f"{role} slot {requirement.slot!r} has two families")
                windows = (known.spec.window, window)
                window = None if None in windows else max(windows)
            spec = Spec(layouts[key], requirement.tokens_per_entry, window)
            slots[key] = ResolvedSlot(requirement.family, spec)
    return slots


def _doubling_budgets(smallest: int, largest: int) -> tuple[int, ...]:
    """``smallest`` doubling while below ``largest``, then ``largest``; none when
    ``smallest`` exceeds it."""

    budgets = []
    budget = smallest
    while budget < largest:
        budgets.append(budget)
        budget *= 2
    return (*budgets, largest) if smallest <= largest else ()


@dataclass(frozen=True, slots=True)
class CheckedOptions:
    """The options and tuning ``ResolvedModel.check`` accepted for the ``speculative`` method,
    with the values it derived once for ``plan``: the LoRA options with their PEFT adapter's,
    the execution dtype, the sequence and running-request bounds and the snapshot policy."""

    options: EngineOptions
    tuning: EngineTuning
    speculative: SpeculativeMethod | None
    lora: LoRAConfig | None
    dtype: torch.dtype
    max_seq_len: int
    max_running_requests: int
    snapshots: SnapshotPolicy


@dataclass(frozen=True, kw_only=True)
class ResolvedModel:
    """The checkpoint's facts (``models.read_checkpoint``) and the registered class serving
    it. Each rank still loads its own (until EngineCore resolves once). The speculative
    method, which the composition root loads against it, comes to ``check``, whose result
    ``plan`` takes."""

    facts: CheckpointFacts
    model_config: ModelConfig
    model_class: type[BaseLLMModel]

    @classmethod
    def load(
        cls, model_path: str, *, hub: ModelHub = "huggingface", text_only: bool = False
    ) -> ResolvedModel:
        """Read the checkpoint, a local directory or a repository id on ``hub``, with
        ``text_only`` as its text-only export, and look its class up; an unsupported field or
        malformed EOS fails before any weights download."""

        facts = read_checkpoint(model_path, hub=hub, text_only=text_only)
        model_class = get_model_class(facts.architecture)
        model_class.validate_hf_config(facts.hf_config.get_text_config())
        model_config = model_class.engine_config(facts.hf_config)
        return cls(facts=facts, model_config=model_config, model_class=model_class)

    def check(
        self,
        options: EngineOptions,
        tuning: EngineTuning,
        *,
        speculative: SpeculativeMethod | None,
    ) -> CheckedOptions:
        """Reject what the options and tuning ask of each other, of this model and of the
        ``speculative`` method: the one place for rejections across sections. The composition
        root runs it before any device state exists, then ``plan`` on what it returns."""

        model_config = self.model_config
        training, lora = options.training, self._lora(options)
        num_ranks = options.parallel.num_ranks
        scheduling = options.scheduling
        dtype = self._dtype(options)
        max_position = model_config.max_position
        max_seq_len = self._max_seq_len(options)
        max_running_requests = self._max_running_requests(options)
        # Longer sequences would read positions past the end of the RoPE cache.
        if max_seq_len > max_position:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds the model's "
                f"max_position_embeddings={max_position}"
            )
        # Checked even without prefix reuse, which leaves the policy unused.
        snapshots = SnapshotPolicy(tuning.snapshot_rows, tuning.snapshot_memory_ratio)
        if any(size > max_running_requests for size in tuning.cuda_graph_bs or ()):
            raise ValueError("cuda_graph_bs must not exceed max_running_requests")
        if any(size > scheduling.max_batch_tokens for size in tuning.graph_row_buckets or ()):
            raise ValueError("graph_row_buckets must not exceed max_batch_tokens")
        if (
            training is not None
            and lora is not None
            and "experts" in lora.targets
            and lora.expert_layout is not ExpertLoRALayout.SHARED_A
        ):
            raise ValueError(
                "training adapts the routed experts in the shared-A layout (one A for each "
                "expert's gate and up); this adapter's separate gate and up A serve only"
            )
        requested = ModelCapability(0)
        if lora is not None:
            requested |= ModelCapability.LORA
        if training is not None:
            requested |= ModelCapability.TRAINING
        if options.cache.prefix == "radix":
            requested |= ModelCapability.PREFIX_REUSE
        if num_ranks > 1:
            requested |= ModelCapability.TENSOR_PARALLEL
        if speculative is not None:
            requested |= ModelCapability.SPECULATIVE_TARGET
        self.model_class.require_capabilities(
            requested,
            model_architecture=model_config.architectures[0],
            lora_targets=() if lora is None else lora.targets,
        )
        if (method := speculative) is not None:
            if missing := requested & ~method.capabilities:
                raise NotImplementedError(
                    f"speculative decoding with {missing.name} is not implemented yet"
                )
            method.check(max_seq_len=max_seq_len)
        return CheckedOptions(
            options,
            tuning,
            speculative,
            lora,
            dtype,
            max_seq_len,
            max_running_requests,
            snapshots,
        )

    def plan(
        self,
        checked: CheckedOptions,
        *,
        device: torch.device,
        capability: DeviceCapability | None,
        device_memory: int,
        online: bool,
    ) -> Plan:
        """Make every choice for this checkpoint, the ``checked`` request's speculative method
        and ``device``, the rank's, with ``device_memory`` its total bytes; the only rejections
        left are of what this device's choices cannot run. ``online`` is for an online server
        rather than offline calls."""

        options, tuning, method = checked.options, checked.tuning, checked.speculative
        num_ranks = options.parallel.num_ranks
        max_running_requests = checked.max_running_requests
        rounds = (
            None
            if method is None
            else method.round_shape(
                max_running_req=max_running_requests,
                max_forward_len=options.scheduling.max_batch_tokens,
            )
        )
        hf_config = self.facts.hf_config
        requirements = {GroupRole.TARGET: self.model_class.attention_requirements(hf_config)}
        if method is not None:
            requirements[GroupRole.DRAFT] = method.attention_requirements()
        dtype = checked.dtype
        attention = resolve_attention_config(
            requirements,
            AttentionRequest(
                target=GroupRole.TARGET,
                backend=tuning.attention_backend,
                page_size=tuning.page_size,
                capability=capability,
                tensor_parallel_size=num_ranks,
                dtype=dtype,
                rounds=rounds,
                # FlashInfer's tensor-core decode forced on or off; None chooses by grouping.
                tensor_cores=tuning.flashinfer_tensor_cores,
            ),
        )
        page_size = choose_page_size(
            tuning.page_size,
            attention.pages,
            [r.tokens_per_entry for role in requirements.values() for r in role],
        )
        supported = self.model_class.CAPABILITIES
        if method is not None:
            supported &= method.capabilities
        prompt_budget = self._prompt_budget(options, device_memory=device_memory, online=online)
        prefix_cache = self._prefix_cache(
            options.cache.prefix,
            supported,
            device=device,
            num_ranks=num_ranks,
            page_size=page_size,
            prompt_budget=prompt_budget,
        )
        max_seq_len = checked.max_seq_len
        max_batch_tokens = options.scheduling.max_batch_tokens
        preprocessing = self.facts.image_preprocessing
        max_image_tokens = (
            0 if preprocessing is None else min(preprocessing.max_image_tokens, max_seq_len)
        )
        return Plan(
            max_batch_tokens=max_batch_tokens,
            prompt_budget=prompt_budget,
            inference_ratio=options.memory.inference_ratio,
            peak_ratio=options.memory.peak_ratio,
            recurrent_to_kv_ratio=options.memory.recurrent_to_kv_ratio,
            lora=checked.lora,
            training=options.training,
            model_path=self.facts.source.folder,
            weights=options.model.weights,
            initialization_seed=options.model.initialization_seed,
            dtype=dtype,
            max_seq_len=max_seq_len,
            image_limits=ImageLimits(max_seq_len, max_image_tokens),
            encoder_rows=max(max_batch_tokens, max_image_tokens),
            max_running_requests=max_running_requests,
            auto_running_requests=options.scheduling.max_running_requests is None,
            num_ranks=num_ranks,
            local_device=0 if device.index is None else device.index,
            experts=(
                ExpertParallel()
                if options.parallel.enable_expert_parallel
                else ExpertTensorParallel()
            ),
            prefix_cache=prefix_cache,
            snapshots=checked.snapshots if prefix_cache == "radix" else None,
            attention=attention,
            page_size=page_size,
            state_slots=_state_slots(requirements, attention.layouts),
            speculative=method,
            rounds=rounds,
            seeds_every_row=method is not None,
            graph_row_buckets=(
                None
                if tuning.graph_row_buckets is None
                else tuple(sorted(set(tuning.graph_row_buckets)))
            ),
            graph_batch_sizes=tuning.cuda_graph_bs,
            graph_max_batch=tuning.cuda_graph_max_bs,
            vision_graph_budgets=(
                _doubling_budgets(preprocessing.min_image_tokens, max_batch_tokens)
                if tuning.vision_graphs and preprocessing is not None and max_image_tokens > 0
                else ()
            ),
            distributed_timeout=tuning.distributed_timeout,
            use_pynccl=tuning.use_pynccl,
            capability=capability,
        )

    def _prefix_cache(
        self,
        prefix: str,
        supported: ModelCapability,
        *,
        device: torch.device,
        num_ranks: int,
        page_size: int,
        prompt_budget: PromptBudget,
    ) -> Literal["naive", "radix"]:
        """The radix cache where asked for, or with ``auto`` where the model reuses
        prefixes, else the naive one. A recurrent prefix restores only states its prefill
        exported: without them an explicit radix cache is refused and an automatic one is
        naive. With them, a prompt step short of its prompt's end ends on the snapshot grid,
        the lcm of ``page_size`` and ``RECURRENT_EXPORT_GRANULARITY``, which
        ``prompt_budget`` must reach."""

        if not (
            prefix == "radix" or (prefix == "auto" and ModelCapability.PREFIX_REUSE in supported)
        ):
            return "naive"
        if self.model_class.exports_prefill_states(
            self.facts.hf_config, device, num_ranks=num_ranks
        ):
            granularity = self.model_class.RECURRENT_EXPORT_GRANULARITY
            if granularity is not None and prompt_budget.tokens < (
                grid := lcm(page_size, granularity)
            ):
                raise ValueError(
                    f"max_prompt_tokens={prompt_budget.tokens} ({prompt_budget.reason}) must "
                    f"reach the {grid}-token grid recurrent prefix snapshots lie on"
                )
            return "radix"
        reason = (
            "the recurrent layers' prefill runs cti-kernels' fla-core provider here, which "
            "captures no states; its CuTe and Triton providers do (Triton from compute "
            "capability 10.0, CuTe where nvidia-cutlass-dsl 4.8.0 compiles and it is not "
            "switched off)"
        )
        if prefix == "radix":
            raise ValueError(
                f'cache prefix "radix" with recurrent layers needs a prefill that exports '
                f'states, but {reason}; use "naive"'
            )
        logger.info("Using the naive cache: %s", reason)
        return "naive"

    def _lora(self, options: EngineOptions) -> LoRAConfig | None:
        """The LoRA options, with their PEFT adapter's targets, rank and alpha."""

        lora = options.lora
        if lora is None or lora.adapter is None:
            return lora
        return peft_lora(lora, self.model_class.LORA_TARGETS)

    def _dtype(self, options: EngineOptions) -> torch.dtype:
        """The model's execution dtype, BF16; ``auto`` checks that the checkpoint stores
        BF16, FP16 or FP32 weights, which load converted."""

        if options.model.dtype == "auto":
            dtype = self.facts.hf_config.dtype
            if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(
                    f"the checkpoint's dtype {dtype} is not float16, bfloat16 or float32"
                )
        return torch.bfloat16

    def _max_seq_len(self, options: EngineOptions) -> int:
        """The option, else the checkpoint's RoPE positions."""

        max_seq_len = options.model.max_seq_len
        return self.model_config.max_position if max_seq_len is None else max_seq_len

    @staticmethod
    def _prompt_budget(options: EngineOptions, *, device_memory: int, online: bool) -> PromptBudget:
        """The option, else on an online server the default of the device's memory tier,
        which keeps prompt steps short for the TTFT median and the decode tail beside them;
        offline, ``max_batch_tokens`` for throughput."""

        scheduling = options.scheduling
        if (tokens := scheduling.max_prompt_tokens) is not None:
            return PromptBudget(tokens, "set by scheduling.max_prompt_tokens")
        if not online:
            return PromptBudget(scheduling.max_batch_tokens, "offline: max_batch_tokens")
        tier = next(t for gib, t in _ONLINE_PROMPT_TOKENS if device_memory < gib * (1 << 30))
        tokens = min(tier, scheduling.max_batch_tokens)
        return PromptBudget(tokens, "online server default for the device's memory")

    @staticmethod
    def _max_running_requests(options: EngineOptions) -> int:
        """The option, else the most the memory budget may choose: 256, as vLLM's default
        ``max_num_seqs`` below 70 GiB (vllm 0.31.0 ``engine/arg_utils.py:2877-2886``), and
        one decode token per request within a step (there, ``:3113-3115``)."""

        scheduling = options.scheduling
        if scheduling.max_running_requests is not None:
            return scheduling.max_running_requests
        return min(_AUTO_MAX_RUNNING_REQUESTS, scheduling.max_batch_tokens)
