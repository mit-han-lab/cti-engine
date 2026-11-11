"""Device memory one training configuration adds to an inference runtime, fixed at startup.

Each component's bytes follow from the selected parameters, the optimizer and the
activation peak of the largest configured microbatch. The live selected parameters belong
to the model and are measured with its weights.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Protocol

import torch

from cti.config import (
    ActivationCheckpointMode,
    BatchLayout,
    StaticBatching,
    TrainingConfig,
    TrainingStateResidency,
)
from cti.models.activation_checkpointing import ROW_TILE
from cti.models.training_contract import TrainingActivations, TrainingModel
from cti.utils import init_logger, mem_GB

from ._optimizer import foreach_group_sizes
from ._parameters import (
    SupportsParameterUniverse,
    TrainableParameterMetadata,
    describe_for_model,
    first_trainable_layer,
)
from ._staging import staging_device_bytes

logger = init_logger(__name__)

_FP32_BYTES = torch.float32.itemsize
# Free segment space a fresh pool may hold beside its live blocks, as a fraction of their
# bytes: training scopes measured up to a fifth (RTX PRO 6000, RTX 5090).
_FRAGMENTATION_FRACTION = 0.25
# AdamW keeps exp_avg and exp_avg_sq, FP32, per updated element.
_ADAMW_MOMENTS = 2
# Offloaded checkpoint inputs on the device: in forward the embedding, the previous input
# awaiting its copy, the current one and its output; in backward the segment's own input,
# the preceding one, fetched while it runs, and the gradient of its output from the segment
# above, live while it recomputes its forward.
_OFFLOADED_FORWARD_INPUTS = 4
_OFFLOADED_BACKWARD_INPUTS = 3
# A frozen layer's input and output beside what its forward allocates.
_FROZEN_LAYER_STREAMS = 2


@dataclass(frozen=True, slots=True)
class TrainingMemoryPlan:
    """Physical bytes of each training component while training holds the device, and the
    ``residency`` that decides which of them stay beside inference (``resident_bytes``)."""

    # FP32 copies of the BF16 selected parameters the optimizer updates.
    master_weights: int
    # The optimizer's per-element state.
    optimizer_state: int
    # FP32 gradients accumulated across a logical batch's microbatches.
    gradients: int
    # The device buffers a fixed SFT shape profile stages its inputs in.
    input_staging: int
    # The optimizer step's temporaries for its largest group of parameters, in the window.
    optimizer_step: int
    # One microbatch's peak: activations, backward temporaries and live BF16 gradients.
    activations: int
    # The segments the window's fresh pool maps beyond those activations.
    activation_segments: int
    residency: TrainingStateResidency = TrainingStateResidency.RESIDENT

    @property
    def resident_bytes(self) -> int:
        """Bytes training keeps on the device while inference runs: what its ``residency``
        does not offload."""

        residency = self.residency
        kept = self.input_staging
        if not residency.offloads_optimizer_state:
            kept += self.optimizer_state
        if not residency.offloads_masters_and_gradients:
            kept += self.master_weights + self.gradients
        return kept

    def components(self) -> dict[str, int]:
        """Each component's bytes, by name."""

        return {name: nbytes for name, nbytes in asdict(self).items() if name != "residency"}

    @property
    def training_bytes(self) -> int:
        """Bytes training holds on the device at its peak, with inference memory released."""

        return sum(self.components().values())

    @property
    def scope_bytes(self) -> int:
        """The most one scope of a training window maps in a fresh pool: a microbatch's
        activations with their segments, or the optimizer step's temporaries."""

        return max(self.activations + self.activation_segments, self.optimizer_step)

    def largest(self, count: int) -> tuple[tuple[str, int], ...]:
        """The ``count`` largest components, by name, largest first."""

        terms = sorted(self.components().items(), key=lambda term: term[1], reverse=True)
        return tuple(terms[:count])


@dataclass(frozen=True, slots=True)
class StateBytes:
    """One training state component on the device: its tensors' storage bytes, the bytes of
    the allocator segments that hold nothing else, and the pools of the segments holding
    them."""

    payload: int
    segments: int
    pools: frozenset[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class TrainingMemoryUsage:
    """What the planned state components (``TrainingMemoryPlan``) hold on the device now."""

    master_weights: StateBytes
    optimizer_state: StateBytes
    gradients: StateBytes
    input_staging: StateBytes

    @classmethod
    def measure(cls, components: dict[str, Sequence[torch.Tensor]]) -> TrainingMemoryUsage:
        """Each component's ``StateBytes``, from the caching allocator's segments."""

        segments = []
        if any(tensor.is_cuda for tensors in components.values() for tensor in tensors):
            segments = torch.cuda.memory_snapshot(include_traces=False)

        def component(tensors: Sequence[torch.Tensor]) -> StateBytes:
            storages = {tensor.untyped_storage().data_ptr(): tensor for tensor in tensors}
            alone, pools = 0, set()
            for segment in segments:
                start, size = segment["address"], segment["total_size"]
                if any(start <= pointer < start + size for pointer in storages):
                    pools.add(tuple(segment["segment_pool_id"]))
                live = [
                    block for block in segment["blocks"] if block["state"] == "active_allocated"
                ]
                if live and all(block["address"] in storages for block in live):
                    alone += size
            payload = sum(tensor.untyped_storage().nbytes() for tensor in storages.values())
            return StateBytes(payload, alone, frozenset(pools))

        return cls(**{name: component(tensors) for name, tensors in components.items()})

    def against(self, plan: TrainingMemoryPlan) -> str:
        """Each component's held bytes beside its planned bytes, for the log."""

        return ", ".join(
            f"{name} {mem_GB(getattr(self, name).payload)} of {mem_GB(getattr(plan, name))}"
            for name in ("master_weights", "optimizer_state", "gradients", "input_staging")
        )


class AllocatorModel(Protocol):
    """The caching allocator's placement, which the worker injects: the segment bytes a pool
    of kept blocks maps, and the sizes of its small and large segments."""

    @property
    def SMALL_SEGMENT_BYTES(self) -> int: ...

    @property
    def LARGE_SEGMENT_BYTES(self) -> int: ...

    def pooled(self, payloads: Iterable[int]) -> int: ...


def plan_training_memory(
    parameters: Sequence[TrainableParameterMetadata],
    activations: TrainingActivations,
    checkpointing: ActivationCheckpointMode,
    *,
    first: int,
    boundary_bytes: int,
    staging_bytes: Sequence[int] = (),
    allocator: AllocatorModel,
    residency: TrainingStateResidency = TrainingStateResidency.RESIDENT,
) -> TrainingMemoryPlan:
    """Plan AdamW training of ``parameters`` with a microbatch's ``activations``.

    Every parameter trains as an FP32 master, which the training kernels cast at each use
    and keep, not a cast, for backward. Each state component is placed by
    ``allocator`` in creation order, so its bytes include segment rounding. Decoder layers before ``first`` run without autograd; each
    checkpointed layer input takes ``boundary_bytes``; ``staging_bytes`` are the input
    staging buffers; ``residency`` says which components stay beside inference.
    """

    fp32_bytes = tuple(math.prod(parameter.shape) * _FP32_BYTES for parameter in parameters)
    master_bytes = fp32_bytes
    # The optimizer creates both moments of a parameter before the next one's at startup;
    # a step takes one FP32 temporary per parameter of a group in the window's pool, after
    # the commit's normalized gradient norm, an FP32 scalar that lives through the step.
    moment_bytes = tuple(nbytes for nbytes in fp32_bytes for _ in range(_ADAMW_MOMENTS))
    step_bytes = max(
        (allocator.pooled((_FP32_BYTES, *group)) for group in foreach_group_sizes(fp32_bytes)),
        default=0,
    )
    # Backward holds every master's new FP32 gradient until accumulation moves it.
    gradient_bytes = sum(fp32_bytes)
    peak = partial(
        _activation_peak_bytes,
        checkpointing=checkpointing,
        first=first,
        boundary_bytes=boundary_bytes,
        gradient_bytes=gradient_bytes,
    )
    activation_bytes = peak(activations)
    segment_bytes = pool_slack_bytes(activation_bytes - gradient_bytes, allocator)
    if max(activations.head_blocks, default=0) > activations.largest_block:
        # No block spans segments: logits chunks wider than every decoder block cannot reuse
        # the decoder's freed segments, so theirs add to the decoder phase's, which keeps a
        # partly used small and large segment beside its peak.
        decoder = peak(replace(activations, head_working=0))
        separate = (
            decoder
            + allocator.SMALL_SEGMENT_BYTES
            + allocator.LARGE_SEGMENT_BYTES
            + allocator.pooled(activations.head_blocks)
        )
        segment_bytes = max(segment_bytes, separate - activation_bytes)
    # The FP32 gradients are small blocks, which a window maps in small-pool segments of
    # their own beside either peak's: none of the slack above holds them. Their payload,
    # counted in the activations too, is a deliberate over-count.
    segment_bytes += allocator.pooled(fp32_bytes)
    return TrainingMemoryPlan(
        master_weights=allocator.pooled(master_bytes),
        optimizer_state=allocator.pooled(moment_bytes),
        gradients=allocator.pooled(fp32_bytes),
        input_staging=allocator.pooled(staging_bytes),
        optimizer_step=step_bytes,
        activations=activation_bytes,
        activation_segments=segment_bytes,
        residency=residency,
    )


def prepare_model_training(
    config: TrainingConfig,
    model: TrainingModel,
    *,
    trainables: SupportsParameterUniverse,
    rank: int,
    stream: torch.cuda.Stream,
    allocator: AllocatorModel,
    residency: TrainingStateResidency | None = None,
) -> TrainingMemoryPlan:
    """Compile the built ``model``'s training ops at the largest microbatch ``config``'s
    batching allows, bind ``stream``'s cuBLAS workspace, then plan training's memory
    (``plan_model_training_memory``) as ``allocator`` places it. All run once inference's warm-up is done and before
    the model's inference state is sized: an op's first launch may grow the device's local
    memory for the life of the process, which the KV budget must then see, and the
    inference warm-up drops every stream's cuBLAS workspace, while the training stream's
    lives as long as the process."""

    layers = model.iter_recurrent_layers()
    for layer in layers:
        layer.warm_up_training(config.batching.policy.max_tokens_per_microbatch)
    if layers:
        # Nothing the warm-up allocated stays cached under the budget.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    with torch.cuda.stream(stream):
        torch.cuda.current_blas_handle()
    plan = plan_model_training_memory(
        config, model, trainables=trainables, rank=rank, allocator=allocator, residency=residency
    )
    logger.info_rank0(
        "Training memory: %s resident beside inference, %s at its peak (%s)",
        mem_GB(plan.resident_bytes),
        mem_GB(plan.training_bytes),
        ", ".join(
            f"{component} {mem_GB(nbytes)}" for component, nbytes in plan.components().items()
        ),
    )
    return plan


def plan_model_training_memory(
    config: TrainingConfig,
    model: object,
    *,
    trainables: SupportsParameterUniverse,
    rank: int,
    allocator: AllocatorModel,
    residency: TrainingStateResidency | None = None,
) -> TrainingMemoryPlan:
    """Plan ``config``'s training of the built ``model``, whose ``trainables`` it selects
    from and whose LoRA targets adapt at ``rank``, at the largest microbatch its batching
    allows, as ``allocator`` places it, before the model's inference state exists;
    ``residency`` as the rank resolved it for its device, else as configured."""

    if not isinstance(model, TrainingModel):
        raise TypeError(f"{type(model).__name__} does not account its training activations")
    parameters = describe_for_model(config.trainable_plan, trainables)
    policy = config.batching.policy
    tokens = policy.max_tokens_per_microbatch
    sequences = (
        policy.microbatch_size
        if isinstance(policy, StaticBatching)
        else policy.max_sequences_per_microbatch
    )
    width, dtype = model.residual_stream_layout()
    profile = config.sft_profile
    activations = model.training_activations(
        tokens,
        sequences,
        rank=rank,
        # Offloaded checkpoints train row-wise segments in tiles.
        row_tile=(
            ROW_TILE
            if config.activation_checkpointing is ActivationCheckpointMode.FULL_OFFLOAD
            else None
        ),
    )
    return plan_training_memory(
        parameters,
        activations,
        config.activation_checkpointing,
        first=first_trainable_layer(trainables, (parameter.name for parameter in parameters)),
        boundary_bytes=tokens * width * dtype.itemsize,
        staging_bytes=(
            ()
            if profile is None
            else staging_device_bytes(profile, packed=config.batching.layout is BatchLayout.PACKED)
        ),
        allocator=allocator,
        residency=residency or config.training_state_residency or TrainingStateResidency.RESIDENT,
    )


def pool_slack_bytes(live_bytes: int, allocator: AllocatorModel) -> int:
    """Segment bytes a fresh pool may map beyond ``live_bytes`` of mixed-size blocks: a
    small and a large segment of which little is used, and fragmentation."""

    if live_bytes == 0:
        return 0
    return (
        allocator.SMALL_SEGMENT_BYTES
        + allocator.LARGE_SEGMENT_BYTES
        + math.ceil(_FRAGMENTATION_FRACTION * live_bytes)
    )


def _activation_peak_bytes(
    activations: TrainingActivations,
    checkpointing: ActivationCheckpointMode,
    *,
    first: int,
    boundary_bytes: int,
    gradient_bytes: int,
) -> int:
    """The larger of the frozen layers' forward peak and the trained layers' peak.

    Layers before ``first`` run without autograd; a layer's forward needs no more than
    what it saves and what its backward allocates.
    """

    frozen = max(
        (
            _FROZEN_LAYER_STREAMS * boundary_bytes + sum(_layer_bytes(segments))
            for segments in activations.layers[:first]
        ),
        default=0,
    )
    return max(
        frozen,
        _trained_peak_bytes(
            activations,
            checkpointing,
            first=first,
            boundary_bytes=boundary_bytes,
            gradient_bytes=gradient_bytes,
        ),
    )


def _trained_peak_bytes(
    activations: TrainingActivations,
    checkpointing: ActivationCheckpointMode,
    *,
    first: int,
    boundary_bytes: int,
    gradient_bytes: int,
) -> int:
    """The larger of the forward peak and the backward peak with every live gradient.

    Plain checkpointing recomputes a whole layer in backward; offloaded checkpointing one
    segment of it, with only that segment's input and the next one's prefetch on the device.
    """

    trained = activations.layers[first:]
    head = activations.head_saved + activations.head_working
    if not trained:
        return head
    whole = [_layer_bytes(segments) for segments in trained]
    if checkpointing is ActivationCheckpointMode.NONE:
        kept = sum(saved for saved, _ in whole) + activations.head_saved
        busiest = max(backward for _, backward in whole)
        return kept + max(activations.head_working, busiest + gradient_bytes)
    if checkpointing is ActivationCheckpointMode.FULL:
        layer_peak = max(saved + backward for saved, backward in whole)
        return len(trained) * boundary_bytes + max(head, layer_peak + gradient_bytes)
    segment_peak = max(saved + backward for segments in trained for saved, backward in segments)
    return max(
        _OFFLOADED_FORWARD_INPUTS * boundary_bytes + head,
        _OFFLOADED_BACKWARD_INPUTS * boundary_bytes + segment_peak + gradient_bytes,
    )


def _layer_bytes(segments: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """A whole layer's ``(saved, backward)`` from its segments in forward order: it keeps
    every segment's saves, and a segment's backward runs once the later ones released
    theirs."""

    later = [sum(saved for saved, _ in segments[index + 1 :]) for index in range(len(segments))]
    backward = max(
        (backward - after for (_, backward), after in zip(segments, later, strict=True)),
        default=0,
    )
    return sum(saved for saved, _ in segments), max(backward, 0)


__all__ = [
    "TrainingMemoryPlan",
    "plan_model_training_memory",
    "prepare_model_training",
    "plan_training_memory",
    "pool_slack_bytes",
]
