from __future__ import annotations

import ctypes
import math
import os
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol, TypeVar

import torch

from cti.config import (
    ActivationCheckpointMode,
    BatchLayout,
    TrainingConfig,
    TrainingStateResidency,
)
from cti.models.activation_checkpointing import ReentrantActivationCheckpoint
from cti.models.training_contract import TrainingModel

from ._batch import TrainingBatch
from ._batching import plan_microbatches
from ._boundary_offload import HostBoundaryOffload
from ._checkpoint import OptimizerState, TrainingCheckpoint
from ._errors import InvalidGradientError, TrainingInputError
from ._host_memory import PinnedHostMemory
from ._memory_plan import TrainingMemoryPlan, TrainingMemoryUsage, prepare_model_training
from ._objective import SFT, ForwardBackwardCompletion, Metrics
from ._optimizer import AdamWOptimizer
from ._parameter_layout import LocalLayout, tensor_parallel_layout
from ._parameters import (
    ModelParameterUniverse,
    SupportsParameterUniverse,
    first_trainable_layer,
    resolve_for_model,
)
from ._staging import SFTStaging
from ._state_residency import training_state_residency

if TYPE_CHECKING:
    from ._parameter_layout import PlacedParameters, Ranks

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class TrainingCommitResult:
    completions: tuple[ForwardBackwardCompletion, ...]
    grad_norm: float


@dataclass(slots=True)
class _AcceptedBatch:
    """One accepted batch's progress: the next microbatch and the completed microbatches'
    metrics, folded."""

    batch: TrainingBatch
    groups: tuple[tuple[int, ...], ...]
    next_microbatch: int = 0
    evidence: Metrics | None = None

    def relocate_evidence(self, device: torch.device | str) -> None:
        if self.evidence is not None:
            self.evidence = self.evidence.relocate(device)


class TrainingAllocation(Protocol):
    """Where training allocates device memory: a residency owner's regions, as
    ``MemoryManager`` provides them, or the default allocator."""

    def persistent_training_allocation(self) -> AbstractContextManager[None]: ...

    def transient_training_allocation(self, scope_bytes: int) -> AbstractContextManager[None]: ...

    def optimizer_training_allocation(self) -> AbstractContextManager[None]: ...


class UnmanagedTrainingAllocation:
    """Training memory no residency owner manages: the default allocator, which never fails
    as an owner."""

    def persistent_training_allocation(self) -> AbstractContextManager[None]:
        return nullcontext()

    def transient_training_allocation(self, scope_bytes: int) -> AbstractContextManager[None]:
        return nullcontext()

    def optimizer_training_allocation(self) -> AbstractContextManager[None]:
        return nullcontext()


class _DeviceStackLimit:
    """The current context's per-thread stack size, which the driver only grows: a kernel
    needing more local memory than it allows grows every resident thread's reserve, outside
    any allocator, for the life of the process."""

    # CU_LIMIT_STACK_SIZE
    _STACK_SIZE = 0

    def __init__(self) -> None:
        self._get_limit = ctypes.CDLL("libcuda.so.1").cuCtxGetLimit

    def __call__(self) -> int:
        size = ctypes.c_size_t()
        status = self._get_limit(ctypes.byref(size), self._STACK_SIZE)
        if status:
            raise RuntimeError(f"cuCtxGetLimit(CU_LIMIT_STACK_SIZE) failed with CUresult {status}")
        return size.value


def device_stack_limit() -> Callable[[], int]:
    """A reader of the current CUDA context's per-thread stack size, which a runner built
    with it checks no command grows once the inference state is sized."""

    return _DeviceStackLimit()


class TrainingRunner:
    """One rank's training execution, its only GPU-side owner: SFT and GRPO
    forward/backward and selected-parameter updates on the training stream, after
    inference, in training's memory.

    It plans a batch (``forward_backward``), runs its microbatches, commits or discards
    the gradients, captures or restores the optimizer, and moves training state off the
    device and back, as the training scheduler asks. It keeps no submission ids and
    checks no counts. A ``Refusal`` leaves training as it was; any other error ends the
    rank's process.
    """

    @classmethod
    def create(
        cls,
        model: TrainingModel,
        config: TrainingConfig,
        *,
        trainables: SupportsParameterUniverse,
        stream: torch.cuda.Stream,
        after: torch.cuda.Stream,
        memory: TrainingAllocation | None,
        scope_bytes: int,
        model_signature: Callable[[], str],
        adapter_writer: Callable[[str, Mapping[str, torch.Tensor]], None],
        ranks: Ranks | None,
        placed: PlacedParameters | None,
        stack_limit: Callable[[], int] | None,
        checks: bool,
        residency: TrainingStateResidency | None = None,
    ) -> TrainingRunner:
        """The runner's only constructor. It resolves ``config``'s trainable plan over
        ``trainables``, the owner of the tensors ``model`` computes with. On
        tensor-parallel ``ranks``, ``placed`` says where they hold each selected parameter
        (no ranks: this rank's are the logical ones). Work runs on ``stream`` after
        ``after``, the inference execution stream; the FP32 master gradients outlive their
        microbatch, so they accumulate in ``memory``'s persistent region (without it,
        training allocates from the default allocator), and each command's temporaries
        take a ``scope_bytes`` transient scope. ``model_signature`` identifies the
        construction a checkpoint belongs to; ``adapter_writer(path, factors)`` writes the
        adapter's whole factors by name as a PEFT adapter; ``stack_limit`` (``device_stack_limit``)
        reads the device's per-thread stack size, which no command may grow past its value
        now, once the inference state is sized; ``checks`` runs training's internal and
        expensive checks."""

        self = super().__new__(cls)
        allocation = UnmanagedTrainingAllocation() if memory is None else memory

        batching = config.batching
        sft_profile = config.sft_profile
        if sft_profile is not None:
            try:
                profile_groups = plan_microbatches(batching.policy, sft_profile.sequence_lengths)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "SFT shape profile exceeds the configured batching policy"
                ) from error
            if profile_groups != (tuple(range(sft_profile.batch_size)),):
                raise ValueError(
                    "an exact SFT shape profile must fit one configured logical microbatch"
                )
            if (
                batching.layout is BatchLayout.PADDED
                and sft_profile.padded_input_tokens > batching.policy.max_tokens_per_microbatch
            ):
                raise ValueError("SFT shape profile's padding exceeds the microbatch token bound")
        checkpointing = config.activation_checkpointing
        trainable_parameters = resolve_for_model(config.trainable_plan, trainables)

        self.model = model
        self.trainable_parameters = trainable_parameters
        self._batching = batching
        self._checks = checks
        named = trainable_parameters.named_parameters()
        layout = (
            LocalLayout()
            if ranks is None
            else tensor_parallel_layout(
                ranks, placed, [name for name, _ in named], device=named[0][1].device
            )
        )
        # Its masters and moments are created by the state's residency owner, below.
        self._optimizer = AdamWOptimizer(
            trainable_parameters, config.optimizer, checks=checks, gradients=layout, deferred=True
        )
        # The masters exist once the state is first held (``_hold_state``), which binds them.
        self._trainables = trainables
        self._bound = False
        # Every rank gathers a checkpoint's logical tensors; the primary publishes it.
        self._publishes = layout.primary
        self._stream = stream
        self._after = after
        self.device = self._optimizer.device
        # What the runner pins on the host: the gradient stash, the offload's arena and the
        # optimizer state's mirror.
        self._pinned = PinnedHostMemory()
        # What the optimizer's state leaves the device with while serving holds it.
        # ``residency`` as the rank resolved it for its device; else as configured.
        self._state = training_state_residency(
            residency or config.training_state_residency or TrainingStateResidency.RESIDENT,
            self._optimizer,
            tuple(parameter for _, parameter in named),
            host=self._pinned,
            allocation=allocation,
            stream=stream,
        )
        if self._optimizer.materialized:
            # A resident state exists from the start, and so does what training reads.
            self._bind()
        self._suspended = False
        self._allocation = allocation
        self._persistent_allocation = allocation.persistent_training_allocation
        self._transient = partial(allocation.transient_training_allocation, scope_bytes)
        self._model_signature = model_signature
        self._adapter_writer = adapter_writer
        self._stack_limit = stack_limit
        self._stack_bytes = None if stack_limit is None else stack_limit()
        self._accepted: _AcceptedBatch | None = None
        # Staging needs the device, so it follows the optimizer.
        self._staging = (
            None if sft_profile is None else SFTStaging(sft_profile, batching.layout, self.device)
        )
        self._boundary_offload: HostBoundaryOffload | None = None
        if checkpointing is not ActivationCheckpointMode.NONE:
            layers = trainables.decoder_layer_trainable_parameters()
            first = first_trainable_layer(trainables, trainable_parameters.selected_names)
            if checkpointing is ActivationCheckpointMode.FULL_OFFLOAD:
                width, dtype = model.residual_stream_layout()
                # Pinning the arena is the last step that may refuse the configuration.
                # Every segment of every layer from first is one checkpoint, and no
                # microbatch holds more tokens than the batching policy's bound.
                self._boundary_offload = HostBoundaryOffload(
                    segments=(len(layers) - first) * model.TRAINING_LAYER_SEGMENTS,
                    tokens=batching.policy.max_tokens_per_microbatch,
                    width=width,
                    dtype=dtype,
                    device=self.device,
                    host=self._pinned,
                )
            model.enable_activation_checkpointing(
                ReentrantActivationCheckpoint(first, self._boundary_offload)
            )
        return self

    def forward_backward(self, batch: TrainingBatch) -> None:
        """Validate and plan one logical batch on the host and accept it; ``microbatch``
        runs it."""

        if self._accepted is not None:
            raise RuntimeError("another forward/backward batch is already accepted")
        if not isinstance(batch, TrainingBatch):
            raise TrainingInputError("batch must be a TrainingBatch")
        try:
            layout = batch.tokens.LAYOUT
            if layout is not self._batching.layout:
                raise ValueError(
                    f"{layout.value} batch does not match configured "
                    f"{self._batching.layout.value} training layout"
                )
            groups = plan_microbatches(self._batching.policy, batch.tokens.sequence_lengths())
            batch.tokens.check_width(groups, self._batching.policy.max_tokens_per_microbatch)
            if self._staged(batch):
                self._staging.require_profile(batch)
            self.model.validate_training_inputs(batch.objective.supplied_token_ids(batch))
            self.model.validate_training_targets(batch.tokens.target_token_ids())
        except (TypeError, ValueError) as error:
            raise TrainingInputError(str(error)) from error

        self._accepted = _AcceptedBatch(batch, groups)

    def microbatch(self) -> Metrics | None:
        """Run the accepted batch's next microbatch; after its last, release the batch and
        return its metrics on the host."""

        accepted = self._accepted
        if accepted is None:
            raise RuntimeError("no forward/backward batch is accepted")
        self._hold_state()
        result = self._run(
            "training forward/backward microbatch", self._advance, accepted, scope=self._transient
        )
        # The device is done with it: an unobserved result keeps no device memory.
        return None if result is None else result.relocate("cpu")

    def _advance(self, accepted: _AcceptedBatch) -> Metrics | None:
        """Run ``accepted``'s next microbatch and fold its metrics into the batch's; return
        them after the last. A failure unwinds the accumulation window and releases the
        batch; one that discards earlier microbatches' gradients ends the process."""

        had_accumulation = self._optimizer.has_pending_accumulation
        indices = accepted.groups[accepted.next_microbatch]
        try:
            self._state.ready_for_forward()
            fragment = self._forward(accepted.batch, indices)
            evidence = self._backward_and_accumulate(fragment, accepted.evidence)
        except BaseException as error:
            self._accepted = None
            self._discard_accumulation()
            if self._boundary_offload is not None:
                self._boundary_offload.abort()
            if had_accumulation:
                raise RuntimeError(
                    "A microbatch failed and discarded an existing accumulation window"
                ) from error
            raise
        accepted.evidence = evidence
        accepted.next_microbatch += 1
        if accepted.next_microbatch < len(accepted.groups):
            return None
        self._accepted = None
        return evidence

    def _staged(self, batch: TrainingBatch) -> bool:
        """Whether the exact SFT shape profile stages ``batch``."""

        return self._staging is not None and isinstance(batch.objective, SFT)

    def _forward(self, batch: TrainingBatch, indices: tuple[int, ...]) -> Metrics:
        rows = (
            self._staging.stage(batch)
            if self._staged(batch)
            else batch.microbatch(indices, self.device)
        )
        hidden_states = self.model.training_hidden_states(rows.input_ids, rows.packing)
        if rows.target_indices is not None:
            hidden_states = hidden_states.index_select(0, rows.target_indices)
        return batch.objective.metrics(self.model, batch, indices, hidden_states, rows)

    def _require_no_accepted(self) -> None:
        if self._accepted is not None:
            raise RuntimeError("the accepted forward/backward batch must complete first")

    def _backward_and_accumulate(self, fragment: Metrics, evidence: Metrics | None) -> Metrics:
        """Run the microbatch's backward, then fold its evidence into the logical batch's."""

        loss_sum = fragment.loss_sum
        # Backward runs on this thread, so it allocates in the pool forward and
        # recomputation use: a pool selected for a thread does not follow the work to
        # autograd's device threads. With one GPU per process no parallelism is lost.
        # Backward adds into the window's gradients, which a resumption may still be
        # copying back.
        self._state.ready_for_accumulation()
        with torch.autograd.set_multithreading_enabled(False):
            loss_sum.backward()
        if self._boundary_offload is not None:
            self._boundary_offload.require_drained()
        # What outlives the microbatch leaves its scope, whose pool a later one may replace;
        # its evidence takes no more device memory however many microbatches follow.
        with self._persistent_allocation():
            self._optimizer.accumulate_gradients()
            detached = fragment.relocate(self.device)
            return detached if evidence is None else evidence.fold(detached)

    def commit(
        self, normalization_count: int, results: tuple[Metrics, ...]
    ) -> TrainingCommitResult:
        """Validate the window's gradients and host ``results`` behind one gate and step the
        optimizer; an invalid window raises InvalidGradientError and is discarded."""

        self._require_no_accepted()
        self._hold_state()
        return self._run(
            "training optimizer commit",
            self._commit,
            normalization_count,
            results,
            # The step's temporaries belong to the window, beside the state allocated at
            # startup.
            scope=self._transient,
        )

    def discard(self) -> None:
        """Discard an uncommitted accumulation window without updating parameters."""

        self._require_no_accepted()
        self._run("training accumulation cleanup", self._discard_accumulation)

    def _discard_accumulation(self) -> None:
        # Gradients a resumption still copies back must land before their memory is freed.
        self._state.ready_for_accumulation()
        self._optimizer.discard_accumulation()

    @property
    def holds_gradients(self) -> bool:
        """Whether a batch is accepted or the device holds gradients."""

        return self._accepted is not None or self._optimizer.has_pending_accumulation

    def memory_usage(self) -> TrainingMemoryUsage:
        """What the planned state components hold on the device now: bytes and pools."""

        staging = () if self._staging is None else self._staging.device_tensors()
        return TrainingMemoryUsage.measure(
            {**self._optimizer.device_state(), "input_staging": staging}
        )

    def accumulated_gradients(self) -> tuple[torch.Tensor, ...]:
        """A read-only view for equivalence checks: host copies of this rank's FP32 gradients
        of the open accumulation window, one per trainable parameter in order."""

        return self._optimizer.accumulated_gradients()

    def checkpoint(self, *, num_optimizer_steps: int, model_revision: int) -> TrainingCheckpoint:
        """Capture the settled optimizer state as a checkpoint of this construction."""

        model_signature = self._model_signature()
        self._require_settled()
        self._hold_state()
        optimizer = self._run(
            "training checkpoint capture",
            self._capture_optimizer,
            scope=self._allocation.persistent_training_allocation,
        )
        return TrainingCheckpoint(
            model_signature=model_signature,
            num_optimizer_steps=num_optimizer_steps,
            model_revision=model_revision,
            optimizer=optimizer,
        )

    def save(
        self, path: str | os.PathLike[str], *, num_optimizer_steps: int, model_revision: int
    ) -> None:
        """Capture a checkpoint on every rank and publish it at ``path`` from the first."""

        checkpoint = self.checkpoint(
            num_optimizer_steps=num_optimizer_steps, model_revision=model_revision
        )
        if self._publishes:
            checkpoint.save(path)

    def save_adapter(self, path: str) -> None:
        """Gather the adapter's trained factors (the masters, published or not) on every rank
        and write them as a PEFT adapter at ``path`` from the first."""

        self._require_settled()
        self._hold_state()
        factors = self._run(
            "adapter capture",
            self._trained_state,
            scope=self._allocation.persistent_training_allocation,
        )
        if self._publishes:
            self._adapter_writer(path, factors)

    def publish(self) -> None:
        """Copy the trained masters into the live factors inference reads: the only way a
        training step reaches inference."""

        self._require_no_accepted()
        self._hold_state()
        self._run("adapter publish", self._publish)

    def restore(self, checkpoint: TrainingCheckpoint) -> None:
        """Restore one checkpoint of this construction in place."""

        self._require_settled()
        if checkpoint.model_signature != self._model_signature():
            raise ValueError("checkpoint model signature does not match this runtime")
        self._run("training checkpoint restore", self._state.restore, checkpoint.optimizer)

    def _hold_state(self) -> None:
        """Have training's state on the device, which training holds: created there at the
        first call under an offloading residency, and then bound into the model, which
        trains on the FP32 masters from here on (inference keeps the published values)."""

        self._state.allocate()
        if not self._bound:
            self._bind()

    def _bind(self) -> None:
        self._trainables.bind_training_weights(self._optimizer.training_weights())
        self._bound = True

    def _publish(self) -> None:
        self._state.ready_for_forward()
        self._optimizer.publish()

    def _trained_state(self) -> dict[str, torch.Tensor]:
        self._state.ready_for_forward()
        return self._optimizer.trained_state()

    def _capture_optimizer(self) -> OptimizerState:
        self._state.ready_for_step()
        return self._optimizer.checkpoint_state()

    def suspend(self) -> None:
        """Move what training's state residency sends to the host there, and the accepted
        batch's metrics, before training memory is released; it runs in a switch, where any
        failure ends the rank's process."""

        with torch.cuda.stream(self._stream):
            self._state.suspend()
        self._stream.synchronize()
        self._suspended = True
        if self._accepted is not None:
            self._accepted.relocate_evidence("cpu")

    def resume(self) -> None:
        """Bring back what ``suspend`` moved off, if anything, once training memory returns;
        it runs in a switch, where any failure ends the rank's process."""

        if not self._suspended:
            self._hold_state()
            return
        self._suspended = False
        with torch.cuda.stream(self._stream), self._persistent_allocation():
            self._state.resume()
            if self._accepted is not None:
                self._accepted.relocate_evidence(self.device)

    def close(self) -> None:
        """Release what the runner holds outside device memory: its pinned host memory.

        The runner trains no more afterwards.
        """

        if self._boundary_offload is not None:
            self._boundary_offload.close()
        self._state.close()
        if self._bound:
            self._trainables.release_training_weights()
        self._pinned.close()

    def _commit(
        self, normalization_count: int, results: tuple[Metrics, ...]
    ) -> TrainingCommitResult:
        self._state.ready_for_step()
        gradients, normalized_norm = self._optimizer.validate_gradients(normalization_count)
        grad_norm = float(normalized_norm)
        if not math.isfinite(grad_norm):
            self._optimizer.discard_accumulation()
            raise InvalidGradientError("Selected gradients must have a finite global norm")

        try:
            completions = Metrics.complete(results)
        except InvalidGradientError:
            self._optimizer.discard_accumulation()
            raise

        self._optimizer.apply_gradients(
            normalization_count,
            gradients=gradients,
            normalized_norm=normalized_norm,
        )
        self._state.after_step()
        return TrainingCommitResult(completions, grad_norm)

    def _require_settled(self) -> None:
        self._require_no_accepted()
        self._optimizer.require_no_pending_accumulation()

    def _run(
        self,
        description: str,
        call: Callable[..., _T],
        *args: object,
        scope: Callable[[], AbstractContextManager[None]] = nullcontext,
    ) -> _T:
        """Run ``call`` on the training stream after inference work and wait for it. Once the
        stream drains, a ``Refusal`` it raises leaves training as it was; any other error
        ends the rank's process."""

        self._stream.wait_stream(self._after)
        try:
            with torch.cuda.stream(self._stream), scope():
                value = call(*args)
        finally:
            self._stream.synchronize()
        self._require_stack_unchanged(description)
        return value

    def _require_stack_unchanged(self, description: str) -> None:
        """Raise if ``description`` grew the device's local memory past the KV budget: its
        kernels must be warmed up before the inference state is sized."""

        if self._stack_limit is None or self._stack_bytes is None:
            return
        stack_bytes = self._stack_limit()
        if stack_bytes > self._stack_bytes:
            raise RuntimeError(
                f"{description} grew the device's per-thread stack from "
                f"{self._stack_bytes} to {stack_bytes} bytes after the inference state was "
                "sized: a kernel first launched now grew local memory outside the budget"
            )


__all__ = [
    "ModelParameterUniverse",
    "SupportsParameterUniverse",
    "TrainingAllocation",
    "TrainingMemoryPlan",
    "TrainingRunner",
    "device_stack_limit",
    "prepare_model_training",
]
