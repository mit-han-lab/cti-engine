"""What training's state leaves the device with while serving holds it: one owner per runner,
built for a ``TrainingStateResidency``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Protocol

import torch

from cti.config import TrainingStateResidency

from ._checkpoint import OptimizerState
from ._errors import TrainingResidencyError
from ._host_memory import PinnedHostMemory
from ._optimizer import AdamWOptimizer, separate_storage


class OptimizerAllocation(Protocol):
    def optimizer_training_allocation(self) -> AbstractContextManager[None]: ...


class ResidentTrainingState:
    """``RESIDENT``: the FP32 masters, AdamW moments and gradients stay on the device
    through serving's turn, created with the runner. Nothing moves at a switch, so every
    method but ``restore`` is a no-op."""

    def __init__(self, optimizer: AdamWOptimizer) -> None:
        self._optimizer = optimizer
        optimizer.materialize()

    def allocate(self) -> None:
        """Have the state on the device, which training holds; called on the training thread
        outside any training allocation scope."""

    def ready_for_forward(self) -> None:
        """Order the current stream's next forward after the masters' copy back."""

    def ready_for_accumulation(self) -> None:
        """Order the current stream's next backward, which adds into the gradients, after
        their copy back."""

    def ready_for_step(self) -> None:
        """Order the current stream's next read or write of the masters and moments (an
        optimizer step, a checkpoint) after every copy of them."""

    def after_step(self) -> None:
        """The masters and moments changed on the current stream: an optimizer step or a
        restore."""

    def restore(self, state: OptimizerState) -> None:
        """Copy a checkpoint into the masters, moments and live parameters in place; the
        caller holds no accepted microbatch or accumulation."""

        self._optimizer.restore_checkpoint_state(state)

    def suspend(self) -> None:
        """Serving takes the device next: wait for the copies its memory must keep, and queue
        on the current stream those of what leaves with it; the caller waits for the
        stream."""

    def resume(self) -> None:
        """Training holds the device again: queue the copies back of what ``suspend`` moved,
        into memory resumed at its addresses."""

    def close(self) -> None:
        """Training ends; the runner's pinned memory owner releases the host copies."""


class _FlatMirror:
    """One class of mirrored state (the moments, or the masters) as views of one flat FP32
    device buffer, created in the optimizer region, beside its pinned host copy: each
    direction is one copy, whatever the number of tensors."""

    def __init__(
        self,
        host: PinnedHostMemory,
        region: Callable[[], AbstractContextManager[None]],
        purpose: str,
    ) -> None:
        self._host = host
        self._region = region
        self._purpose = purpose
        self._device: torch.Tensor | None = None
        self._host_copy: torch.Tensor | None = None

    def storage(self, shapes: Sequence[torch.Size], device: torch.device) -> list[torch.Tensor]:
        """The views, zero, one per shape (``AdamWOptimizer``'s ``StateStorage``)."""

        sizes = [shape.numel() for shape in shapes]
        with self._region():
            self._device = torch.zeros(sum(sizes), dtype=torch.float32, device=device)
        self._host_copy = self._host.allocate(4 * sum(sizes), self._purpose).view(torch.float32)
        return [
            view.view(shape) for view, shape in zip(self._device.split(sizes), shapes, strict=True)
        ]

    def copy_out(self) -> None:
        if self._device is not None:
            self._host_copy.copy_(self._device, non_blocking=True)

    def copy_back(self) -> None:
        if self._device is not None:
            self._device.copy_(self._host_copy, non_blocking=True)


class MirroredTrainingState(ResidentTrainingState):
    """The state that leaves with serving's turn lives in the training-optimizer region, a
    no-backup region the switch pauses, one flat buffer per class beside its pinned host
    mirror, refreshed on a copy stream after every step in one copy per class: a suspension
    waits for that copy, usually long done, and copies nothing; a resumption copies back on
    the copy stream, which only the state's next reader waits for (``ready_for_*``).

    Its tensors are created when training first holds the device (``allocate``), since the
    KV pool sized at startup counts on their bytes, and mirrored at once. A checkpoint
    restored before that writes the live parameters at once, which inference reads, and the
    rest once they exist. Every write into them runs outside autograd: the masters are
    leaves.
    """

    # Whether the masters, beside the moments, live in the region and are mirrored, and
    # whether an open window's gradients go to a pinned stash at a switch.
    _MIRRORS_MASTERS = False
    _STASHES_GRADIENTS = False

    def __init__(
        self,
        optimizer: AdamWOptimizer,
        *,
        host: PinnedHostMemory,
        region: Callable[[], AbstractContextManager[None]],
        stream: torch.cuda.Stream,
    ) -> None:
        self._optimizer = optimizer
        self._host = host
        self._region = region
        self._stream = stream
        # The masters' and the moments' flat buffers with their host copies; the masters are
        # mirrored only where they leave with serving's turn.
        self._masters = (
            _FlatMirror(host, region, "the master weights' mirror")
            if self._MIRRORS_MASTERS
            else None
        )
        self._moments = _FlatMirror(host, region, "the optimizer state mirror")
        self._copy_stream: torch.cuda.Stream | None = None
        self._gradient_stash: torch.Tensor | None = None
        # The copy stream's last event, and those after the masters' and the gradients'
        # copies back; None when nothing is in flight for that reader.
        self._copied: torch.cuda.Event | None = None
        self._masters_back: torch.cuda.Event | None = None
        self._gradients_back: torch.cuda.Event | None = None
        self._stashed = False
        self._pending: OptimizerState | None = None

    def allocate(self) -> None:
        optimizer = self._optimizer
        if optimizer.materialized:
            return
        with torch.cuda.stream(self._stream):
            optimizer.materialize(
                masters=separate_storage if self._masters is None else self._masters.storage,
                moments=self._moments.storage,
            )
            if self._STASHES_GRADIENTS:
                self._gradient_stash = self._host.allocate(
                    optimizer.gradient_bytes, "the gradient stash"
                ).view(torch.float32)
            self._copy_stream = torch.cuda.Stream(optimizer.device)
            if self._pending is not None:
                optimizer.restore_checkpoint_state(self._pending)
                self._pending = None
            self.after_step()

    def ready_for_forward(self) -> None:
        self._wait(self._masters_back)

    def ready_for_accumulation(self) -> None:
        self._wait(self._gradients_back)

    def ready_for_step(self) -> None:
        self._wait(self._copied)

    def after_step(self) -> None:
        self._on_copy_stream(*self._mirrors(), copy=_FlatMirror.copy_out)

    def restore(self, state: OptimizerState) -> None:
        if not self._optimizer.materialized:
            self._optimizer.restore_live_state(state)
            self._pending = state
            return
        self.ready_for_step()
        super().restore(state)
        self.after_step()

    def suspend(self) -> None:
        if self._copied is not None:
            self._copied.synchronize()
        self._stashed = self._gradient_stash is not None and (
            self._optimizer.has_pending_accumulation
        )
        if self._stashed:
            self._optimizer.suspend_gradients(self._gradient_stash)

    def resume(self) -> None:
        masters = () if self._masters is None else (self._masters,)
        self._masters_back = (
            self._on_copy_stream(*masters, copy=_FlatMirror.copy_back) if masters else None
        )
        self._gradients_back = None
        if self._stashed:
            with torch.cuda.stream(self._copy_stream), torch.no_grad():
                self._optimizer.resume_gradients(self._gradient_stash)
                self._gradients_back = self._record()
            self._stashed = False
        self._on_copy_stream(self._moments, copy=_FlatMirror.copy_back)

    def _mirrors(self) -> tuple[_FlatMirror, ...]:
        return (self._moments,) if self._masters is None else (self._masters, self._moments)

    def _on_copy_stream(
        self, *mirrors: _FlatMirror, copy: Callable[[_FlatMirror], None]
    ) -> torch.cuda.Event | None:
        """``copy(mirror)`` for each mirror on the copy stream after the current stream's
        queued work, outside autograd; the event after them."""

        stream = self._copy_stream
        if stream is None:
            return None
        stream.wait_stream(torch.cuda.current_stream(stream.device))
        with torch.cuda.stream(stream), torch.no_grad():
            for mirror in mirrors:
                copy(mirror)
            return self._record()

    def _record(self) -> torch.cuda.Event:
        self._copied = torch.cuda.Event()
        self._copied.record(self._copy_stream)
        return self._copied

    @staticmethod
    def _wait(event: torch.cuda.Event | None) -> None:
        if event is not None:
            torch.cuda.current_stream(event.device).wait_event(event)


class OffloadedOptimizerState(MirroredTrainingState):
    """``OFFLOAD_OPTIMIZER``: the AdamW moments leave with serving's turn; the masters and
    gradients stay on the device (the persistent region stays mapped)."""


class OffloadedTrainingState(MirroredTrainingState):
    """``OFFLOAD_ALL``: the masters leave with the moments, and an open window's gradients
    go to a pinned stash at the switch (only when it holds any); the persistent region is
    paused too."""

    _MIRRORS_MASTERS = True
    _STASHES_GRADIENTS = True


def training_state_residency(
    residency: TrainingStateResidency,
    optimizer: AdamWOptimizer,
    parameters: tuple[torch.Tensor, ...],
    *,
    host: PinnedHostMemory,
    allocation: OptimizerAllocation,
    stream: torch.cuda.Stream,
) -> ResidentTrainingState:
    """The owner of ``optimizer``'s state over the selected ``parameters`` under
    ``residency``; an offloading residency places what leaves in ``allocation``'s optimizer
    region, created on training's ``stream``, and takes BF16 parameters on CUDA only (LoRA
    factors follow the model dtype)."""

    if residency is TrainingStateResidency.RESIDENT:
        return ResidentTrainingState(optimizer)
    others = sum(parameter.dtype != torch.bfloat16 for parameter in parameters)
    if others:
        raise TrainingResidencyError(
            f"training_state_residency {residency.value} trains BF16 parameters only; "
            f"{others} of the {len(parameters)} selected parameters are not BF16"
        )
    if optimizer.device.type != "cuda":
        raise TrainingResidencyError(
            f"training_state_residency {residency.value} needs a CUDA device, not "
            f"{optimizer.device.type}"
        )
    owner = (
        OffloadedTrainingState
        if residency.offloads_masters_and_gradients
        else OffloadedOptimizerState
    )
    return owner(
        optimizer, host=host, region=allocation.optimizer_training_allocation, stream=stream
    )


__all__ = [
    "MirroredTrainingState",
    "OffloadedOptimizerState",
    "OffloadedTrainingState",
    "ResidentTrainingState",
    "training_state_residency",
]
