from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from cti.distributed import ParallelGroup, Topology
from cti.memory import MemoryManager, Mode
from cti.utils import init_logger

from ._device_memory import DeviceMemory
from ._model_host import HostedCheckpoint, ModelHost

if TYPE_CHECKING:
    from collections.abc import Callable

    from cti.plan import Plan

logger = init_logger(__name__)


@dataclass(frozen=True)
class Streams:
    """The rank's streams: inference executes on ``execution`` and prepares its inputs on
    ``preparation``; training runs on ``training``, None without training."""

    execution: torch.cuda.Stream
    preparation: torch.cuda.Stream
    training: torch.cuda.Stream | None


class GPUWorker:
    """Rank-local owner of the device's resources: the parallel group, the streams, the
    device memory and the hosted models. What runs on them, the composition root builds."""

    def __init__(
        self,
        plan: Plan,
        checkpoint: HostedCheckpoint,
        *,
        rank: int = 0,
        init_method: str | None = None,
        memory: MemoryManager | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Hosts ``plan``, the composition root's plan of ``checkpoint`` for this rank's
        device: ``rank`` is the global rank, run on ``plan.local_device``; ``init_method`` is
        the multi-rank rendezvous; ``memory`` places what the worker keeps in its residencies,
        or there are none; ``progress`` hears each long startup phase as it begins.

        A failed construction releases nothing: the rank's process ends, and a peer may
        still be inside a collective that a teardown would wait on."""
        progress = progress or _quiet
        self._memory_manager = memory_manager = memory
        self._resources = resources = ExitStack()

        device = torch.device("cuda", plan.local_device)
        self.ranks = parallel = ParallelGroup(
            Topology(rank, plan.num_ranks),
            device=device,
            timeout_seconds=plan.distributed_timeout,
            init_method=init_method,
            use_pynccl=plan.use_pynccl,
        )
        resources.callback(parallel.close)
        # The ranks' one vote: device memory's readings and a resume's free bytes.
        self._vote = parallel.min_max_int
        # Logged once the process group names rank 0.
        attention = plan.attention
        logger.info_rank0("Selected attention backend %s: %s", attention.selected, attention.reason)
        budget = plan.prompt_budget
        logger.info_rank0("Prompt tokens per step: %d (%s)", budget.tokens, budget.reason)
        logger.info_rank0("Page size %d for %s", plan.page_size, attention.selected)

        torch.manual_seed(plan.initialization_seed)
        self.streams = streams = Streams(
            execution=torch.cuda.Stream(device=device),
            preparation=torch.cuda.Stream(device=device),
            training=None if plan.training is None else torch.cuda.Stream(device=device),
        )
        torch.cuda.set_stream(streams.execution)
        self._device = device
        self.memory = DeviceMemory(plan, device=device, agree=self._vote, manager=memory_manager)
        # What training keeps on the device while serving holds it.
        self.training_state_residency = (
            None
            if plan.training is None
            else self.memory.training_state_residency(plan.training.training_state_residency)
        )
        self.host = ModelHost(
            plan,
            checkpoint,
            device=device,
            ranks=parallel,
            training=streams.training is not None,
            progress=progress,
        )

    # ================================== Lifecycle ==================================

    def release(self, mode: Mode) -> None:
        """Give back the shared device memory ``mode`` holds: serving's KV pool, or
        training's parameters and optimizer state. The side that held it has let go."""

        memory = self._require_memory_manager()
        if mode is Mode.SERVING:
            memory.suspend_kv()
        else:
            residency = self.training_state_residency
            memory.suspend_training(
                keep_persistent=residency is not None
                and not residency.offloads_masters_and_gradients
            )

    def acquire(self, mode: Mode) -> None:
        """Take it back for ``mode``; raises ``ResidencyRestoreError`` on every rank when it
        does not fit any rank's device, before any rank maps it."""

        memory = self._require_memory_manager()
        if mode is Mode.SERVING:
            memory.resume_kv(self._fewest)
        else:
            memory.resume_training(self._fewest)

    def _fewest(self, spare_bytes: int) -> int:
        """The fewest bytes any rank keeps free: each rank's one vote on a resume."""

        return self._vote(spare_bytes)[0]

    def _require_memory_manager(self) -> MemoryManager:
        if self._memory_manager is None:
            raise RuntimeError("switching needs managed memory")
        return self._memory_manager

    def quiesce(self) -> None:
        torch.cuda.synchronize(self._device)
        self.ranks.barrier()

    def close(self) -> None:
        """Release what the worker built, in reverse, for a process that goes on without it; a
        serving rank never does, it exits."""

        self._resources.close()


def _quiet(phase: str) -> None:
    pass


__all__ = ["GPUWorker"]
