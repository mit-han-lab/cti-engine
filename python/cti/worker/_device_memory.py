# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/engine/engine.py`); see THIRD_PARTY_NOTICES.

"""The rank's device memory: the free memory the ranks agree on, the budget taken from it,
and the inference state planned and allocated in what the built models leave."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Protocol

from cti.config import TrainingStateResidency
from cti.memory import GraphPool, free_bytes, native_backing, unified_memory
from cti.utils import init_logger, mem_GB

if TYPE_CHECKING:
    import torch

    from cti.kvcache import (
        InferenceState,
        KVLayout,
        SnapshotPolicy,
        StateCapacity,
        TableGeometry,
    )
    from cti.memory import MemoryManager
    from cti.plan import Plan

logger = init_logger(__name__)


class TrainingFootprint(Protocol):
    """What training keeps on the device (``training.runner.TrainingMemoryPlan``): through
    inference, at its peak, and the components that make up the peak."""

    @property
    def resident_bytes(self) -> int: ...

    @property
    def training_bytes(self) -> int: ...

    def largest(self, count: int) -> tuple[tuple[str, int], ...]: ...


class TrainingMemoryError(ValueError):
    """The configured training does not fit the device beside the inference state."""


class DeviceMemory:
    """A rank's device memory as inference plans it: the free memory at boot, the inference
    budget and the peak reserve taken from it, the inference state's capacity planned in
    what the built models leave, and that state's allocation. Every reading settles the
    allocator and takes the least free memory across the ranks, whose minimum and maximum
    of a value ``agree`` returns; ``manager`` places the
    state in its KV residency, or there is none."""

    def __init__(
        self,
        plan: Plan,
        *,
        device: torch.device,
        agree: Callable[[int], tuple[int, int]],
        manager: MemoryManager | None,
    ) -> None:
        self._device, self._agree, self._manager = device, agree, manager
        self._unified = unified_memory(device)
        self._boot_local_bytes, self.boot_bytes = self._free()
        ratio, source = self._inference_ratio(plan.inference_ratio)
        self.budget_bytes = int(ratio * self.boot_bytes)
        self._peak_ratio = ratio if plan.peak_ratio is None else plan.peak_ratio
        if self._peak_ratio < ratio:
            raise ValueError(
                f"peak_ratio must be in [inference_ratio, 1]; inference_ratio is {source}"
            )
        self.reserve_bytes = self.boot_bytes - int(self._peak_ratio * self.boot_bytes)
        self._recurrent_to_kv_ratio = plan.recurrent_to_kv_ratio
        logger.info_rank0(f"Free memory before loading model: {mem_GB(self.boot_bytes)}")
        logger.info_rank0(f"Inference memory ratio: {source}")
        logger.info_rank0(
            "Memory budget: inference %s, peak reserve %s",
            mem_GB(self.budget_bytes),
            mem_GB(self.reserve_bytes),
        )

    def _inference_ratio(self, explicit: float | None) -> tuple[float, str]:
        """The inference ratio and where it comes from: ``explicit``, else the device's
        default, lower on unified memory, which the host shares."""

        if explicit is not None:
            return explicit, f"explicit {explicit}"

        if self._unified:
            return 0.6, "UMA default 0.6"
        return 0.9, "default 0.9"

    def training_state_residency(
        self, configured: TrainingStateResidency | None
    ) -> TrainingStateResidency:
        """``configured``, or by default what training keeps on this device while serving
        holds it: everything where it shares the host's memory, else all but the optimizer
        state."""

        if configured is not None:
            return configured
        if self._unified or self._device.type != "cuda":
            return TrainingStateResidency.RESIDENT
        return TrainingStateResidency.OFFLOAD_OPTIMIZER

    def free_bytes(self) -> int:
        return self._free()[1]

    def graph_pool(self) -> GraphPool:
        """A new pool inference graphs capture into, inside the manager's graph region."""

        manager = self._manager
        return GraphPool(nullcontext if manager is None else manager.graph_allocation)

    def peak_bytes(self, run: Callable[[], bool]) -> int:
        """The most memory ``run`` allocates beyond what is live, on any rank; 0 when ``run``
        reports it ran nothing."""

        import torch

        device = self._device
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        if not run():
            return 0
        torch.cuda.synchronize(device)
        return self._agree(torch.cuda.max_memory_allocated(device) - baseline)[1]

    def plan(
        self,
        layout: KVLayout,
        *,
        geometry: TableGeometry,
        snapshots: SnapshotPolicy | None,
        workspace_bytes: int,
        image_encoder_bytes: int,
        image_rows_bytes: int,
        beside_bytes: int,
        allocations: Callable[[int, int], tuple[int, ...]] | None,
        fit_slots: bool,
    ) -> StateCapacity:
        """The inference state's capacity in the budget the models leave: less the
        attention backends' ``workspace_bytes``, allocated after the pools, the image
        activations the peak reserve cannot hold, the encoder-output rows and the
        ``beside_bytes`` what runs beside inference keeps; ``allocations`` and ``fit_slots`` as
        ``kvcache.plan_capacity`` takes them. When what runs beside leaves inference too
        little, ``TrainingMemoryError`` says so. Refused under an allocator configuration
        the native placement does not model."""

        from cti import kvcache

        cuda = self._device.type == "cuda"
        options = native_backing.custom_placement_options() if cuda else ()
        if options:
            raise ValueError(
                "The inference state is planned for the caching allocator's native placement; "
                f"the allocator configuration sets {', '.join(options)}"
            )
        model_bytes = self._boot_local_bytes - self._free()[0]
        available, _ = self._agree(self.budget_bytes - model_bytes - workspace_bytes)
        budget = kvcache.StateBudget(
            available,
            native_backing if cuda else None,
            self._recurrent_to_kv_ratio,
            beside=beside_bytes,
            # Image activations share the peak reserve with text execution.
            image_encoder=max(0, image_encoder_bytes - self.reserve_bytes),
            image_rows=image_rows_bytes,
            fit_slots=fit_slots,
        )
        try:
            return kvcache.plan_capacity(
                layout, geometry, budget, snapshots=snapshots, allocations=allocations
            )
        except kvcache.CapacityError as error:
            if not error.beside:
                raise
            raise TrainingMemoryError(
                f"Training keeps {mem_GB(beside_bytes)} beside inference, which "
                f"leaves the inference state too little memory: {error}"
            ) from error

    def allocate(
        self,
        layout: KVLayout,
        capacity: StateCapacity,
        image_rows: tuple[int, int] | None,
        dtype: torch.dtype,
    ) -> tuple[InferenceState, torch.Tensor | None]:
        """Every pool, then the encoder-output rows of shape ``image_rows``, if any, in the KV
        residency."""

        import torch

        from cti.kvcache import InferenceState, StateData

        manager = self._manager
        with nullcontext() if manager is None else manager.kv_allocation():
            data = StateData.allocate(layout, capacity, device=self._device)
            rows = None
            if image_rows is not None:
                rows = torch.empty(image_rows, dtype=dtype, device=self._device)
        # Outside the region: the metadata the state needs beside its data, which a KV
        # suspension must not lose.
        return InferenceState.over(data, device=self._device), rows

    def check_training_memory(self, training: TrainingFootprint) -> None:
        """Training keeps ``training.resident_bytes`` through inference, so that must fit beside
        it; its estimated peak only warns when the peak budget leaves less, since the estimate
        is not a hard bound. Checked once everything inference keeps through training exists.
        Ranks hold equal KV shards, so the least free memory decides; without a manager the
        inference state stays through training."""

        options = native_backing.custom_placement_options()
        if options:
            raise TrainingMemoryError(
                "Training memory is planned for the caching allocator's native placement; "
                f"the allocator configuration sets {', '.join(options)}"
            )
        free_bytes = self.free_bytes()
        if training.resident_bytes > free_bytes:
            raise TrainingMemoryError(
                f"Training keeps {mem_GB(training.resident_bytes)} beside inference, but the "
                f"inference state leaves {mem_GB(max(free_bytes, 0))} free; lower the LoRA rank "
                "or memory.inference_ratio"
            )
        released_bytes = 0 if self._manager is None else self._manager.kv_bytes
        room_bytes = free_bytes + released_bytes - self.reserve_bytes
        if training.training_bytes > room_bytes:
            logger.warning_rank0(
                "Training's estimated peak %s exceeds the %s the peak budget leaves "
                "(memory.peak_ratio %.2f, %s of the inference state released); largest terms: %s",
                mem_GB(training.training_bytes),
                mem_GB(max(room_bytes, 0)),
                self._peak_ratio,
                mem_GB(released_bytes),
                ", ".join(f"{name} {mem_GB(size)}" for name, size in training.largest(3)),
            )

    def _free(self) -> tuple[int, int]:
        """Local and least free memory across the ranks, after settling the allocator."""

        import torch

        torch.cuda.synchronize(self._device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self._device)
        local = free_bytes(self._device, unified=self._unified)
        least, most = self._agree(local)
        if most - least > 2 * 1024 * 1024 * 1024:
            if self._unified:
                # Hosts' unified memory differs by what each keeps cached and mapped, such as
                # the checkpoint pages its load left resident; every rank plans with the least.
                logger.warning_rank0(
                    "Unified memory differs across hosts: min %s, max %s; planning with the min",
                    mem_GB(least),
                    mem_GB(most),
                )
            else:
                logger.error(
                    "Memory across ranks is imbalanced: min %s, max %s",
                    mem_GB(least),
                    mem_GB(most),
                )
                raise RuntimeError("Memory across ranks is imbalanced")
        return local, least
