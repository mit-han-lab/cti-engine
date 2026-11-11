# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors

"""What inference's execution uses of the rank's devices, which the worker owns: the hosted
models, the device budget, and what runs beside inference on the same device."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from cti.kvcache import (
        InferenceState,
        KVLayout,
        SnapshotPolicy,
        StateCapacity,
        TableGeometry,
    )
    from cti.layers import LayerServices
    from cti.memory import GraphPool
    from cti.models import BaseLLMModel, CheckpointFacts, ModelConfig


class HostedCheckpoint(Protocol):
    """What execution reads of the checkpoint the models were built from."""

    facts: CheckpointFacts
    model_config: ModelConfig


class HostedModels(Protocol):
    """The rank's built and loaded models: the target, with the services its layers select
    their ops through, and the speculative draft, if any."""

    checkpoint: HostedCheckpoint
    device: torch.device
    model: BaseLLMModel
    services: LayerServices
    draft: BaseLLMModel | None

    @property
    def models(self) -> tuple[BaseLLMModel, ...]:
        """The target, then the draft."""
        ...

    def prepare_inference(
        self,
        max_rows: int,
        *,
        max_sequences: int,
        progress: Callable[[str], None],
    ) -> int:
        """Compile everything the models' inference forwards run up to ``max_rows`` rows and
        ``max_sequences`` sequences, and assign their shared split-K workspace; its bytes
        (``layers.prepare_inference``)."""
        ...


class DeviceBudget(Protocol):
    """The rank's device memory as the ranks agree on it: the free memory at boot, a run's
    peak, the pools graphs capture into, and the inference state planned and allocated in
    what the models leave."""

    boot_bytes: int

    def free_bytes(self) -> int: ...

    def peak_bytes(self, run: Callable[[], bool]) -> int: ...

    def graph_pool(self) -> GraphPool: ...

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
    ) -> StateCapacity: ...

    def allocate(
        self,
        layout: KVLayout,
        capacity: StateCapacity,
        image_rows: tuple[int, int] | None,
        dtype: torch.dtype,
    ) -> tuple[InferenceState, torch.Tensor | None]: ...


class ResidentBeside(Protocol):
    """What runs beside inference on the device, prepared once inference's ops are warm and
    before its state is sized: the bytes it keeps through inference."""

    @property
    def resident_bytes(self) -> int: ...


__all__ = ["DeviceBudget", "HostedCheckpoint", "HostedModels", "ResidentBeside"]
