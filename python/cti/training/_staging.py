from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from cti.config import BatchLayout, SFTShapeProfile
from cti.models.training_contract import Packing

from ._batch import DeviceRows

if TYPE_CHECKING:
    from ._batch import TrainingBatch


def staging_device_bytes(profile: SFTShapeProfile, *, packed: bool) -> tuple[int, ...]:
    """Bytes of each device buffer the runner stages ``profile`` in, in allocation order."""

    buffers = _device_buffers(profile, packed, torch.device("meta"))
    return tuple(buffer.nbytes for buffer in buffers if buffer is not None)


def _device_buffers(
    profile: SFTShapeProfile, packed: bool, device: torch.device
) -> tuple[torch.Tensor | None, ...]:
    """The device buffers staging ``profile`` takes, in allocation order: inputs, targets,
    target positions (None when every input is a target) and, for packed rows, the
    packing's positions and sequence boundaries."""

    def int64(*shape: int) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.int64, device=device)

    inputs = (
        (profile.total_tokens - profile.batch_size,)
        if packed
        else (profile.batch_size, cast(int, profile.padded_width) - 1)
    )
    targets = cast(int, profile.num_target_tokens)
    every_input = targets == math.prod(inputs)
    packing = (
        (int64(*inputs), torch.empty(profile.batch_size + 1, dtype=torch.int32, device=device))
        if packed
        else ()
    )
    return (int64(*inputs), int64(targets), None if every_input else int64(targets), *packing)


class SFTStaging:
    """One exact SFT shape profile staged on the device: each batch's inputs, targets and,
    unless the profile's targets are fixed, target positions go through two pinned host
    slots in turn into buffers allocated once; a slot is reused once its last copy has
    completed. The packing is fixed."""

    def __init__(self, profile: SFTShapeProfile, layout: BatchLayout, device: torch.device) -> None:
        width = cast(int, profile.padded_width)
        packed = layout is BatchLayout.PACKED
        rows: list[int] = []
        positions: list[int] = []
        boundaries = [0]
        start = 0
        for length in profile.sequence_lengths:
            # Token offsets of each sequence's input positions, in the flat token tensor.
            rows.extend(range(start, start + length - 1))
            positions.extend(range(length - 1))
            boundaries.append(boundaries[-1] + length - 1)
            start += length if packed else width
        self._profile = profile
        self._layout = layout
        inputs, targets, indices, *packing = _device_buffers(profile, packed, device)
        input_index = (
            torch.tensor(rows, dtype=torch.int64)
            if packed
            else torch.arange(profile.batch_size * width).view(-1, width)[:, :-1].reshape(-1)
        )
        self._input_index, self._target_index = input_index, input_index + 1
        sparse = not profile.has_dense_targets
        self._fixed_cpu = None
        self._fixed = None if sparse else indices
        if self._fixed is not None:
            # Dense targets of padded rows skip the padding at fixed positions.
            self._fixed_cpu = torch.tensor(
                [
                    row * (width - 1) + p
                    for row, n in enumerate(profile.sequence_lengths)
                    for p in range(n - 1)
                ],
                dtype=torch.int64,
            )
            self._fixed.copy_(self._fixed_cpu)
        self._packing = None
        if packed:
            packing[0].copy_(torch.tensor(positions, dtype=torch.int64))
            packing[1].copy_(torch.tensor(boundaries, dtype=torch.int32))
            self._packing = Packing(*packing, max(profile.sequence_lengths) - 1)
        self._buffers = (inputs, targets, indices if sparse else None)
        self._pinned = device.type == "cuda"
        # Each pinned slot's host buffers, and the event its last copy recorded.
        self._slots = [
            (
                tuple(
                    None
                    if buffer is None
                    else torch.empty(buffer.shape, dtype=buffer.dtype, pin_memory=True)
                    for buffer in self._buffers
                ),
                torch.cuda.Event(),
            )
            for _ in range(2 if self._pinned else 0)
        ]
        self._next = 0

    def device_tensors(self) -> tuple[torch.Tensor, ...]:
        """The device buffers this staging keeps."""

        packing = (
            () if self._packing is None else (self._packing.positions, self._packing.cu_seqlens)
        )
        held = (self._fixed, *self._buffers, *packing)
        return tuple(
            tensor for tensor in held if tensor is not None and tensor.device.type != "cpu"
        )

    def require_profile(self, batch: TrainingBatch) -> None:
        tokens = batch.tokens
        actual = (tokens.LAYOUT, tuple(tokens.sequence_lengths()), batch.num_targets)
        expected = (self._layout, self._profile.sequence_lengths, self._profile.num_target_tokens)
        width = getattr(tokens, "width", None)
        if actual != expected or (width is not None and width + 1 != self._profile.padded_width):
            raise ValueError(
                f"SFT profile mismatch: expected {self._layout.value} sequence lengths "
                f"{self._profile.sequence_lengths}, padded width {self._profile.padded_width}, "
                f"{self._profile.num_target_tokens} target tokens; got {tokens.LAYOUT.value} "
                f"lengths {actual[1]}, width {None if width is None else width + 1}, "
                f"{batch.num_targets} target tokens"
            )

    def stage(self, batch: TrainingBatch) -> DeviceRows:
        """``batch``, which ``require_profile`` admitted, in the device buffers."""

        inputs, targets, staged = self._buffers
        flat = batch.tokens.token_ids.reshape(-1)
        input_ids = flat.index_select(0, self._input_index).view(inputs.shape)
        target_ids = flat.index_select(0, self._target_index)
        indices = (
            batch.targets.reshape(-1).nonzero().flatten() if staged is not None else self._fixed_cpu
        )
        if indices is not None:
            target_ids = target_ids.index_select(0, indices)
        sources = (input_ids, target_ids, indices)
        if not self._pinned:
            for buffer, source in zip(self._buffers, sources, strict=True):
                if buffer is not None:
                    buffer.copy_(source)
        else:
            host, copied = self._slots[self._next]
            self._next ^= 1
            copied.synchronize()
            for slot, buffer, source in zip(host, self._buffers, sources, strict=True):
                if buffer is not None:
                    slot.copy_(source)
                    buffer.copy_(slot, non_blocking=True)
            copied.record()
        return DeviceRows(inputs, self._packing, self._fixed if staged is None else staged, targets)


__all__ = ["SFTStaging", "staging_device_bytes"]
