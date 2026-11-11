# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/models/weight.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import glob
import json
import os
import struct
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm

from cti.config import Checkpoint, RandomWeights
from cti.utils import init_logger

from .mapping import EXPERT_NAME, CheckpointMapping, CheckpointPlacement

if TYPE_CHECKING:
    from cti.layers import Param

    from ._download import CheckpointSource
    from .base import BaseLLMModel

logger = init_logger(__name__)


def load_weight(
    model_folder: str,
    device: torch.device,
    *,
    mapping: CheckpointMapping,
    params: Sequence[Param],
    show_progress: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream final rank-local runtime tensors onto the execution device.

    ``mapping`` is the model's ``checkpoint_mapping`` and ``params`` its ``params()``: each
    runtime tensor takes its param's shape and dtype, and only float32, bfloat16 and float16
    convert. A param without windows loads whole; a stacked expert of another rank is never
    read. CPU parts wait until their runtime tensor can be filled directly, so each device
    allocation is final, never a temporary merge or cast input. Rank-local windows are mmap
    views held until their merge or expert group completes. ``progress`` hears each file
    of the checkpoint as its read begins. The rank logs its read seconds, bytes and rate,
    and each shard's at debug level.
    """
    start = time.perf_counter()
    targets = {param.path: param for param in params}
    # Parts of a merged tensor that a split projection holds in tensors of its own.
    held = {
        (param.merge[0], slot): (param.path, local, len(param.merge[1]))
        for param in params
        if param.merge is not None
        for local, slot in enumerate(param.merge[1])
    }
    declared = targets.keys() | {source for source, _ in held}
    files = sorted(_checkpoint_files(model_folder))
    shapes: dict[str, tuple[int, ...]] | None = None
    builder = _RuntimeTensorBuilder(targets, device)
    total = 0
    for index, file in enumerate(tqdm(files, desc="Loading weights", disable=not show_progress)):
        shard = f"shard {index + 1}/{len(files)} ({os.path.basename(file)})"
        if progress is not None:
            progress(f"loading weights: {shard}")
        shard_start, shard_bytes = time.perf_counter(), 0
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for checkpoint_name in f.keys():
                placements = [
                    _local_expert(targets, _held(held, placement))
                    for placement in mapping.placements(checkpoint_name, declared)
                ]
                if all(placement is None for placement in placements):
                    continue
                weight_rows = None
                if (weight := mapping.weight_of(checkpoint_name)) is not None:
                    # A companion cuts by its weight's rows, which only the headers record.
                    shapes = _shapes(files) if shapes is None else shapes
                    weight_rows = shapes[weight][0]
                parts = mapping.split(checkpoint_name, f.get_tensor(checkpoint_name), weight_rows)
                for placement, part in zip(placements, parts, strict=True):
                    if placement is None:
                        continue
                    window = _select_window(targets, placement, part)
                    shard_bytes += window.nbytes
                    if (tensor := builder.add(placement, window)) is not None:
                        yield placement.name, tensor
        total += shard_bytes
        logger.debug("%s on %s: %s", shard, device, _rate(shard_bytes, shard_start))
    builder.require_complete()
    builder.release()  # before the caller measures memory for its capacity plan
    logger.info("Loaded %d weight shards onto %s: %s", len(files), device, _rate(total, start))


def _rate(size: int, start: float) -> str:
    """``size`` bytes since ``start`` (``time.perf_counter``), in GB, seconds and GB/s."""

    seconds = time.perf_counter() - start
    return f"{size / 1e9:.2f} GB in {seconds:.2f} s ({size / 1e9 / max(seconds, 1e-9):.2f} GB/s)"


def _shapes(files: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Every checkpoint tensor's shape, read from the files' headers."""

    return {name: tuple(header["shape"]) for name, header in _headers(files).items()}


class CheckpointMethods:
    """The methods the checkpoint in ``model_folder`` stores its weights for, read from its
    documents and tensor headers (no tensor data) before the model is built, so a checkpoint
    this engine cannot serve is refused before any is; ``adopt`` gives them to the built
    model's ops."""

    def __init__(self, model_folder: str, device: torch.device) -> None:
        from cti.layers.quant import CheckpointFormats

        def document(name: str) -> dict[str, Any] | None:
            if not os.path.isfile(path := os.path.join(model_folder, name)):
                return None
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        headers = _headers(_checkpoint_files(model_folder))
        self._modules = tuple(
            module
            for module in dict.fromkeys(name.rpartition(".")[0] for name in headers)
            if f"{module}.weight" in headers
        )
        experts = {module for module in self._modules if EXPERT_NAME.match(f"{module}.weight")}
        self._formats = CheckpointFormats(document, headers, experts, device)

    @property
    def untrainable(self) -> str | None:
        """The recipe of a checkpoint that serves but does not train, by name; else None."""

        return self._formats.untrainable

    def adopt(self, model: BaseLLMModel) -> dict[str, int]:
        """Give each op of ``model`` (built on the meta device, no adapter attached) the
        method its checkpoint modules store their weights for; return an opaque summary of
        them for logs and receipts."""

        mapping, declared = model.checkpoint_mapping, {param.path for param in model.params()}
        ops: dict[str, dict[int, list[str]]] = {}
        for module in self._modules:
            for placement in mapping.placements(f"{module}.weight", declared):
                slots = ops.setdefault(placement.name.rpartition(".")[0], {})
                slots.setdefault(placement.slot, []).append(module)
        return self._formats.adopt(model, ops)


def _checkpoint_files(model_folder: str) -> list[str]:
    files = glob.glob(f"{model_folder}/*.safetensors")
    return [f for f in files if not f.endswith("consolidated.safetensors")] or files


def _headers(files: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Each checkpoint tensor's safetensors header (``dtype`` tag, ``shape``), no data read."""

    headers: dict[str, dict[str, Any]] = {}
    for file in files:
        with open(file, "rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(size))
        header.pop("__metadata__", None)
        if repeated := sorted(headers.keys() & header.keys()):
            raise ValueError(f"checkpoint tensors appear in more than one file: {repeated[:4]}")
        headers |= header
    return headers


def _held(
    held: Mapping[tuple[str, int], tuple[str, int, int]], placement: CheckpointPlacement
) -> CheckpointPlacement:
    """The placement in the tensor that holds a merged tensor's part, where one does."""

    if (where := held.get((placement.name, placement.slot))) is None:
        return placement
    name, slot, num_slots = where
    return replace(placement, name=name, slot=slot, num_slots=num_slots)


@dataclass(frozen=True)
class CheckpointWeights:
    """The weights in a checkpoint's local ``folder`` (``cti.config.Checkpoint``)."""

    folder: str

    def methods(self, device: torch.device) -> CheckpointMethods:
        """The methods the checkpoint stores its weights for, read before the model is
        built; their ``adopt`` gives them to the built model's ops."""

        return CheckpointMethods(self.folder, device)

    def names(self) -> tuple[str, ...]:
        """The checkpoint's tensor names, read from its files' headers."""

        return tuple(sorted(_headers(_checkpoint_files(self.folder))))

    def state_dict(
        self,
        model: BaseLLMModel,
        device: torch.device,
        *,
        show_progress: bool,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """``model``'s tensors from the folder; ``progress`` hears each file as its read
        begins."""

        return dict(
            load_weight(
                self.folder,
                device,
                mapping=model.checkpoint_mapping,
                params=model.params(),
                show_progress=show_progress,
                progress=progress,
            )
        )


@dataclass(frozen=True)
class SeededWeights:
    """Standard normal weights seeded by ``seed`` (``cti.config.RandomWeights``); they
    read no checkpoint."""

    seed: int
    folder: None = None
    # Random weights are of the model dtype, which trains.
    untrainable: None = None

    def methods(self, device: torch.device) -> SeededWeights:
        """Random weights are of the model dtype: their ``adopt`` changes no method."""

        return self

    def adopt(self, model: BaseLLMModel) -> dict[str, int]:
        return {}

    def names(self) -> None:
        """No checkpoint names the weights."""

        return None

    def state_dict(
        self,
        model: BaseLLMModel,
        device: torch.device,
        *,
        show_progress: bool,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device).manual_seed(self.seed)
        return {
            name: torch.randn(target.shape, dtype=target.dtype, device=device, generator=generator)
            for name, target in model.state_dict().items()
        }


Weights = CheckpointWeights | SeededWeights


def open_weights(
    source: CheckpointSource,
    weights: Checkpoint | RandomWeights,
    progress: Callable[[str], None],
) -> Weights:
    """The weights the option ``weights`` names for the checkpoint at ``source``: its
    folder's, downloaded at its revision if needed (``progress`` hearing each file's
    download), or seeded random ones."""

    match weights:
        case RandomWeights(seed=seed):
            return SeededWeights(seed)
        case Checkpoint():
            from ._download import fetch_weights

            fetch_weights(source, progress)
            return CheckpointWeights(source.folder)


def _local_expert(
    targets: Mapping[str, Param],
    placement: CheckpointPlacement,
) -> CheckpointPlacement | None:
    """Renumber a stacked expert within this rank's experts; ``None`` if another rank's."""
    experts = targets[placement.name].experts if placement.name in targets else None
    if placement.expert is None or experts is None:
        return placement
    if placement.expert not in experts:
        return None
    return replace(placement, expert=placement.expert - experts.start)


def _select_window(
    targets: Mapping[str, Param],
    placement: CheckpointPlacement,
    part: torch.Tensor,
) -> torch.Tensor:
    """Return this rank's window of one runtime part. A sharded expert stack takes its experts
    of a stacked checkpoint tensor (a per-expert checkpoint's expert arrives alone), then each
    expert's windows, one axis further into a stacked tensor; a part that holds every
    window's part, such as gate and up rows in one tensor, splits into equal parts first."""
    target = targets[placement.name]
    windows = target.windows
    if target.experts is None:
        return windows[placement.slot].select(part) if windows else part
    stacked = placement.expert is None
    if stacked:
        part = part.narrow(0, target.experts.start, len(target.experts))
    if not windows:
        return part
    if placement.num_slots == len(windows):
        windows = (windows[placement.slot],)
    windows = tuple(replace(window, axis=window.axis + stacked) for window in windows)
    pieces = part.chunk(len(windows), dim=windows[0].axis)
    cut = [window.select(piece) for window, piece in zip(windows, pieces, strict=True)]
    return cut[0] if len(cut) == 1 else torch.cat(cut, windows[0].axis)


class _RuntimeTensorBuilder:
    """Hold rank-local CPU parts until a runtime tensor is complete, then fill it once."""

    # The checkpoint dtypes the loader converts between; any other dtype loads only as itself.
    CASTABLE = frozenset({torch.float32, torch.bfloat16, torch.float16})

    def __init__(self, targets: Mapping[str, Param], device: torch.device) -> None:
        self._targets = targets
        self._device = device
        self._merges: dict[tuple[str, int | None], dict[int, torch.Tensor]] = {}
        self._experts: dict[str, dict[int, tuple[torch.Tensor, ...]]] = {}
        self._staging = _HostStaging()

    def release(self) -> None:
        self._staging.release()

    def add(self, placement: CheckpointPlacement, part: torch.Tensor) -> torch.Tensor | None:
        """Return the final device tensor once ``part`` completes it."""

        name = placement.name
        group: tuple[torch.Tensor, ...] = (part,)
        if placement.num_slots > 1:
            key = (name, placement.expert)
            slots = self._merges.setdefault(key, {})
            slots[placement.slot] = part
            if len(slots) < placement.num_slots:
                return None
            del self._merges[key]
            group = tuple(slots[slot] for slot in range(placement.num_slots))

        target = self._targets[name]
        if placement.expert is None:
            return _materialize_weight(
                name, (group,), device=self._device, target=target, staging=self._staging
            )
        num_experts = target.shape[0]
        if placement.expert >= num_experts:
            raise ValueError(f"{name}: expert {placement.expert} is outside {num_experts} experts")
        experts = self._experts.setdefault(name, {})
        experts[placement.expert] = group
        if len(experts) < num_experts:
            return None
        del self._experts[name]
        groups = [experts[expert] for expert in range(num_experts)]
        return _materialize_weight(
            name,
            groups,
            device=self._device,
            target=target,
            staging=self._staging,
            stacked=True,
        )

    def require_complete(self) -> None:
        incomplete = sorted({name for name, _ in self._merges} | set(self._experts))
        if incomplete:
            raise ValueError(f"Incomplete merge or expert groups in checkpoint: {incomplete}")


class _HostStaging:
    """One reusable host buffer that parts pass through on their way to the device: a copy
    straight from a safetensors mmap view costs about 18 ms per tensor on a GB10 (unified
    memory), whatever the page cache holds, against 0.13 ms from ordinary memory. Sized to
    the largest part so far; ``release`` frees it."""

    def __init__(self) -> None:
        self._buffer: torch.Tensor | None = None

    def stage(self, part: torch.Tensor) -> torch.Tensor:
        """``part``'s values in the buffer, valid until the next ``stage``."""

        if self._buffer is None or self._buffer.numel() < part.nbytes:
            self._buffer = None  # the smaller buffer goes before the larger one is made
            self._buffer = torch.empty(part.nbytes, dtype=torch.uint8)
        staged = self._buffer[: part.nbytes].view(part.dtype).view(part.shape)
        return staged.copy_(part)

    def release(self) -> None:
        self._buffer = None


def _materialize_weight(
    name: str,
    groups: Sequence[Sequence[torch.Tensor]],
    *,
    device: torch.device,
    target: Param,
    staging: _HostStaging,
    stacked: bool = False,
) -> torch.Tensor:
    """Copy completed CPU part groups into one device tensor shaped and typed like ``target``;
    a scalar part fills one row (per-slot scales stack)."""
    if target.shape and not groups[0][0].ndim:
        groups = [[part.reshape(1) for part in group] for group in groups]
    first = groups[0][0]
    rows = sum(part.shape[0] for part in groups[0])
    for index, group in enumerate(groups):
        if (group_rows := sum(part.shape[0] for part in group)) != rows:
            raise ValueError(f"{name}: expert {index} has {group_rows} rows, expected {rows}")
        for part in group:
            if part.dtype != first.dtype or part.shape[1:] != first.shape[1:]:
                raise ValueError(
                    f"{name}: part {part.dtype} {tuple(part.shape)} does not match "
                    f"{first.dtype} {tuple(first.shape)}"
                )
    shape = (rows, *first.shape[1:])
    if stacked:
        shape = (len(groups), *shape)
    if shape != tuple(target.shape):
        raise ValueError(
            f"{name}: checkpoint gives shape {shape}, the model declares {tuple(target.shape)}"
        )
    if (
        first.dtype != target.dtype
        and not {first.dtype, target.dtype} <= _RuntimeTensorBuilder.CASTABLE
    ):
        raise ValueError(f"{name}: checkpoint dtype {first.dtype} cannot load as {target.dtype}")
    output = torch.empty(shape, dtype=target.dtype, device=device)
    for index, group in enumerate(groups):
        destination = output[index] if stacked else output
        offset = 0
        for part in group:
            # A blocking host-to-device copy_ converts dtype and layout on the CPU; it returns
            # once the staged part is read, so the buffer is free for the next.
            destination.narrow(0, offset, part.shape[0]).copy_(staging.stage(part))
            offset += part.shape[0]
    return output
