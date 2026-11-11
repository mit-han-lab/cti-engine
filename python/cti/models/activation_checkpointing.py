# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class CheckpointInputOffload(Protocol):
    """Holds the checkpointed layers' saved inputs between forward and backward.

    Enter ``saving()`` around forward and ``expect(input)`` before each checkpoint.
    """

    def saving(self) -> AbstractContextManager[None]: ...

    def expect(self, tensor: torch.Tensor) -> None: ...


# One step of a decoder layer's training: the residual stream in, the stream after it out.
Segment = Callable[[torch.Tensor], torch.Tensor]

# Rows per tile of a row-wise segment's offloaded checkpoint: its backward holds one
# tile's activations, and a tile still runs its projections as full-speed GEMMs.
ROW_TILE = 4096


@dataclass(frozen=True, slots=True)
class RowWise:
    """A segment each output row of which depends only on the same input row, such as an
    MLP. Under an offload it trains in tiles of ``ROW_TILE`` rows, so its backward holds
    one tile's activations."""

    segment: Segment

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.segment(x)


class ReentrantActivationCheckpoint:
    """Recompute each decoder layer in backward, from the first layer that trains.

    A layer trains as its segments in order, such as its token mixer and then its MLP.
    With an ``offload``, each segment is a checkpoint whose input the offload owns until
    backward, so a segment's backward holds no other segment's saves, and a ``RowWise``
    one recomputes tile by tile; without one, each layer is a checkpoint, since every
    checkpoint input then stays on the device.
    The decoder input must not require grad. Layers before ``first`` must be frozen;
    they run without autograd. Later checkpoints are recomputed once in backward and must
    not modify their inputs in place; such writes raise ``RuntimeError``.
    Grad-disabled forwards run every segment plainly.
    """

    __slots__ = ("_first", "_offload")

    def __init__(self, first: int, offload: CheckpointInputOffload | None = None) -> None:
        if type(first) is not int or first < 0:
            raise ValueError("first must be a non-negative decoder layer index")
        self._first = first
        self._offload = offload

    @property
    def first(self) -> int:
        """The first checkpointed layer; the layers before it run without autograd."""

        return self._first

    def run_blocks(self, layers: Sequence[Sequence[Segment]], x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` after every segment of every layer, in order."""

        if not torch.is_grad_enabled():
            return run_segments(layers, x)
        if x.requires_grad:
            # The layers before first would drop this gradient.
            raise RuntimeError("activation checkpointing requires a decoder input without grad")
        with torch.no_grad():
            x = run_segments(layers[: self._first], x)
        offload = self._offload
        trained = layers[self._first :]
        units = (
            [partial(run_segments, (segments,)) for segments in trained]
            if offload is None
            else [segment for segments in trained for segment in segments]
        )
        # Reentrant checkpoints need a grad-bearing input even when x is frozen.
        anchor = torch.empty(0, device=x.device, requires_grad=True)
        with nullcontext() if offload is None else offload.saving():
            for unit in units:
                if offload is not None:
                    offload.expect(x)
                # Checkpoints save after the unit, too late to detect its writes.
                version = x._version
                output = (
                    _RowTiledCheckpoint.apply(unit.segment, x, anchor)
                    if isinstance(unit, RowWise)
                    else torch_checkpoint(
                        partial(_run_unit, unit),
                        x,
                        anchor,
                        use_reentrant=True,
                        preserve_rng_state=False,
                    )
                )
                if x._version != version:
                    raise RuntimeError("a checkpointed decoder layer wrote its input in place")
                x = output
        return x


def run_segments(layers: Sequence[Sequence[Segment]], x: torch.Tensor) -> torch.Tensor:
    """Return ``x`` after every segment of every layer, in order, without checkpoints."""

    for segments in layers:
        for segment in segments:
            x = segment(x)
    return x


def _run_unit(unit: Segment, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    output = unit(x)
    if torch.is_grad_enabled() and not output.requires_grad:
        # A frozen unit before the first trained one, such as an unadapted mixer: its
        # recomputation must still return an output with grad.
        output = _Anchored.apply(output, anchor)
    return output


class _RowTiledCheckpoint(torch.autograd.Function):
    """A row-wise segment's checkpoint: the forward runs it without autograd tile by tile
    and saves only its input; backward recomputes and differentiates one tile at a time,
    accumulating parameter gradients as a reentrant checkpoint's backward does."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        segment: Segment,
        x: torch.Tensor,
        _anchor: torch.Tensor,
    ) -> torch.Tensor:
        ctx.segment = segment
        ctx.save_for_backward(x)
        rows = x.view(-1, x.shape[-1])
        output = torch.empty_like(rows)
        for tile in _tiles(rows.shape[0]):
            output[tile] = segment(rows[tile])
        return output.view_as(x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx, gradient: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        rows = x.view(-1, x.shape[-1])
        gradients = gradient.reshape(rows.shape)
        grad_x = torch.empty_like(rows)
        for tile in _tiles(rows.shape[0]):
            tile_x = rows[tile].detach().requires_grad_(True)
            with torch.enable_grad():
                tile_output = ctx.segment(tile_x)
            torch.autograd.backward(tile_output, gradients[tile])
            grad_x[tile] = tile_x.grad
        return None, grad_x.view_as(x), None


def _tiles(rows: int) -> list[slice]:
    return [slice(start, min(start + ROW_TILE, rows)) for start in range(0, rows, ROW_TILE)]


class _Anchored(torch.autograd.Function):
    """``output`` itself, attached to the anchor; no gradient flows through it."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, output: torch.Tensor, _anchor: torch.Tensor
    ) -> torch.Tensor:
        return output.view_as(output)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, _gradient: torch.Tensor
    ) -> tuple[None, None]:
        return None, None


__all__ = [
    "ROW_TILE",
    "CheckpointInputOffload",
    "ReentrantActivationCheckpoint",
    "RowWise",
    "Segment",
    "run_segments",
]
