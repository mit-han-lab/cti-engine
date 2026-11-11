"""Recompute audit of activation checkpointing, a test fixture: production computes no digest.

The models look their decoder layers' training segments up on the decoder-layer class at
every forward, so ``record_block_runs`` can wrap them and digest each call's output by
layer, segment and grad mode. A checkpointed segment runs once without grad, inside its
reentrant checkpoint, and again with grad when backward recomputes it. ``assert_recomputed``
checks that the checkpointed forward repeats an eager one bit for bit, that backward
recomputes every segment from layer ``first`` on exactly once, to the same output, last
checkpoint first: a whole layer, its segments in order, or with ``by_segment`` one segment
at a time; and that it never runs the frozen layers before ``first`` again.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import NamedTuple
from unittest import mock

import torch

# Every decoder layer's training segments, in forward order.
SEGMENTS = ("mixer_training", "mlp_training")


class BlockRun(NamedTuple):
    layer: int
    segment: int
    grad_enabled: bool
    digest: str


@contextmanager
def record_block_runs(layer_class: type, layers: Sequence[object]) -> Iterator[list[BlockRun]]:
    """Record every segment call on one of ``layers``, in call order."""

    indices = {id(layer): index for index, layer in enumerate(layers)}
    runs: list[BlockRun] = []

    def recording(segment: int) -> object:
        run = getattr(layer_class, SEGMENTS[segment])

        def recorded(layer: object, *args: object, **kwargs: object) -> torch.Tensor:
            output = run(layer, *args, **kwargs)
            runs.append(
                BlockRun(indices[id(layer)], segment, torch.is_grad_enabled(), _digest(output))
            )
            return output

        return recorded

    with ExitStack() as stack:
        for segment, name in enumerate(SEGMENTS):
            stack.enter_context(mock.patch.object(layer_class, name, recording(segment)))
        yield runs


def assert_recomputed(
    eager: Sequence[BlockRun],
    checkpointed: Sequence[BlockRun],
    *,
    first: int,
    by_segment: bool = False,
) -> None:
    """``eager`` ran every segment once with grad; ``checkpointed`` recomputed from layer
    ``first``, a layer at a time or with ``by_segment`` a segment at a time."""

    count = len(eager)
    num_layers = count // len(SEGMENTS)
    assert [run[:3] for run in eager] == [
        (layer, segment, True) for layer in range(num_layers) for segment in range(len(SEGMENTS))
    ]
    forward, recomputed = checkpointed[:count], checkpointed[count:]
    assert list(forward) == [run._replace(grad_enabled=False) for run in eager]
    units = [
        forward[layer * len(SEGMENTS) : (layer + 1) * len(SEGMENTS)]
        for layer in range(first, num_layers)
    ]
    if by_segment:
        units = [(run,) for unit in units for run in unit]
    assert list(recomputed) == [
        run._replace(grad_enabled=True) for unit in reversed(units) for run in unit
    ]


def _digest(output: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


__all__ = ["SEGMENTS", "BlockRun", "assert_recomputed", "record_block_runs"]
