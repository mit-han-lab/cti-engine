"""Breakable graphs: a forward's hidden states as captured segments between eager breaks.

Capture ends a segment at each ``batch.graph_breaks.eager`` (``cti.forward``) and
starts the next; replay runs each segment, then the break on the batch's real rows into a static
bridge the next segment reads. Segments hold only row-wise work over a whole row bucket and
reference no KV or recurrent state, so padding rows never reach the caches.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from cti.forward import BreakCall, ForwardInputs, GraphBreaks

from ._graphs import RowBuckets, capture_graphs

if TYPE_CHECKING:
    from cti.inference._step import GraphCapture
    from cti.memory import GraphPool
    from cti.models import BaseLLMModel


@dataclass(frozen=True)
class _Step:
    """A captured segment, then the break that follows it (none after the last): ``x`` is a
    weak alias of its input, read right after the segment that wrote it."""

    segment: torch.cuda.CUDAGraph
    call: BreakCall | None = None
    x: torch.Tensor | None = None
    bridge: torch.Tensor | None = None


class _CaptureBreaks(GraphBreaks):
    """Skip every break: on a warm-up forward, returning its bridge; while capturing into
    ``pool``, ending the open segment, recording the break and opening the next. A break
    never keeps the capture batch: replay passes its own."""

    def __init__(self, graph: BreakableGraph, pool: GraphPool | None) -> None:
        self._graph = graph
        self._pool = pool
        self._steps: list[_Step] = []
        self._open: torch.cuda.CUDAGraph | None = None
        if pool is not None:
            self._begin()

    def eager(
        self, call: BreakCall, x: torch.Tensor, batch: ForwardInputs, *, width: int
    ) -> torch.Tensor:
        bridge = self._graph.bridge(width, x.dtype)[: x.shape[0]]
        if self._pool is not None:
            self._steps.append(_Step(self._end(), call, self._graph.weak_alias(x), bridge))
            self._begin()
        return bridge

    def finish(self) -> tuple[_Step, ...]:
        return (*self._steps, _Step(self._end()))

    def abort(self) -> None:
        if self._open is not None:
            self._end()

    def _begin(self) -> None:
        self._open = torch.cuda.CUDAGraph()
        self._pool.begin(self._open)

    def _end(self) -> torch.cuda.CUDAGraph:
        segment, self._open = self._open, None
        self._pool.end(segment)
        return segment


class BreakableGraph:
    """Hidden states of a forward replayed from per-bucket segments around eager breaks;
    replay returns the real rows."""

    def __init__(
        self,
        *,
        model: BaseLLMModel,
        buckets: RowBuckets,
        pool: GraphPool,
        stream: torch.cuda.Stream,
        device: torch.device,
    ) -> None:
        from ctikernels import weak_tensor_alias

        self.buckets = buckets
        self.weak_alias = weak_tensor_alias()
        self._model = model
        self._device = device
        self._closed = False
        self._input_ids = torch.zeros(buckets.sizes[-1], dtype=torch.int32, device=device)
        # Static outside the pool: per break width and dtype, and the hidden output.
        self._bridges: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        self._output: torch.Tensor | None = None
        self._graphs: dict[int, tuple[_Step, ...]] = {}

        def capture() -> None:
            # Largest first: its warm-up sizes every static buffer.
            for bucket in reversed(buckets.sizes):
                self._graphs[bucket] = self._capture(bucket, pool, stream)

        try:
            self.captured: GraphCapture = capture_graphs(
                "breakable",
                buckets.sizes,
                capture,
                pool=pool,
                device=device,
                max_rows=buckets.sizes[-1],
            )
        except BaseException:
            self.close()
            raise

    def bridge(self, width: int, dtype: torch.dtype) -> torch.Tensor:
        key = (width, dtype)
        if key not in self._bridges:
            rows = self.buckets.sizes[-1]
            self._bridges[key] = torch.zeros(rows, width, dtype=dtype, device=self._device)
        return self._bridges[key]

    def can_replay(self, batch: ForwardInputs) -> bool:
        """A ragged batch without images whose rows fit a bucket: prompt steps and batches
        of prompts with decodes. Uniform decodes stay with the decode graphs or eager."""

        return (
            not self._closed
            and batch.ragged
            and batch.images is None
            and self.buckets.select(batch.input_ids.numel()) is not None
        )

    def replay(self, batch: ForwardInputs) -> torch.Tensor:
        rows = batch.input_ids.numel()
        bucket = self.buckets.select(rows)
        self._input_ids[:rows].copy_(batch.input_ids)
        for bridge in self._bridges.values():
            bridge[rows:bucket].zero_()
        for step in self._graphs[bucket]:
            step.segment.replay()
            if step.call is not None:
                step.call(step.x[:rows], batch, step.bridge[:rows])
        return self._output[:rows]

    def close(self) -> None:
        self._closed = True
        self._graphs.clear()
        gc.collect()

    def _capture(
        self, bucket: int, pool: GraphPool, stream: torch.cuda.Stream
    ) -> tuple[_Step, ...]:
        with torch.inference_mode(), torch.cuda.stream(stream):
            hidden = self._model.forward_hidden(self._batch(bucket, _CaptureBreaks(self, None)))
            if self._output is None:
                self._output = hidden.new_zeros(bucket, hidden.shape[1])
            torch.cuda.synchronize(self._device)
            with pool.region():
                breaks = _CaptureBreaks(self, pool)
                try:
                    hidden = self._model.forward_hidden(self._batch(bucket, breaks))
                    self._output[:bucket].copy_(hidden)
                    return breaks.finish()
                except BaseException:
                    breaks.abort()
                    raise

    def _batch(self, rows: int, breaks: GraphBreaks) -> ForwardInputs:
        """A prompt of ``rows`` tokens whose forward reads only its static token IDs: every
        break, the only reader of positions and metadata, is skipped, so it has none."""

        return ForwardInputs(
            self._input_ids[:rows], None, None, 1, ragged=True, graph_breaks=breaks
        )


__all__ = ["BreakableGraph"]
