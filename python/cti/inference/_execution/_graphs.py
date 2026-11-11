# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/engine/graph.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import gc
import time
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, ClassVar, List, Literal, Protocol

import torch

from cti.forward import EagerImages, ForwardInputs, ForwardRows, RecurrentExport
from cti.inference._step import GraphCapture, GraphStats, QueryKind
from cti.utils import init_logger

if TYPE_CHECKING:
    from cti.forward import ImageEncoding
    from cti.inference._step import RequestSpec, Row
    from cti.memory import GraphPool
    from cti.models import BaseLLMModel

    from ._breakable_graph import BreakableGraph
    from ._vision_graph import VisionGraph

logger = init_logger(__name__)


def forward_rows(
    entries: Sequence[Row], specs: Sequence[RequestSpec] = (), padding: Sequence[Row] = ()
) -> ForwardRows:
    """What the model side reads of ``entries``, given their request constants ``specs``
    (none for a capture, whose rows take no rope offset), then of the ``padding`` rows a
    captured graph size adds. Extensions lead the batch."""

    if not entries:
        raise ValueError("inference batch must contain at least one entry")
    kinds = [entry.query_kind for entry in entries]
    num_extend = kinds.count(QueryKind.EXTEND)
    if QueryKind.DECODE in kinds[:num_extend]:
        raise ValueError("extend entries must precede decode entries")
    offsets = [0] * len(entries)
    if specs:
        offsets = [
            spec.rope_offset if entry.query_kind is QueryKind.DECODE else 0
            for entry, spec in zip(entries, specs, strict=True)
        ]
    padded = (*entries, *padding)
    return ForwardRows(
        size=len(entries),
        num_extend=num_extend,
        slots=tuple(entry.request_index for entry in padded),
        kv_prefix_lens=tuple(entry.kv_prefix_len for entry in padded),
        sequence_lens=tuple(entry.sequence_len for entry in padded),
        rope_offsets=(*offsets, *(0,) * len(padding)),
        restores=tuple(
            (entry.restore_row, entry.request_index)
            for entry in entries[:num_extend]
            if entry.restore_row is not None
        ),
        captures=tuple(
            tuple(RecurrentExport(capture.position, capture.row) for capture in entry.captures)
            for entry in entries
        ),
        logit_rows=tuple(end - 1 for end in accumulate(entry.extend_len for entry in entries)),
    )


@dataclass(frozen=True)
class RowBuckets:
    """The ascending sizes a kind of graph is captured at: rows of a breakable graph,
    requests of a decode graph or a DFlash round. A batch replays the smallest size that
    holds it, padded with dummy rows (none on an exact fit); only a batch past the largest
    runs eagerly. The dummy rows of decode graphs and rounds follow the request slots: decode
    graphs repeat the first, a round takes distinct ones."""

    # Breakable-graph rows as (first, last, step) runs up to 512. Fine steps keep the padding
    # small, so a padded prompt rarely crosses a cuBLAS algorithm boundary that changes its
    # rows' bits (Qwen3-Coder-30B qkv on RTX PRO 6000: 11 of 29 prompts at step 64, none here).
    SPACING: ClassVar[tuple[tuple[int, int, int], ...]] = (
        (4, 32, 4),
        (48, 256, 16),
        (288, 512, 32),
    )

    sizes: tuple[int, ...]

    @classmethod
    def for_limits(cls, max_forward_len: int, override: tuple[int, ...] | None) -> RowBuckets:
        """Breakable graph rows: ``override`` when given, else every ``SPACING`` size that a
        forward of at most ``max_forward_len`` rows can fill."""

        if override is not None:
            return cls(override)
        ladder = (
            size for first, last, step in cls.SPACING for size in range(first, last + 1, step)
        )
        return cls(tuple(rows for rows in ladder if rows <= max_forward_len))

    def select(self, rows: int) -> int | None:
        """The smallest bucket that holds ``rows``, or None."""

        index = bisect_left(self.sizes, rows)
        return self.sizes[index] if rows > 0 and index < len(self.sizes) else None

    def padding_rows(self, max_round: int) -> int:
        """The dummy rows past the request slots when rounds of at most ``max_round``
        requests (0 without rounds) replay these sizes: at most ``padding_rows_bound``."""

        sizes = (0, *(size for size in self.sizes if size <= max_round))
        return max(1, max((high - low - 1 for low, high in pairwise(sizes)), default=0))

    @staticmethod
    def padding_rows_bound(max_round: int) -> int:
        """``padding_rows`` of any sizes, for what is allocated before they are known."""

        return max(1, max_round - 1)


class GraphParticipant(Protocol):
    """Each mixer backend's extension point into forwards and decode CUDA graphs: per forward
    it prepares its metadata as the input it fills, at capture it fills its metadata into the
    inputs, at replay it copies the step's metadata into its static buffers."""

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    def prepare_inputs(self, rows: ForwardRows) -> dict[str, object]: ...

    def prepare_for_capture(self, rows: ForwardRows, inputs: ForwardInputs) -> ForwardInputs: ...

    def prepare_for_replay(self, inputs: ForwardInputs, bs: int) -> None: ...


@dataclass
class _DecodeGraphBuffer:
    """Static decode-graph inputs; ``rope_positions`` is int64 ``[bs]``, one row that every
    rotary axis reads, and ``logit_rows`` every row, fixed per size. ``positions`` are not
    graph inputs."""

    input_ids: torch.Tensor
    rope_positions: torch.Tensor
    logit_rows: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(
        cls,
        bs: int,
        vocab_size: int,
        device: torch.device,
        logits_dtype: torch.dtype,
    ) -> _DecodeGraphBuffer:
        return cls(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            rope_positions=torch.zeros(bs, dtype=torch.int64, device=device),
            logit_rows=torch.arange(bs, dtype=torch.int64, device=device),
            logits=torch.empty(bs, vocab_size, dtype=logits_dtype, device=device),
        )

    def copy_from(self, inputs: ForwardInputs, bs: int) -> None:
        self.input_ids[:bs] = inputs.input_ids
        self.rope_positions[:bs] = inputs.rope_positions


def resolve_decode_graph_batch_sizes(
    cuda_graph_bs: Sequence[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
    max_running_requests: int,
) -> List[int]:
    """The decode batch sizes to capture, which DFlash rounds share: ``cuda_graph_bs`` as
    given, else every size to 4 and multiples of 8 up to ``cuda_graph_max_bs`` (by default
    sized from ``free_memory``) and the request capacity."""

    if cuda_graph_bs is not None:
        graph_batch_sizes = sorted(set(cuda_graph_bs))
        unreachable = [size for size in graph_batch_sizes if size > max_running_requests]
        if unreachable:
            raise ValueError(
                "decode CUDA graph batch sizes must not exceed the running-request "
                f"capacity {max_running_requests}; got {unreachable}"
            )
        return graph_batch_sizes

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    graph_ceiling = min(cuda_graph_max_bs, max_running_requests)
    graph_batch_sizes = set(range(1, min(4, graph_ceiling) + 1))
    graph_batch_sizes.update(range(8, graph_ceiling + 1, 8))
    graph_batch_sizes.add(graph_ceiling)
    return sorted(graph_batch_sizes)


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


def capture_graphs(
    kind: Literal["decode", "breakable", "round", "vision"],
    sizes: Sequence[int],
    capture: Callable[[], None],
    *,
    pool: GraphPool,
    device: torch.device,
    max_rows: int,
) -> GraphCapture:
    """Run ``capture``, the warm-up and capture of one group of graphs of at most
    ``max_rows`` rows each, in the group's pool workspace, and report the group. Runs before
    serving: it synchronizes the device around the group to measure it."""

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, start = get_free_memory(device), time.perf_counter()
    with pool.workspace(max_rows):
        capture()
    torch.cuda.synchronize(device)
    captured = GraphCapture(
        kind, tuple(sizes), time.perf_counter() - start, free - get_free_memory(device)
    )
    logger.info_rank0("Captured %s graphs of sizes %s: %s", kind, list(sizes), captured)
    return captured


class DecodeGraph:
    """Whole-forward graphs of uniform decode batches, one per captured batch size; replay
    returns the logits. Batches are padded to the next captured size with dummy rows."""

    def __init__(
        self,
        *,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        participants: tuple[GraphParticipant, ...],
        batch_sizes: Sequence[int],
        max_seq_len: int,
        vocab_size: int,
        logits_dtype: torch.dtype,
        dummy_entry: Row,
        pool: GraphPool,
    ) -> None:
        self._participants = participants
        self.batch_sizes = tuple(sorted(batch_sizes))
        self._buckets = RowBuckets(self.batch_sizes)
        self._dummy_entry = dummy_entry
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.captured: GraphCapture | None = None
        if not self.batch_sizes:
            logger.info_rank0("CUDA graph is disabled.")
            return
        for participant in participants:
            participant.init_capture_graph(max_seq_len, list(self.batch_sizes))

        def capture() -> None:
            self._buffer = _DecodeGraphBuffer.init(
                self.batch_sizes[-1], vocab_size, device, logits_dtype
            )
            for bs in reversed(self.batch_sizes):
                rows = forward_rows((dummy_entry,) * bs)
                buffer = self._buffer
                batch = ForwardInputs(
                    buffer.input_ids[:bs],
                    buffer.rope_positions[:bs],
                    None,
                    bs,
                    False,
                    logit_rows=buffer.logit_rows[:bs],
                )
                for participant in participants:
                    batch = participant.prepare_for_capture(rows, batch)
                self._graphs[bs] = graph = torch.cuda.CUDAGraph()
                with torch.inference_mode():
                    self._buffer.logits[:bs] = model.forward(batch)
                    with pool.capture(graph, stream):
                        self._buffer.logits[:bs] = model.forward(batch)

        self.captured = capture_graphs(
            "decode",
            self.batch_sizes,
            capture,
            pool=pool,
            device=device,
            max_rows=self.batch_sizes[-1],
        )

    def can_replay(self, rows: ForwardRows) -> bool:
        return rows.is_uniform_decode and self._buckets.select(rows.size) is not None

    def rows(self, entries: tuple[Row, ...], specs: tuple[RequestSpec, ...]) -> ForwardRows:
        """``entries``' rows padded up to the smallest captured size they fit, or not at all."""

        bucket = self._buckets.select(len(entries))
        if bucket is None or any(entry.query_kind is QueryKind.EXTEND for entry in entries):
            return forward_rows(entries, specs)
        return forward_rows(entries, specs, (self._dummy_entry,) * (bucket - len(entries)))

    def replay(self, rows: ForwardRows, inputs: ForwardInputs) -> torch.Tensor:
        bs = rows.padded_size
        self._buffer.copy_from(inputs, bs)
        for participant in self._participants:
            participant.prepare_for_replay(inputs, bs)
        self._graphs[bs].replay()
        return self._buffer.logits[: rows.size]

    def close(self) -> None:
        """Drop the graphs; before NCCL resources are freed, or the program hangs."""

        self._graphs.clear()
        gc.collect()


class GraphedForward:
    """A model's inference forward through its graphs: a decode graph for uniform decode
    batches, else a breakable graph for the hidden states when one fits, else eager; its
    images through the vision graphs when captured."""

    def __init__(
        self,
        model: BaseLLMModel,
        decode: DecodeGraph,
        breakable: BreakableGraph | None,
        vision: VisionGraph | None,
    ) -> None:
        self._model = model
        self._decode = decode
        self._breakable = breakable
        self._vision = vision
        # How a forward's inputs have the model encode its images.
        self.images: ImageEncoding = EagerImages() if vision is None else vision
        # Forwards that replayed a decode or breakable graph, or ran eagerly.
        self._decode_replays = self._breakable_replays = self._eager = 0

    def rows(self, entries: tuple[Row, ...], specs: tuple[RequestSpec, ...]) -> ForwardRows:
        return self._decode.rows(entries, specs)

    def logits(self, rows: ForwardRows, batch: ForwardInputs) -> torch.Tensor:
        if self._decode.can_replay(rows):
            self._decode_replays += 1
            return self._decode.replay(rows, batch)
        return self._model.compute_logits(self.hidden(batch), batch)

    def hidden(self, batch: ForwardInputs) -> torch.Tensor:
        breakable = self._breakable
        if breakable is not None and breakable.can_replay(batch):
            self._breakable_replays += 1
            return breakable.replay(batch)
        self._eager += 1
        return self._model.forward_hidden(batch)

    def stats(self, rounds: GraphCapture | None) -> GraphStats:
        """The graphs' metric, with the ``rounds`` the speculative runtime captured."""

        captures = (
            self._breakable and self._breakable.captured,
            self._decode.captured,
            rounds,
            self._vision and self._vision.captured,
        )
        return GraphStats(
            tuple(capture for capture in captures if capture is not None),
            self._decode_replays,
            self._breakable_replays,
            self._eager,
        )
