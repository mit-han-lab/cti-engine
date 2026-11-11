"""The decode-graph runner over a Gated DeltaNet layer's recurrent backend on CPU."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

import cti.inference._execution._graphs as graph_module
from cti.forward import ForwardInputs, ForwardRows
from cti.inference._execution._graphs import DecodeGraph
from cti.inference._step import QueryKind, Row
from tests.support.forward_rows import decode_entry, extend_entry
from tests.support.gated_delta_net import (
    NUM_SLOTS,
    PROJECTED_WIDTH,
    bind_reference,
    gdn_layer,
    pool_snapshot,
    projected_rows,
    run_layer,
)


def test_decode_graph_runner_captures_padding_and_replays_slots_through_its_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU stand-in for CUDA graphs: replay re-runs the captured forward on the
    captured batch, so it reads only the static buffers bound at capture."""

    vocab_size = 4
    events: list[tuple[str, object]] = []
    layer = gdn_layer()
    bound = bind_reference(layer)
    run_layer(layer, bound.backend, projected_rows(4), extend_entry(0, 0, 4))
    run_layer(layer, bound.backend, projected_rows(5, seed=8), extend_entry(3, 0, 5))
    before_state, before_conv = pool_snapshot(bound.pool)

    class Attention:
        def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
            events.append(("attention.init", tuple(bs_list)))

        def prepare_for_capture(self, rows: ForwardRows, inputs: object) -> object:
            events.append(("attention.capture", rows.size))
            return inputs

        def prepare_for_replay(self, batch: object, bs: int) -> None:
            slots = bound.backend._capture_slot_ids
            events.append(("attention.replay", None if slots is None else slots.tolist()))

    class Model:
        def __init__(self) -> None:
            self.projected = torch.zeros(4, PROJECTED_WIDTH, dtype=torch.bfloat16)
            self.capturing: Graph | None = None
            self.capture_slots: list[list[int]] = []

        def forward(self, batch: ForwardInputs) -> torch.Tensor:
            if self.capturing is not None:
                self.capturing.batch = batch
                self.capture_slots.append(batch.recurrent_metadata.slot_ids.tolist())
            # A captured batch's size is its padded size.
            y = layer.forward(self.projected[: batch.size], batch)
            return y[:, :vocab_size].float()

    model = Model()

    class Graph:
        def __init__(self) -> None:
            self.batch: ForwardInputs | None = None

        def replay(self) -> None:
            assert self.batch is not None
            runner._buffer.logits[: self.batch.size] = model.forward(self.batch)

    class Pool:
        @contextmanager
        def workspace(self, max_rows: int) -> Iterator[None]:
            yield

        @contextmanager
        def capture(self, graph: Graph, stream: object) -> Iterator[None]:
            model.capturing = graph
            try:
                yield
            finally:
                model.capturing = None

    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(graph_module, "get_free_memory", lambda _device: 0)

    runner = DecodeGraph(
        stream=None,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        model=model,  # type: ignore[arg-type]
        participants=(Attention(), bound.backend),  # type: ignore[arg-type]
        batch_sizes=[1, 2, 4],
        max_seq_len=64,
        vocab_size=vocab_size,
        logits_dtype=torch.float32,
        dummy_entry=Row(NUM_SLOTS, QueryKind.DECODE, 0, 1, 1),
        pool=Pool(),  # type: ignore[arg-type]
    )

    # Decode reads one int64 rotary row, M-RoPE included.
    rope_buffer = runner._buffer.rope_positions
    assert (rope_buffer.dtype, rope_buffer.shape) == (torch.int64, (4,))
    assert events == [
        ("attention.init", (1, 2, 4)),
        ("attention.capture", 4),
        ("attention.capture", 2),
        ("attention.capture", 1),
    ]
    assert model.capture_slots == [[-1] * 4, [-1] * 2, [-1]]
    assert torch.equal(bound.pool.layer_views(0)[0], before_state)
    assert torch.equal(bound.pool.layer_views(0)[1], before_conv)

    twin_layer = gdn_layer()
    twin = bind_reference(twin_layer)
    run_layer(twin_layer, twin.backend, projected_rows(4), extend_entry(0, 0, 4))
    run_layer(twin_layer, twin.backend, projected_rows(5, seed=8), extend_entry(3, 0, 5))

    def replay(seed: int, *decodes: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay a decode of each ``(slot, position)`` beside the twin's eager one."""

        step = projected_rows(len(decodes), seed=seed)
        eager = run_layer(twin_layer, twin.backend, step, *(decode_entry(*d) for d in decodes))
        entries = tuple(Row(slot, QueryKind.DECODE, p, p + 1, p + 1) for slot, p in decodes)
        rows = runner.rows(entries, ())
        assert rows.padded_size == 4
        batch = ForwardInputs(
            torch.zeros(4, dtype=torch.int32),
            torch.zeros_like(rope_buffer),
            None,
            rows.size,
            ragged=False,
            recurrent_metadata=bound.backend.prepare_metadata(rows),
        )
        model.projected[: len(decodes)] = step
        return runner.replay(rows, batch), eager[:, :vocab_size].float()

    events.clear()
    # A full bucket first, so the next, padded replay of the bucket starts from real slots.
    full, eager_full = replay(11, (3, 5), (0, 4), (1, 7), (2, 9))
    assert bound.backend._capture_slot_ids.tolist() == [3, 0, 1, 2]
    assert torch.equal(full, eager_full)
    slot_2_rows = [view[2].clone() for view in bound.pool.layer_views(0)]

    padded, eager_padded = replay(12, (3, 6), (0, 5), (1, 8))

    # Attention prepares before the recurrent backend, so it still sees the previous slots.
    assert events == [
        ("attention.replay", [-1, -1, -1, -1]),
        ("attention.replay", [3, 0, 1, 2]),
    ]
    # The padding row is rewritten to -1: it gives zero and slot 2 does not advance.
    assert bound.backend._capture_slot_ids.tolist() == [3, 0, 1, -1]
    assert torch.equal(runner._buffer.logits[3], torch.zeros(vocab_size))
    assert all(
        torch.equal(view[2], before)
        for view, before in zip(bound.pool.layer_views(0), slot_2_rows, strict=True)
    )
    assert torch.equal(padded, eager_padded)
    assert torch.equal(bound.pool.layer_views(0)[0], twin.pool.layer_views(0)[0])
    assert torch.equal(bound.pool.layer_views(0)[1], twin.pool.layer_views(0)[1])
