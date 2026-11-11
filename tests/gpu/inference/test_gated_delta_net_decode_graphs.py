"""The decode-graph runner on the GPU over a Gated DeltaNet layer's recurrent backend, bound
to ``ctikernels.gdn``: captured graphs replay slots and padding through the backend as
participant, bitwise against an eager twin.

Each test runs one child process that prints a receipt; the parent asserts on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.support.gated_delta_net_gpu import (
    NUM_SLOTS,
    PROJECTED_WIDTH,
    bind_layer,
    component_receipt,
    gdn_layer,
    pools_equal,
    projected_rows,
    relative_error,
    run_component,
    run_layer,
)

pytestmark = pytest.mark.checkpoints()

if TYPE_CHECKING:
    import torch

GRAPH_BATCH_SIZES = (1, 2, 4, 8)
VOCAB_SIZE = 16
# Decode replays after capture: padded (3 -> 4) straight after capture, then shrinking
# and growing batches; the last two fill bucket 4 and then pad it, so a padding row that
# kept the previous replay's slot would advance that slot.
REPLAY_STEPS = ((5, 2, 7), (1,), (0, 3, 6, 4, 2, 5, 7, 1), (6, 0), (4, 1, 3, 2), (0, 6, 5))


# ============================== Child process ==============================


def _graphs(device: torch.device) -> dict[str, Any]:
    """Decode CUDA graphs through ``DecodeGraph`` with the recurrent backend, replayed
    against an eager twin."""

    import torch

    from cti.forward import ForwardInputs, ForwardRows
    from cti.inference._execution._graphs import DecodeGraph
    from cti.inference._step import QueryKind, Row
    from cti.memory import GraphPool
    from tests.support.forward_rows import decode_entry, extend_entry

    receipt: dict[str, Any] = {}
    graph_layer = gdn_layer(device)
    graph_backend, graph_pool = bind_layer(graph_layer, device)
    twin_layer = gdn_layer(device)
    twin_backend, twin_pool = bind_layer(twin_layer, device)
    positions: dict[int, int] = {}
    for slot in range(NUM_SLOTS):
        rows = projected_rows(5 + slot, 20 + slot, device)
        for layer, backend in ((graph_layer, graph_backend), (twin_layer, twin_backend)):
            run_layer(layer, backend, rows, extend_entry(slot, 0, 5 + slot))
        positions[slot] = 5 + slot
    torch.cuda.synchronize(device)
    before_capture = [view.clone() for view in graph_pool.layer_views(0)]

    class Model:
        def __init__(self) -> None:
            self.projected = torch.zeros(
                max(GRAPH_BATCH_SIZES), PROJECTED_WIDTH, dtype=torch.bfloat16, device=device
            )

        def forward(self, batch: ForwardInputs) -> torch.Tensor:
            # A captured batch's size is its padded size.
            y = graph_layer.forward(self.projected[: batch.size], batch)
            return y[:, :VOCAB_SIZE].float()

    class NoAttention:
        """The layer reads no attention metadata."""

        def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
            pass

        def prepare_for_capture(self, rows: ForwardRows, inputs: object) -> object:
            return inputs

        def prepare_for_replay(self, batch: object, bs: int) -> None:
            pass

    model = Model()
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        runner = DecodeGraph(
            stream=stream,
            device=device,
            model=model,  # type: ignore[arg-type]
            participants=(NoAttention(), graph_backend),  # type: ignore[arg-type]
            batch_sizes=list(GRAPH_BATCH_SIZES),
            max_seq_len=64,
            vocab_size=VOCAB_SIZE,
            logits_dtype=torch.float32,
            dummy_entry=Row(NUM_SLOTS, QueryKind.DECODE, 0, 1, 1),
            pool=GraphPool(),
        )
    torch.cuda.synchronize(device)
    receipt["capture_left_pool_untouched"] = all(
        torch.equal(now, then)
        for now, then in zip(graph_pool.layer_views(0), before_capture, strict=True)
    )

    steps = []
    for step, slots in enumerate(REPLAY_STEPS):
        rows = projected_rows(len(slots), 100 + step, device)
        entries = tuple(
            Row(slot, QueryKind.DECODE, positions[slot], positions[slot] + 1, positions[slot] + 1)
            for slot in slots
        )
        forward_rows = runner.rows(entries, ())
        padded = forward_rows.padded_size
        batch = ForwardInputs(
            torch.zeros(padded, dtype=torch.int32, device=device),
            torch.zeros(padded, dtype=torch.int64, device=device),
            None,
            forward_rows.size,
            ragged=False,
            recurrent_metadata=graph_backend.prepare_metadata(forward_rows),
        )
        previous_slots = graph_backend._capture_slot_ids[:padded].tolist()
        padding_slots = [slot for slot in previous_slots[len(slots) :] if slot >= 0]
        before = [view[padding_slots].clone() for view in graph_pool.layer_views(0)]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            model.projected[: len(slots)] = rows
            logits = runner.replay(forward_rows, batch).clone()
            padding_logits = runner._buffer.logits[len(slots) : padded].clone()
        stream.synchronize()
        eager = run_layer(
            twin_layer, twin_backend, rows, *(decode_entry(slot, positions[slot]) for slot in slots)
        )[:, :VOCAB_SIZE].float()
        torch.cuda.synchronize(device)
        steps.append(
            {
                "slots": list(slots),
                "padded": padded,
                "previous_padding_slots": padding_slots,
                "bound_slots": graph_backend._capture_slot_ids[:padded].tolist(),
                "padding_logits_zero": bool(torch.count_nonzero(padding_logits) == 0),
                "previous_padding_slots_untouched": all(
                    torch.equal(view[padding_slots], then)
                    for view, then in zip(graph_pool.layer_views(0), before, strict=True)
                ),
                "relative_error": relative_error(logits, eager),
                "bitwise": torch.equal(logits, eager),
                "pools_equal": pools_equal(graph_pool, twin_pool),
            }
        )
        for slot in slots:
            positions[slot] += 1
    receipt["replays"] = steps
    runner.close()
    return receipt


# ============================== Tests ==============================


def test_decode_graphs_replay_slots_and_padding_through_the_participant(
    tmp_path: Path,
) -> None:
    receipt = component_receipt(
        tmp_path, "tests.gpu.inference.test_gated_delta_net_decode_graphs", "graphs"
    )

    assert receipt["capture_left_pool_untouched"] is True
    replays = receipt["replays"]
    assert [replay["slots"] for replay in replays] == [list(slots) for slots in REPLAY_STEPS]
    assert [replay["padded"] for replay in replays] == [4, 1, 8, 2, 4, 4]
    for replay in replays:
        padded, slots = replay["padded"], replay["slots"]
        assert replay["bound_slots"] == slots + [-1] * (padded - len(slots))
        assert replay["padding_logits_zero"] is True
        assert replay["previous_padding_slots_untouched"] is True
        assert replay["bitwise"] is True, replay
        assert replay["pools_equal"] is True, replay
    # The last replay pads bucket 4 right after a full replay of it: its padding row
    # held slot 2, which must neither be bound again nor advance.
    assert replays[-1]["previous_padding_slots"] == [2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", required=True, choices=("graphs",))
    run_component(parser.parse_args().component, {"graphs": _graphs})


if __name__ == "__main__":
    main()
