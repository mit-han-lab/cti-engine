from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cti.forward import ForwardRows, RecurrentExport
from cti.kvcache import RecurrentStatePool
from cti.layers.gated_delta_net.layer import gated_delta_net_state
from cti.recurrent import RecurrentBackend
from tests.support.forward_rows import Entry, decode_entry, extend_entry, forward_rows

NUM_SLOTS = 4
CPU = torch.device("cpu")


_DUMMY = decode_entry(NUM_SLOTS, 0)


def _backend() -> RecurrentBackend:
    pool = RecurrentStatePool([gated_delta_net_state(1, 2, 3, 4)], num_slots=NUM_SLOTS, device=CPU)
    return RecurrentBackend(pool, device=CPU)


def _batch(*entries: Entry, padding: int = 0) -> ForwardRows:
    return forward_rows(*entries, padding=padding, padding_slot=NUM_SLOTS)


def test_metadata_addresses_slots_extend_lengths_and_padding() -> None:
    backend = _backend()
    batch = _batch(
        extend_entry(3, 0, 5),
        extend_entry(1, 5, 9),
        decode_entry(0, 7),
        decode_entry(2, 11),
        padding=2,
    )

    metadata = backend.prepare_metadata(batch)

    assert metadata.slot_ids.dtype == torch.int32
    assert metadata.slot_ids.tolist() == [3, 1, 0, 2, -1, -1]
    assert metadata.num_extend_requests == 2
    assert metadata.num_extend_tokens == 9
    assert metadata.cu_seqlens is not None and metadata.cu_seqlens.dtype == torch.int32
    assert metadata.cu_seqlens.tolist() == [0, 5, 9]
    assert metadata.has_initial_state is not None
    assert metadata.has_initial_state.dtype == torch.bool
    assert metadata.has_initial_state.tolist() == [False, True]
    assert metadata.slot_ids.device == CPU


def test_decode_metadata_marks_graph_padding_and_carries_no_sequence_bounds() -> None:
    backend = _backend()
    batch = _batch(decode_entry(2, 4), decode_entry(0, 9), decode_entry(3, 1), padding=1)

    metadata = backend.prepare_metadata(batch)

    assert metadata.slot_ids.tolist() == [2, 0, 3, -1]
    assert metadata.num_extend_requests == 0
    assert metadata.num_extend_tokens == 0
    assert metadata.cu_seqlens is None
    assert metadata.has_initial_state is None


def test_capture_binds_an_all_padding_static_buffer_that_replay_overwrites() -> None:
    backend = _backend()
    backend.init_capture_graph(max_seq_len=64, bs_list=[1, 2, 4])

    capture = _batch(*(_DUMMY,) * 4)
    captured = backend.capture_metadata(capture)
    captured_slots = captured.slot_ids
    assert captured_slots.tolist() == [-1, -1, -1, -1]
    assert captured.num_extend_requests == 0

    replay = _batch(decode_entry(3, 5), decode_entry(0, 4), padding=2)
    metadata = backend.prepare_metadata(replay)
    backend.prepare_for_replay(SimpleNamespace(recurrent_metadata=metadata), replay.padded_size)
    assert captured_slots.tolist() == [3, 0, -1, -1]

    # A smaller bucket captures a prefix view of the same buffer, reset to padding.
    smaller = _batch(_DUMMY, _DUMMY)
    smaller_slots = backend.capture_metadata(smaller).slot_ids
    assert smaller_slots.data_ptr() == captured_slots.data_ptr()
    assert captured_slots.tolist() == [-1, -1, -1, -1]


def test_graph_participation_rejects_unprepared_or_non_decode_use() -> None:
    backend = _backend()

    with pytest.raises(RuntimeError, match="no recurrent decode graph for batch size 1"):
        backend.capture_metadata(_batch(_DUMMY))
    backend.init_capture_graph(max_seq_len=8, bs_list=[1, 4])
    with pytest.raises(RuntimeError, match="already initialized"):
        backend.init_capture_graph(max_seq_len=8, bs_list=[1])
    with pytest.raises(RuntimeError, match="batch size 2"):
        backend.capture_metadata(_batch(_DUMMY, _DUMMY))
    extend = _batch(extend_entry(0, 0, 2))
    extend_metadata = backend.prepare_metadata(extend)
    with pytest.raises(RuntimeError, match="only decode batches"):
        backend.prepare_for_replay(
            SimpleNamespace(recurrent_metadata=extend_metadata), extend.padded_size
        )


@pytest.mark.parametrize("copies", [1, 4, 17])
def test_snapshot_restores_preserve_other_rows_across_layers(copies):
    pool = RecurrentStatePool(
        [gated_delta_net_state(1, 2, 3, 4)] * 2,
        num_slots=copies + 1,
        num_snapshot_rows=copies,
        device=CPU,
    )
    backend = RecurrentBackend(pool, device=CPU)
    sources = list(range(copies + 1, copies * 2 + 1))
    batch = _batch(
        *(extend_entry(i, 64, 128, restore_row=s) for i, s in enumerate(sources)),
        decode_entry(copies, 7),
    )
    views = [view for layer in range(pool.num_layers) for view in pool.layer_views(layer)]
    originals = [view.normal_().clone() for view in views]
    metadata = backend.prepare_metadata(batch)
    assert metadata.restore.tolist() == [sources, list(range(copies))]
    assert metadata.capture_rows is None
    backend.restore(SimpleNamespace(recurrent_metadata=metadata))
    for view, original in zip(views, originals, strict=True):
        assert torch.equal(view[:copies], original[sources])
        assert torch.equal(view[copies:], original[copies:])


def _capturing(index: int, start: int, end: int, *positions: int) -> Entry:
    captures = tuple(RecurrentExport(p, 10 + 3 * index + k) for k, p in enumerate(positions))
    return extend_entry(index, start, end, captures=captures)


def test_captures_address_each_extend_by_one_row_and_minus_one_for_none() -> None:
    pool = RecurrentStatePool(
        [gated_delta_net_state(1, 2, 3, 4)], num_slots=4, num_snapshot_rows=16, device=CPU
    )
    backend = RecurrentBackend(pool, device=CPU)
    # Each capture lies at the most whole 64-token multiples past its extend's start.
    batch = _batch(
        _capturing(0, 64, 300, 256),
        _capturing(1, 0, 5),
        _capturing(2, 0, 129, 128),
        _capturing(3, 0, 128, 128),
    )
    metadata = backend.prepare_metadata(batch)
    assert metadata.capture_rows.dtype == torch.int32
    assert metadata.capture_rows.tolist() == [10, -1, 16, 19]

    batch = _batch(_capturing(2, 0, 129, 128), decode_entry(0, 9))
    assert backend.prepare_metadata(batch).capture_rows.tolist() == [16]


def test_decode_exports_copy_the_advanced_rows_after_the_step() -> None:
    pool = RecurrentStatePool(
        [gated_delta_net_state(1, 2, 3, 4)] * 2, num_slots=4, num_snapshot_rows=8, device=CPU
    )
    backend = RecurrentBackend(pool, device=CPU)
    batch = _batch(
        _capturing(0, 0, 128, 128),
        decode_entry(1, 63, captures=(RecurrentExport(64, 6),)),
        decode_entry(2, 9),
        decode_entry(3, 127, captures=(RecurrentExport(128, 7),)),
    )
    metadata = backend.prepare_metadata(batch)
    assert metadata.exports == ((1, 6), (3, 7))
    assert metadata.capture_rows.tolist() == [10]
    views = [view for layer in range(pool.num_layers) for view in pool.layer_views(layer)]
    originals = [view.normal_().clone() for view in views]
    backend.export(SimpleNamespace(recurrent_metadata=metadata))
    for view, original in zip(views, originals, strict=True):
        assert torch.equal(view[6], original[1]) and torch.equal(view[7], original[3])
        assert torch.equal(view[:6], original[:6]) and torch.equal(view[8:], original[8:])
