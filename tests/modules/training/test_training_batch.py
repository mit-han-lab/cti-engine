from __future__ import annotations

import pytest
import torch

from cti.config import BatchLayout
from cti.training import SFT, PackedTokens, PaddedTokens, SFTDatum, TrainingBatch

_CPU = torch.device("cpu")
_DATA = (
    SFTDatum((11, 12), (False, True)),
    SFTDatum((21, 22, 23, 24), (False, True, False, True)),
    SFTDatum((31, 32, 33, 34, 35), (False, True, True, True, True)),
)
_LAYOUTS = pytest.mark.parametrize("layout", list(BatchLayout), ids=lambda layout: layout.value)


def _batch(layout: BatchLayout) -> TrainingBatch:
    padding = {"pad_token_id": 99, "padded_length": 6} if layout is BatchLayout.PADDED else {}
    return TrainingBatch.sft(_DATA, layout, **padding)


@_LAYOUTS
def test_sft_data_place_the_same_targets_in_either_layout(layout: BatchLayout) -> None:
    batch = _batch(layout)

    assert batch.tokens.LAYOUT is layout
    assert batch.tokens.sequence_lengths() == [2, 4, 5]
    assert batch.row_target_counts.tolist() == [1, 2, 4]
    assert batch.objective.normalization_count(batch) == batch.num_targets == 7
    rows = batch.microbatch((0, 1, 2), _CPU)
    assert rows.target_ids.tolist() == [12, 22, 24, 32, 33, 34, 35]


@pytest.mark.parametrize(
    ("layout", "indices", "inputs", "packing", "target_indices", "target_ids"),
    (
        (BatchLayout.PADDED, (1,), [[21, 22, 23, 24, 99]], None, [0, 2], [22, 24]),
        (
            BatchLayout.PADDED,
            (0, 2),
            [[11, 12, 99, 99, 99], [31, 32, 33, 34, 35]],
            None,
            [0, 5, 6, 7, 8],
            [12, 32, 33, 34, 35],
        ),
        (
            BatchLayout.PACKED,
            (1, 2),
            [21, 22, 23, 31, 32, 33, 34],
            ([0, 1, 2, 0, 1, 2, 3], [0, 3, 7], 4),
            [0, 2, 3, 4, 5, 6],
            [22, 24, 32, 33, 34, 35],
        ),
    ),
    ids=("padded-row", "padded-rows", "packed-span"),
)
def test_a_microbatch_moves_its_rows_inputs_and_targets(
    layout, indices, inputs, packing, target_indices, target_ids
) -> None:
    rows = _batch(layout).microbatch(indices, _CPU)

    assert rows.input_ids.tolist() == inputs
    if packing is None:
        assert rows.packing is None
    else:
        positions, cu_seqlens, longest = packing
        assert rows.packing.positions.tolist() == positions
        assert rows.packing.cu_seqlens.tolist() == cu_seqlens
        assert rows.packing.max_sequence_length == longest
    assert rows.target_indices.tolist() == target_indices
    assert rows.target_ids.tolist() == target_ids


@_LAYOUTS
def test_a_microbatch_of_every_input_position_takes_no_target_indices(layout: BatchLayout):
    data = (SFTDatum((1, 2, 3), (False, True, True)), SFTDatum((4, 5, 6), (False, True, True)))
    padding = {"pad_token_id": 0} if layout is BatchLayout.PADDED else {}

    rows = TrainingBatch.sft(data, layout, **padding).microbatch((0, 1), _CPU)

    assert rows.target_indices is None
    assert rows.target_ids.tolist() == [2, 3, 5, 6]


def test_packed_microbatches_take_one_contiguous_span() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _batch(BatchLayout.PACKED).microbatch((0, 2), _CPU)


@pytest.mark.parametrize(
    ("tokens", "message"),
    (
        (
            lambda: PaddedTokens(torch.zeros((2, 3, 1), dtype=torch.int64), torch.tensor([2]), 0),
            "matrix",
        ),
        (lambda: PaddedTokens(torch.tensor([[1]]), torch.tensor([1]), 0), "two columns"),
        (lambda: PaddedTokens(torch.tensor([[1, 2, 0]]), torch.tensor([4]), 0), "between two"),
        (lambda: PaddedTokens(torch.tensor([[1, 2, 3]]), torch.tensor([2]), 0), "right-padded"),
        (lambda: PaddedTokens(torch.tensor([[1, 2]]), torch.tensor([2]), -1), "pad_token_id"),
        (
            lambda: PackedTokens(torch.tensor([[1, 2]]), torch.tensor([0, 2], dtype=torch.int32)),
            "vector",
        ),
        (lambda: PackedTokens(torch.tensor([1, 2]), torch.tensor([0, 2])), "int32"),
        (
            lambda: PackedTokens(torch.tensor([1, 2]), torch.tensor([1, 2], dtype=torch.int32)),
            "zero",
        ),
        (
            lambda: PackedTokens(torch.tensor([1, 2, 3]), torch.tensor([0, 2], dtype=torch.int32)),
            "token count",
        ),
        (
            lambda: PackedTokens(
                torch.tensor([1, 2, 3, 4]), torch.tensor([0, 1, 4], dtype=torch.int32)
            ),
            "two tokens",
        ),
    ),
)
def test_token_layouts_refuse_malformed_tensors(tokens, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        tokens()


@pytest.mark.parametrize(
    ("targets", "message"),
    (
        (torch.ones((1, 3), dtype=torch.int64), "bool"),
        (torch.ones((1, 2), dtype=torch.bool), "target coordinates"),
        (torch.tensor([[True, False, True]]), "padding"),
        (torch.zeros((1, 3), dtype=torch.bool), "at least one"),
    ),
)
def test_a_batch_refuses_targets_off_its_layout(targets: torch.Tensor, message: str) -> None:
    tokens = PaddedTokens(torch.tensor([[1, 2, 0, 0]]), torch.tensor([2]), 0)

    with pytest.raises((TypeError, ValueError), match=message):
        TrainingBatch(tokens, targets, SFT())


@pytest.mark.parametrize(
    ("token_ids", "target_mask", "message"),
    (
        ((1,), (False,), "at least two"),
        ((1, 2), (False,), "same length"),
        ((1, -2), (False, True), "non-negative"),
        ((1, 2), (0, True), "booleans"),
        ((1, 2), (True, True), "first token"),
    ),
)
def test_sft_datum_rejects_invalid_semantics(token_ids, target_mask, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SFTDatum(token_ids=token_ids, target_mask=target_mask)
