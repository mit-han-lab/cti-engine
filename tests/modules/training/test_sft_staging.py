from __future__ import annotations

import pytest
import torch

from cti.config import BatchLayout, SFTShapeProfile
from cti.training import SFTDatum, TrainingBatch
from cti.training._staging import SFTStaging

_CPU = torch.device("cpu")
_LAYOUTS = pytest.mark.parametrize("layout", list(BatchLayout), ids=lambda layout: layout.value)


def _batch(layout: BatchLayout, *masks: tuple[bool, ...]) -> TrainingBatch:
    sequences = ((4, 5), (1, 2, 3, 4))
    masks = masks or ((False, True), (False, True, True, True))
    data = [SFTDatum(tokens, mask) for tokens, mask in zip(sequences, masks, strict=True)]
    padding = {"pad_token_id": 0} if layout is BatchLayout.PADDED else {}
    return TrainingBatch.sft(data, layout, **padding)


@pytest.mark.parametrize(
    ("sequence_lengths", "error_type", "message"),
    [
        ((), ValueError, "must not be empty"),
        ((1,), ValueError, "at least two"),
    ],
)
def test_shape_profile_rejects_invalid_sequence_lengths(sequence_lengths, error_type, message):
    with pytest.raises(error_type, match=message):
        SFTShapeProfile(sequence_lengths)


@pytest.mark.parametrize("sparse", (False, True), ids=("dense", "sparse"))
@_LAYOUTS
def test_staging_matches_the_batchs_own_microbatch_in_reused_buffers(
    layout: BatchLayout, sparse: bool
) -> None:
    masks = ((False, True), (False, False, True, False)) if sparse else ()
    batch = _batch(layout, *masks)
    profile = SFTShapeProfile((2, 4), num_target_tokens=batch.num_targets)
    staging = SFTStaging(profile, layout, _CPU)

    first = staging.stage(batch)
    pointers = (first.input_ids.data_ptr(), first.target_ids.data_ptr())
    expected = batch.microbatch((0, 1), _CPU)

    for actual, wanted in (
        (first.input_ids, expected.input_ids),
        (first.target_ids, expected.target_ids),
        (first.target_indices, expected.target_indices),
    ):
        assert (actual is None) == (wanted is None)
        if wanted is not None:
            torch.testing.assert_close(actual, wanted)
    if layout is BatchLayout.PACKED:
        torch.testing.assert_close(first.packing.positions, expected.packing.positions)
        torch.testing.assert_close(first.packing.cu_seqlens, expected.packing.cu_seqlens)
        assert first.packing.max_sequence_length == expected.packing.max_sequence_length
    second = staging.stage(batch)
    assert (second.input_ids.data_ptr(), second.target_ids.data_ptr()) == pointers


def test_full_padded_staging_takes_no_target_indices() -> None:
    data = [SFTDatum((1, 2, 3), (False, True, True)), SFTDatum((4, 5, 6), (False, True, True))]
    staging = SFTStaging(SFTShapeProfile((3, 3)), BatchLayout.PADDED, _CPU)

    staged = staging.stage(TrainingBatch.sft(data, BatchLayout.PADDED, pad_token_id=0))

    assert staged.target_indices is None
    torch.testing.assert_close(staged.target_ids, torch.tensor((2, 3, 5, 6)))


@_LAYOUTS
def test_staging_refuses_a_batch_off_its_profile(layout: BatchLayout) -> None:
    staging = SFTStaging(SFTShapeProfile((2, 5)), layout, _CPU)

    with pytest.raises(ValueError, match="profile mismatch"):
        staging.require_profile(_batch(layout))
