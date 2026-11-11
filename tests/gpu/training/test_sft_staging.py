from __future__ import annotations

import pytest
import torch

from cti.config import BatchLayout, SFTShapeProfile
from cti.training import SFT, PackedTokens, PaddedTokens, TrainingBatch
from cti.training._staging import SFTStaging

pytestmark = pytest.mark.checkpoints()


def _padded_batch(offset: int, *, sparse: bool) -> TrainingBatch:
    return TrainingBatch(
        PaddedTokens(
            torch.tensor(
                ((offset + 1, offset + 2, offset + 3, 0), (offset + 4, offset + 5, 0, 0)),
                dtype=torch.int64,
            ),
            torch.tensor((3, 2), dtype=torch.int64),
            0,
        ),
        torch.tensor(
            ((False, True, False), (True, False, False))
            if sparse
            else ((True, True, False), (True, False, False))
        ),
        SFT(),
    )


def _packed_batch(offset: int, *, sparse: bool) -> TrainingBatch:
    return TrainingBatch(
        PackedTokens(
            torch.tensor(
                (offset + 1, offset + 2, offset + 3, offset + 4, offset + 5),
                dtype=torch.int64,
            ),
            torch.tensor((0, 3, 5), dtype=torch.int32),
        ),
        torch.tensor((False, True, True)) if sparse else torch.ones(3, dtype=torch.bool),
        SFT(),
    )


@pytest.mark.parametrize("packed", [False, True], ids=("padded", "packed"))
@pytest.mark.parametrize("sparse", [False, True], ids=("dense", "sparse"))
def test_fixed_sft_staging_preserves_three_accepted_cpu_batches(
    packed: bool,
    sparse: bool,
) -> None:
    device = torch.device("cuda")
    profile = SFTShapeProfile(
        (3, 2),
        padded_width=4,
        num_target_tokens=2 if sparse else None,
    )
    staging = SFTStaging(profile, BatchLayout.PACKED if packed else BatchLayout.PADDED, device)
    stream = torch.cuda.Stream(device=device)
    snapshots: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    device_pointers: tuple[int, int] | None = None

    for offset in (0, 10, 20):
        batch = (
            _packed_batch(offset, sparse=sparse)
            if packed
            else _padded_batch(
                offset,
                sparse=sparse,
            )
        )
        expected = batch.microbatch((0, 1), torch.device("cpu"))
        # Padded rows on the host are views of the batch's tokens, which the loop refills.
        expected_inputs, expected_targets = expected.input_ids.clone(), expected.target_ids.clone()

        with torch.cuda.stream(stream):
            staged = staging.stage(batch)
            pointers = (staged.input_ids.data_ptr(), staged.target_ids.data_ptr())
            if device_pointers is None:
                device_pointers = pointers
            else:
                assert pointers == device_pointers
            snapshots.append(
                (
                    staged.input_ids.clone(),
                    staged.target_ids.clone(),
                    expected_inputs,
                    expected_targets,
                )
            )

        # The accepted command owns a pinned copy, not the caller's borrowed CPU
        # storage. Reusing the source immediately must not change queued work.
        batch.tokens.token_ids.fill_(-1)

    stream.synchronize()
    for actual_inputs, actual_targets, expected_inputs, expected_targets in snapshots:
        torch.testing.assert_close(actual_inputs.cpu(), expected_inputs)
        torch.testing.assert_close(actual_targets.cpu(), expected_targets)


def test_full_padded_staging_omits_identity_target_indices() -> None:
    device = torch.device("cuda")
    profile = SFTShapeProfile((4, 4))
    staging = SFTStaging(profile, BatchLayout.PADDED, device)
    batch = TrainingBatch(
        PaddedTokens(
            torch.tensor(((1, 2, 3, 4), (5, 6, 7, 8)), dtype=torch.int64),
            torch.tensor((4, 4), dtype=torch.int64),
            0,
        ),
        torch.ones((2, 3), dtype=torch.bool),
        SFT(),
    )

    staged = staging.stage(batch)
    torch.cuda.current_stream().synchronize()

    assert staged.target_indices is None
    torch.testing.assert_close(staged.target_ids.cpu(), torch.tensor((2, 3, 4, 6, 7, 8)))
