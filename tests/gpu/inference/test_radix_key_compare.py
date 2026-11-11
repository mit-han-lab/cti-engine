"""ctikernels' radix key comparison, which needs its AOT build: image key ids are negative
int32, and the first mismatch must not depend on sign."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.checkpoints()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fast_compare_key_finds_the_first_mismatch_among_negative_keys(dtype) -> None:
    from ctikernels import fast_compare_key

    key = torch.tensor([7, -(2**31), -1, -5, 9], dtype=dtype)
    for position in range(len(key)):
        other = key.clone()
        other[position] = -other[position] - 1
        assert fast_compare_key(key, other) == position
    assert fast_compare_key(key, key[:3]) == 3
    assert fast_compare_key(key, key.clone()) == len(key)
