"""The ranks' residual reduction over real gloo groups (``gloo_ranks``): off the fused
exchange it is the group's all-reduce followed by the unfused add and norm, bit for bit."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cti.distributed.runtime import ParallelGroup
from tests.support.gloo_ranks import gloo_ranks
from tests.support.reference_ops import bind_norm_references
from tests.support.tp_shards import default_dtype

pytestmark = pytest.mark.xdist_group("gloo")

_ROWS, _HIDDEN, _EPS = 5, 64, 1e-6


def _reduce(group: ParallelGroup, zero_centered: bool) -> tuple[str, bool, bool]:
    from ctikernels.norm import inference as inference_norm

    # The rank process exits after its scenario; the CPU norm is the reference.
    bind_norm_references(pytest.MonkeyPatch())
    generator = torch.Generator().manual_seed(7)
    shares = torch.randn(group.size, _ROWS, _HIDDEN, generator=generator).bfloat16()
    residual = torch.randn(_ROWS, _HIDDEN, generator=generator).bfloat16()
    weight = torch.randn(_HIDDEN, generator=generator).bfloat16()
    # Built as a model's layers are: on the meta device.
    with torch.device("meta"), default_dtype(torch.bfloat16):
        reduction = group.residual_reduction(
            hidden_size=_HIDDEN, eps=_EPS, zero_centered=zero_centered
        )
    x, stream = reduction(shares[group.rank].clone(), residual.clone(), weight)
    summed = group.all_reduce(shares[group.rank].clone())
    expected, expected_stream = inference_norm.add_rms_norm_(
        summed, residual.clone(), weight, _EPS, zero_centered=zero_centered
    )
    return reduction.provider, torch.equal(x, expected), torch.equal(stream, expected_stream)


@pytest.mark.parametrize("zero_centered", [True, False])
def test_the_reduction_off_the_exchange_is_the_all_reduce_then_the_norm(
    tmp_path: Path, zero_centered: bool
) -> None:
    with gloo_ranks(_reduce, 2, zero_centered, tmp_path) as results:
        assert results == [("nccl", True, True)] * 2
