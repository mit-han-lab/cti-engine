"""The caching allocator maps the segment ``native_backing.segment`` predicts for one request
from an empty pool: it rounds the request to a 512-byte block before it picks the segment
class, so a request just below 10 MiB takes a segment of its own."""

from __future__ import annotations

import pytest
import torch

from cti.memory import native_backing

pytestmark = pytest.mark.checkpoints()

_MIB = 1 << 20


@pytest.mark.parametrize(
    "size",
    [512, _MIB, _MIB + 1, 10 * _MIB - 512, 10 * _MIB - 511, 10 * _MIB - 160, 10 * _MIB + 1],
)
def test_one_request_maps_the_modelled_segment(size: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    if native_backing.custom_placement_options():
        pytest.skip("the allocator configuration changes the native placement")
    pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(pool):
        before = torch.cuda.memory_reserved()
        block = torch.empty(size, dtype=torch.uint8, device="cuda")
        mapped = torch.cuda.memory_reserved() - before
    del block
    assert mapped == native_backing.segment(size)
