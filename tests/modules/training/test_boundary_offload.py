"""The ``full_offload`` host arena and the runner's pinned host memory: size,
registration, budget and lifetime.

``HostBoundaryOffload`` pins exactly one slot per offloaded layer from the runner's
``PinnedHostMemory``, which checks the host budget before it maps anything and
unregisters every allocation when closed or collected. The policy-level behavior, the
bit-for-bit training and the copy order, is tested with the stub model in
``tests/modules/models``.
"""

from __future__ import annotations

import gc
import mmap
from collections.abc import Iterator

import pytest
import torch

import cti.training._host_memory as host_memory
from cti.training._boundary_offload import HostBoundaryOffload
from cti.training._host_memory import PinnedHostMemory
from tests.support.host_offload import GIB, FakeCuda

_CUDA = torch.device("cuda")


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeCuda]:
    fake = FakeCuda().install(monkeypatch)
    yield fake
    fake.require_released()


@pytest.fixture
def mappings(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the size of every anonymous mapping the pinned host memory makes."""

    sizes: list[int] = []
    create = mmap.mmap

    def record(fileno: int, length: int, *args: object, **kwargs: object) -> mmap.mmap:
        sizes.append(length)
        return create(fileno, length, *args, **kwargs)

    monkeypatch.setattr(host_memory.mmap, "mmap", record)
    return sizes


def _offload(host: PinnedHostMemory, segments: int = 4) -> HostBoundaryOffload:
    """Slots of five tokens of eight BF16 values, 80 bytes each, pinned from ``host``."""

    return HostBoundaryOffload(
        segments=segments, tokens=5, width=8, dtype=torch.bfloat16, device=_CUDA, host=host
    )


def test_the_arena_pins_exactly_one_slot_per_offloaded_layer(
    fake_cuda: FakeCuda,
    mappings: list[int],
) -> None:
    host = PinnedHostMemory()
    offload = _offload(host)

    # Four checkpoints keep the last input on the device: three slots of five tokens of
    # eight BF16 values.
    nbytes = 3 * 5 * 8 * 2
    assert offload.nbytes == host.nbytes == nbytes
    assert mappings == [nbytes]
    ((address, registered),) = fake_cuda.registered
    assert registered == nbytes and address % mmap.PAGESIZE == 0
    assert [slot.data_ptr() for slot in offload._slots] == [address, address + 80, address + 160]
    assert all(slot.dtype == torch.bfloat16 and slot.numel() == 40 for slot in offload._slots)

    offload.close()
    assert fake_cuda.unregistered == []
    host.close()
    host.close()
    assert fake_cuda.unregistered == [address]
    assert host.nbytes == 0


@pytest.mark.parametrize("segments", [0, 1])
def test_without_an_offloaded_layer_nothing_is_mapped(
    fake_cuda: FakeCuda,
    mappings: list[int],
    segments: int,
) -> None:
    offload = _offload(fake_cuda.host, segments)

    assert offload.nbytes == 0
    assert mappings == [] and fake_cuda.registered == []
    offload.close()


def test_collecting_unclosed_pinned_memory_unregisters_it(fake_cuda: FakeCuda) -> None:
    host = PinnedHostMemory()
    _offload(host)
    ((address, _),) = fake_cuda.registered

    del host
    gc.collect()

    assert fake_cuda.unregistered == [address]


@pytest.mark.parametrize(
    ("total_bytes", "available_bytes"),
    [
        # 70% of 342 bytes is 239, one short of the 240-byte arena.
        pytest.param(342, 1 * GIB, id="share-of-host-memory"),
        pytest.param(64 * GIB, 239, id="available-memory"),
    ],
)
def test_an_allocation_over_the_host_budget_is_refused_before_mapping(
    fake_cuda: FakeCuda,
    mappings: list[int],
    total_bytes: int,
    available_bytes: int,
) -> None:
    fake_cuda.host_memory = (total_bytes, available_bytes)

    with pytest.raises(ValueError, match="full_offload needs 0.00 GiB of pinned host memory"):
        _offload(fake_cuda.host)

    assert mappings == [] and fake_cuda.registered == []
    assert fake_cuda.host.nbytes == 0


def test_the_host_budget_counts_every_allocation_of_the_runner(
    fake_cuda: FakeCuda,
    mappings: list[int],
) -> None:
    # Two 240-byte allocations fit 70% of 686 bytes (480); a third does not.
    fake_cuda.host_memory = (686, 1 * GIB)
    host = fake_cuda.host
    _offload(host)
    host.allocate(240, "the gradient stash")

    with pytest.raises(ValueError, match="already pins 0.00 GiB"):
        _offload(host)
    assert len(mappings) == 2 and host.nbytes == 480


def test_the_offload_requires_a_cuda_device(
    fake_cuda: FakeCuda,
    mappings: list[int],
) -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        HostBoundaryOffload(
            segments=4,
            tokens=5,
            width=8,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            host=fake_cuda.host,
        )
    assert mappings == []


def test_host_memory_reads_the_total_and_available_bytes() -> None:
    total, available = host_memory._host_memory()

    assert 0 < available <= total
