from __future__ import annotations

import weakref
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

from cti.memory import MemoryManager, ResidencyRestoreError, free_bytes, unified_memory
from tests.support.residency import one_rank


class _MemorySaver:
    def __init__(
        self,
        *,
        failure: str | None = None,
        failure_tag: str | None = None,
    ) -> None:
        self.events: list[str] = []
        self.failure = failure
        self.failure_tag = failure_tag

    def _fails(self, operation: str, tag: str) -> bool:
        return self.failure == operation and (self.failure_tag is None or self.failure_tag == tag)

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False) -> Iterator[None]:
        self.events.append(f"region({tag}, backup={enable_cpu_backup})+")
        if self._fails("allocation", tag):
            raise RuntimeError("allocation failed")
        try:
            yield
        finally:
            self.events.append(f"region({tag})-")
            if self._fails("exit", tag):
                raise RuntimeError("exit failed")

    def pause(self, tag: str) -> None:
        self.events.append(f"pause({tag})")
        if self._fails("pause", tag):
            raise RuntimeError("pause failed")

    def resume(self, tag: str) -> None:
        self.events.append(f"resume({tag})")
        if self._fails("resume", tag):
            raise RuntimeError("resume failed")


class _Pool:
    def __init__(self, index: int) -> None:
        self.index = index


class _AllocatorPools:
    """Stand-in for the allocator's private pools: records creation, routing and release.

    It also records binding the thread's cuBLAS workspace for the current stream.
    """

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._pools: list[weakref.ref[_Pool]] = []

    @property
    def live(self) -> list[int]:
        return [pool.index for reference in self._pools if (pool := reference()) is not None]

    def create(self) -> _Pool:
        pool = _Pool(len(self._pools))
        self._pools.append(weakref.ref(pool))
        weakref.finalize(pool, self._events.append, f"pool-{pool.index}-")
        self._events.append(f"pool-{pool.index}+")
        return pool

    @contextmanager
    def use(self, pool: _Pool) -> Iterator[None]:
        self._events.append(f"use(pool-{pool.index})+")
        try:
            yield
        finally:
            self._events.append(f"use(pool-{pool.index})-")

    def bind_blas_workspace(self) -> int:
        self._events.append("cublas-workspace")
        return 0


def _allocator_pools(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> _AllocatorPools:
    pools = _AllocatorPools(events)
    monkeypatch.setattr(torch.cuda, "MemPool", pools.create)
    monkeypatch.setattr(torch.cuda, "use_mem_pool", pools.use)
    monkeypatch.setattr(torch.cuda, "current_blas_handle", pools.bind_blas_workspace)
    return pools


class _Device:
    """The allocator's reserved bytes and the device's free bytes, as a test sets them."""

    def __init__(self) -> None:
        self.reserved = 0
        self.free = 1 << 40

    def allocate(self, nbytes: int) -> None:
        self.reserved += nbytes
        self.free -= nbytes


@pytest.fixture(autouse=True)
def device(monkeypatch: pytest.MonkeyPatch) -> _Device:
    device = _Device()
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_args: device.reserved)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_args: (device.free, 1 << 41))
    return device


def _training_allocation(
    manager: MemoryManager, region: str
) -> Callable[[], AbstractContextManager[None]]:
    if region == "transient":
        return partial(manager.transient_training_allocation, 0)
    return manager.persistent_training_allocation


def _allocate(manager: MemoryManager) -> None:
    with manager.kv_allocation():
        pass


def test_memory_manager_owns_one_kv_region_without_cpu_backup() -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)

    _allocate(manager)

    assert provider.events == ["region(kv-cache, backup=False)+", "region(kv-cache)-"]
    with pytest.raises(RuntimeError, match="KV memory is resident, expected unallocated"):
        _allocate(manager)


def test_memory_manager_requires_valid_residency_transitions() -> None:
    manager = MemoryManager(_MemorySaver(), unified=False)

    with pytest.raises(RuntimeError, match="KV memory is unallocated, expected resident"):
        manager.suspend_kv()
    with pytest.raises(RuntimeError, match="KV memory is unallocated, expected suspended"):
        manager.resume_kv(one_rank)

    _allocate(manager)
    with pytest.raises(RuntimeError, match="KV memory is resident, expected unallocated"):
        _allocate(manager)


def test_memory_manager_suspend_and_resume_are_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    _allocate(manager)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: provider.events.append("empty_cache"))

    manager.suspend_kv()
    manager.suspend_kv()
    manager.resume_kv(one_rank)
    manager.resume_kv(one_rank)

    assert provider.events[2:] == ["pause(kv-cache)", "empty_cache", "resume(kv-cache)"]


def test_memory_manager_empties_allocator_cache_before_provider_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    _allocate(manager)
    manager.suspend_kv()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: provider.events.append("empty_cache"))

    manager.resume_kv(one_rank)

    assert provider.events[-2:] == ["empty_cache", "resume(kv-cache)"]


def test_kv_resume_that_does_not_fit_raises_and_stays_suspended(device: _Device) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    required = 60
    with manager.kv_allocation():
        for nbytes in (10, 20, 30):
            device.allocate(nbytes)
    manager.suspend_kv()
    device.free = required - 1

    with pytest.raises(ResidencyRestoreError) as raised:
        manager.resume_kv(one_rank)

    assert (raised.value.region, raised.value.required_bytes) == ("KV", required)
    assert not any(event.startswith("resume(") for event in provider.events)
    device.free = required
    manager.resume_kv(one_rank)
    assert sum(event.startswith("resume(") for event in provider.events) == 1


@pytest.mark.parametrize(
    ("integrated", "device_type", "unified"),
    [(1, "cuda", True), (0, "cuda", False), (1, "cpu", False)],
)
def test_unified_memory_is_an_integrated_cuda_device(
    monkeypatch: pytest.MonkeyPatch, integrated: int, device_type: str, unified: bool
) -> None:
    properties = SimpleNamespace(is_integrated=integrated)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)

    assert unified_memory(torch.device(device_type)) is unified


def test_unified_memory_reads_the_hosts_available_memory(
    monkeypatch: pytest.MonkeyPatch, device: _Device
) -> None:
    """GB10's own reading is MemFree, which counts the page cache as used."""

    monkeypatch.setattr("cti.memory._host_available_bytes", lambda: device.free + 7)

    assert free_bytes(unified=True) == device.free + 7
    assert free_bytes(unified=False) == device.free


def test_unified_resume_fits_what_the_host_has_available(
    monkeypatch: pytest.MonkeyPatch, device: _Device
) -> None:
    """A checkpoint load leaves the page cache full: the device's own reading falls short of
    the KV, the host's available memory does not (GB10, 10-28: KV 14.60 GiB, MemFree 11.76 GiB,
    MemAvailable over 57 GiB)."""

    available = {"bytes": 0}
    monkeypatch.setattr("cti.memory._host_available_bytes", lambda: available["bytes"])
    manager = MemoryManager(_MemorySaver(), unified=True)
    required = 60
    with manager.kv_allocation():
        device.allocate(required)
    manager.suspend_kv()
    device.free, available["bytes"] = required - 1, required - 1

    with pytest.raises(ResidencyRestoreError) as raised:
        manager.resume_kv(one_rank)

    assert raised.value.free_bytes == required - 1
    available["bytes"] = required
    manager.resume_kv(one_rank)


def test_kv_resume_checks_the_graph_region_with_the_kv_before_mapping_either(
    device: _Device,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    with manager.kv_allocation():
        device.allocate(10)
    with manager.graph_allocation():
        device.allocate(5)
    manager.suspend_kv()
    device.free = 14

    with pytest.raises(ResidencyRestoreError) as raised:
        manager.resume_kv(one_rank)

    assert raised.value.required_bytes == 15
    assert not any(event.startswith("resume(") for event in provider.events)
    device.free = 15
    manager.resume_kv(one_rank)
    assert provider.events[-2:] == ["resume(inference-graph)", "resume(kv-cache)"]


def test_a_resume_a_peer_rank_cannot_back_is_refused_here_too(device: _Device) -> None:
    """Every rank refuses or none does: the vote reports the tightest rank's free bytes."""

    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    with manager.kv_allocation():
        device.allocate(10)
    manager.suspend_kv()  # this device fits it; a peer keeps 3 bytes too few

    with pytest.raises(ResidencyRestoreError) as raised:
        manager.resume_kv(lambda spare: min(spare, -3))

    assert (raised.value.required_bytes, raised.value.free_bytes) == (10, 7)
    assert not any(event.startswith("resume(") for event in provider.events)


def test_training_resume_that_does_not_fit_raises_and_stays_suspended(device: _Device) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    with manager.persistent_training_allocation():
        device.allocate(40)
    # A failed allocation scope still leaves its segments in the region.
    with pytest.raises(RuntimeError, match="accumulation failed"):
        with manager.persistent_training_allocation():
            device.allocate(24)
            raise RuntimeError("accumulation failed")
    manager.suspend_training()
    device.free = 63

    with pytest.raises(ResidencyRestoreError, match="restoring training memory") as raised:
        manager.resume_training(one_rank)

    assert raised.value.required_bytes == 64
    assert "resume(training-persistent)" not in provider.events
    device.free = 64
    manager.resume_training(one_rank)
    assert provider.events[-1] == "resume(training-persistent)"


def test_persistent_training_allocation_reuses_one_no_backup_region() -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)

    with manager.persistent_training_allocation():
        pass
    with manager.persistent_training_allocation():
        pass

    assert provider.events == [
        "region(training-persistent, backup=False)+",
        "region(training-persistent)-",
        "region(training-persistent, backup=False)+",
        "region(training-persistent)-",
    ]


def test_transient_training_allocation_gets_a_fresh_pool_in_every_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    pools = _allocator_pools(monkeypatch, provider.events)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: provider.events.append("empty_cache"))

    for _window in range(2):
        manager.resume_training(one_rank)
        for _microbatch in range(2):
            with manager.transient_training_allocation(0):
                pass
        manager.suspend_training()

    def window(pool: int, *, first: bool) -> list[str]:
        microbatch = [
            "region(training-transient, backup=False)+",
            f"use(pool-{pool})+",
            f"use(pool-{pool})-",
            "region(training-transient)-",
        ]
        return [
            "empty_cache",
            *([] if first else ["resume(training-persistent)"]),
            # The thread's cuBLAS workspace, which lives as long as the process, is bound
            # before any allocation can reach the window's pool.
            "cublas-workspace",
            f"pool-{pool}+",
            *microbatch,
            "cublas-workspace",
            *microbatch,
            # The window's pool is gone before the allocator empties its cache, and the
            # transient region itself is never paused.
            f"pool-{pool}-",
            "pause(training-persistent)",
            "empty_cache",
        ]

    assert provider.events == [*window(0, first=True), *window(1, first=False)]
    assert pools.live == []


def test_a_pool_that_leaves_too_little_free_memory_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
    device: _Device,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    pools = _allocator_pools(monkeypatch, provider.events)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: provider.events.append("empty_cache"))
    manager.resume_training(one_rank)
    device.free = 1 << 30

    for _scope in range(2):
        with manager.transient_training_allocation(1 << 30):
            pass
    # Segments the pool cached for earlier shapes now leave less than a scope needs.
    device.free = (1 << 30) - 1
    with manager.transient_training_allocation(1 << 30):
        pass

    def scope(pool: int) -> list[str]:
        return [
            "region(training-transient, backup=False)+",
            f"use(pool-{pool})+",
            f"use(pool-{pool})-",
            "region(training-transient)-",
        ]

    assert provider.events[1:] == [
        "cublas-workspace",
        "pool-0+",
        *scope(0),
        "cublas-workspace",
        *scope(0),
        "cublas-workspace",
        # The old pool is gone before the allocator frees its segments.
        "pool-0-",
        "empty_cache",
        "pool-1+",
        *scope(1),
    ]
    assert pools.live == [1]


def test_persistent_allocation_opens_inside_a_transient_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    pools = _allocator_pools(monkeypatch, provider.events)

    with manager.transient_training_allocation(0):
        with manager.persistent_training_allocation():
            with pytest.raises(
                RuntimeError,
                match="Training memory is allocating, expected resident",
            ):
                manager.suspend_training()
        with pytest.raises(RuntimeError, match="Training memory is allocating"):
            manager.resume_training(one_rank)

    assert provider.events == [
        "cublas-workspace",
        "pool-0+",
        "region(training-transient, backup=False)+",
        "use(pool-0)+",
        "region(training-persistent, backup=False)+",
        "region(training-persistent)-",
        "use(pool-0)-",
        "region(training-transient)-",
    ]
    assert pools.live == [0]


@pytest.mark.parametrize("region", ["persistent", "transient"])
def test_training_allocation_rejects_reentering_one_region(
    monkeypatch: pytest.MonkeyPatch,
    region: str,
) -> None:
    manager = MemoryManager(_MemorySaver(), unified=False)
    _allocator_pools(monkeypatch, [])
    allocation = _training_allocation(manager, region)

    with allocation():
        with pytest.raises(
            RuntimeError,
            match=f"Training region training-{region} is already allocating",
        ):
            with allocation():
                pass

    with allocation():
        pass


@pytest.mark.parametrize("region", ["persistent", "transient"])
def test_training_body_failure_preserves_the_resident_regions(
    monkeypatch: pytest.MonkeyPatch,
    region: str,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    pools = _allocator_pools(monkeypatch, provider.events)
    allocation = _training_allocation(manager, region)
    body_error = RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed") as raised:
        with allocation():
            raise body_error

    assert raised.value is body_error
    with allocation():
        pass
    assert provider.events.count(f"region(training-{region})-") == 2
    assert pools.live == ([0] if region == "transient" else [])


def test_unallocated_training_resume_flushes_allocator_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: provider.events.append("empty_cache"))

    manager.suspend_training()
    manager.resume_training(one_rank)

    assert provider.events == ["empty_cache"]


def test_training_suspend_and_resume_are_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    with manager.persistent_training_allocation():
        pass
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: provider.events.append("empty_cache"))

    manager.suspend_training()
    manager.suspend_training()
    manager.resume_training(one_rank)
    manager.resume_training(one_rank)

    assert provider.events == [
        "region(training-persistent, backup=False)+",
        "region(training-persistent)-",
        "pause(training-persistent)",
        "empty_cache",
        "empty_cache",
        "resume(training-persistent)",
    ]


@pytest.mark.parametrize("region", ["persistent", "transient"])
def test_training_allocation_requires_resident_training_memory(
    monkeypatch: pytest.MonkeyPatch,
    region: str,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    _allocator_pools(monkeypatch, provider.events)
    allocation = _training_allocation(manager, region)
    with allocation():
        pass
    manager.suspend_training()
    events = provider.events.copy()

    with pytest.raises(RuntimeError, match="Training memory is suspended"):
        with allocation():
            pass

    assert provider.events == events
    manager.resume_training(one_rank)
    with allocation():
        pass


def test_kv_and_training_residency_transition_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _MemorySaver()
    manager = MemoryManager(provider, unified=False)
    _allocate(manager)
    with manager.persistent_training_allocation():
        pass
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: provider.events.append("empty_cache"))

    manager.suspend_training()
    manager.suspend_kv()
    manager.resume_training(one_rank)
    manager.resume_kv(one_rank)

    assert provider.events[-7:] == [
        "pause(training-persistent)",
        "empty_cache",
        "pause(kv-cache)",
        "empty_cache",
        "resume(training-persistent)",
        "empty_cache",
        "resume(kv-cache)",
    ]


@pytest.mark.parametrize("fails", [False, True])
def test_each_graph_group_shares_one_workspace_until_the_pool_closes(
    monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    import ctikernels
    from cti.memory import GraphPool

    events: list[str] = []

    class _Workspace:
        def __init__(self, *, max_rows: int) -> None:
            self.max_rows = max_rows

        def __enter__(self) -> None:
            events.append(f"enter {self.max_rows}")

        def __exit__(self, *exc: object) -> None:
            events.append(f"exit {self.max_rows}")

        def close(self) -> None:
            events.append(f"close {self.max_rows}")

    monkeypatch.setattr(ctikernels, "GraphWorkspace", _Workspace, raising=False)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    pool = GraphPool()

    with pool.workspace(4):
        events.append("capture")
    with pytest.raises(RuntimeError) if fails else nullcontext():
        with pool.workspace(512):
            events.append("capture")
            if fails:
                raise RuntimeError("capture failed")
    assert events == ["enter 4", "capture", "exit 4", "enter 512", "capture", "exit 512"]

    # The graphs hold their group's workspace; the pool's references end with it.
    pool.close()
    assert events[-2:] == ["close 4", "close 512"]
