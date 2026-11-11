from __future__ import annotations

import os
from bisect import bisect_left, insort
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Iterator, Protocol

from cti.utils import mem_GB

if TYPE_CHECKING:
    import torch

    from ctikernels import GraphWorkspace


class Mode(Enum):
    """Who holds the device's shared memory: serving's KV pool or training's state."""

    SERVING = auto()
    TRAINING = auto()


class ResidencyRestoreError(RuntimeError):
    """Suspended memory does not fit the device's free memory; it stays suspended.

    Raised before the provider maps anything, so the owner remains healthy and the caller
    may keep the current residency.
    """

    def __init__(self, region: str, required_bytes: int, free_bytes: int) -> None:
        super().__init__(region, required_bytes, free_bytes)
        self.region = region
        self.required_bytes = required_bytes
        self.free_bytes = free_bytes

    def __str__(self) -> str:
        return (
            f"restoring {self.region} memory needs {mem_GB(self.required_bytes)}, but the "
            f"device has {mem_GB(self.free_bytes)} free"
        )


class MemoryProvider(Protocol):
    """What places device memory in tagged regions it can unmap and map again: the
    process-wide torch_memory_saver in an engine, a stand-in in the CPU suite."""

    def region(
        self,
        tag: str,
        enable_cpu_backup: bool = False,
    ) -> AbstractContextManager[None]: ...

    def pause(self, tag: str) -> None: ...

    def resume(self, tag: str) -> None: ...


# Takes the bytes this device keeps free after a resume; returns the fewest any rank keeps.
Fewest = Callable[[int], int]


def unified_memory(device: torch.device) -> bool:
    """Whether ``device`` shares the host's memory (GB10)."""

    import torch

    return device.type == "cuda" and bool(torch.cuda.get_device_properties(device).is_integrated)


def free_bytes(device: torch.device | None = None, *, unified: bool) -> int:
    """The bytes ``device`` (``None``: the current one) has free. Unified memory's are the
    host's ``MemAvailable``. The device's own reading there is ``MemFree``, which counts the
    page cache, reclaimed as the device allocates, as used: after a checkpoint load it leaves
    out the tens of GiB the load only cached. The memory plan and every later fit check read
    free memory this one way."""

    import torch

    return _host_available_bytes() if unified else torch.cuda.mem_get_info(device)[0]


def _host_available_bytes() -> int:
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable")


def _require_restorable(region: str, required_bytes: int, fewest: Fewest, unified: bool) -> None:
    """Refuse a resume the device cannot back: the provider exits the process when it
    cannot map a resumed allocation. Every rank refuses when any rank's device cannot, as
    ``fewest`` agrees, so that all keep the same residency. The check holds while no other
    process allocates on the device between it and the resume."""

    import torch

    torch.cuda.empty_cache()
    spare = fewest(free_bytes(unified=unified) - required_bytes)
    if spare < 0:
        raise ResidencyRestoreError(region, required_bytes, required_bytes + spare)


@contextmanager
def _measured(record: Callable[[int], None]) -> Iterator[None]:
    """Record the segment bytes the allocator maps while the scope runs, even when it
    fails: the region keeps them."""

    import torch

    before = torch.cuda.memory_reserved()
    try:
        yield
    finally:
        record(max(torch.cuda.memory_reserved() - before, 0))


class _ResidencyState(Enum):
    UNALLOCATED = auto()
    ALLOCATING = auto()
    RESIDENT = auto()
    SUSPENDED = auto()


class _Residency:
    """A residency controller's state and the one guard on its transitions."""

    _MEMORY: ClassVar[str]  # the memory it names in errors
    _residency: _ResidencyState

    def _require_residency(self, *expected: _ResidencyState) -> None:
        if self._residency not in expected:
            names = " or ".join(state.name.lower() for state in expected)
            raise RuntimeError(
                f"{self._MEMORY} memory is {self._residency.name.lower()}, expected {names}"
            )


class _KVResidency(_Residency):
    """Own physical residency for the inference state's one no-backup region."""

    _MEMORY = "KV"
    _TAG = "kv-cache"

    def __init__(self, provider: MemoryProvider) -> None:
        self._provider = provider
        self._residency = _ResidencyState.UNALLOCATED
        # Physical bytes the region maps.
        self.mapped_bytes = 0

    @contextmanager
    def allocation(self) -> Iterator[None]:
        """Capture the inference state's allocations in the region."""

        self._require_residency(_ResidencyState.UNALLOCATED)

        def record(nbytes: int) -> None:
            self.mapped_bytes += nbytes

        self._residency = _ResidencyState.ALLOCATING
        with self._provider.region(self._TAG, enable_cpu_backup=False), _measured(record):
            yield
        self._residency = _ResidencyState.RESIDENT

    def suspend(self) -> None:
        if self._residency is _ResidencyState.SUSPENDED:
            return
        self._require_residency(_ResidencyState.RESIDENT)
        self._provider.pause(self._TAG)
        self._residency = _ResidencyState.SUSPENDED

    @property
    def suspended(self) -> bool:
        return self._residency is _ResidencyState.SUSPENDED

    @property
    def suspended_bytes(self) -> int:
        """Physical bytes the suspended region maps again on resume."""

        return self.mapped_bytes if self.suspended else 0

    def resume(self) -> None:
        """Map the suspended region again; the caller checked it fits."""

        if self._residency is _ResidencyState.RESIDENT:
            return
        self._require_residency(_ResidencyState.SUSPENDED)
        self._provider.resume(self._TAG)
        self._residency = _ResidencyState.RESIDENT


class _GraphResidency:
    """The inference graph pool's one no-backup region: every capture maps its memory there
    and the KV suspension pauses it. Replays write what they read, so none is recaptured."""

    _TAG = "inference-graph"

    def __init__(self, provider: MemoryProvider) -> None:
        self._provider = provider
        self.mapped_bytes = 0
        self._suspended = False

    @contextmanager
    def allocation(self) -> Iterator[None]:
        def record(nbytes: int) -> None:
            self.mapped_bytes += nbytes

        with self._provider.region(self._TAG, enable_cpu_backup=False), _measured(record):
            yield

    def suspend(self) -> None:
        if self.mapped_bytes and not self._suspended:
            self._provider.pause(self._TAG)
            self._suspended = True

    @property
    def suspended_bytes(self) -> int:
        return self.mapped_bytes if self._suspended else 0

    def resume(self) -> None:
        """Map the region again; the caller checked it fits."""

        if self._suspended:
            self._provider.resume(self._TAG)
            self._suspended = False


class _TrainingRegion(Enum):
    """Training memory split by lifetime; each value is the region's provider tag."""

    PERSISTENT = "training-persistent"
    TRANSIENT = "training-transient"
    OPTIMIZER = "training-optimizer"


class _TrainingResidency(_Residency):
    """Own physical residency for training allocations, split by lifetime.

    The persistent region is one reusable no-backup region, suspended and resumed in
    place, that holds the FP32 gradients, a logical batch's folded loss evidence, the
    state restored after a suspension and checkpoint temporaries. The optimizer region,
    used only under ``TrainingStateResidency.OFFLOAD_ALL``, holds the FP32 masters and
    optimizer moments, allocated once and suspended and resumed with it.

    The transient region gives each training window a fresh allocator pool for what one
    microbatch or optimizer step allocates on the training thread: activations,
    recomputation and backward temporaries, the checkpoint inputs ``full_offload``
    fetches back from host memory and the step's temporaries. The offload's pinned host
    arena is not device memory and stays out of every region. A live pool never frees a
    cached segment, so the pool is dropped, its segments freed, whenever it leaves less
    free memory than a scope needs, and at suspension: one reused pool would otherwise
    map every shape so far.
    """

    _MEMORY = "Training"

    def __init__(self, provider: MemoryProvider, *, unified: bool) -> None:
        self._provider, self._unified = provider, unified
        self._residency = _ResidencyState.UNALLOCATED
        self._allocating: set[_TrainingRegion] = set()
        self._transient_pool: torch.cuda.MemPool | None = None
        # Physical bytes the persistent and optimizer regions map, the size their resume
        # must find free.
        self._persistent_bytes = 0
        self._optimizer_bytes = 0
        # Whether the last suspension paused the persistent region: not while training keeps
        # its gradients on the device.
        self._persistent_paused = False

    def persistent_allocation(self) -> AbstractContextManager[None]:
        return self._allocation(_TrainingRegion.PERSISTENT, self._persistent_scope)

    def transient_allocation(self, needed_bytes: int) -> AbstractContextManager[None]:
        return self._allocation(
            _TrainingRegion.TRANSIENT, partial(self._transient_scope, needed_bytes)
        )

    def optimizer_allocation(self) -> AbstractContextManager[None]:
        return self._allocation(_TrainingRegion.OPTIMIZER, self._optimizer_scope)

    def suspend(self, *, keep_persistent: bool) -> None:
        if self._residency in (
            _ResidencyState.UNALLOCATED,
            _ResidencyState.SUSPENDED,
        ):
            return
        self._require_residency(_ResidencyState.RESIDENT)
        import torch

        # Drop the window's pool so the allocator frees its segments. The transient
        # region is never paused: the allocator frees them in empty_cache below and
        # whenever a block that outlived its window dies, and the provider cannot free a
        # paused allocation (torch_memory_saver@6d5bce4 unmaps it again and exits).
        self._transient_pool = None
        self._persistent_paused = not keep_persistent
        if self._persistent_paused:
            self._provider.pause(_TrainingRegion.PERSISTENT.value)
        if self._optimizer_bytes:
            self._provider.pause(_TrainingRegion.OPTIMIZER.value)
        torch.cuda.empty_cache()
        self._residency = _ResidencyState.SUSPENDED

    def resume(self, fewest: Fewest) -> None:
        if self._residency is _ResidencyState.RESIDENT:
            return
        self._require_residency(
            _ResidencyState.UNALLOCATED,
            _ResidencyState.SUSPENDED,
        )
        persistent = self._persistent_bytes if self._persistent_paused else 0
        _require_restorable("training", persistent + self._optimizer_bytes, fewest, self._unified)
        if self._residency is _ResidencyState.SUSPENDED:
            if self._persistent_paused:
                self._provider.resume(_TrainingRegion.PERSISTENT.value)
            if self._optimizer_bytes:
                self._provider.resume(_TrainingRegion.OPTIMIZER.value)
            self._residency = _ResidencyState.RESIDENT

    @contextmanager
    def _allocation(
        self,
        region: _TrainingRegion,
        open_scope: Callable[[], AbstractContextManager[None]],
    ) -> Iterator[None]:
        """Capture allocations in one region; the other region may be open around it."""

        if self._allocating:
            self._require_residency(_ResidencyState.ALLOCATING)
        else:
            self._require_residency(
                _ResidencyState.UNALLOCATED,
                _ResidencyState.RESIDENT,
            )
        if region in self._allocating:
            raise RuntimeError(f"Training region {region.value} is already allocating")
        self._residency = _ResidencyState.ALLOCATING
        self._allocating.add(region)
        try:
            with open_scope():
                yield
        finally:
            # A failed training body leaves the regions reusable.
            self._allocating.discard(region)
            if not self._allocating:
                self._residency = _ResidencyState.RESIDENT

    @contextmanager
    def _transient_scope(self, needed_bytes: int) -> Iterator[None]:
        import torch

        # torch keeps one cuBLAS workspace per thread and stream for the life of the
        # process and allocates it at their first matmul (ATen/cuda/CublasHandlePool.cpp).
        # Bind it here, outside the window's pool: a block that outlives the pool would
        # keep the segment it was carved from mapped after suspension.
        torch.cuda.current_blas_handle()
        if self._transient_pool is not None and free_bytes(unified=self._unified) < needed_bytes:
            # The pool keeps the segments earlier shapes mapped, which this scope may not
            # reuse; dropping it frees them for the memory the scope was planned.
            self._transient_pool = None
            torch.cuda.empty_cache()
        if self._transient_pool is None:
            self._transient_pool = torch.cuda.MemPool()
        with self._provider.region(_TrainingRegion.TRANSIENT.value, enable_cpu_backup=False):
            # Allocations route to the innermost pool; the enclosing region's thread-local
            # tag still places every segment this pool maps in the provider's region.
            with torch.cuda.use_mem_pool(self._transient_pool):
                yield

    @contextmanager
    def _persistent_scope(self) -> Iterator[None]:
        def record(nbytes: int) -> None:
            self._persistent_bytes += nbytes

        with (
            self._provider.region(_TrainingRegion.PERSISTENT.value, enable_cpu_backup=False),
            _measured(record),
        ):
            yield

    @contextmanager
    def _optimizer_scope(self) -> Iterator[None]:
        def record(nbytes: int) -> None:
            self._optimizer_bytes += nbytes

        with (
            self._provider.region(_TrainingRegion.OPTIMIZER.value, enable_cpu_backup=False),
            _measured(record),
        ):
            yield


class MemoryManager:
    """Compose rank-local physical residency controllers. A failure other than a refused
    restore (``ResidencyRestoreError``) ends the rank's process."""

    def __init__(self, provider: MemoryProvider, *, unified: bool) -> None:
        """``unified``: the device shares the host's memory, whose free bytes ``free_bytes``
        reads."""

        self._unified = unified
        self._kv = _KVResidency(provider)
        self._graphs = _GraphResidency(provider)
        self._training = _TrainingResidency(provider, unified=unified)

    @classmethod
    def from_stock_provider(cls, *, unified: bool) -> MemoryManager:
        """Borrow the process-wide provider without initializing it eagerly."""

        from torch_memory_saver import torch_memory_saver

        return cls(torch_memory_saver, unified=unified)

    def kv_allocation(self) -> AbstractContextManager[None]:
        """Capture the inference state in the no-backup region the KV suspension releases."""

        return self._kv.allocation()

    def graph_allocation(self) -> AbstractContextManager[None]:
        """Capture inference graph memory in the no-backup region that the KV suspension
        also releases."""

        return self._graphs.allocation()

    def persistent_training_allocation(self) -> AbstractContextManager[None]:
        """Capture FP32 gradients, loss evidence and restored state in the reusable
        no-backup region.

        Suspension keeps the region's addresses but not its contents: callers move what
        must survive, such as an open gradient accumulation, off the device first.
        """

        return self._training.persistent_allocation()

    def transient_training_allocation(self, needed_bytes: int) -> AbstractContextManager[None]:
        """Capture what one microbatch or optimizer step allocates on this thread in the
        window's pool, first replacing a pool that leaves less than ``needed_bytes`` free.

        A persistent allocation may open inside this scope. What outlives the scope
        belongs in the persistent region: the pool may be replaced at the next scope.
        """

        return self._training.transient_allocation(needed_bytes)

    def optimizer_training_allocation(self) -> AbstractContextManager[None]:
        """Capture the FP32 masters and optimizer moments in their own no-backup region,
        suspended and resumed with the persistent one (``TrainingStateResidency.OFFLOAD_ALL``).

        Suspension keeps the region's addresses but not its contents: the optimizer's host
        mirror brings them back.
        """

        return self._training.optimizer_allocation()

    def suspend_kv(self) -> None:
        """Release physical KV and graph memory after callers quiesce every KV-using stream."""

        self._kv.suspend()
        self._graphs.suspend()

    @property
    def kv_bytes(self) -> int:
        """Physical bytes suspending the KV releases: the recurrent and KV pools and the
        graph pool."""

        return self._kv.mapped_bytes + self._graphs.mapped_bytes

    def resume_kv(self, fewest: Fewest) -> None:
        """Restore physical KV memory; its previous contents remain undefined.

        Raises ``ResidencyRestoreError`` and keeps the KV and graph memory suspended when
        together they do not fit this device or, as ``fewest`` agrees, a peer rank's.
        """

        # The graph region is suspended only with the KV.
        if self._kv.suspended:
            _require_restorable(
                "KV", self._kv.suspended_bytes + self._graphs.suspended_bytes, fewest, self._unified
            )
        self._graphs.resume()
        self._kv.resume()

    def suspend_training(self, *, keep_persistent: bool = False) -> None:
        """Free transient training memory and suspend the optimizer region and, unless
        ``keep_persistent``, the persistent one, after training's state settles."""

        self._training.suspend(keep_persistent=keep_persistent)

    def resume_training(self, fewest: Fewest) -> None:
        """Restore the persistent training region; transient memory starts each window empty.

        Raises ``ResidencyRestoreError`` and keeps the region suspended when it does not fit
        this device or, as ``fewest`` agrees, a peer rank's.
        """

        self._training.resume(fewest)


class GraphPool:
    """A memory pool inference CUDA graphs capture into: one for decode graphs, breakable
    graphs and DFlash rounds, one of their own for vision graphs. A pool's graphs replay one
    at a time on the execution stream and copy what must outlive a replay out of the pool,
    so they share its blocks.

    Captures map the pool's memory inside ``region`` (``MemoryManager.graph_allocation``),
    which the KV suspension releases while training holds the device."""

    def __init__(self, region: Callable[[], AbstractContextManager[None]] = nullcontext) -> None:
        import torch

        self._handle = torch.cuda.graph_pool_handle()
        self.region = region
        self._workspaces: list[GraphWorkspace] = []

    @contextmanager
    def capture(self, graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> Iterator[None]:
        """Capture ``graph`` on ``stream`` into this pool."""

        import torch

        with self.region(), torch.cuda.graph(graph, pool=self._handle, stream=stream):
            yield

    @contextmanager
    def workspace(self, max_rows: int) -> Iterator[None]:
        """Warm up and capture one group of graphs inside, of at most ``max_rows`` rows each,
        in any order: the kernels' graph workspace, sized for every call up to ``max_rows``,
        is shared by the group's graphs rather than held by each, until the pool closes."""

        from ctikernels import GraphWorkspace

        workspace = GraphWorkspace(max_rows=max_rows)
        self._workspaces.append(workspace)
        with workspace:
            yield

    def close(self) -> None:
        """Release the groups' workspaces; their memory returns once their graphs are gone."""

        for workspace in self._workspaces:
            workspace.close()
        self._workspaces.clear()

    def begin(self, graph: torch.cuda.CUDAGraph) -> None:
        """Begin capturing ``graph`` on the current stream; ``end`` finishes it (breakable
        graphs' segments, all inside one ``region()``)."""

        graph.capture_begin(pool=self._handle)

    @staticmethod
    def end(graph: torch.cuda.CUDAGraph) -> None:
        graph.capture_end()


@dataclass(frozen=True, slots=True)
class _NativeBacking:
    """The native CUDA caching allocator's placement, as the inference state's and
    training's memory plans model it: the 512-byte blocks it rounds requests to, the segment
    it maps for a block (1 MiB and below from 2 MiB segments, below 10 MiB from 20 MiB ones,
    larger ones from their own rounded up to 2 MiB), and the allocator options under which
    it places blocks otherwise. Reads the allocator configuration only when asked."""

    MIB: ClassVar[int] = 1024**2
    BLOCK_BYTES: ClassVar[int] = 512
    SMALL_BLOCK_BYTES: ClassVar[int] = MIB
    SMALL_SEGMENT_BYTES: ClassVar[int] = 2 * MIB
    LARGE_SEGMENT_BYTES: ClassVar[int] = 20 * MIB
    # From this block size on, an allocation takes a segment of its own rather than a 20 MiB
    # one; from this request size on, as requests round up to blocks first.
    LARGE_ALLOCATION_BYTES: ClassVar[int] = 10 * MIB
    OWN_SEGMENT_REQUEST_BYTES: ClassVar[int] = LARGE_ALLOCATION_BYTES - BLOCK_BYTES + 1
    # Allocator options that change how blocks are rounded, split or mapped, with the value
    # that keeps the native placement modelled here (None: any value changes it).
    PLACEMENT_OPTIONS: ClassVar[Mapping[str, str | None]] = MappingProxyType(
        {
            "backend": "native",
            "expandable_segments": "false",
            "large_segment_size_mb": "20",
            "max_non_split_rounding_mb": None,
            "max_split_size_mb": None,
            "roundup_power2_divisions": None,
        }
    )

    def segment(self, nbytes: int) -> int:
        """The segment bytes one request of ``nbytes`` maps alone; 0 for none. PyTorch's
        ``DeviceCachingAllocator::malloc`` rounds the request (``round_size``) before it
        picks the pool and the allocation size (``get_allocation_size``)."""

        size = self._block(nbytes)
        if size == 0:
            return 0
        if size <= self.SMALL_BLOCK_BYTES:
            return self.SMALL_SEGMENT_BYTES
        if size < self.LARGE_ALLOCATION_BYTES:
            return self.LARGE_SEGMENT_BYTES
        return -(-size // self.SMALL_SEGMENT_BYTES) * self.SMALL_SEGMENT_BYTES

    def _block(self, payload: int) -> int:
        """The block a request of ``payload`` bytes takes."""

        return -(-payload // self.BLOCK_BYTES) * self.BLOCK_BYTES

    def pooled(self, payloads: Iterable[int]) -> int:
        """Segment bytes the allocator maps for ``payloads`` allocated in order from an empty
        pool and all kept: each block takes the smallest free remainder that holds it, and a
        remainder stays free only past the split threshold (512 B small, 1 MiB large)."""

        free: dict[bool, list[int]] = {True: [], False: []}
        reserved = 0
        for payload in payloads:
            if payload == 0:
                continue
            size = self._block(payload)
            small = size <= self.SMALL_BLOCK_BYTES
            remainders = free[small]
            index = bisect_left(remainders, size)
            if index < len(remainders):
                block = remainders.pop(index)
            else:
                block = self.segment(size)
                reserved += block
            if block - size >= (self.BLOCK_BYTES if small else self.SMALL_BLOCK_BYTES + 1):
                insort(remainders, block - size)
        return reserved

    def custom_placement_options(self) -> tuple[str, ...]:
        """The configured allocator options (``PYTORCH_ALLOC_CONF``, else
        ``PYTORCH_CUDA_ALLOC_CONF``) under which blocks are not placed as modelled here."""

        config = os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", ""
        )
        options = (entry.split(":", 1) for entry in config.split(",") if ":" in entry)
        placement = self.PLACEMENT_OPTIONS
        return tuple(
            sorted(
                name
                for name, value in (
                    (name.strip(), value.strip().lower()) for name, value in options
                )
                if name in placement and value != placement[name]
            )
        )


native_backing = _NativeBacking()


@contextmanager
def configure_memory_saver_subprocess() -> Iterator[None]:
    """Apply the stock provider's preload selection only while spawning a child."""

    from torch_memory_saver import configure_subprocess

    previous_preload = os.environ.get("LD_PRELOAD")
    try:
        with configure_subprocess():
            yield
    finally:
        # The stock context restores an absent variable as an empty string; leave
        # the spawning parent's environment exactly as it was.
        if previous_preload is None:
            os.environ.pop("LD_PRELOAD", None)


__all__ = [
    "GraphPool",
    "MemoryManager",
    "MemoryProvider",
    "Mode",
    "ResidencyRestoreError",
    "configure_memory_saver_subprocess",
    "free_bytes",
    "native_backing",
    "unified_memory",
]
