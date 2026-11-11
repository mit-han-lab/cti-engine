"""Training's pinned host memory: one owner per runner, within a share of the host's memory."""

from __future__ import annotations

import mmap
import weakref
from pathlib import Path

import torch


class PinnedHostMemory:
    """Pin the host memory one runner holds across microbatches and switches: the
    ``full_offload`` arena and the gradient stash. Each allocation is checked against the
    host's available memory and, with what this owner already pins, a share of its total
    before anything is mapped. ``close`` unregisters every allocation, as collecting the
    owner unclosed does."""

    # The share of the host's memory one runner may pin.
    HOST_MEMORY_SHARE = 0.7

    def __init__(self) -> None:
        self._nbytes = 0
        self._arenas: list[torch.Tensor] = []
        self._release = weakref.finalize(self, PinnedHostMemory._unregister, self._arenas)
        # Process exit releases the mappings anyway, and CUDA may be torn down by then.
        self._release.atexit = False

    @property
    def nbytes(self) -> int:
        """The bytes this owner pins."""

        return self._nbytes

    def allocate(self, nbytes: int, purpose: str) -> torch.Tensor:
        """``nbytes`` of page-aligned pinned host memory as bytes, for ``purpose``."""

        if not nbytes:
            return torch.empty(0, dtype=torch.uint8)
        total, available = _host_memory()
        limit = int(self.HOST_MEMORY_SHARE * total)
        if self._nbytes + nbytes > limit or nbytes > available:
            gib = 1 << 30
            raise ValueError(
                f"{purpose} needs {nbytes / gib:.2f} GiB of pinned host memory; this runner "
                f"already pins {self._nbytes / gib:.2f} GiB of at most {limit / gib:.2f} GiB "
                f"({self.HOST_MEMORY_SHARE:.0%} of host memory), and "
                f"{available / gib:.2f} GiB is available"
            )
        mapping = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        # The tensor keeps the mapping alive; registering faults in and locks its pages.
        arena = torch.frombuffer(mapping, dtype=torch.uint8)
        _register_host_memory(arena.data_ptr(), nbytes)
        self._arenas.append(arena)
        self._nbytes += nbytes
        return arena

    def close(self) -> None:
        """Unregister every allocation; their tensors must no longer be used."""

        self._release()
        self._nbytes = 0

    @staticmethod
    def _unregister(arenas: list[torch.Tensor]) -> None:
        for arena in arenas:
            _unregister_host_memory(arena.data_ptr())
        arenas.clear()


def _host_memory() -> tuple[int, int]:
    """The host's total and available memory in bytes, from ``/proc/meminfo``."""

    fields = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if line
    )
    return tuple(int(fields[name].split()[0]) * 1024 for name in ("MemTotal", "MemAvailable"))


def _register_host_memory(address: int, nbytes: int) -> None:
    torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(address, nbytes, 0))


def _unregister_host_memory(address: int) -> None:
    torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(address))


__all__ = ["PinnedHostMemory"]
