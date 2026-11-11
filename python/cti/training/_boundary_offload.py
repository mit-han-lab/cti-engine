"""Byte-exact checkpoint input offload to pinned host memory on CUDA.

Inputs must be contiguous, match the configured dtype and width, and fit the token
capacity. All but the last input wait in the arena until backward. In-place writes,
out-of-order or repeated reads, and undrained microbatches raise ``RuntimeError``.
After failure, call ``abort`` before reuse. ``close`` drains copies and drops the
arena, which the runner's pinned host memory owns; it persists across microbatches and
inference windows.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from ._host_memory import PinnedHostMemory


@dataclass(eq=False, slots=True)
class _SavedInput:
    """The handle autograd holds for one checkpointed layer's saved input."""

    index: int
    shape: torch.Size
    device: torch.device
    version: int
    data_ptr: int
    # Kept until the copy out completes, or backward reads the last input.
    tensor: torch.Tensor | None
    # An offloaded input's device copy, fetched while the layer above runs backward.
    fetched: torch.Tensor | None = None
    released: bool = False

    def release(self) -> None:
        self.tensor = None
        self.fetched = None
        self.released = True


class HostBoundaryOffload:
    """Hold each checkpointed layer's input but the last in a pinned host arena."""

    def __init__(
        self,
        *,
        segments: int,
        tokens: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        host: PinnedHostMemory,
    ) -> None:
        """The arena, one slot per offloaded layer, is pinned from ``host``."""

        if device.type != "cuda":
            raise ValueError("full_offload activation checkpointing requires a CUDA device")
        slots = max(segments - 1, 0)
        slot_numel = tokens * width
        nbytes = slots * slot_numel * dtype.itemsize
        arena = host.allocate(nbytes, "full_offload")

        self._segments = segments
        self._tokens = tokens
        self._width = width
        self._dtype = dtype
        self._slot_numel = slot_numel
        self._device = device
        self._nbytes = nbytes
        self._copy_stream = torch.cuda.Stream(device=device)
        self._ready = tuple(torch.cuda.Event() for _ in range(slots))
        self._done = tuple(torch.cuda.Event() for _ in range(slots))
        self._saved: list[_SavedInput] = []
        self._expected: torch.Tensor | None = None
        self._slots: tuple[torch.Tensor, ...] = (
            arena.view(dtype).view(slots, slot_numel).unbind(0) if nbytes else ()
        )

    @property
    def nbytes(self) -> int:
        """The arena's size: one slot per offloaded layer."""

        return self._nbytes

    @contextmanager
    def saving(self) -> Iterator[None]:
        """Offload the inputs ``expect`` declares while the checkpointed layers run forward."""

        self.require_drained()
        self._saved = []
        self._expected = None
        with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
            yield

    def expect(self, tensor: torch.Tensor) -> None:
        """Declare the input the next checkpoint saves."""

        self._expected = tensor

    def require_drained(self) -> None:
        """Require backward to have read every saved input: none outlives its microbatch."""

        if any(not saved.released for saved in self._saved):
            raise RuntimeError("a checkpointed layer's saved input outlived its microbatch")

    def abort(self) -> None:
        """Drop the saved inputs and prefetches of a failed microbatch."""

        # A copy may still read an input or write a prefetch.
        self._copy_stream.synchronize()
        for saved in self._saved:
            saved.release()
        self._saved = []
        self._expected = None

    def close(self) -> None:
        """Drop everything saved and the arena; the offload is unusable after."""

        self.abort()
        self._slots = ()

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor | _SavedInput:
        if tensor is not self._expected:
            return tensor
        self._expected = None
        index = len(self._saved)
        self._require_fits(tensor)
        saved = _SavedInput(
            index=index,
            shape=tensor.shape,
            device=tensor.device,
            version=tensor._version,
            data_ptr=tensor.data_ptr(),
            tensor=tensor,
        )
        # Listed before its copy starts, so abort reaches it whatever fails next.
        self._saved.append(saved)
        producer = torch.cuda.current_stream(self._device)
        if index < self._segments - 1:
            self._copy_out(saved, producer)
        if index:
            # The previous input's copy overlapped this layer; free it once done.
            previous = self._saved[index - 1]
            producer.wait_event(self._done[index - 1])
            self._require_unchanged(previous)
            previous.tensor = None
        return saved

    def _unpack(self, saved: torch.Tensor | _SavedInput) -> torch.Tensor:
        if not isinstance(saved, _SavedInput):
            return saved
        consumer = torch.cuda.current_stream(self._device)
        if saved.index == self._segments - 1:
            self._require_unchanged(saved)
            tensor = saved.tensor
        else:
            consumer.wait_event(self._done[saved.index])
            tensor = cast(torch.Tensor, saved.fetched)
        saved.release()
        if saved.index:
            # Fetch the layer below while this one recomputes and runs backward.
            self._copy_in(self._saved[saved.index - 1], consumer)
        return tensor

    def _copy_out(self, saved: _SavedInput, producer: torch.cuda.Stream) -> None:
        """Copy an input to its slot once ``producer`` has written it."""

        tensor = saved.tensor
        copy = self._copy_stream
        # Waits capture the current record, so these events can be reused on fetch.
        producer.record_event(self._ready[saved.index])
        copy.wait_event(self._ready[saved.index])
        with torch.cuda.stream(copy):
            self._slots[saved.index][: tensor.numel()].copy_(tensor.view(-1), non_blocking=True)
            copy.record_event(self._done[saved.index])

    def _copy_in(self, saved: _SavedInput, consumer: torch.cuda.Stream) -> None:
        """Fetch an offloaded input back to the device for ``consumer``."""

        fetched = torch.empty(saved.shape, dtype=self._dtype, device=saved.device)
        saved.fetched = fetched
        copy = self._copy_stream
        # The fetched block may still be read by earlier work on the consumer.
        consumer.record_event(self._ready[saved.index])
        copy.wait_event(self._ready[saved.index])
        with torch.cuda.stream(copy):
            fetched.view(-1).copy_(self._slots[saved.index][: fetched.numel()], non_blocking=True)
            copy.record_event(self._done[saved.index])

    def _require_fits(self, tensor: torch.Tensor) -> None:
        if not tensor.is_contiguous():
            raise RuntimeError("a checkpointed layer's input must be contiguous")
        if tensor.dtype != self._dtype or tensor.shape[-1] != self._width:
            raise RuntimeError(
                f"a checkpointed layer's input is {tensor.dtype} of width {tensor.shape[-1]}; "
                f"the offload holds {self._dtype} of width {self._width}"
            )
        if tensor.numel() > self._slot_numel:
            raise RuntimeError(
                f"a checkpointed layer's input has {tensor.numel() // self._width} tokens; "
                f"the offload holds {self._tokens}"
            )

    @staticmethod
    def _require_unchanged(saved: _SavedInput) -> None:
        tensor = saved.tensor
        if tensor._version != saved.version or tensor.data_ptr() != saved.data_ptr:
            raise RuntimeError(
                "a checkpointed layer's input was modified in place after it was saved"
            )


__all__ = ["HostBoundaryOffload"]
