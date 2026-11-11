"""CPU stand-ins for the CUDA calls of ``HostBoundaryOffload``.

The offload copies on its own stream, ordered by events alone, and pins its arena with
``cudaHostRegister``. On CPU every copy completes when it is issued, so the stand-in
streams and events only log, in order, what the offload asks of them: the order is the
contract a GPU relies on. Registration records the address and size, and the host's
memory is a fixed value. The runner trains a CPU model, so ``FakeCuda.install`` also
makes the runner's offloads see a CUDA device and records what each was built with.
Offloads a test builds alone pin from ``FakeCuda.host``, which teardown closes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

import cti.training._host_memory as host_memory
import cti.training.runner as runner_module
from cti.training._boundary_offload import HostBoundaryOffload

if TYPE_CHECKING:
    import pytest

GIB = 1 << 30


class FakeEvent:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeStream:
    """Log each stream operation as ``(stream, operation, operand)``."""

    def __init__(
        self, name: str, log: list[tuple[str, str, str]], new_event: Callable[[], FakeEvent]
    ) -> None:
        self.name = name
        self._log = log
        self._new_event = new_event

    def wait_stream(self, other: FakeStream) -> None:
        self.wait_event(other.record_event())

    def record_event(self, event: FakeEvent | None = None) -> FakeEvent:
        if event is None:
            event = self._new_event()
        self._log.append((self.name, "record", event.name))
        return event

    def wait_event(self, event: FakeEvent) -> None:
        self._log.append((self.name, "wait_event", event.name))

    def synchronize(self) -> None:
        self._log.append((self.name, "synchronize", ""))


class FakeCuda:
    """The streams, registration and host memory the offload sees on CPU."""

    def __init__(self, *, total_bytes: int = 64 * GIB, available_bytes: int = 48 * GIB) -> None:
        self.log: list[tuple[str, str, str]] = []
        self.events: list[FakeEvent] = []
        self.training = FakeStream("training", self.log, self._new_event)
        self._current = self.training
        self.host_memory = (total_bytes, available_bytes)
        # Every registration and unregistration in order; a freed address may come back.
        self.registered: list[tuple[int, int]] = []
        self.unregistered: list[int] = []
        self._live: set[int] = set()
        # Each offload the runner built and the keywords it was built with.
        self.offloads: list[tuple[dict[str, object], HostBoundaryOffload]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeCuda:
        # The host has no CUDA but these stand-ins, with or without a GPU visible: a host
        # runner built under them takes a host stream, as on a CPU-only host.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "Stream", self._new_stream)
        monkeypatch.setattr(torch.cuda, "Event", self._new_event)
        monkeypatch.setattr(torch.cuda, "current_stream", self._current_stream)
        monkeypatch.setattr(torch.cuda, "stream", self._stream)
        monkeypatch.setattr(host_memory, "_register_host_memory", self._register)
        monkeypatch.setattr(host_memory, "_unregister_host_memory", self._unregister)
        monkeypatch.setattr(host_memory, "_host_memory", lambda: self.host_memory)
        monkeypatch.setattr(runner_module, "HostBoundaryOffload", self._runner_offload)
        self.host = host_memory.PinnedHostMemory()
        return self

    def operations(self, stream: str) -> list[tuple[str, str]]:
        return [(operation, operand) for name, operation, operand in self.log if name == stream]

    def require_released(self) -> None:
        """Every arena registered was unregistered: a test closes the runners it builds."""

        self.host.close()
        assert not self._live

    def _new_event(self) -> FakeEvent:
        event = FakeEvent(f"e{len(self.events)}")
        self.events.append(event)
        return event

    def _new_stream(self, device: torch.device) -> FakeStream:
        assert device.type == "cuda"
        return FakeStream("copy", self.log, self._new_event)

    def _current_stream(self, device: torch.device | None = None) -> FakeStream:
        return self._current

    @contextmanager
    def _stream(self, stream: FakeStream | object) -> Iterator[None]:
        if not isinstance(stream, FakeStream):
            # A host runner's stream orders nothing: work stays on the training stream.
            yield
            return
        previous = self._current
        self._current = stream
        self.log.append((stream.name, "issue", ""))
        try:
            yield
        finally:
            self._current = previous

    def _register(self, address: int, nbytes: int) -> None:
        assert address not in self._live
        self._live.add(address)
        self.registered.append((address, nbytes))

    def _unregister(self, address: int) -> None:
        self._live.remove(address)
        self.unregistered.append(address)

    def _runner_offload(self, **options: object) -> HostBoundaryOffload:
        offload = HostBoundaryOffload(**{**options, "device": torch.device("cuda")})
        self.offloads.append((options, offload))
        return offload
