"""Counts the messages that cross an engine's process boundaries during a test.

The parent's side of its command pipe, by command name, and of its request link, by message
type, is counted in this process; calls in the engine's own processes (collectives, ring
writes) are counted there by ``Count`` injections that append to files in the probe's
directory.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from cti.protocol.link import ZmqPushQueue
from tests.support.fault_injection import Count


def _command_name(command: object) -> str:
    """A command's name: its type's, or a training command's verb in CamelCase."""

    verb = getattr(command, "verb", None)
    return type(command).__name__ if verb is None else verb.name.title().replace("_", "")


class IpcProbe:
    """One test's message counts."""

    # Collectives a rank's hot loop must not make.
    COLLECTIVES = (
        "torch.distributed.distributed_c10d:broadcast",
        "torch.distributed.distributed_c10d:all_reduce",
        "torch.distributed.distributed_c10d:barrier",
    )

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self.sent: Counter[str] = Counter()
        self.received: Counter[str] = Counter()
        self.requests: Counter[str] = Counter()  # put on the request link

    def injections(self) -> tuple[Count, ...]:
        return tuple(Count(target, directory=str(self._directory)) for target in self.COLLECTIVES)

    @contextmanager
    def parent(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """Count what this process sends to and receives from the engine, by type."""

        send, recv, put = Connection.send, Connection.recv, ZmqPushQueue.put

        def counted_send(connection: Connection, value: object) -> None:
            self.sent[_command_name(value)] += 1
            send(connection, value)

        def counted_recv(connection: Connection) -> object:
            value = recv(connection)
            self.received[type(value).__name__] += 1
            return value

        def counted_put(queue: ZmqPushQueue, values: Sequence[object]) -> None:
            # One link message per non-empty put, and each message in it by type.
            self.requests["message"] += bool(values)
            self.requests.update(type(value).__name__ for value in values)
            put(queue, values)

        with monkeypatch.context() as patch:
            patch.setattr(Connection, "send", counted_send)
            patch.setattr(Connection, "recv", counted_recv)
            patch.setattr(ZmqPushQueue, "put", counted_put)
            yield

    @property
    def replies(self) -> int:
        """The engine's answers received."""

        return self.received["Answered"] + self.received["Refused"]

    def calls(self) -> Counter[str]:
        """Calls counted in the engine's processes, by target."""

        counted: Counter[str] = Counter()
        for path in self._directory.iterdir():
            counted.update(path.read_text().split())
        return counted
