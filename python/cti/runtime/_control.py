"""The ranks' control plane: rank 0 takes in the parent's commands and the online requests
and relays them, in order, to its peers."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.connection import Connection

    from cti.inference import RequestSeeds
    from cti.protocol import InferenceReply, ServingEndpoints
    from cti.protocol.link import ZmqRequestTransport
    from cti.runtime._protocol import CoreFailed


@dataclass(frozen=True, slots=True)
class ControlRecord:
    """What one iteration of rank 0's loop took in, in arrival order: the parent's commands
    and the online requests. Every rank applies the same records in the same order, so
    their schedulers decide alike."""

    commands: tuple[object, ...] = ()
    requests: tuple[object, ...] = ()


class ControlChannel:
    """Rank 0 writes each record once to every peer; a peer reads them in order. One pipe
    per peer keeps the order and bounds nothing: a record may hold a training batch."""

    def __init__(self, connections: tuple[Connection, ...]) -> None:
        self._connections = connections

    def send(self, record: ControlRecord) -> None:
        if not self._connections:
            return  # no peers
        data = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        for connection in self._connections:
            connection.send_bytes(data)

    def receive(self) -> ControlRecord:
        (connection,) = self._connections
        return pickle.loads(connection.recv_bytes())


class Completion(Protocol):
    """Device work the loop may wait for besides its input, such as a CUDA event."""

    def query(self) -> bool:
        """Whether it has completed."""
        ...


class ControlPlane(Protocol):
    """A rank's end of the control plane: where its records come from and its answers go."""

    def start(self, *, seeds: RequestSeeds | None) -> None:
        """Open what the rank's loop thread reads from; rank 0 stamps every request it
        takes in with ``seeds``, when given."""
        ...

    def take(self, *, block: bool, until: Completion | None) -> ControlRecord:
        """The next record; ``block`` waits a moment for one when the rank is idle, and
        ``until`` for one or that step's completion, whichever comes first."""
        ...

    def reply(self, reply: object) -> None:
        """Answer the parent."""
        ...

    def publish(self, replies: Sequence[InferenceReply]) -> None:
        """Send the requests' replies to their frontend."""
        ...

    def fail(self, failure: CoreFailed) -> None:
        """Report the rank's failure to the parent, as it exits."""
        ...


class Leader:
    """Rank 0's end: reads the parent's commands from ``connection`` and the online
    requests from ``endpoints``, relays each record to ``peers``, answers the parent and
    publishes the requests' replies."""

    def __init__(
        self, connection: Connection, peers: ControlChannel, endpoints: ServingEndpoints
    ) -> None:
        self._connection = connection
        self._peers = peers
        self._endpoints = endpoints
        self._transport: ZmqRequestTransport | None = None  # once started
        self._seeds: RequestSeeds | None = None

    def start(self, *, seeds: RequestSeeds | None) -> None:
        from cti.protocol.link import ZmqRequestTransport

        self._transport = ZmqRequestTransport(self._endpoints)
        self._seeds = seeds

    def take(self, *, block: bool, until: Completion | None) -> ControlRecord:
        if until is not None:
            # Spin until the step completes or input arrives, as a blocking settle would
            # spin on the event: a request that arrives meanwhile launches as the device
            # frees.
            while not (until.query() or self._connection.poll(0) or self._transport.ready()):
                pass
        elif block and not self._connection.poll(0):
            # An idle rank 0 wakes on either input at once: a request never waits out a poll.
            self._transport.wait(self._connection.fileno())
        commands = []
        while self._connection.poll(0):
            commands.append(self._connection.recv())
        requests = list(self._transport.receive())
        if self._seeds is not None:
            requests = self._seeds.stamp(requests)
        record = ControlRecord(tuple(commands), tuple(requests))
        self._peers.send(record)
        return record

    def reply(self, reply: object) -> None:
        self._connection.send(reply)

    def publish(self, replies: Sequence[InferenceReply]) -> None:
        if replies:
            self._transport.publish(replies)

    def fail(self, failure: CoreFailed) -> None:
        self._connection.send(failure)


class Follower:
    """A peer's end: rank 0's records, in order; its answers and replies are rank 0's to
    send, and its failure goes to the parent on ``failures``, its own one-way pipe."""

    def __init__(self, leader: ControlChannel, failures: Connection) -> None:
        self._leader = leader
        self._failures = failures

    def start(self, *, seeds: RequestSeeds | None) -> None:
        pass

    def take(self, *, block: bool, until: Completion | None) -> ControlRecord:
        return self._leader.receive()

    def reply(self, reply: object) -> None:
        pass

    def publish(self, replies: Sequence[InferenceReply]) -> None:
        pass

    def fail(self, failure: CoreFailed) -> None:
        self._failures.send(failure)


__all__ = ["Completion", "ControlChannel", "ControlPlane", "ControlRecord", "Follower", "Leader"]
