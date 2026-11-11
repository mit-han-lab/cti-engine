"""An engine whose ranks span hosts (``ParallelOptions.num_nodes``): node 0's parent serves and
runs rank 0; every other node runs a ``NodeAgent`` that joins it and starts that node's ranks.

Only rank 0 talks to the parent, the frontend and the tokenizers, so they all stay on node 0.
What crosses hosts is each remote rank's two links, rank 0's ordered relay to it and its
failure report to the parent, as authenticated TCP ``multiprocessing`` connections in place
of local pipes, and the process group's rendezvous, at a ``tcp://`` address node 0 picks.

A node joins with a handshake: node 0 refuses one that resolves another model, draft or
parallel layout (but for its ``node_rank``), engine or kernel version or checkpoint, or that
names a node rank out of range or already taken; node 0 sends it the rest of its options, such
as scheduling and training, which a node's command line need not spell. Either side's end is the other's failure: a lost link fails node 0's
engine, which reclaims its ranks; a joined node reclaims its own once node 0's link closes.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import socket
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from functools import partial
from importlib import metadata
from multiprocessing.connection import (
    Client,
    Connection,
    answer_challenge,
    deliver_challenge,
    wait,
)
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from cti.env import ENV
from cti.utils import init_logger

from ._process import ChildProcess, ProgressWatch
from ._protocol import RankProgress

if TYPE_CHECKING:
    from cti.config import EngineOptions, EngineTuning

logger = init_logger(__name__)

# A peer that stops answering is lost within IDLE + INTERVAL * COUNT seconds.
_KEEPALIVE_IDLE_SECONDS = 5
_KEEPALIVE_INTERVAL_SECONDS = 2
_KEEPALIVE_COUNT = 3
_DIAL_RETRY_SECONDS = 1.0
# How long a node waits for the others, and its ranks for node 0: an engine's startup stall.
_JOIN_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class NodeHello:
    """A joining node: who it is and what it would run."""

    node_rank: int
    num_nodes: int
    identity: str


@dataclass(frozen=True, slots=True)
class NodeWelcome:
    """What a joined node's ranks run: node 0's options, tuning and training checkpoint,
    the process group's rendezvous, and whether the engine serves online."""

    options: EngineOptions
    tuning: EngineTuning | None
    checkpoint: str | None
    init_method: str
    online: bool


@dataclass(frozen=True, slots=True)
class NodeRefused:
    reason: str


@dataclass(frozen=True, slots=True)
class NodeFailed:
    """An agent's word to node 0 of why it gives up, before it closes its link."""

    reason: str


@dataclass(frozen=True, slots=True)
class EngineShutDown:
    """Node 0's word to an agent that the engine shut down: its ranks are done."""


@dataclass(frozen=True, slots=True)
class RankHello:
    """A remote rank's link: ``relay`` carries rank 0's records to it, ``report`` its failure
    to the parent."""

    rank: int
    role: Literal["relay", "report"]


def node_identity(options: EngineOptions) -> str:
    """What every node resolves on its own host and must agree on: the model, its draft and
    the parallel layout but for ``node_rank``, the engine's and the kernels' versions, and
    the checkpoint a local folder holds."""

    parallel = replace(options.parallel, node_rank=0)
    digest = hashlib.sha256(repr((options.model, options.speculative, parallel)).encode())
    for distribution in ("cti", "cti-kernels"):
        try:
            digest.update(f"{distribution}=={metadata.version(distribution)}".encode())
        except metadata.PackageNotFoundError:
            digest.update(f"{distribution} from source".encode())
    folder = Path(options.model.path).expanduser()
    if folder.is_dir():  # a Hugging Face id is its options' part; a folder, its files
        if (config := folder / "config.json").is_file():
            digest.update(config.read_bytes())
        for weights in sorted(folder.glob("*.safetensors")):
            digest.update(f"{weights.name}:{weights.stat().st_size}".encode())
    return digest.hexdigest()


def _authkey() -> bytes:
    key = ENV.dist_authkey
    if key is None:
        raise RuntimeError("CTI_DIST_AUTHKEY is unset; every node of a multi-node engine needs it")
    return key


def _address(options: EngineOptions) -> tuple[str, int]:
    host, _, port = options.parallel.dist_init_addr.rpartition(":")
    return host, int(port)


def _keep_alive(connection: Connection) -> Connection:
    """``connection``, its socket probing an idle peer, so a host that froze or lost its link
    is noticed as a closed connection."""

    with socket.socket(fileno=os.dup(connection.fileno())) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        probe.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, _KEEPALIVE_IDLE_SECONDS)
        probe.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, _KEEPALIVE_INTERVAL_SECONDS)
        probe.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, _KEEPALIVE_COUNT)
    return connection


def _dial(address: tuple[str, int], deadline: float) -> Connection:
    """A connection to node 0 at ``address``, redialled until ``deadline``: the nodes start in
    any order."""

    while True:
        try:
            return _keep_alive(Client(address, family="AF_INET", authkey=_authkey()))
        except (ConnectionRefusedError, ConnectionResetError, TimeoutError):
            if time.monotonic() > deadline:
                raise TimeoutError(f"node 0 did not accept at {address[0]}:{address[1]}") from None
            time.sleep(_DIAL_RETRY_SECONDS)


class NodeListener:
    """Node 0's side: admits every other node, then takes each remote rank's two links."""

    def __init__(self, options: EngineOptions) -> None:
        self._options = options
        self._identity = node_identity(options)
        self._authkey = _authkey()
        self._server = socket.create_server(_address(options))
        self.nodes: list[Connection] = []  # each joined node's agent, which ends with it

    def _next(self, deadline: float, waiting_for: str) -> tuple[Connection, object]:
        """The next peer that proves it holds the key, and its first message, by
        ``deadline``; as ``multiprocessing.connection.Listener`` admits, with a deadline."""

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{waiting_for} did not join in time")
            self._server.settimeout(remaining)
            try:
                peer, _ = self._server.accept()
            except TimeoutError:
                raise TimeoutError(f"{waiting_for} did not join in time") from None
            peer.setblocking(True)
            connection = Connection(peer.detach())
            try:
                deliver_challenge(connection, self._authkey)
                answer_challenge(connection, self._authkey)
            except mp.AuthenticationError:
                logger.warning("refused a node whose CTI_DIST_AUTHKEY is a mismatch")
                connection.close()
                continue
            except (EOFError, OSError):
                connection.close()
                continue
            _keep_alive(connection)
            if not connection.poll(max(deadline - time.monotonic(), 0.0)):
                connection.close()
                raise TimeoutError(f"{waiting_for} connected but said nothing")
            return connection, connection.recv()

    def admit(self, welcome: Callable[[], NodeWelcome], deadline: float) -> None:
        """Admit every other node, each sent ``welcome()`` once all have joined."""

        num_nodes = self._options.parallel.num_nodes
        joined: dict[int, Connection] = {}
        while len(joined) < num_nodes - 1:
            connection, hello = self._next(
                deadline, f"node {sorted(set(range(1, num_nodes)) - set(joined))}"
            )
            match hello:
                case NodeHello() if hello.num_nodes != num_nodes:
                    reason = f"num_nodes {hello.num_nodes} differs from node 0's {num_nodes}"
                case NodeHello() if (
                    not 0 < hello.node_rank < num_nodes or hello.node_rank in joined
                ):
                    reason = f"node_rank {hello.node_rank} is out of range or already joined"
                case NodeHello() if hello.identity != self._identity:
                    reason = (
                        "its model, parallel layout, engine or kernel versions, or checkpoint "
                        "differ from node 0's"
                    )
                case NodeHello():
                    joined[hello.node_rank] = connection
                    continue
                case _:
                    reason = "it did not introduce itself as a node"
            connection.send(NodeRefused(reason))
            connection.close()
            raise RuntimeError(f"refused a node: {reason}")
        message = welcome()
        for node_rank in sorted(joined):
            joined[node_rank].send(message)
        self.nodes = [joined[node_rank] for node_rank in sorted(joined)]

    def rank_links(self, ranks: range, deadline: float) -> dict[int, tuple[Connection, Connection]]:
        """Each remote rank's relay and report links, as its rank dialled them."""

        links: dict[tuple[int, str], Connection] = {}
        expected = {(rank, role) for rank in ranks for role in ("relay", "report")}
        while expected - links.keys():
            connection, hello = self._next(
                deadline, f"remote ranks {sorted({r for r, _ in expected - links.keys()})}"
            )
            key = (hello.rank, hello.role) if isinstance(hello, RankHello) else None
            if key not in expected or key in links:
                connection.close()
                raise RuntimeError(f"a remote rank's link was unexpected: {hello!r}")
            links[key] = connection
        return {rank: (links[rank, "relay"], links[rank, "report"]) for rank in ranks}

    def init_method(self) -> str:
        """The process group's rendezvous: a free port on node 0's address."""

        host, _ = _address(self._options)
        with socket.socket() as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]
        return f"tcp://{host}:{port}"

    def close(self) -> None:
        self._server.close()


def run_remote_rank(
    welcome: NodeWelcome, *, rank: int, address: tuple[str, int], progress: RankProgress
) -> None:
    """A remote rank's entry: dial node 0 for its two links, then run as a peer."""

    from ._control import ControlChannel, Follower
    from ._core import run_engine_core

    deadline = time.monotonic() + _JOIN_SECONDS
    relay = _dial(address, deadline)
    relay.send(RankHello(rank, "relay"))
    report = _dial(address, deadline)
    report.send(RankHello(rank, "report"))
    plane = Follower(ControlChannel((relay,)), report)
    run_engine_core(
        welcome.options,
        welcome.tuning,
        plane,
        welcome.checkpoint,
        rank=rank,
        init_method=welcome.init_method,
        online=welcome.online,
        progress=progress,
    )


class NodeAgent:
    """A node other than 0: joins node 0, starts this node's ranks, and lives as long as they
    and node 0's link do; whichever ends first, it reclaims the rest."""

    def __init__(self, options: EngineOptions) -> None:
        if options.parallel.node_rank == 0:
            raise ValueError("node 0 runs the engine itself; an agent joins it")
        self._options = options
        self._ranks: list[ChildProcess] = []
        self._progress: dict[str, RankProgress] = {}

    def run(self) -> int:
        """Join, run this node's ranks, and return the exit code: 0 once the engine shut down."""

        from cti.memory import configure_memory_saver_subprocess

        parallel = self._options.parallel
        address = _address(self._options)
        node = _dial(address, time.monotonic() + _JOIN_SECONDS)
        node.send(NodeHello(parallel.node_rank, parallel.num_nodes, node_identity(self._options)))
        if not node.poll(_JOIN_SECONDS):
            raise TimeoutError("node 0 did not answer the handshake")
        match node.recv():
            case NodeRefused(reason=reason):
                raise RuntimeError(f"node 0 refused this node: {reason}")
            case NodeWelcome() as welcome:
                pass
        logger.info("Joined node 0 as node %d", parallel.node_rank)
        context = mp.get_context("spawn")
        try:
            for rank in parallel.node_ranks:
                progress = RankProgress.shared(context)
                child = ChildProcess.start(
                    context,
                    partial(run_remote_rank, rank=rank, address=address, progress=progress),
                    (welcome,),
                    name=f"cti-rank-{rank}",
                    preload=configure_memory_saver_subprocess,
                )
                self._ranks.append(child)
                self._progress[child.name] = progress
            return self._watch(node)
        finally:
            for child in self._ranks:
                try:
                    child.stop()
                except RuntimeError as survived:
                    logger.error("%s", survived)
            node.close()

    def _watch(self, node: Connection) -> int:
        """Wait for node 0's word that the engine shut down, 0, or for its link to close, a
        rank to exit or a rank to hang first, a failure; a hang node 0 hears about."""

        sentinels = {child.sentinel: child for child in self._ranks}
        ranks = {child.name: child for child in self._ranks}
        watch = ProgressWatch(self._progress, lambda name: ranks[name].suspended())
        while True:
            ready = wait([node, *sentinels], ProgressWatch.CHECK_SECONDS)
            if (hung := watch.hung(time.monotonic())) is not None:
                logger.error("This node's ranks hung: %s", hung)
                with suppress(OSError):
                    node.send(NodeFailed(f"its ranks hung: {hung}"))
                return 1
            for sentinel in (s for s in ready if s in sentinels):
                child = sentinels[sentinel]
                while child.exitcode is None:
                    time.sleep(0.01)
                logger.error("%s exited with code %s", child.name, child.exitcode)
                return 1
            if node not in ready:
                continue
            try:
                if isinstance(node.recv(), EngineShutDown):
                    return 0
            except (EOFError, OSError):
                logger.error("node 0's link closed; reclaiming this node's ranks")
                return 1


__all__ = [
    "EngineShutDown",
    "NodeAgent",
    "NodeFailed",
    "NodeListener",
    "NodeWelcome",
    "node_identity",
    "run_remote_rank",
]
