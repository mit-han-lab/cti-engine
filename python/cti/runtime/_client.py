"""The parent's client of one engine: its rank processes, commands out, replies back by id,
and the training session it hands out.

The engine owns every piece of its state; the client keeps none of it. The parent spawns
the ranks itself and talks to rank 0 alone; it numbers its commands, and one reader thread
hands each reply to the call waiting for its id. Any failure of the engine, reported or
not, reclaims every rank and fails every call, now and later, with the same error. An
online engine's client also runs the tokenizer workers between the engine and its
frontend, and fails with them.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import partial
from multiprocessing.connection import wait
from multiprocessing.util import Finalize
from threading import Lock, Thread
from typing import TYPE_CHECKING

from cti.runtime._process import ChildProcess, ProgressWatch
from cti.runtime._protocol import (
    Answered,
    CoreFailed,
    EngineInfo,
    FailurePhase,
    Progress,
    RankProgress,
    ReadRevision,
    Refused,
    Shutdown,
    Started,
    Starting,
    TrainingCommand,
)
from cti.training import TrainingVerb
from cti.utils import init_logger

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from cti.config import EngineOptions, EngineTuning
    from cti.protocol import (
        SamplingPrompt,
        SamplingResult,
        SamplingSpec,
    )
    from cti.tokenizer import PromptEncoder

    from ._sampler import OfflineSampler
    from ._training import TrainingSession

logger = init_logger(__name__)

# The verbs whose answer acknowledges a state change, bounded by ``ACK_SECONDS``.
_ACKNOWLEDGED = frozenset(
    {
        TrainingVerb.ENTER_TRAINING_SESSION,
        TrainingVerb.PAUSE_FOR_SERVING,
        TrainingVerb.RESUME_TRAINING,
        TrainingVerb.CLOSE_TRAINING_SESSION,
        TrainingVerb.ABORT_TRAINING_SESSION,
    }
)


class EngineProcessError(RuntimeError):
    """The engine can no longer execute commands; ``phase`` is where it failed."""

    def __init__(self, message: str, phase: FailurePhase) -> None:
        super().__init__(message)
        self.phase = phase


# What a call waits for: rank 0's answers to its command, or the engine's failure.
_Answer = Answered | Progress | Refused | EngineProcessError


class EngineClient:
    """Spawns the engine's ranks for ``options`` and ``tuning`` (None keeps its defaults)
    and sends rank 0 commands, from any thread; ``checkpoint`` is a training checkpoint the
    engine resumes from. With ``num_tokenizers`` it serves online: it also runs the
    detokenizer and that many tokenizer workers, which serve a frontend on fresh
    ``endpoints``; without, it samples offline itself, on private ``endpoints`` it removes
    as it closes.

    Its reader thread is the one watcher of every child: the engine's first failure, a
    rank's or a tokenizer worker's exit included, calls ``on_failure`` once, from the
    thread that saw it."""

    STARTUP_STALL_SECONDS = 300.0  # the longest startup phase, between progress reports
    # A rank's live descendant of these is a JIT build in progress: it renews a startup wait.
    COMPILERS = frozenset({"nvcc", "ninja", "cicc", "ptxas"})
    ACK_SECONDS = 75.0  # a switch may take a control and a recovery acknowledgement
    PROGRESS_SECONDS = 3600.0  # the longest progress reports may keep renewing a wait
    SHUTDOWN_SECONDS = 10.0
    WORKER_EXIT_SECONDS = 30.0  # for the tokenizer workers to exit once interrupted
    EXITED_SECONDS = 1.0  # for an exited child to be named, once a pipe saw it go

    def __init__(
        self,
        options: EngineOptions,
        tuning: EngineTuning | None = None,
        *,
        checkpoint: str | None = None,
        num_tokenizers: int | None = None,
        on_failure: Callable[[], None] | None = None,
    ) -> None:
        self._on_failure = on_failure
        self._ids = itertools.count()
        self._send_lock = Lock()
        self._calls: dict[int, queue.SimpleQueue[_Answer]] = {}
        self._failure: EngineProcessError | None = None
        self._failure_lock = Lock()
        self._phase = FailurePhase.STARTUP  # where a failure now would happen
        self.info: EngineInfo | None = None  # once the engine started
        self._startup_phase = "spawning its ranks"
        self._ranks: list[ChildProcess] = []
        self._workers: list[ChildProcess] = []  # the tokenizer workers, when online
        self._connection: Connection | None = None  # rank 0's, once spawned
        self._rank_failures: tuple[Connection, ...] = ()  # each peer's failure report
        # The other nodes' agents, and the ranks they run, when the engine spans hosts.
        self._nodes: tuple[Connection, ...] = ()
        self._remote_ranks: frozenset[int] = frozenset()
        self._progress: dict[str, RankProgress] = {}  # each local rank's, as spawned
        self._encoder: PromptEncoder | None = None  # loaded once, under _loading
        self._loading = Lock()
        self._sampler: OfflineSampler | None = None  # when offline, once started
        if options.parallel.node_rank != 0:
            raise ValueError("only node 0 runs the engine; another node joins it as an agent")
        # The ranks' rendezvous and, offline, the request link's endpoints: private to
        # this engine and removed with it.
        self._directory = tempfile.TemporaryDirectory(prefix="cti-")
        online = num_tokenizers is not None
        if online and num_tokenizers < 1:
            raise ValueError("an online engine needs at least one tokenizer")
        from cti.protocol import ServingEndpoints  # not loaded with the runtime

        # Where a frontend sends this engine requests; only rank 0's Leader takes them.
        self.endpoints = ServingEndpoints.create(
            num_tokenizers if online else 0, directory=None if online else self._directory.name
        )
        try:
            self._start(options, tuning, checkpoint, online=online)
            self._await_startup()
            self.check()
            if online:
                self._start_workers()
            else:
                from ._sampler import OfflineSampler

                self._sampler = OfflineSampler(self.endpoints, self)
        except BaseException as error:
            self._fail(self._error("the engine's startup was interrupted"))
            try:
                self._stop_workers()
            except RuntimeError as survived:
                error.add_note(str(survived))
            raise
        self._phase = FailurePhase.RUNNING
        Thread(target=self._read, name="cti-client", daemon=True).start()

    def _await_startup(self) -> None:
        """Wait until the engine starts or fails. Each startup phase rank 0 reports renews
        the wait, and so does a rank's JIT compiler at work (a build's one long step reports
        nothing), the latter only within ``PROGRESS_SECONDS`` of the startup's beginning:
        only a stalled phase times out."""

        deadline = time.monotonic() + self.PROGRESS_SECONDS
        while self.info is None and self._failure is None:
            received = self._receive(self.STARTUP_STALL_SECONDS)
            if received is None:
                compilers = self._compiling()
                if compilers and time.monotonic() < deadline:
                    logger.info(
                        "Engine startup is compiling (%s) while %s",
                        ", ".join(sorted(compilers)),
                        self._startup_phase,
                    )
                    continue
                if compilers:
                    stalled = (
                        f"was still compiling ({', '.join(sorted(compilers))}) after "
                        f"{self.PROGRESS_SECONDS} s"
                    )
                else:
                    stalled = f"made no startup progress in {self.STARTUP_STALL_SECONDS} s"
                self._fail(self._error(f"the engine {stalled} while {self._startup_phase}"))
            elif not received:
                self._fail(self._exited())

    def _compiling(self) -> frozenset[str]:
        """The JIT compilers running under any rank."""

        return frozenset().union(
            *(rank.descendant_commands() & self.COMPILERS for rank in self._ranks)
        )

    def _start(
        self,
        options: EngineOptions,
        tuning: EngineTuning | None,
        checkpoint: str | None,
        *,
        online: bool,
    ) -> None:
        """Spawn this host's ranks: rank 0 talks to this client and relays what it takes in
        to its peers, one ordered link each, a pipe to a local peer and a TCP connection to
        a rank another node runs (``_nodes``)."""

        from cti.memory import configure_memory_saver_subprocess
        from cti.runtime._control import ControlChannel, Follower, Leader
        from cti.runtime._core import run_engine_core

        context = mp.get_context("spawn")
        parallel = options.parallel
        local = parallel.node_ranks
        init_method, remote = f"file://{self._directory.name}/group", {}
        if parallel.num_nodes > 1:
            init_method, remote = self._join_nodes(options, tuning, checkpoint, online=online)
        self._connection, theirs = context.Pipe(duplex=True)
        relays = {rank: context.Pipe(duplex=False) for rank in local[1:]}
        # Each peer reports its failure on a link of its own; rank 0 on the connection.
        failures = {rank: context.Pipe(duplex=False) for rank in local[1:]}
        peers = range(1, parallel.num_ranks)
        self._rank_failures = tuple(
            failures[rank][0] if rank in failures else remote[rank][1] for rank in peers
        )
        channel = ControlChannel(
            tuple(relays[rank][1] if rank in relays else remote[rank][0] for rank in peers)
        )
        planes = [Leader(theirs, channel, self.endpoints)]
        planes += [
            Follower(ControlChannel((relays[rank][0],)), failures[rank][1]) for rank in local[1:]
        ]
        try:
            for rank, plane in enumerate(planes):
                progress = RankProgress.shared(context)
                target = partial(
                    run_engine_core,
                    rank=rank,
                    init_method=init_method,
                    online=online,
                    progress=progress,
                )
                child = ChildProcess.start(
                    context,
                    target,
                    (options, tuning, plane, checkpoint),
                    name=f"cti-rank-{rank}",
                    preload=configure_memory_saver_subprocess,
                )
                self._ranks.append(child)
                self._progress[child.name] = progress
        finally:
            theirs.close()  # each rank holds its own ends now
            for receive, send in relays.values():
                receive.close()
                send.close()
            for _, report in failures.values():
                report.close()
            for relay, _ in remote.values():
                relay.close()

    def _join_nodes(
        self,
        options: EngineOptions,
        tuning: EngineTuning | None,
        checkpoint: str | None,
        *,
        online: bool,
    ) -> tuple[str, dict[int, tuple[Connection, Connection]]]:
        """Admit the other nodes and take their ranks' links: the process group's
        rendezvous, and each remote rank's relay and report connections."""

        from ._nodes import NodeListener, NodeWelcome

        self._startup_phase = "waiting for the other nodes to join"
        listener = NodeListener(options)
        try:
            deadline = time.monotonic() + self.STARTUP_STALL_SECONDS
            init_method = listener.init_method()
            listener.admit(
                lambda: NodeWelcome(options, tuning, checkpoint, init_method, online), deadline
            )
            self._nodes = tuple(listener.nodes)
            parallel = options.parallel
            remote = range(parallel.ranks_per_node, parallel.num_ranks)
            self._remote_ranks = frozenset(remote)
            return init_method, listener.rank_links(remote, deadline)
        finally:
            listener.close()

    def _start_workers(self) -> None:
        """Spawn the detokenizer and the tokenizer workers on the checkpoint the engine
        resolved; return once each reported ready."""

        from cti.tokenizer import tokenize_worker

        context = mp.get_context("spawn")
        info = self.info
        worker = partial(
            tokenize_worker,
            model_path=info.model_path,
            hf_config=info.hf_config,
            image_limits=info.image_limits,
            endpoints=self.endpoints,
        )
        names = {"cti-detokenizer": None}
        names.update({f"cti-tokenizer-{i}": i for i in range(self.endpoints.num_tokenizers)})
        ready: list[tuple[str, Connection]] = []
        try:
            for name, tokenizer_id in names.items():
                reader, writer = context.Pipe(duplex=False)
                ready.append((name, reader))
                try:
                    target = partial(worker, tokenizer_id=tokenizer_id, ready_connection=writer)
                    self._workers.append(
                        ChildProcess.start(context, target, (), name=name, preload=nullcontext)
                    )
                finally:
                    writer.close()
            for name, reader in ready:
                try:
                    logger.info(reader.recv())
                except EOFError as error:
                    raise RuntimeError(f"{name} exited before reporting readiness") from error
        finally:
            for _, reader in ready:
                reader.close()

    def _stop_workers(self) -> None:
        """Interrupt the tokenizer workers, which close their queues and exit, and reclaim
        them, terminating those that do not exit in time."""

        for worker in self._workers:
            worker.interrupt()
        deadline = time.monotonic() + self.WORKER_EXIT_SECONDS
        survivors = []
        for worker in self._workers:
            try:
                worker.stop(max(deadline - time.monotonic(), 0.0))
            except RuntimeError as survived:
                survivors.append(str(survived))
        if survivors:
            raise RuntimeError("; ".join(survivors))

    def close_at_exit(self, owner: object) -> Finalize:
        """Reclaim the engine when ``owner`` is collected or the interpreter exits."""

        return Finalize(owner, self._close_now, exitpriority=10)

    def kill_workers(self) -> None:
        """SIGKILL the tokenizer workers now, waiting for nothing, so a signal handler may
        call it; ``close`` still reaps them."""

        for worker in self._workers:
            worker.interrupt(force=True)

    @property
    def failure(self) -> EngineProcessError | None:
        """The engine's first failure, once it has one."""

        return self._failure

    def check(self) -> None:
        """Raise the engine's failure, once it has one."""

        if self._failure is not None:
            raise self._failure

    def fail(self, reason: str) -> None:
        """Fail the engine for its owner's ``reason``, as its own failure does."""

        self._fail(self._error(reason))

    def close(self) -> None:
        """Shut the engine down, then its tokenizer workers; every process is gone when it
        returns. Closing an engine that already failed only reclaims what is left of it."""

        self._phase = FailurePhase.SHUTDOWN
        try:
            if self._failure is None:
                self._call(Shutdown, "Shutdown", timeout=self.SHUTDOWN_SECONDS)
                self._tell_nodes_shut_down()
        finally:
            self._close_now()

    def _tell_nodes_shut_down(self) -> None:
        """Tell every other node the engine shut down, so it reclaims its ranks as done; a
        node whose link closes without word fails."""

        if not self._nodes:
            return
        from ._nodes import EngineShutDown

        for node in self._nodes:
            try:
                node.send(EngineShutDown())
            except OSError:  # it is gone already
                pass

    def _close_now(self) -> None:
        self._fail(EngineProcessError("the engine is closed", FailurePhase.SHUTDOWN))
        if self._sampler is not None:
            self._sampler.close()
        self._stop_workers()

    def training_session(self, owner: object) -> TrainingSession:
        """A training session on this engine; it keeps ``owner``, the facade that owns this
        client, alive while it lives."""

        from ._training import TrainingSession

        return TrainingSession(owner, self, checks=self.info.checks)

    def prompt_encoder(self) -> PromptEncoder:
        """The engine's prompt encoder, loaded with the first call."""

        with self._loading:
            if self._encoder is None:
                from cti.tokenizer import PromptEncoder

                info = self.info
                self._encoder = PromptEncoder.from_checkpoint(
                    info.model_path, hf_config=info.hf_config, limits=info.image_limits
                )
            return self._encoder

    # Commands

    def sample(
        self, prompts: Sequence[SamplingPrompt], policies: Sequence[SamplingSpec]
    ) -> SamplingResult:
        """Sample ``prompts`` as one command, each with its policy (``SamplingParams.specs``),
        offline: an online engine takes requests from its server."""

        if self._sampler is None:
            raise RuntimeError("an online engine samples through its server")
        return self._sampler.sample(prompts, policies)

    def model_revision(self) -> int:
        return self._call(ReadRevision, "ReadRevision")

    def call(self, verb: TrainingVerb, *args: object) -> object:
        """Send the training command ``verb`` with ``args``; its answer's outcome."""

        timeout = self.ACK_SECONDS if verb in _ACKNOWLEDGED else None
        command = partial(TrainingCommand, verb=verb, args=args)
        return self._call(command, verb.name, timeout=timeout)

    # Calls and replies

    def _call(
        self, command: Callable[[int], object], name: str, *, timeout: float | None = None
    ) -> object:
        """Send ``command(id)``, named ``name`` in errors, and return its answer's outcome;
        ``timeout`` bounds the wait for each answer, a progress report renewing it."""

        self.check()
        command_id = next(self._ids)
        replies: queue.SimpleQueue[_Answer] = queue.SimpleQueue()
        self._calls[command_id] = replies
        deadline = time.monotonic() + self.PROGRESS_SECONDS
        try:
            with self._send_lock:
                self._connection.send(command(command_id))
            while True:
                try:
                    answer = replies.get(timeout=timeout)
                except queue.Empty:
                    self._fail(self._error(f"timed out waiting for {name}"))
                    raise self._failure from None
                match answer:
                    case Answered(value=value):
                        return value
                    case Refused(error=error, remote_traceback=remote):
                        error.add_note(f"Remote traceback:\n{remote}")
                        raise error
                    case EngineProcessError():
                        raise answer
                if time.monotonic() > deadline:  # a progress report renewed the wait
                    self._fail(self._error(f"{name} took too long"))
                    raise self._failure
        except OSError:  # rank 0 closed its end
            self._fail(self._exited())
            raise self._failure from None
        except BaseException as error:
            if self._failure is None and not isinstance(error, Exception):
                # Interrupted once it may have been sent, the command's outcome is unknown.
                self._fail(self._error(f"{name} was interrupted"))
            raise
        finally:
            self._calls.pop(command_id, None)

    def _read(self) -> None:
        """Deliver every reply until the engine fails, a rank exits or a local rank hangs."""

        ranks = {child.name: child for child in self._ranks}
        watch = ProgressWatch(self._progress, lambda name: ranks[name].suspended())
        while self._failure is None:
            try:
                if self._receive(ProgressWatch.CHECK_SECONDS) is False:
                    self._fail(self._exited())
                elif (hung := watch.hung(time.monotonic())) is not None:
                    self._fail(self._error(f"the engine hung: {hung}"))
            except BaseException:  # noqa: BLE001 - the engine is gone either way
                self._fail(self._exited())

    def _receive(self, timeout: float | None) -> bool | None:
        """Deliver rank 0's next reply; False once a rank or a tokenizer worker exited
        without one, None at ``timeout``. A reply that arrived wins over an exit seen at
        the same time."""

        connection = self._connection
        children = (*self._ranks, *self._workers)
        ready = wait(
            (
                connection,
                *self._rank_failures,
                *self._nodes,
                *(child.sentinel for child in children),
            ),
            timeout,
        )
        if not ready:
            return None
        if connection not in ready:
            for rank, failures in enumerate(self._rank_failures, start=1):
                if failures in ready:
                    try:
                        self._fail_remotely(failures.recv(), rank)
                    except (EOFError, OSError):
                        if rank in self._remote_ranks:
                            self._fail(self._error(f"rank {rank}'s link from its node closed"))
                            return True
                        return False  # the peer exited without reporting why
                    return True
            for node, link in enumerate(self._nodes, start=1):
                if link in ready:  # an agent sends only why it gives up, then closes
                    try:
                        reason = f"node {node} failed: {link.recv().reason}"
                    except (EOFError, OSError):
                        reason = f"node {node}'s link closed"
                    self._fail(self._error(reason))
                    return True
            return False
        try:
            reply = connection.recv()
        except (EOFError, OSError):
            return False
        match reply:
            case Starting(phase=phase):
                self._startup_phase = phase
            case Started(info=info):
                self.info = info
            case Answered() | Progress() | Refused():
                if (replies := self._calls.get(reply.id)) is not None:
                    replies.put(reply)
            case CoreFailed():
                self._fail_remotely(reply, 0)
        return True

    def _fail_remotely(self, reply: CoreFailed, rank: int) -> None:
        """Fail with the error ``rank`` reported."""

        error = EngineProcessError(
            f"the engine failed while {reply.phase.value} on rank {rank}: "
            f"{reply.type_name}: {reply.message}",
            reply.phase,
        )
        error.add_note(f"Remote engine traceback:\n{reply.remote_traceback}")
        # Ranks fail in turn, a peer's own error often first: name every report already sent.
        for peer, failures in enumerate(self._rank_failures, start=1):
            if peer != rank and failures.poll(0):
                try:
                    other = failures.recv()
                except (EOFError, OSError):
                    continue
                error.add_note(
                    f"Rank {peer} also failed while {other.phase.value}: "
                    f"{other.type_name}: {other.message}\n{other.remote_traceback}"
                )
        self._fail(error)

    def _exited(self) -> EngineProcessError:
        """A rank or a tokenizer worker exited without reporting why: named, with its exit
        code, once its sentinel is ready within EXITED_SECONDS."""

        children = (*self._ranks, *self._workers)
        deadline = time.monotonic() + self.EXITED_SECONDS
        ended = wait([child.sentinel for child in children], self.EXITED_SECONDS)
        for child in children:
            if child.sentinel in ended:
                # The sentinel is ready as the child exits, a moment before it is reaped.
                while child.exitcode is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                return self._error(f"{child.name} exited unexpectedly (exit code {child.exitcode})")
        return self._error("an engine process exited unexpectedly")

    def _error(self, message: str) -> EngineProcessError:
        return EngineProcessError(message, self._phase)

    def _fail(self, error: EngineProcessError) -> None:
        """Record the engine's first failure, reclaim every rank, fail every waiting call
        and report it to ``on_failure``, then remove the engine's directory; later failures
        change nothing."""

        with self._failure_lock:
            first = self._failure is None
            if first:
                self._failure = error
            for rank in self._ranks:
                try:
                    rank.stop()
                except RuntimeError as survived:
                    self._failure.add_note(str(survived))
            if self._connection is not None:
                self._connection.close()
            for failures in self._rank_failures:
                failures.close()
            for node in self._nodes:  # its agent reclaims that node's ranks
                node.close()
        for replies in list(self._calls.values()):
            replies.put(self._failure)
        if first:
            if self._on_failure is not None:
                self._on_failure()
            self._remove_directory()

    def _remove_directory(self) -> None:
        """Remove the engine's directory if it can; a file still open on a network file
        system may keep it, and that never hides the engine's failure."""

        try:
            self._directory.cleanup()
        except OSError as error:
            logger.warning(
                "could not remove the engine's directory %s: %s", self._directory.name, error
            )


__all__ = ["EngineClient", "EngineProcessError"]
