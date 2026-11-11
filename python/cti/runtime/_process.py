"""Process supervision: run a child bound to its parent's lifetime, as the leader of its own
process group, and reclaim the group as one."""

from __future__ import annotations

import os
import signal
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from threading import Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext
    from multiprocessing.process import BaseProcess

    from cti.runtime._protocol import RankProgress


def _bind_lifetime_to_parent(expected_parent_pid: int) -> None:
    if sys.platform != "linux":
        return

    if os.getppid() != expected_parent_pid:
        raise RuntimeError("parent exited before child lifetime binding completed")

    def watch_parent() -> None:
        # getppid follows the owning process, including when the thread that
        # spawned us exits. Reparenting persists after owner death, so neither
        # death before this thread starts nor PID reuse can lose the event.
        # Polling also works on Python builds without os.pidfd_open. Detection
        # takes at most one 100 ms interval, subject to thread scheduling.
        while os.getppid() == expected_parent_pid:
            time.sleep(0.1)
        os._exit(1)

    Thread(target=watch_parent, name="parent-lifetime", daemon=True).start()


def run_parent_owned_process(
    expected_parent_pid: int,
    target: Callable[..., None],
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
) -> None:
    """Run one child entry point bound to its parent process lifetime on Linux."""

    _bind_lifetime_to_parent(expected_parent_pid)
    try:
        target(*args, **({} if kwargs is None else kwargs))
    finally:
        # Once the entry point has returned the child is committed to exiting,
        # but it still looks alive to the parent while the interpreter tears
        # down. A graceful-stop SIGINT from the supervisor would then land
        # inside an atexit hook as KeyboardInterrupt ("Exception ignored in
        # atexit callback"). Nothing is left to interrupt, so ignore it.
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _live_processes() -> dict[int, tuple[int, str]]:
    """Each live (not zombie) process on this host: its parent pid and command name."""

    processes = {}
    with suppress(FileNotFoundError):
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/stat", "rb") as stat:
                    line = stat.read().decode(errors="replace")
            except OSError:  # it exited while we listed
                continue
            # "pid (command) state ppid ...": the command may hold spaces and parentheses.
            head, _, tail = line.rpartition(")")
            state, parent = tail.split()[:2]
            if state != "Z":
                processes[int(entry.name)] = (int(parent), head.partition("(")[2])
    return processes


@dataclass(frozen=True)
class _GroupLeader:
    """Runs ``target`` in a process that leads its own group and exits with its parent."""

    parent: int
    target: Callable[..., None]

    def __call__(self, *args: object) -> None:
        os.setpgid(0, 0)
        run_parent_owned_process(self.parent, self.target, args)


class ChildProcess:
    """A spawned child that leads its own process group: started, watched, and reclaimed
    together with every process it spawned.

    ``stop`` is the one way it ends: after ``grace`` for it to exit by itself, SIGTERM the
    group and reap its leader, then wait until no member is left; whatever outlives its
    timeout gets SIGKILL. Until the child leads its group it has spawned nothing, and it
    is signalled alone.
    """

    TERMINATE_SECONDS = 5.0
    KILL_SECONDS = 5.0
    POLL_SECONDS = 0.01

    def __init__(self, process: BaseProcess) -> None:
        self._process = process
        self._stopped = False

    @classmethod
    def start(
        cls,
        context: BaseContext,
        target: Callable[..., None],
        args: tuple[object, ...],
        *,
        name: str,
        preload: Callable[[], AbstractContextManager[object]],
    ) -> ChildProcess:
        """Spawn ``target(*args)``, started inside ``preload``, the child's provider hook."""

        leader = _GroupLeader(os.getpid(), target)
        process = context.Process(target=leader, args=args, name=name, daemon=False)
        with preload():
            process.start()
        return cls(process)

    @property
    def name(self) -> str:
        return self._process.name

    @property
    def sentinel(self) -> int:
        return self._process.sentinel

    def descendant_commands(self) -> frozenset[str]:
        """The command names of the child's live descendants, read from ``/proc`` on this
        host; empty where there is no ``/proc``."""

        if not self.alive():
            return frozenset()
        children: dict[int, list[tuple[int, str]]] = {}
        for pid, (parent, command) in _live_processes().items():
            children.setdefault(parent, []).append((pid, command))
        commands, pending = set(), [self._process.pid]
        while pending:
            for pid, command in children.get(pending.pop(), ()):
                commands.add(command)
                pending.append(pid)
        return frozenset(commands)

    @property
    def exitcode(self) -> int | None:
        return None if self._stopped else self._process.exitcode

    def alive(self) -> bool:
        return not self._stopped and self._process.is_alive()

    def suspended(self) -> bool:
        """Whether the child is stopped by a signal or a tracer, as ``/proc`` on this host
        shows; False where there is no ``/proc``."""

        if not self.alive():
            return False
        try:
            with open(f"/proc/{self._process.pid}/stat", encoding="ascii") as stat:
                state = stat.read().rpartition(")")[2].split()[0]
        except (OSError, IndexError):
            return False
        return state in ("T", "t")

    def interrupt(self, *, force: bool = False) -> None:
        """Ask the child alone to exit: SIGINT, which it may handle, or with ``force``
        SIGKILL. Waits for nothing, so a signal handler may call it."""

        # ValueError: a signal handler interrupted ``stop`` as it closed the process.
        with suppress(ProcessLookupError, ValueError):
            if self.alive():
                os.kill(self._process.pid, signal.SIGKILL if force else signal.SIGINT)

    def stop(self, grace: float = 0.0) -> None:
        """Reclaim the group; raises ``RuntimeError`` if a member survives SIGKILL. Stopping
        a stopped child does nothing."""

        if self._stopped:
            return
        process = self._process
        if grace > 0:
            process.join(grace)
        group = self._signal(signal.SIGTERM, process.terminate)
        if process.is_alive():
            process.join(self.TERMINATE_SECONDS)
        if process.is_alive():
            group = self._signal(signal.SIGKILL, process.kill) or group
            process.join(self.KILL_SECONDS)
            if process.is_alive():
                raise RuntimeError(f"{process.name} survived SIGTERM and SIGKILL")
        # A reaped leader no longer answers for its group: wait until its members are gone.
        if group and not self._group_gone(self.TERMINATE_SECONDS):
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            if not self._group_gone(self.KILL_SECONDS):
                raise RuntimeError(f"processes of {process.name} survived SIGTERM and SIGKILL")
        process.close()
        self._stopped = True

    def _signal(self, signum: int, alone: Callable[[], None]) -> bool:
        """Send ``signum`` to the group, or to the child ``alone`` while it leads none;
        whether the group exists."""

        try:
            os.killpg(self._process.pid, signum)
        except ProcessLookupError:
            if self._process.is_alive():
                alone()
            return False
        return True

    def _group_gone(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            try:
                os.killpg(self._process.pid, 0)
            except ProcessLookupError:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(self.POLL_SECONDS)


class ProgressWatch:
    """A parent's view of its ranks' progress records, checked at most once a
    ``CHECK_SECONDS``. A rank that has work and finishes no loop turn within its allowance
    has hung, whether it is stopped or waits in a collective that never completes; the
    parent sees that without the rank's help, so a rank blocked in a kernel launch or
    holding its GIL is seen as well."""

    CHECK_SECONDS = 1.0

    def __init__(self, ranks: Mapping[str, RankProgress], stopped: Callable[[str], bool]) -> None:
        """``stopped`` says whether the named rank's process is stopped, which a hang's
        report names."""

        self._ranks = dict(ranks)
        self._stopped = stopped
        self._seen: dict[str, tuple[float, float]] = {}  # steps, and since when unchanged
        self._next = 0.0

    def hung(self, now: float) -> str | None:
        """Why the ranks hung as of ``now``, if any did: each one that has had work for longer
        than its allowance without finishing an iteration."""

        if now < self._next:
            return None
        self._next = now + self.CHECK_SECONDS
        hung = []
        for name, rank in self._ranks.items():
            steps, allowance = rank.steps, rank.allowance
            seen, since = self._seen.get(name, (-1.0, now))
            if steps != seen or allowance == 0:
                self._seen[name] = (steps, now)
            elif now - since > allowance:
                hung.append(f"{name} for {now - since:.0f} s")
        if not hung:
            return None
        stopped = [name for name in self._ranks if self._stopped(name)]
        reason = "made no progress with work in hand: " + ", ".join(hung)
        return reason + (f" ({', '.join(stopped)} stopped)" if stopped else "")


__all__ = ["ChildProcess", "ProgressWatch", "run_parent_owned_process"]
