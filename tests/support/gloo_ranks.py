"""Tensor-parallel ranks as spawned processes, each owning a real ``ParallelGroup`` whose
device groups run over gloo (CPU hosts have no NCCL). Ranks report through Pipes and never
raise, so no child prints a traceback."""

from __future__ import annotations

import multiprocessing as mp
import traceback
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from cti.distributed import Topology
from cti.distributed.runtime import ParallelGroup

RANK_TIMEOUT_SECONDS = 120.0

# Runs on every rank with its live group and the shared argument. It is pickled by
# reference, so it must be a module-level function. Its result must not hold tensors:
# they would cross the Pipe as shared memory owned by an exiting rank.
RankScenario = Callable[[ParallelGroup, Any], Any]


def _map_device_groups_to_gloo() -> None:
    """CPU-only hosts refuse ProcessGroupNCCL; give each device group to gloo."""

    import torch.distributed as dist

    new_group = dist.new_group

    def gloo_group(*args: Any, backend: str | None = None, **kwargs: Any) -> object:
        if backend == "nccl":
            # Gloo connects eagerly anyway; it has no CUDA device to bind.
            kwargs.pop("device_id", None)
            backend = "gloo"
        return new_group(*args, backend=backend, **kwargs)

    dist.new_group = gloo_group


def _run_scenario(
    scenario: RankScenario, rank: int, size: int, init_method: str, argument: Any
) -> Any:
    # torch deprecates all_gather_into_tensor; keep rank output free of the notice.
    warnings.filterwarnings("ignore", category=FutureWarning)
    # torch's CPU exp, log and kin call MKL VML. On a Xeon 8559C (MKL 2024.2) its first calls,
    # when several threads make them at once, can return one thread's part far less accurate
    # (2,500 ulp); one single-threaded call first settles it.
    torch.ones(1).exp()
    _map_device_groups_to_gloo()
    group = ParallelGroup(
        Topology(rank, size),
        device=torch.device("cpu"),
        timeout_seconds=RANK_TIMEOUT_SECONDS,
        init_method=init_method,
        use_pynccl=False,
    )
    try:
        return scenario(group, argument)
    finally:
        group.close()


def _run_rank(
    scenario: RankScenario,
    rank: int,
    size: int,
    init_method: str,
    argument: Any,
    connection,
) -> None:
    """Report every outcome through the Pipe; never let a rank print a traceback."""

    try:
        result = ("ok", _run_scenario(scenario, rank, size, init_method, argument))
    except BaseException:  # noqa: BLE001 - the parent asserts on the reported failure
        result = ("error", traceback.format_exc())
    try:
        connection.send(result)
    finally:
        connection.close()


@contextmanager
def gloo_ranks(
    scenario: RankScenario,
    size: int,
    argument: Any,
    rendezvous: Path,
) -> Iterator[list[Any]]:
    """Run ``scenario`` on ``size`` spawned ranks; yield their results in rank order.

    ``rendezvous`` is a private directory for the ranks' file rendezvous.
    """

    context = mp.get_context("spawn")
    init_method = f"file://{rendezvous}/group"
    processes, connections = [], []
    try:
        for rank in range(size):
            parent, child = context.Pipe(duplex=False)
            process = context.Process(
                target=_run_rank,
                args=(scenario, rank, size, init_method, argument, child),
                name=f"tp-gloo-rank-{rank}",
            )
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
        results = []
        for rank, connection in enumerate(connections):
            assert connection.poll(RANK_TIMEOUT_SECONDS), f"rank {rank} reported nothing"
            status, value = connection.recv()
            assert status == "ok", f"rank {rank} failed:\n{value}"
            results.append(value)
        for process in processes:
            process.join(RANK_TIMEOUT_SECONDS)
            assert process.exitcode == 0, f"{process.name} exited with {process.exitcode}"
        yield results
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
            process.close()
        for connection in connections:
            connection.close()


__all__ = ["RANK_TIMEOUT_SECONDS", "RankScenario", "gloo_ranks"]
