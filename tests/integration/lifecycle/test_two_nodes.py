"""An engine whose ranks span two nodes, both on this host over localhost TCP and the CPU
stand-in: node 1's agent joins node 0, runs rank 1, and lives and fails with the engine."""

from __future__ import annotations

import multiprocessing
import multiprocessing.util
import os
import shutil
import socket
import tempfile
import time
from collections.abc import Iterator
from dataclasses import replace
from threading import Thread

import pytest

from cti.config import EngineOptions, ModelOptions, ParallelOptions, SchedulingOptions
from cti.env import ENV
from cti.llm import LLM
from cti.protocol import SamplingParams
from cti.runtime import EngineClient, EngineProcessError
from tests.support.cpu_engine import CpuEngine
from tests.support.fault_injection import Exit, FaultInjector, Raise
from tests.support.memory_saver import use_memory_saver_provider

_KEY = "two-node-test-key"


def _options(port: int, *, node_rank: int = 0, **parallel: object) -> EngineOptions:
    return EngineOptions(
        model=ModelOptions(path="cpu-model"),
        parallel=ParallelOptions(
            num_ranks=2,
            num_nodes=parallel.pop("num_nodes", 2),
            node_rank=node_rank,
            dist_init_addr=f"127.0.0.1:{port}",
        ),
        scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=256),
    )


def _run_agent(options: EngineOptions, key: str) -> None:
    os.environ["CTI_DIST_AUTHKEY"] = key
    from cti.runtime import NodeAgent

    raise SystemExit(NodeAgent(options).run())


class _Nodes:
    """Node 0's engine, in this process, and node 1's agent, a process of its own."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.agents: list[multiprocessing.Process] = []
        self.engines: list[LLM] = []

    def agent(self, options: EngineOptions, *, key: str = _KEY) -> multiprocessing.Process:
        context = multiprocessing.get_context("spawn")
        process = context.Process(target=_run_agent, args=(options, key), name="cti-node-1")
        process.start()
        self.agents.append(process)
        return process

    def engine(self, options: EngineOptions) -> LLM:
        self.engines.append(LLM(options))
        return self.engines[-1]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def nodes(monkeypatch: pytest.MonkeyPatch) -> Iterator[callable]:
    """Starts node 0 and node 1 on the CPU stand-in, with ``faults``; reclaims both after."""

    made: list[_Nodes] = []
    cleanup: list[str] = []

    def start(*faults: object) -> _Nodes:
        use_memory_saver_provider(monkeypatch)
        # The engine's ipc sockets and multiprocessing's fd server stay within AF_UNIX's limit.
        monkeypatch.setattr(tempfile, "tempdir", "/tmp")
        assert len(multiprocessing.util.get_temp_dir()) < 64
        short = tempfile.mkdtemp(prefix="cti-")
        cleanup.append(short)
        monkeypatch.setattr(tempfile, "tempdir", short)
        CpuEngine.frontend(monkeypatch)
        injector = FaultInjector((*CpuEngine.injections(), *faults))
        for key, value in injector.environment(os.environ.get("PYTHONPATH", "")).items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("CTI_DIST_AUTHKEY", _KEY)
        # This process read its environment before the key was set.
        monkeypatch.setattr("cti.runtime._nodes.ENV", replace(ENV, dist_authkey=_KEY.encode()))
        made.append(_Nodes(_free_port()))
        return made[-1]

    yield start
    for pair in made:
        for engine in pair.engines:
            engine.shutdown()
        for agent in pair.agents:
            agent.join(30)
            if agent.is_alive():
                agent.kill()
                agent.join()
    for directory in cleanup:
        shutil.rmtree(directory, ignore_errors=True)


def _sample(llm: LLM) -> tuple[int, ...]:
    params = SamplingParams(temperature=0, max_new_tokens=3)
    return tuple(llm.generate([[1, 2]], params)[0]["token_ids"])


def test_two_nodes_serve_as_one_engine_and_shut_down_together(nodes) -> None:
    pair = nodes()
    agent = pair.agent(_options(pair.port, node_rank=1))
    llm = pair.engine(_options(pair.port))

    assert _sample(llm) == (CpuEngine.TOKEN,) * 3
    llm.shutdown()
    agent.join(30)

    assert agent.exitcode == 0


def test_a_node_may_join_after_node_0_started(nodes) -> None:
    pair = nodes()
    started: list[LLM] = []
    node_0 = Thread(target=lambda: started.append(pair.engine(_options(pair.port))))
    node_0.start()
    time.sleep(3.0)  # node 0 waits for node 1
    pair.agent(_options(pair.port, node_rank=1))
    node_0.join(120)

    assert _sample(started[0]) == (CpuEngine.TOKEN,) * 3


def test_a_node_that_starts_first_waits_for_node_0(nodes) -> None:
    pair = nodes()
    pair.agent(_options(pair.port, node_rank=1))
    time.sleep(3.0)  # node 1 redials until node 0 listens
    llm = pair.engine(_options(pair.port))

    assert _sample(llm) == (CpuEngine.TOKEN,) * 3


def test_a_node_that_never_joins_fails_the_startup(nodes, monkeypatch) -> None:
    pair = nodes()
    monkeypatch.setattr(EngineClient, "STARTUP_STALL_SECONDS", 2.0)

    with pytest.raises(TimeoutError, match=r"node \[1\] did not join in time"):
        pair.engine(_options(pair.port))


@pytest.mark.parametrize(
    ("agent_options", "reason"),
    (
        (
            lambda port: replace(
                _options(port, node_rank=1), model=ModelOptions(path="cpu-model", dtype="auto")
            ),
            "its model, parallel layout, engine or kernel versions, or checkpoint differ",
        ),
        (
            lambda port: replace(
                _options(port),
                parallel=ParallelOptions(
                    num_ranks=4, num_nodes=4, node_rank=1, dist_init_addr=f"127.0.0.1:{port}"
                ),
            ),
            "num_nodes 4 differs from node 0's 2",
        ),
    ),
    ids=("options", "num-nodes"),
)
def test_node_0_refuses_a_node_that_runs_something_else(nodes, agent_options, reason) -> None:
    pair = nodes()
    agent = pair.agent(agent_options(pair.port))

    with pytest.raises(RuntimeError, match=f"refused a node: {reason}"):
        pair.engine(_options(pair.port))
    agent.join(30)

    assert agent.exitcode == 1


def test_a_node_without_the_shared_key_is_not_admitted(nodes, monkeypatch) -> None:
    pair = nodes()
    monkeypatch.setattr(EngineClient, "STARTUP_STALL_SECONDS", 3.0)
    agent = pair.agent(_options(pair.port, node_rank=1), key="another-key")

    with pytest.raises(TimeoutError, match="did not join in time"):
        pair.engine(_options(pair.port))
    agent.join(30)

    assert agent.exitcode == 1


def test_a_remote_rank_that_dies_fails_node_0_and_its_agent(nodes) -> None:
    pair = nodes(Exit("tests.support.cpu_engine:CpuWorker.submit", "cti-rank-1"))
    agent = pair.agent(_options(pair.port, node_rank=1))
    llm = pair.engine(_options(pair.port))

    with pytest.raises(EngineProcessError, match="rank 1's link from its node closed"):
        _sample(llm)
    agent.join(30)

    assert agent.exitcode == 1


def test_node_0_failing_reclaims_the_other_node(nodes) -> None:
    pair = nodes(Raise("tests.support.cpu_engine:CpuWorker.submit", "cti-rank-0"))
    agent = pair.agent(_options(pair.port, node_rank=1))
    llm = pair.engine(_options(pair.port))

    with pytest.raises(EngineProcessError, match="on rank 0"):
        _sample(llm)
    agent.join(30)

    assert agent.exitcode == 1
    assert not agent.is_alive()


def test_only_node_0_runs_the_engine() -> None:
    with pytest.raises(ValueError, match="only node 0 runs the engine"):
        EngineClient(_options(1, node_rank=1))


@pytest.mark.parametrize(
    ("num_ranks", "num_nodes", "node_rank", "ranks", "devices"),
    (
        (2, 2, 1, range(1, 2), (0,)),
        (4, 2, 1, range(2, 4), (0, 1)),
        (4, 1, 0, range(4), (0, 1, 2, 3)),
    ),
)
def test_each_node_runs_an_equal_share_of_the_ranks_on_its_own_devices(
    num_ranks: int, num_nodes: int, node_rank: int, ranks: range, devices: tuple[int, ...]
) -> None:
    parallel = ParallelOptions(
        num_ranks=num_ranks, num_nodes=num_nodes, node_rank=node_rank, dist_init_addr="a:1"
    )

    assert parallel.node_ranks == ranks
    assert tuple(parallel.local_device(rank) for rank in ranks) == devices


def test_a_node_runs_node_0s_options_beyond_what_it_resolves_itself(nodes) -> None:
    """Scheduling, like training, comes from node 0: node 1's command line need not spell it."""

    pair = nodes()
    agent = pair.agent(
        replace(
            _options(pair.port, node_rank=1),
            scheduling=SchedulingOptions(max_running_requests=2, max_batch_tokens=128),
        )
    )
    llm = pair.engine(_options(pair.port))

    assert _sample(llm) == (CpuEngine.TOKEN,) * 3
    llm.shutdown()
    agent.join(30)
    assert agent.exitcode == 0
