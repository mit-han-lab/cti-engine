from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import partial
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

import cti.inference as inference_module
import cti.training.runner as training_module
import cti.worker as worker_module
from cti.config import (
    ActivationCheckpointMode,
    AdamWConfig,
    BatchLayout,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.distributed import Topology
from cti.memory import MemoryManager, Mode, native_backing
from cti.runtime._core import Rank


@pytest.fixture(autouse=True)
def _avoid_cuda_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: SimpleNamespace(device=device))
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)


class _Parallel:
    def __init__(self, events: list[str], size: int = 1) -> None:
        self._events = events
        self.size = size
        events.append("parallel+")

    def barrier(self) -> None:
        self._events.append("ranks.barrier")

    def min_max_int(self, value: int) -> tuple[int, int]:
        return value, value

    def close(self) -> None:
        self._events.append("parallel.close")


class _Memory:
    def __init__(self, events: list[str], manager: object | None) -> None:
        self._events = events
        self.manager = manager
        events.append("memory+")

    def check_training_memory(self, training: object) -> None:
        assert training is _TRAINING_PLAN
        self._events.append("memory.fits")

    def training_state_residency(self, configured: object) -> object:
        return configured


class _Host:
    def __init__(self, events: list[str], plan: object, checkpoint: object, training: bool) -> None:
        self.plan = plan
        self.checkpoint = checkpoint
        self.training = training
        self.model = object()
        self.adapter = None
        events.append("host+")

    def signature(self) -> str:
        return "test-model-signature"

    def save_adapter(self, path: str, factors: object) -> None:
        raise AssertionError("the lifecycle saves no adapter")


class _ModelRunner:
    def __init__(self, events: list[str], host: _Host, ranks: object) -> None:
        self._events = events
        self.plan = host.plan
        self.ranks = ranks
        self.model = host.model
        events.append("runner+")

    def close_graphs(self) -> None:
        self._events.append("graphs.close")


class _ExecutionStream:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name
        self.sync_error: BaseException | None = None
        self._num_recorded_events = 0

    def __enter__(self) -> None:
        self._events.append(f"{self._name}.stream+")

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        self._events.append(f"{self._name}.stream-")

    def wait_stream(self, stream: _ExecutionStream) -> None:
        self._events.append(f"{self._name}.wait:{stream._name}")

    def synchronize(self) -> None:
        self._events.append(f"{self._name}.sync")
        if self.sync_error is not None:
            raise self.sync_error

    def record_event(self) -> _CompletionEvent:
        event = _CompletionEvent(
            f"{self._name}-submission-{self._num_recorded_events}",
            self._events,
        )
        self._num_recorded_events += 1
        self._events.append(f"{self._name}.record_event")
        return event


class _TrainingMemoryManager:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def suspend_kv(self) -> None:
        self._events.append("memory.suspend_kv")

    def resume_training(self, _fewest: object) -> None:
        self._events.append("memory.resume_training")

    def suspend_training(self, *, keep_persistent: bool = False) -> None:
        self._events.append("memory.suspend_training")

    def resume_kv(self, _fewest: object) -> None:
        self._events.append("memory.resume_kv")

    @contextmanager
    def persistent_training_allocation(self) -> Iterator[None]:
        self._events.append("training.persistent+")
        try:
            yield
        finally:
            self._events.append("training.persistent-")

    @contextmanager
    def transient_training_allocation(self, needed_bytes: int) -> Iterator[None]:
        self._events.append(f"training.transient({needed_bytes})+")
        try:
            yield
        finally:
            self._events.append("training.transient-")


class _Trainer:
    def __init__(self, events: list[str], *args: object, **kwargs: object) -> None:
        self._events = events
        self.args = args
        self.kwargs = kwargs
        events.append("training+")

    def close(self) -> None:
        self._events.append("training.close")

    def memory_usage(self) -> SimpleNamespace:
        return SimpleNamespace(against=lambda _plan: "nothing held")


class _CompletionEvent:
    def __init__(
        self,
        name: str,
        call_order: list[str],
        failure: BaseException | None = None,
    ) -> None:
        self._name = name
        self._call_order = call_order
        self._failure = failure

    def synchronize(self) -> None:
        self._call_order.append(f"sync:{self._name}")
        if self._failure is not None:
            raise self._failure

    def query(self) -> bool:
        return True


def _plan(
    *, num_ranks: int = 1, local_device: int = 0, training: TrainingConfig | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        model_path="unused",
        dtype=torch.bfloat16,
        attention=SimpleNamespace(selected="fi", reason="test"),
        page_size=1,
        num_ranks=num_ranks,
        local_device=local_device,
        training=training,
        distributed_timeout=30,
        use_pynccl=True,
        max_batch_tokens=16,
        prompt_budget=SimpleNamespace(tokens=16, reason="test"),
        rounds=None,
        initialization_seed=0,
    )


def _worker(plan: SimpleNamespace, **composition: object) -> Rank:
    """A rank composed as the engine core composes one, of a fixed model with checks on;
    ``composition`` replaces any of its keywords."""

    return Rank(plan, _MODEL, **{"checks": True, **composition})  # type: ignore[arg-type]


_MODEL = SimpleNamespace(
    model_config=SimpleNamespace(hidden_size=128, vocab_size=256),
    stop_token_ids=(2,),
)


def _factories(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    def create_parallel(topology: Topology, **_kwargs: object) -> _Parallel:
        return _Parallel(events, topology.size)

    def create_memory(_plan: object, *, device: torch.device, agree, manager):
        memory = _Memory(events, manager)
        memory.device = device
        return memory

    def create_host(plan, checkpoint, *, device, ranks, training, progress):
        host = _Host(events, plan, checkpoint, training)
        host.device = device
        return host

    training_streams: list[object] = []

    def prepare_training(
        _config: object,
        _model: object,
        *,
        trainables: object,
        rank: int,
        stream: object,
        allocator: object,
        residency: object,
    ) -> SimpleNamespace:
        events.append("training.prepare")
        training_streams.append(stream)
        return _TRAINING_PLAN

    def create_model_runner(
        host: _Host,
        memory: _Memory,
        plan: object,
        *,
        ranks: _Parallel,
        preparation_stream: object,
        execution_stream: object,
        resources: ExitStack,
        beside: object | None,
        checks: bool,
        progress: object | None,
    ) -> _ModelRunner:
        assert checks and plan is host.plan and host.training == (beside is not None)
        events.append("runner.warm")
        # What runs beside it prepares once its models are warm, before it plans.
        training = None if beside is None else beside()
        training_stream = training_streams[-1] if training_streams else None
        assert len({id(preparation_stream), id(execution_stream), id(training_stream)}) == 3
        runner = _ModelRunner(events, host, ranks)
        runner.device = host.device
        runner.memory = memory
        runner.training = training
        runner.execution_stream = execution_stream
        runner.training_stream = training_stream
        resources.callback(runner.close_graphs)
        return runner

    monkeypatch.setattr(worker_module, "ParallelGroup", create_parallel)
    monkeypatch.setattr(worker_module, "DeviceMemory", create_memory)
    monkeypatch.setattr(worker_module, "ModelHost", create_host)
    monkeypatch.setattr(inference_module, "build_executor", create_model_runner)
    monkeypatch.setattr(training_module, "prepare_model_training", prepare_training)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: stream)


_TRAINING_PLAN = SimpleNamespace(
    scope_bytes=1 << 20, resident_bytes=3 << 20, training_bytes=5 << 20
)


def _training_config(
    *,
    packed_sft: bool = False,
    activation_checkpointing: ActivationCheckpointMode = ActivationCheckpointMode.NONE,
) -> TrainingConfig:
    return TrainingConfig(
        trainable_plan=TrainablePlan(("weight",)),
        optimizer=AdamWConfig(),
        batching=TrainingBatching(
            BatchLayout.PACKED if packed_sft else BatchLayout.PADDED,
            StaticBatching(microbatch_size=1, max_tokens_per_sample=64),
        ),
        activation_checkpointing=activation_checkpointing,
    )


def _stack_size() -> int:
    """The device's per-thread stack size, as the rank reads it under checks."""

    return 1024


def _worker_with_training(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manage_training_memory: bool = False,
) -> tuple[Rank, _Trainer, list[str]]:
    events: list[str] = []
    _factories(monkeypatch, events)
    engines: list[_Trainer] = []

    def create_trainer(*args: object, **kwargs: object) -> _Trainer:
        engines.append(_Trainer(events, *args, **kwargs))
        return engines[-1]

    monkeypatch.setattr(training_module, "TrainingRunner", SimpleNamespace(create=create_trainer))
    monkeypatch.setattr(training_module, "device_stack_limit", lambda: _stack_size)
    memory_manager = _TrainingMemoryManager(events) if manage_training_memory else None
    worker = _worker(_plan(training=_training_config()), memory=memory_manager)
    return worker, engines[0], events


def test_worker_constructs_parallel_before_runner_and_closes_in_reverse_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _factories(monkeypatch, events)

    worker = _worker(_plan())
    assert worker.executor.ranks is worker.worker.ranks
    assert worker.executor.training is None

    worker.close()
    worker.close()

    assert events == [
        "parallel+",
        "memory+",
        "host+",
        "runner.warm",
        "runner+",
        "graphs.close",
        "parallel.close",
    ]


def test_worker_joins_its_global_rank_on_its_local_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = {}

    def create_parallel(topology, *, device, init_method, **_kwargs):
        received.update(topology=topology, device=device, init_method=init_method)
        raise RuntimeError("stop after the group")

    monkeypatch.setattr(worker_module, "ParallelGroup", create_parallel)

    with pytest.raises(RuntimeError, match="stop after the group"):
        # Rank 1 of two, one per host: its host's first GPU.
        _worker(_plan(num_ranks=2, local_device=0), rank=1, init_method="file:///group")

    assert received == {
        "topology": Topology(1, 2),
        "device": torch.device("cuda:0"),
        "init_method": "file:///group",
    }


def test_worker_owns_and_lends_the_memory_manager_to_its_device_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _factories(monkeypatch, events)
    memory_manager = object.__new__(MemoryManager)

    worker = _worker(_plan(), memory=memory_manager)

    assert worker.worker._memory_manager is memory_manager
    assert worker.executor.memory.manager is memory_manager


def test_worker_plans_training_memory_with_its_config_on_the_built_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned: list[tuple[object, ...]] = []

    def plan(
        *args: object,
        trainables: object,
        rank: int,
        stream: object,
        allocator: object,
        residency: object,
    ) -> SimpleNamespace:
        planned.append((*args, trainables, rank, stream, allocator))
        return _TRAINING_PLAN

    events: list[str] = []
    _factories(monkeypatch, events)
    monkeypatch.setattr(training_module, "prepare_model_training", plan)
    monkeypatch.setattr(
        training_module, "TrainingRunner", SimpleNamespace(create=partial(_Trainer, events))
    )
    worker = _worker(_plan(training=_training_config()))

    # Prepared once, on the training stream and with the allocator's placement, though the
    # runner and the fit check both read it; the host's universe, which the runner trains too,
    # at no adapter's rank.
    stream, host = worker.training.kwargs["stream"], worker.training.kwargs["trainables"]
    assert isinstance(host, _Host)
    model = worker.executor.model
    assert planned == [(_training_config(), model, host, 0, stream, native_backing)]
    assert worker.executor.training is _TRAINING_PLAN


def test_worker_quiescence_fences_device_before_the_ranks_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _factories(monkeypatch, events)
    worker = _worker(_plan())
    runner = worker.executor
    runner.device = torch.device("cuda:0")
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: events.append(f"cuda.synchronize:{device}"),
    )

    worker.worker.quiesce()

    assert events[-2:] == ["cuda.synchronize:cuda:0", "ranks.barrier"]


def test_worker_constructs_training_owner_after_model_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, training, events = _worker_with_training(monkeypatch, manage_training_memory=True)
    model_runner = worker.executor

    # Training prepares once inference's ops are warm, before the runner plans; it fits once
    # the inference state it shares the device with exists.
    assert events == [
        "parallel+",
        "memory+",
        "host+",
        "runner.warm",
        "training.prepare",
        "runner+",
        "memory.fits",
        "training+",
    ]
    assert model_runner.training is _TRAINING_PLAN
    assert worker.training is training
    assert training.args[:2] == (model_runner.model, _training_config())
    signature = training.kwargs.pop("model_signature")
    assert signature() == "test-model-signature"
    # The host owns the parameter universe the runner trains.
    host = training.kwargs.pop("trainables")
    assert isinstance(host, _Host) and host.model is model_runner.model
    # It writes the trained adapter.
    assert training.kwargs.pop("adapter_writer") == host.save_adapter
    assert training.kwargs == {
        "stream": model_runner.training_stream,
        "after": model_runner.execution_stream,
        "memory": worker.worker._memory_manager,
        "scope_bytes": _TRAINING_PLAN.scope_bytes,
        "ranks": None,
        "placed": None,
        "stack_limit": _stack_size,
        "checks": True,
        "residency": None,
    }


def test_worker_releases_the_complete_kv_tail_for_every_training_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, _training, events = _worker_with_training(monkeypatch, manage_training_memory=True)
    events.clear()

    for _ in range(2):
        worker.worker.release(Mode.SERVING)
        worker.worker.acquire(Mode.TRAINING)
        worker.worker.release(Mode.TRAINING)
        worker.worker.acquire(Mode.SERVING)

    # The worker only moves the regions; training sets its own state aside.
    window = ["memory.suspend_kv", "memory.resume_training"]
    leave = ["memory.suspend_training", "memory.resume_kv"]
    assert events == [*window, *leave] * 2


def test_worker_closes_its_training_engine_before_the_model_runner_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, _training, events = _worker_with_training(monkeypatch)

    worker.close()
    worker.close()

    assert events[-3:] == ["training.close", "graphs.close", "parallel.close"]
    assert events.count("training.close") == 1


def test_every_rank_assembles_and_releases_in_the_same_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction runs collectives, so each rank of a group must build in one order."""

    orders = []
    for rank in (0, 1):
        events: list[str] = []
        _factories(monkeypatch, events)
        monkeypatch.setattr(
            training_module, "TrainingRunner", SimpleNamespace(create=partial(_Trainer, events))
        )
        plan = _plan(num_ranks=2, local_device=rank, training=_training_config())
        worker = _worker(plan, rank=rank)
        assert worker.executor.device == torch.device(f"cuda:{rank}")
        worker.close()
        orders.append(events)

    assert (
        orders[0]
        == orders[1]
        == [
            "parallel+",
            "memory+",
            "host+",
            "runner.warm",
            "training.prepare",
            "runner+",
            "memory.fits",
            "training+",
            "training.close",
            "graphs.close",
            "parallel.close",
        ]
    )
