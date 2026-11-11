from __future__ import annotations

import gc
import logging
import re
import weakref
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

import cti.distributed.runtime as runtime_module
from cti.distributed import Topology
from cti.distributed.runtime import ParallelGroup
from cti.utils.logger import init_logger


class _Work:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    def wait(self) -> None:
        self._events.append(f"{self._name}.wait")


class _Group:
    def __init__(self, events: list[str], name: str, backend: str) -> None:
        self.events = events
        self.name = name
        self.backend = backend

    def barrier(self) -> _Work:
        self.events.append(f"{self.name}.barrier")
        return _Work(self.events, f"{self.name}.barrier")

    def broadcast(self, _tensor: torch.Tensor, *, root: int) -> _Work:
        self.events.append(f"{self.name}.broadcast:{root}")
        return _Work(self.events, f"{self.name}.broadcast")


class _Store:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.port = 32123

    def __del__(self) -> None:
        self._events.append("store-")


class _FakeDistributed:
    ProcessGroup = _Group
    Store = object
    ReduceOp = SimpleNamespace(MIN="min", SUM="sum")

    def __init__(
        self,
        *,
        fail_stage: str | None = None,
        fail_destroy: str | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self.events: list[str] = []
        self.fail_stage = fail_stage
        self.fail_destroy = fail_destroy
        self.failure = failure
        self.store_calls: list[dict[str, object]] = []
        self.default_group_calls: list[dict[str, object]] = []
        self.subgroup_calls: list[dict[str, object]] = []
        self.group = SimpleNamespace(WORLD=None)
        self._initialized = False

    def _raise_failure(self, stage: str) -> None:
        if self.fail_stage == stage:
            raise self.failure or RuntimeError(f"{stage} failed")

    def TCPStore(self, **kwargs: object) -> _Store:
        self.events.append("store+")
        self.store_calls.append(kwargs)
        self._raise_failure("store")
        return _Store(self.events)

    def init_process_group(self, **kwargs: object) -> None:
        self.events.append("default+")
        call = dict(kwargs)
        store = call.get("store")
        if store is not None:
            call["store"] = weakref.ref(store)
        self.default_group_calls.append(call)
        self._initialized = True
        self.group.WORLD = _Group(self.events, "default", "gloo")
        self._raise_failure("default")

    def new_group(self, *, backend: str, **kwargs: object) -> _Group:
        name = "cpu" if backend == "gloo" else "device"
        self.events.append(f"{name}+")
        self.subgroup_calls.append({"backend": backend, **kwargs})
        self._raise_failure(name)
        return _Group(self.events, name, backend)

    def destroy_process_group(self, group: _Group | None = None) -> None:
        group = self.group.WORLD if group is None else group
        assert group is not None
        self.events.append(f"{group.name}-")
        if group is self.group.WORLD:
            self.group.WORLD = None
            self._initialized = False
        if self.fail_destroy == group.name:
            raise RuntimeError(f"{group.name} cleanup failed")

    def is_initialized(self) -> bool:
        return self._initialized

    @staticmethod
    def get_backend(group: _Group) -> str:
        return group.backend

    def all_reduce(self, *_args, **_kwargs) -> None:
        self.events.append("native.all_reduce")

    def all_gather_into_tensor(self, *_args, **_kwargs) -> None:
        self.events.append("native.all_gather")


def _group(monkeypatch: pytest.MonkeyPatch, native: _FakeDistributed) -> ParallelGroup:
    monkeypatch.setattr(runtime_module, "dist", native)
    monkeypatch.setattr(
        runtime_module,
        "_PyNCCL",
        lambda *_args, **_kwargs: pytest.fail("a size-one group must not construct PyNCCL"),
    )
    return ParallelGroup(
        Topology(0, 1),
        device=torch.device("cuda", 0),
        timeout_seconds=1,
        # The client names a rendezvous on every rank count; one rank keeps its private Store.
        init_method="file:///private/group",
        use_pynccl=True,
    )


def _new_group(rank: int, size: int, **options: object) -> ParallelGroup:
    arguments: dict[str, object] = {
        "device": torch.device("cuda", rank),
        "timeout_seconds": 1,
        "init_method": "file:///private/group",
        "use_pynccl": False,
    }
    arguments.update(options)
    return ParallelGroup(Topology(rank, size), **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(("rank", "size"), [(-1, 2), (2, 2), (0, 0)])
def test_topology_rejects_a_rank_outside_its_group(rank: int, size: int) -> None:
    with pytest.raises(ValueError, match="is not in a group"):
        Topology(rank, size)


def test_pynccl_communicator_reduces_in_place_without_a_staging_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctikernels as kernel_module

    arguments: dict[str, object] = {}
    reduced: list[tuple[torch.Tensor, str]] = []

    class _Communicator:
        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            reduced.append((tensor, op))

    def init_pynccl(**kwargs: object) -> _Communicator:
        arguments.update(kwargs)
        return _Communicator()

    monkeypatch.setattr(kernel_module, "init_pynccl", init_pynccl)
    cpu_group = _Group([], "cpu", "gloo")

    pynccl = runtime_module._PyNCCL(Topology(1, 2), cpu_group)  # type: ignore[arg-type]

    # No max_size_bytes: the communicator maps no symmetric window and copies nothing.
    assert arguments == {"tp_rank": 1, "tp_size": 2, "tp_cpu_group": cpu_group}
    tensor = torch.tensor([1.0])
    assert pynccl.all_reduce(tensor) is tensor
    assert reduced == [(tensor, "sum")]


def test_size_one_group_is_real_for_control_and_local_for_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeDistributed()
    group = _group(monkeypatch, native)

    assert native.store_calls == [
        {
            "host_name": "127.0.0.1",
            "port": 0,
            "world_size": 1,
            "is_master": True,
            "timeout": timedelta(seconds=1),
            "wait_for_workers": True,
        }
    ]
    store_ref = native.default_group_calls[0]["store"]
    assert isinstance(store_ref, weakref.ReferenceType)
    assert isinstance(store_ref(), _Store)
    assert native.default_group_calls == [
        {
            "backend": "gloo",
            "rank": 0,
            "world_size": 1,
            "timeout": timedelta(seconds=1),
            "store": store_ref,
            "init_method": None,
        }
    ]
    assert native.subgroup_calls == [
        {"backend": "gloo", "ranks": [0], "timeout": timedelta(seconds=1)},
        {"backend": "nccl", "ranks": [0], "timeout": timedelta(seconds=1)},
    ]
    assert "device_id" not in native.subgroup_calls[1]

    assert (group.rank, group.size, group.is_primary()) == (0, 1, True)
    assert group.backends == ("gloo", "nccl")
    group.barrier()
    group.broadcast_cpu(torch.tensor(1), root=0)
    assert group.min_max_int(17) == (17, 17)
    group.require_same(17, "one rank")

    tensor = torch.tensor([1.0])
    assert group.all_reduce(tensor) is tensor
    assert group.all_gather(tensor) is tensor
    group.broadcast_device(tensor)
    assert "native.all_reduce" not in native.events
    assert "native.all_gather" not in native.events

    group.close()
    group.close()

    assert native.events == [
        "store+",
        "default+",
        "cpu+",
        "device+",
        "cpu.barrier",
        "cpu.barrier.wait",
        "cpu.broadcast:0",
        "cpu.broadcast.wait",
        "device-",
        "cpu-",
        "default-",
        "store-",
    ]
    assert not native.is_initialized()


class _HalfFloatGather:
    """PyNCCL's gather: 16-bit floats only; every rank sends as many bytes as rank 0."""

    def __init__(self, ranks: list[torch.Tensor]) -> None:
        self._bytes = [rank.reshape(-1).view(torch.uint8) for rank in ranks]

    def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None:
        assert input.dtype in (torch.float16, torch.bfloat16)
        sent = input.view(torch.uint8)
        assert torch.equal(sent[: self._bytes[0].numel()], self._bytes[0])
        # Another rank's pad byte holds whatever its buffer held.
        padded = [torch.cat((b, b.new_full((sent.numel() - b.numel(),), 255))) for b in self._bytes]
        output.view(torch.uint8).copy_(torch.cat(padded))


@pytest.mark.parametrize("offset", (0, 1))
@pytest.mark.parametrize("shape", ((3,), (2, 3)))
@pytest.mark.parametrize(
    "dtype", (torch.int64, torch.float32, torch.bfloat16, torch.uint8, torch.bool)
)
def test_pynccl_gathers_any_tensor(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, shape: tuple[int, ...], offset: int
) -> None:
    import ctikernels as kernel_module

    numel = torch.Size(shape).numel()
    ranks = [
        ((torch.arange(numel + 1) * 3 + r) % 7).to(dtype)[offset : offset + numel].view(shape)
        for r in range(2)
    ]
    monkeypatch.setattr(kernel_module, "init_pynccl", lambda **_: _HalfFloatGather(ranks))
    pynccl = runtime_module._PyNCCL(Topology(0, 2), _Group([], "cpu", "gloo"))  # type: ignore[arg-type]

    assert torch.equal(pynccl.all_gather(ranks[0]), torch.cat(ranks))


def test_multi_rank_group_connects_its_device_communicator_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeDistributed()
    monkeypatch.setattr(runtime_module, "dist", native)

    group = _new_group(1, 2)

    assert native.store_calls == []
    assert native.default_group_calls[0]["init_method"] == "file:///private/group"
    # The device group connects before any memory sizing or first collective.
    assert native.subgroup_calls[1] == {
        "backend": "nccl",
        "ranks": [0, 1],
        "timeout": timedelta(seconds=1),
        "device_id": torch.device("cuda", 1),
    }
    assert (group.rank, group.size, group.is_primary()) == (1, 2, False)
    group.close()
    assert native.events[-3:] == ["device-", "cpu-", "default-"]


def test_multi_rank_group_runs_its_collectives_over_pynccl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeDistributed()
    monkeypatch.setattr(runtime_module, "dist", native)
    built: list[tuple[Topology, object]] = []

    class _FakePyNCCL:
        reduces = staticmethod(runtime_module._PyNCCL.reduces)

        def __init__(self, topology: Topology, cpu_group: object) -> None:
            built.append((topology, cpu_group))

        def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
            native.events.append("pynccl.all_reduce")
            return tensor

    monkeypatch.setattr(runtime_module, "_PyNCCL", _FakePyNCCL)

    group = _new_group(0, 2, use_pynccl=True)

    ((topology, cpu_group),) = built
    assert topology == Topology(0, 2)
    assert isinstance(cpu_group, _Group) and cpu_group.backend == "gloo"
    tensor = torch.tensor([1.0], dtype=torch.bfloat16)
    assert group.all_reduce(tensor) is tensor
    assert native.events[-1] == "pynccl.all_reduce"
    assert "native.all_reduce" not in native.events
    # PyNCCL sums 16-bit floats only: an FP32 sum takes the native device group.
    group.all_reduce(torch.tensor([1.0]))
    assert native.events[-1] == "native.all_reduce"
    group.close()


def test_multi_rank_group_requires_its_rendezvous(monkeypatch: pytest.MonkeyPatch) -> None:
    native = _FakeDistributed()
    monkeypatch.setattr(runtime_module, "dist", native)

    with pytest.raises(ValueError, match="rendezvous"):
        _new_group(0, 2, init_method=None)
    assert native.events == []


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("store", ["store+"]),
        ("default", ["store+", "default+", "default-", "store-"]),
        ("cpu", ["store+", "default+", "cpu+", "default-", "store-"]),
        (
            "device",
            ["store+", "default+", "cpu+", "device+", "cpu-", "default-", "store-"],
        ),
    ],
)
def test_partial_construction_releases_only_acquired_resources(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    expected: list[str],
) -> None:
    primary = RuntimeError(f"{stage} failed")
    native = _FakeDistributed(fail_stage=stage, failure=primary)
    monkeypatch.setattr(runtime_module, "dist", native)

    with pytest.raises(RuntimeError) as caught:
        _new_group(0, 1, use_pynccl=True)
    assert caught.value is primary
    store_ref = native.default_group_calls[0]["store"] if stage != "store" else None
    primary.__traceback__ = None
    del caught
    # pytest owns the caught traceback until this point; collecting it verifies
    # that the failed ParallelGroup retained no Store reference after cleanup.
    gc.collect()
    if store_ref is not None:
        assert isinstance(store_ref, weakref.ReferenceType)
        assert store_ref() is None
    assert native.events == expected
    assert not native.is_initialized()


def _log_rank_prefix() -> int | None:
    init_logger("cti.tests.parallel_runtime")
    record = logging.LogRecord("cti.tests", logging.INFO, __file__, 1, "message", None, None)
    # The package handler comes first; pytest adds its capture handlers after it.
    handler = logging.getLogger("cti").handlers[0]
    assert handler.filter(record)
    match = re.search(r"\|rank=(\d+)\]", handler.format(record))
    return None if match is None else int(match.group(1))


def test_a_rank_drops_rank0_records_off_rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    native = _FakeDistributed()
    monkeypatch.setattr(runtime_module, "dist", native)
    init_logger("cti.tests.parallel_runtime")
    # The package handler comes first; pytest adds its capture handlers after it.
    handler = logging.getLogger("cti").handlers[0]
    record = logging.LogRecord("cti.tests", logging.INFO, __file__, 1, "m", None, None)
    record.rank0 = True

    group = _new_group(1, 2, device=torch.device("cpu"), init_method="tcp://unused")
    assert not handler.filter(record)
    group.close()
    assert handler.filter(record)


def test_each_group_owns_the_log_rank_for_its_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeDistributed()
    monkeypatch.setattr(runtime_module, "dist", native)

    for rank, size in ((0, 1), (1, 2)):
        group = _new_group(rank, size, device=torch.device("cpu"), init_method="tcp://unused")
        assert _log_rank_prefix() == rank
        # A second group is refused before it touches the live group's rank.
        with pytest.raises(RuntimeError, match="initialized default process group"):
            _group(monkeypatch, native)
        assert _log_rank_prefix() == rank
        group.close()
        assert _log_rank_prefix() is None


def test_failed_construction_clears_the_log_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module, "dist", _FakeDistributed(fail_stage="default"))

    with pytest.raises(RuntimeError, match="default failed"):
        _new_group(0, 1, device=torch.device("cpu"), init_method="unused", use_pynccl=True)

    assert _log_rank_prefix() is None


def test_cleanup_failure_does_not_skip_later_owners_or_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _FakeDistributed(fail_destroy="device")
    group = _group(monkeypatch, native)

    with pytest.raises(RuntimeError, match="device cleanup failed"):
        group.close()
    events = list(native.events)
    group.close()

    assert native.events == events
    assert native.events[-4:] == ["device-", "cpu-", "default-", "store-"]
    assert not native.is_initialized()
