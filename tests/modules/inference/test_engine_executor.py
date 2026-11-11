"""The engine reaches execution only through the ``Executor`` contract: ``limits`` sizes its
cache and scheduler, and each step is submitted, then settled from its result."""

from __future__ import annotations

import pytest

import cti.inference._control._engine as engine_module
from cti.inference import (
    Forward,
    GraphStats,
    InferenceEngine,
    RequestTable,
    Round,
    ServingLimits,
)
from cti.kvcache import StateCapacity
from tests.support.inference.inference_steps import user_msg
from tests.support.inference.scheduling import sampled

_CAPACITY = StateCapacity(
    page_size=4,
    num_slots=3,
    max_seq_len=30,
    table_width=32,
    num_pages=50,
    snapshot_rows=range(3, 5),
    sink_slot=50,
    guard_slots=range(51, 51),
    snapshot_grid=4,
    export_granularity=4,
)


def _limits(*, prefix_cache: str = "naive") -> ServingLimits:
    return ServingLimits(
        capacity=_CAPACITY,
        rounds=None,
        stop_token_ids=(2,),
        vocab_size=1 << 20,
        image_rows=0,
        prefix_cache=prefix_cache,
        max_batch_tokens=16,
        max_prompt_tokens=16,
    )


class _Completion:
    def __init__(self, log: list[str], index: int) -> None:
        self._log, self._index = log, index

    def synchronize(self) -> None:
        self._log.append(f"sync:{self._index}")


class _Result:
    def __init__(self, log: list[str], index: int, rows: int) -> None:
        self.completion_event = _Completion(log, index)
        self._rows = rows

    def output(self) -> object:
        return sampled([5] * self._rows)


class _Executor:
    """Nothing but the contract: an attribute the engine reads past it fails."""

    __slots__ = ("limits", "log")

    def __init__(self, limits: ServingLimits) -> None:
        self.limits = limits
        self.log: list[str] = []

    def submit(self, op: Forward | Round, table: RequestTable) -> _Result:
        assert isinstance(op, Forward)
        index = sum(entry.startswith("submit") for entry in self.log)
        self.log.append(f"submit:{index}")
        return _Result(self.log, index, len(op.rows))

    def graph_stats(self) -> GraphStats:
        return GraphStats((), 0, 0, 0)

    def reset_inference_state(self) -> None:
        self.log.append("reset")


def _engine(executor: _Executor, replies: list[object]) -> InferenceEngine:
    return InferenceEngine(
        executor.limits,
        executor,
        publish=replies.extend,
        agree_on_step=None,
        text_decoder=None,
        overlap=True,
        checks=True,
    )


def test_the_engine_plans_submits_and_settles_through_the_executor_alone() -> None:
    executor, replies = _Executor(_limits()), []
    engine = _engine(executor, replies)

    engine.handle([user_msg(1, [10, 11, 12], max_tokens=2)], admitting=True)
    while engine.busy:
        engine.step()

    assert executor.log == ["submit:0", "submit:1", "sync:0", "sync:1"]
    assert [(reply.uid, reply.token_ids, reply.finished) for reply in replies] == [
        (1, (5,), False),
        (1, (5,), True),
    ]


def test_serving_resets_the_executors_device_state_when_it_holds_the_device_again() -> None:
    executor, replies = _Executor(_limits()), []
    engine = _engine(executor, replies)

    # Training holds the device: the request waits.
    engine.handle([user_msg(1, [10, 11, 12], max_tokens=1)], admitting=False)
    engine.switched(True, None)
    while engine.busy:
        engine.step()

    assert executor.log[:2] == ["reset", "submit:0"]
    # Refused the device, serving keeps nothing to reset.
    executor.log.clear()
    engine.switched(False, None)
    assert executor.log == []


@pytest.mark.parametrize("prefix_cache", ["naive", "radix"])
def test_the_cache_takes_the_executors_capacity(
    monkeypatch: pytest.MonkeyPatch, prefix_cache: str
) -> None:
    built: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Built(Exception):
        pass

    def record(*args: object, **kwargs: object) -> None:
        built.append((args, kwargs))
        raise Built

    monkeypatch.setattr(engine_module, "CacheManager", record)
    with pytest.raises(Built):
        _engine(_Executor(_limits(prefix_cache=prefix_cache)), [])

    ((args, kwargs),) = built
    assert args == (_CAPACITY,)
    assert kwargs == {"prefix": prefix_cache, "checks": True}
