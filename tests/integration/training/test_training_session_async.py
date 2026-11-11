from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

import pytest

from cti.runtime import TrainingFuture, TrainingSession
from cti.training import (
    ForwardBackwardCompletion,
    OptimizerStepResult,
    PackedTokens,
    TrainingBatch,
    TrainingVerb,
)
from cti.training._protocol import ForwardBackwardObservation
from tests.support.training import packed_sft_batch

T = TypeVar("T")


def _batch() -> TrainingBatch:
    return packed_sft_batch(((1, 2, 3),))


async def _await_future(future: TrainingFuture[T]) -> T:
    return await future


class _SuccessfulEndpoint:
    def __init__(self) -> None:
        self.completion = ForwardBackwardCompletion(3.0, 2, {})
        self.step = OptimizerStepResult(1, 2, 0.25)
        self.num_forward_observations = 0
        self.num_optimizer_observations = 0
        self.events: list[str] = []

    def call(self, verb: TrainingVerb, *args: object) -> object:
        """The client's one training call, answered by the method its verb names."""

        return getattr(self, verb.value)(*args)

    def enter_training_session(self) -> None:
        self.events.append("enter")

    def submit_forward_backward(self, batch: TrainingBatch) -> int:
        assert isinstance(batch.tokens, PackedTokens)
        self.events.append("submit_forward_backward")
        return 0

    def observe_forward_backward(
        self,
        submission_id: int,
    ) -> tuple[ForwardBackwardObservation, ...]:
        assert submission_id == 0
        self.num_forward_observations += 1
        self.events.append("observe_forward_backward")
        return (ForwardBackwardObservation(0, self.completion),)

    def submit_optimizer_step(self) -> int:
        self.events.append("submit_optim_step")
        return 0

    def observe_optimizer_step(
        self,
        submission_id: int,
    ) -> tuple[OptimizerStepResult, tuple[ForwardBackwardObservation, ...]]:
        assert submission_id == 0
        self.num_optimizer_observations += 1
        self.events.append("observe_optim_step")
        observations = (
            ()
            if self.num_forward_observations
            else (ForwardBackwardObservation(0, self.completion),)
        )
        return self.step, observations

    def discard_accumulation(self) -> None:
        self.events.append("discard")

    def save_training_checkpoint(self, path: str) -> None:
        assert path
        self.events.append("checkpoint")

    def pause_for_serving(self) -> None:
        self.events.append("pause_for_serving")

    def resume_training(self) -> None:
        self.events.append("resume_training")

    def close_training_session(self) -> None:
        self.events.append("close")

    def abort_training_session(self) -> None:
        self.events.append("abort")


def test_concurrent_optimizer_waiters_share_one_observation() -> None:
    endpoint = _SuccessfulEndpoint()
    session = TrainingSession(object(), endpoint, checks=True)
    session.__enter__()
    forward = session.forward_backward(_batch())
    optimizer = session.optim_step()

    with ThreadPoolExecutor(max_workers=2) as executor:
        waiters = tuple(executor.submit(optimizer.result) for _ in range(2))
        assert tuple(waiter.result(timeout=2) for waiter in waiters) == (
            endpoint.step,
            endpoint.step,
        )

    assert endpoint.num_optimizer_observations == 1
    assert forward.result() == endpoint.completion
    assert optimizer.result() == endpoint.step
    session.close()


def test_observing_a_later_forward_future_settles_the_fifo_prefix() -> None:
    endpoint = _SuccessfulEndpoint()
    second_completion = ForwardBackwardCompletion(5.0, 2, {})

    def submit(batch: TrainingBatch) -> int:
        del batch
        submission_id = endpoint.events.count("submit_forward_backward")
        endpoint.events.append("submit_forward_backward")
        return submission_id

    def observe(
        submission_id: int,
    ) -> tuple[ForwardBackwardObservation, ...]:
        assert submission_id == 1
        endpoint.num_forward_observations += 1
        return (
            ForwardBackwardObservation(0, endpoint.completion),
            ForwardBackwardObservation(1, second_completion),
        )

    endpoint.submit_forward_backward = submit  # type: ignore[method-assign]
    endpoint.observe_forward_backward = observe  # type: ignore[method-assign]
    session = TrainingSession(object(), endpoint, checks=True)
    session.__enter__()
    first = session.forward_backward(_batch())
    second = session.forward_backward(_batch())

    assert second.result() == second_completion
    assert first.result() == endpoint.completion
    assert endpoint.num_forward_observations == 1
    session.discard_accumulation()
    session.close()


def test_cancelled_await_does_not_cancel_an_accepted_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _SuccessfulEndpoint()

    async def run_test() -> None:
        observation_started = asyncio.Event()
        observation_release = asyncio.Event()

        async def controlled_to_thread(
            operation: Callable[[], T],
        ) -> T:
            observation_started.set()
            await observation_release.wait()
            return operation()

        monkeypatch.setattr(asyncio, "to_thread", controlled_to_thread)
        session = TrainingSession(object(), endpoint, checks=True)
        session.__enter__()
        future = session.forward_backward(_batch())
        waiter = asyncio.create_task(_await_future(future))
        await observation_started.wait()

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        observation_release.set()
        assert await future == endpoint.completion
        session.discard_accumulation()
        session.close()

    asyncio.run(run_test())
    assert endpoint.num_forward_observations == 1


def test_future_retries_if_its_shared_observation_task_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _SuccessfulEndpoint()

    async def run_test() -> None:
        first_observation_started = asyncio.Event()
        never_release = asyncio.Event()
        attempt = 0

        async def controlled_to_thread(operation: Callable[[], T]) -> T:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                first_observation_started.set()
                await never_release.wait()
            return operation()

        monkeypatch.setattr(asyncio, "to_thread", controlled_to_thread)
        session = TrainingSession(object(), endpoint, checks=True)
        session.__enter__()
        future = session.forward_backward(_batch())
        waiter = asyncio.create_task(_await_future(future))
        await first_observation_started.wait()

        observation_task = future._async_task
        assert observation_task is not None
        observation_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await asyncio.sleep(0)

        assert await future == endpoint.completion
        assert endpoint.num_forward_observations == 1
        session.discard_accumulation()
        session.close()

    asyncio.run(run_test())


def test_cancelled_async_entry_before_thread_start_does_not_enter_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _SuccessfulEndpoint()

    async def run_test() -> None:
        handoff_created = asyncio.Event()
        never_start = asyncio.Event()

        async def pending_to_thread(operation: Callable[[], T]) -> T:
            handoff_created.set()
            await never_start.wait()
            return operation()

        monkeypatch.setattr(asyncio, "to_thread", pending_to_thread)
        session = TrainingSession(object(), endpoint, checks=True)
        entering = asyncio.create_task(session.__aenter__())
        await handoff_created.wait()
        entering.cancel()

        with pytest.raises(asyncio.CancelledError):
            await entering
        assert endpoint.events == []

    asyncio.run(run_test())


def test_cancelled_async_entry_after_thread_start_aborts_accepted_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _SuccessfulEndpoint()

    async def run_test() -> None:
        entry_started = asyncio.Event()
        release_entry = asyncio.Event()

        async def started_to_thread(operation: Callable[[], T]) -> T:
            result = operation()
            entry_started.set()
            await release_entry.wait()
            return result

        monkeypatch.setattr(asyncio, "to_thread", started_to_thread)
        session = TrainingSession(object(), endpoint, checks=True)
        entering = asyncio.create_task(session.__aenter__())
        await asyncio.wait_for(entry_started.wait(), timeout=2)
        entering.cancel()
        release_entry.set()

        with pytest.raises(asyncio.CancelledError):
            await entering
        assert endpoint.events == ["enter", "abort"]

    asyncio.run(run_test())


def test_async_context_keeps_the_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _SuccessfulEndpoint()

    async def run_test() -> None:
        async def yielding_to_thread(
            operation: Callable[[], T],
        ) -> T:
            await asyncio.sleep(0)
            return operation()

        monkeypatch.setattr(asyncio, "to_thread", yielding_to_thread)
        session = TrainingSession(object(), endpoint, checks=True)
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        entering = asyncio.create_task(session.__aenter__())
        assert await entering is session
        assert heartbeat.is_set()

        heartbeat.clear()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        closing = asyncio.create_task(session.__aexit__(None, None, None))
        assert await closing is False
        assert heartbeat.is_set()

    asyncio.run(run_test())


def test_observation_failure_is_replayed_without_reusing_the_exception() -> None:
    endpoint = _SuccessfulEndpoint()

    def fail_observation(
        submission_id: int,
    ) -> tuple[ForwardBackwardObservation, ...]:
        assert submission_id == 0
        raise ValueError("observation failed")

    endpoint.observe_forward_backward = fail_observation  # type: ignore[method-assign]
    session = TrainingSession(object(), endpoint, checks=True)
    session.__enter__()
    future = session.forward_backward(_batch())

    with pytest.raises(ValueError, match="observation failed") as first:
        future.result()
    with pytest.raises(ValueError, match="observation failed") as second:
        future.result()
    assert first.value is not second.value

    session.discard_accumulation()
    session.close()


def test_accepted_failure_settles_only_its_own_future() -> None:
    endpoint = _SuccessfulEndpoint()
    num_submissions = 0

    def submit(batch: TrainingBatch) -> int:
        nonlocal num_submissions
        del batch
        submission_id = num_submissions
        num_submissions += 1
        return submission_id

    def observe(
        submission_id: int,
    ) -> tuple[ForwardBackwardObservation, ...]:
        if submission_id == 1:
            raise ValueError("accepted execution failed")
        assert submission_id == 0
        return (ForwardBackwardObservation(0, endpoint.completion),)

    endpoint.submit_forward_backward = submit  # type: ignore[method-assign]
    endpoint.observe_forward_backward = observe  # type: ignore[method-assign]
    session = TrainingSession(object(), endpoint, checks=True)
    session.__enter__()
    first = session.forward_backward(_batch())
    second = session.forward_backward(_batch())

    with pytest.raises(ValueError, match="accepted execution failed"):
        second.result()
    assert first.result() == endpoint.completion
    with pytest.raises(ValueError, match="accepted execution failed"):
        second.result()

    session.discard_accumulation()
    session.close()


def test_explicit_serving_pause_preserves_the_training_session() -> None:
    endpoint = _SuccessfulEndpoint()
    session = TrainingSession(object(), endpoint, checks=True)
    session.__enter__()
    completion = session.forward_backward(_batch())

    session.pause_for_serving()
    session.resume_training()
    assert completion.result() == endpoint.completion
    assert session.optim_step().result() == endpoint.step
    session.close()
    assert endpoint.events == [
        "enter",
        "submit_forward_backward",
        "pause_for_serving",
        "resume_training",
        "observe_forward_backward",
        "submit_optim_step",
        "observe_optim_step",
        "close",
    ]


def test_async_serving_transitions_do_not_block_the_event_loop() -> None:
    endpoint = _SuccessfulEndpoint()

    async def run_test() -> None:
        session = TrainingSession(object(), endpoint, checks=True)
        session.__enter__()

        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await session.pause_for_serving_async()
        assert heartbeat.is_set()

        heartbeat.clear()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await session.resume_training_async()
        assert heartbeat.is_set()
        session.close()

    asyncio.run(run_test())
    assert endpoint.events == [
        "enter",
        "pause_for_serving",
        "resume_training",
        "close",
    ]


@pytest.mark.parametrize("optimizer_failed", [False, True])
def test_a_rejected_abort_keeps_the_session_open_for_retry(optimizer_failed):
    from cti.training import InvalidGradientError, SwitchRejectedError

    endpoint = _SuccessfulEndpoint()
    session = TrainingSession(object(), endpoint, checks=True)
    session.__enter__()
    if optimizer_failed:

        def failed_step(submission_id):
            raise InvalidGradientError("non-finite gradient")

        endpoint.observe_optimizer_step = failed_step
        session.forward_backward(_batch())
        session.optim_step()

    def rejected():
        raise SwitchRejectedError("restoring KV memory needs 2 GiB")

    endpoint.abort_training_session = rejected
    error = ValueError("body failed")
    assert not session.__exit__(ValueError, error, None)

    # The retryable outcome wins over the optimizer's, and the session stays open.
    assert "remains open" in error.__notes__[0]
    assert ("non-finite gradient" in "".join(error.__notes__)) is optimizer_failed
    assert not session._closed
    session.close()
    assert endpoint.events[-1] == "close"


def test_a_rejected_close_keeps_the_session_open_ahead_of_an_optimizer_error():
    from cti.training import InvalidGradientError, SwitchRejectedError

    endpoint = _SuccessfulEndpoint()
    session = TrainingSession(object(), endpoint, checks=True)
    session.__enter__()

    def failed_step(submission_id):
        raise InvalidGradientError("non-finite gradient")

    def rejected():
        raise SwitchRejectedError("restoring KV memory needs 2 GiB")

    endpoint.observe_optimizer_step = failed_step
    close = endpoint.close_training_session
    endpoint.close_training_session = rejected
    session.forward_backward(_batch())
    session.optim_step()

    # As for abort, the retryable outcome comes first and carries the optimizer's.
    with pytest.raises(SwitchRejectedError) as raised:
        session.close()
    assert "non-finite gradient" in "".join(raised.value.__notes__)
    assert not session._closed

    # Once memory allows, the retry closes.
    endpoint.close_training_session = close
    session.close()
    assert session._closed and endpoint.events[-1] == "close"
