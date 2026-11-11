from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from threading import Lock
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from cti import training
from cti.runtime._client import EngineProcessError
from cti.training import (
    ForwardBackwardCompletion,
    ForwardBackwardObservation,
    OptimizerStepResult,
    SwitchRejectedError,
    TrainingVerb,
)

if TYPE_CHECKING:
    from cti.runtime._client import EngineClient

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _FailureSnapshot:
    """Replay one failure without retaining its traceback or object graph."""

    error_type: type[BaseException]
    message: str
    notes: tuple[str, ...]

    @classmethod
    def capture(cls, error: BaseException) -> _FailureSnapshot:
        return cls(
            error_type=type(error),
            message=str(error),
            notes=tuple(getattr(error, "__notes__", ())),
        )

    def replay(self) -> BaseException:
        try:
            error = self.error_type(self.message)
        except Exception:  # noqa: BLE001 - arbitrary exception constructors vary
            error = RuntimeError(
                f"{self.error_type.__module__}.{self.error_type.__qualname__}: {self.message}"
            )
        for note in self.notes:
            error.add_note(note)
        return error


def _carry(refused: BaseException, pending_error: BaseException | None) -> None:
    """Note on a retryable refusal the pending optimizer step's failure, which its retry
    no longer reports."""

    if pending_error is not None:
        refused.add_note(f"The pending optimizer step failed: {pending_error!r}")


class _LifecycleCallCancelled(Exception):
    """Internal marker for a lifecycle call cancelled before its thread starts."""


class TrainingFuture(Generic[T]):
    """Lightweight result handle for one accepted training command; ``checks`` runs its
    internal invariant checks."""

    __slots__ = ("_async_task", "_checks", "_error", "_resolver", "_result", "_submission_id")

    def __init__(
        self,
        submission_id: int,
        resolver: Callable[[TrainingFuture[T]], T],
        *,
        checks: bool,
    ) -> None:
        self._submission_id = submission_id
        self._checks = checks
        self._resolver: Callable[[TrainingFuture[T]], T] | None = resolver
        self._result: T | None = None
        self._error: _FailureSnapshot | None = None
        self._async_task: asyncio.Task[T] | None = None

    def result(self) -> T:
        resolver = self._resolver
        if self._result is not None:
            return self._result
        if self._error is not None:
            raise self._error.replay()
        if resolver is not None:
            return resolver(self)
        if self._checks:
            raise RuntimeError("training future has no resolver or terminal result")
        return self._result

    async def _wait(self) -> T:
        if self._resolver is None:
            return self.result()
        task = self._async_task
        if task is not None and task.cancelled():
            self._async_task = None
            task = None
        if task is None:
            task = asyncio.create_task(asyncio.to_thread(self.result))
            task.add_done_callback(self._consume_async_completion)
            self._async_task = task
        return await asyncio.shield(task)

    def __await__(self):
        return self._wait().__await__()

    def _set_result(self, result: T) -> None:
        if self._result is None and self._error is None:
            self._result = result
            self._resolver = None

    def _set_error(self, error: BaseException) -> None:
        if self._result is None and self._error is None:
            self._error = _FailureSnapshot.capture(error)
            self._resolver = None

    def _consume_async_completion(self, task: asyncio.Task[T]) -> None:
        if task.cancelled():
            if self._async_task is task:
                self._async_task = None
            return
        task.exception()


def _resolved_path(path: str | os.PathLike[str], what: str) -> str:
    raw_path = os.fspath(path)
    if type(raw_path) is not str:
        raise TypeError(f"{what} path must resolve to a string")
    if not raw_path:
        raise ValueError(f"{what} path must not be empty")
    return os.path.abspath(os.path.expanduser(raw_path))


class TrainingSession:
    """Single-writer training commands with concurrency-safe result observation."""

    def __init__(self, owner: object, channel: EngineClient, *, checks: bool) -> None:
        """``checks`` runs the session's internal invariant checks."""

        # Keep the public LLM and its finalizer alive for this session's lifetime.
        self._owner = owner
        self._channel = channel
        self._checks = checks
        # Facts about this handle; the engine's scheduler owns the session's state.
        self._entered = False
        self._closed = False
        self._state_lock = Lock()
        self._async_lifecycle_lock = asyncio.Lock()
        self._forward_backward_futures: dict[int, TrainingFuture[ForwardBackwardCompletion]] = {}
        self._optim_future: TrainingFuture[OptimizerStepResult] | None = None

    def __enter__(self) -> TrainingSession:
        if self._entered or self._closed:
            raise RuntimeError("training session cannot be entered more than once")
        self._channel.call(TrainingVerb.ENTER_TRAINING_SESSION)
        self._entered = True
        return self

    async def __aenter__(self) -> TrainingSession:
        async with self._async_lifecycle_lock:
            entered, cancellation = await self._run_lifecycle_call(
                self.__enter__,
                cancel_if_pending=True,
            )
            if self._checks and entered is not self:
                raise RuntimeError("training-session entry returned an unexpected object")
            if cancellation is None:
                return self
            try:
                await self._run_lifecycle_call(self._abort)
            except BaseException as cleanup_error:
                cleanup_error.add_note(
                    "Training-session entry was cancelled before cleanup failed."
                )
                raise
            raise cancellation

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del error_type, traceback
        if self._closed:
            return False
        if error is None:
            self.close()
            return False
        try:
            self._abort()
        except SwitchRejectedError as cleanup_error:
            error.add_note(f"Training session remains open: {cleanup_error}")
            for note in getattr(cleanup_error, "__notes__", ()):
                error.add_note(note)
        return False

    async def __aexit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        async with self._async_lifecycle_lock:
            suppressed, cancellation = await self._run_lifecycle_call(
                partial(self.__exit__, error_type, error, traceback)
            )
            if cancellation is not None and error is None:
                raise cancellation
            return suppressed

    def forward_backward(
        self,
        # Through the lazy package: the batch's torch loads only when the hint resolves.
        batch: training.TrainingBatch,
    ) -> TrainingFuture[ForwardBackwardCompletion]:
        with self._state_lock:
            self._require_open()
            submission_id = self._channel.call(TrainingVerb.SUBMIT_FORWARD_BACKWARD, batch)
            future = TrainingFuture[ForwardBackwardCompletion](
                submission_id,
                self._resolve_forward_backward,
                checks=self._checks,
            )
            self._forward_backward_futures[submission_id] = future
            return future

    def optim_step(self) -> TrainingFuture[OptimizerStepResult]:
        with self._state_lock:
            self._require_open()
            submission_id = self._channel.call(TrainingVerb.SUBMIT_OPTIMIZER_STEP)
            future = TrainingFuture[OptimizerStepResult](
                submission_id, self._resolve_optim_step, checks=self._checks
            )
            self._optim_future = future
            return future

    def discard_accumulation(self) -> None:
        with self._state_lock:
            self._require_open()
            self._channel.call(TrainingVerb.DISCARD_ACCUMULATION)
            self._fail_forward_backward(RuntimeError("training accumulation was discarded"))

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        with self._state_lock:
            self._require_open()
            resolved_path = _resolved_path(path, "checkpoint")
            self._channel.call(TrainingVerb.SAVE_TRAINING_CHECKPOINT, resolved_path)

    def save_adapter(self, path: str | os.PathLike[str]) -> None:
        """Write the trained LoRA adapter as a PEFT adapter in the folder ``path``:
        ``adapter_config.json`` and ``adapter_model.safetensors``, which PEFT 0.19 and
        Transformers 5 load (routed experts through ``target_parameters``)."""

        with self._state_lock:
            self._require_open()
            self._channel.call(TrainingVerb.SAVE_ADAPTER, _resolved_path(path, "adapter"))

    def publish(self) -> int:
        """Publish the trained adapter to inference, which until now serves the last
        published one, and return the new model revision. Optimizer steps change only the
        trained adapter; no accumulation window may be open."""

        with self._state_lock:
            self._require_open()
            return self._channel.call(TrainingVerb.PUBLISH_ADAPTER)

    def pause_for_serving(self) -> None:
        """Establish serving residency at the configured training boundary."""

        with self._state_lock:
            self._require_open()
            self._channel.call(TrainingVerb.PAUSE_FOR_SERVING)

    async def pause_for_serving_async(self) -> None:
        """Establish serving residency without blocking the caller's event loop."""

        async with self._async_lifecycle_lock:
            _, cancellation = await self._run_lifecycle_call(
                self.pause_for_serving,
                cancel_if_pending=True,
            )
            if cancellation is None:
                return
            try:
                await self._run_lifecycle_call(self.resume_training)
            except BaseException as cleanup_error:
                cleanup_error.add_note(
                    "Serving pause was cancelled after starting and its compensating "
                    "training resume failed."
                )
                raise
            raise cancellation

    def resume_training(self) -> None:
        """Release explicit serving demand and restore training residency."""

        with self._state_lock:
            self._require_open()
            self._channel.call(TrainingVerb.RESUME_TRAINING)

    async def resume_training_async(self) -> None:
        """Restore training residency without blocking the caller's event loop."""

        async with self._async_lifecycle_lock:
            _, cancellation = await self._run_lifecycle_call(
                self.resume_training,
                cancel_if_pending=True,
            )
            if cancellation is not None:
                raise cancellation

    def close(self) -> None:
        if self._closed:
            return
        pending_error = self._settle_optim_future()
        with self._state_lock:
            if self._forward_backward_futures:
                raise RuntimeError(
                    "training accumulation must be committed or discarded before session close"
                )
            try:
                self._require_open()
                self._channel.call(TrainingVerb.CLOSE_TRAINING_SESSION)
            except SwitchRejectedError as refused:
                # The session stays open: the retryable outcome comes first, as for abort.
                _carry(refused, pending_error)
                raise
            self._closed = True
        if pending_error is not None:
            raise pending_error

    def _resolve_forward_backward(
        self,
        future: TrainingFuture[ForwardBackwardCompletion],
    ) -> ForwardBackwardCompletion:
        with self._state_lock:
            if future._result is not None:
                return future._result
            if future._error is not None:
                raise future._error.replay()
            submission_id = future._submission_id
            if self._checks and self._forward_backward_futures.get(submission_id) is not future:
                raise RuntimeError("forward/backward future is stale")
            try:
                self._require_open()
                observations = self._channel.call(
                    TrainingVerb.OBSERVE_FORWARD_BACKWARD, submission_id
                )
            except EngineProcessError as error:
                self._fail_forward_backward(error)
                raise
            except Exception as error:  # refused: only this submission failed
                self._forward_backward_futures.pop(submission_id)
                future._set_error(error)
                raise
            except BaseException as error:
                self._fail_forward_backward(error)
                raise
            self._accept_forward_backward_observations(observations)
            completion = future._result
            if self._checks and completion is None:
                raise RuntimeError("forward/backward observation did not resolve its future")
            return completion

    def _resolve_optim_step(
        self,
        future: TrainingFuture[OptimizerStepResult],
    ) -> OptimizerStepResult:
        with self._state_lock:
            if future._result is not None:
                return future._result
            if future._error is not None:
                raise future._error.replay()
            submission_id = future._submission_id
            if self._checks and self._optim_future is not future:
                raise RuntimeError("optimizer future is stale")
            try:
                self._require_open()
                result, observations = self._channel.call(
                    TrainingVerb.OBSERVE_OPTIMIZER_STEP, submission_id
                )
            except BaseException as error:
                future._set_error(error)
                self._optim_future = None
                self._fail_forward_backward(error)
                raise
            self._accept_forward_backward_observations(observations)
            future._set_result(result)
            self._optim_future = None
            return result

    def _accept_forward_backward_observations(
        self,
        observations: tuple[ForwardBackwardObservation, ...],
    ) -> None:
        for observation in observations:
            future = self._forward_backward_futures.pop(observation.submission_id, None)
            if future is not None:
                future._set_result(observation.completion)
            elif self._checks:
                raise RuntimeError("forward/backward observation has no pending future")

    def _fail_forward_backward(self, error: BaseException) -> None:
        for future in self._forward_backward_futures.values():
            future._set_error(error)
        self._forward_backward_futures.clear()

    def _settle_optim_future(self) -> BaseException | None:
        """Wait for the pending optimizer step; return its failure instead of raising it."""

        optim_future = self._optim_future
        if optim_future is None:
            return None
        try:
            optim_future.result()
        except BaseException as error:  # noqa: BLE001 - close and abort still settle the session
            return error
        return None

    def _abort(self) -> None:
        self._require_open()
        pending_error = self._settle_optim_future()
        with self._state_lock:
            try:
                self._channel.call(TrainingVerb.ABORT_TRAINING_SESSION)
            except SwitchRejectedError as refused:
                # The session stays open: the retryable outcome comes first, as for close.
                _carry(refused, pending_error)
                raise
            finally:
                # Abort discards the work even when the switch is refused.
                self._fail_forward_backward(
                    pending_error or RuntimeError("training session was aborted")
                )
            self._closed = True
        if pending_error is not None:
            raise pending_error

    def _require_open(self) -> None:
        if not self._entered or self._closed:
            raise RuntimeError("training session is not open")

    @staticmethod
    async def _run_lifecycle_call(
        operation: Callable[[], T],
        *,
        cancel_if_pending: bool = False,
    ) -> tuple[T, asyncio.CancelledError | None]:
        # The thread and a pending cancellation race for this claim; the first one wins.
        claim = Lock()

        def call() -> T:
            if not claim.acquire(blocking=False):
                raise _LifecycleCallCancelled
            return operation()

        task = asyncio.create_task(asyncio.to_thread(call))
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if cancel_if_pending and claim.acquire(blocking=False):
                    task.cancel()
                    with suppress(asyncio.CancelledError, _LifecycleCallCancelled):
                        await task
                    raise error
                if cancellation is None:
                    cancellation = error
            except BaseException:  # noqa: BLE001 - task.result() preserves the failure
                break
        try:
            result = task.result()
        except BaseException as error:
            if cancellation is not None and error is not cancellation:
                error.add_note(
                    "Task cancellation arrived while this asynchronous "
                    "training-session operation was completing."
                )
            raise
        return result, cancellation


__all__ = ["TrainingFuture", "TrainingSession"]
