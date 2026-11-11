"""Training's scheduler: the session, the accepted batch and its microbatches, the
forward/backward FIFO, the accumulation window, the optimizer step count, the model
revision and the parent's replies, over one rank's TrainingRunner. It schedules training only:
when training holds the device is the mode switch's decision."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial, wraps
from typing import TYPE_CHECKING, Protocol

from cti.config import TrainingPauseBoundary

from ._batch import TrainingBatch
from ._errors import (
    InvalidGradientError,
    SwitchRejectedError,
    TrainingInputError,
)
from ._objective import GRPO, SFT, Metrics
from ._protocol import (
    CommandStateError,
    ForwardBackwardObservation,
    OptimizerStepResult,
    Refusal,
    TrainingVerb,
)

if TYPE_CHECKING:
    import os

    from cti.memory import ResidencyRestoreError

    from .runner import TrainingRunner


@dataclass(frozen=True, slots=True)
class _Accepted:
    """The accepted batch's submission; the runner owns its progress."""

    submission_id: int
    objective: type[SFT | GRPO]
    normalization_count: int


@dataclass(frozen=True, slots=True)
class _Executed:
    submission_id: int
    normalization_count: int
    result: Metrics


class TrainingReplies(Protocol):
    """Where the scheduler's answers to the parent's commands go, by command id; the
    runtime implements it."""

    def answer(self, command_id: int, value: object = None) -> None:
        """The command returned ``value``."""
        ...

    def progress(self, command_id: int) -> None:
        """The command still runs: its caller's wait is renewed."""
        ...

    def refuse(self, command_id: int, error: Refusal) -> None:
        """The command raised ``error``; training goes on as before it."""
        ...


def _queued(
    *, settled: bool, resident: bool = False, resumes: bool = False
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """A session's one queued command, run once training holds the device and no batch
    runs; a ``settled`` one only after a submitted optimizer step's result is observed.
    While paused for serving, a ``resident`` one is refused, as the pause would hold it
    forever. A ``resumes`` one takes the device back whenever serving holds it, paused or
    not: serving drains for it instead of keeping the device as long as requests come."""

    def decorate(run: Callable[..., None]) -> Callable[..., None]:
        @wraps(run)
        def queue(self: TrainingScheduler, command_id: int, *args: object) -> None:
            self._require_open(run.__name__, paused=not resident)
            if self._command is not None:
                raise CommandStateError("another training command is already queued")

            def start() -> None:
                if settled and self._optim is not None:
                    raise CommandStateError(_OPTIM_FIRST)
                run(self, command_id, *args)

            self._command = (command_id, start)
            self._command_resumes = resumes

        return queue

    return decorate


class TrainingScheduler:
    """Schedule training on one rank, as one side of the mode switch (``Switchable``).

    The parent's commands come in through ``handle``; their answers go out through
    ``replies``, by the command's id. A command runs once training holds the device and no
    batch runs; ``step`` runs it, or the accepted batch's next microbatch. A command that
    needs the device elsewhere sets ``requested_holds``, and ``switched`` answers it once the
    switch has moved the device, or refuses it. A ``Refusal`` a command raises refuses it
    and keeps the session; any other error raises on to the process's root.
    """

    def __init__(
        self,
        runner: TrainingRunner,
        pause_boundary: TrainingPauseBoundary,
        *,
        replies: TrainingReplies,
    ) -> None:
        """``pause_boundary`` says where an accepted batch may yield the device."""

        self._runner = runner
        self._pause_boundary = pause_boundary
        self._replies = replies
        self.model_revision = 0
        self._session = _Session.CLOSED
        self._awaiting: _SwitchCommand | None = None  # answered once the device moves
        self._renewed = 0.0  # when the awaiting command's wait was last renewed
        self._holds_device = False  # where the switch last said the device is
        self._command_resumes = False  # the queued command takes the device back from serving
        self._last_optim = -1
        # The command that waits for the device or a batch, by its id.
        self._command: tuple[int, Callable[[], None]] | None = None
        # The error an accepted batch's execution owes its observation, by submission id.
        self._failed: tuple[int, Refusal] | None = None
        self._optim: _Optim | None = None  # a submitted step's outcome, until observed
        # Completed submissions not yet observed: those a committed step completed.
        self._completed: deque[ForwardBackwardObservation] = deque()
        self._next_submission_id = 0
        self._active: _Accepted | None = None
        self._pending: deque[_Executed] = deque()
        # The open accumulation window: its objective and its observed normalization count.
        self._window: type[SFT | GRPO] | None = None
        self._observed_count = 0
        self._num_optimizer_steps = 0

    @property
    def has_active(self) -> bool:
        """Whether an accepted batch has microbatches left to run."""

        return self._active is not None

    @property
    def _has_pending_accumulation(self) -> bool:
        return self._active is not None or bool(self._pending) or self._observed_count != 0

    def _accept(self, batch: TrainingBatch, *, model_revision: int) -> int:
        """Validate and plan one batch on CPU; only an accepted batch takes an id."""

        if self._active is not None:
            raise RuntimeError("another forward/backward submission is already accepted")
        objective = type(batch.objective)
        batch.objective.admit(batch, model_revision)
        if self._window is not None and self._window is not objective:
            raise InvalidGradientError("one accumulation window cannot mix SFT and GRPO objectives")
        self._runner.forward_backward(batch)
        self._active = _Accepted(
            self._next_submission_id, objective, batch.objective.normalization_count(batch)
        )
        self._next_submission_id += 1
        return self._active.submission_id

    def _advance(self) -> bool:
        """Run the accepted batch's next microbatch; return whether the batch is done."""

        accepted = self._active
        if accepted is None:
            raise RuntimeError("no forward/backward submission is accepted")
        try:
            result = self._runner.microbatch()
        except BaseException:
            self._active = None
            self._close_window_if_discarded()
            raise
        if result is None:
            return False
        self._pending.append(
            _Executed(accepted.submission_id, accepted.normalization_count, result)
        )
        self._window = accepted.objective
        self._active = None
        return True

    def _observe_window(
        self,
        submission_id: int,
    ) -> tuple[ForwardBackwardObservation, ...]:
        """Observe the executed FIFO prefix through ``submission_id`` exactly once."""

        pending = self._pending
        if not pending or submission_id < pending[0].submission_id:
            if submission_id < self._next_submission_id:
                raise CommandStateError(
                    f"forward/backward submission {submission_id} was already observed"
                )
            raise CommandStateError(f"forward/backward submission {submission_id} is unknown")
        if submission_id > pending[-1].submission_id:
            raise CommandStateError(f"forward/backward submission {submission_id} is unknown")

        prefix = tuple(item for item in pending if item.submission_id <= submission_id)
        try:
            completions = Metrics.complete(tuple(item.result for item in prefix))
        except InvalidGradientError:
            self._discard_window()
            raise
        for item in prefix:
            pending.popleft()
            self._observed_count += item.normalization_count
        return tuple(
            ForwardBackwardObservation(item.submission_id, completion)
            for item, completion in zip(prefix, completions, strict=True)
        )

    def _take_completed(self, submission_id: int) -> tuple[ForwardBackwardObservation, ...]:
        """The completed observations through ``submission_id``, observed once."""

        taken = []
        while not taken or taken[-1].submission_id != submission_id:
            taken.append(self._completed.popleft())
        return tuple(taken)

    def _optim_step(self) -> tuple[OptimizerStepResult, tuple[ForwardBackwardObservation, ...]]:
        """Commit the window behind one validation gate, then observe its unobserved rest."""

        self._require_no_accepted("optim_step")
        pending = tuple(self._pending)
        count = self._observed_count + sum(item.normalization_count for item in pending)
        if count == 0:
            raise InvalidGradientError("optim_step requires a preceding forward_backward call")
        try:
            commit = self._runner.commit(count, tuple(item.result for item in pending))
        except InvalidGradientError:
            self._close_window_if_discarded()
            raise
        self._close_window()
        self._num_optimizer_steps += 1
        return OptimizerStepResult(self._num_optimizer_steps, count, commit.grad_norm), tuple(
            ForwardBackwardObservation(item.submission_id, completion)
            for item, completion in zip(pending, commit.completions, strict=True)
        )

    def _discard_window(self) -> None:
        self._require_no_accepted("discard_accumulation")
        try:
            self._runner.discard()
        except BaseException:
            self._close_window_if_discarded()
            raise
        self._close_window()

    def _require_no_pending_accumulation(self) -> None:
        """Require the window to be committed or discarded, logically and on the device."""

        if self._has_pending_accumulation or self._runner.holds_gradients:
            raise CommandStateError(
                "training accumulation must be committed or discarded before suspension"
            )

    def _save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Publish a settled checkpoint at ``path`` without advancing logical training."""

        self._require_no_pending_accumulation()
        self._runner.save(
            path, num_optimizer_steps=self._num_optimizer_steps, model_revision=self.model_revision
        )

    # ================================ The session ================================

    @property
    def requested_holds(self) -> bool | None:
        """Where a command waits for the switch to move the device: to training (True), to
        serving (False), or nowhere (None)."""

        if self._awaiting is not None:
            return self._awaiting.holds
        return True if self._takes_device_back else None

    @property
    def _takes_device_back(self) -> bool:
        """Whether the queued command waits for the device serving holds."""

        return self._command is not None and self._command_resumes and not self._holds_device

    def restore(self, path: str | os.PathLike[str]) -> None:
        """Resume from the training checkpoint at ``path`` at startup. It copies into tensors
        that stay resident while serving holds the device, so no switch is needed."""

        from ._checkpoint import TrainingCheckpoint

        checkpoint = TrainingCheckpoint.load(path)
        self._require_no_pending_accumulation()
        self._runner.restore(checkpoint)
        self._num_optimizer_steps = checkpoint.num_optimizer_steps
        self.model_revision = checkpoint.model_revision

    # Switchable

    def wants_to_run(self) -> bool:
        return self._session is _Session.OPEN and (self.has_active or self._command is not None)

    def step(self) -> None:
        """Run the queued command once no batch runs, else the next microbatch."""

        self._next()

    def renew_awaiting(self) -> None:
        """The device's holder still drains toward the move a command waits for: renew the
        parent's wait for it, at most once a second."""

        if self._awaiting is not None:
            command_id = self._awaiting.command_id
        elif self._takes_device_back:  # a queued close or abort
            command_id = self._command[0]
        else:
            return
        now = time.monotonic()
        if now - self._renewed >= _RENEW_SECONDS:
            self._renewed = now
            self._replies.progress(command_id)

    def drained(self) -> bool:
        """Whether training may yield the device now: between batches, or between
        microbatches when the pause boundary is the microbatch."""

        return self._active is None or self._pause_boundary is TrainingPauseBoundary.MICROBATCH

    def release(self) -> None:
        """Move the gradients, and the accepted batch's metrics, off the device before the
        worker takes training's memory away."""

        self._runner.suspend()

    def switched(self, holds: bool, refusal: ResidencyRestoreError | None) -> None:
        """Holding the device, bring back what ``release`` moved off. Answer the command
        that waited for the device to move, or refuse it. Training's own memory not coming
        back fails, except on entry."""

        if holds:
            self._runner.resume()
        awaiting = self._awaiting
        # An awaiting command chose the move (``requested_holds``); else a queued one did.
        took_back = awaiting is None and self._takes_device_back
        self._holds_device = holds
        if took_back:  # the queued command runs, or is refused, as the move went
            if refusal is None:
                self._session = _Session.OPEN
            else:
                (command_id, _), self._command = self._command, None
                self._replies.refuse(command_id, SwitchRejectedError(str(refusal)))
        elif refusal is None:
            if awaiting is not None and awaiting.holds is holds:
                self._awaiting = None
                if holds:
                    self._session = _Session.OPEN
                elif awaiting.verb is _PAUSE:
                    self._session = _Session.PAUSED
                else:  # closed or aborted
                    self._session, self._failed = _Session.CLOSED, None
                self._replies.answer(awaiting.command_id)
        elif holds:  # serving's memory did not fit: training keeps the device
            if awaiting is not None and not awaiting.holds:
                self._refuse_switch(awaiting, _NO_FIT)
        elif awaiting is not None and awaiting.verb is _ENTER:
            self._refuse_switch(awaiting, str(refusal))
        else:
            raise RuntimeError("training memory did not come back") from refusal

    # Commands

    def handle(
        self, command_id: int, verb: TrainingVerb, args: tuple[object, ...], *, holds: bool
    ) -> None:
        """Run the parent's command ``command_id``: ``verb`` with ``args``. A queued command
        runs at once when training ``holds`` the device and no batch runs."""

        self._attempt(command_id, partial(self._command_of(verb), command_id, *args))
        if holds and not self.has_active:
            self._next()

    def _command_of(self, verb: TrainingVerb) -> Callable[..., None]:
        match verb:
            case TrainingVerb.ENTER_TRAINING_SESSION:
                return self.enter_training_session
            case TrainingVerb.SUBMIT_FORWARD_BACKWARD:
                return self.submit_forward_backward
            case TrainingVerb.OBSERVE_FORWARD_BACKWARD:
                return self.observe_forward_backward
            case TrainingVerb.SUBMIT_OPTIMIZER_STEP:
                return self.submit_optimizer_step
            case TrainingVerb.OBSERVE_OPTIMIZER_STEP:
                return self.observe_optimizer_step
            case TrainingVerb.DISCARD_ACCUMULATION:
                return self.discard_accumulation
            case TrainingVerb.SAVE_TRAINING_CHECKPOINT:
                return self.save_training_checkpoint
            case TrainingVerb.SAVE_ADAPTER:
                return self.save_adapter
            case TrainingVerb.PUBLISH_ADAPTER:
                return self.publish_adapter
            case TrainingVerb.CLOSE_TRAINING_SESSION:
                return self.close_training_session
            case TrainingVerb.ABORT_TRAINING_SESSION:
                return self.abort_training_session
            case TrainingVerb.PAUSE_FOR_SERVING:
                return self.pause_for_serving
            case TrainingVerb.RESUME_TRAINING:
                return self.resume_training

    def _attempt(self, command_id: int, run: Callable[[], None]) -> None:
        """Run a command; a ``Refusal`` it raises refuses it."""

        try:
            run()
        except Refusal as error:
            self._replies.refuse(command_id, error)

    def enter_training_session(self, command_id: int) -> None:
        if self._session is not _Session.CLOSED:
            raise CommandStateError("a training session is already active")
        self._await(command_id, _ENTER, holds=True)

    def pause_for_serving(self, command_id: int) -> None:
        self._require_open("pause_for_serving")
        self._await(command_id, _PAUSE, holds=False)

    def resume_training(self, command_id: int) -> None:
        if self._session is not _Session.PAUSED:
            raise CommandStateError("training is not paused for serving")
        self._await(command_id, _RESUME, holds=True)

    def _await(self, command_id: int, verb: TrainingVerb, *, holds: bool) -> None:
        """Answer the command once the switch has moved the device where ``holds`` says."""

        if self._awaiting is not None:
            raise CommandStateError("another training command waits for the device")
        self._awaiting = _SwitchCommand(command_id, verb, holds=holds)

    def _require_open(self, operation: str, *, paused: bool = False) -> None:
        if self._session is _Session.CLOSED:
            raise CommandStateError(f"{operation} needs an open training session")
        if self._session is _Session.PAUSED and not paused:
            raise CommandStateError(
                f"{operation} is unavailable while training is paused for serving"
            )

    def _next(self) -> None:
        """The queued command once no batch runs, else the next microbatch."""

        if self.has_active:
            self._microbatch()
        elif self._command is not None:
            (command_id, run), self._command = self._command, None
            self._attempt(command_id, run)

    def _microbatch(self) -> None:
        submission_id = self._active.submission_id
        try:
            self._advance()
        except Refusal as error:  # owed to the batch's observation
            self._failed = (submission_id, error)

    @_queued(settled=True, resident=True)
    def submit_forward_backward(self, command_id: int, batch: TrainingBatch) -> None:
        if self._failed is not None:
            self._replies.refuse(command_id, self._failed[1])
            return
        submission_id = self._accept(batch, model_revision=self.model_revision)
        self._replies.answer(command_id, submission_id)

    @_queued(settled=False)
    def observe_forward_backward(self, command_id: int, submission_id: int) -> None:
        optim = self._optim
        if self._failed is not None and self._failed[0] == submission_id:
            (_, error), self._failed = self._failed, None
            self._replies.refuse(command_id, error)
        elif optim is not None and optim.rejection is not None:
            self._replies.refuse(command_id, optim.rejection)
        elif any(item.submission_id == submission_id for item in self._completed):
            self._replies.answer(command_id, self._take_completed(submission_id))
        else:
            self._replies.answer(command_id, self._observe_window(submission_id))

    @_queued(settled=True, resident=True)
    def submit_optimizer_step(self, command_id: int) -> None:
        if self._failed is not None:
            self._replies.refuse(command_id, self._failed[1])
            return
        self._last_optim += 1
        submission_id = self._last_optim
        self._replies.answer(command_id, submission_id)
        try:
            result, observations = self._optim_step()
        except Refusal as error:  # owed to the step's observation
            self._optim = _Optim(submission_id, error)
            return
        self._optim = _Optim(submission_id, None, result)
        self._completed.extend(observations)

    @_queued(settled=False)
    def observe_optimizer_step(self, command_id: int, submission_id: int) -> None:
        optim, self._optim = self._optim, None
        if optim is None:
            raise CommandStateError(f"optimizer step {submission_id} is unknown")
        if optim.rejection is not None:
            self._replies.refuse(command_id, optim.rejection)
        else:
            observations, self._completed = tuple(self._completed), deque()
            self._replies.answer(command_id, (optim.result, observations))

    @_queued(settled=True, resident=True)
    def discard_accumulation(self, command_id: int) -> None:
        self._discard_window()
        self._failed = None
        self._replies.answer(command_id)

    @_queued(settled=True, resident=True)
    def save_training_checkpoint(self, command_id: int, path: str) -> None:
        try:
            self._save_checkpoint(path)
        except OSError as error:  # the caller's path: training is as it was
            raise TrainingInputError(f"cannot save a training checkpoint: {error}") from error
        self._replies.answer(command_id)

    @_queued(settled=True, resident=True)
    def save_adapter(self, command_id: int, path: str) -> None:
        self._require_no_pending_accumulation()
        try:
            self._runner.save_adapter(path)
        except (OSError, ValueError) as error:  # the caller's path or adapter: as it was
            raise TrainingInputError(f"cannot save the adapter: {error}") from error
        self._replies.answer(command_id)

    @_queued(settled=True, resident=True)
    def publish_adapter(self, command_id: int) -> None:
        """Publish the trained adapter to inference: a new model revision, the only one a
        step reaches serving and rollouts through."""

        self._require_no_pending_accumulation()
        self._runner.publish()
        self.model_revision += 1
        self._replies.answer(command_id, self.model_revision)

    @_queued(settled=True, resumes=True)
    def close_training_session(self, command_id: int) -> None:
        self._require_no_pending_accumulation()
        self._await(command_id, _CLOSE, holds=False)

    @_queued(settled=True, resumes=True)
    def abort_training_session(self, command_id: int) -> None:
        self._failed = None
        if self._has_pending_accumulation:
            self._discard_window()
        self._await(command_id, _ABORT, holds=False)

    def _refuse_switch(self, awaiting: _SwitchCommand, reason: str) -> None:
        """The device could not move: the command that waited for it is refused."""

        self._awaiting = None
        self._replies.refuse(awaiting.command_id, SwitchRejectedError(reason))

    def _close_window(self) -> None:
        self._pending.clear()
        self._window = None
        self._observed_count = 0

    def _close_window_if_discarded(self) -> None:
        """After a command error, close the window if the device holds no gradients."""

        if not self._runner.holds_gradients:
            self._close_window()

    def _require_no_accepted(self, operation: str) -> None:
        if self._active is not None:
            raise RuntimeError(
                f"{operation} cannot run while forward/backward execution is accepted"
            )


_RENEW_SECONDS = 1.0  # well inside the parent's acknowledgement bound
_OPTIM_FIRST = "the optimizer result must be observed before the next training boundary"
_NO_FIT = "inference memory does not fit while training"


@dataclass(frozen=True, slots=True)
class _Optim:
    submission_id: int
    rejection: Refusal | None = None
    result: OptimizerStepResult | None = None


class _Session(Enum):
    CLOSED = auto()
    OPEN = auto()  # whichever side holds the device
    PAUSED = auto()  # for serving, by the parent


_ENTER = TrainingVerb.ENTER_TRAINING_SESSION
_PAUSE = TrainingVerb.PAUSE_FOR_SERVING
_RESUME = TrainingVerb.RESUME_TRAINING
_CLOSE = TrainingVerb.CLOSE_TRAINING_SESSION
_ABORT = TrainingVerb.ABORT_TRAINING_SESSION


@dataclass(frozen=True, slots=True)
class _SwitchCommand:
    """A command answered once the switch gives training the device (``holds``) or takes
    it away."""

    command_id: int
    verb: TrainingVerb
    holds: bool


__all__ = ["TrainingReplies", "TrainingScheduler"]
