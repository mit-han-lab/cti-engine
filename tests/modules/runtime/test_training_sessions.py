"""TrainingScheduler's sessions through its public surface: the parent's commands in,
their answers out, and ``ModeSwitch`` moving the device between it and a stand-in serving
side, over a stand-in worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from cti.config import TrainingPauseBoundary
from cti.memory import Mode, ResidencyRestoreError
from cti.runtime._core import ModeSwitch
from cti.training import SFT, PaddedTokens, TrainingBatch, TrainingInputError, TrainingVerb
from cti.training._objective import Metrics, Reduction
from cti.training.runner import TrainingCommitResult
from cti.training.scheduler import TrainingScheduler
from tests.support.training import RecordedReplies, Reply

_BATCH = TrainingBatch(
    PaddedTokens(
        torch.tensor([[1, 2, 3]], dtype=torch.int64), torch.tensor([3], dtype=torch.int64), 0
    ),
    torch.ones((1, 2), dtype=torch.bool),
    SFT(),
)


class _Command(NamedTuple):
    id: int
    verb: TrainingVerb
    args: tuple[object, ...] = ()


# Each command's id names what its answer shows.
_Verb = TrainingVerb
_ENTER = _Command(0, _Verb.ENTER_TRAINING_SESSION)
_SUBMIT = _Command(1, _Verb.SUBMIT_FORWARD_BACKWARD, (_BATCH,))
_OBSERVE = _Command(2, _Verb.OBSERVE_FORWARD_BACKWARD, (0,))
_OPTIM = _Command(3, _Verb.SUBMIT_OPTIMIZER_STEP)
_OBSERVE_OPTIM = _Command(4, _Verb.OBSERVE_OPTIMIZER_STEP, (0,))
_DISCARD = _Command(5, _Verb.DISCARD_ACCUMULATION)
_CLOSE = _Command(6, _Verb.CLOSE_TRAINING_SESSION)
_ABORT = _Command(7, _Verb.ABORT_TRAINING_SESSION)
_PAUSE = _Command(8, _Verb.PAUSE_FOR_SERVING)
_RESUME = _Command(9, _Verb.RESUME_TRAINING)
_ANSWERS = (
    lambda _: ("entered",),
    lambda submission: ("submitted", submission),
    lambda observations: ("observed", *(o.submission_id for o in observations)),
    lambda submission: ("optim-submitted", submission),
    lambda outcome: ("optim", outcome[0].step),
    lambda _: ("discarded",),
    lambda _: ("closed",),
    lambda _: ("closed",),
    lambda _: ("serving-ready",),
    lambda _: ("resumed",),
)


class _Runner:
    """Two microbatches a batch, each logged."""

    MICROBATCHES = 2

    def __init__(self, log: list[str], fail: set[str]) -> None:
        self._log = log
        self._fail = fail
        self._gradients = False
        self._accepted: SimpleNamespace | None = None

    def forward_backward(self, batch: TrainingBatch) -> None:
        if "forward_backward" in self._fail:
            raise TrainingInputError("injected")
        self._accepted = SimpleNamespace(normalization_count=batch.num_targets, run=0)

    def microbatch(self) -> object:
        work = self._accepted
        work.run += 1
        self._log.append(f"microbatch {work.run}")
        self._gradients = True
        if work.run < self.MICROBATCHES:
            return None
        self._accepted = None
        count = work.normalization_count
        return Metrics((("loss_sum", Reduction.SUM, torch.tensor(1.0)),), count, count)

    def commit(self, count: int, results: tuple[object, ...]) -> TrainingCommitResult:
        self._gradients = False
        return TrainingCommitResult(Metrics.complete(results), 0.5)

    def discard(self) -> None:
        self._log.append("discard")
        self._gradients = False

    @property
    def holds_gradients(self) -> bool:
        return self._gradients

    def suspend(self) -> None:
        pass

    def resume(self) -> None:
        pass


@dataclass
class _Serving:
    """A serving side that wants the device while ``wants``, until it next holds it or a
    switch is refused."""

    wants: bool = False

    def wants_to_run(self) -> bool:
        return self.wants

    def step(self) -> None:
        pass

    def drained(self) -> bool:
        return True

    def release(self) -> None:
        return None

    def switched(self, holds: bool, refusal: ResidencyRestoreError | None) -> None:
        if holds or refusal is not None:
            self.wants = False


@dataclass
class _Worker:
    """Moves the memory (logged), refused once per mode in ``fail``."""

    log: list[str]
    fail: set[str]

    def quiesce(self) -> None:
        pass

    def release(self, mode: Mode) -> None:
        other = Mode.TRAINING if mode is Mode.SERVING else Mode.SERVING
        self.log.append(f"to {other.name.lower()}")

    def acquire(self, mode: Mode) -> None:
        if mode.name in self.fail:
            self.fail.discard(mode.name)
            self.log.append("refused")
            raise ResidencyRestoreError(mode.name.lower(), 2, 1)


def _seen(reply: Reply) -> tuple[object, ...]:
    """What the parent's call sees of one reply."""

    if reply.renews:
        return ("progress",)
    try:
        value = reply.outcome()
    except Exception as error:  # noqa: BLE001 - a refusal raises its error
        return ("refused", type(error).__name__)
    return _ANSWERS[reply.id](value)


def _drive(
    script: tuple[object, ...],
    *,
    fail: frozenset[str] = frozenset(),
    boundary: TrainingPauseBoundary = TrainingPauseBoundary.LOGICAL_BATCH,
    serve_after: int | None = None,
) -> tuple[tuple[tuple[object, ...], ...], tuple[str, ...]]:
    """Each command once the last was answered; requests wait from command
    ``serve_after`` on. What the parent saw, and what the device did."""

    log: list[str] = []
    replies: list[object] = []
    failing = set(fail)
    training = TrainingScheduler(
        _Runner(log, failing), boundary, replies=RecordedReplies(replies.append)
    )
    serving = _Serving()
    switch = ModeSwitch(serving, training, _Worker(log, failing))
    for index, command in enumerate(script):
        serving.wants |= index == serve_after
        training.handle(command.id, command.verb, command.args, holds=switch.mode is Mode.TRAINING)
        _run_until_idle(switch)
    return tuple(_seen(reply) for reply in replies), tuple(log)


def _run_until_idle(switch: ModeSwitch) -> None:
    """EngineCore's loop with no input arriving: once the switch is not busy, it blocks."""

    for _ in range(50):
        switch.advance()
        if not switch.switching:
            switch.runs().step()
        if not switch.busy:
            return


_BATCH_RUN = ("microbatch 1", "microbatch 2")
_IN, _OUT = ("to training",), ("to serving",)


@pytest.mark.parametrize(
    ("script", "fail", "seen", "device"),
    [
        pytest.param(
            (_ENTER, _SUBMIT, _OBSERVE, _OPTIM, _OBSERVE_OPTIM, _CLOSE),
            frozenset(),
            (
                ("entered",),
                ("submitted", 0),
                ("observed", 0),
                ("optim-submitted", 0),
                ("optim", 1),
                ("closed",),
            ),
            (*_IN, *_BATCH_RUN, *_OUT),
            id="train-step",
        ),
        pytest.param(
            (_ENTER, _SUBMIT, _SUBMIT, _OPTIM, _OBSERVE, _OBSERVE_OPTIM, _CLOSE),
            frozenset(),
            (
                ("entered",),
                ("submitted", 0),
                ("submitted", 1),
                ("optim-submitted", 0),
                ("observed", 0),
                ("optim", 1),
                ("closed",),
            ),
            (*_IN, *_BATCH_RUN, *_BATCH_RUN, *_OUT),
            id="the-step-observes-the-rest",
        ),
        pytest.param(
            (_ENTER, _OPTIM, _OBSERVE_OPTIM, _CLOSE),
            frozenset(),
            (
                ("entered",),
                ("optim-submitted", 0),
                ("refused", "InvalidGradientError"),
                ("closed",),
            ),
            (*_IN, *_OUT),
            id="optim-without-accumulation",
        ),
        pytest.param(
            (_ENTER, _SUBMIT, _CLOSE, _DISCARD, _CLOSE),
            frozenset(),
            (
                ("entered",),
                ("submitted", 0),
                ("refused", "CommandStateError"),
                ("discarded",),
                ("closed",),
            ),
            (*_IN, *_BATCH_RUN, "discard", *_OUT),
            id="close-with-accumulation",
        ),
        pytest.param(
            (_ENTER, _SUBMIT, _ABORT),
            frozenset(),
            (("entered",), ("submitted", 0), ("closed",)),
            (*_IN, *_BATCH_RUN, "discard", *_OUT),
            id="abort-runs-the-batch-then-discards",
        ),
        pytest.param(
            (_ENTER, _SUBMIT, _CLOSE),
            frozenset({"forward_backward"}),
            (("entered",), ("refused", "TrainingInputError"), ("closed",)),
            (*_IN, *_OUT),
            id="refused-submit-keeps-the-session",
        ),
        pytest.param(
            (_ENTER,),
            frozenset({"TRAINING"}),
            (("refused", "SwitchRejectedError"),),
            (*_IN, "refused"),
            id="entry-refused",
        ),
        pytest.param(
            (_ENTER, _CLOSE, _CLOSE),
            frozenset({"SERVING"}),
            (("entered",), ("refused", "SwitchRejectedError"), ("closed",)),
            (*_IN, *_OUT, "refused", *_OUT),
            id="close-refused-keeps-the-session",
        ),
        pytest.param(
            (_ENTER, _SUBMIT, _PAUSE, _RESUME, _OBSERVE, _DISCARD, _CLOSE),
            frozenset(),
            (
                ("entered",),
                ("submitted", 0),
                ("serving-ready",),
                ("resumed",),
                ("observed", 0),
                ("discarded",),
                ("closed",),
            ),
            (*_IN, *_BATCH_RUN, *_OUT, *_IN, "discard", *_OUT),
            id="explicit-pause-and-resume",
        ),
    ],
)
def test_commands_answer_as_the_parent_expects(
    script: tuple[object, ...],
    fail: frozenset[str],
    seen: tuple[tuple[object, ...], ...],
    device: tuple[str, ...],
) -> None:
    assert _drive(script, fail=fail) == (seen, device)


@pytest.mark.parametrize(
    ("boundary", "device"),
    [
        (TrainingPauseBoundary.MICROBATCH, (*_OUT, *_IN, *_BATCH_RUN)),
        (TrainingPauseBoundary.LOGICAL_BATCH, (*_BATCH_RUN, *_OUT, *_IN)),
    ],
    ids=lambda value: getattr(value, "value", None),
)
def test_held_requests_take_the_device_at_the_pause_boundary(
    boundary: TrainingPauseBoundary, device: tuple[str, ...]
) -> None:
    seen, log = _drive(
        (_ENTER, _SUBMIT, _OBSERVE, _DISCARD, _CLOSE), boundary=boundary, serve_after=1
    )

    assert seen == (
        ("entered",),
        ("submitted", 0),
        ("observed", 0),
        ("discarded",),
        ("closed",),
    )
    assert log == (*_IN, *device, "discard", *_OUT)


def test_a_command_queued_behind_a_batch_moves_the_device_without_new_input() -> None:
    log: list[str] = []
    replies: list[object] = []
    training = TrainingScheduler(
        _Runner(log, set()),
        TrainingPauseBoundary.LOGICAL_BATCH,
        replies=RecordedReplies(replies.append),
    )
    switch = ModeSwitch(_Serving(), training, _Worker(log, set()))
    training.handle(_ENTER.id, _ENTER.verb, _ENTER.args, holds=False)
    _run_until_idle(switch)
    training.handle(_SUBMIT.id, _SUBMIT.verb, _SUBMIT.args, holds=True)
    training.handle(_ABORT.id, _ABORT.verb, _ABORT.args, holds=True)  # queued: the batch runs first

    _run_until_idle(switch)

    assert tuple(_seen(reply) for reply in replies) == (
        ("entered",),
        ("submitted", 0),
        ("closed",),
    )
    assert tuple(log) == (*_IN, *_BATCH_RUN, "discard", *_OUT)


@dataclass
class _Requests:
    """A serving side whose requests each take ``steps`` serving steps. As EngineCore hands
    requests over, one that arrives while the switch admits none waits for serving's next
    turn on the device."""

    log: list[str]
    steps: int = 3
    running: dict[str, int] = field(default_factory=dict)
    waiting: list[str] = field(default_factory=list)

    def handle(self, names: tuple[str, ...], *, admitting: bool) -> None:
        self.waiting.extend(names)
        if admitting:
            self._admit()

    def wants_to_run(self) -> bool:
        return bool(self.running or self.waiting)

    def step(self) -> None:
        for name in list(self.running):
            self.running[name] -= 1
            if not self.running[name]:
                del self.running[name]
                self.log.append(f"finish {name}")

    def drained(self) -> bool:
        return not self.running

    def release(self) -> None:
        return None

    def switched(self, holds: bool, refusal: ResidencyRestoreError | None) -> None:
        if holds:
            self._admit()

    def _admit(self) -> None:
        for name in self.waiting:
            self.running[name] = self.steps
            self.log.append(f"admit {name}")
        self.waiting.clear()


def _serve_while_training(
    arrivals: dict[int, tuple[str, ...]],
    *,
    microbatches: int,
    commands: dict[int, _Command] | None = None,
    turns: int = 30,
) -> tuple[str, ...]:
    """EngineCore's turns over one accepted batch of ``microbatches`` (none for 0),
    requests arriving at the turns ``arrivals`` names and ``commands`` at theirs: each turn
    hands its command and arrivals over, then the switch moves and the holder steps. What
    the device did, each answer logged where it came."""

    log: list[str] = []
    runner = _Runner(log, set())
    runner.MICROBATCHES = microbatches
    training = TrainingScheduler(
        runner,
        TrainingPauseBoundary.MICROBATCH,
        replies=RecordedReplies(
            lambda reply: log.append(f"{'progress' if reply.renews else 'answer'} {reply.id}")
        ),
    )
    serving = _Requests(log)
    switch = ModeSwitch(serving, training, _Worker(log, set()))
    training.handle(_ENTER.id, _ENTER.verb, _ENTER.args, holds=False)
    _run_until_idle(switch)
    if microbatches:
        training.handle(_SUBMIT.id, _SUBMIT.verb, _SUBMIT.args, holds=True)
    for turn in range(turns):
        if commands and turn in commands:
            command = commands[turn]
            training.handle(
                command.id, command.verb, command.args, holds=switch.mode is Mode.TRAINING
            )
        serving.handle(arrivals.get(turn, ()), admitting=switch.admits_requests)
        switch.advance()
        if not switch.switching:
            switch.runs().step()
    return tuple(log)


def test_serving_admits_arrivals_until_no_request_runs_then_training_resumes() -> None:
    log = _serve_while_training({1: ("a",), 3: ("b",)}, microbatches=3)

    # b arrives while a runs: serving takes it at once instead of finishing a, running a
    # microbatch and only then serving b.
    assert log == (
        *_IN,
        "answer 0",
        "answer 1",
        "microbatch 1",
        *_OUT,
        "admit a",
        "admit b",
        "finish a",
        "finish b",
        *_IN,
        "microbatch 2",
        "microbatch 3",
    )


def test_training_waits_out_a_steady_stream_of_requests() -> None:
    stream = {turn: (f"r{turn}",) for turn in range(1, 21)}

    log = _serve_while_training(stream, microbatches=3)

    # Serving admits every arrival and keeps the device until the last request finishes;
    # only then does training run its next microbatch.
    assert log.count("to serving") == 1
    assert sum(entry.startswith("admit") for entry in log) == 20
    assert log.index("finish r20") < log.index("microbatch 2")


@pytest.mark.parametrize(
    ("command", "microbatches"),
    [pytest.param(_CLOSE, 0, id="close"), pytest.param(_ABORT, 3, id="abort-mid-batch")],
)
def test_a_close_or_abort_takes_the_device_from_a_steady_stream(
    command: _Command, microbatches: int
) -> None:
    stream = {turn: (f"r{turn}",) for turn in range(1, 30)}

    log = _serve_while_training(stream, microbatches=microbatches, commands={4: command})

    # The session's end does not wait for serving to run out: serving drains for it, as for
    # a paused session's, the wait renewed, and takes the stream back once it is answered.
    answer = log.index(f"answer {command.id}")
    assert f"progress {command.id}" in log[:answer]
    assert answer < log.index("admit r29")
    assert log[-1].startswith(("admit", "finish"))
    if microbatches:  # the abort runs once the batch has, serving between its microbatches
        assert log.index(f"microbatch {microbatches}") < answer
    else:  # the close takes the device back once, and only for itself
        assert log[:answer].count("to training") == 2
