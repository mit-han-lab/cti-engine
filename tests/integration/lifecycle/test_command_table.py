"""The engine's command table, black box: each row drives the public API of a real engine
(its processes, protocol, scheduler, training accounting and switch order) whose device is
the CPU stand-in, and pins what the caller sees: the ACK, the completion or rejection, how a
failure is disposed of, the device's switch order and the messages that crossed processes.
Faults are injected into the engine's processes from here; the library has no hooks.
"""

from __future__ import annotations

import multiprocessing.util
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Thread

import pytest
import torch

from cti.config import (
    AdamWConfig,
    BatchLayout,
    EngineOptions,
    ModelOptions,
    ParallelOptions,
    SchedulingOptions,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
    TrainingPauseBoundary,
)
from cti.llm import LLM
from cti.protocol import RejectionKind, RequestRejectedError, SamplingParams
from cti.runtime import EngineClient, EngineProcessError, FailurePhase
from cti.training import SFT, PaddedTokens, TrainingBatch
from tests.support.cpu_engine import CpuEngine
from tests.support.fault_injection import Exit, FaultInjector, Raise
from tests.support.ipc_probe import IpcProbe
from tests.support.memory_saver import use_memory_saver_provider


def _batch() -> TrainingBatch:
    tokens = PaddedTokens(
        torch.tensor([[1, 2, 3]], dtype=torch.int64), torch.tensor([3], dtype=torch.int64), 0
    )
    return TrainingBatch(tokens, torch.ones((1, 2), dtype=torch.bool), SFT())


def _outcome(call: Callable[[], object]) -> object:
    """What a call returned, or the name of what it raised."""

    try:
        return call()
    except Exception as error:  # noqa: BLE001 - the row pins which error
        return type(error).__name__


@dataclass
class _Engine:
    llm: LLM
    log: Path
    tmp_path: Path

    def device(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def running(self) -> bool:
        """Whether the engine's first forward/backward has yet to finish a microbatch."""

        return not any(" microbatch " in call for call in self.device())

    def generate(self, prompt: list[int], tokens: int) -> object:
        params = SamplingParams(temperature=0, max_new_tokens=tokens)
        return _outcome(lambda: tuple(self.llm.generate([prompt], params)[0]["token_ids"]))


# Rows: each drives the engine and returns what the caller saw.


def _sample(engine: _Engine) -> object:
    return engine.generate([1, 2, 3], 3)


def _seeded_sample(engine: _Engine) -> object:
    """Rank 0 seeds each request before relaying it: every rank runs the same seeds."""

    tokens = engine.generate([1, 2, 3], 3)
    seeds = [
        [call.split(" ", 2)[2] for call in engine.device() if call.startswith(f"rank{rank} seeds")]
        for rank in (0, 1)
    ]
    return tokens, bool(seeds[0]) and seeds[0] == seeds[1] and "None" not in str(seeds[0])


def _sample_together(count: int) -> Callable[[_Engine], object]:
    """One call's prompts reach the engine as one message and run in one forward: a
    repeated call is batched alike."""

    def drive(engine: _Engine) -> object:
        outputs = engine.llm.generate(
            [[index + 1] * 3 for index in range(count)],
            SamplingParams(temperature=0, max_new_tokens=2),
        )
        widths = [
            len(call.split(" ", 2)[2].split(","))
            for call in engine.device()
            if call.startswith("rank0 seeds")
        ]
        return tuple(tuple(output["token_ids"]) for output in outputs), widths[0]

    return drive


def _chat_together(engine: _Engine) -> object:
    """A call's chats, preprocessed concurrently, still reach the engine as one message and
    sample as the prompts their encoder builds."""

    chats = [
        [{"role": "user", "content": f"{index + 1} {index + 1} {index + 1}"}] for index in range(4)
    ]
    result = engine.llm.chat(chats, SamplingParams(temperature=0, max_new_tokens=2))
    return tuple((sample.prompt_token_ids, sample.sampled_token_ids) for sample in result.samples)


def _sample_too_long(engine: _Engine) -> object:
    return engine.generate(list(range(3, 130)), 8)


def _sample_past_the_context(engine: _Engine) -> object:
    """The engine's admission rejects a completion past the context; the error names the
    request and whom the rejection blames."""

    params = SamplingParams(temperature=0, max_new_tokens=200)
    try:
        engine.llm.generate([[1, 2, 3]], params)
    except RequestRejectedError as error:
        return error.request_id, error.kind
    return None


def _bad_sampling(engine: _Engine) -> object:
    """A policy the sampling law refuses is rejected, and the engine samples on."""

    params = SamplingParams(max_new_tokens=3, temperature=1.0, top_p=1.5)
    refused = _outcome(lambda: engine.llm.generate([[1, 2, 3]], params))
    return refused, engine.generate([1, 2, 3], 3)


def _train_step(engine: _Engine) -> object:
    with engine.llm.training_session() as session:
        trained = session.forward_backward(_batch()).result().normalization_count
        step = session.optim_step().result()
    return trained, step.step, step.normalization_count


def _two_forward_backwards(engine: _Engine) -> object:
    with engine.llm.training_session() as session:
        first, second = session.forward_backward(_batch()), session.forward_backward(_batch())
        counts = (first.result().normalization_count, second.result().normalization_count)
        return counts, session.optim_step().result().normalization_count


def _optim_without_accumulation(engine: _Engine) -> object:
    with engine.llm.training_session() as session:
        refused = _outcome(lambda: session.optim_step().result())
        session.forward_backward(_batch()).result()
        return refused, session.optim_step().result().step


def _discard(engine: _Engine) -> object:
    with engine.llm.training_session() as session:
        session.forward_backward(_batch()).result()
        session.discard_accumulation()
        return _outcome(lambda: session.optim_step().result())


def _close_with_accumulation(engine: _Engine) -> object:
    session = engine.llm.training_session()
    session.__enter__()
    session.forward_backward(_batch()).result()
    refused = _outcome(session.close)
    session.discard_accumulation()
    session.close()
    return refused


def _abort_and_reenter(engine: _Engine) -> object:
    with pytest.raises(KeyError), engine.llm.training_session() as session:
        session.forward_backward(_batch())
        running = engine.running()
        raise KeyError("the caller fails mid-batch")
    return running, _train_step(engine)


def _abort_behind_a_batch(engine: _Engine) -> object:
    """The abort waits for the running batch, then moves the device with no new input to
    wake the engine; a failed abort shows as a note on the caller's error."""

    with pytest.raises(KeyError) as raised, engine.llm.training_session() as session:
        submitted = session.forward_backward(_batch())
        running = engine.running()
        raise KeyError("the caller fails mid-batch")
    return running, _outcome(submitted.result), getattr(raised.value, "__notes__", [])


def _stale_session(engine: _Engine) -> object:
    with engine.llm.training_session() as session:
        pass
    return _outcome(lambda: session.forward_backward(_batch()))


def _sample_between_sessions(engine: _Engine) -> object:
    _train_step(engine)
    return engine.generate([1, 2], 2)


def _shutdown_with_open_session(engine: _Engine) -> object:
    session = engine.llm.training_session()
    session.__enter__()
    session.forward_backward(_batch())
    return engine.running(), _outcome(engine.llm.shutdown)


def _requests_take_the_device_between_microbatches(engine: _Engine) -> object:
    """A request that arrives during a batch takes the device at the next microbatch's
    boundary; the batch then finishes."""

    with engine.llm.training_session() as session:
        submitted = session.forward_backward(_batch())
        sampled = engine.generate([1, 2], 2)
        trained = submitted.result().normalization_count
        return sampled, trained, session.optim_step().result().step


def _pause_after_serving_took_the_device(engine: _Engine) -> object:
    """Requests take the device once the accepted batch ends; a pause then has nothing
    to move and is answered at once."""

    with engine.llm.training_session() as session:
        submitted = session.forward_backward(_batch())
        sampled = engine.generate([1, 2], 2)
        session.pause_for_serving()
        session.resume_training()
        trained = submitted.result().normalization_count
        return sampled, trained, session.optim_step().result().step


def _behind_a_long_request(paused: bool) -> Callable[[_Engine], object]:
    """A switch command waits out a request longer than its acknowledgement bound: the
    drain renews the wait instead of failing the engine. Entering a session waits for
    the device; so does closing a paused one, which takes the device back first."""

    def drive(engine: _Engine) -> object:
        session = engine.llm.training_session()
        if paused:
            session.__enter__()
            session.pause_for_serving()
        sampled: list[object] = []
        request = Thread(target=lambda: sampled.append(engine.generate([1, 2, 3], 40)))
        request.start()
        while not any(call.startswith("rank0 seeds") for call in engine.device()):
            time.sleep(0.01)
        if not paused:
            session.__enter__()
        session.close()
        request.join()
        return sampled == [(7,) * 40]

    return drive


def _save_checkpoint(engine: _Engine) -> object:
    path = engine.tmp_path / "checkpoint"
    with engine.llm.training_session() as session:
        session.forward_backward(_batch()).result()
        session.optim_step().result()
        session.save_checkpoint(path)
    return path.exists()


def _save_checkpoint_under_a_file(engine: _Engine) -> object:
    """A path the engine cannot write refuses the save; the session saves elsewhere."""

    (engine.tmp_path / "file").touch()
    with engine.llm.training_session() as session:
        refused = _outcome(lambda: session.save_checkpoint(engine.tmp_path / "file" / "checkpoint"))
        session.save_checkpoint(engine.tmp_path / "checkpoint")
    return refused, (engine.tmp_path / "checkpoint").exists()


def _failure(call: Callable[[], object]) -> tuple[FailurePhase, str]:
    """Where the engine failed, and the head of its message, as ``call`` sees it."""

    try:
        call()
    except EngineProcessError as error:
        return error.phase, str(error).partition(":")[0]
    raise AssertionError("the call did not fail")


def _failed_forward_backward(engine: _Engine) -> object:
    """An error a training command did not mean as a refusal ends the engine."""

    session = engine.llm.training_session()
    session.__enter__()
    failed = _failure(lambda: session.forward_backward(_batch()).result())
    return failed, _failure(
        lambda: engine.llm.generate([[1, 2]], SamplingParams(temperature=0, max_new_tokens=2))
    )


def _failed_sample(engine: _Engine) -> object:
    """Where the engine failed, and the head of its message, as two samples see it."""

    def sample() -> object:
        return engine.llm.generate([[1, 2, 3]], SamplingParams(temperature=0, max_new_tokens=3))

    return _failure(sample), _failure(sample)


@dataclass(frozen=True)
class Row:
    name: str
    drive: Callable[[_Engine], object]
    sees: object
    device: tuple[str, ...] | None = None  # the device calls, in order, when pinned
    sent: dict[str, int] = field(default_factory=dict)  # parent -> engine, by type
    requests: dict[str, int] = field(default_factory=dict)  # parent -> request link, by type
    faults: tuple[Raise, ...] = ()
    ranks: int = 1
    training: bool = True
    microbatch_seconds: float = 0.0  # long enough for the row to act while one runs
    step_seconds: float = 0.0  # each forward's duration
    seeded: bool = False  # the engine serves only seeded rows
    ack_seconds: float = EngineClient.ACK_SECONDS  # bounds a switch's acknowledgement
    unanswered: int = 0  # the commands the engine's failure leaves unanswered
    startup_seconds: tuple[float | str, ...] = ()  # the stand-in's startup phases, each reported
    weight_shards: tuple[float, ...] = ()  # the stand-in's checkpoint shards' read times
    # Where training yields to waiting requests; None keeps the engine's default.
    pause_boundary: TrainingPauseBoundary | None = None


_STARTED = ("rank0 quiesce",)
_ENTER = ("rank0 quiesce", "rank0 switch_to_training")
_BATCH = ("rank0 microbatch 0.0", "rank0 microbatch 0.1")
_NEXT_BATCH = ("rank0 microbatch 1.0", "rank0 microbatch 1.1")
_LEAVE = ("rank0 quiesce", "rank0 suspend_training", "rank0 resume_inference")
# Serving takes the device between two microbatches and gives it back: training sets its
# batch aside and brings it back.
_YIELD = (
    "rank0 quiesce",
    "rank0 suspend_training_execution",
    "rank0 suspend_training",
    "rank0 resume_inference",
    "rank0 quiesce",
    "rank0 switch_to_training",
    "rank0 resume_training_execution",
)
_ROWS = (
    Row("sample", _sample, (7, 7, 7), sent={"ReadRevision": 0}, training=False),
    *(
        Row(
            f"sample-{count}-together",
            _sample_together(count),
            (((7, 7),) * count, count),
            requests={"message": 1, "UserMsg": count},
            training=False,
            seeded=True,
        )
        for count in (2, 4)
    ),
    Row(
        "chat-4-together",
        _chat_together,
        tuple(((index,) * 3, (7, 7)) for index in range(1, 5)),
        requests={"message": 1, "UserMsg": 4},
        training=False,
    ),
    Row("sample-too-long", _sample_too_long, "ValueError", sent={"ReadRevision": 0}),
    Row(
        "sample-past-the-context",
        _sample_past_the_context,
        (0, RejectionKind.CONTEXT_LENGTH_EXCEEDED),
        training=False,
    ),
    # The frontend refuses it first; without that check the engine's admission rejects it.
    Row("bad-sampling", _bad_sampling, ("SamplingError", (7, 7, 7)), training=False),
    Row(
        "train-step",
        _train_step,
        (2, 1, 2),
        device=(*_STARTED, *_ENTER, *_BATCH, *_LEAVE),
        sent={
            "EnterTrainingSession": 1,
            "SubmitForwardBackward": 1,
            "ObserveForwardBackward": 1,
            "SubmitOptimizerStep": 1,
            "ObserveOptimizerStep": 1,
            "CloseTrainingSession": 1,
        },
    ),
    Row("two-forward-backwards", _two_forward_backwards, ((2, 2), 4)),
    Row("optim-without-accumulation", _optim_without_accumulation, ("InvalidGradientError", 1)),
    Row("discard", _discard, "InvalidGradientError"),
    Row("close-with-accumulation", _close_with_accumulation, "CommandStateError"),
    Row(
        "abort-and-reenter",
        _abort_and_reenter,
        (True, (2, 1, 2)),
        device=(*_STARTED, *_ENTER, *_BATCH, *_LEAVE, *_ENTER, *_NEXT_BATCH, *_LEAVE),
        sent={"AbortTrainingSession": 1, "ObserveForwardBackward": 1},
        microbatch_seconds=0.5,
    ),
    Row(
        "abort-behind-a-batch",
        _abort_behind_a_batch,
        (True, "RuntimeError", []),
        device=(*_STARTED, *_ENTER, *_BATCH, *_LEAVE),
        sent={"AbortTrainingSession": 1, "ObserveForwardBackward": 0},
        microbatch_seconds=0.5,
        ack_seconds=10.0,  # the abort once waited for input that never came
    ),
    Row("stale-session", _stale_session, "RuntimeError"),
    Row("sample-between-sessions", _sample_between_sessions, (7, 7)),
    Row(
        "shutdown-with-open-session",
        _shutdown_with_open_session,
        (True, None),
        device=(*_STARTED, *_ENTER, *_BATCH, "rank0 quiesce"),
        sent={"ObserveForwardBackward": 0, "Shutdown": 1},
        microbatch_seconds=0.5,
    ),
    Row(
        "requests-take-the-device-between-microbatches",
        _requests_take_the_device_between_microbatches,
        ((7, 7), 2, 1),
        device=(*_STARTED, *_ENTER, _BATCH[0], *_YIELD, _BATCH[1], *_LEAVE),
        microbatch_seconds=0.5,
        pause_boundary=TrainingPauseBoundary.MICROBATCH,
    ),
    Row(
        "pause-after-serving-took-the-device",
        _pause_after_serving_took_the_device,
        ((7, 7), 2, 1),
        device=(*_STARTED, *_ENTER, *_BATCH, *_LEAVE, *_ENTER, *_LEAVE),
        ack_seconds=10.0,  # the pause once hung: fail it here, not after the default
        # Requests wait out the accepted batch: the device moves only after it.
        pause_boundary=TrainingPauseBoundary.LOGICAL_BATCH,
    ),
    *(
        Row(
            f"{name}-behind-a-long-request",
            _behind_a_long_request(paused),
            True,
            sent={"EnterTrainingSession": 1, "CloseTrainingSession": 1},
            seeded=True,  # each forward logs its seeds: the request runs
            step_seconds=0.1,  # 40 tokens take 4 s
            ack_seconds=2.0,
        )
        for name, paused in (("enter", False), ("close-paused", True))
    ),
    Row("save-checkpoint", _save_checkpoint, True),
    Row(
        "save-checkpoint-under-a-file", _save_checkpoint_under_a_file, ("TrainingInputError", True)
    ),
    Row(
        "failed-forward-backward",
        _failed_forward_backward,
        ((FailurePhase.RUNNING, "the engine failed while running on rank 0"),) * 2,
        sent={"SubmitForwardBackward": 1},
        faults=(Raise("tests.support.cpu_engine:CpuTrainingRunner.forward_backward"),),
        unanswered=1,
    ),
    Row(
        "failed-sample",
        _failed_sample,
        ((FailurePhase.RUNNING, "the engine failed while running on rank 0"),) * 2,
        faults=(Raise("tests.support.cpu_engine:CpuWorker.submit"),),
        training=False,
    ),
    Row(
        "tp-first-failure",
        _failed_sample,
        # A peer reports its own failure, on its own pipe.
        ((FailurePhase.RUNNING, "the engine failed while running on rank 1"),) * 2,
        faults=(Raise("tests.support.cpu_engine:CpuWorker.submit", "cti-rank-1"),),
        ranks=2,
        training=False,
    ),
    Row(
        "tp-peer-killed",
        _failed_sample,
        # A peer that dies without reporting is named, with its exit code.
        ((FailurePhase.RUNNING, "cti-rank-1 exited unexpectedly (exit code 9)"),) * 2,
        faults=(Exit("tests.support.cpu_engine:CpuWorker.submit", "cti-rank-1"),),
        ranks=2,
        training=False,
    ),
    Row("tp-seeds", _seeded_sample, ((7, 7, 7), True), ranks=2, training=False, seeded=True),
)


def _training(pause_boundary: TrainingPauseBoundary | None) -> TrainingConfig:
    training = TrainingConfig(
        trainable_plan=TrainablePlan(("weight",)),
        optimizer=AdamWConfig(),
        batching=TrainingBatching(
            BatchLayout.PADDED, StaticBatching(microbatch_size=1, max_tokens_per_sample=64)
        ),
    )
    if pause_boundary is None:
        return training
    return replace(training, training_pause_boundary=pause_boundary)


@pytest.fixture
def start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Callable[..., _Engine]]:
    """Starts engines on the CPU stand-in, with the row's faults; shuts them down after."""

    engines: list[_Engine] = []
    cleanup: list[str] = []

    def engine(row: Row, probe: IpcProbe) -> _Engine:
        use_memory_saver_provider(monkeypatch)
        # A training batch's tensors reach the engine as shared fds, served from a socket
        # in multiprocessing's temp dir (fixed at its first use), and its request link is an
        # ipc socket in the engine's own temp dir: keep both within AF_UNIX's path limit.
        monkeypatch.setattr(tempfile, "tempdir", "/tmp")
        assert len(multiprocessing.util.get_temp_dir()) < 64
        short = tempfile.mkdtemp(prefix="cti-")
        cleanup.append(short)
        monkeypatch.setattr(tempfile, "tempdir", short)
        CpuEngine.frontend(monkeypatch)
        injector = FaultInjector((*CpuEngine.injections(), *probe.injections(), *row.faults))
        for key, value in injector.environment(os.environ.get("PYTHONPATH", "")).items():
            monkeypatch.setenv(key, value)
        log = tmp_path / "device.log"
        monkeypatch.setenv(CpuEngine.LOG, str(log))
        monkeypatch.setenv(CpuEngine.MICROBATCH_SECONDS, str(row.microbatch_seconds))
        monkeypatch.setenv(CpuEngine.STEP_SECONDS, str(row.step_seconds))
        monkeypatch.setattr(EngineClient, "ACK_SECONDS", row.ack_seconds)
        if row.seeded:
            monkeypatch.setenv(CpuEngine.SEEDED, "1")
        if row.startup_seconds:
            monkeypatch.setenv(CpuEngine.STARTUP_SECONDS, ",".join(map(str, row.startup_seconds)))
        if row.weight_shards:
            monkeypatch.setenv(
                CpuEngine.WEIGHT_SHARD_SECONDS, ",".join(map(str, row.weight_shards))
            )
        llm = LLM(
            EngineOptions(
                model=ModelOptions(path="cpu-model"),
                parallel=ParallelOptions(num_ranks=row.ranks),
                scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=256),
                training=_training(row.pause_boundary) if row.training else None,
            )
        )
        engines.append(_Engine(llm, log, tmp_path))
        return engines[-1]

    yield engine
    for running in engines:
        running.llm.shutdown()
    for directory in cleanup:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.parametrize("row", _ROWS, ids=lambda row: row.name)
def test_command_table(start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, row: Row) -> None:
    (tmp_path / "calls").mkdir()
    probe = IpcProbe(tmp_path / "calls")
    engine = start(row, probe)

    with probe.parent(monkeypatch):
        assert row.drive(engine) == row.sees
    if row.device is not None:
        assert tuple(engine.device()) == row.device
    assert {name: probe.sent[name] for name in row.sent} == row.sent
    assert {name: probe.requests[name] for name in row.requests} == row.requests
    # Every command answered once, but those the engine's failure cut short.
    assert probe.replies == probe.sent.total() - row.unanswered
    assert not probe.calls()  # no collective outside a TP group's own steps


def test_an_offline_engine_removes_its_own_endpoints(start, tmp_path: Path) -> None:
    """Without serving endpoints the engine samples and trains on private ones, which are
    gone once it shuts down."""

    (tmp_path / "calls").mkdir()
    engine = start(Row("offline", _train_step, None), IpcProbe(tmp_path / "calls"))

    owned = list(Path(tempfile.gettempdir()).glob("cti-*"))
    assert (engine.generate([1, 2], 2), _train_step(engine)) == ((7, 7), (2, 1, 2))
    engine.llm.shutdown()
    assert len(owned) == 1 and not owned[0].exists()


_STALL_SECONDS = 4.0  # long enough for a rank to spawn and report its first phase


def test_a_startup_that_reports_progress_outlives_the_stall_window(
    start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each reported phase renews the wait: a startup longer than the stall window serves."""

    monkeypatch.setattr(EngineClient, "STARTUP_STALL_SECONDS", _STALL_SECONDS)
    phases = (_STALL_SECONDS * 0.6,) * 3
    (tmp_path / "calls").mkdir()
    began = time.monotonic()
    engine = start(
        Row("slow-startup", _sample, None, startup_seconds=phases), IpcProbe(tmp_path / "calls")
    )

    assert time.monotonic() - began > _STALL_SECONDS
    assert _sample(engine) == (7, 7, 7)


def test_a_stalled_startup_fails_after_the_stall_window(
    start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A phase that reports nothing for the stall window fails startup, long before it ends."""

    monkeypatch.setattr(EngineClient, "STARTUP_STALL_SECONDS", _STALL_SECONDS)
    (tmp_path / "calls").mkdir()
    began = time.monotonic()
    with pytest.raises(
        EngineProcessError, match="no startup progress .* stand-in phase 0"
    ) as error:
        start(
            Row("stalled-startup", _sample, None, startup_seconds=(60.0,)),
            IpcProbe(tmp_path / "calls"),
        )

    assert error.value.phase is FailurePhase.STARTUP
    assert _STALL_SECONDS < time.monotonic() - began < 30.0


def test_a_startup_whose_rank_runs_a_compiler_outlives_the_stall_window(
    start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A JIT build's one long step reports nothing, but a rank's live compiler renews the
    wait: a build longer than the stall window serves."""

    monkeypatch.setattr(EngineClient, "STARTUP_STALL_SECONDS", _STALL_SECONDS)
    (tmp_path / "calls").mkdir()
    began = time.monotonic()
    engine = start(
        Row(
            "compiling-startup", _sample, None, startup_seconds=(f"compile:{_STALL_SECONDS * 1.5}",)
        ),
        IpcProbe(tmp_path / "calls"),
    )

    assert time.monotonic() - began > _STALL_SECONDS * 1.5
    assert _sample(engine) == (7, 7, 7)


def test_a_compiling_startup_fails_past_the_progress_bound(
    start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A compiler renews the wait only within ``PROGRESS_SECONDS`` of the startup's
    beginning: one that never ends fails startup."""

    monkeypatch.setattr(EngineClient, "STARTUP_STALL_SECONDS", _STALL_SECONDS)
    monkeypatch.setattr(EngineClient, "PROGRESS_SECONDS", _STALL_SECONDS * 1.5)
    (tmp_path / "calls").mkdir()
    began = time.monotonic()
    with pytest.raises(
        EngineProcessError, match=r"still compiling \(nvcc\) after .* stand-in phase 0"
    ) as error:
        start(
            Row("endless-compile", _sample, None, startup_seconds=("compile:60",)),
            IpcProbe(tmp_path / "calls"),
        )

    assert error.value.phase is FailurePhase.STARTUP
    assert _STALL_SECONDS * 1.5 < time.monotonic() - began < 30.0


def test_a_weight_load_that_reports_each_shard_outlives_the_stall_window(
    start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each checkpoint shard the loader reads renews the wait: a load longer than the
    stall window serves."""

    monkeypatch.setattr(EngineClient, "STARTUP_STALL_SECONDS", _STALL_SECONDS)
    shards = (_STALL_SECONDS * 0.6,) * 3
    (tmp_path / "calls").mkdir()
    began = time.monotonic()
    engine = start(
        Row("slow-weights", _sample, None, weight_shards=shards), IpcProbe(tmp_path / "calls")
    )

    assert time.monotonic() - began > _STALL_SECONDS
    assert _sample(engine) == (7, 7, 7)


def test_a_stalled_weight_shard_fails_startup_naming_it(
    start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A shard whose read reports nothing for the stall window fails startup."""

    monkeypatch.setattr(EngineClient, "STARTUP_STALL_SECONDS", _STALL_SECONDS)
    (tmp_path / "calls").mkdir()
    began = time.monotonic()
    with pytest.raises(
        EngineProcessError,
        match=r"no startup progress .* loading weights: shard 2/2 \(shard-1.safetensors\)",
    ):
        start(
            Row("stalled-weights", _sample, None, weight_shards=(0.1, 60.0)),
            IpcProbe(tmp_path / "calls"),
        )

    assert _STALL_SECONDS < time.monotonic() - began < 30.0


def test_a_rank_that_exits_mid_startup_fails_startup_as_an_exit(
    start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rank's exit fails startup at once as an exit, not as a stalled phase."""

    monkeypatch.setattr(EngineClient, "STARTUP_STALL_SECONDS", 60.0)
    (tmp_path / "calls").mkdir()
    began = time.monotonic()
    with pytest.raises(EngineProcessError, match="exited unexpectedly") as error:
        start(
            Row("exiting-startup", _sample, None, startup_seconds=(0.1, "exit")),
            IpcProbe(tmp_path / "calls"),
        )

    assert error.value.phase is FailurePhase.STARTUP
    assert time.monotonic() - began < 30.0


def test_a_directory_that_cannot_be_removed_still_fails_waiting_calls(
    start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The engine's failure reaches the waiting call, with its cause, even when its private
    directory cannot be removed (a network TMPDIR keeps open files' ``.nfs*`` stand-ins)."""

    def busy(directory: tempfile.TemporaryDirectory) -> None:
        raise OSError(39, "Directory not empty", directory.name)

    (tmp_path / "calls").mkdir()
    row = Row(
        "undeletable-directory",
        _sample,
        None,
        faults=(Raise("tests.support.cpu_engine:CpuTrainingRunner.forward_backward"),),
    )
    engine = start(row, IpcProbe(tmp_path / "calls"))
    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", busy)
    session = engine.llm.training_session()
    session.__enter__()
    seen: list[BaseException] = []

    def wait() -> None:
        try:
            session.forward_backward(_batch()).result()
        except EngineProcessError as error:
            seen.append(error)

    waiter = Thread(target=wait, daemon=True)
    waiter.start()
    waiter.join(30.0)

    assert len(seen) == 1, "the waiting call never saw the engine's failure"
    assert seen[0].phase is FailurePhase.RUNNING
    assert str(seen[0]).startswith("the engine failed while running on rank 0: ")


def test_a_peers_failure_names_its_rank_phase_and_exception(start, tmp_path: Path) -> None:
    """A peer's own report reaches the parent whole: its rank, the phase and its error."""

    (tmp_path / "calls").mkdir()
    row = Row(
        "tp-peer-report",
        _sample,
        None,
        faults=(Raise("tests.support.cpu_engine:CpuWorker.submit", "cti-rank-1"),),
        ranks=2,
        training=False,
    )
    engine = start(row, IpcProbe(tmp_path / "calls"))
    with pytest.raises(EngineProcessError) as error:
        engine.llm.generate([[1, 2, 3]], SamplingParams(temperature=0, max_new_tokens=3))

    assert str(error.value) == (
        "the engine failed while running on rank 1: builtins.RuntimeError: injected fault"
    )
    assert any("injected fault" in note for note in error.value.__notes__)
