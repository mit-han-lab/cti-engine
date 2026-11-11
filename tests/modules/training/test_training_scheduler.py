"""TrainingScheduler over a host TrainingRunner, driven through its commands: submission
accounting, and the runner's stream-ordered, region-scoped execution."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from cti.config import (
    AdamWConfig,
    BatchLayout,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
    TrainingPauseBoundary,
)
from cti.memory import MemoryManager, ResidencyRestoreError
from cti.protocol import SampledSequence, SamplingResult, SamplingSpec
from cti.sampling import LogprobMode, TokenSampling
from cti.training import (
    SFT,
    CommandStateError,
    InvalidGradientError,
    PaddedTokens,
    SwitchRejectedError,
    TrainingBatch,
    TrainingInputError,
    TrainingVerb,
)
from cti.training._checkpoint import TrainingCheckpoint
from cti.training._optimizer import AdamWOptimizer
from cti.training.runner import TrainingRunner
from tests.support.training import TrainingDriver, TrainingModelStub

_SIGNATURE = "test-model-signature"
_SCOPE_BYTES = 1 << 20


class _Stream:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name
        self.sync_error: BaseException | None = None

    def __enter__(self) -> None:
        self._events.append(f"{self._name}.stream+")

    def __exit__(self, *_exc: object) -> None:
        self._events.append(f"{self._name}.stream-")

    def wait_stream(self, stream: _Stream) -> None:
        self._events.append(f"{self._name}.wait:{stream._name}")

    def synchronize(self) -> None:
        self._events.append(f"{self._name}.sync")
        if self.sync_error is not None:
            raise self.sync_error


class _Memory:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    @contextmanager
    def persistent_training_allocation(self) -> Iterator[None]:
        self._events.append("persistent+")
        yield
        self._events.append("persistent-")

    @contextmanager
    def transient_training_allocation(self, scope_bytes: int) -> Iterator[None]:
        assert scope_bytes == _SCOPE_BYTES
        self._events.append("transient+")
        yield
        self._events.append("transient-")


class _Model(TrainingModelStub):
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.eye(8), requires_grad=False)
        self.loss_scale = 1.0

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (("weight", self.weight),)

    def forward_training(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = F.one_hot(input_ids.remainder(8), num_classes=8).float() @ self.trained(
            self.weight
        )
        return logits * self.loss_scale


class _Harness:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        boundary: TrainingPauseBoundary = TrainingPauseBoundary.LOGICAL_BATCH,
        signature: str = _SIGNATURE,
        memory: object | None = None,
    ) -> None:
        self.events: list[str] = []
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: stream)
        self.model = _Model()
        self.stream = _Stream(self.events, "training")
        self.memory = _Memory(self.events) if memory is None else memory
        # The device's per-thread stack size, as each command reads it.
        self.stack_bytes = 1024
        self.runner = TrainingRunner.create(
            self.model,
            TrainingConfig(
                TrainablePlan(("weight",)),
                AdamWConfig(lr=0.1, max_grad_norm=None),
                TrainingBatching(
                    BatchLayout.PADDED, StaticBatching(microbatch_size=1, max_tokens_per_sample=8)
                ),
            ),
            trainables=self.model,
            stream=self.stream,  # type: ignore[arg-type]
            after=_Stream(self.events, "inference"),  # type: ignore[arg-type]
            memory=self.memory,
            scope_bytes=_SCOPE_BYTES,
            model_signature=lambda: signature,
            adapter_writer=lambda path, factors: self.events.append(f"save adapter {path}"),
            ranks=None,
            placed=None,
            stack_limit=lambda: self.stack_bytes,
            checks=True,
        )
        self.driver = TrainingDriver(self.runner, boundary)
        self.scheduler = self.driver.scheduler


def _batch(rows: int = 1) -> TrainingBatch:
    tokens = PaddedTokens(
        torch.arange(1, 3 * rows + 1, dtype=torch.int64).reshape(rows, 3),
        torch.full((rows,), 3, dtype=torch.int64),
        0,
    )
    return TrainingBatch(tokens, torch.ones((rows, 2), dtype=torch.bool), SFT())


def _grpo_batch(revision: int) -> TrainingBatch:
    spec = SamplingSpec(
        TokenSampling(0.7, logprobs=LogprobMode.PROCESSED), max_new_tokens=2, seed=0
    )
    samples = tuple(
        SampledSequence(
            request_id=request_id,
            prompt_token_ids=(1, 2),
            sampled_token_ids=(3, 4),
            sampling_logprobs=(-0.4, -0.7),
            cached_prompt_tokens=0,
            sampling=replace(spec, seed=request_id),
            finish_reason="length",
        )
        for request_id in range(2)
    )
    result = SamplingResult(samples=samples, texts=("", ""), model_revision=revision)
    return TrainingBatch.grpo(result, [1.0, 3.0], BatchLayout.PADDED, group_size=2, pad_token_id=0)


def _assert_window_empty(driver: TrainingDriver) -> None:
    with pytest.raises(InvalidGradientError, match="requires a preceding forward_backward"):
        driver.optim_step()


def _checkpoint(driver: TrainingDriver, path: object) -> TrainingCheckpoint:
    driver.call(TrainingVerb.SAVE_TRAINING_CHECKPOINT, str(path))
    return TrainingCheckpoint.load(path)


def test_each_microbatch_runs_fenced_on_the_training_stream_in_a_transient_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    driver, scheduler = harness.driver, harness.scheduler

    submitted = driver.send(TrainingVerb.SUBMIT_FORWARD_BACKWARD, _batch(rows=2))
    assert harness.events == []
    scheduler.step()
    assert scheduler.has_active
    assert harness.events[:4] == [
        "training.wait:inference",
        "training.stream+",
        "transient+",
        "persistent+",
    ]
    assert harness.events[-3:] == ["transient-", "training.stream-", "training.sync"]
    scheduler.step()
    assert not scheduler.has_active
    (observation,) = driver.observe(driver.answer(submitted))
    assert observation.completion.normalization_count == 4

    harness.events.clear()
    result, _ = driver.optim_step()
    assert result.step == 1
    assert harness.events[:3] == ["training.wait:inference", "training.stream+", "transient+"]


@pytest.mark.parametrize("stack_bytes", [1024, 2112])
def test_a_command_that_grows_the_device_stack_after_sizing_fails(
    monkeypatch: pytest.MonkeyPatch, stack_bytes: int
) -> None:
    harness = _Harness(monkeypatch)
    # A kernel the command first launches needs more local memory than the device allowed.
    harness.stack_bytes = stack_bytes

    if stack_bytes == 1024:
        harness.runner.discard()
        return
    with pytest.raises(RuntimeError, match="from 1024 to 2112 bytes"):
        harness.runner.discard()


@pytest.mark.parametrize(
    ("boundary", "between_microbatches"),
    [(TrainingPauseBoundary.LOGICAL_BATCH, False), (TrainingPauseBoundary.MICROBATCH, True)],
)
def test_training_is_at_a_boundary_between_batches_and_by_policy_between_microbatches(
    monkeypatch: pytest.MonkeyPatch, boundary: TrainingPauseBoundary, between_microbatches: bool
) -> None:
    harness = _Harness(monkeypatch, boundary)
    scheduler = harness.scheduler

    assert scheduler.drained()
    harness.driver.send(TrainingVerb.SUBMIT_FORWARD_BACKWARD, _batch(rows=2))
    scheduler.step()
    assert scheduler.drained() is between_microbatches
    scheduler.step()
    assert scheduler.drained()


def test_ids_count_accepted_batches_and_one_commit_takes_every_window_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _Harness(monkeypatch).driver

    first = driver.forward_backward(_batch(rows=1))
    second = driver.forward_backward(_batch(rows=2))
    assert (first, second) == (0, 1)
    (observed,) = driver.observe(first)
    with pytest.raises(RuntimeError, match="already observed"):
        driver.observe(first)
    with pytest.raises(RuntimeError, match="unknown"):
        driver.observe(5)

    result, rest = driver.optim_step()

    assert observed.completion.normalization_count == 2
    assert (result.step, result.normalization_count) == (1, 6)
    assert [item.submission_id for item in rest] == [second]
    _assert_window_empty(driver)


def test_a_command_waits_for_the_accepted_batch_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)
    driver = harness.driver
    submitted = driver.send(TrainingVerb.SUBMIT_FORWARD_BACKWARD, _batch(rows=2))
    discard = driver.send(TrainingVerb.DISCARD_ACCUMULATION)

    assert driver.answered(submitted) and not driver.answered(discard)
    driver.run()
    driver.answer(discard)
    assert not harness.runner.holds_gradients
    _assert_window_empty(driver)


def test_a_paused_session_refuses_training_commands_and_keeps_its_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    driver = harness.driver
    scheduler = harness.scheduler
    driver.forward_backward(_batch())
    with pytest.raises(CommandStateError, match="not paused for serving"):
        driver.call(TrainingVerb.RESUME_TRAINING)
    pause = driver.send(TrainingVerb.PAUSE_FOR_SERVING)
    scheduler.switched(False, None)
    driver.answer(pause)

    for verb, args in (
        (TrainingVerb.SUBMIT_FORWARD_BACKWARD, (_batch(),)),
        (TrainingVerb.SUBMIT_OPTIMIZER_STEP, ()),
        (TrainingVerb.DISCARD_ACCUMULATION, ()),
        (TrainingVerb.SAVE_TRAINING_CHECKPOINT, ("unused",)),
        (TrainingVerb.SAVE_ADAPTER, ("unused",)),
        (TrainingVerb.PUBLISH_ADAPTER, ()),
        (TrainingVerb.PAUSE_FOR_SERVING, ()),
    ):
        command_id = driver.send(verb, *args, holds=False)
        with pytest.raises(CommandStateError, match="paused for serving"):
            driver.answer(command_id)

    resume = driver.send(TrainingVerb.RESUME_TRAINING, holds=False)
    scheduler.switched(True, None)
    driver.answer(resume)
    assert harness.runner.holds_gradients
    driver.call(TrainingVerb.DISCARD_ACCUMULATION)


@pytest.mark.parametrize("refused", [False, True], ids=["fits", "refused-then-retried"])
@pytest.mark.parametrize(
    "verb", [TrainingVerb.CLOSE_TRAINING_SESSION, TrainingVerb.ABORT_TRAINING_SESSION]
)
def test_ending_a_paused_session_takes_the_device_back_first(
    monkeypatch: pytest.MonkeyPatch, verb: TrainingVerb, refused: bool
) -> None:
    harness = _Harness(monkeypatch)
    driver, scheduler = harness.driver, harness.scheduler
    pause = driver.send(TrainingVerb.PAUSE_FOR_SERVING)
    scheduler.switched(False, None)
    driver.answer(pause)

    end = driver.send(verb, holds=False)
    assert scheduler.requested_holds is True
    if refused:  # training's memory does not fit: the session stays paused
        scheduler.switched(False, ResidencyRestoreError("training", 2, 1))
        with pytest.raises(SwitchRejectedError):
            driver.answer(end)
        assert scheduler.requested_holds is None
        end = driver.send(verb, holds=False)
    scheduler.switched(True, None)
    driver.run()
    assert scheduler.requested_holds is False
    scheduler.switched(False, None)
    driver.answer(end)
    with pytest.raises(CommandStateError, match="needs an open training session"):
        driver.call(TrainingVerb.PAUSE_FOR_SERVING)


def test_commands_outside_a_session_or_behind_a_queued_one_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _Harness(monkeypatch).driver
    queued = driver.send(TrainingVerb.SUBMIT_OPTIMIZER_STEP, holds=False)
    with pytest.raises(CommandStateError, match="already queued"):
        driver.answer(driver.send(TrainingVerb.DISCARD_ACCUMULATION, holds=False))
    driver.run()
    with pytest.raises(InvalidGradientError):  # the queued command ran, not overwritten
        driver.call(TrainingVerb.OBSERVE_OPTIMIZER_STEP, driver.answer(queued))

    close = driver.send(TrainingVerb.CLOSE_TRAINING_SESSION)
    driver.run()
    driver.scheduler.switched(False, None)
    driver.answer(close)
    for verb, args in (
        (TrainingVerb.PAUSE_FOR_SERVING, ()),
        (TrainingVerb.SUBMIT_FORWARD_BACKWARD, (_batch(),)),
        (TrainingVerb.SAVE_TRAINING_CHECKPOINT, ("unused",)),
        (TrainingVerb.SAVE_ADAPTER, ("unused",)),
        (TrainingVerb.PUBLISH_ADAPTER, ()),
    ):
        with pytest.raises(CommandStateError, match="needs an open training session"):
            driver.answer(driver.send(verb, *args, holds=False))


def test_one_window_trains_one_objective_from_the_revision_that_sampled_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    driver = harness.driver
    before = harness.model.weight.detach().clone()
    driver.forward_backward(_batch())
    driver.optim_step()
    assert harness.scheduler.model_revision == 0
    torch.testing.assert_close(harness.model.weight, before, rtol=0, atol=0)
    assert driver.publish() == 1
    assert not torch.equal(harness.model.weight, before)

    with pytest.raises(ValueError, match="does not match policy revision 1"):
        driver.forward_backward(_grpo_batch(revision=0))
    driver.forward_backward(_batch())
    with pytest.raises(InvalidGradientError, match="cannot mix"):
        driver.forward_backward(_grpo_batch(revision=1))
    # The refused batches left the open window as it was.
    assert driver.optim_step()[0].normalization_count == 2


def test_an_unexpected_microbatch_failure_ends_the_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)
    driver = harness.driver

    def fail(*_args: object) -> torch.Tensor:
        raise ValueError("injected microbatch failure")

    monkeypatch.setattr(harness.model, "training_loss", fail)
    # Not a Refusal: it raises on to the process's root instead of refusing the observation.
    with pytest.raises(ValueError, match="injected"):
        driver.observe(driver.forward_backward(_batch(rows=2)))


@pytest.mark.parametrize("nonfinite", ["observed", "committed"])
def test_a_nonfinite_loss_discards_the_window_without_failing_training(
    monkeypatch: pytest.MonkeyPatch, nonfinite: str
) -> None:
    harness = _Harness(monkeypatch)
    driver = harness.driver
    harness.model.loss_scale = math.nan
    submission_id = driver.forward_backward(_batch())

    with pytest.raises(InvalidGradientError, match="finite"):
        if nonfinite == "observed":
            driver.observe(submission_id)
        else:
            driver.optim_step()

    _assert_window_empty(driver)
    harness.model.loss_scale = 1.0
    driver.forward_backward(_batch())
    assert driver.optim_step()[0].step == 1


def test_a_training_failure_raises_to_the_process_root(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)
    driver = harness.driver
    driver.forward_backward(_batch())
    failure = RuntimeError("injected optimizer failure")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(AdamWOptimizer, "apply_gradients", fail)
    with pytest.raises(RuntimeError) as raised:
        driver.optim_step()

    assert raised.value is failure


def test_a_failed_stream_fence_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)
    harness.stream.sync_error = RuntimeError("device lost")

    with pytest.raises(RuntimeError, match="device lost"):
        harness.driver.forward_backward(_batch())


def test_the_runner_asks_its_memory_only_what_memory_manager_offers(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    driver = _Harness(monkeypatch, memory=MagicMock(spec=MemoryManager)).driver

    driver.observe(driver.forward_backward(_batch()))
    driver.optim_step()
    driver.forward_backward(_batch())
    driver.discard()

    assert _checkpoint(driver, tmp_path / "checkpoint").num_optimizer_steps == 1


def test_runner_suspension_stashes_gradients_in_place_and_moves_metrics_back_persistently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, TrainingPauseBoundary.MICROBATCH)
    driver, scheduler = harness.driver, harness.scheduler
    driver.forward_backward(_batch())
    driver.send(TrainingVerb.SUBMIT_FORWARD_BACKWARD, _batch(rows=2))
    scheduler.step()
    harness.events.clear()

    assert scheduler.drained()
    held = harness.runner.memory_usage().gradients
    scheduler.release()
    # The gradients keep their device memory; the stash holds their values.
    assert harness.runner.memory_usage().gradients == held
    scheduler.switched(True, ResidencyRestoreError("serving", 2, 1))  # serving did not fit

    assert harness.events == [
        "training.stream+",
        "training.stream-",
        "training.sync",
        "training.stream+",
        "persistent+",
        "persistent-",
        "training.stream-",
    ]
    assert driver.optim_step()[0].normalization_count == 6


def test_an_adapter_saves_its_settled_factors_and_a_refused_save_leaves_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    driver = harness.driver
    driver.forward_backward(_batch())
    with pytest.raises(CommandStateError, match="committed or discarded"):
        driver.call(TrainingVerb.SAVE_ADAPTER, "/adapter")
    driver.optim_step()

    driver.call(TrainingVerb.SAVE_ADAPTER, "/adapter")
    assert harness.events[-1] == "save adapter /adapter"
    saved = {}
    monkeypatch.setattr(
        harness.runner, "_adapter_writer", lambda path, factors: saved.update(factors)
    )
    driver.call(TrainingVerb.SAVE_ADAPTER, "/adapter")
    torch.testing.assert_close(
        saved["weight"], harness.model.trained.masters["weight"], rtol=0, atol=0
    )
    assert not torch.equal(saved["weight"], harness.model.weight)
    assert harness.scheduler.model_revision == 0

    def refuse(path: str, factors: object) -> None:
        raise ValueError("no PEFT form")

    monkeypatch.setattr(harness.runner, "_adapter_writer", refuse)
    with pytest.raises(TrainingInputError, match="cannot save the adapter: no PEFT form"):
        driver.call(TrainingVerb.SAVE_ADAPTER, "/adapter")
    driver.forward_backward(_batch())  # training goes on
    driver.optim_step()


def test_a_checkpoint_carries_the_step_and_restores_only_into_its_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    source = _Harness(monkeypatch)
    source.driver.forward_backward(_batch())
    source.driver.optim_step()
    assert source.driver.publish() == 1
    with_window = source.driver.forward_backward(_batch())
    with pytest.raises(RuntimeError, match="committed or discarded"):
        _checkpoint(source.driver, tmp_path / "early")
    source.driver.observe(with_window)
    source.driver.discard()

    checkpoint = _checkpoint(source.driver, tmp_path / "checkpoint")
    target = _Harness(monkeypatch)
    target.scheduler.restore(tmp_path / "checkpoint")

    assert (checkpoint.num_optimizer_steps, checkpoint.model_revision) == (1, 1)
    # The published weight, which inference reads, moved with the step and came back.
    assert not torch.equal(source.model.weight, torch.eye(8))
    torch.testing.assert_close(target.model.weight, source.model.weight, rtol=0, atol=0)
    target.driver.forward_backward(_batch())
    assert target.driver.optim_step()[0].step == 2
    with pytest.raises(ValueError, match="model signature"):
        _Harness(monkeypatch, signature="another").scheduler.restore(tmp_path / "checkpoint")


def test_close_releases_the_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)
    harness.driver.forward_backward(_batch())

    harness.runner.close()

    assert harness.runner.holds_gradients
