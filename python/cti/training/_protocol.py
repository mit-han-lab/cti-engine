"""Tensor-free training values shared across ownership boundaries: the training commands'
vocabulary, their refusals and their results. Imports no torch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class Refusal(Exception):
    """A command the engine refuses, its state as before the command: the engine goes on.
    Only a ``Refusal`` refuses a command; any other error a command raises ends the engine."""


class CommandStateError(Refusal, RuntimeError):
    """The command does not fit the state the engine or its training session is in: the
    caller sent it out of order."""


class TrainingVerb(Enum):
    """A training command's verb, which the ``TrainingScheduler`` runs with the command's id
    and args."""

    ENTER_TRAINING_SESSION = "enter_training_session"
    SUBMIT_FORWARD_BACKWARD = "submit_forward_backward"  # (batch,); answered with its id
    OBSERVE_FORWARD_BACKWARD = "observe_forward_backward"  # (submission_id,); observations
    SUBMIT_OPTIMIZER_STEP = "submit_optimizer_step"  # answered with its submission id
    OBSERVE_OPTIMIZER_STEP = "observe_optimizer_step"  # (submission_id,); (result, observations)
    DISCARD_ACCUMULATION = "discard_accumulation"
    SAVE_TRAINING_CHECKPOINT = "save_training_checkpoint"  # (path,)
    SAVE_ADAPTER = "save_adapter"  # (path,)
    PUBLISH_ADAPTER = "publish_adapter"  # answered with the new model revision
    CLOSE_TRAINING_SESSION = "close_training_session"
    ABORT_TRAINING_SESSION = "abort_training_session"
    PAUSE_FOR_SERVING = "pause_for_serving"  # ``Progress`` after each microbatch renews it
    RESUME_TRAINING = "resume_training"


@dataclass(frozen=True, slots=True)
class ForwardBackwardCompletion:
    """Host result of one forward/backward submission: its summed loss, the count its
    gradients normalize by, and its objective's other metrics."""

    loss_sum: float
    normalization_count: int
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ForwardBackwardObservation:
    """One completed training submission identified inside the training engine."""

    submission_id: int
    completion: ForwardBackwardCompletion


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    """Host result of one successfully committed logical optimizer step."""

    step: int
    normalization_count: int
    grad_norm: float


__all__ = [
    "CommandStateError",
    "ForwardBackwardCompletion",
    "ForwardBackwardObservation",
    "OptimizerStepResult",
    "Refusal",
    "TrainingVerb",
]
