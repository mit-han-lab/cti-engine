"""The parent's commands to its engine and the engine's replies, and the progress record each
rank shares with the parent that watches it.

The parent and the engine's rank 0 are one trusted run, so values are pickled as they are.
Each is a frozen value: ``EngineCore`` matches the commands and ``EngineClient`` the
replies. A command carries the id its answer replies to.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from cti.training import Refusal, TrainingVerb

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

    from transformers import PretrainedConfig

    from cti.multimodal import ImageLimits


class FailurePhase(Enum):
    """Where an engine failed; its value completes "the engine failed while ..."."""

    STARTUP = "starting"
    RUNNING = "running"
    SHUTDOWN = "shutting down"


@dataclass(frozen=True, slots=True)
class EngineInfo:
    """What a frontend needs to know of the engine it serves, fixed once it starts."""

    model_path: str  # the checkpoint the engine resolved
    hf_config: PretrainedConfig  # the checkpoint's, for its prompt encoder
    image_limits: ImageLimits  # how many images a request may carry, and of what size
    checks: bool  # whether the engine runs its internal invariant checks; the parent's too
    chat_template: str | None = None  # the tokenizer's, which tool-call parsing reads


# ================================ Parent -> engine ================================


@dataclass(frozen=True, slots=True)
class TrainingCommand:
    """The training session's command ``verb`` with its ``args``."""

    id: int
    verb: TrainingVerb
    args: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class ReadRevision:
    """Answered with the revision of the weights serving runs on."""

    id: int


@dataclass(frozen=True, slots=True)
class Shutdown:
    """Answered once the engine is quiescent; the parent then reclaims its processes."""

    id: int


# ================================ Engine -> parent ================================


@dataclass(frozen=True, slots=True)
class Answered:
    """The command with ``id`` returned ``value``."""

    id: int
    value: object = None


@dataclass(frozen=True, slots=True)
class Progress:
    """The command with ``id`` is still running; its caller's wait is renewed."""

    id: int


@dataclass(frozen=True, slots=True)
class Refused:
    """The command with ``id`` raised ``error``; the engine goes on."""

    id: int
    error: Refusal
    remote_traceback: str

    @classmethod
    def capture(cls, command_id: int, error: Refusal) -> Refused:
        return cls(command_id, error, "".join(traceback.format_exception(error)))


@dataclass(frozen=True, slots=True)
class Starting:
    """The engine's startup entered ``phase``; the parent's wait for ``Started`` is renewed."""

    phase: str


@dataclass(frozen=True, slots=True)
class Started:
    """The engine serves."""

    info: EngineInfo


@dataclass(frozen=True, slots=True)
class CoreFailed:
    """The engine failed and every rank is exiting."""

    phase: FailurePhase
    type_name: str
    message: str
    remote_traceback: str

    @classmethod
    def capture(cls, phase: FailurePhase, error: BaseException) -> CoreFailed:
        error_type = type(error)
        return cls(
            phase,
            f"{error_type.__module__}.{error_type.__qualname__}",
            str(error),
            "".join(traceback.format_exception(error)),
        )


class RankProgress:
    """One rank's progress, in memory it shares with the parent that watches it: ``steps``,
    the loop turns it finished, and ``allowance``, the seconds its current work may take to
    finish one, 0 while it waits for work."""

    def __init__(self, shared: object) -> None:
        self._shared = shared

    @classmethod
    def shared(cls, context: BaseContext) -> RankProgress:
        """A record in shared memory, handed to the rank as it is spawned."""

        return cls(context.Array("d", 2, lock=False))

    @property
    def steps(self) -> float:
        return self._shared[0]

    @property
    def allowance(self) -> float:
        return self._shared[1]

    def working(self, allowance: float) -> None:
        self._shared[1] = allowance

    def idle(self) -> None:
        self._shared[1] = 0.0

    def finished(self) -> None:
        self._shared[0] += 1


__all__ = [
    "Answered",
    "CoreFailed",
    "EngineInfo",
    "FailurePhase",
    "Progress",
    "RankProgress",
    "ReadRevision",
    "Refused",
    "Shutdown",
    "Started",
    "Starting",
    "TrainingCommand",
]
