"""The engine's control plane. The parent's side, this facade, spawns the engine's rank
processes and the tokenizer workers, numbers its commands and routes each reply by id, and
hands out the training session; each rank runs one ``EngineCore`` loop behind it.

Importing it loads neither torch nor a rank's side: the training session loads the training
types, and torch with them, on first use.
"""

from __future__ import annotations

from cti.runtime._client import EngineClient, EngineProcessError
from cti.runtime._protocol import FailurePhase
from cti.utils import lazy_getattr

__getattr__ = lazy_getattr(
    __name__,
    {"NodeAgent": "._nodes", "TrainingFuture": "._training", "TrainingSession": "._training"},
    globals(),
)

__all__ = [
    "EngineClient",
    "EngineProcessError",
    "FailurePhase",
    "NodeAgent",
    "TrainingFuture",
    "TrainingSession",
]
