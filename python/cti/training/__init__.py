from __future__ import annotations

from typing import TYPE_CHECKING

from cti.utils import lazy_getattr

from ._errors import (
    InvalidGradientError,
    SwitchRejectedError,
    TrainingInputError,
)
from ._protocol import CommandStateError, Refusal, TrainingVerb

if TYPE_CHECKING:
    from ._batch import PackedTokens, PaddedTokens, SFTDatum, TrainingBatch
    from ._checkpoint import CheckpointFormatError, TrainingCheckpoint
    from ._objective import GRPO, SFT
    from ._protocol import (
        ForwardBackwardCompletion,
        ForwardBackwardObservation,
        OptimizerStepResult,
    )

# Every training type another package uses; the runner and the scheduler keep public modules.
# An import loads only its defining module (PEP 562).
_LAZY_ATTRS = {
    "CheckpointFormatError": "._checkpoint",
    "TrainingCheckpoint": "._checkpoint",
    "ForwardBackwardObservation": "._protocol",
    "PackedTokens": "._batch",
    "PaddedTokens": "._batch",
    "SFTDatum": "._batch",
    "TrainingBatch": "._batch",
    "ForwardBackwardCompletion": "._protocol",
    "GRPO": "._objective",
    "SFT": "._objective",
    "OptimizerStepResult": "._protocol",
}

__getattr__ = lazy_getattr(__name__, _LAZY_ATTRS, globals())


__all__ = [
    "CheckpointFormatError",
    "CommandStateError",
    "ForwardBackwardCompletion",
    "ForwardBackwardObservation",
    "GRPO",
    "InvalidGradientError",
    "OptimizerStepResult",
    "PackedTokens",
    "PaddedTokens",
    "Refusal",
    "SFT",
    "SFTDatum",
    "SwitchRejectedError",
    "TrainingBatch",
    "TrainingCheckpoint",
    "TrainingInputError",
    "TrainingVerb",
]
