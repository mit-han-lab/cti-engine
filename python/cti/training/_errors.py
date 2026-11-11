from ._protocol import Refusal


class InvalidGradientError(Refusal, RuntimeError):
    """The current accumulation window cannot be committed safely."""


class TrainingInputError(Refusal, ValueError):
    """A training batch, policy or checkpoint path is invalid; training is as it was."""


class TrainingResidencyError(ValueError):
    """The configured ``TrainingStateResidency`` does not apply to this training."""


class SwitchRejectedError(Refusal, RuntimeError):
    """A switch between serving and training did not fit the device; the current phase
    keeps it."""


__all__ = [
    "InvalidGradientError",
    "SwitchRejectedError",
    "TrainingInputError",
    "TrainingResidencyError",
]
