from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import torch

from cti.config import AdamWConfig

_FORMAT = "cti-training-checkpoint"
_VERSION = 5
# What each older version lacks, for its refusal.
_CHANGES = MappingProxyType(
    {
        3: ", which dropped the residual_dtype field with the FP32 residual stream",
        4: ", which keeps every parameter's FP32 master beside its published live value",
    }
)
# PyTorch's non-capturable AdamW step is an FP32 scalar: increments reach 2**24 and
# then leave it unchanged. The logical step count stays a separate native integer.
ADAMW_BACKEND_STEP_SATURATION = 1 << 24
_KEYS = frozenset(
    {
        "format",
        "version",
        "model_signature",
        "num_optimizer_steps",
        "model_revision",
        "config",
        "names",
        "live",
        "master",
        "adamw",
    }
)


class CheckpointFormatError(ValueError):
    """A training checkpoint does not satisfy the versioned disk contract."""


class OptimizerState(NamedTuple):
    """Selected parameters and their AdamW state on the host, as logical (unsharded)
    tensors in parameter order: each parameter's published live value (what inference
    reads), its FP32 master (what training reached, published or not), and torch AdamW's
    own ``state_dict()``, whose ``state`` is keyed by the same order."""

    config: AdamWConfig
    names: tuple[str, ...]
    live: tuple[torch.Tensor, ...]
    master: tuple[torch.Tensor, ...]
    adamw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """State required to resume one committed training boundary, on any rank count."""

    model_signature: str
    num_optimizer_steps: int
    model_revision: int
    optimizer: OptimizerState

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically publish this checkpoint at *path* on one filesystem."""

        optimizer = self.optimizer
        payload = {
            "format": _FORMAT,
            "version": _VERSION,
            "model_signature": self.model_signature,
            "num_optimizer_steps": self.num_optimizer_steps,
            "model_revision": self.model_revision,
            "config": asdict(optimizer.config),
            "names": optimizer.names,
            "live": optimizer.live,
            "master": optimizer.master,
            "adamw": optimizer.adamw,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> TrainingCheckpoint:
        """Load and validate a checkpoint on the host without touching a runtime."""

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
            raise CheckpointFormatError(f"cannot decode training checkpoint: {error}") from error
        try:
            return _validated(payload)
        except CheckpointFormatError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise CheckpointFormatError(f"invalid training checkpoint: {error}") from error


def _validated(payload: object) -> TrainingCheckpoint:
    """The one check of a loaded checkpoint: schema, names, shapes, dtypes, finite values
    and steps that agree with the logical optimizer-step count."""

    if type(payload) is not dict or payload.get("format") != _FORMAT:
        raise CheckpointFormatError("unsupported checkpoint format")
    if (version := payload.get("version")) != _VERSION:
        change = _CHANGES.get(version, "")
        raise CheckpointFormatError(
            f"checkpoint version {version} is not supported: this engine reads version "
            f"{_VERSION}{change}"
        )
    if frozenset(payload) != _KEYS:
        raise CheckpointFormatError(f"checkpoint keys do not match: {sorted(payload)}")
    num_optimizer_steps = payload["num_optimizer_steps"]
    model_revision = payload["model_revision"]
    for owner, count in (
        ("num_optimizer_steps", num_optimizer_steps),
        ("model_revision", model_revision),
    ):
        if type(count) is not int or count < 0:
            raise CheckpointFormatError(f"{owner} must be a non-negative integer")
    signature = payload["model_signature"]
    if type(signature) is not str or not signature:
        raise CheckpointFormatError("model_signature must be a non-empty string")

    names, live, master, adamw = (payload[key] for key in ("names", "live", "master", "adamw"))
    if (
        not names
        or len(set(names)) != len(names)
        or any(type(n) is not str or not n for n in names)
    ):
        raise CheckpointFormatError("parameter names must be unique non-empty strings")
    states = adamw["state"]
    if not len(live) == len(master) == len(names) == len(states):
        raise CheckpointFormatError("parameters, masters and AdamW states must align")
    step = min(num_optimizer_steps, ADAMW_BACKEND_STEP_SATURATION)
    for index, (name, value, fp32) in enumerate(zip(names, live, master, strict=True)):
        _require_finite(name, value, value.shape, value.dtype)
        _require_finite(f"{name} master", fp32, value.shape, torch.float32)
        state = states[index]
        if float(state["step"]) != step:
            raise CheckpointFormatError(
                f"{name}: AdamW step {float(state['step'])} does not match the optimizer-step "
                f"count {num_optimizer_steps}"
            )
        for moment in ("exp_avg", "exp_avg_sq"):
            _require_finite(f"{name} {moment}", state[moment], value.shape, torch.float32)
    return TrainingCheckpoint(
        model_signature=signature,
        num_optimizer_steps=num_optimizer_steps,
        model_revision=model_revision,
        optimizer=OptimizerState(AdamWConfig(**payload["config"]), names, live, master, adamw),
    )


def _require_finite(owner: str, tensor: object, shape: torch.Size, dtype: torch.dtype) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype:
        raise CheckpointFormatError(f"{owner} must be a {dtype} tensor of shape {tuple(shape)}")
    if dtype not in (torch.float32, torch.bfloat16) or not bool(torch.isfinite(tensor).all()):
        raise CheckpointFormatError(f"{owner} must contain only finite FP32 or BF16 values")


__all__ = ["CheckpointFormatError", "OptimizerState", "TrainingCheckpoint"]
