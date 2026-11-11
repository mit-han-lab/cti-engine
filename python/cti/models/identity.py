from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ._download import CheckpointSource

_SIGNATURE_DOMAIN = "cti-training-model-v2"


def model_signature(
    *,
    source: CheckpointSource,
    config: object,
    dtype: torch.dtype,
    trainable: Mapping[str, object] | None,
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> str:
    """Identify one immutable model construction without hashing tensor values: ``source``
    names the checkpoint (``CheckpointSource.identity``), and ``config`` is the model
    class's own configuration (``CONFIG``), its whole geometry. Its tensor
    layout is each tensor's name and dtype, not its shape: a rank's shapes depend on the
    rank count, while a checkpoint holds logical tensors whose shapes restore checks.
    ``trainable`` is the trainable part's own identity (``Adapter.identity``)."""

    tensors = []
    names: set[str] = set()
    for name, tensor in named_tensors:
        if type(name) is not str or not name:
            raise ValueError("model tensor names must be non-empty strings")
        if name in names:
            raise ValueError(f"model tensor name is duplicated: {name!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model state {name!r} must be a tensor")
        names.add(name)
        tensors.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
        )
    if not tensors:
        raise ValueError("model tensor layout must not be empty")

    payload = {
        "source": source.identity(),
        "config": asdict(config),
        "dtype": str(dtype).removeprefix("torch."),
        "trainable": None if trainable is None else dict(trainable),
        "tensor_layout": sorted(tensors, key=lambda tensor: tensor["name"]),
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=_enum_value,
    )
    digest = sha256(f"{_SIGNATURE_DOMAIN}\0{canonical}".encode()).hexdigest()
    return f"sha256:{digest}"


def _enum_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"{type(value).__name__} is not part of a model signature")


__all__ = ["model_signature"]
