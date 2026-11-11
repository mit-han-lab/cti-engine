from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

import cti.utils.torch_utils as torch_utils
from cti.utils import nvtx_annotate


class _Layer:
    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id

    def forward(self, value: int) -> int:
        return value + self.layer_id


def test_the_probe_finds_nvtx_exactly_on_cuda_builds() -> None:
    # CPU and CUDA builds of torch both run this, so a moved binding fails here instead of
    # silently dropping every range on a CUDA build.
    assert torch_utils._torch_has_nvtx() is torch.backends.cuda.is_built()


def test_without_nvtx_the_method_is_left_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    # A CPU-only torch build has no NVTX binding, and its ranges raise.
    monkeypatch.delattr(torch._C, "_nvtx", raising=False)

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("NVTX functions not installed. Are you sure you have a CUDA build?")

    monkeypatch.setattr(torch.cuda.nvtx, "range", unavailable)

    forward = nvtx_annotate("Layer_{}", layer_id_field="layer_id")(_Layer.forward)

    assert forward is _Layer.forward
    assert forward(_Layer(3), 2) == 5


def test_with_nvtx_each_call_runs_inside_its_named_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch._C, "_nvtx", object(), raising=False)
    events: list[str] = []

    @contextmanager
    def record(name: str) -> Iterator[None]:
        events.append(f"enter {name}")
        yield
        events.append(f"exit {name}")

    monkeypatch.setattr(torch.cuda.nvtx, "range", record)

    def forward(layer: _Layer, value: int) -> int:
        events.append("call")
        return _Layer.forward(layer, value)

    annotated = nvtx_annotate("Layer_{}", layer_id_field="layer_id")(forward)

    assert annotated.__wrapped__ is forward
    assert annotated(_Layer(3), 2) == 5
    assert events == ["enter Layer_3", "call", "exit Layer_3"]
