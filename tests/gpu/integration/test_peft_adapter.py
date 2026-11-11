"""A PEFT adapter loaded at launch on one GPU follows Transformers with the same adapter,
teacher-forced."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.peft_adapter import ADAPTERS, assert_launch_adapter_follows_transformers


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(model, marks=pytest.mark.checkpoints(checkpoint), id=model)
        for model, (checkpoint, _, _) in ADAPTERS.items()
    ],
)
def test_a_launch_adapter_follows_transformers_with_peft(tmp_path: Path, model: str) -> None:
    assert_launch_adapter_follows_transformers(tmp_path, num_ranks=1, model=model)
