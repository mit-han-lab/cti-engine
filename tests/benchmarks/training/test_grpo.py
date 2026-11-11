from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.training.grpo import (
    PROFILE_NAME,
    GRPOConfig,
    _graph_batch_sizes,
    _parse_args,
    _rank_rewards,
)


def _config(tmp_path: Path, **overrides) -> GRPOConfig:
    values = {
        "model_path": tmp_path / "model",
        "output": tmp_path / "receipt.json",
        "layout": "packed",
        "checkpoint": "none",
        "num_prompts": 4,
        "group_size": 8,
        "prompt_tokens": 256,
        "completion_tokens": 64,
        "microbatch_size": 4,
        "warmup": 1,
        "measure": 5,
        "rank": 32,
        "alpha": 64.0,
        "seed": 7,
        "inference_memory_ratio": 0.9,
        "peak_memory_ratio": 0.9,
    }
    values.update(overrides)
    return GRPOConfig(**values)


def test_grpo_config_resolves_rollout_batch_and_rejects_partial_microbatch(
    tmp_path: Path,
) -> None:
    assert _config(tmp_path).rollout_batch_size == 32
    with pytest.raises(ValueError, match="divisible"):
        _config(tmp_path, microbatch_size=3)


def test_grpo_rewards_and_graph_sizes_cover_exact_groups() -> None:
    assert _rank_rewards(8, 4) == (0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0)
    assert _graph_batch_sizes(5) == [1, 2, 4, 5]
    with pytest.raises(ValueError, match="complete groups"):
        _rank_rewards(7, 4)


def test_grpo_cli_defaults_define_edge_owned_profile(tmp_path: Path) -> None:
    config = _parse_args(
        [
            "--model-path",
            str(tmp_path / "model"),
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    assert PROFILE_NAME == "cti-grpo-v1"
    assert config.layout == "packed"
    assert config.rollout_batch_size == 32
    assert config.microbatch_size == 4
