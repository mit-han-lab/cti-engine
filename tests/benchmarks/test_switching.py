from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.switching import PROFILE_NAME, SwitchingConfig, _graph_batch_sizes, _parse_args


def test_switching_config_rejects_invalid_memory_ratios(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="memory ratios"):
        SwitchingConfig(
            model_path=tmp_path / "model",
            output=tmp_path / "receipt.json",
            batch_size=4,
            prompt_tokens=256,
            warmup=1,
            measure=5,
            rank=32,
            alpha=64.0,
            seed=7,
            inference_memory_ratio=0.9,
            peak_memory_ratio=0.8,
        )


def test_switching_cli_defaults_define_all_layer_release_profile(tmp_path: Path) -> None:
    config = _parse_args(
        [
            "--model-path",
            str(tmp_path / "model"),
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    assert PROFILE_NAME == "cti-switching-v1"
    assert config.cache_type == "radix"
    assert config.batch_size == 4
    assert config.measure == 5
    assert _graph_batch_sizes(config.batch_size) == [1, 2, 4]
