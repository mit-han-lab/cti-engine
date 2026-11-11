from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.training.sft import PROFILE_NAME, SFTConfig, _build_batch, _parse_args


def _config(tmp_path: Path, **overrides) -> SFTConfig:
    values = {
        "model_path": tmp_path / "model",
        "output": tmp_path / "receipt.json",
        "layout": "packed",
        "checkpoint": "none",
        "batch_size": 4,
        "microbatch_size": 2,
        "sequence_length": 16,
        "warmup": 1,
        "measure": 5,
        "rank": 32,
        "alpha": 64.0,
        "seed": 7,
        "inference_memory_ratio": 0.9,
        "peak_memory_ratio": 0.9,
    }
    values.update(overrides)
    return SFTConfig(**values)


@pytest.mark.parametrize("layout", ("padded", "packed"))
def test_sft_workload_has_exact_shape_and_target_count(tmp_path: Path, layout: str) -> None:
    config = _config(tmp_path, layout=layout)
    batch = _build_batch(config, vocab_size=512, pad_token_id=0)

    assert batch.batch_size == 4
    assert batch.num_targets == 4 * 15


def test_sft_config_rejects_invalid_accumulation_and_memory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divisible"):
        _config(tmp_path, microbatch_size=3)
    with pytest.raises(ValueError, match="memory ratios"):
        _config(tmp_path, inference_memory_ratio=0.9, peak_memory_ratio=0.8)


def test_sft_cli_defaults_define_edge_owned_profile(tmp_path: Path) -> None:
    config = _parse_args(
        [
            "--model-path",
            str(tmp_path / "model"),
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    assert PROFILE_NAME == "cti-sft-v1"
    assert config.layout == "packed"
    assert config.batch_size == config.microbatch_size == 4
    assert config.measure == 5
