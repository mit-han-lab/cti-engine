from __future__ import annotations

from pathlib import Path

import pytest

from benchmark._runtime import REPO_ROOT
from benchmark.inference.online import (
    PROFILE_NAME,
    OnlineCase,
    OnlineConfig,
    _parse_args,
    poisson_arrivals,
)


def test_online_case_parser_names_dimensions_and_rejects_invalid_shapes() -> None:
    case = OnlineCase.parse("512x64x32x4.5")

    assert case == OnlineCase(512, 64, 32, 4.5)
    assert case.name == "i512-o64-n32-r4p5"

    with pytest.raises(ValueError, match="positive"):
        OnlineCase(0, 64, 32, 4)
    with pytest.raises(ValueError, match="at least two"):
        OnlineCase(64, 1, 32, 4)
    with pytest.raises(ValueError, match="request rate"):
        OnlineCase(64, 64, 32, 0)
    with pytest.raises(ValueError, match="sequence limit"):
        OnlineCase(32768, 64, 32, 4)


def test_poisson_arrivals_are_seeded_monotonic_and_start_immediately() -> None:
    first = poisson_arrivals(8, 4.0, seed=7)
    second = poisson_arrivals(8, 4.0, seed=7)
    changed = poisson_arrivals(8, 4.0, seed=8)

    assert first == second
    assert first != changed
    assert first[0] == 0.0
    assert all(left < right for left, right in zip(first, first[1:], strict=False))


def test_online_config_requires_unique_cases_and_external_artifacts(tmp_path: Path) -> None:
    case = OnlineCase(64, 64, 8, 4)
    config = OnlineConfig(tmp_path, (case,), 1, 2, 19400)
    assert config.cases == (case,)

    with pytest.raises(ValueError, match="unique"):
        OnlineConfig(tmp_path, (case, case), 1, 2, 19400)
    with pytest.raises(ValueError, match="outside"):
        OnlineConfig(REPO_ROOT / "artifacts", (case,), 1, 2, 19400)


def test_cli_defaults_define_a_distinct_online_profile(tmp_path: Path) -> None:
    config = _parse_args(["--output-dir", str(tmp_path)])

    assert PROFILE_NAME == "cti-online-inference-api-v1"
    assert config.runs == 3
    assert config.warmup_requests == 4
    assert config.cases == (
        OnlineCase(64, 64, 32, 8),
        OnlineCase(512, 64, 32, 4),
    )
