from __future__ import annotations

import pytest

from tests.support.residency import (
    MEMORY_TOLERANCE_BYTES,
    PhysicalMemoryContract,
    assert_warmed_training_windows,
)


def _window(training: int, inference: int = 2_000_000_000) -> dict[str, int]:
    return {"free_training_bytes": training, "free_inference_bytes": inference}


def test_cold_pool_capacity_is_excluded_only_from_training_stability() -> None:
    # A failure seen on an RTX 5090: the cold pool retains 458 MiB more than later pools.
    warmup = _window(12_807_241_728)
    windows = [_window(13_287_489_536) for _ in range(10)]
    assert_warmed_training_windows(PhysicalMemoryContract.EXACT, warmup, windows, 64 << 20)


@pytest.mark.parametrize("phase", ["training", "inference"])
@pytest.mark.parametrize("index", [0, 5, 9])
@pytest.mark.parametrize("direction", [-1, 1])
def test_every_measured_window_still_has_the_original_tolerance(
    phase: str, index: int, direction: int
) -> None:
    warmup = _window(12_807_241_728)
    windows = [_window(13_287_489_536) for _ in range(10)]
    windows[index][f"free_{phase}_bytes"] += direction * (MEMORY_TOLERANCE_BYTES + 1)
    with pytest.raises(AssertionError, match=f"{phase}-phase free memory"):
        assert_warmed_training_windows(PhysicalMemoryContract.EXACT, warmup, windows, 64 << 20)


@pytest.mark.parametrize(("moved", "passes"), [(10 << 20, True), (33 << 20, False)])
def test_the_first_switch_may_only_load_kernel_modules(moved: int, passes: bool) -> None:
    # Cold kernel caches load modules on the first switch: 10 MiB on an RTX 5090.
    warmup = _window(12_807_241_728, 2_000_000_000 + moved)
    windows = [_window(13_287_489_536) for _ in range(10)]
    if passes:
        assert_warmed_training_windows(PhysicalMemoryContract.EXACT, warmup, windows, 64 << 20)
        return
    with pytest.raises(AssertionError, match="across the first switch"):
        assert_warmed_training_windows(PhysicalMemoryContract.EXACT, warmup, windows, 64 << 20)


def test_cold_window_must_release_its_extra_capacity_before_inference() -> None:
    warmup = _window(12_807_241_728, 2_000_000_000 - 480_247_808)
    windows = [_window(13_287_489_536) for _ in range(10)]
    with pytest.raises(AssertionError, match="inference-phase free memory"):
        assert_warmed_training_windows(PhysicalMemoryContract.EXACT, warmup, windows, 64 << 20)


@pytest.mark.parametrize(
    "device", ["NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 6000 Blackwell Server Edition"]
)
@pytest.mark.parametrize("phase", ["entry", "exit"])
def test_qwen35_switch_gate_rejects_latency_regressions(device, phase, monkeypatch, tmp_path):
    from tests.support.switching import (
        QWEN35_SWITCH_LATENCY_BOUNDS_MS,
        assert_qwen35_switch_latency,
    )

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    key = (device, (12, 0))
    medians = dict(QWEN35_SWITCH_LATENCY_BOUNDS_MS[key])
    assert_qwen35_switch_latency(key, medians, "none")
    medians[phase] += 0.01
    with pytest.raises(AssertionError):
        assert_qwen35_switch_latency(key, medians, "full_offload")
    assert "GATED" in summary.read_text()


def test_unqualified_switch_device_is_explicitly_not_gated(monkeypatch, tmp_path, capsys):
    from tests.support.switching import assert_qwen35_switch_latency

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert_qwen35_switch_latency(("unknown", (99, 0)), {"entry": 10000.0, "exit": 10000.0}, "none")
    assert "NOT GATED" in summary.read_text()
    assert "NOT GATED" in capsys.readouterr().out


@pytest.mark.parametrize("path", ["direct_entry", "direct_exit", "session_entry", "session_exit"])
def test_qwen3_switch_gate_rejects_each_path_regression(path, monkeypatch, tmp_path):
    from tests.support.switching import assert_qwen3_switch_latency

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    device = ("NVIDIA GeForce RTX 5090", (12, 0))
    medians = {"direct_entry": 60.0, "direct_exit": 4.0, "session_entry": 60.0, "session_exit": 4.0}
    assert_qwen3_switch_latency(device, medians)
    medians[path] = 150.0 if path.endswith("entry") else 20.0
    with pytest.raises(AssertionError):
        assert_qwen3_switch_latency(device, medians)
    assert "Qwen3 switching" in summary.read_text()
    assert "NOT GATED" not in summary.read_text()


@pytest.mark.parametrize("device", ["NVIDIA RTX PRO 6000 Blackwell Server Edition", "unknown"])
def test_qwen3_ungated_devices_reach_ci_summary(device, monkeypatch, tmp_path, capsys):
    from tests.support.switching import assert_qwen3_switch_latency, assert_qwen35_switch_latency

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert_qwen35_switch_latency(("unknown", (99, 0)), {"entry": 1.0, "exit": 1.0}, "none")
    assert_qwen3_switch_latency(
        (device, (12, 0)),
        {
            "direct_entry": 10000.0,
            "direct_exit": 10000.0,
            "session_entry": 10000.0,
            "session_exit": 10000.0,
        },
    )
    report = summary.read_text()
    assert len(report.splitlines()) == 2
    assert f"Qwen3 switching — {device}" in report
    assert report.count("NOT GATED") == 2
    assert "NOT GATED" in capsys.readouterr().out
