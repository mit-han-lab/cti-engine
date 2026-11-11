from __future__ import annotations

import os
from pathlib import Path

# Conservative public-session median limits for both checkpoint modes, in ms.
# Rounded worst observed entry/exit medians (RTX 5090: 15.30/2.05, RTX PRO 6000: 61.04/72.92).
QWEN35_SWITCH_LATENCY_BOUNDS_MS = {
    ("NVIDIA GeForce RTX 5090", (12, 0)): {"entry": 50.0, "exit": 10.0},
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0)): {"entry": 150.0, "exit": 160.0},
}


# Upper bounds on the median entry and exit latency of both the direct and the
# session path, per qualified device (name, compute capability). On an RTX 5090, 14
# runs (2026-09-08 to 09-26) measured entry 59.9-61.3 ms and exit
# 3.5-4.4 ms; the bounds leave about 2.5x and 4.5x head room for host noise, and a
# larger slowdown fails the release gate. Entry time grows with the KV bytes it
# unmaps, so another device is gated only once its own bounds are measured.
QWEN3_SWITCH_LATENCY_BOUNDS_MS = {
    ("NVIDIA GeForce RTX 5090", (12, 0)): {"entry": 150.0, "exit": 20.0},
    # Qwen3 on PRO 6000 has no measured baseline yet; explicitly ungated.
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0)): None,
}


def assert_qwen35_switch_latency(
    device: tuple[str, tuple[int, int]], medians: dict[str, float], mode: str
) -> None:
    """Enforce qualified latency bounds and report devices without a latency gate."""

    bounds = QWEN35_SWITCH_LATENCY_BOUNDS_MS.get(device)
    report = _report_switch_latency("Qwen3.5", device, medians, bounds, mode)
    if bounds is not None:
        assert medians["entry"] <= bounds["entry"], report
        assert medians["exit"] <= bounds["exit"], report


def assert_qwen3_switch_latency(
    device: tuple[str, tuple[int, int]], medians: dict[str, float]
) -> None:
    """Gate all four Qwen3 paths where measured and report the other devices as ungated."""
    bounds = QWEN3_SWITCH_LATENCY_BOUNDS_MS.get(device)
    report = _report_switch_latency("Qwen3", device, medians, bounds, "direct/session")
    if bounds is not None:
        for path, median in medians.items():
            assert median < bounds[path.rpartition("_")[2]], report


def _report_switch_latency(
    model: str,
    device: tuple[str, tuple[int, int]],
    medians: dict[str, float],
    bounds: dict[str, float] | None,
    mode: str,
) -> str:
    status = "GATED" if bounds is not None else "NOT GATED (未门禁)"
    report = (
        f"{model} switching — {device[0]} {device[1]}, {mode}: {status}; "
        f"medians_ms={medians}, bounds_ms={bounds}"
    )
    print(report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as output:
            output.write(f"- {report}\n")
    return report
