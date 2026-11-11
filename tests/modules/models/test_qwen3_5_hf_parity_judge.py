"""Qwen3.5's default accuracy tier on synthetic rows: ``measure``, ``check`` and
``derive_bounds``."""

from __future__ import annotations

import math

import pytest
import torch

from tests.support import qwen3_5_hf_parity_judge as absolute
from tests.support.qwen3_5_hf_parity_judge import SELF_CHECKS
from tests.support.qwen3_5_hf_parity_judge import synthetic_measure as _measure

ABS_BOUNDS = {"gap.max": 0.5, "gap.sum": 1.0, "err.max": 10.0, "err.mean": 1.0}


@pytest.mark.parametrize(
    "fault", "gap.max finite rank.max gap.sum err.mean self_check coverage".split()
)
def test_absolute_counterexamples(fault):
    z = torch.eye(8, 32) * 8
    e, options = z.clone(), {}
    if fault == "gap.max":
        z = z.roll(1, 0)
    if fault == "finite":
        z[1, 3] = float("nan")
        outside = _measure(z, e, stop_ids=(0,))
        assert outside["finite"] and not outside["finite_beyond"]
    if fault == "rank.max":
        z[0, :9] = torch.arange(9).float()
    if fault == "gap.sum":
        z[:, 31] = 8.3
    if fault == "err.mean":
        options["selected"] = e.log_softmax(-1).max(-1).values - 2
    if fault == "self_check":
        options["self_checks"] = {**SELF_CHECKS, "first_logprob_error": 0.663}
    if fault == "coverage":
        options.update(stop_ids=(0,), coverage_critical=True)
    failures = {
        name for _, name in absolute.check({"r": _measure(z, e, **options)}, ABS_BOUNDS)["failures"]
    }
    assert fault in failures
    report = absolute.check({"r": _measure(z, e, **options)}, None, report_only=True)
    assert (report["status"] == "FAIL") == (fault in ("finite", "self_check", "coverage"))
    if fault in ("gap.sum", "err.mean"):
        assert failures == {fault}


def test_absolute_stop_report_bounds_and_alignment():
    m = _measure(stop_ids=(1,))
    assert m["gated"] == 2 and m["stats"]["gap.sum"] == 0
    assert absolute.check({"r": m}, None)["status"] == "REPORT_ONLY"
    assert absolute.check({"r": m}, ABS_BOUNDS)["status"] == "PASS"
    assert absolute.check({"r": m}, ABS_BOUNDS, required=("missing",))["status"] == "FAIL"
    assert absolute.QUALIFICATION_BOUNDS == {}
    with pytest.raises(ValueError, match="REPORT"):
        absolute.check({"r": m}, ABS_BOUNDS, report_only=True)
    z = torch.eye(8, 32) * 8
    compact = _measure(z, prompt_tokens=100, aligned=True)
    assert compact["stats"] == _measure(z)["stats"]
    for bad in (
        {},
        {**ABS_BOUNDS, "typo": 1},
        {**ABS_BOUNDS, "gap.sum": 0.1},
        *({**ABS_BOUNDS, "err.max": v} for v in (-1, float("nan"), float("inf"), True)),
    ):
        with pytest.raises(ValueError, match="illegal bounds"):
            absolute.check({"r": m}, bad)
    with pytest.raises(ValueError, match="do not pick"):
        _measure(tokens=torch.arange(8).roll(1))


@pytest.mark.parametrize("value", [4.5, 10.5])
def test_dense_bounds_use_full_vocabulary_and_only_eligible_gated_samples(value):
    z = torch.zeros(2, 64)
    z[:, :2] = torch.tensor([10.0, 9.5])
    e = z.clone()
    e[:, 1], e[:, 2:41] = value, 6.0
    m = _measure(z, e)
    assert (1 in e[0].topk(20).indices) == (value > 10)
    assert m["near_tie"]["ids"].tolist() == [1, 1]
    assert m["near_tie"]["margin_shift"].tolist() == [value - 9.5] * 2
    sample = {"id": "same", "source": "hf-bf16-eager", "measurement": m}
    excluded = {**sample, "id": "excluded", "measurement": {**m, "bf16_reference": True}}
    records = [sample, sample, {**excluded, "image": True}, {**excluded, "proxy": True}]
    result = absolute.derive_bounds(records)
    error = float((e.log_softmax(-1) - z.log_softmax(-1)).abs()[:, 1].max())
    assert result["hf_floor"]["err.max"] == pytest.approx(error)
    for name, grid in absolute.BOUND_GRIDS.items():
        observed = max(m["stats"][name], m["dense"].get(name, 0))
        assert result["bounds"][name] == pytest.approx(math.ceil(1.25 * observed / grid) * grid)
    assert result["unique_samples"] == 1 and result["setters"]["err.max"]["id"] == "same"
