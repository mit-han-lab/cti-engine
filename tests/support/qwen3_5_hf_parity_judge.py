"""Qwen3.5's default accuracy tier: an engine row's error against a Transformers reference
row on the same tokens (``measure``), checked against bounds (``check``) derived from calibration
samples (``derive_bounds``).
The FP32-truth parity judge itself is model-neutral: ``tests/support/hf_judge.py``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

# Default-tier bounds are intentionally unfrozen; tests.support.hf_judge.BOUNDS belong only to
# legacy G1.
QUALIFICATION_BOUNDS: dict[str, dict[str, float]] = {}
BOUND_GRIDS = {"gap.max": 0.125, "gap.sum": 0.125, "err.max": 0.05, "err.mean": 0.005}


def _statistics(values: torch.Tensor) -> dict[str, float]:
    if not values.numel():
        return {}
    return {name: float(getattr(values.float(), name)()) for name in ("max", "sum", "mean")}


def measure(
    reference: torch.Tensor,
    engine: torch.Tensor,
    tokens: torch.Tensor,
    selected: torch.Tensor,
    *,
    prompt_tokens: int,
    max_tokens: int,
    self_checks: Mapping[str, Any],
    stop_ids: tuple[int, ...] = (),
    coverage_critical: bool = False,
) -> dict[str, Any]:
    """Measure full HF P+t or aligned rows; hooks attest reference settings and indexing."""
    n, vocab = engine.shape
    shape_ok = reference.shape in ((prompt_tokens + n, vocab), (n, vocab))
    if not (0 < n <= max_tokens and prompt_tokens > 0 and shape_ok):
        raise ValueError("misaligned P+t / emitted rows")
    if not (tokens.shape == selected.shape == (n,) and tokens.dtype in (torch.int32, torch.int64)):
        raise ValueError("invalid tokens/selected logprobs")
    if not bool(((tokens >= 0) & (tokens < vocab)).all()) or any(
        t < 0 or t >= vocab for t in stop_ids
    ):
        raise ValueError("invalid token/stop indices")
    z = reference if len(reference) == n else reference[prompt_tokens - 1 : prompt_tokens + n - 1]
    z, e = z.float(), engine.float()
    if bool((e.isfinite().all(-1) & (e.argmax(-1) != tokens)).any()):
        raise ValueError("engine rows do not pick their tokens")
    stop = next((i for i, t in enumerate(tokens.tolist()) if t in stop_ids), n)
    gated = min(n, max_tokens, 32, stop + 1)
    top, top_ids = z.max(-1)
    chosen = z.gather(1, tokens.long()[:, None]).flatten()
    lp, ep = z.log_softmax(-1), e.log_softmax(-1)
    error = (selected.float() - lp.gather(1, tokens.long()[:, None]).flatten()).abs()
    rows = {"gap": top - chosen, "rank": (z > chosen[:, None]).sum(-1), "err": error}
    finite = z.isfinite().all(-1) & e.isfinite().all(-1) & selected.isfinite()
    positions, ids = (
        (top[:, None] - z <= 1).scatter(1, top_ids[:, None], False).nonzero(as_tuple=True)
    )
    lead = e[positions, top_ids[positions]] - e[positions, ids]
    shifts = top[positions] - z[positions, ids] - lead
    alternatives = (ep[positions, ids] - lp[positions, ids]).abs()
    hooks = "dtype ieee disabled_modules indices use_cache_false ignore_eos not_held".split()
    first_error = self_checks.get("first_logprob_error", math.inf)
    self_ok = all(self_checks.get(k) is True for k in hooks)
    self_ok &= reference.dtype in (torch.float32, torch.bfloat16)
    self_ok &= 0 <= first_error <= (1e-4 if reference.dtype == torch.float32 else 0.4)
    stats = {f"{k}.{s}": v for k, r in rows.items() for s, v in _statistics(r[:gated]).items()}
    dense = {"gap.max": shifts[positions < gated], "err.max": alternatives[positions < gated]}
    return {
        "stats": stats,
        "rows": rows,
        "gated": gated,
        "self_check": self_ok,
        "bf16_reference": reference.dtype == torch.bfloat16,
        "coverage": gated >= (min(max_tokens, 32) if coverage_critical else 1),
        "finite": bool(finite[:gated].all()),
        "finite_beyond": bool(finite[gated:].all()),
        "beyond": {k: _statistics(r[gated:]) for k, r in rows.items()},
        "near_tie": {
            "positions": positions,
            "ids": ids,
            "margin_shift": shifts,
            "alternative_error": alternatives,
        },
        "dense": {k: max(0.0, _statistics(v).get("max", 0.0)) for k, v in dense.items()},
        "stop_gaps": top[: min(max_tokens, 32), None] - z[: min(max_tokens, 32), list(stop_ids)],
        "coverage_critical": coverage_critical,
    }


def check(
    measurements, bounds, *, required=(), calibrating=False, report_only=False
) -> dict[str, Any]:
    """Check frozen bounds (None only reports); required names every (case, pass, request)."""
    if report_only and bounds is not None:
        raise ValueError("REPORT cannot apply frozen bounds")
    if bounds is not None:
        legal = set(bounds) == set(BOUND_GRIDS)
        legal &= all(
            type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in bounds.values()
        )
        if not legal or bounds["gap.sum"] < bounds["gap.max"]:
            raise ValueError("illegal bounds keys/values or gap.sum < gap.max")
    failures = [(key, "coverage") for key in required if key not in measurements]
    if not measurements:
        failures.append((None, "coverage"))
    for key, m in measurements.items():
        gates = {name: m[name] for name in ("finite", "self_check", "coverage")}
        gates["rank.max"] = report_only or m["stats"]["rank.max"] <= 7
        for name, bound in (bounds or {}).items():
            gates[name] = m["stats"][name] <= bound + (1e-5 if name.startswith("gap.") else 0)
        if calibrating and bounds is not None and m["coverage_critical"]:
            gates["stop_safety"] = bool((m["stop_gaps"] > bounds["gap.max"]).all())
        failures.extend((key, name) for name, passed in gates.items() if not passed)
    status = "FAIL" if failures else "REPORT_ONLY" if bounds is None else "PASS"
    return {"status": status, "failures": failures}


def derive_bounds(samples) -> dict[str, Any]:
    """Propose from records: id (bitwise-copy identity), source, image, proxy, measurement."""
    eligible = [s for s in samples if not s.get("proxy")]
    eligible = [s for s in eligible if not s.get("image") or not s["measurement"]["bf16_reference"]]
    sources = ("engine", "hf-bf16-eager", "hf-bf16-sdpa", "holdout")
    if not eligible or any(s["source"] not in sources for s in eligible):
        raise ValueError("empty/unknown calibration class")
    if any(check({s["id"]: s["measurement"]}, None)["failures"] for s in eligible):
        raise ValueError("invalid calibration measurement")
    bounds, setters, floors = {}, {}, {}
    for name, grid in BOUND_GRIDS.items():
        candidates = [
            (max(s["measurement"]["stats"][name], s["measurement"]["dense"].get(name, 0)), s)
            for s in eligible
        ]
        value, setter = max(candidates, key=lambda item: item[0])
        bounds[name] = math.ceil(1.25 * value / grid) * grid
        setters[name] = {"id": setter["id"], "source": setter["source"], "s": value}
        floors[name] = max(
            (v for v, s in candidates if s["source"].startswith("hf-bf16")), default=None
        )
    bounds["gap.sum"] = max(bounds["gap.sum"], bounds["gap.max"])
    safety = check({i: s["measurement"] for i, s in enumerate(eligible)}, bounds, calibrating=True)
    return {
        "bounds": bounds,
        "setters": setters,
        "hf_floor": floors,
        "unique_samples": len({s["id"] for s in eligible}),
        "stop_safety": safety["failures"],
    }


SELF_CHECKS = dict.fromkeys(
    "dtype ieee disabled_modules indices use_cache_false ignore_eos not_held".split(), True
) | {"first_logprob_error": 0.0}


def synthetic_measure(z=None, e=None, *, aligned=False, **kwargs):
    z = torch.eye(8, 32) * 8 if z is None else z
    e = z.clone() if e is None else e
    ids = kwargs.pop("tokens", e.argmax(-1))
    selected = kwargs.pop("selected", e.log_softmax(-1).gather(1, ids[:, None]).flatten())
    options = {"prompt_tokens": 1, "max_tokens": len(z), "self_checks": SELF_CHECKS} | kwargs
    reference = z if aligned else torch.cat([z, z[-1:]])
    return measure(reference, e, ids, selected, **options)


__all__ = [
    "BOUND_GRIDS",
    "QUALIFICATION_BOUNDS",
    "SELF_CHECKS",
    "check",
    "derive_bounds",
    "measure",
    "synthetic_measure",
]
