from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import isolated_runtime_env, require_cuda_device, run_isolated_process
from tests.support.lora_backward import compare_backward_snapshots
from tests.support.paths import REPO_ROOT as _REPO_ROOT

from .profile import (
    QUALIFIED_DEVICES,
    RECEIPT_PREFIX,
    build_case_payload,
    comparison_spec,
)

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

_RUNNER_MODULE = "tests.gpu.training.lora_backward_hf.runner"


def _resolve_model_snapshot() -> Path:
    snapshot = QWEN3_0_6B.snapshot().resolve()
    if snapshot.name != QWEN3_0_6B.revision:
        raise RuntimeError(f"resolved unexpected model snapshot: {snapshot}")
    return snapshot


def _runner_command(
    *,
    role: str,
    model_path: Path,
    case_path: Path,
    output_path: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        _RUNNER_MODULE,
        "--role",
        role,
        "--model-path",
        str(model_path),
        "--case",
        str(case_path),
        "--output",
        str(output_path),
    )


def test_qwen3_lora_backward_matches_hf_peft(tmp_path: Path) -> None:
    """The engine's error against HF PEFT FP64 stays within HF PEFT FP32's (profile.py)."""

    device = torch.cuda.get_device_name(0)
    if device not in QUALIFIED_DEVICES:
        pytest.skip(f"qualification requires one of {QUALIFIED_DEVICES}, found {device!r}")
    spec = comparison_spec(device)
    require_cuda_device(device, spec.expected_profile.capability)
    env, log_root = isolated_runtime_env(tmp_path, "lora-backward-hf")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    model_path = _resolve_model_snapshot()
    case_path = tmp_path / "case.pt"
    case_payload = build_case_payload()
    torch.save(case_payload, case_path)

    snapshot_paths = {role: tmp_path / f"{role}.pt" for role in ("reference", "hf", "candidate")}
    for role, output_path in snapshot_paths.items():
        run_isolated_process(
            _runner_command(
                role=role,
                model_path=model_path,
                case_path=case_path,
                output_path=output_path,
            ),
            log_root / f"{role}.log",
            env=env,
            cwd=_REPO_ROOT,
            timeout=600,
        )

    snapshots = {
        role: torch.load(path, map_location="cpu", weights_only=True)
        for role, path in snapshot_paths.items()
    }
    receipt = compare_backward_snapshots(**snapshots, spec=spec)
    expected_initial = case_payload["initial_lora"]
    for role, snapshot in snapshots.items():
        actual_initial = snapshot["initial_lora"]
        if tuple(actual_initial) != tuple(expected_initial):
            raise AssertionError(f"{role} initial LoRA order differs from the fixed fixture")
        for name, expected in expected_initial.items():
            # The reference holds the fixture cast exactly to FP64.
            if not torch.equal(actual_initial[name], expected.to(actual_initial[name].dtype)):
                raise AssertionError(f"{role} initial LoRA tensor differs from fixture: {name}")
    serialized_receipt = json.dumps(receipt, sort_keys=True)
    (log_root / "comparison-receipt.json").write_text(
        serialized_receipt + "\n",
        encoding="utf-8",
    )
    print(RECEIPT_PREFIX + serialized_receipt, flush=True)
