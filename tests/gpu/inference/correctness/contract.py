from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from benchmark._runtime import (
    git_source_state,
    run_environment_verifier,
    sha256_file,
)
from tests.support import inference_correctness as spec_module
from tests.support.gpu import isolated_runtime_env, run_isolated_process
from tests.support.graph_captures import capture_line
from tests.support.inference_correctness import (
    HF_RUNNER,
    RECEIPT_PREFIX,
    REPO_ROOT,
    WORKSPACE_PYTHON_ROOT,
    CorrectnessProfile,
    ExecutionInvariant,
    RuntimeMode,
    expected_golden_results,
    planned_workload,
)

from . import runner as runner_module

_RUNNER_MODULE = "tests.gpu.inference.correctness.runner"
_CLAIM_SCOPES = {
    "hf-vs-target": "exact BF16 greedy token and text parity against Hugging Face",
    "eager-vs-graph": "exact token and text parity with CUDA graph as the only setting change",
    "serial-vs-overlap": "exact token and text parity with overlap as the only setting change",
}
_SOURCE_FILES = (
    Path(spec_module.__file__).resolve(),
    Path(runner_module.__file__).resolve(),
    Path(__file__).resolve(),
    REPO_ROOT / "tests/support/gpu.py",
    REPO_ROOT / "benchmark/_runtime.py",
)


class _ComponentFailure(RuntimeError):
    def __init__(self, runner: str, cause: BaseException) -> None:
        self.runner = runner
        self.cause = cause
        super().__init__(f"{runner} component failed: {cause}")


def _component_command(profile: CorrectnessProfile, runner: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        _RUNNER_MODULE,
        "--profile",
        profile.name,
        "--runner",
        runner,
    )


def _parse_receipt(output: str, profile: CorrectnessProfile, runner: str) -> dict:
    receipts = [
        json.loads(line.removeprefix(RECEIPT_PREFIX))
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    assert len(receipts) == 1, f"{runner}: expected one receipt, found {len(receipts)}"
    assert receipts[0]["profile"] == profile.name
    return receipts[0]


def _first_divergent_step(expected: list[int], actual: list[int]) -> int | None:
    for index, (expected_id, actual_id) in enumerate(zip(expected, actual, strict=False)):
        if expected_id != actual_id:
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def _setting_differences(reference: dict, candidate: dict) -> dict[str, dict]:
    differences = {}
    for name in sorted(reference.keys() | candidate.keys()):
        reference_present = name in reference
        candidate_present = name in candidate
        if reference_present == candidate_present and reference.get(name) == candidate.get(name):
            continue
        differences[name] = {
            "reference_present": reference_present,
            "reference": reference.get(name),
            "candidate_present": candidate_present,
            "candidate": candidate.get(name),
        }
    return differences


def _comparison_receipt(
    profile: CorrectnessProfile,
    invariant: ExecutionInvariant,
    reference: dict,
    candidate: dict,
    expected_names: list[str],
) -> dict:
    expected_counts = Counter(expected_names)
    reference_names = [case["name"] for case in reference["cases"]]
    candidate_names = [case["name"] for case in candidate["cases"]]
    reference_counts = Counter(reference_names)
    candidate_counts = Counter(candidate_names)
    reference_cases = {case["name"]: case for case in reference["cases"]}
    candidate_cases = {case["name"]: case for case in candidate["cases"]}
    expected_name_set = set(expected_names)
    metadata_fields = (
        "batch_index",
        "workload_batch_size",
        "prompt_ids",
        "max_tokens",
    )
    cases = []
    for name in expected_names:
        if name not in reference_cases or name not in candidate_cases:
            continue
        reference_case = reference_cases[name]
        candidate_case = candidate_cases[name]
        reference_ids = reference_case["result"]["token_ids"]
        candidate_ids = candidate_case["result"]["token_ids"]
        cases.append(
            {
                "name": name,
                "workload_match": all(
                    reference_case[field] == candidate_case[field] for field in metadata_fields
                ),
                "generate_call_batch_size_match": (
                    reference_case["generate_call_batch_size"]
                    == candidate_case["generate_call_batch_size"]
                ),
                "reference": reference_case,
                "candidate": candidate_case,
                "first_divergent_step": _first_divergent_step(reference_ids, candidate_ids),
                "result_match": reference_case["result"] == candidate_case["result"],
            }
        )

    return {
        "schema_version": 1,
        "profile": profile.name,
        "invariant": invariant.as_receipt(),
        "reference_model": reference["model"],
        "candidate_model": candidate["model"],
        "model_match": reference["model"] == candidate["model"],
        "reference_settings": reference["settings"],
        "candidate_settings": candidate["settings"],
        "reference_source_file": reference.get("source_file"),
        "candidate_source_file": candidate.get("source_file"),
        "source_match": reference.get("source_file") == candidate.get("source_file"),
        "expected_case_count": len(expected_names),
        "reference_case_count": len(reference_names),
        "candidate_case_count": len(candidate_names),
        "expected_duplicate_names": sorted(
            name for name, count in expected_counts.items() if count > 1
        ),
        "reference_duplicate_names": sorted(
            name for name, count in reference_counts.items() if count > 1
        ),
        "candidate_duplicate_names": sorted(
            name for name, count in candidate_counts.items() if count > 1
        ),
        "missing_from_reference": sorted(expected_name_set - reference_cases.keys()),
        "unexpected_in_reference": sorted(reference_cases.keys() - expected_name_set),
        "missing_from_candidate": sorted(expected_name_set - candidate_cases.keys()),
        "unexpected_in_candidate": sorted(candidate_cases.keys() - expected_name_set),
        "setting_differences": _setting_differences(reference["settings"], candidate["settings"]),
        "cases": cases,
    }


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def _worktree_fingerprint(repo_root: Path) -> dict[str, object]:
    completed = subprocess.run(
        ("git", "-C", str(repo_root), "ls-files", "-co", "--exclude-standard", "-z"),
        check=True,
        capture_output=True,
        text=True,
    )
    records = []
    for relative_path in sorted(filter(None, completed.stdout.split("\0"))):
        path = repo_root / relative_path
        if path.is_symlink():
            kind = "symlink"
            content = str(path.readlink())
        elif path.is_file():
            kind = "file"
            content = sha256_file(path)
        elif path.exists():
            kind = "directory"
            content = None
        else:
            kind = "missing"
            content = None
        mode = None if kind == "missing" else path.lstat().st_mode & 0o777
        records.append(
            {
                "path": relative_path,
                "kind": kind,
                "mode": mode,
                "content": content,
            }
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(records),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _source_snapshot() -> dict[str, object]:
    revision, dirty = git_source_state()
    return {
        "revision": revision,
        "dirty": dirty,
        "worktree": _worktree_fingerprint(REPO_ROOT),
        "files": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in _SOURCE_FILES},
    }


def _assert_source_stable(expected: dict[str, object]) -> None:
    actual = _source_snapshot()
    if actual != expected:
        raise RuntimeError(
            f"correctness source changed during execution: expected={expected}, actual={actual}"
        )


def _run_component(
    profile: CorrectnessProfile,
    runner: str,
    base_env: dict[str, str],
    profile_root: Path,
    mode: RuntimeMode | None = None,
) -> tuple[dict, str, Path, Path]:
    try:
        env = base_env.copy()
        if mode is not None:
            env["CTI_DISABLE_OVERLAP_SCHEDULING"] = "1" if mode.disable_overlap else "0"
        log_path = profile_root / f"{runner}.log"
        output = run_isolated_process(
            _component_command(profile, runner),
            log_path,
            env=env,
            cwd=REPO_ROOT,
        )
        receipt = _parse_receipt(output, profile, runner)
        receipt_path = profile_root / f"{runner}-receipt.json"
        _write_json(receipt_path, receipt)
        return receipt, output, receipt_path, log_path
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        raise _ComponentFailure(runner, error) from error


def _assert_comparison(
    profile: CorrectnessProfile,
    invariant: ExecutionInvariant,
    comparison: dict,
) -> None:
    assert comparison["profile"] == profile.name
    assert comparison["model_match"]
    assert comparison["reference_case_count"] == comparison["expected_case_count"]
    assert comparison["candidate_case_count"] == comparison["expected_case_count"]
    assert not comparison["expected_duplicate_names"]
    assert not comparison["reference_duplicate_names"]
    assert not comparison["candidate_duplicate_names"]
    assert not comparison["missing_from_reference"]
    assert not comparison["unexpected_in_reference"]
    assert not comparison["missing_from_candidate"]
    assert not comparison["unexpected_in_candidate"]
    if invariant.expected_setting_differences is not None:
        assert set(comparison["setting_differences"]) == set(invariant.expected_setting_differences)
        assert comparison["source_match"]
    for case in comparison["cases"]:
        assert case["workload_match"]
        if invariant.expected_setting_differences is not None:
            assert case["generate_call_batch_size_match"]
        assert case["first_divergent_step"] is None
        assert case["result_match"]
    if invariant.reference == HF_RUNNER:
        for case in comparison["cases"]:
            assert case["reference"]["min_top2_margin"] >= profile.minimum_hf_top2_margin


def _execute_correctness_profile(
    profile: CorrectnessProfile,
    *,
    base_env: dict[str, str],
    profile_root: Path,
    source: dict[str, object],
    environment: dict[str, object],
    plan: dict[str, object],
    plan_path: Path,
) -> Path:
    expected_names = [case["name"] for batch in plan["batches"] for case in batch["cases"]]

    execution_receipts = {}
    component_artifacts = {}
    hf_receipt, _hf_output, receipt_path, log_path = _run_component(
        profile, HF_RUNNER, base_env, profile_root
    )
    execution_receipts[HF_RUNNER] = hf_receipt
    component_artifacts[HF_RUNNER] = {
        "receipt": _artifact(receipt_path),
        "log": _artifact(log_path),
    }

    expected_golden = expected_golden_results()
    for mode_name, mode in profile.runtime_modes:
        receipt, output, receipt_path, log_path = _run_component(
            profile,
            mode_name,
            base_env,
            profile_root,
            mode,
        )
        execution_receipts[mode_name] = receipt
        component_artifacts[mode_name] = {
            "receipt": _artifact(receipt_path),
            "log": _artifact(log_path),
        }

        assert Path(receipt["source_file"]).is_relative_to(WORKSPACE_PYTHON_ROOT)
        assert receipt["mode"] == mode_name
        assert receipt["golden"]["batch_results"] == expected_golden
        assert receipt["golden"]["single_results"] == expected_golden
        assert "Selected attention backend fi" in output
        expected_graph_log = (
            capture_line("decode") if mode.cuda_graph_bs else "CUDA graph is disabled"
        )
        assert expected_graph_log in output

    claims = []
    for invariant in profile.execution_invariants:
        comparison = _comparison_receipt(
            profile,
            invariant,
            execution_receipts[invariant.reference],
            execution_receipts[invariant.candidate],
            expected_names,
        )
        comparison_path = profile_root / f"{invariant.name}-receipt.json"
        _write_json(comparison_path, comparison)
        _assert_comparison(profile, invariant, comparison)
        claims.append(
            {
                "name": invariant.name,
                "oracle": "exact-token-and-text",
                "scope": _CLAIM_SCOPES[invariant.name],
                "case_count": len(comparison["cases"]),
                "comparison": _artifact(comparison_path),
            }
        )

    _assert_source_stable(source)
    aggregate = {
        "schema_version": 1,
        "profile": profile.name,
        "verdict": "passed",
        "formal_eligible": not source["dirty"],
        "source": source,
        "environment": environment,
        "model": hf_receipt["model"],
        "workload": {
            **_artifact(plan_path),
            "case_count": len(expected_names),
        },
        "claims": claims,
        "components": component_artifacts,
    }
    aggregate_path = profile_root / "receipt.json"
    _write_json(aggregate_path, aggregate)
    return aggregate_path


def _existing_artifacts(profile_root: Path) -> dict[str, dict[str, str]]:
    return {
        path.name: _artifact(path)
        for path in sorted(profile_root.iterdir())
        if path.is_file() and path.name != "receipt.json"
    }


def _comparison_failure_details(
    profile: CorrectnessProfile, profile_root: Path
) -> dict[str, object]:
    details = {
        "comparison": None,
        "case": None,
        "expected": None,
        "actual": None,
        "first_divergent_step": None,
    }
    for invariant in reversed(profile.execution_invariants):
        comparison_path = profile_root / f"{invariant.name}-receipt.json"
        if not comparison_path.is_file():
            continue
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        details["comparison"] = _artifact(comparison_path)
        mismatch = next(
            (case for case in comparison.get("cases", ()) if not case.get("result_match", False)),
            None,
        )
        if mismatch is not None:
            details.update(
                {
                    "case": mismatch["name"],
                    "expected": mismatch["reference"]["result"]["token_ids"],
                    "actual": mismatch["candidate"]["result"]["token_ids"],
                    "first_divergent_step": mismatch["first_divergent_step"],
                }
            )
        return details
    return details


def run_correctness_profile(profile: CorrectnessProfile, tmp_path: Path) -> Path:
    base_env, log_root = isolated_runtime_env(tmp_path, f"inference-correctness-{profile.name}")
    profile_root = log_root / profile.name
    profile_root.mkdir(parents=True, exist_ok=True)
    source = _source_snapshot()
    plan = planned_workload(profile)
    plan_path = profile_root / "planned-workload-receipt.json"
    _write_json(plan_path, plan)
    environment = None
    stage = "environment-verification"
    try:
        environment = run_environment_verifier()
        verifier_receipt = environment["receipt"]
        if (
            verifier_receipt["source_revision"] != source["revision"]
            or verifier_receipt["source_dirty"] != source["dirty"]
        ):
            raise RuntimeError("environment receipt does not match the correctness source snapshot")
        stage = "profile-execution"
        return _execute_correctness_profile(
            profile,
            base_env=base_env,
            profile_root=profile_root,
            source=source,
            environment=environment,
            plan=plan,
            plan_path=plan_path,
        )
    except BaseException as error:
        try:
            if isinstance(error, _ComponentFailure):
                failure_error = error.cause
                failure_stage = "component-execution"
                failure_runner = error.runner
            else:
                failure_error = error
                failure_stage = stage
                failure_runner = None
            comparison_failure = _comparison_failure_details(profile, profile_root)
            final_source = _source_snapshot()
            failure = {
                "schema_version": 1,
                "profile": profile.name,
                "verdict": "failed",
                "formal_eligible": False,
                "source": {
                    "start": source,
                    "end": final_source,
                    "stable": final_source == source,
                },
                "environment": environment,
                "model": plan["model"],
                "workload": plan,
                "failure": {
                    "stage": failure_stage,
                    "runner": failure_runner,
                    "type": type(failure_error).__name__,
                    "message": str(failure_error),
                    **comparison_failure,
                },
                "artifacts": _existing_artifacts(profile_root),
            }
            _write_json(profile_root / "receipt.json", failure)
        except Exception as receipt_error:  # noqa: BLE001 - preserve the primary failure
            if hasattr(error, "add_note"):
                error.add_note(f"failed to capture correctness failure receipt: {receipt_error}")
        raise
