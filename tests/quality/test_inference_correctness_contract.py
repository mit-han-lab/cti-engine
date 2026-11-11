from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.gpu.inference.correctness import contract as contract_module
from tests.support.inference_correctness import (
    BOUNDARY_PROFILE,
    CORRECTNESS_PROFILES,
    HF_RUNNER,
    LONG_PROFILE,
    build_prompt,
    get_profile,
    planned_workload,
)


def test_correctness_profiles_have_resolved_runner_invariants() -> None:
    assert tuple(profile.name for profile in CORRECTNESS_PROFILES) == (
        "boundary",
        "long",
    )
    for profile in CORRECTNESS_PROFILES:
        assert get_profile(profile.name) is profile
        runner_names = {HF_RUNNER, *(name for name, _mode in profile.runtime_modes)}
        for invariant in profile.execution_invariants:
            assert invariant.reference in runner_names
            assert invariant.candidate in runner_names


def test_long_profile_covers_prefill_and_decode_without_exceeding_context() -> None:
    cases = [case for batch in LONG_PROFILE.batches for case in batch]
    assert [(case.name, case.prompt_length, case.max_tokens) for case in cases] == [
        ("prefill-8192", 8192, 8),
        ("prefill-decode-16385", 16385, 1024),
        ("prefill-24577", 24577, 8),
    ]
    assert all(len(batch) == 1 for batch in LONG_PROFILE.batches)
    assert [case.prefix_token for case in cases] == [11580, 7772, 304]
    assert all(
        case.prompt_length + case.max_tokens <= LONG_PROFILE.max_sequence_length for case in cases
    )
    assert LONG_PROFILE.max_batch_tokens == 16384
    assert LONG_PROFILE.hf_attention_implementation == "sdpa"
    assert tuple(name for name, _mode in LONG_PROFILE.runtime_modes) == (
        "graph-overlap-radix",
        "eager-overlap-radix",
    )


def test_profile_plans_preserve_distinct_settings_and_exact_prompts() -> None:
    boundary_plan = planned_workload(BOUNDARY_PROFILE)
    long_plan = planned_workload(LONG_PROFILE)

    assert boundary_plan["profile"] == "boundary"
    assert long_plan["profile"] == "long"
    assert boundary_plan["edge_modes"]["graph-overlap-radix"]["max_sequence_length"] == 512
    assert boundary_plan["edge_modes"]["graph-serial-radix"] == {
        **boundary_plan["edge_modes"]["graph-overlap-radix"],
        "overlap_scheduling": False,
    }
    assert long_plan["edge_modes"]["graph-overlap-radix"]["max_sequence_length"] == 32768
    assert boundary_plan["hf_settings"]["attention_implementation"] == "eager"
    assert long_plan["hf_settings"]["attention_implementation"] == "sdpa"
    assert "logits_to_keep" not in boundary_plan["hf_settings"]
    assert long_plan["hf_settings"]["logits_to_keep"] == 1
    assert long_plan["edge_modes"]["graph-overlap-radix"] == {
        "attention_backend": "fi",
        "cache_type": "radix",
        "cuda_graph_bs": [1],
        "dtype": "bfloat16",
        "ignore_eos": True,
        "max_batch_tokens": 16384,
        "max_running_requests": 1,
        "max_sequence_length": 32768,
        "inference_memory_ratio": 0.9,
        "peak_memory_ratio": 0.9,
        "overlap_scheduling": True,
        "temperature": 0,
    }

    planned_cases = [case for batch in long_plan["batches"] for case in batch["cases"]]
    assert [case["name"] for case in planned_cases] == [
        "prefill-8192",
        "prefill-decode-16385",
        "prefill-24577",
    ]
    for profile_case, planned_case in zip(
        (case for batch in LONG_PROFILE.batches for case in batch),
        planned_cases,
        strict=True,
    ):
        assert planned_case["prompt_ids"] == build_prompt(profile_case)
        assert len(planned_case["prompt_ids"]) == profile_case.prompt_length


def test_source_snapshot_detects_production_changes_in_a_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    orchestration_file = repo_root / "tests" / "harness.py"
    production_file = repo_root / "python" / "model.py"
    orchestration_file.parent.mkdir(parents=True)
    production_file.parent.mkdir(parents=True)
    orchestration_file.write_text("stable\n", encoding="utf-8")
    production_file.write_text("committed\n", encoding="utf-8")
    subprocess.run(("git", "init", "--quiet", str(repo_root)), check=True)
    subprocess.run(
        ("git", "-C", str(repo_root), "add", "tests/harness.py", "python/model.py"),
        check=True,
    )
    production_file.write_text("first dirty state\n", encoding="utf-8")

    monkeypatch.setattr(contract_module, "REPO_ROOT", repo_root)
    monkeypatch.setattr(contract_module, "_SOURCE_FILES", (orchestration_file,))
    monkeypatch.setattr(contract_module, "git_source_state", lambda: ("fixed-head", True))
    before = contract_module._source_snapshot()

    production_file.write_text("second dirty state\n", encoding="utf-8")
    after = contract_module._source_snapshot()

    assert before["revision"] == after["revision"] == "fixed-head"
    assert before["dirty"] is after["dirty"] is True
    assert before["files"] == after["files"]
    assert before["worktree"] != after["worktree"]


def test_failure_receipt_preserves_replayable_contract(monkeypatch, tmp_path) -> None:
    source = {"revision": "test-revision", "dirty": True, "files": {}}
    environment = {"receipt": {"source_revision": "test-revision", "source_dirty": True}}

    monkeypatch.setattr(contract_module, "_source_snapshot", lambda: source)
    monkeypatch.setattr(contract_module, "run_environment_verifier", lambda: environment)

    def fail_component(*_args, **_kwargs):
        pytest.fail("synthetic child failure", pytrace=False)

    monkeypatch.setattr(contract_module, "run_isolated_process", fail_component)

    with pytest.raises(RuntimeError, match="synthetic child failure"):
        contract_module.run_correctness_profile(BOUNDARY_PROFILE, tmp_path)

    receipt = json.loads(
        (
            tmp_path / "logs" / "inference-correctness-boundary" / "boundary" / "receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["verdict"] == "failed"
    assert receipt["formal_eligible"] is False
    assert receipt["source"]["stable"] is True
    assert receipt["environment"] == environment
    assert receipt["workload"] == planned_workload(BOUNDARY_PROFILE)
    assert receipt["failure"] == {
        "stage": "component-execution",
        "runner": HF_RUNNER,
        "type": "Failed",
        "message": "synthetic child failure",
        "comparison": None,
        "case": None,
        "expected": None,
        "actual": None,
        "first_divergent_step": None,
    }
    assert "planned-workload-receipt.json" in receipt["artifacts"]


def test_semantic_failure_receipt_summarizes_the_first_divergence(monkeypatch, tmp_path) -> None:
    source = {"revision": "test-revision", "dirty": True, "files": {}}
    environment = {"receipt": {"source_revision": "test-revision", "source_dirty": True}}

    monkeypatch.setattr(contract_module, "_source_snapshot", lambda: source)
    monkeypatch.setattr(contract_module, "run_environment_verifier", lambda: environment)

    def fail_comparison(_profile, *, profile_root, **_kwargs):
        contract_module._write_json(
            profile_root / "hf-vs-target-receipt.json",
            {
                "cases": [
                    {
                        "name": "first-mismatch",
                        "result_match": False,
                        "reference": {"result": {"token_ids": [11, 22]}},
                        "candidate": {"result": {"token_ids": [11, 33]}},
                        "first_divergent_step": 1,
                    }
                ]
            },
        )
        raise AssertionError("synthetic semantic mismatch")

    monkeypatch.setattr(contract_module, "_execute_correctness_profile", fail_comparison)

    with pytest.raises(AssertionError, match="synthetic semantic mismatch"):
        contract_module.run_correctness_profile(BOUNDARY_PROFILE, tmp_path)

    failure = json.loads(
        (
            tmp_path / "logs" / "inference-correctness-boundary" / "boundary" / "receipt.json"
        ).read_text(encoding="utf-8")
    )["failure"]
    assert failure["case"] == "first-mismatch"
    assert failure["expected"] == [11, 22]
    assert failure["actual"] == [11, 33]
    assert failure["first_divergent_step"] == 1
    assert failure["comparison"]["path"].endswith("hf-vs-target-receipt.json")


def test_failure_receipt_capture_never_masks_the_primary_error(monkeypatch, tmp_path) -> None:
    source = {"revision": "test-revision", "dirty": True, "files": {}}
    calls = 0

    def flaky_source_snapshot():
        nonlocal calls
        calls += 1
        if calls == 1:
            return source
        raise OSError("snapshot capture failed")

    monkeypatch.setattr(contract_module, "_source_snapshot", flaky_source_snapshot)
    monkeypatch.setattr(
        contract_module,
        "run_environment_verifier",
        lambda: {"receipt": {"source_revision": "test-revision", "source_dirty": True}},
    )

    def fail_execution(*_args, **_kwargs):
        raise RuntimeError("primary execution failure")

    monkeypatch.setattr(contract_module, "_execute_correctness_profile", fail_execution)

    with pytest.raises(RuntimeError, match="primary execution failure") as raised:
        contract_module.run_correctness_profile(BOUNDARY_PROFILE, tmp_path)

    assert any("snapshot capture failed" in note for note in raised.value.__notes__)
