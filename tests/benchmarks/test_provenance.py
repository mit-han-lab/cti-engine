from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark._provenance import (
    canonical_json_sha256,
    installed_package_provenance,
    read_receipt,
    seal_receipt,
    validate_receipt_identity,
    write_receipt,
)
from tests.support.paths import REPO_ROOT


@pytest.fixture
def installed_distributions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    paths = {}
    for name, version in (("cti", "4.3.2"), ("cti-kernels", "7.6.5")):
        path = tmp_path / f"{name.replace('-', '_')}-{version}.dist-info"
        path.mkdir()
        (path / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
        paths[name] = path

    def distribution(name: str) -> importlib.metadata.Distribution:
        if name not in paths:
            raise importlib.metadata.PackageNotFoundError(name)
        return importlib.metadata.PathDistribution(paths[name])

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    return paths


def test_index_installs_have_versions_without_requiring_direct_url(installed_distributions) -> None:
    assert installed_package_provenance() == {
        "cti": {"version": "4.3.2"},
        "cti-kernels": {"version": "7.6.5"},
    }


@pytest.mark.parametrize(
    "url",
    (
        "file:///private/candidates/kernel.whl",
        "https://user:secret@example.test/private/kernel.whl?token=secret",
    ),
)
@pytest.mark.parametrize("legacy_hash", (False, True))
def test_candidate_wheel_identity_uses_archive_hash_without_private_url(
    installed_distributions, url: str, legacy_hash: bool
) -> None:
    digest = hashlib.sha256(b"unpublished kernel wheel").hexdigest()
    archive_info = {"hash": f"sha256={digest}"} if legacy_hash else {"hashes": {"sha256": digest}}
    (installed_distributions["cti-kernels"] / "direct_url.json").write_text(
        json.dumps({"url": url, "archive_info": archive_info}), encoding="utf-8"
    )

    assert installed_package_provenance()["cti-kernels"] == {
        "version": "7.6.5",
        "origin": {"archive_info": {"hashes": {"sha256": digest}}},
    }


def test_vcs_identity_records_resolved_commit_not_mutable_requested_revision(
    installed_distributions,
) -> None:
    commit = "a" * 40
    (installed_distributions["cti-kernels"] / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "https://user:secret@example.test/private/kernel.git",
                "vcs_info": {"vcs": "git", "requested_revision": "candidate", "commit_id": commit},
            }
        ),
        encoding="utf-8",
    )

    assert installed_package_provenance()["cti-kernels"] == {
        "version": "7.6.5",
        "origin": {"vcs_info": {"vcs": "git", "commit_id": commit}},
    }


@pytest.mark.parametrize("editable", (False, True))
def test_directory_install_records_mode_without_claiming_immutable_identity(
    installed_distributions, editable: bool
) -> None:
    (installed_distributions["cti-kernels"] / "direct_url.json").write_text(
        json.dumps({"url": "file:///private/kernel", "dir_info": {"editable": editable}}),
        encoding="utf-8",
    )

    assert installed_package_provenance()["cti-kernels"] == {
        "version": "7.6.5",
        "origin": {"dir_info": {"editable": editable}},
    }


@pytest.mark.parametrize("name", ("cti", "cti-kernels"))
def test_missing_package_cannot_produce_a_valid_identity(
    installed_distributions, name: str
) -> None:
    del installed_distributions[name]

    with pytest.raises(importlib.metadata.PackageNotFoundError):
        installed_package_provenance()


def test_unreadable_origin_cannot_silently_fall_back_to_version(installed_distributions) -> None:
    (installed_distributions["cti-kernels"] / "direct_url.json").write_text(
        "not JSON", encoding="utf-8"
    )

    with pytest.raises(json.JSONDecodeError):
        installed_package_provenance()


@pytest.mark.parametrize("changed_identity", ("version", "wheel_hash", "vcs_commit"))
def test_kernel_change_alone_changes_sealed_receipt_identity(
    installed_distributions, tmp_path: Path, changed_identity: str
) -> None:
    receipts = []
    kernel_path = installed_distributions["cti-kernels"]
    for candidate in range(2):
        if changed_identity == "version":
            (kernel_path / "METADATA").write_text(
                f"Name: cti-kernels\nVersion: 7.6.{candidate}\n", encoding="utf-8"
            )
        elif changed_identity == "wheel_hash":
            (kernel_path / "direct_url.json").write_text(
                json.dumps(
                    {
                        "url": "file:///private/kernel.whl",
                        "archive_info": {
                            "hashes": {"sha256": hashlib.sha256(bytes([candidate])).hexdigest()}
                        },
                    }
                ),
                encoding="utf-8",
            )
        else:
            (kernel_path / "direct_url.json").write_text(
                json.dumps(
                    {
                        "url": "https://example.test/kernel.git",
                        "vcs_info": {"vcs": "git", "commit_id": str(candidate) * 40},
                    }
                ),
                encoding="utf-8",
            )
        path = tmp_path / f"receipt-{candidate}.json"
        write_receipt(
            path,
            {
                "source": {"revision": "f" * 40, "dirty": False},
                "runtime": {"packages": installed_package_provenance()},
                "samples": [1.25],
            },
        )
        receipts.append(read_receipt(path))

    first, second = receipts
    assert first["source"] == second["source"]
    assert first["samples"] == second["samples"]
    assert first["runtime"]["packages"]["cti"] == second["runtime"]["packages"]["cti"]
    assert (
        first["runtime"]["packages"]["cti-kernels"] != second["runtime"]["packages"]["cti-kernels"]
    )
    assert validate_receipt_identity(first) != validate_receipt_identity(second)


def test_package_provenance_does_not_import_runtime_packages(installed_distributions) -> None:
    metadata_root = next(iter(installed_distributions.values())).parent
    # Installed/editable packages must not mask malformed test metadata.
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import json, sys; sys.path[:0] = sys.argv[1:]; "
            "from benchmark._provenance import installed_package_provenance; "
            "print(json.dumps(installed_package_provenance())); "
            "assert not {'torch', 'cti', 'ctikernels', 'flashinfer', 'liger_kernel', 'triton'} "
            ".intersection(name.split('.')[0] for name in sys.modules)",
            str(metadata_root),
            str(REPO_ROOT),
        ),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == {
        "cti": {"version": "4.3.2"},
        "cti-kernels": {"version": "7.6.5"},
    }


def test_receipt_identity_is_canonical_and_does_not_mutate_payload() -> None:
    payload = {"samples": [1, 2], "provider": "cti"}
    original = copy.deepcopy(payload)

    sealed = seal_receipt(payload)

    assert payload == original
    assert validate_receipt_identity(sealed) == canonical_json_sha256(payload)
    assert seal_receipt({"provider": "cti", "samples": [1, 2]}) == sealed


def test_receipt_identity_rejects_payload_changes() -> None:
    sealed = seal_receipt({"samples": [1, 2]})
    sealed["samples"] = [1, 3]

    with pytest.raises(RuntimeError, match="does not match"):
        validate_receipt_identity(sealed)


def test_receipt_round_trip_is_fail_closed(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    write_receipt(path, {"kind": "operation_profile", "latency_ms": 1.25})

    receipt = read_receipt(path)

    assert receipt["kind"] == "operation_profile"
    assert validate_receipt_identity(receipt) == receipt["receipt_identity"]["payload_sha256"]

    path.write_text('{"kind": "operation_profile"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="omitted"):
        read_receipt(path)


def test_receipt_rejects_non_finite_json_values() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        seal_receipt({"latency_ms": float("nan")})
