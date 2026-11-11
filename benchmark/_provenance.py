"""Tamper-evident identities shared by development benchmark receipts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path

from . import _runtime

_RECEIPT_IDENTITY_FIELD = "receipt_identity"
_RECEIPT_CANONICALIZATION = "json-sort-keys-compact-v1"
_SOURCE_SNAPSHOT_ALGORITHM = "git-visible-content-v1"


def installed_package_provenance() -> dict[str, dict[str, object]]:
    """Record installed package identities without importing either runtime package.

    Index installs normally have no direct URL metadata. For direct installs, retain
    artifact hashes, resolved VCS commits, or editable mode, but not private URLs or
    local paths. Editable mode alone is not an immutable source identity.
    """

    packages = {}
    for name in ("cti", "cti-kernels"):
        distribution = importlib.metadata.distribution(name)
        package: dict[str, object] = {"version": distribution.version}
        raw_origin = distribution.read_text("direct_url.json")
        if raw_origin is not None:
            direct_url = json.loads(raw_origin)
            origin = {}
            if "archive_info" in direct_url:
                archive_info = direct_url["archive_info"]
                hashes = dict(archive_info.get("hashes", {}))
                if "hash" in archive_info:
                    algorithm, digest = archive_info["hash"].split("=", 1)
                    hashes.setdefault(algorithm, digest)
                origin["archive_info"] = {"hashes": hashes}
            if "vcs_info" in direct_url:
                vcs_info = direct_url["vcs_info"]
                origin["vcs_info"] = {
                    "vcs": vcs_info["vcs"],
                    "commit_id": vcs_info["commit_id"],
                }
            if "dir_info" in direct_url:
                origin["dir_info"] = {"editable": direct_url["dir_info"].get("editable", False)}
            package["origin"] = origin
        packages[name] = package
    return packages


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_snapshot_sha256(repo_root: Path, paths: tuple[str, ...]) -> str:
    """Hash Git-visible source content, including unstaged and untracked files."""

    completed = subprocess.run(
        (
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *paths,
        ),
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    relative_paths = sorted(path for path in completed.stdout.split(b"\0") if path)
    digest = hashlib.sha256(_SOURCE_SNAPSHOT_ALGORITHM.encode())
    for encoded_path in relative_paths:
        relative_path = Path(os.fsdecode(encoded_path))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"git returned an invalid source path: {relative_path}")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        path = repo_root / relative_path
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            digest.update(b"missing")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"unsupported source file type: {relative_path}")
        digest.update((metadata.st_mode & 0o111).to_bytes(2, "big"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def seal_receipt(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a content-addressed receipt without mutating ``payload``."""

    if _RECEIPT_IDENTITY_FIELD in payload:
        raise ValueError("receipt payload is already sealed")
    return {
        **payload,
        _RECEIPT_IDENTITY_FIELD: {
            "algorithm": "sha256",
            "canonicalization": _RECEIPT_CANONICALIZATION,
            "payload_sha256": canonical_json_sha256(payload),
        },
    }


def validate_receipt_identity(payload: Mapping[str, object]) -> str:
    """Validate and return the canonical identity of a sealed receipt."""

    identity = payload.get(_RECEIPT_IDENTITY_FIELD)
    if not isinstance(identity, dict):
        raise RuntimeError("receipt omitted its content identity")
    if set(identity) != {"algorithm", "canonicalization", "payload_sha256"}:
        raise RuntimeError("receipt content identity fields drifted")
    digest = identity.get("payload_sha256")
    if (
        identity.get("algorithm") != "sha256"
        or identity.get("canonicalization") != _RECEIPT_CANONICALIZATION
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise RuntimeError("receipt content identity is invalid")
    unsigned = {name: value for name, value in payload.items() if name != _RECEIPT_IDENTITY_FIELD}
    if canonical_json_sha256(unsigned) != digest:
        raise RuntimeError("receipt content identity does not match its payload")
    return digest


def write_receipt(path: Path, payload: Mapping[str, object]) -> None:
    _runtime.write_json(path, seal_receipt(payload))


def read_receipt(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"receipt is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"receipt must contain a JSON object: {path}")
    validate_receipt_identity(payload)
    return payload
