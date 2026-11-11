"""Narrow repairs for malformed locked third-party wheel metadata."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import platform
from pathlib import Path

from .contract import VerificationError


def repair_cusparselt_arm64_tag(prefix: Path, purelib: Path, version: str) -> None:
    """Repair the locked NVIDIA cuSPARSELt wheel's invalid internal ARM64 tag."""

    if platform.machine().lower() not in ("aarch64", "arm64"):
        return

    prefix = prefix.resolve()
    purelib = purelib.resolve()
    if not purelib.is_relative_to(prefix):
        raise VerificationError(
            f"target Python purelib is outside the Conda prefix: prefix={prefix}, purelib={purelib}"
        )

    wheel_file = purelib / f"nvidia_cusparselt_cu13-{version}.dist-info" / "WHEEL"
    if not wheel_file.is_file():
        raise VerificationError(
            f"target Python purelib has no cuSPARSELt {version} WHEEL metadata: {wheel_file}"
        )
    dist_info = wheel_file.parent
    record_file = dist_info / "RECORD"
    if not record_file.is_file():
        raise VerificationError(f"cuSPARSELt installation has no RECORD: {record_file}")

    contents = wheel_file.read_text(encoding="utf-8")
    invalid = "Tag: py3-none-manylinux2014_sbsa"
    corrected = "Tag: py3-none-manylinux2014_aarch64"
    if contents.count(invalid) != 1 or corrected in contents:
        raise VerificationError(f"unexpected cuSPARSELt ARM64 wheel metadata in {wheel_file}")
    repaired = contents.replace(invalid, corrected)

    wheel_record_path = wheel_file.relative_to(dist_info.parent).as_posix()
    rows = list(csv.reader(record_file.read_text(encoding="utf-8").splitlines()))
    matching_rows = [row for row in rows if row and row[0] == wheel_record_path]
    if len(matching_rows) != 1 or len(matching_rows[0]) != 3:
        raise VerificationError(f"cuSPARSELt RECORD has no unique WHEEL entry: {record_file}")

    old_payload = contents.encode()
    wheel_row = matching_rows[0]
    if tuple(wheel_row[1:]) != _record_values(old_payload):
        raise VerificationError(
            f"cuSPARSELt WHEEL does not match its installed RECORD: {wheel_file}"
        )
    wheel_row[1:] = _record_values(repaired.encode())

    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer, lineterminator="\n").writerows(rows)
    wheel_file.write_text(repaired, encoding="utf-8")
    record_file.write_text(record_buffer.getvalue(), encoding="utf-8")


def _record_values(payload: bytes) -> tuple[str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"sha256={digest.decode()}", str(len(payload))
