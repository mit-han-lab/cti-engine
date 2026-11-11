#!/usr/bin/env python3
"""Create or verify a complete cti source environment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__:
    from .environment.bootstrap import EnvironmentBootstrap
    from .environment.contract import load_device_qualification
    from .environment.verify import EnvironmentVerifier
else:
    from environment.bootstrap import EnvironmentBootstrap
    from environment.contract import load_device_qualification
    from environment.verify import EnvironmentVerifier


def _find_conda(explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    if conda := shutil.which("conda"):
        return conda

    sibling = Path(sys.executable).with_name("conda")
    if sibling.is_file():
        return str(sibling)
    raise RuntimeError(
        "conda was not found on PATH or beside the current Python executable; "
        "pass --conda /path/to/conda"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or verify a cti environment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create and verify a complete environment")
    create.add_argument("--conda", help="path to the conda executable")
    create.add_argument(
        "--name",
        help="Conda environment name (default: the versioned profile name)",
    )
    create.add_argument(
        "--qualification",
        help="also require a named GPU qualification from environments/gpu-qualifications.toml",
    )
    create.add_argument(
        "--kernel-wheel",
        type=Path,
        help="optional local wheel override for the exact cti-kernels version in pyproject.toml",
    )
    create.add_argument(
        "--flash-attn-wheel",
        type=Path,
        help=(
            "a prebuilt flash-attn wheel from environments/flash-attention-prebuilt.toml for "
            "this machine; omitted, flash-attn is built from the locked source"
        ),
    )
    create.add_argument(
        "--no-gpu",
        action="store_true",
        help=(
            "create without a GPU, as a container image build does: requires --flash-attn-wheel "
            "and --cuda-driver-dir, and leaves verification to `verify` on a GPU host"
        ),
    )
    create.add_argument(
        "--cuda-driver-dir",
        type=Path,
        help="with --no-gpu: the directory that will hold the NVIDIA driver's libcuda.so",
    )
    create.add_argument(
        "--cache-dir",
        type=Path,
        help="the directory for JIT-compiled kernels (default: inside the environment)",
    )
    verify = subparsers.add_parser("verify", help="verify the active formal environment")
    verify.add_argument(
        "--qualification",
        help="also require a named GPU qualification from environments/gpu-qualifications.toml",
    )
    verify.add_argument(
        "--kernel-wheel",
        type=Path,
        help="optionally require the installed cti-kernels wheel to match this file",
    )

    args = parser.parse_args()
    qualification = load_device_qualification(args.qualification)
    if args.command == "create":
        EnvironmentBootstrap(
            _find_conda(args.conda),
            qualification,
            environment=args.name,
            kernel_wheel=args.kernel_wheel,
            flash_attn_wheel=args.flash_attn_wheel,
            no_gpu=args.no_gpu,
            cuda_driver_dir=args.cuda_driver_dir,
            cache_dir=args.cache_dir,
        ).create()
        return

    receipt = EnvironmentVerifier(
        qualification=qualification, kernel_wheel=args.kernel_wheel
    ).verify()
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
