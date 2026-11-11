from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.support.paths import PYTHON_ROOT


@pytest.mark.parametrize(
    ("layer_module", "inference_leaf", "training_leaf"),
    (
        (
            "cti.layers.activation",
            "ctikernels.activation.inference",
            "ctikernels.activation.training",
        ),
        (
            "cti.layers.norm",
            "ctikernels.norm.inference",
            "ctikernels.norm.training",
        ),
    ),
)
def test_inference_layer_import_does_not_load_training_kernel(
    layer_module: str,
    inference_leaf: str,
    training_leaf: str,
) -> None:
    source_root = PYTHON_ROOT
    code = """
import importlib
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
layer = importlib.import_module(sys.argv[2])
assert Path(layer.__file__).resolve().is_relative_to(source_root)
assert sys.argv[3] in sys.modules
kernel = importlib.import_module("ctikernels")
assert not Path(kernel.__file__).resolve().is_relative_to(source_root)

for prefix in (sys.argv[4], "liger_kernel"):
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
    )
"""
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(PYTHON_ROOT)

    subprocess.run(
        (
            sys.executable,
            "-c",
            code,
            str(source_root),
            layer_module,
            inference_leaf,
            training_leaf,
        ),
        check=True,
        env=environment,
    )


def test_rotary_layer_import_does_not_load_training_kernel() -> None:
    source_root = PYTHON_ROOT
    code = """
import importlib
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
layer = importlib.import_module("cti.layers.rotary")
assert Path(layer.__file__).resolve().is_relative_to(source_root)
kernel = importlib.import_module("ctikernels")
assert not Path(kernel.__file__).resolve().is_relative_to(source_root)
for prefix in ("ctikernels.rotary.training", "flash_attn"):
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
    )
"""
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(PYTHON_ROOT)

    subprocess.run(
        (sys.executable, "-c", code, str(source_root)),
        check=True,
        env=environment,
    )


@pytest.mark.parametrize(
    "module",
    (
        "cti.moe.fused",
        "cti.inference._execution._sampling",
        "cti.inference._execution._breakable_graph",
    ),
)
def test_engine_kernel_consumer_does_not_parse_the_runtime_environment(module: str) -> None:
    """Importing an engine consumer of the kernel package must not evaluate
    CTI_* variables: an unrelated invalid setting must not break an import."""
    code = """
import importlib
import sys
from pathlib import Path

importlib.import_module(sys.argv[1])
assert "cti.env" not in sys.modules, "kernel import evaluated the runtime environment"
kernel = importlib.import_module("ctikernels")
assert not Path(kernel.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
"""
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(PYTHON_ROOT)

    subprocess.run(
        (sys.executable, "-c", code, module, str(PYTHON_ROOT)), check=True, env=environment
    )
