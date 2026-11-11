"""Installed engine/kernel pair contract using independently released artifacts."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import subprocess
import sys
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns

from tests.support.paths import REPO_ROOT


def _install_explicit_kernel_wheel(tmp_path: Path, environment: dict[str, str]) -> Path | None:
    configured = os.environ.get("CTI_KERNEL_WHEEL")
    if configured is None:
        return None
    wheel = Path(configured).expanduser().resolve()
    assert wheel.is_file() and wheel.suffix == ".whl", (
        f"CTI_KERNEL_WHEEL must name an existing wheel: {wheel}"
    )
    target = tmp_path / "kernel-installed"
    subprocess.run(
        (
            sys.executable,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-compile",
            "--target",
            str(target),
            str(wheel),
        ),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return target.resolve()


def test_engine_wheel_interoperates_with_external_kernel_distribution(tmp_path: Path) -> None:
    """Build only cti; the kernel must come from an external installation."""
    source = tmp_path / "source"
    copytree(
        REPO_ROOT / "python" / "cti",
        source / "python" / "cti",
        ignore=ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES"):
        copy2(REPO_ROOT / name, source / name)

    artifacts = tmp_path / "artifacts"
    subprocess.run(
        (
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(artifacts),
        ),
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    engine_wheel = next(artifacts.glob("cti-*.whl"))

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")

    kernel_root = _install_explicit_kernel_wheel(tmp_path, environment)
    if kernel_root is None:
        kernel_distribution = metadata.distribution("cti-kernels")
        kernel_root = Path(kernel_distribution.locate_file("")).resolve()
        assert kernel_root != REPO_ROOT

    engine_target = (tmp_path / "engine-installed").resolve()
    subprocess.run(
        (
            sys.executable,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-compile",
            "--target",
            str(engine_target),
            str(engine_wheel),
        ),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    code = """
import importlib
import importlib.metadata as metadata
import importlib.util
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

engine_root = Path(sys.argv[1]).resolve()
kernel_root = Path(sys.argv[2]).resolve()
assert engine_root != kernel_root
sys.path[:0] = [str(engine_root), str(kernel_root)]

engine_distribution = metadata.distribution("cti")
kernel_distribution = metadata.distribution("cti-kernels")
assert Path(engine_distribution.locate_file("")).resolve() == engine_root
assert Path(kernel_distribution.locate_file("")).resolve() == kernel_root
kernel_version = kernel_distribution.version
kernel_requirements = [
    requirement
    for requirement in map(Requirement, engine_distribution.requires or [])
    if canonicalize_name(requirement.name) == "cti-kernels"
]
assert len(kernel_requirements) == 1, kernel_requirements
requirement = kernel_requirements[0]
assert requirement.url is None and requirement.marker is None and requirement.extras == {"cute", "gdn", "varlen"}
pins = tuple(requirement.specifier)
assert len(pins) == 1 and pins[0].operator == "==" and not pins[0].version.endswith(".*")
assert Version(pins[0].version) == Version(kernel_version), (requirement, kernel_version)

providers = {
    "flashinfer", "sgl_kernel", "flash_attn", "liger_kernel", "triton",
    "quack", "tvm_ffi", "torch_memory_saver",
}
for name in (
    "cti", "cti.inference", "cti.training",
    "cti.models", "cti.llm",
):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(engine_root), module.__file__
kernel = importlib.import_module("ctikernels")
assert Path(kernel.__file__).resolve().is_relative_to(kernel_root), kernel.__file__
assert not Path(kernel.__file__).resolve().is_relative_to(engine_root)
assert importlib.util.find_spec("cti.kernel") is None

import torch
import torch.nn.functional as F
import cti.layers.linear as engine_linear
from cti.config import LoRAConfig
from cti.layers import ExpertParallel, LayerServices, Site


class SingleRank:
    rank, size = 0, 1

    def all_reduce(self, tensor):
        return tensor

    def all_gather(self, tensor):
        return tensor


torch.manual_seed(23)
site = Site(SingleRank(), ExpertParallel(), torch.device("cpu"), services=LayerServices())
layer = engine_linear.LinearReplicated(4, 3, has_bias=True, site=site)
layer.weight = torch.randn(3, 4, dtype=torch.float64)
layer.bias = torch.randn(3, dtype=torch.float64)
scale = 0.75
lora = layer.attach_lora(LoRAConfig(targets=("attention",), rank=2, alpha=2 * scale))
lora.materialize(device=torch.device("cpu"), dtype=torch.float64, generator=torch.Generator())
a, b = lora.factors
with torch.no_grad():
    a.copy_(torch.randn(2, 4, dtype=torch.float64))
    b.copy_(torch.randn(3, 2, dtype=torch.float64))
x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
inputs = (x, a, b, layer.weight, layer.bias)
snapshots = tuple(value.detach().clone() for value in inputs)
x_ref, a_ref, b_ref = (value.detach().clone().requires_grad_() for value in (x, a, b))
expected = F.linear(x_ref, layer.weight, layer.bias) + scale * F.linear(F.linear(x_ref, a_ref), b_ref)
actual = layer.forward_training(x)
torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
upstream = torch.randn_like(actual)
actual.backward(upstream)
expected.backward(upstream)
for observed, reference in ((x, x_ref), (a, a_ref), (b, b_ref)):
    assert observed.grad is not None and reference.grad is not None
    torch.testing.assert_close(observed.grad, reference.grad, rtol=1e-12, atol=1e-12)
assert layer.weight.grad is None and layer.bias.grad is None
with torch.no_grad():
    inference = layer.forward(x)
torch.testing.assert_close(inference, expected.detach(), rtol=1e-12, atol=1e-12)
for value, snapshot in zip(inputs, snapshots, strict=True):
    torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
for name, module in tuple(sys.modules.items()):
    if name == "cti" or name.startswith("cti."):
        assert Path(module.__file__).resolve().is_relative_to(engine_root), (name, module.__file__)
    if name == "ctikernels" or name.startswith("ctikernels."):
        assert Path(module.__file__).resolve().is_relative_to(kernel_root), (name, module.__file__)
loaded_providers = providers.intersection(name.split(".")[0] for name in sys.modules)
assert not loaded_providers, loaded_providers
assert not torch.cuda.is_initialized()
"""
    completed = subprocess.run(
        (sys.executable, "-I", "-c", code, str(engine_target), str(kernel_root)),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
