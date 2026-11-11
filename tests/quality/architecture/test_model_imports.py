"""Models are peers: a model imports no other model, and only the registry (``register.py``,
re-exported by ``cti.models``) names one; the DFlash drafts, only the speculative seam,
whose draft table names them. A model is one module or one package. Models take kernels from
layers and never import ctikernels; they reach attention only through its value leaf
``attention.requirement``."""

from __future__ import annotations

from pathlib import Path

import pytest

from cti.models.register import MODELS
from tests.support.architecture import Imports, Package, Rule, module_spellings
from tests.support.imports import imports_from_source, module_name
from tests.support.paths import PYTHON_ROOT

_DRAFTS = "cti.models.dflash"
# The module, or for a package model the package, that each registry entry loads, and the
# drafts.
_MODELS = sorted(
    {
        "cti.models." + module_path.lstrip(".").partition(".")[0]
        for module_path, _ in MODELS.values()
    }
    | {_DRAFTS}
)
_REGISTRY_MODULES = frozenset({"cti.models", "cti.models.register"})
_SPECULATIVE_SEAM = "cti.inference._execution._speculative"

# Runtime, annotation-only and literal dynamic imports all count.
_NO_KERNELS = Rule(
    "models-kernels",
    Imports("ctikernels"),
    probes=(
        (
            "models/qwen3_5/vision.py",
            (
                *module_spellings("ctikernels.attention.varlen"),
                "if TYPE_CHECKING:\n    from ctikernels.norm import inference",
                "def _f():\n    from ctikernels.norm import inference",
                'import importlib\nimportlib.import_module("ctikernels.norm")',
            ),
        ),
    ),
    scope=("models/",),
)


def _imports(path: Path) -> frozenset[str]:
    module = module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    return imports_from_source(path.read_text(encoding="utf-8"), package=package)


def _within(module: str, model: str) -> bool:
    return module == model or module.startswith(f"{model}.")


def test_model_match_requires_a_module_boundary() -> None:
    assert _within("cti.models.qwen3_5.config", "cti.models.qwen3_5")
    assert not _within("cti.models.qwen3_5.config", "cti.models.qwen3")
    assert not _within("cti.models.qwen3_moe.model", "cti.models.qwen3")


@pytest.mark.parametrize("model", _MODELS, ids=lambda model: model.rpartition(".")[2])
def test_only_the_model_itself_and_its_table_import_a_model(model: str) -> None:
    importers = []
    for path in sorted((PYTHON_ROOT / "cti").rglob("*.py")):
        module = module_name(path)
        table = (
            _within(module, _SPECULATIVE_SEAM) if model == _DRAFTS else module in _REGISTRY_MODULES
        )
        if _within(module, model) or table:
            continue
        importers.extend(
            (module, imported) for imported in sorted(_imports(path)) if _within(imported, model)
        )

    assert importers == []


def test_models_import_no_kernels(package: Package) -> None:
    _NO_KERNELS.check(package)


def test_the_kernel_rule_catches_every_spelling(package: Package) -> None:
    _NO_KERNELS.check_probes(package)


def test_the_attention_requirement_models_import_is_a_pure_value_leaf() -> None:
    # Models and layers (the MHA KV-head rule) may import attention at run time only through
    # this module, so it stays stdlib-only: no torch, no backend, no other
    # attention module.
    path = PYTHON_ROOT / "cti" / "attention" / "requirement.py"

    assert {imported.partition(".")[0] for imported in _imports(path)} <= {
        "__future__",
        "dataclasses",
    }
