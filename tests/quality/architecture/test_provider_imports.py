"""No new direct native-provider imports outside ``ctikernels``.

The kernel package owns numerical providers (FlashInfer, SGLang kernels,
FlashAttention, Liger, Triton, QuACK, TVM-FFI). Engine residency intentionally
owns torch-memory-saver; it is not a numerical kernel provider. This gate permits
that one runtime-owned dependency and rejects every other direct provider import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.imports import imports_from_source, module_name
from tests.support.paths import PYTHON_ROOT

PROVIDERS = frozenset(
    {
        "flashinfer",
        "sgl_kernel",
        "flash_attn",
        "liger_kernel",
        "triton",
        "quack",
        "tvm_ffi",
        "torch_memory_saver",
    }
)

# Engine memory residency owns this dependency; the standalone kernel package
# neither owns engine memory switching nor depends on torch-memory-saver.
ALLOWED_OUTSIDE_KERNEL: dict[str, frozenset[str]] = {
    "cti.memory": frozenset({"torch_memory_saver"}),
}


def _provider_imports(path: Path) -> frozenset[str]:
    module = module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    imported = imports_from_source(path.read_text(encoding="utf-8"), package=package)
    return frozenset(name.partition(".")[0] for name in imported) & PROVIDERS


def _sources_outside_kernel() -> list[Path]:
    return sorted((PYTHON_ROOT / "cti").rglob("*.py"))


@pytest.mark.parametrize(
    "path",
    _sources_outside_kernel(),
    ids=module_name,
)
def test_no_unapproved_provider_imports_outside_kernel(path: Path) -> None:
    module = module_name(path)
    allowed = ALLOWED_OUTSIDE_KERNEL.get(module, frozenset())
    unexpected = _provider_imports(path) - allowed
    assert not unexpected, (
        f"{module} imports native providers {sorted(unexpected)} directly; "
        "route them through ctikernels"
    )


def test_allow_list_names_only_live_exceptions() -> None:
    """A listed exception that no longer imports its provider must be removed from the list."""
    by_module = {module_name(path): _provider_imports(path) for path in _sources_outside_kernel()}
    stale = {
        module: sorted(allowed - by_module.get(module, frozenset()))
        for module, allowed in ALLOWED_OUTSIDE_KERNEL.items()
        if allowed - by_module.get(module, frozenset())
    }
    assert not stale, f"allow-list entries no longer needed: {stale}"
