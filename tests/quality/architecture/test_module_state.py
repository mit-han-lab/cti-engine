"""Modules keep no state of their own (MS).

OOP first: a new module-level mutable table, registry
or singleton fails; the stock is ledgered and only shrinks as its owners move it into
classes. MS never takes an addition. It matches code written the normal way.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from tests.support.architecture import GuardBook, Module, ModuleState, Package, Rule
from tests.support.paths import REPO_ROOT


class ModuleStateGuards:
    """MS over the engine, benchmark, under tests the support, GPU and quality trees, and every
    conftest.py; ordinary test files' data is not scanned, the guards' own files are. Engine
    paths are relative to cti, the others to the repository."""

    TOOLS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("benchmark", "benchmark"),
        ("tests", "tests"),
    )
    TEST_TREES: ClassVar[tuple[str, ...]] = ("tests/support/", "tests/gpu/", "tests/quality/")
    GUARDS: ClassVar[str] = "tests/quality/architecture/"
    ENGINE: ClassVar[str] = "python/cti/"

    MS: ClassVar[Rule] = Rule(
        "MS",
        ModuleState(),
        probes=(
            (
                "tests/support/paths.py",
                (
                    "_TABLE = {}",
                    "_ROWS: list[int] = []",
                    "class _Registry:\n    pass\n_REGISTRY = _Registry()",
                    "class _Owner:\n    ROWS = []",
                    "import functools\n@functools.cache\ndef _load():\n    return 1",
                    "def _set():\n    global _STATE\n    _STATE = 1",
                    "def _add(x):\n    REPO_ROOT.append(x)",
                ),
            ),
            (
                "protocol/_request.py",
                (
                    "from cti.kvcache import SUPPORTED_CACHE_MANAGER\n"
                    'SUPPORTED_CACHE_MANAGER["x"] = 1',
                    'import cti.kvcache as kv\nkv.SUPPORTED_CACHE_MANAGER["x"] = 1',
                ),
            ),
        ),
        ledger="""
inference/__init__.py
    mutable:_LAZY_ATTRS
models/__init__.py
    mutable:_LAZY_ATTRS
multimodal/__init__.py
    mutable:_LAZY_ATTRS
server/openai/requests.py
    mutable:_UNSUPPORTED 3
tests/gpu/conftest.py
    mutable:FAST_TIER
    mutable:FULL_TIER
    mutable:TENSOR_PARALLEL_TIER
tests/gpu/inference/correctness/contract.py
    mutable:_CLAIM_SCOPES
    mutable:_SOURCE_FILES
tests/gpu/training/lora_backward_hf/profile.py
    mutable:_EXPECTED_PARAMETER_SCHEMA
tests/gpu/training/lora_backward_hf/runner.py
    mutable:_CASE_KEYS
    mutable:_EDGE_PROJECTION_NAMES
    mutable:_EDGE_TARGETS
    mutable:_LORA_KEYS
tests/quality/architecture/test_model_imports.py
    mutable:_MODELS
tests/quality/architecture/test_parallel_owners.py
    mutable:_CEILINGS
    mutable:_KV_HEAD_RULE_USERS
tests/quality/architecture/test_provider_imports.py
    mutable:ALLOWED_OUTSIDE_KERNEL
tests/support/dflash.py
    mutable:TINY_DRAFT_CONFIG
    mutable:TINY_TARGET_CONFIG
tests/support/dflash_prefix_gate.py
    mutable:_GRAPH_SIZES
    mutable:_TOPICS
tests/support/gpu.py
    mutable:_JIT_CACHE_DIRECTORIES
tests/support/hf_truth.py
    mutable:ARCHITECTURES
    mutable:IEEE_FP32_PRECISION
tests/support/imports.py
    mutable:_COMPOUND
tests/support/inference/qwen3_5_serving.py
    mutable:CPU
tests/support/inference_correctness.py
    mutable:WORKSPACE_PYTHON_ROOT
tests/support/lora_backward.py
    mutable:_SNAPSHOT_KEYS
tests/support/qwen3_5.py
    mutable:INTERLEAVED_QWEN3_5_CONFIG
    mutable:QWEN3_5_9B_CONFIG
    mutable:QWEN3_8_27B_CONFIG
    mutable:TINY_QWEN3_5_CONFIG
    mutable:TINY_QWEN3_5_VL_CONFIG
    mutable:_HYBRID_LAYER_PATTERN
tests/support/qwen3_5_engine.py
    mutable:STEPS
    mutable:TP_TOLERANCE
tests/support/qwen3_5_hf_parity_judge.py
    mutable:BOUND_GRIDS
    mutable:QUALIFICATION_BOUNDS
    mutable:SELF_CHECKS
tests/support/qwen3_5_lora.py
    mutable:TARGET_MODULES
    mutable:_BY_A
    mutable:_BY_B
    mutable:_PLACEMENTS
tests/support/qwen3_5_prompts.py
    mutable:IMAGE_RECIPES
    mutable:TEXTS
    mutable:_FROZEN
tests/support/qwen3_5_reference_cache.py
    mutable:ROW_FORMATS
tests/support/switching.py
    mutable:QWEN35_SWITCH_LATENCY_BOUNDS_MS
    mutable:QWEN3_SWITCH_LATENCY_BOUNDS_MS
tests/support/tiny_configs.py
    mutable:TINY_GEOMETRY
tests/support/tp_parity.py
    mutable:MAX_TIED_PICK_LOGPROB
tests/support/tp_shards.py
    mutable:CPU
training/__init__.py
    mutable:_LAZY_ATTRS
utils/logger.py
    global
    mutable:_COLORS
""",
    )

    @classmethod
    def package(cls, engine: Package, repository: Path = REPO_ROOT) -> Package:
        """The engine and the tool trees, as far as MS scans them, with every conftest.py of
        the repository."""

        kept = dict(engine.modules)
        for folder, root in cls.TOOLS:  # only the files MS scans are parsed
            for file in sorted((repository / folder).rglob("*.py")):
                path = file.relative_to(repository).as_posix()
                if cls._scanned(path):
                    source = file.read_text(encoding="utf-8")
                    kept[path] = Module(path, source, root, f"{folder}/")
        for file in sorted(repository.rglob("conftest.py")):
            path = file.relative_to(repository).as_posix()
            folder = path.rpartition("/")[0]
            hidden = any(part.startswith(".") for part in folder.split("/"))
            if path not in kept and not hidden and not path.startswith(cls.ENGINE):
                source = file.read_text(encoding="utf-8")
                root, base = folder.replace("/", "."), f"{folder}/" if folder else ""
                kept[path] = Module(path, source, root, base)
        return Package(kept, engine.root)

    @classmethod
    def _scanned(cls, path: str) -> bool:
        name = path.rpartition("/")[2]
        if not path.startswith("tests/") or name == "conftest.py" or path.startswith(cls.GUARDS):
            return True
        return path.startswith(cls.TEST_TREES) and not name.startswith("test_")


@pytest.fixture(scope="module")
def trees(package: Package) -> Package:
    return ModuleStateGuards.package(package)


def test_positions_match_the_ledger(trees: Package) -> None:
    ModuleStateGuards.MS.check(trees)


def test_rule_catches_its_probes(trees: Package) -> None:
    ModuleStateGuards.MS.check_probes(trees)


def test_module_state_takes_no_addition() -> None:
    assert not any(row.rule == "MS" for row in GuardBook().additions)


@pytest.mark.parametrize(
    "source",
    [
        "_LIMIT = 4\n_NAMES = ('a', 'b')\n_IDS = frozenset({1, 2})",
        "import types\n_KINDS = types.MappingProxyType({'a': 1})",
        "import logging\nlogger = logging.getLogger(__name__)",
        "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass _V:\n    x: int\n"
        "_ZERO = _V(0)",
        "__all__ = ['a', 'b']",
        "if __name__ == '__main__':\n    parser = []\n    parser.append(1)",
        "class _C:\n    def __init__(self):\n        self.items = []\n    def add(self, x):\n"
        "        self.items.append(x)",
        "def _f():\n    items = []\n    items.append(1)\n    items[0] = 2",
        "_LIMIT = 4\ndef _f(_LIMIT):\n    _LIMIT.update(k=1)",
    ],
)
def test_constants_and_values_keep_no_state(source: str, trees: Package) -> None:
    module = trees.parse("tests/support/paths.py", source)
    assert not list(ModuleState()(trees, module))


def test_every_conftest_of_the_repository_is_scanned(tmp_path: Path) -> None:
    for folder in ("", "tools/", "tests/gpu/"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        (tmp_path / folder / "conftest.py").write_text("TABLE = {}\n")
    scanned = ModuleStateGuards.package(Package.of({}), tmp_path)
    assert {"conftest.py", "tools/conftest.py", "tests/gpu/conftest.py"} <= scanned.modules.keys()
    found = ModuleStateGuards.MS.positions(scanned)
    assert sum(count for (path, *_), count in found.items() if path.endswith("conftest.py")) == 3
