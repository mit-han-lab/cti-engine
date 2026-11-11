"""Parallelism stays behind distributed and the placing layers.

``test_parallel_owners.py`` keeps its line ceilings; these rules add AST positions.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from types import MappingProxyType
from typing import ClassVar

import pytest

from tests.support.architecture import (
    Hit,
    Imports,
    Module,
    Names,
    Package,
    Rule,
    attribute_spellings,
    calls,
    dotted_spellings,
    module_spellings,
    within,
)
from tests.support.imports import MODULE_SCOPE

_GROUP_TYPES = ("ParallelGroup", "Topology", "CollectiveGroup")
_LOCKSTEP_FILES = (
    "inference/_execution/runner.py",
    "inference/_control/_engine.py",
    "training/_optimizer.py",
    "worker/__init__.py",
)
# Packages that know nothing of ranks.
_AGNOSTIC = (
    "forward.py",
    "inference/_step.py",
    "inference/_execution/_graphs.py",
    "inference/_execution/_sampling.py",
    "inference/_execution/_speculative/",
    "inference/_control/_cache.py",
    "inference/_control/_prefix_state.py",
    "inference/_control/_scheduling/",
    "protocol/",
    "kvcache/",
    "llm/",
    "models/dflash/",
    "moe/",
    "multimodal/",
    "recurrent.py",
    "training/_objective.py",
)
_FORWARD = re.compile(r"forward\w*|project|argmax|_enter|_exit")
_RANK_SIZES = frozenset({"tp_size", "_tp_size", "num_ranks"})
_CONSTRUCTED_IN = MappingProxyType(
    {"Placement": "layers/placement.py", "WeightShard": "layers/", "ParallelGroup": "worker/"}
)


def _lockstep(package: Package, module: Module) -> Iterator[Hit]:
    lockstep = r"^(broadcast_cpu|broadcast_device|min_max_int|require_same|barrier|is_primary)$"
    yield from Names(lockstep)(package, module)
    if module.path in _LOCKSTEP_FILES:
        yield from Names(r"^(tp|ep)$")(package, module)


def _rank_branches(package: Package, module: Module) -> Iterator[Hit]:
    compares = (
        node
        for scope, node in module.nodes
        if isinstance(node, ast.Compare) and _FORWARD.fullmatch(scope.rpartition(".")[2])
    )
    for token in within(module, compares):
        group_size = token.kind == "attribute" and id(token.node) not in module.callees
        if token.ident in _RANK_SIZES or (token.ident == "size" and group_size):
            yield token.qualname, token.ident


def _one_owner(package: Package, module: Module) -> Iterator[Hit]:
    path = module.path
    for scope, call in calls(module, _CONSTRUCTED_IN):
        if not path.startswith(_CONSTRUCTED_IN[ident := module.ident(call.func)]):
            yield scope, ident
    for scope, node in module.nodes:
        definition = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        if definition and node.name == "place" and not path.startswith("layers/"):
            yield scope, "def_place"
    if not path.startswith("distributed/"):
        global_state = [
            part
            for scope, node in module.nodes
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign | ast.NamedExpr)
            for part in ([node] if scope == MODULE_SCOPE else module.global_values(node))
        ]
        for token in within(module, global_state):
            if token.ident in {"ContextVar", *_GROUP_TYPES}:
                yield token.qualname, token.ident
    if not path.startswith(("layers/", "inference/_execution/runner.py")):
        for token in module.tokens:
            if token.kind in {"attribute", "import"} and token.ident == "place":
                yield token.qualname, ".place"


class ParallelSeams:
    """The P rules, with their probes and ledgers."""

    # P1 who imports distributed; the worker constructs the groups.
    P1: ClassVar[Rule] = Rule(
        "P1",
        Imports("cti.distributed"),
        probes=(("inference/_control/_cache.py", module_spellings("cti.distributed")),),
        owners=("distributed/", "worker/"),
        ledger="""
inference/_execution/runner.py
    cti.distributed.ParallelGroup
layers/embedding.py
    cti.distributed.CollectiveGroup
layers/linear.py
    cti.distributed.CollectiveGroup
layers/site.py
    cti.distributed.CollectiveGroup
""",
    )
    # P2 collectives: distributed, the exits of placing layers, and training's gradient layout
    # over the ranks the worker injects.
    P2: ClassVar[Rule] = Rule(
        "P2",
        Names(r"^(all_reduce|all_gather|reduce_scatter)$"),
        probes=(
            ("inference/_execution/runner.py", attribute_spellings("all_reduce")),
            ("training/_optimizer.py", attribute_spellings("all_gather")),
        ),
        owners=("distributed/", "training/_parameter_layout.py"),
        ledger="""
layers/embedding.py
    all_gather
    all_reduce
layers/linear.py
    all_reduce
""",
    )
    # P3 lockstep calls, and lockstep files reading .tp or .ep.
    P3: ClassVar[Rule] = Rule(
        "P3",
        _lockstep,
        probes=(
            ("inference/_control/_cache.py", attribute_spellings("min_max_int")),
            ("worker/__init__.py", attribute_spellings("require_same")),
            ("inference/_control/_engine.py", attribute_spellings("tp")),
            ("training/_optimizer.py", attribute_spellings("ep")),
        ),
        owners=("distributed/",),
        ledger="""
inference/_execution/runner.py
    broadcast_device 2
    is_primary 2
runtime/_core.py
    require_same
worker/__init__.py
    barrier
    min_max_int
""",
    )
    # P4 parallel-agnostic packages name no axis, rank or group type.
    P4: ClassVar[Rule] = Rule(
        "P4",
        Names(
            r"(?i)(^|_)(tp|ep|sp)(_|$)|(tensor|expert|sequence)_?parallel"
            r"|^(rank|num_ranks|world_size|ParallelGroup|Topology|CollectiveGroup)$"
        ),
        probes=(
            ("inference/_control/_scheduling/request.py", attribute_spellings("num_ranks")),
            ("kvcache/radix_cache.py", attribute_spellings("tp_group")),
        ),
        scope=_AGNOSTIC,
        ledger="",
    )
    # P5 forward paths do not branch on the rank count.
    P5: ClassVar[Rule] = Rule(
        "P5",
        _rank_branches,
        probes=(
            (
                "layers/norm.py",
                attribute_spellings("tp_size", "def forward(self):\n    return {} == 1")
                + tuple(
                    f"def _exit(self, x):\n    return {size} > 1" for size in ("self.group.size",)
                ),
            ),
        ),
        ledger="",
    )
    # P6 one owner each: Placement, WeightShard, ParallelGroup, place(), module-level groups.
    P6: ClassVar[Rule] = Rule(
        "P6",
        _one_owner,
        probes=(
            (
                "inference/_execution/_speculative/dflash.py",
                (
                    "_object.place(_context)",
                    "_place = _object.place",
                    "from _module import place",
                    "def place(self, context):\n    pass",
                    "async def place(self, context):\n    pass",
                    "_GROUP: ParallelGroup | None = None",
                    "if _GROUP := ParallelGroup:\n    pass",
                    "def _init():\n    global _GROUP\n    _GROUP = ParallelGroup",
                    "def _init():\n    global _GROUP\n    [_GROUP := ParallelGroup for _ in ()]",
                    "def _init():\n    global _GROUP\n    _x, *_GROUP = 1, ParallelGroup",
                    "_GROUPS = []\n_GROUPS += [ParallelGroup]",
                    "def _init():\n    global _GROUPS\n    _GROUPS += [ParallelGroup]",
                )
                + dotted_spellings("contextvars.ContextVar", '_CONTEXT = {}("context")'),
            ),
            (
                "models/qwen3/model.py",
                dotted_spellings("cti.layers.placement.WeightShard", "{}(0, 0, 1)"),
            ),
        ),
        ledger="",
    )
    # P7 tp_group in layers and the worker; models count in test_parallel_owners.
    P7: ClassVar[Rule] = Rule(
        "P7",
        Names(r"^tp_group$"),
        probes=(("layers/norm.py", attribute_spellings("tp_group")),),
        scope=("layers/", "worker/"),
        ledger="",
    )

    @classmethod
    def rules(cls) -> tuple[Rule, ...]:
        return (cls.P1, cls.P2, cls.P3, cls.P4, cls.P5, cls.P6, cls.P7)


@pytest.mark.parametrize("rule", ParallelSeams.rules(), ids=lambda rule: rule.name)
def test_positions_match_the_ledger(rule: Rule, package: Package) -> None:
    rule.check(package)


@pytest.mark.parametrize("rule", ParallelSeams.rules(), ids=lambda rule: rule.name)
def test_rule_catches_every_spelling(rule: Rule, package: Package) -> None:
    rule.check_probes(package)


@pytest.mark.parametrize(
    "statement",
    [
        "ready, local = True, ContextVar('local')",
        "ready, *local = True, ContextVar('local')",
        "(ready, *_), local = (True, 1), ContextVar('local')",
        "local += [ContextVar('local')]",
    ],
)
def test_one_owner_scans_only_what_a_function_stores_globally(
    statement: str, package: Package
) -> None:
    source = f"def f():\n    global ready\n    {statement}\n"
    assert not list(_one_owner(package, package.parse("worker/__init__.py", source)))
