"""State kinds and prefix reuse stay with their owners."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from typing import ClassVar

import pytest

from tests.support.architecture import (
    PLAN_PREFIX_KIND,
    Hit,
    Imports,
    Module,
    Names,
    Package,
    References,
    Rule,
    attribute_spellings,
    calls,
    dotted_spellings,
    isinstance_targets,
    module_spellings,
    under,
)

_SPECULATIVE = ("inference/_execution/_speculative/",)
_STATE_FLAG = re.compile(r"recurrent_\w*|_?blocks")


def _kvcache_classes(package: Package) -> frozenset[str]:
    return frozenset(
        node.name
        for path, module in package.modules.items()
        if path.startswith("kvcache/")
        for _, node in module.nodes
        if isinstance(node, ast.ClassDef)
    )


def _state_flags(package: Package, module: Module) -> Iterator[Hit]:
    yield from Names(r"^has_recurrent_layers$")(package, module)
    yield from PLAN_PREFIX_KIND(package, module)
    for scope, node in module.nodes:
        if isinstance(node, ast.Compare):
            operands = (node.left, *node.comparators)
            for op, left, right in zip(node.ops, operands, operands[1:], strict=False):
                for value, other in ((left, right), (right, left)):
                    none = isinstance(other, ast.Constant) and other.value is None
                    flag = _STATE_FLAG.fullmatch(ident := module.ident(value))
                    if isinstance(op, ast.Is | ast.IsNot) and none and flag:
                        yield scope, f"{ident}_is_None"


def _kvcache_isinstance(package: Package, module: Module) -> Iterator[Hit]:
    for scope, chain in isinstance_targets(module):
        name = chain.rpartition(".")[2]
        if under(chain, ("cti.kvcache",)) or name in _kvcache_classes(package):
            yield scope, name


def _kvcache_submodule_imports(package: Package, module: Module) -> Iterator[Hit]:
    for scope, chain in Imports("cti.kvcache")(package, module):
        part = chain.split(".")[2:3]
        if part and f"kvcache/{part[0]}.py" in package.modules:
            yield scope, chain


def _kernel_granularity(package: Package, module: Module) -> Iterator[Hit]:
    for scope, call in calls(module, ("lcm",)):
        if any(isinstance(arg, ast.Constant) and arg.value == 64 for arg in call.args):
            yield scope, "lcm_64"


class StateOwners:
    """The S rules, with their probes and ledgers."""

    # S1 concrete pool and cache types stay in kvcache, attention, recurrent and the GDN
    # layer; speculative modules belong to D4.
    S1: ClassVar[Rule] = Rule(
        "S1",
        Names(r"^(PageGroup|RecurrentStatePool)$"),
        probes=(("plan.py", dotted_spellings("cti.kvcache.PageGroup")),),
        scope=("forward.py", "inference/", "plan.py", "resolution.py"),
        owners=_SPECULATIVE,
    )
    # S2 state-axis flags: has_recurrent_layers, the Plan's prefix_cache kind, recurrent_* /
    # blocks is None.
    S2: ClassVar[Rule] = Rule(
        "S2",
        _state_flags,
        probes=(
            (
                "inference/_control/_engine.py",
                (
                    *attribute_spellings("recurrent_state", "{} is None"),
                    "_other is self.recurrent_state is not None",
                ),
            ),
        ),
        scope=(
            "forward.py",
            "inference/",
            "models/",
            "plan.py",
            "resolution.py",
            "worker/",
        ),
        ledger="""
inference/_execution/runner.py
    prefix_cache
plan.py
    prefix_cache
resolution.py
    prefix_cache 3
""",
    )
    # S3 no isinstance on kvcache classes outside kvcache.
    S3: ClassVar[Rule] = Rule(
        "S3",
        _kvcache_isinstance,
        probes=(
            (
                "inference/_control/_cache.py",
                dotted_spellings(
                    "cti.kvcache.radix_cache.StateCoordinator", "isinstance(_cache, {})"
                ),
            ),
        ),
        owners=("kvcache/",),
    )
    # S4 inference reaches kvcache through its package API only.
    S4: ClassVar[Rule] = Rule(
        "S4",
        _kvcache_submodule_imports,
        probes=(("inference/_control/_engine.py", module_spellings("cti.kvcache.radix_cache")),),
        scope=("inference/",),
    )
    # S5 prefix owners read no wall clock: any use of the time module.
    S5: ClassVar[Rule] = Rule(
        "S5",
        References("time"),
        probes=(("inference/_control/_prefix_state.py", module_spellings("time")),),
        scope=(
            "kvcache/radix_cache.py",
            "kvcache/state.py",
            "inference/_control/_cache.py",
            "inference/_control/_prefix_state.py",
        ),
        ledger="",
    )
    # S6 inference spells no kernel granularity such as lcm(..., 64).
    S6: ClassVar[Rule] = Rule(
        "S6",
        _kernel_granularity,
        probes=(
            (
                "inference/_control/_cache.py",
                (*dotted_spellings("math.lcm", "{}(_rows, 64)"),),
            ),
        ),
        scope=("inference/",),
        ledger="",
    )

    @classmethod
    def rules(cls) -> tuple[Rule, ...]:
        return (cls.S1, cls.S2, cls.S3, cls.S4, cls.S5, cls.S6)


@pytest.mark.parametrize("rule", StateOwners.rules(), ids=lambda rule: rule.name)
def test_positions_match_the_ledger(rule: Rule, package: Package) -> None:
    rule.check(package)


@pytest.mark.parametrize("rule", StateOwners.rules(), ids=lambda rule: rule.name)
def test_rule_catches_every_spelling(rule: Rule, package: Package) -> None:
    rule.check_probes(package)


def test_s7_plan_row_fields_are_frozen(package: Package) -> None:
    """A new state field on a plan row is a design review, not an edit here."""

    module = package.modules["inference/_step.py"]
    fields = {
        node.target.id
        for scope, node in module.nodes
        if scope == "Row" and isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert fields == {
        "request_index",
        "query_kind",
        "kv_prefix_len",
        "sequence_len",
        "write_position",
        "sampling_position",
        "restore_row",
        "captures",
    }
