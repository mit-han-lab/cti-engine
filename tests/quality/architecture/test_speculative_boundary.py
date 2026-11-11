"""Speculative decoding stays behind one seam."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
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
    class_patterns,
    dotted_spellings,
    isinstance_targets,
    module_spellings,
    under,
)


@dataclass(frozen=True)
class RoundTypeTests:
    """isinstance and match class patterns on round types (Block* and Round*)."""

    ROUND_TYPE: ClassVar[re.Pattern[str]] = re.compile(r"^(Block|Round)\w*$")

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        for scope, target in chain(isinstance_targets(module), class_patterns(module)):
            if self.ROUND_TYPE.match(name := target.rpartition(".")[2]):
                yield scope, name


@dataclass(frozen=True)
class SeamImports:
    """Outside the seam, the speculative modules each module may import: the runner, the step
    costs and resolution the package. Inside it, no runtime owner or pool, though the execution
    modules beside the runner are free."""

    SEAM: ClassVar[str] = "inference/_execution/_speculative/"
    MODULES: ClassVar[tuple[str, ...]] = ("cti.inference._execution._speculative",)
    ALLOWED: ClassVar[tuple[tuple[str, tuple[str, ...]], ...]] = (
        ("inference/_execution/runner.py", ("cti.inference._execution._speculative",)),
        # The package entry, which exports load_speculative to the composition root.
        ("inference/__init__.py", ("cti.inference._execution._speculative",)),
    )
    BELOW: ClassVar[tuple[str, ...]] = tuple(
        f"cti.{name}"
        for name in ("client control core distributed inference kvcache training worker").split()
    )
    # Execution modules a speculative module may import: all but the runner, and the step.
    BESIDE: ClassVar[tuple[str, ...]] = (
        *(
            f"cti.inference._execution.{name}"
            for name in "_breakable_graph _graphs _hosting _sampling _speculative".split()
        ),
        "cti.inference._step",
    )

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        if module.path.startswith(self.SEAM):
            for scope, chain in Imports(*self.BELOW)(package, module):
                if not under(chain, self.BESIDE):
                    yield scope, chain
            return
        allowed = next((modules for path, modules in self.ALLOWED if path == module.path), ())
        for scope, chain in Imports(*self.MODULES)(package, module):
            if package.owning_module(chain) not in allowed:
                yield scope, chain


@dataclass(frozen=True)
class RaisingHooks:
    """Hooks whose body ends in a raise, abstract methods aside."""

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        for scope, node in module.nodes:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                abstract = "abstractmethod" in {module.ident(item) for item in node.decorator_list}
                if not abstract and isinstance(node.body[-1], ast.Raise):
                    yield scope, "raise"


class SpeculativeGuards:
    """The D rules, with their probes and ledgers."""

    SPECULATIVE: ClassVar[tuple[str, ...]] = ("inference/_execution/_speculative/",)

    # D1 speculative names, the method's own included, stay in their owners.
    D1: ClassVar[Rule] = Rule(
        "D1",
        Names(r"(?i)dflash|draft|proposer|hidden_taps|HiddenStateTaps|mask_token"),
        probes=(("inference/_control/_cache.py", attribute_spellings("draft_rows")),),
        owners=(
            *SPECULATIVE,
            "models/dflash/",
            "models/config.py",
            # The worker's ModelHost builds and loads the draft; the runner reads it through
            # HostedModels.
            "worker/_model_host.py",
            "inference/_execution/_hosting.py",
        ),
        ledger="""
attention/fixed_query.py
    DraftAttention 4
inference/_control/_scheduling/rounds.py
    accepted_drafts 3
    num_accepted_drafts 4
""",
    )
    # D2 no isinstance or match on round types but in the runner, which dispatches the step
    # ops.
    D2: ClassVar[Rule] = Rule(
        "D2",
        RoundTypeTests(),
        probes=(
            (
                "worker/__init__.py",
                dotted_spellings("cti.inference._step.Round", "isinstance(_op, {})"),
            ),
            (
                "worker/__init__.py",
                dotted_spellings(
                    "cti.inference._step.Round", "match _op:\n    case {}():\n        pass"
                ),
            ),
        ),
        owners=("inference/_execution/runner.py",),
        ledger="",
    )
    # D3 outside the seam only the runner, the step costs and resolution import the
    # speculative package; speculative modules import no runtime owner or pool.
    D3: ClassVar[Rule] = Rule(
        "D3",
        SeamImports(),
        probes=(
            (
                "inference/_control/_engine.py",
                module_spellings("cti.inference._execution._speculative.dflash"),
            ),
            (
                "inference/_execution/_speculative/dflash.py",
                module_spellings("cti.inference._control._cache"),
            ),
        ),
        ledger="""
inference/_execution/_speculative/dflash.py
    cti.kvcache.InferenceState
    cti.kvcache.next_snapshot_position
""",
    )
    # D4 speculative modules name no KV address or pool.
    D4: ClassVar[Rule] = Rule(
        "D4",
        Names(
            r"cache_index_table|cache_write_indices|store_kv|kv_pages|PageGroup|kv_indices"
            r"|kv_indptr|guard_start|draft_kv_cache"
        ),
        probes=(
            ("inference/_execution/_speculative/dflash.py", attribute_spellings("store_kv_pair")),
        ),
        scope=SPECULATIVE,
    )
    # D5 base classes keep no hook that only raises.
    D5: ClassVar[Rule] = Rule(
        "D5",
        RaisingHooks(),
        probes=(
            (
                "models/base.py",
                (
                    "class _Probe:\n    def hook(self):\n        raise NotImplementedError",
                    'def _hook():\n    """Doc."""\n    raise RuntimeError("no hook")',
                ),
            ),
        ),
        scope=("models/base.py", "attention/base.py"),
    )
    # D6 round state lives in the round ledger only; each request's slot holds its share,
    # which only the ledger reads or writes.
    D6: ClassVar[Rule] = Rule(
        "D6",
        Names(r"^(round_credits|round_device_row|_discarded_rows|_undone_rows|_block_cursor)$"),
        probes=(
            ("inference/_control/_scheduling/request.py", attribute_spellings("_undone_rows")),
            ("inference/_control/_scheduling/request.py", attribute_spellings("round_credits")),
        ),
        owners=(
            "inference/_control/_scheduling/rounds.py",
            "inference/_control/_scheduling/table.py",
        ),
    )

    # D7 speculative method names, written or bound, in strings and comments, stay in their
    # owners. It never takes an addition.
    @classmethod
    def rules(cls) -> tuple[Rule, ...]:
        return (cls.D1, cls.D2, cls.D3, cls.D4, cls.D5, cls.D6)


@pytest.mark.parametrize("rule", SpeculativeGuards.rules(), ids=lambda rule: rule.name)
def test_positions_match_the_ledger(rule: Rule, package: Package) -> None:
    rule.check(package)


@pytest.mark.parametrize("rule", SpeculativeGuards.rules(), ids=lambda rule: rule.name)
def test_rule_catches_every_spelling(rule: Rule, package: Package) -> None:
    rule.check_probes(package)


def test_seam_imports_name_exact_modules() -> None:
    paths = (
        "inference/_execution/_speculative/__init__.py",
        "inference/_execution/_speculative/records/__init__.py",
    )
    stand_in = Package.of(dict.fromkeys(paths, ""))
    runner = stand_in.parse(
        "inference/_execution/runner.py",
        "from cti.inference._execution._speculative import Runtime\n"
        "from cti.inference._execution._speculative.records import Record\n",
    )
    assert [chain for _, chain in SeamImports()(stand_in, runner)] == [
        "cti.inference._execution._speculative.records.Record"
    ]
