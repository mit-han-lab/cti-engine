"""The axis-guard scanner counts each spelling of a name once per occurrence."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from tests.support.architecture import (
    Addition,
    GuardBook,
    Imports,
    Ledger,
    Module,
    Names,
    Package,
    References,
    Rule,
    attribute_spellings,
    isinstance_targets,
)
from tests.support.imports import imports_from_source, runtime_imports_from_source

_SOURCE = '''
from typing import TYPE_CHECKING
import cti.distributed.runtime as rt
from cti.distributed.runtime import ParallelGroup as Group
from time import monotonic_ns
if TYPE_CHECKING:
    from cti.kvcache import PageGroup

def run(group):
    """all_reduce in a docstring does not count."""
    group.all_reduce(1)  # all_reduce in a comment does not count
    reduce = group.all_reduce
    "all_reduce"
    return rt.ParallelGroup, Group, monotonic_ns()

def shadow(monotonic_ns):
    import math as rt
    return monotonic_ns(), rt.ParallelGroup
'''


def _positions(hits) -> Counter:
    module = Module("engine/probe.py", _SOURCE)
    return Counter(hits(Package.of({}), module))


def test_names_resolve_aliases_and_method_references() -> None:
    assert _positions(Names(r"^all_reduce$")) == {("run", "all_reduce"): 2}
    assert _positions(Names(r"^ParallelGroup$")) == {
        ("<module>", "ParallelGroup"): 1,
        ("run", "ParallelGroup"): 2,
        ("shadow", "ParallelGroup"): 1,
    }
    assert _positions(References("time")) == {
        ("<module>", "time.monotonic_ns"): 1,
        ("run", "time.monotonic_ns"): 1,
    }
    assert _positions(References("math")) == {("shadow", "math"): 2}
    assert _positions(References("cti.distributed")) == {
        ("<module>", "cti.distributed.runtime"): 1,
        ("<module>", "cti.distributed.runtime.ParallelGroup"): 1,
        ("run", "cti.distributed.runtime"): 1,
        ("run", "cti.distributed.runtime.ParallelGroup"): 1,
    }
    assert _positions(Imports("cti.kvcache")) == {("TYPE_CHECKING", "cti.kvcache.PageGroup"): 1}


def _check(ledger: str, path: str, body: str, book: GuardBook | None = None) -> None:
    rule = Rule("P2", Names(r"^all_reduce$"), probes=(), ledger=ledger)
    rule.check(Package.of({path: f"class Row:\n{body}"}), book)


_FORWARD = "    def forward(self):\n        self.group.all_reduce(x)\n"


def test_ledger_is_a_multiset_of_positions() -> None:
    ledger = "layers/linear.py\n    all_reduce\n"
    _check(ledger, "layers/linear.py", _FORWARD)
    _check(ledger, "layers/linear.py", _FORWARD.replace("forward", "backward"))  # a rename

    repeated = _FORWARD + "        self.group.all_reduce(y)\n"
    for body in (repeated, "    pass\n"):
        with pytest.raises(AssertionError, match="leaks only shrink"):
            _check(ledger, "layers/linear.py", body)


def test_additions_land_in_their_own_ledgers() -> None:
    landed = "layers/moe.py\n    all_reduce\n"
    row = Addition("P2", "PA7", "layers/moe.py", "all_reduce", 1, "exit", landed)
    book = GuardBook(additions=(row,))
    _check("", "layers/moe.py", _FORWARD, book)

    with pytest.raises(AssertionError, match="leaks only shrink"):
        _check("", "layers/moe.py", _FORWARD * 2, book)
    for ledger in (landed.replace("moe", "norm"), landed.replace("all_reduce", "all_reduce 2")):
        book = GuardBook(additions=(replace(row, ledger=ledger),))
        with pytest.raises(AssertionError, match="exceeds its row"):
            _check("", "layers/moe.py", _FORWARD, book)


@pytest.mark.parametrize(
    ("path", "symbol", "granted"),
    [
        ("layers/placement.py", "cti.distributed.CollectiveGroup", True),
        ("layers/placement.py", "cti.distributed.CollectiveGroup.size", True),
        ("layers/placement.py", "cti.distributed.CollectiveGroups", False),
        ("layers/placement.py", "cti.distributed.runtime.ParallelGroup", False),
        ("layers/linear.py", "cti.distributed.CollectiveGroup", False),
    ],
)
def test_an_addition_grants_its_path_and_symbol_only(path: str, symbol: str, granted: bool) -> None:
    row = Addition("P1", "QL5", "layers/placement.py", "cti.distributed.CollectiveGroup", 1, "")
    assert row.grants((path, symbol)) is granted


@pytest.mark.parametrize("count", ["0", "-2"])
def test_ledger_counts_are_positive(count: str) -> None:
    with pytest.raises(AssertionError, match="positive"):
        Ledger.parse(f"layers/moe.py\n    all_reduce {count}\n")


def test_a_missing_path_fails(package: Package) -> None:
    with pytest.raises(AssertionError, match="no file under"):
        Rule("probe", Names("x"), probes=(), scope=("engine/missing.py",)).check(package)


@pytest.mark.parametrize(
    ("name", "defined"),
    [
        ("cti.kvcache.StateCapacity", "cti.kvcache.state.StateCapacity"),
        (
            "cti.training.TrainingCheckpoint",
            "cti.training._checkpoint.TrainingCheckpoint",
        ),
        ("cti.forward.ForwardInputs", "cti.forward.ForwardInputs"),
        ("cti.kvcache.SUPPORTED_CACHE_MANAGER", "cti.kvcache.SUPPORTED_CACHE_MANAGER"),
    ],
)
def test_facade_names_resolve_to_their_defining_module(
    name: str, defined: str, package: Package
) -> None:
    assert package.defining(name) == defined


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ('_TABLE = {"Worker": "cti.worker"}', "cti.worker"),
        (
            '_TABLE = {"Worker": ("._scheduling.request", "Request")}',
            "cti.inference._control._scheduling.request",
        ),
        (
            'class Catalog:\n    MODELS = MappingProxyType({"X": ("cti.worker", "X")})',
            "cti.worker",
        ),
        (
            'MODELS: Mapping = MappingProxyType({"X": ("cti.worker", "X")})',
            "cti.worker",
        ),
    ],
)
def test_import_tables_count_as_runtime_imports(source: str, target: str) -> None:
    module = Module("inference/_control/_cache.py", source)
    assert [(name, only) for _, name, only in Package.of({}).imported(module)] == [(target, False)]


def test_probes_report_a_missed_spelling(package: Package) -> None:
    def bare_names(package, module):
        for token in module.tokens:
            if token.kind == "name" and token.ident == "probed":
                yield token.qualname, token.ident

    rule = Rule(
        "probe", bare_names, probes=(("inference/_step.py", attribute_spellings("probed")),)
    )

    with pytest.raises(AssertionError, match="misses"):
        rule.check_probes(package)


@pytest.mark.parametrize(
    "call",
    [
        'importlib.import_module("cti.inference._execution.runner")',
        'from importlib import import_module as load\nload("cti.inference._execution." + "runner")',
        'importlib.import_module(name="cti.inference._execution.runner")',
        '__import__("cti.inference._execution.runner")',
        'importlib.import_module(".runner", package=__package__)',
        'importlib.import_module(".runner", "cti.inference._execution")',
    ],
)
def test_import_helpers_count_literal_dynamic_imports(call: str) -> None:
    package, module = "cti.inference._execution", "cti.inference._execution.runner"
    assert module in imports_from_source(call, package=package)
    assert module in dict(runtime_imports_from_source(call, package=package))


def test_import_helpers_skip_non_literal_and_foreign_imports() -> None:
    for call in ("importlib.import_module(name)", 'registry.import_module("cti")'):
        assert not imports_from_source(call, package="cti")


@pytest.mark.parametrize(
    ("guard", "skipped"),
    [
        ("from typing import TYPE_CHECKING as TC\nif TC:", True),
        ("import typing as t\nif t.TYPE_CHECKING:", True),
        (
            "from typing import TYPE_CHECKING\ndef f(TYPE_CHECKING=True):\n  if TYPE_CHECKING:",
            False,
        ),
        ("import typing as t\nt = Flags\nif t.TYPE_CHECKING:", False),
        ("t = typing\nif t.TYPE_CHECKING:", False),
        ("import typing\nt = typing\nif t.TYPE_CHECKING:", True),
        ("if Flags.TYPE_CHECKING:", False),
        ("if typing.TYPE_CHECKING:", False),
        ("def TYPE_CHECKING():", False),
        (
            "from typing import TYPE_CHECKING as TC\nmatch x:\n    case TC:\n        pass\nif TC:",
            False,
        ),
    ],
)
def test_only_typing_type_checking_skips_imports(guard: str, skipped: bool) -> None:
    found = runtime_imports_from_source(f"{guard}\n    import cti.worker\n", package="x")
    assert any(module == "cti.worker" for module, _ in found) is not skipped


def test_type_checking_definitions_and_unions_keep_their_scope() -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    def hook(cache):\n"
        "        return isinstance(cache, A | (B, C | D))\n"
    )
    found = list(isinstance_targets(Module("engine/probe.py", source)))
    assert found == [("TYPE_CHECKING.hook", name) for name in "ABCD"]


@pytest.mark.parametrize(
    ("source", "count"),
    [
        (
            "import torch as t\ndef f(x):\n    g = lambda t: t\n    [t for t in x]\n    return t.mm",
            1,
        ),
        ("import torch as t\nclass A:\n    t = None\n    def f(self):\n        return t.mm", 1),
        ("import torch as t\n@t.mm\ndef f(t=t.mm):\n    return t", 2),
        ("import torch\nmm = torch.mm\ndef f(a, b):\n    return mm(a, b)", 2),
    ],
)
def test_names_resolve_in_their_lexical_scope(source: str, count: int) -> None:
    tokens = Module("engine/probe.py", source).tokens
    assert sum(token.chain == "torch.mm" for token in tokens) == count


@pytest.mark.parametrize(
    "source",
    [
        "import torch as t\nt.mm\nimport math as t",
        "try:\n    import torch as t\nexcept ImportError:\n    t = None\nt.mm",
        "t = None\ndef load():\n    global t\n    import torch as t\ndef f():\n    return t.mm",
        "def f():\n    import torch as t\n    def g():\n        nonlocal t\n        t = 1\n    return t.mm",
        "import torch\ndef f():\n    t = torch\n    t.mm\n    t = None",
        "import torch as t\nclass C:\n    t.mm\n    t = None",
        "import torch as t\nclass C:\n    t = None\n    del t\n    t.mm",
        "import torch as t\nclass C:\n    t = None\n    try:\n        1 / 0\n    except E as t:\n        pass\n    t.mm",
        "import torch as t\nclass C:\n    if x:\n        t = None\n    t.mm",
    ],
)
def test_a_rebound_import_resolves_to_nothing(source: str) -> None:
    tokens = Module("engine/probe.py", source).tokens
    assert not any(token.chain == "torch.mm" for token in tokens)


def test_a_local_named_like_a_module_is_not_the_module() -> None:
    tokens = Module("engine/probe.py", "def f(torch):\n    return torch.mm").tokens
    assert [token.chain for token in tokens if token.kind == "attribute"] == ["?.torch.mm"]
