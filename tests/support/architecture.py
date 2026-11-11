"""AST scans for the axis seam guards.

A ``Rule``'s hits outside its owners form a multiset of ``(path, qualname, symbol)``
positions that must equal its ledger plus its ``GuardBook`` additions; leaks only shrink.
Rules scan a ``Package``, one source tree parsed once, which resolves names across its
modules. Names resolve as Python binds them (``tests.support.imports.Scopes``), every
``Attribute`` counts, and literal dynamic imports count as the module they name. Rules
match code written the normal way; deliberate workarounds are review's. Paths are
relative to the package root and a prefix covers every file it starts.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from importlib.util import resolve_name
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import pytest

from tests.support.imports import (
    MODULE_SCOPE,
    TYPE_CHECKING,
    Scopes,
    literal_string,
)
from tests.support.paths import PYTHON_ROOT

Hit = tuple[str, str]
Position = tuple[str, str]  # (path, symbol)


@dataclass(frozen=True)
class Addition:
    """Positions ``commit`` may add to ``rule`` under ``path`` and ``symbol``, a legal owner
    addition. Landed positions go in ``ledger``, at most ``count`` of them (None: no cap)."""

    rule: str
    commit: str
    path: str
    symbol: str
    count: int | None
    reason: str
    ledger: str = ""

    def grants(self, position: Position) -> bool:
        path, symbol = position
        return path.startswith(self.path) and (
            not self.symbol or symbol == self.symbol or symbol.startswith(f"{self.symbol}.")
        )


@dataclass(frozen=True)
class GuardBook:
    """What is registered beside the rules' ledgers: ``additions``, the only rows that let a
    rule's positions grow; a new row needs review. Code that moves updates the ledgers it moves
    between."""

    _DISTRIBUTED: ClassVar[str] = "cti.distributed"
    _AUTOGRAD: ClassVar[str] = "cti.distributed.autograd"

    additions: tuple[Addition, ...] = (
        # I1 edges that arrive with a commit.
        Addition("I1-edges", "PA9", "layers/linear.py", _AUTOGRAD, 2, "f, g"),
        Addition("I1-edges", "PA9", "layers/embedding.py", _AUTOGRAD, 3, "f, g"),
        Addition("I1-edges", "PA9", "training/_optimizer.py", _DISTRIBUTED, 1, "Context"),
        Addition("I1-edges", "PA9", "training/runner.py", _DISTRIBUTED, 1, "group type"),
        Addition("I1-edges", "QL9", "layers/moe.py", _AUTOGRAD, 1, "MoE-LoRA"),
        # GR1: every inference graph captures into memory's one pool.
        Addition(
            "I1-edges",
            "GR1",
            "layers/gated_delta_net/replay.py",
            "cti.memory.GraphPool",
            1,
            "the one graph pool",
            ledger="layers/gated_delta_net/replay.py\n    cti.memory.GraphPool\n",
        ),
        Addition(
            "I1-edges",
            "GR1",
            "recurrent.py",
            "cti.memory.GraphPool",
            1,
            "the one graph pool",
            ledger="recurrent.py\n    cti.memory.GraphPool\n",
        ),
        Addition(
            "P1",
            "PA6",
            "layers/moe.py",
            _DISTRIBUTED,
            1,
            "Topology for Experts.shard",
            ledger="layers/moe.py\n    cti.distributed.Topology",
        ),
        Addition("P1", "PA9", "training/_optimizer.py", _DISTRIBUTED, 1, "Context"),
        Addition("P1", "PA9", "training/runner.py", _DISTRIBUTED, 1, "group type"),
        Addition("P1", "PA9", "layers/linear.py", f"{_AUTOGRAD}.copy_to_parallel", 1, "f"),
        Addition("P1", "PA9", "layers/linear.py", f"{_AUTOGRAD}.reduce_from_parallel", 1, "g"),
        Addition("P1", "PA9", "layers/embedding.py", f"{_AUTOGRAD}.copy_to_parallel", 1, "f"),
        Addition("P1", "PA9", "layers/embedding.py", f"{_AUTOGRAD}.gather_from_parallel", 1, "g"),
        Addition("P1", "PA9", "layers/embedding.py", f"{_AUTOGRAD}.reduce_from_parallel", 1, "g"),
        Addition("P1", "QL9", "layers/moe.py", _AUTOGRAD, 1, "MoE-LoRA"),
        Addition("P2", "PA7", "layers/moe.py", "all_reduce", 1, "MoE exit all-reduce"),
        Addition("P2", "PA9", "layers/embedding.py", "", 1, "vocab-parallel loss"),
        Addition("P2", "PA9", "training/", "", 3, "checkpoint gather, FP32 optimizer commit"),
        Addition("P2", "PA10", "layers/norm.py", "", 1, "fused norm exit"),
        Addition("P2", "QL9", "layers/moe.py", "", 2, "MoE-LoRA backward dots and dX"),
        Addition("P3", "PA9", "training/_optimizer.py", "is_primary", 1, "FP32 commit"),
        Addition("P3", "PA9", "training/_optimizer.py", "broadcast_device", 1, "FP32 commit"),
        Addition("P3", "PA9", "training/_optimizer.py", "min_max_int", 1, "FP32 commit"),
        Addition(
            "Q1",
            "T3-API",
            "moe/fused.py",
            "",
            None,
            "cti-kernels' MoE LoRA weights take the shared-A layout's four factors or a"
            " per-projection adapter's six (MoELoRAWeights.separate, served only); its"
            " routed training sizes itself (fused_experts_lora_training_bytes)",
            ledger="""
moe/fused.py
    MoELoRAWeights 3
    fused_experts_lora_training_bytes 2
    lora
""",
        ),
        Addition(
            "P4",
            "T3-API",
            "moe/",
            "",
            None,
            "LoRA rank, the kernel's keyword; not a parallel rank",
            ledger="""
moe/base.py
    rank
moe/fused.py
    rank 3
""",
        ),
        Addition(
            "Q1",
            "LO-1+STE",
            "moe/fused.py",
            "",
            None,
            "adapted expert stacks call the cti-kernels expert LoRA kernel from the MoE backend"
            " (FP8 first, then BF16, which deleted the torch provider, net shrink)",
            ledger="""
moe/fused.py
    ExpertLoRA 3
    MoELoRAWeights 2
    fused_experts_lora 2
    lora 13
""",
        ),
        Addition(
            "Q1",
            "QL10",
            "moe/",
            "",
            None,
            "MoE-LoRA provider and training route",
            ledger="""
moe/__init__.py
    ExpertLoRA
moe/base.py
    ExpertLoRA 3
    lora 2
moe/fused.py
    ExpertLoRA 2
    lora 1
""",
        ),
        Addition(
            "Q1",
            "9617f8dd3",
            "moe/base.py",
            "ExpertLoRA",
            1,
            "ExpertLoRA.training returns the same protocol over bound FP32 masters",
            ledger="moe/base.py\n    ExpertLoRA",
        ),
        # PA6 construction: the engine builds each layer's Site and LayerServices.
        Addition(
            "I3-private",
            "PA6",
            "layers/base.py",
            "_param",
            1,
            "the op-tree walk asks each op for its Param",
            ledger="layers/base.py\n    _param",
        ),
        # GroupRole names the draft's KV groups,
        # and resolution keys the method's requirements by it.
        Addition(
            "D1",
            "ST6b",
            "kvcache/state.py",
            "DRAFT",
            1,
            "GroupRole keys the draft's paged slots apart from the target's",
            ledger="kvcache/state.py\n    DRAFT",
        ),
        Addition(
            "D1",
            "ST6b",
            "resolution.py",
            "DRAFT",
            1,
            "the speculative method's requirements serve the draft's role",
            ledger="resolution.py\n    DRAFT",
        ),
        # Dual-host start: a remote rank's entry runs the rank
        # unit as _client does, and every node reads the shared key from the env owner.
        Addition(
            "I1-edges",
            "DH1",
            "runtime/_nodes.py",
            "",
            4,
            "a remote rank's entry and the nodes' shared key",
            ledger="""
runtime/_nodes.py
    cti.env.ENV
    cti.runtime._control.ControlChannel
    cti.runtime._control.Follower
    cti.runtime._core.run_engine_core
""",
        ),
        Addition(
            "I1-facades",
            "DH1",
            "runtime/_nodes.py",
            "cti.env.ENV",
            2,
            "the nodes' shared key",
            ledger="runtime/_nodes.py\n    cti.env.ENV 2",
        ),
        Addition(
            "I1-facades",
            "DH1",
            "runtime/__init__.py",
            "+NodeAgent",
            1,
            "a joining node's agent",
            ledger="runtime/__init__.py\n    +NodeAgent",
        ),
        Addition(
            "I1-facades",
            "DH1",
            "server/launch.py",
            "cti.runtime.NodeAgent",
            1,
            "the CLI runs a joining node's agent",
            ledger="server/launch.py\n    cti.runtime.NodeAgent",
        ),
        # G4: the services resolve the gate op unfused gated products share with the fused
        # gated Linear.
        Addition(
            "Q6",
            "G4",
            "layers/services.py",
            "ctikernels.linear.gated_input",
            1,
            "the unfused gate, bitwise the fused Linear's",
            ledger="layers/services.py\n    ctikernels.linear.gated_input",
        ),
        # An adapted product on a quantized or split base adds its delta by cti-kernels'
        # factor delta.
        Addition(
            "Q6",
            "AP9-3",
            "layers/lora.py",
            "ctikernels.linear.lora_delta_factors_",
            1,
            "the adapter delta after a quantized or split base's product",
            ledger="layers/lora.py\n    ctikernels.linear.lora_delta_factors_",
        ),
    )


class Ledger:
    """A rule's ledger text: a path line, then ``symbol [count]`` lines under it."""

    @staticmethod
    def parse(text: str) -> Counter[Position]:
        ledger: Counter[Position] = Counter()
        path = ""
        for line in text.splitlines():
            if not line.strip():
                continue
            if not line.startswith(" "):
                path = line.strip()
                continue
            symbol, *count = line.split()
            times = int(count[0]) if count else 1
            assert times > 0, f"ledger count must be positive: {line.strip()!r}"
            ledger[(path, symbol)] += times
        return ledger

    @staticmethod
    def render(positions: Counter[Position]) -> str:
        lines: list[str] = []
        for (path, symbol), count in sorted(positions.items()):
            if path not in lines:
                lines.append(path)
            lines.append(f"    {symbol}" + (f" {count}" if count > 1 else ""))
        return "\n".join(lines)


@dataclass(frozen=True)
class Token:
    """One name a module spells at ``node``; ``chain`` expands the binding at its root."""

    qualname: str
    kind: str  # "name", "attribute" or "import"
    chain: str
    node: ast.AST = field(compare=False, repr=False)
    imported: bool = False

    @property
    def ident(self) -> str:
        return self.chain.rpartition(".")[2]


class Module(Scopes):
    """One parsed source file with its scoped nodes and tokens; ``root`` names the package
    its path, past ``base``, is relative to."""

    MODULE_PATH: ClassVar[re.Pattern[str]] = re.compile(r"(\.+|cti\.)[A-Za-z_][\w.]*")

    def __init__(self, path: str, source: str, root: str = "cti", base: str = "") -> None:
        self.path = path
        self.source = source
        self.naming = (root, base)  # what parse() names a mutation of it by
        parts = path.removeprefix(base).removesuffix(".py").split("/")
        named = [root, *parts[: -1 if parts[-1] == "__init__" else None]]
        self.name = ".".join(part for part in named if part)
        self.tree = ast.parse(source)
        super().__init__(
            self.tree,
            self.name if path.endswith("__init__.py") else self.name.rpartition(".")[0],
        )
        # Each callee's id to the call that applies it.
        self.callees = {id(node.func): node for _, node in self.nodes if isinstance(node, ast.Call)}
        self.tokens = tuple(
            token for scope, node in self.nodes for token in self._tokens(scope, node)
        )

    def ident(self, node: ast.AST) -> str:
        return self.chain(node).rpartition(".")[2]

    def _tokens(self, scope: str, node: ast.AST) -> Iterator[Token]:
        if isinstance(node, ast.Import | ast.ImportFrom | ast.Call):
            modules = self.imported_modules(node)
            # ``from x import y`` imports x and x.y: count the leaves once.
            for module in sorted(modules):
                if not any(other.startswith(f"{module}.") for other in modules):
                    yield Token(scope, "import", module, node)
        elif isinstance(node, ast.Name):
            yield Token(scope, "name", self.chain(node), node, self.imported(node))
        elif isinstance(node, ast.Attribute):
            yield Token(scope, "attribute", self.chain(node), node)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield Token(scope, "name", node.name, node)
        elif isinstance(node, ast.arg | ast.keyword) and node.arg:
            yield Token(scope, "name", node.arg, node)

    def import_tables(self) -> Iterator[tuple[str, str, str, bool]]:
        """``(qualname, key, module, lazy)`` of each entry of a literal dict, at module level
        or in a class body, that maps a name to a module path or to a ``(module path, name)``
        pair: lazy tables (``lazy_getattr``) and registries (``import_module``). Relative paths
        anchor at the module's package."""

        for scope, node in self.nodes:
            value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
            if isinstance(value, ast.Call) and len(value.args) == 1:
                value = value.args[0]  # MappingProxyType({...})
            if not isinstance(value, ast.Dict) or scope.count(".") > 1:
                continue
            for key, item in zip(value.keys, value.values, strict=True):
                lazy = isinstance(item, ast.Constant)
                path = literal_string(item if lazy else (getattr(item, "elts", None) or [None])[0])
                if path and self.MODULE_PATH.fullmatch(path):
                    target = resolve_name(path, self.package) if path.startswith(".") else path
                    yield scope, literal_string(key) or "?", target, lazy

    def facade_names(self) -> dict[str, str]:
        """The names a package ``__init__`` binds to other modules' names: imports, aliases
        and lazily loaded names."""

        names: dict[str, str] = {}
        for scope, node in self.nodes:
            if scope not in {MODULE_SCOPE, TYPE_CHECKING}:
                continue
            if isinstance(node, ast.ImportFrom) and (sources := self.imported_modules(node)):
                source = min(sources, key=len)
                names.update(
                    (alias.asname or alias.name, f"{source}.{alias.name}") for alias in node.names
                )
            elif isinstance(node, ast.Import):
                names.update((alias.asname, alias.name) for alias in node.names if alias.asname)
            elif isinstance(node, ast.Assign) and under(self.chain(node.value), (self.package,)):
                names.update((name, self.chain(node.value)) for name in bound_names(node))
        names.update(
            (key, f"{target}.{key}") for _, key, target, lazy in self.import_tables() if lazy
        )
        return names

    @cached_property
    def annotations(self) -> frozenset[int]:
        """The nodes inside parameter, variable and return annotations."""

        return frozenset(
            id(child)
            for _, node in self.nodes
            for annotation in (
                getattr(node, "annotation", None),
                getattr(node, "returns", None),
            )
            if isinstance(annotation, ast.AST)
            for child in ast.walk(annotation)
        )


class Package:
    """The modules of one source tree, keyed by path relative to its root (``root`` names
    it), parsed once; names resolve across them through package facades."""

    ENGINE: ClassVar[Path] = PYTHON_ROOT / "cti"
    PROPERTIES: ClassVar[frozenset[str]] = frozenset({"property", "cached_property"})
    ENUMS: ClassVar[frozenset[str]] = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"})

    def __init__(self, modules: Mapping[str, Module], root: str = "cti") -> None:
        self.modules = MappingProxyType(dict(modules))
        self.root = root
        self._families: dict[str, frozenset[str]] = {}

    @classmethod
    def scan(cls, folder: Path, root: str) -> Package:
        """Every ``.py`` file under ``folder``, the package named ``root``."""

        paths = sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*.py"))
        return cls(
            {
                path: Module(path, (folder / path).read_text(encoding="utf-8"), root)
                for path in paths
            },
            root,
        )

    @classmethod
    def engine(cls) -> Package:
        return cls.scan(cls.ENGINE, "cti")

    @classmethod
    def of(cls, sources: Mapping[str, str], root: str = "cti") -> Package:
        """A stand-in package of the given files."""

        return cls({path: Module(path, source, root) for path, source in sources.items()}, root)

    def parse(self, path: str, source: str) -> Module:
        """``source`` as the module at ``path`` of this tree (named as the module there)."""

        if (present := self.modules.get(path)) is not None:
            return Module(path, source, *present.naming)
        return Module(path, source, self.root)

    def built(self, prefix: str) -> bool:
        """Whether a file under ``prefix`` exists."""

        return any(path.startswith(prefix) for path in self.modules)

    def is_module(self, path: str) -> bool:
        """Whether ``path`` (no suffix) is a module or a package."""

        return f"{path}.py" in self.modules or f"{path}/__init__.py" in self.modules

    @cached_property
    def names(self) -> frozenset[str]:
        """The dotted names of the modules."""

        return frozenset(module.name for module in self.modules.values())

    def owning_module(self, name: str) -> str:
        """The longest prefix of a dotted name that names a module, else ``""``."""

        while name and name not in self.names:
            name = name.rpartition(".")[0]
        return name

    @cached_property
    def facades(self) -> Mapping[str, Mapping[str, str]]:
        """Each package's names bound to other modules' names: imports, aliases and lazy
        table entries of its ``__init__``."""

        return MappingProxyType(
            {
                module.package: MappingProxyType(module.facade_names())
                for path, module in self.modules.items()
                if path.endswith("__init__.py")
            }
        )

    def defining(self, name: str) -> str:
        """``name`` with package facades followed to the module that defines it, for the name
        or any prefix of it (``pkg.Space.Base`` through ``pkg``'s re-export of ``Space``)."""

        applied: set[tuple[str, str]] = set()  # each facade entry once, so aliases cannot cycle
        while True:
            parts = name.split(".")
            found = (
                (key, target, parts[end:])
                for end in range(len(parts), 1, -1)
                if (key := (".".join(parts[: end - 1]), parts[end - 1])) not in applied
                and (target := self.facades.get(key[0], {}).get(key[1]))
            )
            key, target, rest = next(found, (None, None, []))
            if key is None:
                return name
            applied.add(key)
            name = ".".join([target, *rest])

    def subclasses(self, root: str) -> frozenset[str]:
        """``root`` and the classes that derive from it, by class name."""

        if root not in self._families:
            bases = {
                node.name: {module.ident(base) for base in node.bases}
                for module in self.modules.values()
                for _, node in module.nodes
                if isinstance(node, ast.ClassDef)
            }
            family = {root}
            while grown := {name for name, parents in bases.items() if parents & family} - family:
                family |= grown
            self._families[root] = frozenset(family)
        return self._families[root]

    def classes(self, dotted: str) -> list[tuple[Module, ast.ClassDef]]:
        """The module-level classes a dotted name defines, each with its module."""

        name = self.owning_module(dotted)
        return [
            (module, body)
            for module in self.modules.values()
            if module.name == name
            for body in class_bodies(module, dotted.removeprefix(f"{name}."))
        ]

    def lineage(self, module: Module, name: str) -> Iterator[tuple[ast.ClassDef, Module]]:
        """The bodies of the module-level classes ``name`` of ``module`` and of the package
        classes they derive from, each with its module."""

        pending = [(module, body) for body in class_bodies(module, name)]
        seen: set[int] = set()
        while pending:
            owner, body = pending.pop()
            if id(body) not in seen:
                seen.add(id(body))
                yield body, owner
                for base in body.bases:
                    dotted = owner.chain(unwrap(base))
                    pending += [(owner, local) for local in class_bodies(owner, dotted[2:])]
                    if not dotted.startswith("?"):
                        pending += [(owner, local) for local in class_bodies(owner, dotted)]
                        pending += self.classes(self.defining(dotted))

    def members(self, module: Module, name: str, public: bool = True) -> frozenset[str]:
        """Names (public ones by default) class ``name`` of ``module`` and its package
        ancestors define: what a body binds and what methods set on ``self``."""

        found: set[str] = set()
        for body, _ in self.lineage(module, name):
            for item in _statements(body.body):
                found |= bound_names(item)
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    found |= {
                        node.attr
                        for node in ast.walk(item)
                        if isinstance(node, ast.Attribute)
                        and not isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "self"
                    }
        return frozenset(member for member in found if not (public and member.startswith("_")))

    def fields(self, module: Module, name: str) -> frozenset[str]:
        """The fields of class ``name`` of ``module`` and its package ancestors: annotated
        names but class variables, and properties; an enum's members."""

        found: set[str] = set()
        bodies = list(self.lineage(module, name))
        enum = any(
            {owner.ident(unwrap(base)) for base in body.bases} & self.ENUMS
            for body, owner in bodies
        )
        for body, owner in bodies:
            for item in body.body:
                if enum and isinstance(item, ast.Assign):
                    found |= bound_names(item)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if "ClassVar" not in {owner.ident(node) for node in ast.walk(item.annotation)}:
                        found.add(item.target.id)
                elif isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and (
                    {owner.ident(decorator) for decorator in item.decorator_list} & self.PROPERTIES
                ):
                    found.add(item.name)
        return frozenset(found)

    def imported(self, module: Module) -> Iterator[tuple[str, str, bool]]:
        """``(qualname, name, annotation_only)`` of each package name ``module`` imports, reads
        off an imported module that does not define it (a submodule or a facade re-export), or
        names in an import table."""

        for token in module.tokens:
            imports = token.kind == "import"
            if (imports or self._reaches_module(token)) and under(token.chain, (self.root,)):
                # An import call in an annotation runs when the definition does.
                annotated = id(token.node) in module.annotations and not isinstance(
                    token.node, ast.Call
                )
                only = id(token.node) in module.annotation_only or annotated
                yield token.qualname, token.chain, only
        for qualname, _, target, _ in module.import_tables():
            yield qualname, target, False

    def _reaches_module(self, token: Token) -> bool:
        receiver = token.chain.rpartition(".")[0]
        if token.kind != "attribute" or not under(receiver, (self.root,)):
            return False
        return receiver in self.names and self.owning_module(self.defining(token.chain)) != receiver


# What a rule counts in one module of a package.
Hits = Callable[[Package, Module], Iterable[Hit]]


@dataclass(frozen=True)
class Rule:
    """One guard: its ``(path, symbol)`` positions outside ``owners`` within ``scope`` equal
    its ledger plus its additions. Each probe is a file the rule covers and code that must
    add a position and fail ``check``."""

    name: str
    hits: Hits
    probes: tuple[tuple[str, tuple[str, ...]], ...]
    ledger: str = ""
    scope: tuple[str, ...] = ("",)
    owners: tuple[str, ...] = ()

    def covers(self, path: str) -> bool:
        return path.startswith(self.scope) and not path.startswith(self.owners)

    def positions(
        self, package: Package, modules: Iterable[Module] | None = None
    ) -> Counter[Position]:
        """The positions in ``modules`` (the package's by default)."""

        return Counter(
            (module.path, symbol)
            for module in (package.modules.values() if modules is None else modules)
            if self.covers(module.path)
            for _, symbol in self.hits(package, module)
        )

    def check(self, package: Package, book: GuardBook | None = None) -> None:
        named = (*self.scope, *self.owners, *(path for path, _ in self.probes))
        unknown = [prefix for prefix in named if not package.built(prefix)]
        assert not unknown, f"{self.name}: no file under {unknown}"
        self._compare(self.positions(package), book or GuardBook())

    def check_probes(self, package: Package, book: GuardBook | None = None) -> None:
        found = self.positions(package)
        for path, codes in self.probes:
            assert self.covers(path), f"{self.name}: probe {path} is outside the rule"
            module = package.modules[path]
            others = found - self.positions(package, [module])
            base = self.positions(package, [module])
            for code in codes:
                mutated = self.positions(
                    package, [package.parse(path, f"{module.source}\n{code}\n")]
                )
                assert mutated - base, f"{self.name} misses in {path}:\n{code}"
                with pytest.raises(AssertionError, match="leaks only shrink"):
                    self._compare(others + mutated, book or GuardBook())

    def _compare(self, found: Counter[Position], book: GuardBook) -> None:
        added: Counter[Position] = Counter()
        for addition in book.additions:
            if addition.rule == self.name:
                landed = Ledger.parse(addition.ledger)
                assert all(map(addition.grants, landed)) and (
                    addition.count is None or landed.total() <= addition.count
                ), f"{self.name}: the {addition.commit} ledger exceeds its row"
                added += landed
        expected = Ledger.parse(self.ledger) + added
        new, gone = found - expected, expected - found
        assert not new and not gone, (
            f"{self.name}: leaks only shrink; a new position needs a GuardBook addition\n"
            f"new:\n{Ledger.render(new)}\ngone:\n{Ledger.render(gone)}\n"
            f"ledger now:\n{Ledger.render(found - added)}"
        )


@dataclass(frozen=True)
class Names:
    """Names, attributes and imported names whose last segment matches ``pattern``."""

    pattern: str

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        for token in module.tokens:
            if re.search(self.pattern, token.ident):
                yield token.qualname, token.ident


@dataclass(frozen=True)
class ScopedNames:
    """``Names`` hits counted only in ``paths``, keyword-argument names aside (they name the
    callee's parameter; the value passed is the read): a rename keeps the readers it
    replaces."""

    pattern: str
    paths: frozenset[str]

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        if module.path not in self.paths:
            return
        for token in module.tokens:
            if not isinstance(token.node, ast.keyword) and re.search(self.pattern, token.ident):
                yield token.qualname, token.ident


# Reads of the Plan's prefix-cache kind, counted where the boolean reuses_prefixes was read:
# typing the flag does not pay the debt of its readers.
PLAN_PREFIX_KIND = ScopedNames(
    r"^prefix_cache$",
    frozenset({"inference/_execution/runner.py", "plan.py", "resolution.py"}),
)


@dataclass(frozen=True, init=False)
class Imports:
    """Imports of ``modules`` or their submodules, static or literal dynamic."""

    modules: tuple[str, ...]

    def __init__(self, *modules: str) -> None:
        object.__setattr__(self, "modules", modules)

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        for token in module.tokens:
            if token.kind == "import" and under(token.chain, self.modules):
                yield token.qualname, token.chain


@dataclass(frozen=True, init=False)
class References:
    """Imports of ``modules`` and every use of a name bound by one."""

    modules: tuple[str, ...]

    def __init__(self, *modules: str) -> None:
        object.__setattr__(self, "modules", modules)

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        for token in module.tokens:
            used = token.kind == "name" and token.imported
            if (used or token.kind == "import") and under(token.chain, self.modules):
                yield token.qualname, token.chain


def under(name: str, modules: Iterable[str]) -> bool:
    return any(name == module or name.startswith(f"{module}.") for module in modules)


def within(module: Module, roots: Iterable[ast.AST]) -> Iterator[Token]:
    """Tokens of ``module`` spelled inside ``roots``."""

    inside = {id(node) for root in roots for node in ast.walk(root)}
    return (token for token in module.tokens if id(token.node) in inside)


def calls(module: Module, functions: Iterable[str]) -> Iterator[tuple[str, ast.Call]]:
    """Calls whose callee's last segment is one of ``functions``."""

    for scope, node in module.nodes:
        if isinstance(node, ast.Call) and module.ident(node.func) in functions:
            yield scope, node


def isinstance_targets(module: Module) -> Iterator[Hit]:
    """``(qualname, chain)`` of each class an ``isinstance`` or ``issubclass`` tests."""

    for scope, call in calls(module, ("isinstance", "issubclass")):
        if len(call.args) > 1:
            for item in _alternatives(call.args[1]):
                yield scope, module.chain(item)


def class_patterns(module: Module) -> Iterator[Hit]:
    """``(qualname, chain)`` of each class a ``match`` class pattern tests."""

    for scope, node in module.nodes:
        if isinstance(node, ast.MatchClass):
            yield scope, module.chain(node.cls)


def _alternatives(node: ast.AST) -> Iterator[ast.AST]:
    """The classes a tuple or ``|`` union names, nested ones included."""

    if isinstance(node, ast.Tuple):
        for item in node.elts:
            yield from _alternatives(item)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        yield from _alternatives(node.left)
        yield from _alternatives(node.right)
    else:
        yield node


def lexical(scope: str) -> str:
    """A ledger qualname as a lexical scope: ``TYPE_CHECKING`` blocks open no scope."""

    parts = [part for part in scope.split(".") if part != TYPE_CHECKING]
    return ".".join(parts) or MODULE_SCOPE


def class_bodies(module: Module, name: str) -> list[ast.ClassDef]:
    """The module-level classes ``name`` of ``module``."""

    return [
        node
        for scope, node in module.nodes
        if lexical(scope) == name and isinstance(node, ast.ClassDef) and node.name == name
    ]


def unwrap(base: ast.expr) -> ast.expr:
    """A class without its type arguments."""

    return base.value if isinstance(base, ast.Subscript) else base


def private(name: str) -> bool:
    """Whether ``name`` is private: one leading underscore, or two without trailing ones."""

    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def protocol(module: Module, node: ast.ClassDef) -> bool:
    return any(module.ident(unwrap(base)) == "Protocol" for base in node.bases)


def dataclass_flags(module: Module, node: ast.ClassDef) -> set[str]:
    """The keywords set true on a class's ``dataclass`` decorator."""

    return {
        keyword.arg
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call) and module.ident(decorator.func) == "dataclass"
        for keyword in decorator.keywords
        if keyword.arg and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    }


def _statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """The statements of a block and of the blocks its compound statements hold, not those
    of nested definitions."""

    for statement in body:
        yield statement
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for block in ("body", "orelse", "finalbody"):
            yield from _statements(getattr(statement, block, []))
        for clause in getattr(statement, "handlers", []):
            yield from _statements(clause.body)


def bound_names(statement: ast.AST) -> set[str]:
    """The names a statement binds in its own scope: definitions, imports and assignment,
    loop and ``with`` targets."""

    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {statement.name}
    if isinstance(statement, ast.Import | ast.ImportFrom):
        return {(alias.asname or alias.name).partition(".")[0] for alias in statement.names}
    targets: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        targets = list(statement.targets)
    elif isinstance(statement, ast.AnnAssign | ast.AugAssign | ast.For | ast.AsyncFor):
        targets = [statement.target]
    elif isinstance(statement, ast.With | ast.AsyncWith):
        targets = [item.optional_vars for item in statement.items if item.optional_vars]
    return {
        node.id
        for target in targets
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)
    }


def exports(module: Module) -> frozenset[str]:
    """A module's ``__all__``: a literal list or tuple of strings, ``list(TABLE)`` of a
    module-level literal dict, list or tuple, or a sum of these; ``?`` for anything else. With no
    ``__all__``, its public module-level names."""

    found = [node for node in module.tree.body if "__all__" in bound_names(node)]
    if not found:
        names = set().union(*map(bound_names, _statements(module.tree.body)))
        return frozenset(name for name in names if not name.startswith("_"))
    return _exported(module, getattr(found[-1], "value", None))


def _exported(module: Module, value: ast.expr | None) -> frozenset[str]:
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        return _exported(module, value.left) | _exported(module, value.right)
    if isinstance(value, ast.Call) and module.ident(value.func) == "list" and value.args:
        table = module.ident(value.args[0])
        tables = [
            getattr(node, "value", None) for node in module.tree.body if table in bound_names(node)
        ]
        value = tables[-1] if tables else None
        if isinstance(value, ast.Dict):
            items = value.keys
        else:
            items = value.elts if isinstance(value, ast.List | ast.Tuple) else [None]
    else:
        items = value.elts if isinstance(value, ast.List | ast.Tuple) else [None]
    return frozenset(literal_string(item) or "?" for item in items)


class Types:
    """Which expressions of a module hold an instance of the class ``name``, as code
    normally spells it: the class itself; a call of it, of one of ``factories``, or of a
    function or method annotated to return it; a parameter or name annotated with it (inside ``Optional``, unions and
    strings too); a name assigned such a value in the same scope; and an attribute a class
    body annotates with it, or that methods assign such a value."""

    def __init__(self, module: Module, name: str, *, factories: tuple[str, ...] = ()) -> None:
        self.module, self.name, self.factories = module, name, factories
        self.returns = {
            node.name
            for _, node in module.nodes
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and self.mentions(node.returns)
        }
        self.attributes: set[str] = set()
        self.names: set[tuple[str, str]] = set()
        for qualname, node in module.nodes:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                scope = lexical(qualname)
                arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                self.names |= {(scope, a.arg) for a in arguments if self.mentions(a.annotation)}
        while True:  # assignments of typed values, until none adds a name
            before = len(self.attributes) + len(self.names)
            for scope, node in module.nodes:
                if isinstance(node, ast.Assign | ast.AnnAssign):
                    annotated = isinstance(node, ast.AnnAssign) and self.mentions(node.annotation)
                    if annotated or self.holds(scope, node.value):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        for target in targets:
                            if isinstance(target, ast.Name):
                                self.names.add((lexical(scope), target.id))
                            elif isinstance(target, ast.Attribute):
                                self.attributes.add(target.attr)
                    if annotated and isinstance(node.target, ast.Name) and "." not in scope:
                        self.attributes.add(node.target.id)  # a class body's field
            if len(self.attributes) + len(self.names) == before:
                break

    def holds(self, scope: str, node: ast.AST | None) -> bool:
        """Whether ``node``, in ``scope``, holds an instance (or is the class)."""

        if isinstance(node, ast.Call):
            callee = self.module.ident(node.func)
            return callee == self.name or callee in self.returns or callee in self.factories
        if isinstance(node, ast.Attribute):
            return node.attr in self.attributes or self.module.chain(node).endswith(f".{self.name}")
        if isinstance(node, ast.Name):
            local = (lexical(scope), node.id) in self.names
            return local or (MODULE_SCOPE, node.id) in self.names or node.id == self.name
        return False

    def mentions(self, annotation: ast.AST | None) -> bool:
        """Whether an annotation says the class: itself or a generic of it, in ``Optional``,
        ``Union`` or ``|`` unions, or in a string; not as a container's item."""

        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                return self.mentions(ast.parse(annotation.value, mode="eval").body)
            except SyntaxError:
                return False
        if isinstance(annotation, ast.BinOp):
            return self.mentions(annotation.left) or self.mentions(annotation.right)
        if isinstance(annotation, ast.Tuple):
            return any(map(self.mentions, annotation.elts))
        if isinstance(annotation, ast.Subscript):  # a generic of the class, or a union
            if self.module.ident(annotation.value) in {"Optional", "Union"}:
                return self.mentions(annotation.slice)
            return self.mentions(annotation.value)
        return self.module.ident(annotation) == self.name if annotation is not None else False


class State:
    """What module-level and class-body values hold state (singletons and module state):
    a display, a comprehension or a call, unless the call is one of NONE's immutable
    constructors, class factories and field specifications, or makes a frozen dataclass, an
    enum or a named tuple of the package; a tuple or a conditional holding one; an operation
    as its left operand. A subscript's index (a type argument) holds nothing."""

    NONE: ClassVar[frozenset[str]] = frozenset(
        {
            "bool",
            "bytes",
            "float",
            "frozenset",
            "int",
            "range",
            "str",
            "tuple",
            "collections.namedtuple",
            "cti.utils.logger.init_logger",
            "dataclasses.field",
            "enum.auto",
            "functools.partial",
            "logging.getLogger",
            "pydantic.ConfigDict",
            "pydantic.Field",
            "re.compile",
            "struct.Struct",
            "types.MappingProxyType",
            "typing.NewType",
            "typing.TypeVar",
        }
    )
    FROZEN_BASES: ClassVar[frozenset[str]] = frozenset(
        {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "NamedTuple"}
    )

    def __init__(self, package: Package, module: Module) -> None:
        self.package, self.module = package, module

    def holds(self, node: ast.AST | None) -> bool:
        if node is None or isinstance(node, ast.Lambda):
            return False
        if isinstance(
            node, ast.List | ast.Dict | ast.Set | ast.ListComp | ast.DictComp | ast.SetComp
        ):
            return True
        if isinstance(node, ast.Call):
            made = self.package.defining(self.module.chain(unwrap(node.func)))
            return made.removeprefix("builtins.") not in self.NONE and not self._frozen(made)
        if isinstance(node, ast.Subscript | ast.BinOp):
            return self.holds(getattr(node, "value", None) or node.left)
        if isinstance(node, ast.Tuple | ast.IfExp | ast.BoolOp):
            return any(map(self.holds, ast.iter_child_nodes(node)))
        return False

    def _frozen(self, dotted: str) -> bool:
        module = self.module
        local = class_bodies(module, dotted.removeprefix("?.").removeprefix(f"{module.name}."))
        found = [(module, body) for body in local] or self.package.classes(dotted)
        return bool(found) and all(
            "frozen" in dataclass_flags(owner, body)
            or bool({owner.ident(unwrap(base)) for base in body.bases} & self.FROZEN_BASES)
            for owner, body in found
        )


@dataclass(frozen=True)
class ModuleState:
    """State a module keeps: assignments at module level and in class bodies whose value
    holds state as State counts it; ``global`` statements; functools caches on module-level
    functions; and writes into a module-level variable, its own or one imported from its
    package: item and attribute stores and deletes, augmented assignments to them, mutating
    method calls, and registering decorators (``@NAME.route(...)``). A script's ``if
    __name__ == "__main__":`` body, a package's ``__all__`` and PEP 562 hooks keep none."""

    CACHES: ClassVar[frozenset[str]] = frozenset({"functools.cache", "functools.lru_cache"})
    FREE: ClassVar[frozenset[str]] = frozenset({"__all__", "__getattr__", "__dir__"})
    MUTATORS: ClassVar[frozenset[str]] = frozenset(
        {
            "add",
            "append",
            "clear",
            "difference_update",
            "discard",
            "extend",
            "insert",
            "intersection_update",
            "pop",
            "popitem",
            "remove",
            "reverse",
            "setdefault",
            "sort",
            "symmetric_difference_update",
            "update",
        }
    )

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        state = State(package, module)
        script = {
            id(child)
            for node in module.tree.body
            if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
            for child in ast.walk(node)
        }
        variables = {
            name
            for node in module.tree.body
            if isinstance(node, ast.Assign | ast.AnnAssign) and id(node) not in script
            for name in bound_names(node)
        }
        classes = {lexical(s) for s, node in module.nodes if isinstance(node, ast.ClassDef)}
        local = self._locals(module)
        for scope, node in module.nodes:
            if id(node) in script:
                continue
            if isinstance(node, ast.Global):
                yield scope, "global"
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                names = sorted(bound_names(node))
                shared = scope == MODULE_SCOPE or lexical(scope) in classes
                if shared and set(names) - self.FREE and state.holds(node.value):
                    yield scope, "mutable:" + ",".join(names)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                for decorator in node.decorator_list:
                    called = decorator.func if isinstance(decorator, ast.Call) else decorator
                    top = scope.count(".") == 0 and not isinstance(node, ast.ClassDef)
                    if top and module.chain(called) in self.CACHES:
                        yield scope, f"cache:{node.name}"
                    elif isinstance(called, ast.Attribute):  # registers where it is defined
                        enclosing = scope.rpartition(".")[0] or MODULE_SCOPE
                        yield from self._writes(module, enclosing, called.value, variables, local)
            target = self._written(node)
            if target is not None:
                yield from self._writes(module, scope, target, variables, local)

    def _written(self, node: ast.AST) -> ast.expr | None:
        """The object a store, delete, augmented assignment or mutating call writes into."""

        if isinstance(node, ast.Subscript | ast.Attribute) and not isinstance(node.ctx, ast.Load):
            return node.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return node.func.value if node.func.attr in self.MUTATORS else None
        return None

    def _writes(
        self,
        module: Module,
        scope: str,
        target: ast.expr,
        variables: set[str],
        local: dict[str, set[str]],
    ) -> Iterator[Hit]:
        if isinstance(target, ast.Attribute):  # a variable read off an imported module
            chain = module.chain(target)
            if "." in chain and chain.partition(".")[0] == module.name.partition(".")[0]:
                yield scope, f"mutates:{chain}"
            return
        if not isinstance(target, ast.Name) or target.id in local.get(lexical(scope), set()):
            return
        chain = module.chain(target)
        if target.id in variables:
            yield scope, f"mutates:{target.id}"
        elif "." in chain and chain.partition(".")[0] == module.name.partition(".")[0]:
            yield scope, f"mutates:{chain}"  # imported from the module's package

    @staticmethod
    def _locals(module: Module) -> dict[str, set[str]]:
        """Each function's parameters and the names it binds."""

        found: dict[str, set[str]] = {}
        for scope, node in module.nodes:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                arguments = node.args
                names = {
                    a.arg for a in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
                }
                names |= {a.arg for a in (arguments.vararg, arguments.kwarg) if a}
                names |= {
                    child.id
                    for child in ast.walk(node)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
                }
                found[lexical(scope)] = names
        return found


def attribute_spellings(name: str, template: str = "{}") -> tuple[str, ...]:
    """Attribute ``name`` read off an object, and as a bare name, inside ``template``."""

    return (template.format(f"_object.{name}"), template.format(name))


def dotted_spellings(symbol: str, template: str = "{}") -> tuple[str, ...]:
    """``module.name`` imported and used inside ``template``: through the module, and by
    name."""

    module, _, name = symbol.rpartition(".")
    return (
        f"import {module}\n{template.format(symbol)}",
        f"from {module} import {name}\n{template.format(name)}",
    )


def module_spellings(module: str) -> tuple[str, ...]:
    """Imports of ``module``: the module, a name from it, and the module from its parent."""

    parent, _, last = module.rpartition(".")
    return (
        f"import {module}",
        f"from {module} import _name",
        *((f"from {parent} import {last}",) if parent else ()),
    )
