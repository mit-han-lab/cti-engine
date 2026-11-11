"""Import-graph helpers shared by architecture boundary tests."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Iterator
from importlib.util import resolve_name
from pathlib import Path

from tests.support.paths import PYTHON_ROOT

MODULE_SCOPE = "<module>"
TYPE_CHECKING = "TYPE_CHECKING"
IMPORTERS = frozenset({"importlib.import_module", "importlib.__import__", "__import__"})
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
# Statements whose bodies may not run, so a class body binding in them is not certain.
_COMPOUND = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    getattr(ast, "TryStar", ast.Try),  # Python 3.11+
    ast.With,
    ast.AsyncWith,
    ast.Match,
)
_NAMED = (*_DEFINITIONS, ast.ExceptHandler, ast.MatchAs, ast.MatchStar)

# What a name is bound to: an import's dotted name, an assigned expression, or None.
Binding = str | ast.expr | None

# Runs argv[1] in a fresh interpreter, then prints the loaded modules under argv[2:].
_LOADED_MODULES = """
import json
import sys

exec(sys.argv[1])
roots = sys.argv[2:]
print(json.dumps(sorted(
    name for name in sys.modules if any(name == root or name.startswith(root + ".") for root in roots)
)))
"""


def module_name(path: Path) -> str:
    """Dotted module name of a source file under ``python/`` (the package for ``__init__.py``)."""

    parts = list(path.relative_to(PYTHON_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports_from_source(source: str, *, package: str) -> frozenset[str]:
    """Return imported modules with relative and imported-child names resolved."""

    scopes = Scopes(ast.parse(source), package)
    return frozenset(module for _, node in scopes.nodes for module in scopes.imported_modules(node))


def runtime_imports_from_source(source: str, *, package: str) -> tuple[tuple[str, int], ...]:
    """Return ``(module, line)`` for every import that can run.

    ``if typing.TYPE_CHECKING:`` bodies are skipped (their ``else`` branches are kept);
    function-level imports count. Names resolve as in ``imports_from_source``.
    """

    scopes = Scopes(ast.parse(source), package)
    return tuple(
        (module, node.lineno)
        for _, node in scopes.nodes
        if id(node) not in scopes.annotation_only
        for module in sorted(scopes.imported_modules(node))
    )


class Scopes:
    """One module's nodes with their qualnames, and its names resolved as Python binds them.

    Scopes follow Python: class bodies do not enclose functions, lambdas or
    comprehensions, and see the enclosing scope until a top-level statement binds the
    name. A name whose one binding is an import, or an assigned reference that resolves,
    stands for that dotted name; any other local spells ``?.name``. ``if`` bodies on a
    resolved ``typing.TYPE_CHECKING`` are ``annotation_only``.
    """

    def __init__(self, tree: ast.Module, package: str) -> None:
        self.package = package
        self._module = id(tree)
        self._scope: dict[int, int] = {}
        self._parent: dict[int, int | None] = {self._module: None}
        self._classes: set[int] = set()
        self._comprehensions: set[int] = set()
        self._bindings: dict[tuple[int, str], set[Binding]] = {}
        self._declared: dict[tuple[int, str], type] = {}
        self._moved: dict[tuple[int, str], int] = {}
        self._assigned: dict[int, tuple[int, Binding]] = {}
        self._class_bound: dict[tuple[int, str], tuple[int, int]] = {}
        self._resolving: set[int] = set()
        self._visit(tree, self._module)
        self._bind_declared()
        self.annotation_only: set[int] = set()
        self.nodes = tuple(self._walk(tree.body, MODULE_SCOPE, annotation=False))

    def chain(self, node: ast.AST) -> str:
        """Dotted name of a reference; ``?`` stands for an unnamed expression."""

        if isinstance(node, ast.Attribute):
            return _unqualified(f"{self.chain(node.value)}.{node.attr}")
        if not isinstance(node, ast.Name):
            return "?"
        bindings = self._bindings_of(node)
        if not bindings:
            return node.id
        if isinstance(node.ctx, ast.Load) and len(bindings) == 1:
            dotted = self._expand(next(iter(bindings)))
            if dotted is not None:
                return dotted
        return f"?.{node.id}"

    def imported(self, node: ast.Name) -> bool:
        return bool(self._bindings_of(node)) and not self.chain(node).startswith("?")

    def global_values(
        self, node: ast.Assign | ast.AnnAssign | ast.AugAssign | ast.NamedExpr
    ) -> Iterator[ast.AST]:
        """The parts of an assignment's value a function stores in names it declares
        ``global``; unpacking pairs targets with values where it can."""

        scope = self._binding_scope(node, self._scope[id(node)])
        for target in node.targets if isinstance(node, ast.Assign) else (node.target,):
            yield from self._stored_globally(scope, target, node.value)

    def _stored_globally(self, scope: int, target: ast.expr, value: ast.expr) -> Iterator[ast.expr]:
        target = target.value if isinstance(target, ast.Starred) else target
        sequences = ast.Tuple | ast.List
        pairs = None
        if isinstance(target, sequences) and isinstance(value, sequences):
            pairs = _unpack(target.elts, value.elts)
        if pairs is not None:
            for item, parts in pairs:
                for part in parts:
                    yield from self._stored_globally(scope, item, part)
        elif any(
            self._moved.get((scope, name.id)) == self._module
            for name in ast.walk(target)
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
        ):
            yield value

    def imported_modules(self, node: ast.AST) -> set[str]:
        """Modules one node imports: ``import``, ``from``, and literal dynamic imports.

        ``importlib.import_module(literal[, package])`` and ``__import__(literal)`` count
        under any alias; a literal may be a constant concatenation.
        """

        if not isinstance(node, ast.Call):
            return _static_imports(node, self.package)
        if self.chain(node.func) not in IMPORTERS:
            return set()
        arguments = {item.arg: item.value for item in node.keywords}
        module = literal_string((*node.args[:1], arguments.get("name"))[0])
        if module is None or not module.startswith("."):
            return set() if module is None else {module}
        anchor = (*node.args[1:2], arguments.get("package"))[0]
        anchor_name = (
            self.package if self.chain(anchor) == "__package__" else literal_string(anchor)
        )
        return set() if anchor_name is None else {resolve_name(module, anchor_name)}

    def _visit(self, node: ast.AST, scope: int) -> None:
        self._scope[id(node)] = scope
        self._bind(node, scope)
        if isinstance(node, ast.arg):
            return  # its annotation is visited in the outer scope
        outer, inner = _split(node)
        for child in outer:
            self._visit(child, scope)
        if inner:
            self._parent[id(node)] = self._parent[scope] if scope in self._classes else scope
            if isinstance(node, ast.ClassDef):
                self._classes.add(id(node))
                self._order_class(node)
            elif isinstance(node, _COMPREHENSIONS):
                self._comprehensions.add(id(node))
            for child in inner:
                self._visit(child, id(node))

    def _bind(self, node: ast.AST, scope: int) -> None:
        if isinstance(node, ast.Global | ast.Nonlocal):
            self._declared.update(((scope, name), type(node)) for name in node.names)
        elif isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr):
            scope = self._binding_scope(node, scope)
            value = (
                node.value if isinstance(node.value, ast.Name | ast.Attribute | ast.Call) else None
            )
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            self._assigned.update((id(target), (scope, value)) for target in targets)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            scope, value = self._assigned.get(id(node), (scope, None))
            self._add(scope, node.id, value)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.partition(".")[0]
                self._add(scope, alias.asname or root, alias.name if alias.asname else root)
        elif isinstance(node, ast.ImportFrom) and (modules := _static_imports(node, self.package)):
            source = min(modules, key=len)
            for alias in node.names:
                if alias.name != "*":
                    self._add(scope, alias.asname or alias.name, f"{source}.{alias.name}")
        elif isinstance(node, _NAMED) and node.name:
            self._add(scope, node.name, None)
        elif isinstance(node, ast.arg):
            self._add(scope, node.arg, None)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            self._add(scope, node.rest, None)

    def _binding_scope(self, node: ast.AST, scope: int) -> int:
        # A walrus in a comprehension binds in the enclosing function.
        while isinstance(node, ast.NamedExpr) and scope in self._comprehensions:
            scope = self._parent[scope]
        return scope

    def _add(self, scope: int, name: str, binding: Binding) -> None:
        self._bindings.setdefault((scope, name), set()).add(binding)

    def _order_class(self, node: ast.ClassDef) -> None:
        """Record where an unconditional statement of a class body binds a name for sure;
        a name the class deletes anywhere is never sure."""

        deleted = {
            name.id
            for name in ast.walk(node)
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Del)
        }
        # ``except ... as name`` deletes the name when the handler ends.
        deleted |= {h.name for h in ast.walk(node) if isinstance(h, ast.ExceptHandler) and h.name}
        for statement in node.body:
            if isinstance(statement, _COMPOUND):
                continue
            end = (statement.end_lineno, statement.end_col_offset)
            for name in set(_binds(statement)) - deleted:
                self._class_bound.setdefault((id(node), name), end)

    def _bind_declared(self) -> None:
        """Move ``global`` and ``nonlocal`` bindings to the scope that holds them."""

        for (scope, name), kind in self._declared.items():
            target = self._module
            if kind is ast.Nonlocal:
                target = self._parent[scope]
                while (target, name) not in self._bindings and self._parent[target] is not None:
                    target = self._parent[target]
            self._moved[(scope, name)] = target
            moved = self._bindings.pop((scope, name), set())
            self._bindings.setdefault((target, name), set()).update(moved)

    def _bindings_of(self, node: ast.Name) -> set[Binding]:
        """What ``node`` may be bound to; empty for a global or builtin it never binds."""

        scope = self._scope.get(id(node), self._module)
        found: set[Binding] = set()
        while scope is not None:
            scope = self._moved.get((scope, node.id), scope)
            if (scope, node.id) in self._bindings:
                found |= self._bindings[(scope, node.id)]
                bound = self._class_bound.get((scope, node.id))
                if scope not in self._classes or (
                    bound and bound <= (node.lineno, node.col_offset)
                ):
                    break
            scope = self._parent[scope]
        return found

    def _expand(self, binding: Binding) -> str | None:
        """The dotted name a binding stands for, or None for an opaque one."""

        if isinstance(binding, str):
            return _unqualified(binding)
        if binding is None or id(binding) in self._resolving:
            return None
        self._resolving.add(id(binding))
        dotted = self.chain(binding)
        self._resolving.discard(id(binding))
        return None if dotted.startswith("?") else dotted

    def _walk(
        self, children: Iterable[ast.AST], qualname: str, *, annotation: bool
    ) -> Iterator[tuple[str, ast.AST]]:
        for child in children:
            inner = qualname
            if isinstance(child, _DEFINITIONS):
                inner = child.name if qualname == MODULE_SCOPE else f"{qualname}.{child.name}"
            if annotation:
                self.annotation_only.add(id(child))
            yield inner, child
            if not self._annotates(child):
                yield from self._walk(ast.iter_child_nodes(child), inner, annotation=annotation)
                continue
            segment = TYPE_CHECKING if qualname == MODULE_SCOPE else f"{qualname}.{TYPE_CHECKING}"
            yield from self._walk((child.test, *child.orelse), qualname, annotation=annotation)
            yield from self._walk(child.body, segment, annotation=True)

    def _annotates(self, node: ast.AST) -> bool:
        """Whether ``node`` is an ``if`` whose body runs only under a type checker."""

        if not isinstance(node, ast.If):
            return False
        return self._imported_root(node.test) and self.chain(node.test) == "typing.TYPE_CHECKING"

    def root(self, node: ast.AST) -> ast.Name | None:
        """The name a reference reads attributes off, or None for a call or other value."""

        while not isinstance(node, ast.Name):
            if isinstance(node, ast.Attribute):
                node = node.value
            else:
                return None
        return node

    def _imported_root(self, node: ast.AST) -> bool:
        """Whether a reference's root is bound only to an import, directly or by aliases."""

        root = self.root(node)
        bindings = self._bindings_of(root) if root else set()
        binding = next(iter(bindings)) if len(bindings) == 1 else None
        if binding is None or isinstance(binding, str):
            return binding is not None
        if id(binding) in self._resolving:
            return False
        self._resolving.add(id(binding))
        found = self._imported_root(binding)
        self._resolving.discard(id(binding))
        return found


def _split(node: ast.AST) -> tuple[list[ast.AST], list[ast.AST]]:
    """A node's children evaluated in the enclosing scope, and those in the scope it opens."""

    if isinstance(node, ast.ClassDef):
        return [*node.decorator_list, *node.bases, *node.keywords], node.body
    if isinstance(node, _FUNCTIONS):
        arguments = node.args
        params = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        params += [param for param in (arguments.vararg, arguments.kwarg) if param]
        outer = [*getattr(node, "decorator_list", ()), *arguments.defaults, *arguments.kw_defaults]
        outer += [param.annotation for param in params] + [getattr(node, "returns", None)]
        body = node.body if isinstance(node.body, list) else [node.body]
        return [child for child in outer if child], [*params, *body]
    if isinstance(node, _COMPREHENSIONS):
        first = node.generators[0]
        inner = [child for child in ast.iter_child_nodes(node) if child is not first]
        return [first.iter], [first.target, *first.ifs, *inner]
    return list(ast.iter_child_nodes(node)), []


def _binds(statement: ast.stmt) -> Iterator[str]:
    """Names a simple statement binds once it has run."""

    if isinstance(statement, _DEFINITIONS):
        yield statement.name
    elif isinstance(statement, ast.Import | ast.ImportFrom):
        yield from ((alias.asname or alias.name).partition(".")[0] for alias in statement.names)
    elif isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign | ast.Expr):
        if statement.value is None:  # a bare annotation binds nothing
            return
        targets = statement.targets if isinstance(statement, ast.Assign) else []
        if isinstance(statement, ast.AnnAssign | ast.AugAssign):
            targets = [statement.target]
        if isinstance(statement.value, ast.NamedExpr):
            targets = [*targets, statement.value.target]
        stored = (name for target in targets for name in ast.walk(target))
        yield from (
            name.id
            for name in stored
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
        )


def _unpack(
    targets: list[ast.expr], values: list[ast.expr]
) -> list[tuple[ast.expr, list[ast.expr]]] | None:
    """Each target with the literal values it receives, or None when they do not line up."""

    if any(isinstance(value, ast.Starred) for value in values):
        return None
    stars = [index for index, target in enumerate(targets) if isinstance(target, ast.Starred)]
    if not stars:
        return (
            [(t, [v]) for t, v in zip(targets, values, strict=True)]
            if len(targets) == len(values)
            else None
        )
    star, after = stars[0], len(targets) - stars[0] - 1
    if len(values) < len(targets) - 1:
        return None
    split = len(values) - after
    return [
        *((target, [value]) for target, value in zip(targets[:star], values[:star], strict=True)),
        (targets[star], values[star:split]),
        *(
            (target, [value])
            for target, value in zip(targets[star + 1 :], values[split:], strict=True)
        ),
    ]


def _unqualified(dotted: str) -> str:
    """``builtins.getattr`` and ``__builtins__.getattr`` are ``getattr``."""

    for prefix in ("builtins.", "__builtins__."):
        dotted = dotted.removeprefix(prefix)
    return dotted


def _static_imports(node: ast.AST, package: str) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if not isinstance(node, ast.ImportFrom):
        return set()
    imported_from = node.module or ""
    if node.level:
        imported_from = resolve_name("." * node.level + imported_from, package)
    if not imported_from:
        return set()
    return {
        imported_from,
        *(f"{imported_from}.{alias.name}" for alias in node.names if alias.name != "*"),
    }


def literal_string(node: ast.AST | None) -> str | None:
    """The value of a string literal or a constant concatenation, else None."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal_string(node.left), literal_string(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = [literal_string(value) for value in node.values]
        return None if None in parts else "".join(parts)
    return None


def loaded_modules(setup: str, *roots: str) -> frozenset[str]:
    """Modules under ``roots`` loaded by a fresh interpreter after running ``setup``."""

    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(PYTHON_ROOT)
    completed = subprocess.run(
        (sys.executable, "-c", _LOADED_MODULES, setup, *roots),
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    return frozenset(json.loads(completed.stdout.splitlines()[-1]))


def assert_not_loaded(setup: str, *roots: str) -> None:
    loaded = loaded_modules(setup, *roots)
    assert not loaded, f"{setup!r} loaded {sorted(loaded)}"
