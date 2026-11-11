"""Parallel names stay with their owners, one rule divides KV heads, and ranks agree only
where registered.

Other files keep their parallel-name lines under ceilings that may only go down."""

from __future__ import annotations

import ast
import re
from collections import Counter

from tests.support.paths import PYTHON_ROOT

_PACKAGE = PYTHON_ROOT / "cti"
_AXIS = re.compile(r"(\b|_)(tp|ep|sp)(_|\b)|(tensor|expert|sequence)[_ -]?parallel", re.I)
_GROUP_TYPE = re.compile(r"ParallelGroup|Topology|CollectiveGroup")
# Models name an axis only to declare the capability, or to list it in ModelCapability.ALL.
_CAPABILITY = re.compile(
    r"ModelCapability\.TENSOR_PARALLEL|^\s*TENSOR_PARALLEL = auto\(\)|^\s*ALL = "
)
# Axis names: where work is placed on ranks (paths under cti; dirs end in /).
_AXIS_OWNERS = tuple(
    """
    distributed/ training/ worker/
    layers/__init__.py layers/site.py layers/base.py layers/placement.py layers/linear.py
    layers/embedding.py layers/moe.py layers/gated_delta_net/layer.py
    config.py plan.py resolution.py
    server/args.py
    """.split()
)
# Group types also: where ranks run in lockstep.
_LOCKSTEP_OWNERS = tuple(
    """
    inference/_execution/runner.py inference/_control/_engine.py runtime/ runtime/_core.py
    """.split()
)
# Lines left for the follow-ups that place layers and name ranks.
_CEILINGS = {
    "attention/base.py": 1,
    "attention/family.py": 1,
    "attention/flashinfer.py": 1,
    "attention/mha.py": 3,
}
# DFlash is orthogonal to TP/EP: its modules name no rank, group or collective. A KV group
# (one of the inference state's groups) is no rank group.
_SPECULATIVE = re.compile(
    r"rank|group|collective|all_(gather|reduce)|broadcast|distributed|\bdist\b", re.I
)
_KV_GROUPS = re.compile(r"\bstate\.groups\b|\bKV groups\b")
# Where ranks agree on a value (outside distributed/): startup memory sizing and step costs, and
# SPMD-shaped transitional exceptions, each deleted by the commit named.
# (path, function, method, calls, deleted by)
_AGREEMENTS = (
    # Each batch's digest, under CTI_CHECKS.
    ("inference/_control/_engine.py", "InferenceEngine.step", "agree_on_step", 1, "CP3"),
)
_AGREEMENT_METHODS = frozenset({"min_max_int", "require_same", "agree_on_step"})
_KV_HEADS = re.compile(r"kv_head|num_kv|key_value_head")
# The MHA family's KV-head rule, which the QKV projection's shard and the family's row
# layout read.
_KV_HEAD_RULE = "attention/requirement.py"
_KV_HEAD_RULE_USERS = {_KV_HEAD_RULE, "attention/mha.py", "layers/linear.py"}


def _sources() -> dict[str, str]:
    return {
        path.relative_to(_PACKAGE).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(_PACKAGE.rglob("*.py"))
    }


def _outside_lines(path: str, text: str) -> list[str]:
    if path.startswith(_AXIS_OWNERS):
        return []
    return [
        f"{path}:{number}: {line.strip()}"
        for number, line in enumerate(text.splitlines(), 1)
        if not (path.startswith("models/") and _CAPABILITY.search(line))
        and (_AXIS.search(line) or (_GROUP_TYPE.search(line) and path not in _LOCKSTEP_OWNERS))
    ]


def _names_kv_heads(node: ast.AST) -> bool:
    return any(
        _KV_HEADS.search(name.id if isinstance(name, ast.Name) else name.attr)
        for name in ast.walk(node)
        if isinstance(name, ast.Name | ast.Attribute)
    )


def _dividend(node: ast.AST) -> ast.AST | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv | ast.Div | ast.Mod):
        return node.left
    if isinstance(node, ast.Call) and node.args:
        function = node.func
        name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", "")
        return node.args[0] if name in {"_div_even", "div_ceil", "divmod"} else None
    return None


def _kv_head_rule_breaks(path: str, text: str) -> list[str]:
    breaks = []
    for node in ast.walk(ast.parse(text, filename=path)):
        dividend = _dividend(node)
        if dividend is not None and path != _KV_HEAD_RULE and _names_kv_heads(dividend):
            breaks.append(f"{path}:{node.lineno} divides KV heads")
        if isinstance(node, ast.arg | ast.keyword) and node.arg == "allow_replicate":
            breaks.append(f"{path}:{node.lineno} replicates KV heads")
        named = getattr(node, "attr", None) or getattr(node, "id", None)
        if named in {"kv_head_window", "qo_head_window"} and path not in _KV_HEAD_RULE_USERS:
            breaks.append(f"{path}:{node.lineno} calls a head rule")
    return breaks


def test_parallel_names_stay_with_their_owners() -> None:
    assert all((_PACKAGE / owner).exists() for owner in _AXIS_OWNERS + _LOCKSTEP_OWNERS)
    outside = {
        path: lines for path, text in _sources().items() if (lines := _outside_lines(path, text))
    }
    grown = [
        line
        for path, lines in outside.items()
        if len(lines) > _CEILINGS.get(path, 0)
        for line in lines
    ]

    assert not grown, "parallel names outside their owners:\n" + "\n".join(grown)
    assert {path: len(lines) for path, lines in outside.items()} == _CEILINGS, "lower the ceilings"


def _agreements(text: str) -> Counter[tuple[str, str]]:
    """``(function qualname, method)`` -> calls of a rank-agreement method."""

    calls: Counter[tuple[str, str]] = Counter()
    scope: list[str] = []

    def visit(node: ast.AST) -> None:
        named = isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        if named:
            scope.append(node.name)
        call = node.func if isinstance(node, ast.Call) else None
        # An injected agreement is held privately (``self._agree_on_step``).
        method = call.attr.lstrip("_") if isinstance(call, ast.Attribute) else None
        if method in _AGREEMENT_METHODS:
            calls[".".join(scope), method] += 1
        for child in ast.iter_child_nodes(node):
            visit(child)
        if named:
            scope.pop()

    visit(ast.parse(text))
    return calls


def test_ranks_agree_only_where_registered() -> None:
    found = Counter(
        {
            (path, function, method): count
            for path, text in _sources().items()
            if not path.startswith("distributed/")
            for (function, method), count in _agreements(text).items()
        }
    )
    registered: Counter[tuple[str, str, str]] = Counter()
    for path, function, method, count, _deleted_by in _AGREEMENTS:
        registered[path, function, method] += count

    assert found == registered, "register rank agreements in _AGREEMENTS"


def test_one_rule_divides_kv_heads() -> None:
    breaks = [
        line for path, text in _sources().items() for line in _kv_head_rule_breaks(path, text)
    ]

    assert not breaks, "KV heads outside attention.requirement.kv_head_window:\n" + "\n".join(
        breaks
    )


def test_speculative_modules_name_no_rank_group_or_collective() -> None:
    lines = [
        f"{path}:{number}: {line.strip()}"
        for path, text in _sources().items()
        # The draft config parses checkpoint keys, such as conv_group_size, not ranks.
        if ("dflash" in path or "speculative" in path) and not path.endswith("dflash/config.py")
        for number, line in enumerate(text.splitlines(), 1)
        if _SPECULATIVE.search(_KV_GROUPS.sub("", line))
    ]

    assert lines == [], "rank-aware lines in speculative modules:\n" + "\n".join(lines)
