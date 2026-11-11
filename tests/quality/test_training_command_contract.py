# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors

"""Every training verb names a scheduler command its senders call with matching arguments.

``TrainingScheduler.handle`` runs ``getattr(scheduler, verb.value)(command_id, *args)``, so
a verb without its method, or a send whose args do not fit that method, fails only at run
time. A sender is any ``<x>.call(TrainingVerb.<VERB>, ...)`` in the engine's source. The
scheduler's public names are its commands, one per verb, plus ``NON_COMMANDS``: adding a
public name forces a choice between a verb and that list.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

import pytest

from cti.training import TrainingVerb
from cti.training.scheduler import TrainingScheduler
from tests.support.paths import PYTHON_ROOT

# The scheduler's public surface that is not a command: what the worker and the mode
# switch call on it directly.
NON_COMMANDS = frozenset(
    {
        "drained",
        "handle",
        "has_active",
        "release",
        "renew_awaiting",
        "requested_holds",
        "restore",
        "step",
        "switched",
        "wants_to_run",
    }
)


@dataclass(frozen=True)
class Send:
    """One ``.call(TrainingVerb.<verb>, ...)`` call site."""

    where: str
    verb: str
    positional: int
    keywords: tuple[str, ...]
    starred: bool


def sends(source: str, where: str) -> Iterator[Send]:
    """The training-verb sends in ``source``."""

    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "call"
            and node.args
        ):
            continue
        verb, *args = node.args
        if not (isinstance(verb, ast.Attribute) and _is_verb_enum(verb.value)):
            continue
        yield Send(
            where=f"{where}:{node.lineno}",
            verb=verb.attr,
            positional=sum(not isinstance(arg, ast.Starred) for arg in args),
            keywords=tuple(keyword.arg or "**" for keyword in node.keywords),
            starred=any(isinstance(arg, ast.Starred) for arg in args),
        )


def _is_verb_enum(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TrainingVerb") or (
        isinstance(node, ast.Attribute) and node.attr == "TrainingVerb"
    )


def _public_names(scheduler: type) -> set[str]:
    return {
        name
        for cls in scheduler.__mro__
        if cls is not object
        for name in vars(cls)
        if not name.startswith("_")
    }


def _command_params(scheduler: type, name: str) -> list[inspect.Parameter] | str:
    """The parameters a command takes after ``command_id``, or why it is no command."""

    method = inspect.getattr_static(scheduler, name, None)
    if not inspect.isfunction(method):
        return "names no scheduler method"
    params = list(inspect.signature(method).parameters.values())[1:]  # self
    if not params or params[0].name != "command_id":
        return "is a method whose first parameter is not command_id"
    params = params[1:]
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if any(param.kind not in positional for param in params):
        return "is a method taking more than fixed positional args"
    return params


def violations(
    verbs: type[Enum], scheduler: type, non_commands: frozenset[str], all_sends: list[Send]
) -> list[str]:
    """Where ``verbs``, ``scheduler``'s commands and the sends disagree."""

    found = []
    commands = {}
    for verb in verbs:
        params = _command_params(scheduler, verb.value)
        if isinstance(params, str):
            found.append(f"verb {verb.name}: {verb.value!r} {params}")
        else:
            commands[verb.name] = params

    public = _public_names(scheduler)
    values = {verb.value for verb in verbs}
    for name in sorted(public - values - non_commands):
        found.append(f"scheduler.{name} is public but neither a verb nor a non-command")
    for name in sorted(non_commands - public):
        found.append(f"non-command {name} is no public scheduler name")
    for name in sorted(non_commands & values):
        found.append(f"non-command {name} is also a verb")

    sent = set()
    for send in all_sends:
        sent.add(send.verb)
        if send.verb not in verbs.__members__:
            found.append(f"{send.where}: sends unknown verb {send.verb}")
            continue
        params = commands.get(send.verb)
        if params is None:
            continue
        if send.starred:
            found.append(f"{send.where}: {send.verb} forwards *args; spell its args out")
            continue
        names = [param.name for param in params]
        unknown = [name for name in send.keywords if name not in names[send.positional :]]
        if unknown:
            found.append(f"{send.where}: {send.verb} passes unknown keywords {unknown}")
        given = send.positional + len(send.keywords)
        required = sum(param.default is inspect.Parameter.empty for param in params)
        if not required <= given <= len(params):
            found.append(
                f"{send.where}: {send.verb} sends {given} args, "
                f"{verbs[send.verb].value}{tuple(names)} takes {required}..{len(params)}"
            )
    for verb in verbs:
        if verb.name not in sent:
            found.append(f"verb {verb.name} is never sent")
    return found


def engine_sends() -> list[Send]:
    root = PYTHON_ROOT / "cti"
    return [
        send
        for path in sorted(root.rglob("*.py"))
        for send in sends(path.read_text(), str(path.relative_to(root)))
    ]


def test_training_verbs_match_scheduler_and_senders() -> None:
    all_sends = engine_sends()
    assert all_sends, "found no training-verb sends to check"
    assert violations(TrainingVerb, TrainingScheduler, NON_COMMANDS, all_sends) == []


# ================================ Probes ================================


class _Verb(Enum):
    START = "start"
    SAVE = "save"


class _Scheduler:
    def handle(self, command) -> None: ...

    def start(self, command_id: int) -> None: ...

    def save(self, command_id: int, path: str) -> None: ...


_CLEAN = """
channel.call(TrainingVerb.START)
self._channel.call(protocol.TrainingVerb.SAVE, path)
"""


def _probe(source: str, verbs=_Verb, scheduler=_Scheduler, non_commands=("handle",)):
    found = violations(verbs, scheduler, frozenset(non_commands), list(sends(source, "probe")))
    return "\n".join(found)


def test_probe_clean_source_passes() -> None:
    assert _probe(_CLEAN) == ""


class _Unbacked(Enum):
    START = "start"
    SAVE = "save"
    LOAD = "load"


class _Unverbed(_Scheduler):
    def load(self, command_id: int) -> None: ...


@pytest.mark.parametrize(
    ("source", "kwargs", "expected"),
    [
        (_CLEAN + "c.call(TrainingVerb.LOAD)", {"verbs": _Unbacked}, "'load' names no scheduler"),
        (_CLEAN, {"scheduler": _Unverbed}, "scheduler.load is public but neither"),
        (_CLEAN, {"non_commands": ("handle", "gone")}, "non-command gone is no public"),
        (_CLEAN + "c.call(TrainingVerb.SAVE)", {}, "SAVE sends 0 args"),
        (_CLEAN + "c.call(TrainingVerb.START, 1)", {}, "START sends 1 args"),
        (_CLEAN + "c.call(TrainingVerb.SAVE, file=p)", {}, "unknown keywords ['file']"),
        (_CLEAN + "c.call(TrainingVerb.SAVE, *args)", {}, "SAVE forwards *args"),
        (_CLEAN + "c.call(TrainingVerb.STOP)", {}, "sends unknown verb STOP"),
        ("c.call(TrainingVerb.START)", {}, "verb SAVE is never sent"),
    ],
)
def test_probe_catches_mismatch(source: str, kwargs: dict, expected: str) -> None:
    assert expected in _probe(source, **kwargs)
