"""Read a tree of frozen dataclasses from a YAML document or from command-line flags.

A tree's fields are leaves or nested dataclasses. A document spells each dataclass as a
mapping keyed by field name; an absent key keeps the field's default, and a field typed
``X | None`` holds a section that exists only when its key does. A field whose metadata
sets ``inline`` spells its own fields at its parent's level. A field typed as a union of
dataclasses is one leaf whose mapping names its member by ``kind``, the snake_case class
name. The command line has exactly one flag per leaf, ``--`` and the field path with
underscores spelled as hyphens (``--model.max-seq-len``); its value is read as YAML, except
that a leaf whose every value is a string takes the text verbatim.

Nothing is renamed or aliased: a key is a field name, a flag is a field path. The tree is
the only type check of a document: each leaf is decoded strictly against its annotation.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import os
import re
import types
import typing
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import yaml

T = TypeVar("T")
_FieldPath = tuple[str, ...]


class _Loader(yaml.SafeLoader):
    """Safe YAML with string keys and no duplicate key; no global constructor registration."""

    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError(f"keys must be strings at {key_node.start_mark}")
            if key in keys:
                raise ValueError(f"duplicate key {key!r} at {key_node.start_mark}")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


@dataclasses.dataclass(frozen=True)
class _Leaf:
    """One leaf field of a tree: its path from the root and its type."""

    path: _FieldPath
    hint: Any
    text: bool

    @property
    def flag(self) -> str:
        return "--" + ".".join(part.replace("_", "-") for part in self.path)

    @property
    def dest(self) -> str:
        return ".".join(self.path)

    def parse(self, value: str) -> object:
        """The document value a flag's ``value`` spells."""

        return value if self.text else DataclassTree.parse_yaml(value)


class DataclassTree(Generic[T]):
    """The tree rooted at the dataclass ``root``. ``exclude`` names top-level sections that
    get no leaves, and so no flags; documents still spell them."""

    def __init__(self, root: type[T], *, exclude: Collection[str] = ()) -> None:
        self._root = root
        self.leaves = tuple(leaf for leaf in self._walk(root, ()) if leaf.path[0] not in exclude)

    @staticmethod
    def read_yaml(path: str | os.PathLike[str]) -> object:
        """The document at ``path``, read as strict YAML."""

        with Path(path).expanduser().open(encoding="utf-8") as stream:
            return yaml.load(stream, Loader=_Loader)

    @staticmethod
    def parse_yaml(text: str) -> object:
        """One YAML value, read as strictly as a document."""

        return yaml.load(text, Loader=_Loader)

    @staticmethod
    def kind(cls: type) -> str:
        """The ``kind`` that selects ``cls`` in a union: its snake_case name."""

        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", cls.__name__)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    def decode(self, document: object, overrides: Mapping[_FieldPath, object] | None = None) -> T:
        """The root ``document`` spells, with each override's value set at its path; missing
        or null sections on the way are created."""

        if overrides:
            document = self._overlay(document, overrides)
        return self._decode_dataclass(self._root, document, ())

    def add_flags(self, parser: argparse.ArgumentParser) -> None:
        """One flag per leaf; a flag given twice is an error. The parser must not accept
        abbreviated flags, which would alias them."""

        for leaf in self.leaves:
            parser.add_argument(
                leaf.flag,
                dest=leaf.dest,
                default=argparse.SUPPRESS,
                metavar="TEXT" if leaf.text else "YAML",
                action=_Once,
                help=f"{leaf.dest}: {_describe(leaf.hint)}",
            )

    def flag_values(self, namespace: argparse.Namespace) -> dict[_FieldPath, object]:
        """The document value of each leaf whose flag was given."""

        given = vars(namespace)
        return {
            leaf.path: leaf.parse(given[leaf.dest]) for leaf in self.leaves if leaf.dest in given
        }

    def _walk(self, cls: type, at: _FieldPath) -> list[_Leaf]:
        found = []
        for field, hint in _fields(cls):
            path = at if _inline(field) else (*at, field.name)
            section = _section(hint)
            if section is not None:
                found.extend(self._walk(section, path))
            else:
                found.append(_Leaf(path, hint, _is_text(hint)))
        return found

    def _overlay(self, document: object, values: Mapping[_FieldPath, object]) -> dict:
        if not isinstance(document, Mapping):
            raise ValueError(f"the document must be a mapping, got {_show(document)}")
        result = dict(document)
        for path, value in values.items():
            node = result
            for depth, key in enumerate(path[:-1]):
                child = node.get(key)
                if child is None:
                    child = {}
                elif not isinstance(child, Mapping):
                    raise ValueError(f"{'.'.join(path[: depth + 1])} must be a mapping")
                node[key] = node = dict(child)
            node[path[-1]] = value
        return result

    def _decode_dataclass(self, cls: type, document: object, at: _FieldPath) -> Any:
        if not isinstance(document, Mapping):
            raise ValueError(f"{_where(at)} must be a mapping, got {_show(document)}")
        unknown = sorted(set(document) - _keys(cls))
        if unknown:
            raise ValueError(f"{_where(at)} has unknown keys: {', '.join(unknown)}")
        return self._build(cls, document, at)

    def _build(self, cls: type, document: Mapping, at: _FieldPath) -> Any:
        values = {}
        for field, hint in _fields(cls):
            if _inline(field):
                values[field.name] = self._build(hint, document, at)
            elif field.name in document:
                values[field.name] = self._decode(hint, document[field.name], (*at, field.name))
            elif (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                raise ValueError(f"{_where((*at, field.name))} is required")
        try:
            return cls(**values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{_where(at)}: {error}") from error

    def _decode(self, hint: Any, value: object, at: _FieldPath) -> object:
        options = _union(hint)
        if options is not None:
            if value is None and types.NoneType in options:
                return None
            members = tuple(option for option in options if option is not types.NoneType)
            if len(members) == 1:
                return self._decode(members[0], value, at)
            return self._decode_member(members, value, at)
        if dataclasses.is_dataclass(hint):
            return self._decode_dataclass(hint, value, at)
        origin = typing.get_origin(hint)
        if origin is Literal:
            choices = typing.get_args(hint)
            if not any(type(value) is type(choice) and value == choice for choice in choices):
                raise ValueError(f"{_where(at)} must be one of {_choices(choices)}")
            return value
        if origin is tuple:
            return self._decode_tuple(typing.get_args(hint), value, at)
        if isinstance(hint, type) and issubclass(hint, enum.Enum):
            choices = tuple(member.value for member in hint)
            if not any(type(value) is type(choice) and value == choice for choice in choices):
                raise ValueError(f"{_where(at)} must be one of {_choices(choices)}")
            return hint(value)
        if hint is float:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{_where(at)} must be a number, got {_show(value)}")
            return float(value)
        if hint in (bool, int, str):
            if type(value) is not hint:
                raise ValueError(f"{_where(at)} must be {_describe(hint)}, got {_show(value)}")
            return value
        raise TypeError(f"{_where(at)} has a type the tree cannot read: {hint!r}")

    def _decode_member(self, members: tuple[type, ...], value: object, at: _FieldPath) -> object:
        if not all(dataclasses.is_dataclass(member) for member in members):
            raise TypeError(f"{_where(at)} unions types the tree cannot tell apart: {members!r}")
        by_kind = {self.kind(member): member for member in members}
        selected = value.get("kind") if isinstance(value, Mapping) else None
        if not isinstance(selected, str) or selected not in by_kind:
            raise ValueError(f"{_where(at)} must be a mapping with kind: {_choices(by_kind)}")
        fields = {key: item for key, item in value.items() if key != "kind"}
        return self._decode_dataclass(by_kind[selected], fields, at)

    def _decode_tuple(self, items: tuple[Any, ...], value: object, at: _FieldPath) -> tuple:
        if not isinstance(value, list | tuple):
            raise ValueError(f"{_where(at)} must be a list, got {_show(value)}")
        if len(items) == 2 and items[1] is Ellipsis:
            items = (items[0],) * len(value)
        elif len(value) != len(items):
            raise ValueError(f"{_where(at)} must have {len(items)} items, got {len(value)}")
        return tuple(
            self._decode(hint, item, (*at[:-1], f"{at[-1]}[{index}]"))
            for index, (hint, item) in enumerate(zip(items, value, strict=True))
        )


class _Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if self.dest in vars(namespace):
            parser.error(f"{option_string} given more than once")
        vars(namespace)[self.dest] = values


def _fields(cls: type) -> tuple[tuple[dataclasses.Field, Any], ...]:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    return tuple((field, hints[field.name]) for field in dataclasses.fields(cls) if field.init)


def _inline(field: dataclasses.Field) -> bool:
    return bool(field.metadata.get("inline"))


def _keys(cls: type) -> set[str]:
    """The keys a document spelling ``cls`` may hold, inline fields' keys included."""

    keys = set()
    for field, hint in _fields(cls):
        keys |= _keys(hint) if _inline(field) else {field.name}
    return keys


def _union(hint: Any) -> tuple[Any, ...] | None:
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return typing.get_args(hint)
    return None


def _section(hint: Any) -> type | None:
    """The dataclass a field nests, alone or optional; None for a leaf."""

    options = _union(hint)
    if options is not None:
        members = tuple(option for option in options if option is not types.NoneType)
        hint = members[0] if len(members) == 1 else None
    return hint if dataclasses.is_dataclass(hint) else None


def _is_text(hint: Any) -> bool:
    """Whether a flag passes its text through: every value the leaf takes is a string, so
    none is null."""

    for option in _union(hint) or (hint,):
        if typing.get_origin(option) is Literal:
            if not all(isinstance(choice, str) for choice in typing.get_args(option)):
                return False
        elif not (isinstance(option, type) and issubclass(option, str)):
            return False
    return True


def _describe(hint: Any) -> str:
    names = {bool: "a boolean", int: "an integer", float: "a number", str: "a string"}
    if hint in names:
        return names[hint]
    if isinstance(hint, type):
        return hint.__name__
    return str(hint).replace("typing.", "")


def _choices(choices) -> str:
    return ", ".join(repr(choice) for choice in choices)


def _where(at: _FieldPath) -> str:
    return ".".join(at) if at else "the document"


def _show(value: object) -> str:
    return f"{type(value).__name__} {value!r}"


__all__ = ["DataclassTree"]
