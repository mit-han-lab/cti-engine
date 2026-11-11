from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import pytest

from cti.utils.dataclass_tree import DataclassTree


class Mode(StrEnum):
    FAST = "fast"
    SAFE = "safe"


@dataclass(frozen=True, kw_only=True)
class Seeded:
    seed: int


@dataclass(frozen=True, kw_only=True)
class Loaded:
    pass


@dataclass(frozen=True, kw_only=True)
class Inner:
    name: str
    ratio: float = 0.5
    sizes: tuple[int, ...] = ()
    pair: tuple[float, float] = (0.9, 0.99)
    source: Seeded | Loaded = Loaded()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")


@dataclass(frozen=True, kw_only=True)
class Limits:
    count: int = 4
    max_count: int = 8
    enabled: bool = True
    mode: Mode = Mode.SAFE
    prefix: Literal["auto", "radix"] = "auto"
    cap: int | None = None
    label: str | None = "default"


@dataclass(frozen=True, kw_only=True)
class Root:
    inner: Inner
    limits: Limits = field(default_factory=Limits)
    extra: Inner | None = None


@dataclass(frozen=True)
class Launch:
    root: Root = field(metadata={"inline": True})
    server: Limits = Limits()


def decode(cls: type, document: object, overrides: dict | None = None) -> object:
    return DataclassTree(cls).decode(document, overrides)


def test_nested_sections_take_defaults_for_absent_keys() -> None:
    root = decode(Root, {"inner": {"name": "x"}, "limits": {"count": 8}})

    assert root == Root(inner=Inner(name="x"), limits=Limits(count=8))
    launch = decode(Launch, {"inner": {"name": "x"}, "server": {"count": 2}})
    assert launch == Launch(root=Root(inner=Inner(name="x")), server=Limits(count=2))


def test_optional_section_exists_only_when_its_key_does() -> None:
    assert decode(Root, {"inner": {"name": "x"}}).extra is None
    assert decode(Root, {"inner": {"name": "x"}, "extra": None}).extra is None
    assert decode(Root, {"inner": {"name": "x"}, "extra": {"name": "y"}}).extra == Inner(name="y")


def test_union_members_are_chosen_by_kind() -> None:
    assert DataclassTree.kind(Seeded) == "seeded"
    assert DataclassTree.kind(argparse.ArgumentParser) == "argument_parser"

    inner = decode(Inner, {"name": "x", "source": {"kind": "seeded", "seed": 7}})

    assert inner.source == Seeded(seed=7)
    assert decode(Inner, {"name": "x", "source": {"kind": "loaded"}}).source == Loaded()
    with pytest.raises(ValueError, match=r"source has unknown keys: seed"):
        decode(Inner, {"name": "x", "source": {"kind": "loaded", "seed": 7}})


@pytest.mark.parametrize("source", ({"seed": 7}, {"kind": []}, {"kind": {}}, "seeded"))
def test_a_union_without_a_known_kind_names_its_path(source: object) -> None:
    with pytest.raises(ValueError, match=r"source must be a mapping with kind: 'seeded', 'loaded'"):
        decode(Inner, {"name": "x", "source": source})


def test_enums_and_literals_take_their_values() -> None:
    limits = decode(Limits, {"mode": "fast", "prefix": "radix"})

    assert limits.mode is Mode.FAST
    assert limits.prefix == "radix"
    with pytest.raises(ValueError, match=r"mode must be one of 'fast', 'safe'"):
        decode(Limits, {"mode": "slow"})
    with pytest.raises(ValueError, match=r"prefix must be one of 'auto', 'radix'"):
        decode(Limits, {"prefix": "naive"})


def test_tuples_have_fixed_or_variable_length() -> None:
    inner = decode(Inner, {"name": "x", "sizes": [1, 2], "pair": [0.8, 1]})

    assert inner.sizes == (1, 2)
    assert inner.pair == (0.8, 1.0)
    assert decode(Inner, {"name": "x", "sizes": []}).sizes == ()
    with pytest.raises(ValueError, match=r"pair must have 2 items, got 3"):
        decode(Inner, {"name": "x", "pair": [1, 2, 3]})
    with pytest.raises(ValueError, match=r"sizes\[1\] must be an integer, got str 'b'"):
        decode(Inner, {"name": "x", "sizes": [1, "b"]})


@pytest.mark.parametrize(
    ("document", "message"),
    (
        ({"count": True}, r"count must be an integer, got bool True"),
        ({"count": 1.0}, r"count must be an integer, got float 1.0"),
        ({"enabled": 1}, r"enabled must be a boolean, got int 1"),
        ({"cap": "3"}, r"cap must be an integer, got str '3'"),
        ({"label": 3}, r"label must be a string, got int 3"),
    ),
)
def test_booleans_integers_and_strings_are_strict(document: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode(Limits, document)


def test_numbers_take_integers_but_not_booleans() -> None:
    assert decode(Inner, {"name": "x", "ratio": 1}).ratio == 1.0
    with pytest.raises(ValueError, match=r"ratio must be a number, got bool True"):
        decode(Inner, {"name": "x", "ratio": True})


@pytest.mark.parametrize(
    ("document", "message"),
    (
        ({"inner": {"name": "x"}, "inners": {}}, r"the document has unknown keys: inners"),
        ({"inner": {"name": "x", "label": "y"}}, r"inner has unknown keys: label"),
        ({}, r"inner is required"),
        ({"inner": {}}, r"inner.name is required"),
        ({"inner": None}, r"inner must be a mapping, got NoneType None"),
        ({"inner": {"name": ""}}, r"inner: name must not be empty"),
        ("inner", r"the document must be a mapping, got str 'inner'"),
    ),
)
def test_errors_name_the_path(document: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode(Root, document)


@dataclass(frozen=True)
class Unreadable:
    names: frozenset[str] = frozenset()
    either: int | str = 0


def test_types_the_tree_cannot_read_are_refused() -> None:
    with pytest.raises(TypeError, match=r"names has a type the tree cannot read"):
        decode(Unreadable, {"names": ["a"]})
    with pytest.raises(TypeError, match=r"either unions types the tree cannot tell apart"):
        decode(Unreadable, {"either": 1})


def test_documents_reject_duplicate_and_non_string_keys(tmp_path) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text("inner:\n  name: x\n  name: y\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"duplicate key 'name'"):
        DataclassTree.read_yaml(path)
    with pytest.raises(ValueError, match=r"keys must be strings"):
        DataclassTree.parse_yaml("{1: x}")


def test_each_leaf_has_exactly_one_flag() -> None:
    assert [leaf.flag for leaf in DataclassTree(Root).leaves] == [
        "--inner.name",
        "--inner.ratio",
        "--inner.sizes",
        "--inner.pair",
        "--inner.source",
        "--limits.count",
        "--limits.max-count",
        "--limits.enabled",
        "--limits.mode",
        "--limits.prefix",
        "--limits.cap",
        "--limits.label",
        "--extra.name",
        "--extra.ratio",
        "--extra.sizes",
        "--extra.pair",
        "--extra.source",
    ]
    flags = [leaf.flag for leaf in DataclassTree(Launch, exclude=("extra",)).leaves]
    assert (flags[0], flags[11], flags[12]) == ("--inner.name", "--limits.label", "--server.count")


def _parse(argv: list[str]) -> dict:
    tree = DataclassTree(Root)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    tree.add_flags(parser)
    return tree.flag_values(parser.parse_args(argv))


def test_flags_are_read_as_yaml_and_override_the_document() -> None:
    values = _parse(
        [
            "--limits.count",
            "16",
            "--inner.sizes",
            "[]",
            "--inner.source",
            "{kind: seeded, seed: 3}",
            "--limits.label",
            "null",
        ]
    )

    root = decode(Root, {"inner": {"name": "x", "sizes": [1, 2]}}, values)

    assert root.limits == Limits(count=16, label=None)
    assert root.inner == Inner(name="x", sizes=(), source=Seeded(seed=3))


def test_a_flag_creates_its_missing_or_null_section() -> None:
    root = decode(Root, {"inner": {"name": "x"}, "extra": None}, _parse(["--extra.name", "y"]))

    assert root.extra == Inner(name="y")
    with pytest.raises(ValueError, match=r"extra.name is required"):
        decode(Root, {"inner": {"name": "x"}}, _parse(["--extra.ratio", "0.25"]))


def test_only_leaves_that_are_never_null_take_the_flag_verbatim() -> None:
    values = _parse(
        [
            "--inner.name",
            "123",
            "--limits.mode",
            "fast",
            "--limits.prefix",
            "auto",
            "--limits.label",
            "'null'",
        ]
    )

    assert values == {
        ("inner", "name"): "123",
        ("limits", "mode"): "fast",
        ("limits", "prefix"): "auto",
        ("limits", "label"): "null",
    }


def test_a_flag_spells_underscores_as_hyphens_only() -> None:
    assert _parse(["--limits.max-count", "16"]) == {("limits", "max_count"): 16}
    with pytest.raises(SystemExit):
        _parse(["--limits.max_count", "16"])


def test_a_flag_given_twice_or_abbreviated_is_an_error() -> None:
    with pytest.raises(SystemExit):
        _parse(["--limits.count", "1", "--limits.count", "2"])
    with pytest.raises(SystemExit):
        _parse(["--limits.coun", "1"])


def test_an_override_refuses_to_replace_a_value_with_a_section() -> None:
    with pytest.raises(ValueError, match=r"inner must be a mapping"):
        decode(Root, {"inner": "x"}, {("inner", "name"): "y"})
