"""Compact fixtures of real quantized checkpoints: their documents and tensor headers.

A fixture file holds each JSON document the recipes read and one line per family of
tensor headers. Dotted name components that are integers compact into ``{a-b,c}``
ranges; a long name list becomes ``{"$names": [...]}`` and a long module-keyed map
``{"$modules": [[name, value], ...]}``. Such lists are sets to every recipe, so the
fixture keeps their members, not their order; everything else round-trips exactly. ``python -m tests.support.quant_checkpoints``
rewrites every fixture from the Hub at its pinned revision (network).
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

FIXTURES = Path(__file__).parent / "checkpoints"
# Fixture stem -> (repo, revision, documents read beside the headers).
CHECKPOINTS = MappingProxyType(
    {
        "qwen3_8b_fp8": ("Qwen/Qwen3-8B-FP8", "220b46e3b2180893580a4454f21f22d3ebb187d3", ()),
        "qwen3_8_27b_fp8": ("Qwen/Qwen3.8-27B-FP8", "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a", ()),
        "qwen3_8_27b_nvfp4": (
            "nvidia/Qwen3.8-27B-NVFP4",
            "482ca0f3832238542f8f5295dde86b5f22711d80",
            ("hf_quant_config.json",),
        ),
        "qwen3_5_35b_a3b_fp8": (
            "Qwen/Qwen3.5-35B-A3B-FP8",
            "9d1823d2dee688a6b25e77009dc727688c44936e",
            (),
        ),
        "qwen3_30b_a3b_fp8": (
            "Qwen/Qwen3-30B-A3B-FP8",
            "d206ba732169f29bb77fbf80fc2c4b81d4d30782",
            (),
        ),
    }
)
_MIN_COMPACT = 8
_INDEX = re.compile(r"^(\d+)(\*?)$")
_RANGES = re.compile(r"\{([0-9,-]+)\}")


def load(stem: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """``(documents, headers)``: each JSON document by file name, and each tensor's header
    (``dtype`` tag and ``shape``) by name, as the checkpoint has them."""

    documents: dict[str, dict[str, Any]] = {}
    headers: dict[str, dict[str, Any]] = {}
    section, lines = "", []
    text = (FIXTURES / f"{stem}.txt").read_text(encoding="utf-8")
    for line in [*text.splitlines(), "[end]"]:
        if line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            if section and section != "tensors":
                documents[section] = _expand_document(json.loads("\n".join(lines)))
            section, lines = line[1:-1], []
        elif section == "tensors" and line:
            header, pattern = line.split(" ", 1)
            dtype, shape = header.split("[", 1)
            for name in expand(pattern):
                headers[name] = {"dtype": dtype, "shape": json.loads(f"[{shape}")}
        else:
            lines.append(line)
    return documents, headers


def write_headers_only(stem: str, folder: Path) -> None:
    """``folder`` as the checkpoint looks to resolution: its documents and one safetensors
    file holding every tensor's header and no data."""

    documents, headers = load(stem)
    folder.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (folder / name).write_text(json.dumps(document), encoding="utf-8")
    header = json.dumps(
        {name: {**entry, "data_offsets": [0, 0]} for name, entry in headers.items()}
    ).encode()
    (folder / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)


def expand(pattern: str) -> list[str]:
    """Every name a ``{a-b,c}`` pattern spells, in index order."""

    choices = [_indices(spec) for spec in _RANGES.findall(pattern)]
    template = _RANGES.sub("{}", pattern)
    return [template.format(*combo) for combo in itertools.product(*choices)]


def compact(names: Iterable[str]) -> list[str]:
    """The fewest ``{a-b,c}`` patterns that expand to exactly ``names``."""

    families: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for name in names:
        template, indices = _template(name)
        families[template].add(indices)
    patterns = []
    for template, tuples in sorted(families.items()):
        for product in _products(tuples):
            patterns.append(template.format(*("{" + _ranges(sorted(s)) + "}" for s in product)))
    return patterns


def _indices(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        out.extend(range(int(first), int(last or first) + 1))
    return out


def _template(name: str) -> tuple[str, tuple[int, ...]]:
    parts, indices = [], []
    for part in name.split("."):
        match = _INDEX.match(part)
        if match:
            indices.append(int(match[1]))
        parts.append("{}" + match[2] if match else part)
    return ".".join(parts), tuple(indices)


def _products(tuples: set[tuple[int, ...]]) -> list[tuple[frozenset[int], ...]]:
    """Split a set of index tuples into products of per-position index sets."""

    if not tuples or not next(iter(tuples)):
        return [()] if tuples else []
    heads: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for item in tuples:
        heads[item[1:]].add(item[0])
    rests: dict[frozenset[int], set[tuple[int, ...]]] = defaultdict(set)
    for rest, firsts in heads.items():
        rests[frozenset(firsts)].add(rest)
    return [
        (firsts, *product)
        for firsts, group in sorted(rests.items(), key=lambda item: min(item[0]))
        for product in _products(group)
    ]


def _ranges(values: list[int]) -> str:
    runs = []
    for _, group in itertools.groupby(enumerate(values), lambda pair: pair[1] - pair[0]):
        run = [value for _, value in group]
        runs.append(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}")
    return ",".join(runs)


def _expand_document(node: Any) -> Any:
    if isinstance(node, dict):
        if set(node) == {"$names"}:
            return [name for pattern in node["$names"] for name in expand(pattern)]
        if set(node) == {"$modules"}:
            return {
                name: _expand_document(value)
                for pattern, value in node["$modules"]
                for name in expand(pattern)
            }
        return {key: _expand_document(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand_document(value) for value in node]
    return node


def _compact_document(node: Any) -> Any:
    if isinstance(node, dict):
        if _modules(list(node)):
            groups: dict[str, list[str]] = defaultdict(list)
            for key, value in node.items():
                groups[json.dumps(value, sort_keys=True)].append(key)
            return {
                "$modules": [
                    [pattern, json.loads(value)]
                    for value, keys in sorted(groups.items())
                    for pattern in compact(keys)
                ]
            }
        return {key: _compact_document(value) for key, value in node.items()}
    if isinstance(node, list):
        if _modules(node):
            return {"$names": compact(node)}
        return [_compact_document(value) for value in node]
    return node


def _modules(node: list[Any]) -> bool:
    """Whether ``node`` is a long list of distinct module names (mostly dotted)."""

    strings = all(isinstance(item, str) for item in node)
    dotted = strings and 2 * sum("." in item for item in node) >= len(node)
    return len(node) >= _MIN_COMPACT and dotted and len(set(node)) == len(node)


def _dump(node: Any, pad: str = "", *, split: bool = False) -> str:
    """JSON with a line per mapping entry and per item of a ``$names`` list or of a list
    holding containers; other lists stay inline."""

    inner = pad + " "
    if isinstance(node, dict) and node:
        items = [
            f"{inner}{json.dumps(key)}: {_dump(value, inner, split=key == '$names')}"
            for key, value in node.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(node, list) and (split or any(isinstance(i, dict | list) for i in node)):
        return "[\n" + ",\n".join(inner + _dump(item, inner) for item in node) + f"\n{pad}]"
    return json.dumps(node)


def _canonical(node: Any) -> Any:
    """Order-free form of a document: compacted lists compare as sets."""

    if isinstance(node, dict):
        return {key: _canonical(value) for key, value in node.items()}
    if isinstance(node, list):
        items = [_canonical(value) for value in node]
        return sorted(items) if _modules(node) else items
    return node


def render(
    source: str,
    documents: Mapping[str, Mapping[str, Any]],
    headers: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = [
        f"# {source}: documents and safetensors headers, compacted by",
        "# tests/support/quant_checkpoints.py",
    ]
    for name, document in documents.items():
        lines += [f"[{name}]", _dump(_compact_document(document))]
    families: dict[str, list[str]] = defaultdict(list)
    for name, header in headers.items():
        shape = ",".join(map(str, header["shape"]))
        families[f"{header['dtype']}[{shape}]"].append(name)
    lines.append("[tensors]")
    lines += sorted(
        f"{header} {pattern}" for header, names in families.items() for pattern in compact(names)
    )
    return "\n".join(lines) + "\n"


def _fetch(stem: str) -> None:
    from huggingface_hub import get_safetensors_metadata, hf_hub_download

    repo, revision, extra = CHECKPOINTS[stem]
    documents = {}
    for name in ("config.json", *extra):
        path = hf_hub_download(repo, name, revision=revision)
        documents[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = get_safetensors_metadata(repo, revision=revision)
    headers = {
        name: {"dtype": info.dtype, "shape": list(info.shape)}
        for file in metadata.files_metadata.values()
        for name, info in file.tensors.items()
    }
    (FIXTURES / f"{stem}.txt").write_text(render(f"{repo}@{revision}", documents, headers))
    loaded_documents, loaded_headers = load(stem)
    assert loaded_headers == headers, stem
    assert _canonical(loaded_documents) == _canonical(documents), stem


if __name__ == "__main__":
    FIXTURES.mkdir(exist_ok=True)
    for stem in sys.argv[1:] or CHECKPOINTS:
        _fetch(stem)
