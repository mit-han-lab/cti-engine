# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/message/utils.py`); see THIRD_PARTY_NOTICES.

"""The protocol's typed codec. A message family encodes a message as ``[index, *fields]``,
each field by its declared type: a value msgpack carries as it is stays as it is, a tuple is
rebuilt from its declared shape, an enum travels as its value, a nested dataclass as its own
fields, a tensor as its dtype, shape and bytes. Every codec is built once from the class's
type hints, by the family that carries the class, so encoding names no field and walks no
value it need not."""

from __future__ import annotations

import dataclasses
import enum
import sys
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

# A codec half: None when msgpack carries the value as it is.
_Half = Callable[[Any], Any] | None


class _Codec(NamedTuple):
    encode: _Half
    decode: _Half


_PLAIN = _Codec(None, None)
_SCALARS = (int, float, str, bool, bytes, types.NoneType)


def _apply(half: _Half, value: Any) -> Any:
    return value if half is None else half(value)


def _imported_for_type_checking(name: str) -> Any:
    """A name a protocol annotation uses but its module imports only for type checking:
    torch, which only a process that holds tensors loads, or a multimodal or sampling type."""

    if name == "torch":
        import torch

        return torch
    from cti import multimodal, sampling

    for module in (multimodal, sampling):
        if hasattr(module, name):
            return getattr(module, name)
    raise NameError(name)


def _hints(cls: type) -> tuple[tuple[str, Any], ...]:
    """``cls``'s constructor fields and their resolved types, in order."""

    module, names = sys.modules[cls.__module__], {}
    while True:
        try:
            hints = typing.get_type_hints(cls, globalns=vars(module), localns=names)
            break
        except NameError as missing:
            if missing.name in names:
                raise
            names[missing.name] = _imported_for_type_checking(missing.name)
    return tuple((field.name, hints[field.name]) for field in dataclasses.fields(cls) if field.init)


def _tensor() -> _Codec:
    def encode(tensor: Any) -> list[Any]:
        torch = sys.modules["torch"]
        tensor = tensor.detach().contiguous()
        # Raw bytes keep every dtype bit-exact, bfloat16 included.
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return [str(tensor.dtype).removeprefix("torch."), list(tensor.shape), raw]

    def decode(data: Sequence[Any]) -> Any:
        import torch

        dtype_name, shape, raw = data
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Cannot decode tensor dtype {dtype_name!r}")
        if not raw:
            return torch.empty(shape, dtype=dtype)
        # bytearray gives the tensor writable memory of its own.
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(shape)

    return _Codec(encode, decode)


def _sequence(item: _Codec, build: type) -> _Codec:
    encode = None if item.encode is None else lambda v: [item.encode(x) for x in v]
    if item.decode is None:
        return _Codec(encode, build)
    return _Codec(encode, lambda data: build(item.decode(x) for x in data))


def _fixed_tuple(items: tuple[_Codec, ...]) -> _Codec:
    def encode(value: Sequence[Any]) -> list[Any]:
        return [_apply(c.encode, x) for c, x in zip(items, value, strict=True)]

    def decode(data: Sequence[Any]) -> tuple[Any, ...]:
        return tuple(_apply(c.decode, x) for c, x in zip(items, data, strict=True))

    return _Codec(encode, decode)


def _optional(inner: _Codec) -> _Codec:
    if inner == _PLAIN:
        return _PLAIN
    return _Codec(
        None if inner.encode is None else lambda v: None if v is None else inner.encode(v),
        None if inner.decode is None else lambda d: None if d is None else inner.decode(d),
    )


class Family:
    """A link's message classes: a message encodes as ``[its index, *its fields]``. A class's
    codec is built when the family first codes one, so a process loads only what its link
    carries. An open base in ``open_bases`` travels as ``[class name, *fields]``, its
    implementation found by name as the resolver returns it."""

    def __init__(
        self,
        classes: Sequence[type],
        *,
        open_bases: Mapping[type, Callable[[str], type]] | None = None,
    ) -> None:
        self._classes = tuple(classes)
        self._index = {cls: index for index, cls in enumerate(self._classes)}
        self._open = dict(open_bases or {})
        self._codecs: dict[Any, _Codec] = {}
        self._fields: dict[type, tuple[tuple[str, _Codec], ...]] = {}

    def encode(self, message: Any) -> list[Any]:
        cls = type(message)
        try:
            index = self._index[cls]
        except KeyError:
            raise TypeError(f"{cls.__name__} is not a message of this link") from None
        return [index, *self._encode_fields(cls, message)]

    def decode(self, data: Sequence[Any]) -> Any:
        return self._decode_fields(self._classes[data[0]], data[1:])

    def codec(self, hint: Any) -> _Codec:
        """The codec of values of the declared type ``hint``."""

        built = self._codecs.get(hint)
        if built is None:
            built = self._codecs[hint] = self._build(hint)
        return built

    def _field_codecs(self, cls: type) -> tuple[tuple[str, _Codec], ...]:
        fields = self._fields.get(cls)
        if fields is None:
            fields = self._fields[cls] = tuple(
                (name, self.codec(hint)) for name, hint in _hints(cls)
            )
        return fields

    def _encode_fields(self, cls: type, value: Any) -> list[Any]:
        return [_apply(c.encode, getattr(value, name)) for name, c in self._field_codecs(cls)]

    def _decode_fields(self, cls: type, data: Sequence[Any]) -> Any:
        fields = self._field_codecs(cls)
        if len(data) != len(fields):
            raise ValueError(f"{cls.__name__} has {len(fields)} fields, the wire {len(data)}")
        return cls(
            **{name: _apply(c.decode, item) for (name, c), item in zip(fields, data, strict=True)}
        )

    def _build(self, hint: Any) -> _Codec:
        origin, args = typing.get_origin(hint), typing.get_args(hint)
        if hint is Any or hint in _SCALARS or origin is typing.Literal:
            return _PLAIN
        if origin in (typing.Union, types.UnionType):
            options = tuple(arg for arg in args if arg is not types.NoneType)
            inner = self.codec(options[0]) if len(options) == 1 else self._union(options)
            return _optional(inner) if len(options) < len(args) else inner
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return _sequence(self.codec(args[0]), tuple)
            return _fixed_tuple(tuple(self.codec(arg) for arg in args))
        if origin is list:
            return _sequence(self.codec(args[0]), list) if args else _PLAIN
        if origin in (dict, Mapping) or hint in (dict, list) or typing.is_typeddict(hint):
            return _PLAIN  # JSON-like: msgpack carries it, keys included
        if isinstance(hint, type) and issubclass(hint, enum.Enum):
            return _Codec(lambda v: v.value, hint)
        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            return _Codec(
                lambda v: self._encode_fields(hint, v), lambda d: self._decode_fields(hint, d)
            )
        if isinstance(hint, type) and hint in self._open:
            resolve = self._open[hint]
            return _Codec(
                lambda v: [type(v).__name__, *self._encode_fields(type(v), v)],
                lambda d: self._decode_fields(resolve(d[0]), d[1:]),
            )
        torch = sys.modules.get("torch")
        if torch is not None and hint is torch.Tensor:
            return _tensor()
        raise TypeError(f"no codec for {hint!r}")

    def _union(self, options: tuple[Any, ...]) -> _Codec:
        """A union of dataclasses travels as ``[index, *fields]`` by its place in the union."""

        if all(isinstance(o, type) and dataclasses.is_dataclass(o) for o in options):
            index = {cls: i for i, cls in enumerate(options)}
            return _Codec(
                lambda v: [index[type(v)], *self._encode_fields(type(v), v)],
                lambda d: self._decode_fields(options[d[0]], d[1:]),
            )
        if all(self.codec(o) == _PLAIN for o in options):
            return _PLAIN
        raise TypeError(f"no codec for the union {options!r}")


__all__ = ["Family"]
