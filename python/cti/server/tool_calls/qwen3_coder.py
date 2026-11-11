"""Qwen3-Coder calls (Qwen3.5): ``<tool_call><function=NAME><parameter=KEY>VALUE</parameter>…
</function></tool_call>``, each tag on its own line."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from . import CallDelta, ToolCallParser

_FUNCTION = re.compile(r"\s*<function=([^>\n]+)>")
_PARAMETER = re.compile(r"\s*<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)
_FUNCTION_END = re.compile(r"\s*</function>\s*")


class Qwen3CoderParser(ToolCallParser):
    """A call's name goes out with the arguments' opening brace, each parameter whole once
    its end tag arrives, typed by the tool's JSON schema, and the brace at ``</function>``."""

    def _open_call(self) -> None:
        self._function: Mapping[str, Any] | None = None
        self._parameters = 0

    def _body(self, calls: list[CallDelta]) -> bool | None:
        if self._function is None:
            head = _FUNCTION.match(self._pending)
            if head is None:
                return False if self.CALL_END in self._pending else None
            name = head.group(1).strip()
            if name not in self._tools:
                return False
            self._function = self._tools[name]
            self._emit(calls, name, "{")
            self._pending = self._pending[head.end() :]
        while parameter := _PARAMETER.match(self._pending):
            key = parameter.group(1).strip()
            value = json.dumps(self._typed(key, _unwrap(parameter.group(2))), ensure_ascii=False)
            separator = ", " if self._parameters else ""
            self._emit(calls, None, f"{separator}{json.dumps(key, ensure_ascii=False)}: {value}")
            self._parameters += 1
            self._pending = self._pending[parameter.end() :]
        if end := _FUNCTION_END.match(self._pending):
            if self.CALL_END not in self._pending[end.end() :]:
                return None
            self._emit(calls, None, "}")
            after = self._pending.index(self.CALL_END) + len(self.CALL_END)
            self._pending = self._pending[after:]
            return True
        return None

    def _typed(self, key: str, value: str) -> object:
        """``value`` as the type the tool's schema gives ``key``; a string when it has none
        or the value does not parse as it."""

        schema = (self._function or {}).get("parameters") or {}
        kind = (schema.get("properties") or {}).get(key, {}).get("type", "string")
        kinds = kind if isinstance(kind, list) else [kind]
        for kind in kinds:
            if kind == "string":
                return value
            try:
                if kind == "integer":
                    return int(value)
                if kind == "number":
                    number = float(value)
                    return int(number) if number.is_integer() and "." not in value else number
                if kind == "boolean" and value.lower() in ("true", "false"):
                    return value.lower() == "true"
                if kind == "null" and value.lower() in ("null", "none"):
                    return None
                if kind in ("object", "array"):
                    return json.loads(value)
            except ValueError:
                continue
        return value


def _unwrap(value: str) -> str:
    """A parameter's value without the newline the template puts on each side of it."""

    value = value[1:] if value.startswith("\n") else value
    return value[:-1] if value.endswith("\n") else value
