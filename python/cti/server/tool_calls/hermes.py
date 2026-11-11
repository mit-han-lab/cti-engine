"""Hermes calls (Qwen3): ``<tool_call>{"name": NAME, "arguments": {...}}</tool_call>``."""

from __future__ import annotations

import json
import re

from .._text import held_tail
from . import CallDelta, ToolCallParser

# The call object up to its arguments' first character; its name is a JSON string.
_HEAD = re.compile(r'\s*\{\s*"name"\s*:\s*("(?:[^"\\]|\\.)*")\s*,\s*"arguments"\s*:\s*(?=\S)')


class HermesParser(ToolCallParser):
    """A call's name goes out once its head is complete, then its arguments' JSON text as it
    arrives, minus a tail that may be the call object's closing brace or the end tag."""

    def _open_call(self) -> None:
        self._named = False

    def _body(self, calls: list[CallDelta]) -> bool | None:
        if not self._named:
            head = _HEAD.match(self._pending)
            if head is None:
                return False if self.CALL_END in self._pending else None
            name = json.loads(head.group(1))
            if name not in self._tools:
                return False
            self._emit(calls, name, "")
            self._named = True
            self._pending = self._pending[head.end() :]
        end = self._pending.find(self.CALL_END)
        if end >= 0:
            arguments = self._pending[:end].rstrip()
            if arguments.endswith("}"):
                arguments = arguments[:-1].rstrip()  # the call object's own brace
            if arguments:
                self._emit(calls, None, arguments)
            self._pending = self._pending[end + len(self.CALL_END) :]
            return True
        released = self._open_end(self._pending)
        if released:
            self._emit(calls, None, self._pending[:released])
            self._pending = self._pending[released:]
        return None

    def _open_end(self, text: str) -> int:
        """Where the text that may still end the arguments begins: a start of the end tag,
        the blanks before it, and a brace that may be the call object's."""

        rest = text[: len(text) - held_tail(text, (self.CALL_END,))].rstrip()
        return len(rest) - 1 if rest.endswith("}") else len(rest)
