"""Tool calls in a chat completion's text, as the model's chat template formats them.

A parser takes a choice's visible text delta by delta (``feed``) and, at its end, ``finish``;
a whole response is one ``feed`` and the ``finish``, so streamed and whole responses parse
alike. Text outside calls passes through as content. Inside a ``<think>`` block only a
``<tool_call>`` at the start of a line is a call, ending the thinking, as SGLang's Qwen3
reasoning parser takes it; a call that names no requested tool stays content.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, get_args, get_type_hints

from cti.config import ServerOptions

from .._text import held_tail

_THINK = "<think>"
_THINK_END = "</think>"
_CALL = "<tool_call>"
_CALL_END = "</tool_call>"


@dataclass(frozen=True, slots=True)
class CallDelta:
    """A part of a choice's ``index``-th call: its name on the first, then argument text."""

    index: int
    name: str | None
    arguments: str


@dataclass(frozen=True, slots=True)
class ParseDelta:
    content: str = ""
    calls: tuple[CallDelta, ...] = ()


class ToolCallParser:
    """The text around calls; a format parses each call's body (``_body``).

    ``_pending`` holds the text not yet released: outside a call, a tail that may begin a
    tag; inside one, its body so far."""

    TAGS: ClassVar[tuple[str, ...]] = (_THINK, _THINK_END, _CALL)
    CALL_END: ClassVar[str] = _CALL_END

    def __init__(self, tools: Sequence[Mapping[str, Any]], *, in_think: bool = False) -> None:
        """``in_think``: the prompt ends inside an open ``<think>``, so the text begins in
        thought."""

        self._tools = {tool["function"]["name"]: tool["function"] for tool in tools}
        self._pending = ""
        self._in_call = False
        self._started = False  # some part of the current call went out
        self._in_think = in_think
        self._line_start = True
        self._calls = 0

    def feed(self, text: str) -> ParseDelta:
        self._pending += text
        content: list[str] = []
        calls: list[CallDelta] = []
        while True:
            if self._in_call:
                if not self._call(content, calls):
                    break
            elif not self._outside(content):
                break
        return self._delta(content, calls)

    def finish(self) -> ParseDelta:
        """What the text's end releases: the held tail, and an unfinished call's raw text
        if no part of it went out."""

        content: list[str] = []
        if not self._in_call:
            self._release(content, self._pending)
        elif not self._started:
            self._release(content, _CALL + self._pending)
        self._pending = ""
        self._in_call = False
        return self._delta(content, [])

    def _outside(self, content: list[str]) -> bool:
        """Release text up to the next tag; False when the rest must wait for more."""

        text = self._pending
        start = 0
        while True:
            found = [(text.find(tag, start), tag) for tag in self.TAGS]
            found = [(at, tag) for at, tag in found if at >= 0]
            if not found:
                held = held_tail(text, self.TAGS)
                self._release(content, text[: len(text) - held])
                self._pending = text[len(text) - held :]
                return False
            at, tag = min(found)
            if tag == _CALL and self._in_think and not self._at_line_start(text, at):
                start = at + len(tag)  # a mention in thought, not a call
                continue
            break
        if tag == _CALL:
            self._release(content, text[:at])
            self._pending = text[at + len(tag) :]
            self._in_call, self._in_think, self._started = True, False, False
            self._open_call()
            return True
        self._release(content, text[: at + len(tag)])
        self._pending = text[at + len(tag) :]
        self._in_think = tag == _THINK
        return True

    def _call(self, content: list[str], calls: list[CallDelta]) -> bool:
        """Parse the call body so far; False when it needs more text."""

        ended = self._body(calls)
        if ended is None:
            return False
        if ended is False:  # not a call of a requested tool: its text stays content
            end = self._pending.find(_CALL_END)
            if end < 0:
                return False
            self._release(content, _CALL + self._pending[: end + len(_CALL_END)])
            self._pending = self._pending[end + len(_CALL_END) :]
            self._in_call = False
            return True
        self._in_call = False
        self._calls += 1
        return True

    def _emit(self, calls: list[CallDelta], name: str | None, arguments: str) -> None:
        self._started = True
        calls.append(CallDelta(self._calls, name, arguments))

    def _release(self, content: list[str], text: str) -> None:
        if text:
            content.append(text)
            self._line_start = text.endswith("\n")

    def _at_line_start(self, text: str, at: int) -> bool:
        return text[at - 1] == "\n" if at else self._line_start

    @staticmethod
    def _delta(content: list[str], calls: list[CallDelta]) -> ParseDelta:
        return ParseDelta("".join(content), tuple(calls))

    def _open_call(self) -> None:
        """Reset the format's per-call state."""

    def _body(self, calls: list[CallDelta]) -> bool | None:
        """Consume ``_pending`` through the call's end: True when it ended, False when it
        names no requested tool, None when it needs more text."""
        raise NotImplementedError


def parser_for(chat_template: str) -> str | None:
    """The parser of the call format a chat template renders, None when it takes no tools."""

    if "<function=" in chat_template and "<parameter=" in chat_template:
        return "qwen3_coder"
    if _CALL in chat_template:
        return "hermes"
    return None


def opens_think(chat_template: str) -> bool:
    """Whether the template's generation prompt, with its own thinking default, ends inside
    an open ``<think>``, so a response begins in thought (Qwen3.5 thinking)."""

    from jinja2 import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.globals["raise_exception"] = raise_exception
    try:
        prompt = environment.from_string(chat_template).render(
            messages=[{"role": "user", "content": "x"}], add_generation_prompt=True
        )
    except TemplateError:
        return False
    tail = prompt.rstrip()
    return tail.endswith(_THINK) and _THINK_END not in tail[tail.rfind(_THINK) :]


# The parsers ServerOptions.tool_call_parser names, but for "auto".
PARSERS = tuple(
    name for name in get_args(get_type_hints(ServerOptions)["tool_call_parser"]) if name != "auto"
)


@dataclass(frozen=True, slots=True)
class ToolCallFormat:
    """How a model's chat template writes calls: the parser of their format, and whether
    its generation prompt opens thinking."""

    name: str  # one of PARSERS
    in_think: bool = False

    @classmethod
    def of(cls, chat_template: str | None, parser: str = "auto") -> ToolCallFormat | None:
        """The format of ``chat_template``, ``parser`` naming its parser unless ``auto``;
        None when the template takes no tools."""

        if not chat_template:
            return None if parser == "auto" else cls(parser)
        name = parser_for(chat_template) if parser == "auto" else parser
        return None if name is None else cls(name, opens_think(chat_template))

    def parser(self, tools: Sequence[Mapping[str, Any]]) -> ToolCallParser:
        """A choice's parser of calls to ``tools``."""

        from .hermes import HermesParser
        from .qwen3_coder import Qwen3CoderParser

        parser = {"hermes": HermesParser, "qwen3_coder": Qwen3CoderParser}[self.name]
        return parser(tools, in_think=self.in_think)


__all__ = [
    "PARSERS",
    "CallDelta",
    "ParseDelta",
    "ToolCallFormat",
    "ToolCallParser",
    "opens_think",
    "parser_for",
]
