"""How a call's replies become what its clients see: each choice's deltas, its text held back
while it may begin a stop string, and the tool calls parsed out of it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from cti.protocol import PromptReply, TokenUsage, UserReply
from cti.sampling import TokenLogprobs

from ..tool_calls import CallDelta, ToolCallFormat, ToolCallParser

if TYPE_CHECKING:
    pass
from .._text import held_tail
from .requests import GenerationRequest, OpenAIError


@dataclass(frozen=True, slots=True)
class ChoiceDelta:
    """What a choice's client sees of one reply: new text with the token ids and logprobs it
    releases, the prompt and its logprobs on the choice's first delta when the request
    repeats it, and the choice's end. ``token_texts`` names every token those logprobs
    name."""

    text: str
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[TokenLogprobs, ...] = ()
    prompt: PromptReply | None = None
    # With ``prompt``, when the request asks for them: None for its first token.
    prompt_logprobs: tuple[TokenLogprobs | None, ...] = ()
    token_texts: dict[int, str] = field(default_factory=dict)
    usage: TokenUsage | None = None
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None
    # The parts of tool calls the text releases; ``text`` is then the content around them.
    tool_calls: tuple[CallDelta, ...] = ()


class ChoiceRenderer:
    """Turns one choice's replies into what its client sees.

    The engine names the stop string that ended the request, so the text ends where that
    string begins; output text is held back while its tail may begin one. With
    ``include_stop_str`` the text ends after the string, and nothing is held back. A token's
    id and logprobs leave with the last of its text, and the end releases every token, a
    stop token and those past a cut stop string too, as vLLM reports them. A request that
    repeats its prompt sees it first, whichever order the prompt and output replies arrive
    in, with its logprobs when it asks for them. A ``prompt_only`` request shows no output:
    the engine samples one token for the prompt's usage.
    """

    def __init__(
        self,
        stop: tuple[str, ...],
        *,
        include_stop_str: bool,
        wants_prompt: bool,
        prompt_only: bool,
    ) -> None:
        self._stop = stop
        self._longest_stop = max(map(len, stop), default=0)
        self._include_stop_str = include_stop_str
        self._wants_prompt = wants_prompt
        self._prompt_only = prompt_only
        self._prompt: PromptReply | None = None
        self._early: list[UserReply] = []
        self._held = ""
        # The released text's last characters, which a stop string may have begun in.
        self._released_tail = ""
        self._opened = False
        # The output's length so far, and the tokens not yet released, each with where its
        # text ends in the output.
        self._received = 0
        self._pending: list[tuple[int, int, TokenLogprobs | None]] = []
        # The texts of the tokens the replies' logprobs name.
        self._texts: dict[int, str] = {}

    def add(self, reply: PromptReply | UserReply) -> list[ChoiceDelta]:
        """The deltas ``reply`` releases."""

        self._texts.update(reply.token_texts)
        if isinstance(reply, PromptReply):
            self._prompt = reply
            early, self._early = self._early, []
            return [self._delta(output) for output in early]
        if self._wants_prompt and self._prompt is None:
            self._early.append(reply)
            return []
        return [self._delta(reply)]

    def _delta(self, reply: UserReply) -> ChoiceDelta:
        prompt = None if self._opened else self._prompt
        self._opened = True
        prompt_logprobs = () if prompt is None else reply.prompt_logprobs or ()
        if self._prompt_only:
            usage = None if reply.usage is None else replace(reply.usage, completion_tokens=0)
            finish_reason = "length" if reply.finished else None
            return ChoiceDelta(
                "",
                prompt=prompt,
                prompt_logprobs=prompt_logprobs,
                token_texts=self._named((), prompt, prompt_logprobs),
                usage=usage,
                finish_reason=finish_reason,
            )
        self._received += len(reply.incremental_output)
        logprobs = reply.logprobs or (None,) * len(reply.token_ids)
        self._pending += [
            (self._received, token_id, logprob)
            for token_id, logprob in zip(reply.token_ids, logprobs, strict=True)
        ]
        text = self._visible_text(reply)
        released = self._released_tokens(reply.finished)
        logprobs = tuple(logprob for _, _, logprob in released if logprob is not None)
        return ChoiceDelta(
            text,
            tuple(token_id for _, token_id, _ in released),
            logprobs,
            prompt,
            prompt_logprobs,
            self._named(logprobs, prompt, prompt_logprobs),
            reply.usage,
            reply.finish_reason,
        )

    def _named(
        self,
        logprobs: Sequence[TokenLogprobs],
        prompt: PromptReply | None,
        prompt_logprobs: Sequence[TokenLogprobs | None],
    ) -> dict[int, str]:
        """The texts of the tokens a delta's logprobs name, its prompt's first too."""

        ids = [t for entry in (*logprobs, *prompt_logprobs) if entry for t in entry.token_ids]
        if prompt is not None and prompt_logprobs:
            ids.append(prompt.token_ids[0])
        return {token_id: self._texts[token_id] for token_id in ids}

    def _visible_text(self, reply: UserReply) -> str:
        text = self._held + reply.incremental_output
        if reply.finished:
            self._held = ""
            seen = self._released_tail + text
            if reply.stop_str is not None and (begin := seen.find(reply.stop_str)) >= 0:
                end = begin + len(reply.stop_str) if self._include_stop_str else begin
                return text[: max(0, end - len(self._released_tail))]
            return text
        # Released text then never holds the start of a stop string the output is cut at.
        held = 0 if self._include_stop_str else held_tail(text, self._stop)
        released, self._held = text[: len(text) - held], text[len(text) - held :]
        if self._longest_stop > 1:
            tail = self._released_tail + released
            self._released_tail = tail[max(0, len(tail) - self._longest_stop + 1) :]
        return released

    def _released_tokens(self, finished: bool) -> list[tuple[int, int, TokenLogprobs | None]]:
        """The pending tokens whose text has all been released; at the end, all of them."""

        shown = self._received - len(self._held)
        count = len(self._pending)
        if not finished:
            count = next((i for i, (end, _, _) in enumerate(self._pending) if end > shown), count)
        released, self._pending = self._pending[:count], self._pending[count:]
        return released


class ToolChoiceRenderer:
    """A chat choice's deltas with the tool calls parsed out of their text; a choice that
    called a tool and stopped ends with ``tool_calls``, as SGLang and vLLM end it."""

    def __init__(self, renderer: ChoiceRenderer, parser: ToolCallParser) -> None:
        self._renderer = renderer
        self._parser = parser
        self._called = False

    def add(self, reply: PromptReply | UserReply) -> list[ChoiceDelta]:
        return [self._parsed(delta) for delta in self._renderer.add(reply)]

    def _parsed(self, delta: ChoiceDelta) -> ChoiceDelta:
        parsed = self._parser.feed(delta.text)
        content, calls = parsed.content, parsed.calls
        finish_reason = delta.finish_reason
        if finish_reason is not None:
            end = self._parser.finish()
            content, calls = content + end.content, calls + end.calls
        self._called = self._called or bool(calls)
        if finish_reason == "stop" and self._called:
            finish_reason = "tool_calls"
        return replace(delta, text=content, tool_calls=calls, finish_reason=finish_reason)


class CallRenderer:
    """Turns a call's replies into its choices' deltas; choice ``prompt_index * n + j`` is
    the ``j``-th of a prompt's ``n``. Only a prompt's choice 0 scores it
    (``sampling``), so its siblings open with its prompt and prompt logprobs, and
    hold their deltas until it has opened."""

    def __init__(
        self, request: GenerationRequest, prompts: int, tool_calls: ToolCallFormat | None
    ) -> None:
        """``tool_calls``: how the model's chat template writes calls, None when it takes no
        tools."""

        self._n = request.n
        self._renderers = [self._choice(request, tool_calls) for _ in range(prompts * self._n)]
        self._shares = request.prompt_logprobs_count is not None and self._n > 1
        # Each prompt's choice-0 opening delta, and its siblings' deltas held until then.
        self._openings: dict[int, ChoiceDelta] = {}
        self._held: dict[int, list[tuple[int, ChoiceDelta]]] = {}

    @staticmethod
    def _choice(
        request: GenerationRequest, tool_calls: ToolCallFormat | None
    ) -> ChoiceRenderer | ToolChoiceRenderer:
        """A renderer of one of ``request``'s choices; one that offers tools parses its calls
        out, and a 400 refuses tools the chat template does not take."""

        renderer = ChoiceRenderer(
            request.stop_strings,
            include_stop_str=request.include_stop_str_in_output,
            wants_prompt=request.return_prompt,
            prompt_only=request.prompt_only,
        )
        if not (tools := request.offered_tools):
            return renderer
        if tool_calls is None:
            raise OpenAIError("this model's chat template takes no tools", param="tools")
        return ToolChoiceRenderer(renderer, tool_calls.parser(tools))

    def add(self, index: int, reply: PromptReply | UserReply) -> list[tuple[int, ChoiceDelta]]:
        """The deltas, each with its choice's index, that ``reply`` to choice ``index``
        releases."""

        deltas = [(index, delta) for delta in self._renderers[index].add(reply)]
        if not self._shares or not deltas:
            return deltas
        prompt, choice = divmod(index, self._n)
        if choice == 0:
            if deltas[0][1].prompt is None:
                return deltas
            self._openings[prompt] = deltas[0][1]
            return deltas + [self._opened(prompt, held) for held in self._held.pop(prompt, [])]
        if prompt not in self._openings:
            self._held.setdefault(prompt, []).extend(deltas)
            return []
        return [self._opened(prompt, entry) for entry in deltas]

    def _opened(self, prompt: int, entry: tuple[int, ChoiceDelta]) -> tuple[int, ChoiceDelta]:
        """A sibling's delta, its opening one with choice 0's prompt and prompt logprobs."""

        index, delta = entry
        if delta.prompt is None:
            return entry
        opening = self._openings[prompt]
        return index, replace(
            delta,
            prompt=opening.prompt,
            prompt_logprobs=opening.prompt_logprobs,
            token_texts={**delta.token_texts, **opening.token_texts},
        )


@dataclass(slots=True)
class ChoiceOutput:
    """One choice's whole output, gathered delta by delta."""

    fragments: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[TokenLogprobs] = field(default_factory=list)
    prompt: PromptReply | None = None
    prompt_logprobs: tuple[TokenLogprobs | None, ...] = ()
    token_texts: dict[int, str] = field(default_factory=dict)
    usage: TokenUsage | None = None
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None
    # Each call's name and arguments.
    tool_calls: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.fragments)

    def add(self, delta: ChoiceDelta) -> None:
        if delta.prompt is not None:
            self.prompt = delta.prompt
            self.prompt_logprobs = delta.prompt_logprobs
        self.fragments.append(delta.text)
        self.token_ids.extend(delta.token_ids)
        self.logprobs.extend(delta.logprobs)
        self.token_texts.update(delta.token_texts)
        for call in delta.tool_calls:
            if call.name is not None:
                self.tool_calls.append((call.name, []))
            self.tool_calls[call.index][1].append(call.arguments)
        if delta.finish_reason is not None:
            self.usage = delta.usage
            self.finish_reason = delta.finish_reason
