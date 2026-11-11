"""The OpenAI response shapes of the API server: each endpoint's whole response and its
stream's chunks."""

from __future__ import annotations

import secrets
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from cti.protocol import PromptReply, TokenUsage
from cti.sampling import TokenLogprobs

from ..tool_calls import CallDelta

if TYPE_CHECKING:
    pass
from .rendering import ChoiceDelta, ChoiceOutput
from .requests import GenerationRequest


def _usage_payload(usage: TokenUsage) -> dict[str, object]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_prompt_tokens},
    }


class ResponseFormat(ABC):
    """How one generation endpoint shapes its response and its stream's chunks. Choice
    ``prompt_index * n + j`` is the ``j``-th of a prompt's ``n`` choices."""

    ID_PREFIX: ClassVar[str]
    OBJECT: ClassVar[str]
    CHUNK_OBJECT: ClassVar[str]

    def __init__(self, request: GenerationRequest, request_id: int) -> None:
        self._request = request
        self._id = f"{self.ID_PREFIX}-{request_id}"
        self._created = int(time.time())

    def response(self, choices: Sequence[ChoiceOutput]) -> dict[str, object]:
        payload = self._envelope(
            self.OBJECT,
            [self._choice(index, choice) for index, choice in enumerate(choices)],
        )
        payload["usage"] = _usage_payload(self._usage([choice.usage for choice in choices]))
        return payload

    def delta_chunk(self, index: int, delta: ChoiceDelta, first: bool) -> dict[str, object]:
        """The chunk of one delta; the first of each choice opens the choice."""

        return self._chunk(self._delta(index, delta, first))

    def end_chunk(self, index: int, finish_reason: str | None) -> dict[str, object]:
        return self._chunk(self._end(index, finish_reason))

    def usage_chunk(self, usages: Sequence[TokenUsage]) -> dict[str, object]:
        """The chunk of the call's usage, from each choice's, in choice order."""

        payload = self._envelope(self.CHUNK_OBJECT, [])
        payload["usage"] = _usage_payload(self._usage(usages))
        return payload

    def _usage(self, usages: Sequence[TokenUsage]) -> TokenUsage:
        """Every choice's output and each prompt once, as its first choice counts it; vLLM
        counts a prompt once for its ``n`` choices."""

        n = self._request.n
        return sum(
            (
                usage if index % n == 0 else TokenUsage(0, usage.completion_tokens)
                for index, usage in enumerate(usages)
            ),
            start=TokenUsage(0, 0),
        )

    @abstractmethod
    def _choice(self, index: int, choice: ChoiceOutput) -> dict[str, object]: ...

    @abstractmethod
    def _delta(self, index: int, delta: ChoiceDelta, first: bool) -> dict[str, object]: ...

    @abstractmethod
    def _end(self, index: int, finish_reason: str | None) -> dict[str, object]: ...

    def _envelope(self, obj: str, choices: list[dict[str, object]]) -> dict[str, object]:
        return {
            "id": self._id,
            "object": obj,
            "created": self._created,
            "model": self._request.model,
            "choices": choices,
        }

    def _chunk(self, choice: dict[str, object]) -> dict[str, object]:
        payload = self._envelope(self.CHUNK_OBJECT, [choice])
        if self._request.include_usage:
            payload["usage"] = None
        return payload

    def _add_token_ids(self, choice: dict[str, object], token_ids: Sequence[int]) -> None:
        if self._request.return_token_ids:
            choice["token_ids"] = list(token_ids)


class ChatCompletionFormat(ResponseFormat):
    ID_PREFIX = "chatcmpl"
    OBJECT = "chat.completion"
    CHUNK_OBJECT = "chat.completion.chunk"

    def response(self, choices: Sequence[ChoiceOutput]) -> dict[str, object]:
        payload = super().response(choices)
        if self._request.return_token_ids:
            payload["prompt_token_ids"] = list(choices[0].prompt.token_ids)
        return payload

    def delta_chunk(self, index: int, delta: ChoiceDelta, first: bool) -> dict[str, object]:
        payload = super().delta_chunk(index, delta, first)
        if delta.prompt is not None:
            payload["prompt_token_ids"] = list(delta.prompt.token_ids)
        return payload

    def _choice(self, index: int, choice: ChoiceOutput) -> dict[str, object]:
        message: dict[str, object] = {"role": "assistant", "content": choice.text}
        if choice.tool_calls:
            message["content"] = choice.text or None
            message["tool_calls"] = [
                {
                    "id": _call_id(),
                    "type": "function",
                    "function": {"name": name, "arguments": "".join(arguments)},
                }
                for name, arguments in choice.tool_calls
            ]
        result: dict[str, object] = {
            "index": index,
            "message": message,
            "logprobs": self._logprobs(choice),
            "finish_reason": choice.finish_reason,
        }
        self._add_token_ids(result, choice.token_ids)
        return result

    def _delta(self, index: int, delta: ChoiceDelta, first: bool) -> dict[str, object]:
        message: dict[str, object] = {"role": "assistant"} if first else {}
        if delta.text:
            message["content"] = delta.text
        if delta.tool_calls:
            message["tool_calls"] = _call_chunks(delta.tool_calls)
        result: dict[str, object] = {
            "index": index,
            "delta": message,
            "logprobs": self._logprobs(delta),
            "finish_reason": None,
        }
        self._add_token_ids(result, delta.token_ids)
        return result

    def _end(self, index: int, finish_reason: str | None) -> dict[str, object]:
        result: dict[str, object] = {
            "index": index,
            "delta": {},
            "logprobs": None,
            "finish_reason": finish_reason,
        }
        self._add_token_ids(result, [])
        return result

    def _logprobs(self, output: ChoiceOutput | ChoiceDelta) -> dict[str, object] | None:
        """OpenAI's ``logprobs.content``: each token with its UTF-8 bytes and alternatives."""

        if self._request.top_logprobs_count is None:
            return None

        def token(token_id: int, logprob: float) -> dict[str, object]:
            text = output.token_texts[token_id]
            return {
                "token": text,
                "logprob": logprob,
                "bytes": list(text.encode("utf-8", errors="replace")),
            }

        return {
            "content": [
                {
                    **token(entry.token_id, entry.logprob),
                    "top_logprobs": [token(*alternative) for alternative in entry.top],
                }
                for entry in output.logprobs
            ]
        }


def _call_id() -> str:
    return f"call_{secrets.token_hex(12)}"


def _call_chunks(calls: Sequence[CallDelta]) -> list[dict[str, object]]:
    """A stream chunk's ``tool_calls``: one entry per call, its first with the call's id,
    type and name."""

    chunks: dict[int, dict[str, Any]] = {}
    for call in calls:
        chunk = chunks.setdefault(call.index, {"index": call.index})
        if call.name is not None:
            chunk.update(id=_call_id(), type="function", function={"name": call.name})
        function = chunk.setdefault("function", {})
        function["arguments"] = function.get("arguments", "") + call.arguments
    return list(chunks.values())


class CompletionFormat(ResponseFormat):
    ID_PREFIX = "cmpl"
    OBJECT = "text_completion"
    CHUNK_OBJECT = "text_completion"

    def __init__(self, request: GenerationRequest, request_id: int) -> None:
        super().__init__(request, request_id)
        # The text each streamed choice has sent, where its next logprobs' offsets start.
        self._sent: dict[int, int] = {}

    def _choice(self, index: int, choice: ChoiceOutput) -> dict[str, object]:
        echo = self._echo(choice.prompt)
        result: dict[str, object] = {
            "index": index,
            "text": echo + choice.text,
            "logprobs": self._logprobs(choice, 0),
            "finish_reason": choice.finish_reason,
        }
        self._add_prompt_token_ids(result, choice.prompt)
        self._add_token_ids(result, choice.token_ids)
        return result

    def _delta(self, index: int, delta: ChoiceDelta, first: bool) -> dict[str, object]:
        echo = self._echo(delta.prompt)
        sent = self._sent.get(index, 0)
        self._sent[index] = sent + len(echo) + len(delta.text)
        result: dict[str, object] = {
            "index": index,
            "text": echo + delta.text,
            "logprobs": self._logprobs(delta, sent),
            "finish_reason": None,
        }
        self._add_prompt_token_ids(result, delta.prompt)
        self._add_token_ids(result, delta.token_ids)
        return result

    def _end(self, index: int, finish_reason: str | None) -> dict[str, object]:
        result: dict[str, object] = {
            "index": index,
            "text": "",
            "logprobs": None,
            "finish_reason": finish_reason,
        }
        self._add_token_ids(result, [])
        return result

    def _logprobs(
        self, output: ChoiceOutput | ChoiceDelta, offset: int
    ) -> dict[str, object] | None:
        """The legacy completions logprobs: each token, the echoed prompt's first, its
        logprob, its own and its alternatives' logprobs by text, and where its text starts,
        the tokens' texts laid end to end from ``offset``, as vLLM lays them. The prompt's
        first token has no logprobs."""

        if self._request.top_logprobs_count is None:
            return None
        texts = output.token_texts
        entries: list[tuple[int, TokenLogprobs | None]] = []
        if output.prompt is not None and output.prompt_logprobs:
            entries += zip(output.prompt.token_ids, output.prompt_logprobs, strict=True)
        entries += [(entry.token_id, entry) for entry in output.logprobs]
        tokens: list[str] = []
        token_logprobs: list[float | None] = []
        top_logprobs: list[dict[str, float] | None] = []
        text_offset: list[int] = []
        for token_id, entry in entries:
            text = texts[token_id]
            tokens.append(text)
            text_offset.append(offset)
            offset += len(text)
            if entry is None:
                token_logprobs.append(None)
                top_logprobs.append(None)
                continue
            token_logprobs.append(entry.logprob)
            top = {text: entry.logprob}
            top.update((texts[t], logprob) for t, logprob in entry.top)
            top_logprobs.append(top)
        return {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "text_offset": text_offset,
        }

    def _echo(self, prompt: PromptReply | None) -> str:
        return prompt.text if prompt is not None and self._request.echo else ""

    def _add_prompt_token_ids(self, choice: dict[str, object], prompt: PromptReply | None) -> None:
        if prompt is not None and self._request.return_token_ids:
            choice["prompt_token_ids"] = list(prompt.token_ids)
