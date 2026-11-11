# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/message/tokenizer.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from cti.sampling import TokenLogprobs

from ._codec import Family
from ._request import AdmissionPriority, RejectionKind, RequestRejectedError, SamplingSpec
from ._usage import TokenUsage


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def family() -> Family:
        # The replies the engine sends the detokenizer, and the calls the frontend sends a
        # tokenizer; each worker refuses the other link's messages.
        return Family((DetokenizeMsg, RequestRejectedMsg, RequestAbortedMsg, TokenizeMsg, AbortMsg))


@dataclass(frozen=True, slots=True)
class RequestEnd:
    """How a request ended, on its last step's reply: its ``usage``, the output limit
    admission applied (``max_new_tokens``), its stop: a stop token, the step's last,
    which shows no text, or the stop string the last token completes; and the
    ``model_revision`` it ran at, which training never changes while a request runs."""

    usage: TokenUsage
    max_new_tokens: int
    is_stop_token: bool = False
    stop_str: str | None = None
    model_revision: int = 0

    def __post_init__(self) -> None:
        if type(self.model_revision) is not int or self.model_revision < 0:
            raise ValueError("model_revision must be a non-negative integer")
        if self.is_stop_token and self.stop_str is not None:
            raise ValueError("A request stops at a stop token or a stop string, not both")

    @property
    def finish_reason(self) -> Literal["stop", "length"]:
        return "stop" if self.is_stop_token or self.stop_str is not None else "length"


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    """One request's output from one engine step: ``token_ids``, the tokens the step
    committed (a speculative round commits several), and their ``logprobs``, one each, when
    the request asks for them. A request's first reply carries its ``prompt_logprobs`` (None
    for the first prompt token); its last carries its ``end``."""

    uid: int
    token_ids: tuple[int, ...]
    skip_special_tokens: bool = True
    logprobs: tuple[TokenLogprobs, ...] = ()
    prompt_logprobs: tuple[TokenLogprobs | None, ...] | None = None
    end: RequestEnd | None = None

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("A reply carries at least one token")
        if self.logprobs and len(self.logprobs) != len(self.token_ids):
            raise ValueError("A reply's logprobs are one per token or none")

    @property
    def finished(self) -> bool:
        return self.end is not None


def context_length_exceeded(
    prompt_tokens: int, max_new_tokens: int, max_seq_len: int
) -> str | None:
    """Why a request breaks OpenAI's rule that its prompt and its whole requested completion
    fit in ``max_seq_len`` tokens (``RejectionKind.CONTEXT_LENGTH_EXCEEDED``), or None."""

    requested = prompt_tokens + max_new_tokens
    if requested <= max_seq_len:
        return None
    return (
        f"This model's maximum context length is {max_seq_len} tokens. However, you "
        f"requested {requested} tokens ({prompt_tokens} in the prompt, {max_new_tokens} in the "
        "completion). Please reduce the length of the prompt or completion."
    )


@dataclass
class RequestRejectedMsg(BaseTokenizerMsg):
    """A request's rejection, from the engine or its tokenizer; the tokenizer forwards the
    engine's to the frontend as is."""

    uid: int
    reason: str
    kind: RejectionKind = RejectionKind.UNAVAILABLE

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("Request rejection reason must not be empty")
        self.kind = RejectionKind(self.kind)

    def error(self, request_id: int) -> RequestRejectedError:
        """The rejection as the frontend raises it, for its call's ``request_id``."""
        return RequestRejectedError(request_id, self.reason, self.kind)


@dataclass
class RequestAbortedMsg(BaseTokenizerMsg):
    """Published after an aborted request's last token; releases its decode state."""

    uid: int


# What the rank sends its tokenizer for one request.
InferenceReply: TypeAlias = DetokenizeMsg | RequestAbortedMsg | RequestRejectedMsg


@dataclass(frozen=True, slots=True)
class TextPrompt:
    text: str


# OpenAI's chat content parts; the API server validates requests against them.
_FORBID_EXTRA = MappingProxyType({"extra": "forbid"})


class ChatTextPart(TypedDict):
    __pydantic_config__ = _FORBID_EXTRA

    type: Literal["text"]
    text: str


class ChatImageURL(TypedDict):
    __pydantic_config__ = _FORBID_EXTRA

    url: str | bytes  # a base64 data: URL, or the bytes the API server fetched
    detail: NotRequired[Literal["auto", "low", "high"]]  # ignored, as vLLM does


class ChatImagePart(TypedDict):
    __pydantic_config__ = _FORBID_EXTRA

    type: Literal["image_url"]
    image_url: ChatImageURL


class ChatTurn(TypedDict):
    """A message; an assistant's may hold only ``tool_calls``, each call's arguments a
    mapping as chat templates iterate them, and a tool's answers ``tool_call_id``."""

    role: str
    content: str | list[ChatTextPart | ChatImagePart]
    name: NotRequired[str]
    tool_calls: NotRequired[list[dict[str, Any]]]
    tool_call_id: NotRequired[str]


@dataclass(frozen=True, slots=True)
class ChatPrompt:
    """Chat messages, rendered with the model's chat template, the tools (OpenAI function
    definitions) the template offers the model and the template's own variables, such as
    ``enable_thinking``."""

    messages: tuple[ChatTurn, ...]
    tools: tuple[dict[str, Any], ...] = ()
    template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenPrompt:
    token_ids: tuple[int, ...]


Prompt: TypeAlias = TextPrompt | ChatPrompt | TokenPrompt


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    """A request to tokenize and send to the engine; ``return_prompt`` has the tokenizer
    answer first with the prompt's token ids and text."""

    uid: int
    prompt: Prompt
    sampling: SamplingSpec
    priority: AdmissionPriority = AdmissionPriority.NORMAL
    return_prompt: bool = False

    def __post_init__(self) -> None:
        self.priority = AdmissionPriority.parse(self.priority)


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
