# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/message/frontend.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cti.sampling import TokenLogprobs

from ._codec import Family
from ._tokenizer import RequestRejectedMsg
from ._usage import TokenUsage


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def family() -> Family:
        # A rejection reaches the frontend as the engine or the tokenizer made it.
        return Family((UserReply, PromptReply, RequestRejectedMsg))


@dataclass
class UserReply(BaseFrontendMsg):
    """New output of one request from one engine step (a speculative round commits several
    tokens). ``token_ids`` are the output tokens it accounts for, a stop token too, though
    its text never shows; ``stop_str`` is the stop string whose completing token ends the
    request. ``logprobs`` are those tokens' when the request asks for them, one each, and a
    request's first reply carries its ``prompt_logprobs`` (None for the first prompt token).
    ``token_texts`` holds each token id those name, decoded alone with special tokens shown,
    as vLLM shows a logprob token. A terminal reply carries ``max_new_tokens``, the output
    limit admission applied, and the ``model_revision`` the request ran at."""

    uid: int
    incremental_output: str
    finished: bool
    usage: TokenUsage | None = None
    finish_reason: Literal["stop", "length"] | None = None
    token_ids: tuple[int, ...] = ()
    stop_str: str | None = None
    logprobs: tuple[TokenLogprobs, ...] = ()
    prompt_logprobs: tuple[TokenLogprobs | None, ...] | None = None
    token_texts: dict[int, str] = field(default_factory=dict)
    max_new_tokens: int | None = None
    model_revision: int | None = None


@dataclass
class PromptReply(BaseFrontendMsg):
    """A request's prompt as tokenized, sent before the request reaches the engine when it
    asks for it. ``token_texts`` names the first token, which no prompt logprobs name, when
    the request asks for them."""

    uid: int
    token_ids: tuple[int, ...]
    text: str
    token_texts: dict[int, str] = field(default_factory=dict)
