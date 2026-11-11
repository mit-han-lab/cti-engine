"""The engine's replies one token at a time: each step's reply holds a request's tokens of the
step, and its end is its last token's."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, NamedTuple

from cti.protocol import DetokenizeMsg, TokenUsage
from cti.sampling import TokenLogprobs


class TokenReply(NamedTuple):
    uid: int
    next_token: int
    logprobs: TokenLogprobs | None
    finished: bool
    is_stop_token: bool
    stop_str: str | None
    usage: TokenUsage | None
    finish_reason: Literal["stop", "length"] | None
    max_new_tokens: int | None


def per_token(replies: Iterable[DetokenizeMsg]) -> list[TokenReply]:
    tokens = []
    for reply in replies:
        last = len(reply.token_ids) - 1
        for index, token_id in enumerate(reply.token_ids):
            end = reply.end if index == last else None
            tokens.append(
                TokenReply(
                    reply.uid,
                    token_id,
                    reply.logprobs[index] if reply.logprobs else None,
                    end is not None,
                    end is not None and end.is_stop_token,
                    None if end is None else end.stop_str,
                    None if end is None else end.usage,
                    None if end is None else end.finish_reason,
                    None if end is None else end.max_new_tokens,
                )
            )
    return tokens


__all__ = ["TokenReply", "per_token"]
