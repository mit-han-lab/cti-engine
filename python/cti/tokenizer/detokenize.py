# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/tokenizer/detokenize.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, List, NamedTuple

if TYPE_CHECKING:
    from cti.protocol import DetokenizeMsg

    from .text import TextStream, TokenizerTextDecoder


class _Step(NamedTuple):
    """What one decode pass takes of a request: its tokens, whether they end it and whether
    the last is a stop token, which shows no text."""

    uid: int
    token_ids: Sequence[int]
    finished: bool
    stop_token: bool
    skip_special_tokens: bool

    @classmethod
    def of(cls, msg: DetokenizeMsg) -> _Step:
        end = msg.end
        stop = end is not None and end.is_stop_token
        return cls(msg.uid, msg.token_ids, end is not None, stop, msg.skip_special_tokens)


class IncrementalDetokenizer:
    """Streams each request's output text: the final text of its ``TextStream``, the text
    the scheduler matches stop strings in, so the stream and the stop check cannot drift."""

    def __init__(self, decoder: TokenizerTextDecoder) -> None:
        """``decoder`` is the one the scheduler matches stop strings with."""

        self.streams: Dict[int, TextStream] = {}
        self._decoder = decoder

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """The text each message, a request's tokens of one step, adds, in message order. A
        finished request flushes all held text, so its strings concatenate to the decode of
        its tokens."""

        texts = []
        for msg in msgs:
            step = _Step.of(msg)
            stream = self.streams.get(step.uid)
            if stream is None:
                stream = self.streams[step.uid] = self._decoder.stream(
                    skip_special_tokens=step.skip_special_tokens
                )
            # A stop token shows no text.
            shown = step.token_ids[:-1] if step.stop_token else step.token_ids
            text = stream.feed(shown)
            if step.finished:
                text += stream.finish()
                del self.streams[step.uid]
            texts.append(text)
        return texts

    def token_texts(self, msgs: Sequence[DetokenizeMsg]) -> list[dict[int, str]]:
        """Each message's logprob token ids, each decoded alone with special tokens shown,
        as vLLM shows a logprob token; one decode for the batch."""

        named = [
            [
                t
                for entry in (*msg.logprobs, *(msg.prompt_logprobs or ()))
                if entry
                for t in entry.token_ids
            ]
            for msg in msgs
        ]
        ids = list(dict.fromkeys(token_id for msg_ids in named for token_id in msg_ids))
        decoded = (
            self._decoder.batch_decode([[t] for t in ids], skip_special_tokens=False) if ids else []
        )
        texts = dict(zip(ids, decoded, strict=True))
        return [{token_id: texts[token_id] for token_id in msg_ids} for msg_ids in named]

    def abort(self, uid: int) -> None:
        """Release a request that ended without a terminal token."""
        self.streams.pop(uid, None)

    @staticmethod
    def render_text(
        decoder: TokenizerTextDecoder, token_ids: Sequence[int], *, skip_special_tokens: bool
    ) -> str:
        """A whole output's text as the detokenizer streams it, so text cut at a stop string
        offline is the text an online client sees."""

        stream = decoder.stream(skip_special_tokens=skip_special_tokens)
        return stream.feed(token_ids) + stream.finish()
