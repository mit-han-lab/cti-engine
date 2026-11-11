# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/utils/hf.py`); see THIRD_PARTY_NOTICES.

"""Loading a checkpoint's tokenizer and decoding its output text."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

from cti.utils import find_model_file

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        if path := find_model_file(model_path, "chat_template.json"):
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
    return tokenizer


def load_text_decoder(model_path: str) -> TokenizerTextDecoder:
    """The text decoder of ``model_path``'s tokenizer; a process builds one and shares it."""

    return TokenizerTextDecoder(load_tokenizer(model_path))


class TokenizerTextDecoder:
    """Decodes output text as it streams; ``renders`` tells which ids its text shows at all.

    Space cleanup stays off so text decoded in pieces equals the whole. ``prefix_stable``
    holds only for a fast byte-level-BPE tokenizer whose every token has a byte (Qwen's):
    a prefix's text then matches but for a trailing partial character (at most 4 bytes),
    which 3 more rendered ids complete or break. Other tokenizers decode token by token.
    ``stream`` opens one output's ``TextStream``.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        from tokenizers import Tokenizer, decoders

        self.tokenizer = tokenizer
        # A fast tokenizer drops the added tokens marked special before decoding.
        self._special_ids = frozenset(
            token_id for token_id, token in tokenizer.added_tokens_decoder.items() if token.special
        )
        backend = getattr(tokenizer, "backend_tokenizer", None)
        # Only a fast tokenizer tells which ids its vocabulary assigns.
        self._backend = backend if isinstance(backend, Tokenizer) else None
        # The decoder's fields (prefix space, offsets, regex) act on encoding only.
        self.prefix_stable = (
            self._backend is not None
            and isinstance(self._backend.decoder, decoders.ByteLevel)
            and self._backend.token_to_id("") is None
        )

    def stream(self, *, skip_special_tokens: bool) -> TextStream:
        return TextStream(self, skip_special_tokens=skip_special_tokens)

    def renders(self, token_id: int, *, skip_special_tokens: bool) -> bool:
        """Whether decoding ``token_id`` shows anything: ids the vocabulary does not assign
        render nothing."""

        if skip_special_tokens and token_id in self._special_ids:
            return False
        return self._backend is None or self._backend.id_to_token(token_id) is not None

    def batch_decode(self, sequences: list[list[int]], *, skip_special_tokens: bool) -> list[str]:
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )


class _Decoder(Protocol):
    """What a ``TextStream`` decodes with: a ``TokenizerTextDecoder``, or a stand-in."""

    prefix_stable: bool

    def renders(self, token_id: int, *, skip_special_tokens: bool) -> bool: ...

    def batch_decode(
        self, sequences: list[list[int]], *, skip_special_tokens: bool
    ) -> list[str]: ...


class TextStream:
    """One output's text as its tokens arrive, as final text: text no later token changes.
    The detokenizer streams exactly this text, and the scheduler matches stop strings in it.

    With a prefix-stable decoder, the decoded text is final but for a trailing incomplete
    character, which 3 more rendered ids complete or break for good. Any other decoder may
    revise earlier text, so it is decoded token by token and its text is final once it ends
    on a complete character or 8 ids are pending, broken characters included. With a
    prefix-stable decoder a trailing run of replacement characters shows once a complete
    character follows it, as the full decode of the output so far shows it: until then a
    longer run may still follow. ``finish`` returns what is held at the end.
    """

    # A character is at most 4 bytes and every token holds at least one, so once 3 more
    # tokens follow a point, the character pending there is complete or broken for good.
    SETTLING_TOKENS = 3
    # A decoder that may revise earlier text commits once this many ids pend.
    PENDING_LIMIT = 8

    def __init__(self, decoder: _Decoder, *, skip_special_tokens: bool) -> None:
        self._decoder = decoder
        self._skip = skip_special_tokens
        # The rendered ids from where decoding restarts: those before ``_read`` are the
        # committed context the pending ids are decoded after.
        self._ids: list[int] = []
        self._read = 0
        # How much of the pending text has been returned as final, and how many replacement
        # characters before it are final but held: the output's trailing run.
        self._shown = 0
        self._held = 0
        self._split: Callable[[], list[str]] = list

    def feed(self, token_ids: Sequence[int]) -> str:
        """The final text ``token_ids``, the output's next tokens, add."""

        if not self._decoder.prefix_stable:
            texts = [self._feed_one(token_id) for token_id in token_ids]
            self._split = lambda: texts
            return "".join(texts)
        before, read, shown, held = list(self._ids), self._read, self._shown, self._held
        ends = []
        for token_id in token_ids:
            if self._decoder.renders(token_id, skip_special_tokens=self._skip):
                self._ids.append(token_id)
            ends.append(len(self._ids))
        added = list(self._ids[len(before) :])
        self._split = lambda: self._split_stable(before, read, shown, held, added, ends)
        (new,) = self._pending([len(self._ids)])
        text = self._show(new)
        if new and not new.endswith("\ufffd"):
            self._advance(len(self._ids))
        else:
            self._settle(new)
        return text

    def split(self) -> list[str]:
        """The final text each token of the last ``feed`` added, in order; decoded on demand."""

        return self._split()

    def finish(self) -> str:
        """The text still held when the output ends, broken characters included."""

        (new,) = self._pending([len(self._ids)])
        held = "\ufffd" * self._held + new[self._shown :]
        self._ids, self._read, self._shown, self._held = [], 0, 0, 0
        return held

    def _feed_one(self, token_id: int) -> str:
        """A revising decoder's ``feed`` of one token: its text once committed, else ""."""

        if self._decoder.renders(token_id, skip_special_tokens=self._skip):
            self._ids.append(token_id)
        (new,) = self._pending([len(self._ids)])
        pending = len(self._ids) - self._read
        if pending >= self.PENDING_LIMIT or (new and not new.endswith("\ufffd")):
            self._advance(len(self._ids))
            return new
        return ""

    def _release(self, final: str) -> str:
        """``final`` text as it shows: a trailing run of replacement characters held until a
        complete character follows, with the run held before it."""

        text = "\ufffd" * self._held + final
        shown = text.rstrip("\ufffd")
        self._held = len(text) - len(shown)
        return shown

    def _pending(self, ends: list[int]) -> list[str]:
        """The text each of ``self._ids[:end]`` adds to the committed context."""

        context, *decoded = self._decoder.batch_decode(
            [self._ids[: self._read], *(self._ids[:end] for end in ends)],
            skip_special_tokens=self._skip,
        )
        return [text[len(context) :] for text in decoded]

    def _show(self, new: str) -> str:
        """The part of ``new``, the pending text, newly final: all but a trailing run of
        replacement characters, which may be a character still incomplete."""

        final = new.rstrip("\ufffd")
        text = final[self._shown :]
        self._shown = max(self._shown, len(final))
        return self._release(text)

    def _settle(self, new: str) -> None:
        """Commit the longest settled prefix of ``new``, which ends on a trailing incomplete
        or broken character, once enough ids follow it; its broken characters are final, and
        held while they end the output's text."""

        pending = len(self._ids) - self._read
        if pending <= self.SETTLING_TOKENS + 1:
            return
        # The latest three points with enough tokens after them to settle, latest first.
        lags = range(self.SETTLING_TOKENS, self.SETTLING_TOKENS + 3)
        ends = [len(self._ids) - lag for lag in lags if len(self._ids) - lag > self._read]
        for end, text in zip(ends, self._pending(ends), strict=True):
            if text and new.startswith(text):
                shown = max(self._shown, len(text))
                # Beyond what showed, the settled text holds only its trailing run.
                self._held += len(text[self._shown :])
                self._advance(end)
                self._shown = shown - len(text)
                return

    def _advance(self, end: int) -> None:
        """Commit the pending ids before ``end``, which become the context."""

        del self._ids[: self._read]
        self._read = end - self._read
        self._shown = 0

    def _split_stable(
        self,
        before: list[int],
        read: int,
        shown: int,
        held: int,
        added: list[int],
        ends: list[int],
    ) -> list[str]:
        """Each token's share of a prefix-stable ``feed``'s final text, decoded from the
        state before it."""

        ids = before + added
        context, *decoded = self._decoder.batch_decode(
            [ids[:read], *(ids[:end] for end in ends)], skip_special_tokens=self._skip
        )
        through = []
        for text in decoded:
            final = text[len(context) :].rstrip("\ufffd")[shown:]
            through.append("\ufffd" * held + final if final else "")
        return [b[len(a) :] for a, b in zip(["", *through], through, strict=False)]
