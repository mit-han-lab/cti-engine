"""A request's own stop conditions, decided where its tokens settle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from cti.protocol import SamplingSpec


class TextStream(Protocol):
    """One output's text as its tokens arrive, as final text: text no later token changes,
    the text the detokenizer streams."""

    def feed(self, token_ids: Sequence[int]) -> str:
        """The final text ``token_ids``, the output's next tokens, add."""
        ...

    def split(self) -> list[str]:
        """The final text each token of the last ``feed`` added, in order."""
        ...


class TextDecoder(Protocol):
    """Opens an output's text stream for stop strings."""

    def stream(self, *, skip_special_tokens: bool) -> TextStream: ...


@dataclass(frozen=True, slots=True)
class StopHit:
    """The output ends at token ``index`` of the tokens checked: at a stop token when
    ``stop_str`` is None, else at the token completing the stop string ``stop_str``."""

    index: int
    stop_str: str | None


class RequestStops:
    """One request's stop token ids and stop strings, checked in output order. A stop
    string matches the output's final text, the text the detokenizer streams, at the token
    that completes it. The model's own stop tokens stay with the scheduler."""

    def __init__(self, params: SamplingSpec, decoder: TextDecoder | None) -> None:
        if params.stop and decoder is None:
            raise ValueError("stop strings need a text decoder")
        self._stop_token_ids = frozenset(params.law.stop_token_ids)
        self._stop_strings = params.stop
        self._longest = max(map(len, params.stop), default=0)
        self._text = (
            decoder.stream(skip_special_tokens=params.skip_special_tokens) if params.stop else None
        )
        # The final text's last characters that a stop string could still overlap.
        self._tail = ""

    @classmethod
    def of(cls, params: SamplingSpec, decoder: TextDecoder | None) -> RequestStops | None:
        """The stops of a request that has any."""

        return cls(params, decoder) if params.stop or params.law.stop_token_ids else None

    def find(self, tokens: Sequence[int]) -> StopHit | None:
        """The first of ``tokens``, the request's next output tokens, that ends its output."""

        stop_token = next((i for i, t in enumerate(tokens) if t in self._stop_token_ids), None)
        # Only the tokens before a stop token can complete a stop string.
        before = tokens if stop_token is None else tokens[:stop_token]
        if self._text is not None and before and (hit := self._match(before)) is not None:
            return hit
        return None if stop_token is None else StopHit(stop_token, None)

    def _match(self, tokens: Sequence[int]) -> StopHit | None:
        """Where a stop string completes among ``tokens``."""

        text = self._text.feed(tokens)
        # A stop string within the kept tail alone was found when the tail was new.
        if (stop := self._search(self._tail + text)) is None:
            tail = self._tail + text
            self._tail = tail[max(0, len(tail) - self._longest + 1) :]
            return None
        seen = self._tail
        for index, added in enumerate(self._text.split()[:-1]):
            seen += added
            if stop in seen:
                return StopHit(index, stop)
        return StopHit(len(tokens) - 1, stop)

    def _search(self, text: str) -> str | None:
        """The stop string completing first in ``text``, the longest on a tie."""

        found: tuple[int, int, str] | None = None
        for stop in self._stop_strings:
            begin = text.find(stop)
            if begin >= 0 and (found is None or (begin + len(stop), begin) < found[:2]):
                found = (begin + len(stop), begin, stop)
        return None if found is None else found[2]
