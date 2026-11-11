"""The per-request rule that gives rotary positions to a prompt holding images."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class RopePositionLayout(ABC):
    """Rotary positions of one request's prompt and of the tokens decoded after it.

    The input processor of every supported vision-language model returns one layout
    with each request that carries images, and only with those (``ProcessedPrompt``
    checks the pairing). A request without images carries none; each of its tokens sits
    at its sequence index on every axis. How the axes enter the rotary embedding belongs
    to the model's rotary layer, not to the layout.

    The scheduler and the runner may rely on the following:

    - ``positions(start, end)`` depends only on the token range: its columns are those
      of ``positions(0, end)`` from ``start`` on, so a prefill chunk or a cached prefix
      needs nothing from earlier steps.
    - Every token after the prompt sits at its sequence index plus ``decode_offset`` on
      every axis, so decoding needs ``decode_offset`` alone.
    - Implementations are frozen dataclasses: a layout compares by value and crosses the
      inter-process codec as its constructor fields.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def decode_offset(self) -> int:
        """Rotary minus sequence position of every token after the prompt."""

    @abstractmethod
    def positions(self, start: int, end: int) -> torch.Tensor:
        """Positions of tokens ``[start, end)`` as CPU int64 ``[axes, end - start]``.

        There is one row per rotary position axis of the model, and ``end`` may lie past
        the prompt. A range that does not satisfy ``0 <= start <= end`` raises
        ``ValueError``.
        """
