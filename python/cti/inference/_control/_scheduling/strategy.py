# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/scheduler/scheduler.py`); see THIRD_PARTY_NOTICES.

"""How running requests decode, the limits a scheduler is built with, and the totals it
reports."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from cti.protocol import TokenUsage

from .request import DecodeStep, RequestBatch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cti.inference._step import RoundCompletion
    from cti.protocol import UserMsg

    from .prefill import PrefillQueue
    from .request import Request, RowClaim
    from .rounds import DeviceRow, RoundTotals
    from .table import Slot


class DecodeStrategy:
    """How running requests decode, planning their steps: one token per step
    (``TokenSteps``) or in rounds (``RoundSteps``), whose rows it reads back."""

    def admit(self, message: UserMsg) -> UserMsg:
        """``message`` as this strategy decodes it."""
        return message

    def lookahead(self, request: Request) -> int:
        """KV positions a request may write past its output budget."""
        return 0

    def step_end(self, slot: Slot) -> int:
        """The most cache a running request's next step reserves below."""
        return slot.request.device_len

    def plan(
        self,
        running: Sequence[Slot],
        queue: PrefillQueue,
        *,
        max_batch_tokens: int,
        max_prompt_tokens: int,
        reserved_size: int,
    ) -> DecodeStep | None:
        """The next step over the ``running`` requests, in the order they started running, and the
        prompts ``queue`` admits, or ``None`` when nothing can run: the running requests
        first, then prompt extensions within ``max_prompt_tokens`` and the rest of
        ``max_batch_tokens``; ``reserved_size`` is the cache the running requests reserve."""
        raise NotImplementedError

    def settle(
        self, rows: tuple[RowClaim, ...], slots: Sequence[Slot], record: RoundCompletion
    ) -> list[DeviceRow]:
        """A planned round's ``rows``, of ``slots``, as its device ``record`` committed them."""
        raise NotImplementedError("only rounds settle from a round record")

    def report(self, totals: RequestTotals) -> RequestTotals:
        """``totals`` with this strategy's own."""
        return totals


class TokenSteps(DecodeStrategy):
    """Every running request decodes one token per step, beside the prompt extensions that
    fit the prompt budget and the rest of the token budget."""

    def plan(
        self,
        running: Sequence[Slot],
        queue: PrefillQueue,
        *,
        max_batch_tokens: int,
        max_prompt_tokens: int,
        reserved_size: int,
    ) -> DecodeStep | None:
        decode_requests = tuple(slot.request for slot in running)
        decode_tokens = sum(request.extend_len for request in decode_requests)
        if decode_tokens > max_batch_tokens:
            raise RuntimeError("running decode requests exceed the scheduled-token budget")
        extend_requests = queue.schedule_extensions(
            min(max_prompt_tokens, max_batch_tokens - decode_tokens), reserved_size=reserved_size
        )
        if not extend_requests and not decode_requests:
            return None
        return DecodeStep(RequestBatch(extend_requests, decode_requests), None)


@dataclass(frozen=True, slots=True)
class SchedulingLimits:
    """Request-capacity limits resolved before scheduler construction: ``max_prompt_tokens``
    is the prompt tokens one step may carry, within ``max_batch_tokens``."""

    max_running_requests: int
    max_batch_tokens: int
    max_sequence_length: int
    max_prompt_tokens: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_running_requests", self.max_running_requests),
            ("max_batch_tokens", self.max_batch_tokens),
            ("max_sequence_length", self.max_sequence_length),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_batch_tokens < self.max_running_requests:
            raise ValueError("max_batch_tokens must cover one decode token per running request")
        if not (
            type(self.max_prompt_tokens) is int
            and 0 < self.max_prompt_tokens <= self.max_batch_tokens
        ):
            raise ValueError("max_prompt_tokens must be a positive integer within max_batch_tokens")


@dataclass(frozen=True, slots=True)
class RequestTotals:
    """Summed usage of the requests that finished with a result; aborts are not counted.

    ``num_prefix_hits`` counts the requests with at least one cached prompt token;
    ``rounds`` are the round ledger's totals when requests decode in rounds.
    """

    num_requests: int = 0
    num_prefix_hits: int = 0
    usage: TokenUsage = TokenUsage(prompt_tokens=0, completion_tokens=0)
    rounds: RoundTotals | None = None

    def add(self, usage: TokenUsage) -> RequestTotals:
        return replace(
            self,
            num_requests=self.num_requests + 1,
            num_prefix_hits=self.num_prefix_hits + (usage.cached_prompt_tokens > 0),
            usage=self.usage + usage,
        )

    def __str__(self) -> str:
        usage = self.usage
        hit_rate = usage.cached_prompt_tokens / max(usage.prompt_tokens, 1)
        text = (
            f"{self.num_requests} requests finished, {self.num_prefix_hits} with prefix hits; "
            f"prompt {usage.prompt_tokens} tokens, {usage.cached_prompt_tokens} cached "
            f"({hit_rate:.1%}); generated {usage.completion_tokens} tokens"
        )
        if self.rounds is not None and self.rounds.num_rounds:
            text += f"; {self.rounds}"
        return text


__all__ = ["DecodeStrategy", "RequestTotals", "SchedulingLimits", "TokenSteps"]
