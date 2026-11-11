"""Steps of requests that decode in rounds: each step runs the round of every row that can
run, then the waiting prompts the prompt budget and the rest of the token budget admit, in
the round's target forward."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from cti.inference._control._scheduling.strategy import DecodeStrategy

from .request import DecodeStep, RequestBatch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cti.inference._control._scheduling.strategy import RequestTotals
    from cti.inference._step import RoundCompletion
    from cti.protocol import UserMsg

    from .prefill import PrefillQueue
    from .request import Request, RowClaim
    from .rounds import DeviceRow, RoundLedger
    from .table import Slot


class RoundSteps(DecodeStrategy):
    """Requests decode in rounds, which ``ledger`` plans and settles. As vLLM schedules
    running requests before waiting ones, a step is the round of every row that can run,
    beside the prompt extensions that fit the prompt budget and the rest of the token budget;
    a prompt runs alone only when no row can. Alternating prompt steps and rounds instead is
    this one rule."""

    def __init__(self, ledger: RoundLedger) -> None:
        self._ledger = ledger

    def admit(self, message: UserMsg) -> UserMsg:
        return self._ledger.admit(message)

    def lookahead(self, request: Request) -> int:
        return self._ledger.lookahead(request)

    def step_end(self, slot: Slot) -> int:
        return self._ledger.step_end(slot)

    def settle(
        self, rows: tuple[RowClaim, ...], slots: Sequence[Slot], record: RoundCompletion
    ) -> list[DeviceRow]:
        return self._ledger.settle(rows, slots, record)

    def report(self, totals: RequestTotals) -> RequestTotals:
        return replace(totals, rounds=self._ledger.totals)

    def plan(
        self,
        running: Sequence[Slot],
        queue: PrefillQueue,
        *,
        max_batch_tokens: int,
        max_prompt_tokens: int,
        reserved_size: int,
    ) -> DecodeStep | None:
        rounds = self._ledger.plan(running)
        budget = max_batch_tokens
        if rounds is not None:
            budget -= len(rounds[1]) * self._ledger.width
        extensions = queue.schedule_extensions(
            min(max_prompt_tokens, budget), reserved_size=reserved_size
        )
        if rounds is None and not extensions:
            return None
        return DecodeStep(RequestBatch(extensions) if extensions else None, rounds)
