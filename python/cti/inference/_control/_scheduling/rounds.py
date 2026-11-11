"""The scheduler's account of multi-token speculative rounds: what each running request's
unsettled rounds may output, their cache uses, and each round's rows as the device committed
them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cti.inference._control._cache import CacheManager
from cti.inference._step import Round, RoundCompletion
from cti.protocol import UserMsg
from cti.sampling import TokenLogprobs

from .request import RowClaim

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from cti.forward import RoundShape
    from cti.inference._control._scheduling.strategy import SchedulingLimits

    from .request import Request
    from .table import Slot


@dataclass(frozen=True, slots=True)
class DeviceRow:
    """A round row as its device record committed it: the visible tokens (accepted drafts
    and correction or bonus, through a stop token and within the output budget), their
    logprobs if the request asked for them, and the record's frontier and committed
    positions. A finished request's row (a discarded successor) shows no tokens."""

    tokens: list[int]
    logprobs: list[TokenLogprobs] | None
    record: tuple[int, int]


@dataclass(frozen=True, slots=True)
class RoundTotals:
    """Every live request's settled rounds and their accepted drafts, before EOS, own-stop
    and budget clipping; a round's acceptance length is its accepted drafts plus one."""

    num_rounds: int = 0
    num_accepted_drafts: int = 0

    def __str__(self) -> str:
        mean = 1 + self.num_accepted_drafts / max(self.num_rounds, 1)
        return (
            f"Speculative {self.num_rounds} request rounds, "
            f"{self.num_accepted_drafts} drafts accepted, mean acceptance length {mean:.2f}"
        )


class RoundLedger:
    """Running requests decode ``width`` (Q) tokens per round, each with up to ``depth``
    rounds unsettled. The ledger admits their policies, plans rounds on cache uses, and
    reads each round's rows from its device record, in submission order."""

    def __init__(
        self,
        shape: RoundShape,
        *,
        limits: SchedulingLimits,
        cache_manager: CacheManager,
        stop_token_ids: frozenset[int],
        checks: bool = False,
    ) -> None:
        """``checks`` compares every settled row's device record with the host's rules, the
        model's ``stop_token_ids`` among them."""

        self._width, self._depth = shape.width, shape.depth
        self._checks = checks
        self._cache_manager = cache_manager
        self._stop_token_ids = stop_token_ids
        self._max_seq_len = limits.max_sequence_length
        self._max_rows = shape.max_rows
        self._block_cursor = 0
        self._totals = RoundTotals()

    @property
    def totals(self) -> RoundTotals:
        return self._totals

    @property
    def width(self) -> int:
        return self._width

    def admit(self, message: UserMsg) -> UserMsg:
        """``message``, which rounds sample as it is: seeded, and greedy without filters."""

        if self._checks and message.sampling.seed is None:
            # RequestSeeds stamps every request with rounds on the primary rank's ingress.
            raise RuntimeError(f"request {message.uid} reached round scheduling unseeded")
        return message

    def lookahead(self, request: Request) -> int:
        """KV positions a round may write past the request's output budget: a block at its
        last anchor, within the sequence capacity (guard rows take the rest)."""

        return min(self._width - 1, self._max_seq_len - request.max_device_len)

    def step_end(self, slot: Slot) -> int:
        """The end its request's next round reserves below; 0 when it takes none."""

        if not self._round_credit(slot):
            return 0
        return min(self._max_seq_len, self._settled_bound(slot) + self._width)

    def plan(self, running: Iterable[Slot]) -> tuple[Round, tuple[RowClaim, ...]] | None:
        """One round for the requests that may take another unsettled round, rotating when
        more wait than fit. A request its unsettled rounds may finish only rides along: alone,
        the round could be wasted; beside others, it costs one row and saves a round."""

        rows = []
        for slot in sorted(
            running, key=lambda slot: (slot.request.uid < self._block_cursor, slot.request.uid)
        ):
            if max_outputs := self._round_credit(slot):
                rows.append((slot, max_outputs))
            if len(rows) == self._max_rows:
                break
        if all(self._settled_bound(slot) >= slot.request.max_device_len - 1 for slot, _ in rows):
            return None
        requests = [slot.request for slot, _ in rows]
        for request in requests:
            if request.device_len != request.cached_len + 1:
                raise RuntimeError(f"request {request.uid} has no single uncached anchor")
        claims, pages = self._cache_manager.reserve_rounds(
            [request.slot for request in requests],
            [self.step_end(slot) for slot, _ in rows],
        )
        for slot, max_outputs in rows:
            slot.round_credits.append(max_outputs)
        self._block_cursor = requests[-1].uid + 1
        captures = tuple(
            -1 if claim.snapshot_row is None else claim.snapshot_row for claim in claims
        )
        round_op = Round(
            tuple(request.slot.index for request in requests),
            None if all(row < 0 for row in captures) else captures,
            pages=pages,
        )
        return round_op, tuple(
            RowClaim(slot.request, claim, None, max_outputs)
            for (slot, max_outputs), claim in zip(rows, claims, strict=True)
        )

    def _settled_bound(self, slot: Slot) -> int:
        """The furthest a request's anchor can be once its unsettled rounds land: each
        commits at most Q, and none passes the last output position."""

        request = slot.request
        return min(
            request.cached_len + len(slot.round_credits) * self._width, request.max_device_len - 1
        )

    def _round_credit(self, slot: Slot) -> int:
        """The most tokens one more round of the slot's request may output; 0 when it may
        take none: its rounds are at depth or certainly finish it."""

        request, pending = slot.request, slot.round_credits
        # Each unsettled round of a live request commits at least one token.
        remaining = request.max_device_len - 1 - request.cached_len - len(pending)
        if len(pending) >= self._depth or remaining <= 0:
            return 0
        return min(self._width, remaining)

    def settle(
        self, rows: tuple[RowClaim, ...], slots: Sequence[Slot], record: RoundCompletion
    ) -> list[DeviceRow]:
        """A planned round's rows, of ``slots``, as its device record committed them, once the
        record agrees with the host's rules under checks (else raise with nothing changed).
        Live rows count toward the totals with their accepted drafts, before any stop or
        budget cut."""

        views = _RecordViews(record)
        settled = [
            self._device_row(claim, slot, views, row)
            for row, (claim, slot) in enumerate(zip(rows, slots, strict=True))
        ]
        live = accepted_drafts = 0
        for index, (slot, row) in enumerate(zip(slots, settled, strict=True)):
            slot.round_credits.pop(0)
            frontier, committed = row.record
            # While more are unsettled, the next one continues where this one left the row.
            slot.round_device_row = (
                (frontier + committed, views.done[index]) if slot.round_credits else None
            )
            if not slot.request.is_finished:
                live += 1
                accepted_drafts += views.accepted[index]
        totals = self._totals
        self._totals = RoundTotals(
            totals.num_rounds + live, totals.num_accepted_drafts + accepted_drafts
        )
        return settled

    def _device_row(self, claim: RowClaim, slot: Slot, record: _RecordViews, row: int) -> DeviceRow:
        """Row ``row`` as the device committed it: its visible tokens are the committed
        ones. Under ``checks`` its record must agree with the host's rules: its frontier
        continues its predecessor's, and committed and done follow the budget and stop rules."""

        request = claim.request
        tokens = record.tokens(row)
        accepted, committed = record.accepted[row], record.committed[row]
        frontier, done = record.frontiers[row], record.done[row]
        if self._checks:
            self._check(slot, tokens, accepted, committed, frontier, done)
        if request.is_finished:
            return DeviceRow([], None, (frontier, committed))
        visible = list(tokens[:committed])
        if self._checks and len(visible) > claim.max_outputs:
            raise RuntimeError(f"request {request.uid}'s round exceeded its output credit")
        logprobs = None
        if request.sampling.logprobs is not None:
            scores = record.logprobs
            if scores is None:
                raise RuntimeError(f"request {request.uid}'s round returned no log-probabilities")
            top, start = request.sampling.top_logprobs, row * record.width
            logprobs = [scores.at(start + index, top) for index in range(len(visible))]
        return DeviceRow(visible, logprobs, (frontier, committed))

    def _check(
        self,
        slot: Slot,
        tokens: memoryview,
        accepted: int,
        committed: int,
        frontier: int,
        done: bool,
    ) -> None:
        """Without a device row from a settled predecessor, a row starts at the request's
        cached frontier, live: where a request that ended on the host alone (aborted, or
        stopped by the prompt's first token) still runs."""

        request = slot.request
        expected, device_done = slot.round_device_row or (request.cached_len, False)
        if device_done:
            consistent = (accepted, committed, done) == (-1, 0, True)
        else:
            visible, is_stop = self._visible(request, list(tokens), accepted, frontier)
            finished = is_stop or frontier + len(visible) >= request.max_device_len - 1
            consistent = 0 <= accepted < len(tokens) and (committed, done) == (
                len(visible),
                finished,
            )
        if frontier != expected or not consistent:
            raise RuntimeError(
                f"request {request.uid}'s device round diverged from settlement: frontier "
                f"{frontier} (expected {expected}), raw accepted {accepted}, committed "
                f"{committed}, done {done}"
            )

    def _visible(
        self, request: Request, tokens: list[int], accepted: int, frontier: int
    ) -> tuple[list[int], bool]:
        """The tokens a round at ``frontier`` emits: its accepted drafts and correction or
        bonus within the output budget, through the first stop token."""

        visible = tokens[: accepted + 1][: request.max_device_len - 1 - frontier]
        if not request.sampling.ignore_eos:
            stop = next((i for i, t in enumerate(visible) if t in self._stop_token_ids), None)
            if stop is not None:
                return visible[: stop + 1], True
        return visible, False


class _RecordViews:
    """A round's device record, read in place."""

    def __init__(self, record: RoundCompletion) -> None:
        self._tokens = memoryview(record.tokens).cast("i")
        self.width = record.width
        # A full round's scores, row ``b``'s position ``i`` at ``b * width + i``.
        self.logprobs = record.logprobs
        self.accepted = memoryview(record.accepted).cast("i")
        self.committed = memoryview(record.committed).cast("i")
        self.frontiers = memoryview(record.frontiers).cast("q")
        self.done = memoryview(record.done).cast("?")

    def tokens(self, row: int) -> memoryview:
        return self._tokens[row * self.width : (row + 1) * self.width]
