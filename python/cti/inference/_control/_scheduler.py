# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/scheduler/scheduler.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from cti.inference._control._cache import CacheManager
from cti.inference._step import (
    Forward,
    QueryKind,
    RequestSpec,
    RequestTable,
    Round,
    RoundCompletion,
    Sampled,
)
from cti.protocol import (
    AbortBackendMsg,
    BaseBackendMsg,
    DetokenizeMsg,
    RejectionKind,
    RequestAbortedMsg,
    RequestEnd,
    RequestRejectedMsg,
    TokenUsage,
    UserMsg,
    context_length_exceeded,
)
from cti.sampling import SamplingError, TokenLogprobs
from cti.utils import init_logger, log_stage

from ._scheduling.encoder_cache import EncoderCache
from ._scheduling.forward_ops import ForwardBuilder
from ._scheduling.prefill import PrefillQueue, estimated_output
from ._scheduling.request import Request, RequestBatch, RowClaim
from ._scheduling.stops import RequestStops, TextDecoder
from ._scheduling.strategy import DecodeStrategy, RequestTotals, SchedulingLimits
from ._scheduling.table import RequestSlot, SlotTable

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class ScheduledStep:
    """A planned step: its op, its rows (the forward's, then the round's), and the
    constants of the requests admitted since the last planned step, which the scheduler's
    request table already holds."""

    op: Forward | Round
    rows: tuple[RowClaim, ...]
    admit: tuple[RequestSpec, ...]

    @property
    def forward_rows(self) -> tuple[RowClaim, ...]:
        return tuple(row for row in self.rows if row.query_kind is not None)

    @property
    def round_rows(self) -> tuple[RowClaim, ...]:
        return tuple(row for row in self.rows if row.query_kind is None)


@dataclass(frozen=True, slots=True)
class _Output:
    """A row's tokens for its request, cut at the first stop; ``cut`` when the request's own
    stop ends a round inside it, so its KV does not commit."""

    tokens: list[int]
    logprobs: list[TokenLogprobs] | None
    is_stop_token: bool
    stop_str: str | None
    finished: bool
    cut: bool
    prompt_logprobs: tuple[TokenLogprobs | None, ...] | None = None


@dataclass(frozen=True, slots=True)
class _Commit:
    """How a settled row commits: the KV positions it makes valid, the prefix it publishes
    or a round's device record, and its output, if its request takes one."""

    row: RowClaim
    committed: int
    prefix_ids: torch.Tensor | None = None
    record: tuple[int, int] | None = None
    output: _Output | None = None


@dataclass(frozen=True, slots=True)
class SettledStep:
    replies: list[DetokenizeMsg]


class InferenceScheduler:
    """Owns inference request policy and logical settlement state."""

    def __init__(
        self,
        *,
        limits: SchedulingLimits,
        cache_manager: CacheManager,
        stop_token_ids: tuple[int, ...],
        vocab_size: int,
        decode: DecodeStrategy,
        encoder_cache: EncoderCache,
        text_decoder: TextDecoder | None = None,
        model_revision: Callable[[], int] = lambda: 0,
        checks: bool = False,
    ) -> None:
        """``vocab_size`` is the vocabulary requests sample from. ``decode`` is how running
        requests decode: ``TokenSteps``, or ``RoundSteps`` that plans steps of rounds. ``encoder_cache`` accounts for the runner's encoder-output
        rows; ``text_decoder`` decodes outputs for requests' stop strings;
        ``model_revision`` reads the revision a request's end reports (0 without training).
        Admission decides every request once; ``checks`` re-checks the internal invariants
        each step relies on."""
        if type(stop_token_ids) is not tuple or any(
            type(token_id) is not int for token_id in stop_token_ids
        ):
            raise TypeError("stop_token_ids must be a tuple of integers")
        if any(token_id < 0 for token_id in stop_token_ids):
            raise ValueError("stop_token_ids must be non-negative")
        # A prompt forward short of its prompt's end ends on the grid, which resolution makes
        # max_prompt_tokens reach.
        if checks and limits.max_prompt_tokens < (grid := cache_manager.prompt_grid):
            raise RuntimeError(f"max_prompt_tokens does not reach the {grid}-token prompt grid")
        # Every admitted request by row and uid, the running ones in order: their one owner.
        self.slots = SlotTable(limits.max_running_requests)
        self._cache_manager = cache_manager
        self._stop_token_ids = frozenset(stop_token_ids)
        self._vocab_size = vocab_size
        self._decode = decode
        self._encoder_cache = encoder_cache
        self._forwards = ForwardBuilder(encoder_cache)
        self._model_revision = model_revision
        self._prefill_queue = PrefillQueue(
            self._cache_manager,
            self.slots,
            encoder_cache=self._encoder_cache,
            lookahead=self._kv_lookahead,
            checks=checks,
        )
        self._checks = checks
        self._text_decoder = text_decoder
        self._max_batch_tokens = limits.max_batch_tokens
        self._max_prompt_tokens = limits.max_prompt_tokens
        self.max_seq_len = limits.max_sequence_length
        self._totals = RequestTotals()
        # Each live slot's request constants, which the executor reads: their one owner.
        self.table = RequestTable()

    @property
    def has_runnable_work(self) -> bool:
        return self._prefill_queue.has_pending or bool(self.slots.num_running)

    @property
    def totals(self) -> RequestTotals:
        return self._decode.report(self._totals)

    def audit_cache(self) -> None:
        self._cache_manager.audit()

    def require_quiescent(self) -> None:
        """Require every admitted request to be settled, and audit the cache under checks."""
        self._require_request_ownership_released()
        self._cache_manager.audit()

    def _require_request_ownership_released(self) -> None:
        if self._prefill_queue.has_pending:
            raise RuntimeError("inference is not quiescent: pending prefill requests")
        self.slots.require_all_released()

    # ============================== Message handling ==============================

    def handle_message(
        self, message: BaseBackendMsg
    ) -> tuple[RequestAbortedMsg | RequestRejectedMsg, ...]:
        replies: list[RequestAbortedMsg | RequestRejectedMsg] = []
        if isinstance(message, UserMsg):
            logger.debug_rank0("Received user msg: %s", message)
            log_stage(logger, "engine_received", (message.uid,))
            if (rejected := self._admit(message)) is not None:
                replies.append(rejected)
        elif isinstance(message, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", message.uid)
            aborted_request = self._prefill_queue.abort(message.uid)
            aborted_request = aborted_request or self._remove_running_request(message.uid)
            if aborted_request is not None:
                self._terminalize_request(aborted_request)
                replies.append(RequestAbortedMsg(uid=message.uid))
        else:
            logger.error("Unknown message type: %s", type(message))
            raise NotImplementedError
        return tuple(replies)

    def abort_all(self) -> tuple[RequestAbortedMsg | RequestRejectedMsg, ...]:
        """Abort every admitted request; their in-flight steps still settle."""

        uids = [
            *self._prefill_queue.pending_ids,
            *(slot.request.uid for slot in self.slots.running),
            *(r.uid for r in self.slots.retracting),
        ]
        return tuple(reply for uid in uids for reply in self.handle_message(AbortBackendMsg(uid)))

    def _admit(self, message: UserMsg) -> RequestRejectedMsg | None:
        """Queue ``message``, or why it cannot be served: every admission rule, decided
        once. Requests and the steps that plan them trust it; admitted, an invalid request
        would raise at planning, which ends serving for every request."""

        def reject(
            reason: str, kind: RejectionKind = RejectionKind.INVALID_REQUEST
        ) -> RequestRejectedMsg:
            return RequestRejectedMsg(uid=message.uid, reason=reason, kind=kind)

        input_len = len(message.input_ids)
        if input_len == 0:
            return reject("prompt must contain at least one token")
        if outside := [t for t, _ in message.sampling.law.logit_bias if t >= self._vocab_size]:
            return reject(
                f"logit_bias token ids {outside} are outside the vocabulary of {self._vocab_size}"
            )
        if message.images is not None or message.rope_layout is not None:
            if reason := self._image_rejection(message):
                return reject(reason)
        # Without a budget, up to the end of the context: a prompt that fills it has no room.
        max_new_tokens = message.sampling.max_new_tokens
        needed = 1 if max_new_tokens is None else max_new_tokens
        if reason := context_length_exceeded(input_len, needed, self.max_seq_len):
            return reject(reason, RejectionKind.CONTEXT_LENGTH_EXCEEDED)
        if max_new_tokens is None:
            try:  # the resolved budget meets the policy's own rules (min_tokens)
                sampling = replace(message.sampling, max_new_tokens=self.max_seq_len - input_len)
            except SamplingError as error:
                return reject(f"invalid sampling policy: {error}")
            message = replace(message, sampling=sampling)
        message = self._decode.admit(message)
        try:
            stops = RequestStops.of(message.sampling, self._text_decoder)
        except ValueError as error:
            return reject(str(error))
        self._prefill_queue.enqueue(message, stops)
        return None

    def _image_rejection(self, message: UserMsg) -> str | None:
        """Why the engine cannot serve a request's images, or None.

        The model's input processor has already checked the placeholders against the
        images; this checks what scheduling relies on.
        """

        if message.images is None or message.rope_layout is None:
            return "images and their rope layout must be present together"
        spans = message.images.spans
        if spans[-1].end > len(message.input_ids):
            return "image placeholders must lie within the prompt"
        largest = max(span.num_tokens for span in spans)
        if largest > self._encoder_cache.rows:
            return f"an image of {largest} tokens exceeds {self._encoder_cache.rows} encoder rows"
        return None

    # ============================== Request planning ==============================

    def plan_next(self) -> ScheduledStep | None:
        """The next submission, or ``None`` when nothing can run: the step the decode policy
        plans, a forward over decodes and prompt extensions, a round, or a round whose target
        forward runs prompt extensions too."""

        if not self._fit_next_step():
            return None
        step = self._decode.plan(
            self.slots.running,
            self._prefill_queue,
            max_batch_tokens=self._max_batch_tokens,
            max_prompt_tokens=self._max_prompt_tokens,
            reserved_size=self._reserved_running_tokens,
        )
        if step is None:
            return None
        op, rows = None, ()
        if step.forward is not None:
            op, rows = self._freeze(step.forward)
        if step.rounds is not None:
            round_op, round_rows = step.rounds
            op, rows = replace(round_op, extend=op), (*rows, *round_rows)
        admitted = self._prefill_queue.take_admitted()
        self.table.admit(admitted)
        return ScheduledStep(op, rows, admitted)

    def _freeze(self, batch: RequestBatch) -> tuple[Forward, tuple[RowClaim, ...]]:
        """Reserve a planned forward's cache and freeze its op and rows."""

        rows = self._forwards.rows(batch)
        slots = tuple(request.slot for request in batch.requests)
        if self._checks:
            if not slots or len(set(slots)) != len(slots):
                raise RuntimeError(
                    "one submission cannot be empty or contain a request slot more than once"
                )
            for slot in slots:
                self.slots.require_live(slot)
                if self.slots.retired(slot) is not None:
                    raise RuntimeError(f"cannot submit table slot {slot} pending reclamation")
        uses, pages = self._cache_manager.reserve(slots, rows)
        rows = tuple(
            replace(row, restore_row=use.restore_row, captures=use.exports)
            if use.restore_row is not None or use.exports
            else row
            for row, use in zip(rows, uses, strict=True)
        )
        forward = self._forwards.forward(batch, rows, pages, self._prefill_queue.continuations)
        claims = tuple(
            RowClaim(request, use, row.query_kind, int(row.write_position is not None))
            for request, use, row in zip(batch.requests, uses, rows, strict=True)
        )
        return forward, claims

    @property
    def _reserved_running_tokens(self) -> int:
        page_padding = self._cache_manager.page_size - 1
        return sum(
            estimated_output(slot.request) + page_padding + self._kv_lookahead(slot.request)
            for slot in self.slots.running
        )

    def _fit_next_step(self) -> bool:
        """Whether the cache holds the next step of every running and prefilling request,
        retracting the least advanced (fewest outputs, then longest prompt, then newest, as
        SGLang retracts) until it does. Admission counts estimated outputs
        (``estimated_output``), so running requests can outgrow the cache. ``False`` waits
        for the steps in flight of retired requests, whose cache then returns."""

        prefilling = self._prefill_queue.continuations
        # A step grows the cache by at most its tokens, each request by up to a page more.
        requests = self.slots.num_running + len(prefilling)
        bound = self._max_batch_tokens + requests * self._cache_manager.page_size
        if bound <= self._cache_manager.available_size:
            return True
        while True:
            ends = [
                *((s.request, self._decode.step_end(s)) for s in self.slots.running),
                *(
                    (r, min(r.device_len + self._max_prompt_tokens, r.prefill_len))
                    for r in prefilling
                ),
            ]
            growth = sum(self._cache_manager.growth_size(r.slot, end) for r, end in ends)
            if growth <= self._cache_manager.available_size:
                return True
            if self.slots.has_retired:
                return False
            if len(ends) <= 1:
                return True
            victim = min(
                (request for request, _ in ends),
                key=lambda r: (r.num_output_tokens, -r.num_prompt_tokens, -r.uid),
            )
            logger.info_rank0(
                "Retracting request %d: the cache cannot hold its next step", victim.uid
            )
            if self.slots.is_running(victim):
                self.slots.stop(victim)
            else:
                self._prefill_queue.retract(victim)
                prefilling = self._prefill_queue.continuations
            self._encoder_cache.pin(victim.uid, None)
            self._retire_request(victim)

    def _kv_lookahead(self, request: Request) -> int:
        return self._decode.lookahead(request)

    def mark_submitted(self, step: ScheduledStep) -> None:
        # A round's rows hold their requests in flight through their own cache uses.
        for row in step.forward_rows:
            request = row.request
            if row.query_kind is QueryKind.EXTEND:
                self._prefill_queue.deferral.track(request, row.use)
            request.mark_submitted()
            if request.can_decode:
                self.slots.run(request)
            else:
                self.slots.stop(request)

    # =========================== Completion settlement ===========================

    def settle_step(self, step: ScheduledStep, output: Sampled | RoundCompletion) -> SettledStep:
        """Settle a step from rank 0's output: a forward's sampled tokens, or a round's
        record with its extensions' tokens. Every row commits its KV, then each live
        request takes its output, cut at its first stop, and one reply. Under checks, a
        forward whose frame does not check raises with nothing changed; a later failure
        raises with the step consumed."""

        forward_rows, round_rows = step.forward_rows, step.round_rows
        commits: list[_Commit] = []
        if forward_rows:
            commits += self._forward_commits(forward_rows, output.sampled if round_rows else output)
        if round_rows:
            commits += self._round_commits(round_rows, output)
        for commit in commits:
            self._cache_manager.settle(
                commit.row.use, commit.committed, prefix_ids=commit.prefix_ids, record=commit.record
            )
        for row in forward_rows:
            self._prefill_queue.deferral.untrack(row.use)
        replies = [self._land(commit) for commit in commits if commit.output is not None]
        # After the tokens land: a retracted request resumes past the last one.
        for row in step.rows:
            self._reclaim_retired_request(row.slot)
        return SettledStep(replies=replies)

    def _forward_commits(self, rows: tuple[RowClaim, ...], sampled: Sampled) -> list[_Commit]:
        token_ids = sampled.token_ids()
        if self._checks and len(token_ids) != len(rows):
            raise RuntimeError("sampled token ID count does not match the forward's rows")
        commits: list[_Commit] = []
        for index, row in enumerate(rows):
            request, use = row.request, row.use
            if request.is_finished:
                commits.append(_Commit(row, 0))
                continue
            if sampled.prompts is not None and (scored := sampled.prompts[index]) is not None:
                request.record_prompt_logprobs(*scored)
            committed = use.end - use.start
            if not row.max_outputs:
                # A prompt chunk publishes its prefix, and with it the snapshots it exported.
                commits.append(_Commit(row, committed, request.prefill_token_ids[: use.end]))
                continue
            if self._checks and request.num_committed_tokens != use.end:
                raise RuntimeError(
                    f"host-committed token frontier for {use.slot} is "
                    f"{request.num_committed_tokens}, expected {use.end}"
                )
            logprobs = None
            if request.sampling.logprobs is not None:
                if sampled.logprobs is None:
                    raise RuntimeError(f"request {request.uid} has no log-probabilities")
                logprobs = [sampled.at(index, request.sampling.top_logprobs)]
            output = replace(
                self._clip(request, [token_ids[index]], logprobs),
                prompt_logprobs=request.take_prompt_logprobs(),
            )
            publish = row.query_kind is QueryKind.EXTEND or output.finished
            prefix_ids = request.token_ids if publish else None
            commits.append(_Commit(row, committed, prefix_ids, output=output))
        return commits

    def _round_commits(self, rows: tuple[RowClaim, ...], record: RoundCompletion) -> list[_Commit]:
        """A round's rows as the device committed them; a row its request's own stop cuts
        commits no KV."""

        commits: list[_Commit] = []
        slots = [self.slots.slot(row.slot) for row in rows]
        for row, device in zip(rows, self._decode.settle(rows, slots, record), strict=True):
            if row.request.is_finished:
                commits.append(_Commit(row, 0, record=device.record))
                continue
            output = self._clip(row.request, device.tokens, device.logprobs)
            committed = 0 if output.cut else len(output.tokens)
            commits.append(_Commit(row, committed, record=device.record, output=output))
        return commits

    def _clip(
        self, request: Request, tokens: list[int], logprobs: list[TokenLogprobs] | None
    ) -> _Output:
        """``request``'s next output ``tokens`` cut at the first stop: a model stop token
        (unless it ignores EOS), or its own stop token or string, matched in the tokens
        before it; and whether they finish it, with its output budget."""

        stop = None
        if not request.sampling.ignore_eos:
            stop = next((i for i, t in enumerate(tokens) if t in self._stop_token_ids), None)
        before = tokens if stop is None else tokens[:stop]
        hit = request.own_stop(before) if before else None
        if hit is not None:
            end, is_stop_token, stop_str = hit.index + 1, hit.stop_str is None, hit.stop_str
        elif stop is not None:
            end, is_stop_token, stop_str = stop + 1, True, None
        else:
            end, is_stop_token, stop_str = len(tokens), False, None
        tokens = tokens[:end]
        finished = (
            is_stop_token
            or stop_str is not None
            or request.num_output_tokens + len(tokens) >= request.max_output_tokens
        )
        return _Output(
            tokens,
            None if logprobs is None else logprobs[:end],
            is_stop_token,
            stop_str,
            finished,
            hit is not None,
        )

    def _land(self, commit: _Commit) -> DetokenizeMsg:
        """Append a row's output to its request, end it if the output finishes it, and its
        reply."""

        row, output = commit.row, commit.output
        request = row.request
        if row.query_kind is None and not output.cut:
            # The anchor and every visible token but the last are now cached.
            request.append_block(output.tokens, cached_len=request.cached_len + len(output.tokens))
        else:
            for token_id in output.tokens:
                request.append_sampled_token(token_id)
        end = None
        if output.finished:
            usage = self._terminalize_request(request)
            self._totals = self._totals.add(usage)
            end = RequestEnd(
                usage,
                request.max_output_tokens,
                output.is_stop_token,
                output.stop_str,
                self._model_revision(),
            )
        return DetokenizeMsg(
            uid=request.uid,
            token_ids=tuple(output.tokens),
            skip_special_tokens=request.policy.skip_special_tokens,
            logprobs=() if output.logprobs is None else tuple(output.logprobs),
            prompt_logprobs=output.prompt_logprobs,
            end=end,
        )

    # =========================== Terminal reclamation ============================

    def _terminalize_request(self, request: Request) -> TokenUsage | None:
        if not request.mark_finished():
            return None
        usage = TokenUsage(
            prompt_tokens=request.num_prompt_tokens,
            completion_tokens=request.num_output_tokens,
            cached_prompt_tokens=request.num_cached_prompt_tokens,
        )
        self.slots.stop(request)
        self._encoder_cache.pin(request.uid, None)
        if request.is_admitted:
            self._retire_request(request)
        return usage

    def _remove_running_request(self, uid: int) -> Request | None:
        """The running or retracting request ``uid``, no longer running."""

        request = self.slots.request(uid)
        if request is None or request.is_finished:
            return None
        self.slots.stop(request)
        return request

    def _retire_request(self, request: Request) -> None:
        self.slots.retire(request)
        self._reclaim_retired_request(request.slot)

    def _reclaim_retired_request(self, slot: RequestSlot) -> None:
        request = self.slots.retired(slot)
        if request is None or self._cache_manager.in_use(slot):
            return
        self._cache_manager.close_slot(slot, prefix_ids=request.token_ids)
        self.slots.reclaimed(slot)
        self.slots.release(slot)
        if not request.is_finished:  # retracted: it resumes from what the cache published
            request.retract()
            self._prefill_queue.requeue(request)
