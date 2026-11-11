# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/scheduler/scheduler.py`); see THIRD_PARTY_NOTICES.

"""A rank's inference: the scheduler's steps submitted to the executor, two in flight, and
the requests that wait while training owns the device."""

from __future__ import annotations

import time
import zlib
from collections import deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from cti.inference._control._cache import CacheManager
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.rounds import RoundLedger
from cti.inference._control._scheduling.rows import RoundSteps
from cti.inference._control._scheduling.strategy import (
    DecodeStrategy,
    SchedulingLimits,
    TokenSteps,
)
from cti.protocol import (
    AbortBackendMsg,
    BaseBackendMsg,
    RequestRejectedMsg,
    UserMsg,
)
from cti.utils import init_logger

if TYPE_CHECKING:
    from cti.inference._control._scheduler import ScheduledStep
    from cti.inference._control._scheduling.stops import TextDecoder
    from cti.inference._step import (
        Completion,
        Executor,
        Forward,
        RequestSpec,
        Round,
        ServingLimits,
        StepResult,
    )
    from cti.memory import ResidencyRestoreError
    from cti.protocol import InferenceReply

logger = init_logger(__name__)


class InferenceEngine:
    """Submits the scheduler's steps to ``executor`` and settles each the step after, so
    step k+1 is on the device before step k completes. It is one side of the mode switch:
    while the switch does not admit requests, arriving ones wait (or are refused with a
    reason) and its KV memory may go to training."""

    DEPTH = 2  # steps in flight; 1 without overlap
    STATS_INTERVAL_SECONDS = 10.0

    def __init__(
        self,
        limits: ServingLimits,
        executor: Executor,
        *,
        publish: Callable[[list[InferenceReply]], None],
        agree_on_step: Callable[[int, str], None] | None,
        text_decoder: TextDecoder | None,
        overlap: bool,
        checks: bool,
        model_revision: Callable[[], int] = lambda: 0,
    ) -> None:
        """``agree_on_step``, under checks with several ranks, raises on every rank unless each
        passed the same digest of the step it planned; ``checks`` runs the cache's integrity scans and the scheduler's comparisons; ``model_revision`` reads the revision a finished request reports."""

        self._executor = executor
        self._checks = checks
        self._publish = publish
        self._agree_on_step = agree_on_step
        self._depth = self.DEPTH if overlap else 1
        self._planned = 0  # steps planned so far, which name a step the ranks disagree on
        capacity = limits.capacity
        self._cache = CacheManager(capacity, prefix=limits.prefix_cache, checks=checks)
        self._cache.warm_up()
        logger.info("Using the %s prefix cache", limits.prefix_cache)
        scheduling = SchedulingLimits(
            max_running_requests=capacity.num_slots,
            max_batch_tokens=limits.max_batch_tokens,
            max_sequence_length=capacity.max_seq_len,
            max_prompt_tokens=limits.max_prompt_tokens,
        )
        # With rounds, requests decode in them.
        decode: DecodeStrategy = TokenSteps()
        if limits.rounds is not None:
            ledger = RoundLedger(
                limits.rounds,
                limits=scheduling,
                cache_manager=self._cache,
                stop_token_ids=frozenset(limits.stop_token_ids),
                checks=checks,
            )
            decode = RoundSteps(ledger)
        self._encoder_cache = EncoderCache(limits.image_rows, checks=checks)
        self._scheduler = InferenceScheduler(
            limits=scheduling,
            cache_manager=self._cache,
            stop_token_ids=limits.stop_token_ids,
            vocab_size=limits.vocab_size,
            decode=decode,
            text_decoder=text_decoder,
            encoder_cache=self._encoder_cache,
            model_revision=model_revision,
            checks=checks,
        )
        self._in_flight: deque[tuple[ScheduledStep, StepResult]] = deque()
        self._parked: Completion | None = None
        self._waiting: list[UserMsg] = []  # while the switch admits no requests
        self._refusal: str | None = None  # why waiting is futile, until serving holds again
        self._stats_at: float | None = None

    @property
    def busy(self) -> bool:
        """Whether a step is in flight or could be planned."""

        return bool(self._in_flight) or self._scheduler.has_runnable_work

    def handle(self, messages: Sequence[BaseBackendMsg], *, admitting: bool) -> None:
        """Admit requests and aborts; unless ``admitting``, a request waits, or is refused."""

        replies: list[InferenceReply] = []
        for message in messages:
            if not admitting and isinstance(message, UserMsg):
                if self._refusal is not None:
                    replies.append(RequestRejectedMsg(uid=message.uid, reason=self._refusal))
                else:
                    self._waiting.append(message)
                continue
            if isinstance(message, AbortBackendMsg) and any(
                waiting.uid == message.uid for waiting in self._waiting
            ):
                self._waiting = [waiting for waiting in self._waiting if waiting.uid != message.uid]
                continue
            replies.extend(self._scheduler.handle_message(message))
        self._publish(replies)

    def step(self) -> None:
        """Launch the next step if there is credit, then settle the oldest once ``DEPTH``
        are in flight. When nothing could be planned, the oldest is parked first: the loop
        waits for it together with new input (``completion``), so a request that arrives
        meanwhile launches as soon as the device frees, and the next call settles it."""

        replies: list[InferenceReply] = []
        launched = False
        if len(self._in_flight) < self._depth:
            planned = self._scheduler.plan_next()
            if planned is not None:
                if self._agree_on_step is not None:
                    # Every rank plans the same step: one that planned differently fails all.
                    self._agree_on_step(
                        step_digest(planned.op, planned.admit),
                        f"planned step {self._planned} ({type(planned.op).__name__})",
                    )
                self._planned += 1
                if self._checks:
                    planned.op.require_vocabulary_ids()
                result = self._executor.submit(planned.op, self._scheduler.table)
                self._scheduler.mark_submitted(planned)
                self._in_flight.append((planned, result))
                launched = True
        if self._in_flight and (len(self._in_flight) >= self._depth or self._parked is not None):
            replies += self._settle_oldest()
        elif self._in_flight and not launched:
            self._parked = self._in_flight[0][1].completion_event
        self._publish(replies)
        self._log_stats()

    def abort_all(self) -> None:
        self._publish(list(self._scheduler.abort_all()))

    # Switchable

    def wants_to_run(self) -> bool:
        return bool(self._waiting) or self.busy

    def drained(self) -> bool:
        """Whether every admitted request has finished."""

        return not self.busy

    def release(self) -> None:
        """The KV memory goes to training: the cache it held is gone. The switch releases
        only a drained side; checks verify it."""

        if self._checks:
            self._scheduler.require_quiescent()
        self._cache.invalidate_all()
        self._encoder_cache.reset()

    def switched(self, holds: bool, refusal: ResidencyRestoreError | None) -> None:
        """Holding the device, admit the waiting requests. Refused it (serving did not fit
        beside training), refuse them and later ones until training's memory changes (the
        next switch)."""

        if holds:
            # Serving's memory may have come back without what the kernels kept beside it.
            self._executor.reset_inference_state()
            waiting, self._waiting, self._refusal = self._waiting, [], None
            self._publish(
                [reply for message in waiting for reply in self._scheduler.handle_message(message)]
            )
        elif refusal is not None:
            self._refusal = f"inference memory does not fit while training: {refusal}"
            refused, self._waiting = self._waiting, []
            self._publish(
                [RequestRejectedMsg(uid=message.uid, reason=self._refusal) for message in refused]
            )

    def settle_all(self) -> None:
        while self._in_flight:
            self._publish(self._settle_oldest())

    @property
    def completion(self) -> Completion | None:
        """The parked step's completion event, which the loop waits for with its input,
        until the step settles."""

        return self._parked

    def _settle_oldest(self) -> list[InferenceReply]:
        self._parked = None
        planned, result = self._in_flight.popleft()
        result.completion_event.synchronize()
        return self._scheduler.settle_step(planned, result.output()).replies

    def _log_stats(self) -> None:
        """Totals, cache occupancy and the executor's graph stats every interval while
        serving, and once as it drains."""

        if self.busy:
            now = time.monotonic()
            if self._stats_at is None:
                self._stats_at = now
            if now - self._stats_at < self.STATS_INTERVAL_SECONDS:
                return
            self._stats_at = now
        elif self._stats_at is None:
            return
        else:
            self._stats_at = None
            self._scheduler.audit_cache()
        logger.info_rank0(
            "Inference: %s; %s; %s",
            self._scheduler.totals,
            self._cache.occupancy,
            self._executor.graph_stats(),
        )


def step_digest(op: Forward | Round, admit: tuple[RequestSpec, ...]) -> int:
    """A step's digest, the same on every rank that planned the same values: the repr of
    its frozen dataclasses of ints, floats, bytes, enums and tuples, not their pickle,
    whose memo follows which objects a rank happened to share."""

    return zlib.crc32(repr((op, admit)).encode()) & 0x7FFFFFFF


__all__ = ["InferenceEngine", "step_digest"]
