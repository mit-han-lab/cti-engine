# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/scheduler/prefill.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch

from cti.protocol import AdmissionPriority
from cti.utils import div_ceil

from .request import Request

if TYPE_CHECKING:
    from cti.inference._step import RequestSpec
    from cti.protocol import UserMsg

    from .._cache import CacheManager, CacheUse
    from .encoder_cache import EncoderCache
    from .stops import RequestStops
    from .table import SlotTable


# A prompt forward's prompt and the positions it exports snapshots at.
Export = tuple[torch.Tensor, tuple[int, ...]]

# The most of a request's remaining output its cache reservation counts, as SGLang clips
# max_new_tokens in its estimate: a request without max_tokens would otherwise reserve the
# whole context. Running requests that outgrow it are retracted (``InferenceScheduler``).
ESTIMATED_OUTPUT_TOKENS = 4096


def estimated_output(request: Request) -> int:
    """The output tokens a request's cache reservation counts from where it stands."""

    remaining = request.max_device_len - max(request.device_len, request.prefill_len)
    return min(remaining, ESTIMATED_OUTPUT_TOKENS)


class SharedPrefixDeferral:
    """Holds a waiting prompt back, at most ``max_deferrals`` plans, while a prompt forward
    planned before it or still unsettled exports a snapshot of a prefix they share and
    restoring there saves at least ``threshold`` tokens over restoring now; the snapshot is
    restorable once that forward settles."""

    def __init__(
        self, cache_manager: CacheManager, *, threshold: int = 256, max_deferrals: int = 4
    ) -> None:
        # 256: on an RTX 5090 (Qwen3.5-9B) holding back won at savings of 256-1216 tokens,
        # idle or behind a running prefill.
        self._cache_manager = cache_manager
        self._threshold = threshold
        self._max_deferrals = max_deferrals
        # Submitted prompt forwards that export, until they settle.
        self._unsettled: dict[CacheUse, Export] = {}
        # A request leaving the queue drops out with it.
        self._deferrals: WeakKeyDictionary[Request, int] = WeakKeyDictionary()

    def track(self, request: Request, reservation: CacheUse) -> None:
        if positions := self._cache_manager.published_positions(reservation):
            self._unsettled[reservation] = (request.prefill_token_ids, positions)

    def untrack(self, reservation: CacheUse) -> None:
        self._unsettled.pop(reservation, None)

    def planned_exports(self, request: Request, start: int) -> Export | None:
        """The prompt of admitted ``request`` and the prefixes its forward planned from
        ``start`` makes restorable, if any."""

        positions = self._cache_manager.restorable_positions(
            request.slot, start, request.device_len
        )
        return (request.prefill_token_ids, positions) if positions else None

    def defers(self, request: Request, planned: Sequence[Export]) -> bool:
        """Whether to hold ``request`` back from a plan whose prompt forwards export
        ``planned``; a deferral counts toward the request's limit."""

        prompt = request.prefill_token_ids
        count = self._deferrals.get(request, 0)
        if (
            count >= self._max_deferrals
            or not request.reads_prefix_cache
            or not (self._unsettled or planned)
        ):
            return False
        exports = [*self._unsettled.values(), *planned]
        prefix_cache = self._cache_manager.prefix_cache
        best = max(
            (
                prefix_cache.restored_len(prompt, leader, position)
                for leader, positions in exports
                for position in positions
            ),
            default=0,
        )
        # Waiting must save the threshold, or all a hit could restore past what is cached now:
        # a sibling (the same prompt, as n > 1 sends) always waits, a hit then restoring all of
        # its prompt but the final token, even past a shorter prefix another prompt cached.
        cached = self._cache_manager.cached_prefix_len(prompt)
        restorable = prefix_cache.hit_limit(len(prompt)) - cached
        if best - cached < max(min(self._threshold, restorable), 1):
            return False
        self._deferrals[request] = count + 1
        return True


class PrefillQueue:
    """Own pending prompts and admit their next extensions into scheduler plans."""

    _PRIORITY_ORDER = (
        AdmissionPriority.HIGH,
        AdmissionPriority.NORMAL,
        AdmissionPriority.LOW,
    )

    def __init__(
        self,
        cache_manager: CacheManager,
        slots: SlotTable,
        *,
        encoder_cache: EncoderCache,
        lookahead: Callable[[Request], int] = lambda _request: 0,
        checks: bool = False,
    ) -> None:
        """``encoder_cache`` gives each admission its encoder rows; ``lookahead`` is the KV
        a request may write past its output budget; ``checks`` runs its requests' checks."""

        self._cache_manager = cache_manager
        self._slots = slots
        self._checks = checks
        # The specs of the requests admitted since the scheduler last took them.
        self._admitted: list[RequestSpec] = []
        self._encoder_cache = encoder_cache
        self._lookahead = lookahead
        self._continuations: deque[Request] = deque()
        self._waiting = {priority: deque[Request]() for priority in self._PRIORITY_ORDER}
        self.deferral = SharedPrefixDeferral(cache_manager)

    def enqueue(self, request: UserMsg, stops: RequestStops | None = None) -> None:
        pending = Request.from_message(request, stops, checks=self._checks)
        self._waiting[pending.priority].append(pending)

    def requeue(self, request: Request) -> None:
        """Queue a retracted request ahead of its priority's waiting ones."""

        self._waiting[request.priority].appendleft(request)

    def retract(self, request: Request) -> None:
        """Drop admitted prompt ``request``, which the scheduler retracts."""

        self._continuations.remove(request)

    def schedule_extensions(
        self,
        token_budget: int,
        *,
        reserved_size: int,
    ) -> tuple[Request, ...]:
        """Admit the next extensions within ``token_budget``, past ``reserved_size`` cache
        planned for running requests: continuations first, then waiting prompts by
        priority, first in first out; a deferred prompt keeps its place, and admission stops
        at the first prompt that cannot fit. An extension ends before an image whose encoder
        rows are not left and where the cache's snapshots need it to
        (``CacheManager.extension_end``); one that stops short to export
        (``CacheManager.export_stop``) leaves admission going."""

        if not self.has_pending:
            return ()
        budget, reserved = token_budget, reserved_size
        encoder = self._encoder_cache.budget()
        requests: list[Request] = []
        exports: list[Export] = []
        cache = self._cache_manager
        page_size = cache.page_size

        def extend(request: Request) -> tuple[int, bool]:
            """Plan the request's next extension: its tokens, and whether it stops to export."""

            nonlocal budget, reserved
            slot, start, prompt_end = request.slot, request.device_len, request.prefill_len
            end = cache.extension_end(slot, min(start + budget, prompt_end), prompt_end)
            if (stop := cache.export_stop(slot, start, end)) is not None:
                end = stop
            for span in request.images(start, end):
                if not encoder.take(span.key, span.num_tokens):
                    cut = max(span.offset, start)
                    # On the grid, unless that leaves nothing: the forward may hold an earlier
                    # image's rows this one waits on. One off the grid exports nothing.
                    end, stop = cache.extension_end(slot, cut, prompt_end), None
                    if end == start:
                        end = cut
                    break
            if not (tokens := end - start):
                return 0, False
            remaining = request.prefill_len - request.cached_len
            budget -= request.plan_prompt_extension(tokens)
            size = remaining + estimated_output(request) + self._lookahead(request)
            reserved += div_ceil(size, page_size) * page_size
            requests.append(request)
            if (export := self.deferral.planned_exports(request, start)) is not None:
                exports.append(export)
            return tokens, stop is not None

        for _ in range(len(self._continuations)):
            continuation = self._continuations[0]
            if budget <= 0:
                return tuple(requests)
            tokens, stops = extend(continuation)
            if not tokens:
                return tuple(requests)
            if not self._continuations.popleft().prompt_fully_scheduled:
                self._continuations.append(continuation)
                if not stops:
                    return tuple(requests)

        for priority in self._PRIORITY_ORDER:
            queue = self._waiting[priority]
            held: list[Request] = []
            try:
                while queue:
                    if budget <= 0 or self._slots.available_size == 0:
                        return tuple(requests)
                    request = queue[0]
                    if self.deferral.defers(request, exports):
                        held.append(queue.popleft())
                        continue
                    if not self._open(request, reserved):
                        return tuple(requests)
                    queue.popleft()
                    _, stops = extend(request)
                    if not request.prompt_fully_scheduled:
                        self._continuations.append(request)
                        if not stops:
                            return tuple(requests)
            finally:
                queue.extendleft(reversed(held))
        return tuple(requests)

    def _open(self, request: Request, reserved_size: int) -> bool:
        """Admit ``request`` into a free slot; ``False`` if its KV does not fit."""

        slot = self._slots.allocate(request)
        cached_len = self._cache_manager.try_open_slot(
            slot,
            request.prefill_token_ids,
            estimated_output(request),
            reserved_size=reserved_size,
            lookahead_tokens=self._lookahead(request),
            prefix=request.reads_prefix_cache,
        )
        if cached_len is None:
            self._slots.release(slot)
            return False
        request.admit(slot, cached_len=cached_len)
        self._admitted.append(request.spec)
        return True

    def take_admitted(self) -> tuple[RequestSpec, ...]:
        """The specs of the requests admitted into slots since the last call."""

        admitted, self._admitted = tuple(self._admitted), []
        return admitted

    def abort(self, uid: int) -> Request | None:
        for request in self._continuations:
            if request.uid == uid:
                self._continuations.remove(request)
                return request
        for waiting in self._waiting.values():
            for request in waiting:
                if request.uid == uid:
                    waiting.remove(request)
                    return request
        return None

    @property
    def continuations(self) -> tuple[Request, ...]:
        """The admitted prompts not yet extended to their end."""
        return tuple(self._continuations)

    @property
    def has_pending(self) -> bool:
        return bool(self._continuations) or any(self._waiting.values())

    @property
    def pending_ids(self) -> list[int]:
        waiting = (request for queue in self._waiting.values() for request in queue)
        return [request.uid for request in (*self._continuations, *waiting)]
