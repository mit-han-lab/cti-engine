# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/core.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING

import torch

from cti.inference._step import RequestSpec
from cti.protocol import AdmissionPriority

if TYPE_CHECKING:
    from cti.inference._control._cache import CacheUse
    from cti.inference._step import QueryKind, Round, Sampled
    from cti.multimodal import ImageInputs, ImageSpan, RopePositionLayout
    from cti.protocol import SamplingSpec, UserMsg
    from cti.sampling import TokenLogprobs, TokenSampling

    from .stops import RequestStops, StopHit
    from .table import RequestSlot


def image_key_ids(span: ImageSpan) -> torch.Tensor:
    """An image's prefix-cache key over its placeholders: its ``key`` as 8 negative int32
    words, repeated over the span, so no text token equals one."""

    words = struct.unpack("<8i", span.key)
    key = torch.tensor([word | -(1 << 31) for word in words], dtype=torch.int32)
    return key.repeat(-(-span.num_tokens // 8))[: span.num_tokens]


class Request:
    """InferenceScheduler-owned state for one request from enqueue through reclamation.

    A prompt holding images keeps their placeholder spans, its rotary position layout and
    their pixels for its lifetime. Its placeholders hold its prefix-cache key ids
    (``image_key_ids``); prompt extensions may end inside an image, which the encoder cache
    keeps for the next. A retracted request gives up its slot and resumes in a new one,
    prefilling every token it had committed (``prefill_len``).
    """

    def __init__(
        self,
        *,
        uid: int,
        prompt_token_ids: torch.Tensor,
        policy: SamplingSpec,
        priority: AdmissionPriority = AdmissionPriority.NORMAL,
        images: ImageInputs | None = None,
        rope_layout: RopePositionLayout | None = None,
        stops: RequestStops | None = None,
        checks: bool = False,
    ) -> None:
        """``stops`` are the request's own stop conditions, if it has any. The scheduler's
        admission decided the prompt and images; ``checks`` re-checks them and every step's
        transition."""
        image_spans = () if images is None else images.spans
        if checks:
            if not prompt_token_ids.is_cpu:
                raise ValueError("prompt_token_ids must be a CPU tensor")
            if prompt_token_ids.dtype is not torch.int32:
                raise TypeError("prompt_token_ids must use torch.int32")
            if prompt_token_ids.ndim != 1 or len(prompt_token_ids) == 0:
                raise ValueError("prompt_token_ids must be a non-empty one-dimensional tensor")
            if (images is None) != (rope_layout is None):
                raise ValueError("images and their rope layout must be present together")
            if image_spans and image_spans[-1].end > len(prompt_token_ids):
                raise ValueError("image placeholders must lie within the prompt")
        self._checks = checks
        self._uid = uid
        key_ids = prompt_token_ids.clone() if image_spans else prompt_token_ids
        for span in image_spans:
            key_ids[span.offset : span.end] = image_key_ids(span)
        self._prompt_token_ids = key_ids
        self._vocabulary_prompt_ids = prompt_token_ids
        self._policy = policy
        self._stops = stops
        self._priority = AdmissionPriority.parse(priority)
        self._image_spans = image_spans
        self._rope_layout = rope_layout
        self._spec: RequestSpec | None = None
        self._pixel_values = None if images is None else images.pixel_values
        # Image i's pixel rows are [_patch_offsets[i], _patch_offsets[i + 1]).
        self._patch_offsets = tuple(
            accumulate((span.num_patches for span in image_spans), initial=0)
        )

        self._slot: RequestSlot | None = None
        self._cached_len = 0
        self._device_len = 0
        # The tokens its admission prefills: the prompt, or all it committed before retraction.
        self._prefill_len = len(key_ids)
        self._admitted_len = 0
        self._num_cached_prompt_tokens: int | None = None
        self._sampled_token_ids: list[int] = []
        self._materialized_token_ids: torch.Tensor | None = key_ids
        self._is_finished = False
        # Prompt log-probabilities gathered chunk by chunk, until the first output token.
        self._prompt_logprobs: list[TokenLogprobs | None] | None = (
            None if policy.law.prompt_logprobs is None else [None]
        )

    @classmethod
    def from_message(
        cls, message: UserMsg, stops: RequestStops | None = None, *, checks: bool = False
    ) -> Request:
        return cls(
            uid=message.uid,
            prompt_token_ids=message.input_ids,
            policy=message.sampling,
            priority=message.priority,
            images=message.images,
            rope_layout=message.rope_layout,
            stops=stops,
            checks=checks,
        )

    # ================================ Identity =================================

    @property
    def uid(self) -> int:
        return self._uid

    @property
    def policy(self) -> SamplingSpec:
        """Its sampling policy, validated where the request was made."""

        return self._policy

    @property
    def sampling(self) -> TokenSampling:
        """The law its tokens are sampled under."""

        return self._policy.law

    @property
    def reads_prefix_cache(self) -> bool:
        """Whether its prompt may restore a cached prefix: not when it scores the prompt,
        which needs every prompt row's hidden state."""

        return self.sampling.prompt_logprobs is None

    def own_stop(self, tokens: list[int]) -> StopHit | None:
        """Where the request's own stop conditions end its output among ``tokens``, its next
        output tokens; the model's stop tokens are checked before."""

        return None if self._stops is None else self._stops.find(tokens)

    @property
    def priority(self) -> AdmissionPriority:
        return self._priority

    # ============================== Token state ================================

    @property
    def prompt_token_ids(self) -> torch.Tensor:
        return self._prompt_token_ids

    def vocabulary_ids(self, start: int, end: int) -> torch.Tensor:
        """Positions ``[start, end)`` as the vocabulary has them: an image's placeholder ids,
        rows the encoder output fills; past the prompt, the sampled tokens."""

        if end <= self.num_prompt_tokens:
            return self._vocabulary_prompt_ids[start:end]
        sampled = torch.tensor(self._sampled_token_ids, dtype=torch.int32)
        return torch.cat((self._vocabulary_prompt_ids, sampled))[start:end]

    @property
    def prefill_len(self) -> int:
        return self._prefill_len

    @property
    def prefill_token_ids(self) -> torch.Tensor:
        """The key ids its admission prefills and publishes."""

        return self.token_ids[: self._prefill_len]

    @property
    def token_ids(self) -> torch.Tensor:
        """Materialize the committed prompt and sampled tokens when required."""

        if self._materialized_token_ids is None:
            sampled = torch.tensor(self._sampled_token_ids, dtype=torch.int32)
            self._materialized_token_ids = torch.cat((self._prompt_token_ids, sampled))
        return self._materialized_token_ids

    @property
    def max_output_tokens(self) -> int:
        return self._policy.max_new_tokens

    @property
    def num_prompt_tokens(self) -> int:
        return len(self._prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._sampled_token_ids)

    @property
    def num_committed_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def scores_prompt(self) -> bool:
        """Whether its forwards score prompt tokens: until its first output token."""

        return self._prompt_logprobs is not None

    def record_prompt_logprobs(self, start: int, scored: Sampled) -> None:
        """Gather a prompt chunk's logprobs, from prompt position ``start`` on, which continue
        the gathered ones."""

        gathered = self._prompt_logprobs
        if gathered is None or start != len(gathered):
            raise RuntimeError(f"request {self.uid}'s prompt log-probabilities are out of order")
        top = self.sampling.prompt_logprobs
        gathered.extend(scored.at(index, top) for index in range(len(scored.token_ids())))

    def take_prompt_logprobs(self) -> tuple[TokenLogprobs | None, ...] | None:
        """The whole prompt's log-probabilities, once, with the first output token."""

        gathered, self._prompt_logprobs = self._prompt_logprobs, None
        if gathered is None:
            return None
        if len(gathered) != self.num_prompt_tokens:
            raise RuntimeError(f"request {self.uid} scored {len(gathered)} prompt tokens")
        return tuple(gathered)

    def append_sampled_token(self, token_id: int) -> None:
        if self._checks and type(token_id) is not int:
            raise TypeError("sampled token ID must be an integer")
        self._sampled_token_ids.append(token_id)
        self._materialized_token_ids = None

    def append_block(self, token_ids: list[int], *, cached_len: int) -> None:
        """Append a settled block's visible tokens; ``cached_len`` is its new KV frontier.

        The frontier advances past the anchor and stays below the last appended token,
        which is the next anchor or the finishing token.
        """

        if not token_ids or any(type(token_id) is not int for token_id in token_ids):
            raise ValueError("a block appends at least one integer token")
        committed_len = self.num_committed_tokens + len(token_ids)
        if not self._cached_len < cached_len < committed_len:
            raise ValueError(f"block frontier {cached_len} is outside the appended tokens")
        self._sampled_token_ids.extend(token_ids)
        self._materialized_token_ids = None
        self._cached_len = cached_len
        self._device_len = committed_len

    # ============================== Image state ================================

    @property
    def rope_layout(self) -> RopePositionLayout | None:
        return self._rope_layout

    @property
    def spec(self) -> RequestSpec:
        """The constants the executor holds for this request's slot, frozen at admission."""

        if self._spec is None:
            raise RuntimeError(f"request {self._uid} is not admitted")
        return self._spec

    @property
    def decode_rope_offset(self) -> int:
        """Rotary minus sequence position of every token after the prompt."""

        return 0 if self._rope_layout is None else self._rope_layout.decode_offset

    def planned_images(self) -> Iterator[tuple[ImageSpan, torch.Tensor]]:
        """Yield each image the planned prompt extension covers part of, with its pixel rows."""

        start, end = self._cached_len, self._device_len
        for index, span in enumerate(self._image_spans):
            if span.end <= start:
                continue
            if span.offset >= end:
                break
            if self._pixel_values is None:
                raise RuntimeError(f"request {self._uid} has released its image pixels")
            yield (
                span,
                self._pixel_values[self._patch_offsets[index] : self._patch_offsets[index + 1]],
            )

    # ============================ Execution state ==============================

    @property
    def is_admitted(self) -> bool:
        return self._slot is not None

    @property
    def slot(self) -> RequestSlot:
        if self._slot is None:
            raise RuntimeError(f"request {self._uid} has not been admitted")
        return self._slot

    @property
    def cached_len(self) -> int:
        return self._cached_len

    @property
    def device_len(self) -> int:
        return self._device_len

    @property
    def num_cached_prompt_tokens(self) -> int:
        """The prompt tokens its first admission restored from the prefix cache."""

        return self._num_cached_prompt_tokens or 0

    @property
    def admitted_len(self) -> int:
        """The tokens its current admission restored from the prefix cache."""

        return self._admitted_len

    @property
    def max_device_len(self) -> int:
        return self.num_prompt_tokens + self.max_output_tokens

    @property
    def num_submitted_output_tokens(self) -> int:
        return max(self._device_len - self.num_prompt_tokens, 0)

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self._device_len

    @property
    def extend_len(self) -> int:
        return self._device_len - self._cached_len

    @property
    def prompt_fully_scheduled(self) -> bool:
        return self._device_len >= self._prefill_len

    @property
    def can_decode(self) -> bool:
        return self.prompt_fully_scheduled and self.remain_len > 0

    def admit(self, slot: RequestSlot, *, cached_len: int) -> None:
        if self._slot is not None:
            raise RuntimeError(f"request {self._uid} is already admitted")
        if not 0 <= cached_len < self._prefill_len:
            raise ValueError("cached_len must leave at least one prompt token to execute")
        self._slot = slot
        self._cached_len = cached_len
        self._device_len = cached_len
        self._admitted_len = cached_len
        if self._num_cached_prompt_tokens is None:
            self._num_cached_prompt_tokens = cached_len
        self._spec = RequestSpec(
            slot=slot.index,
            prompt_len=self.num_prompt_tokens,
            output_end=self.max_device_len,
            sampling=self.sampling,
            seed=self._policy.seed,
            rope_offset=self.decode_rope_offset,
        )

    def images(self, start: int, end: int) -> tuple[ImageSpan, ...]:
        """The images prompt positions ``[start, end)`` cover part of."""

        return tuple(span for span in self._image_spans if span.offset < end and span.end > start)

    @property
    def split_image(self) -> bytes | None:
        """The key of the image the planned prompt extensions end inside, if any."""

        end = self._device_len
        if end == self._admitted_len:  # nothing planned since admission
            return None
        return next((span.key for span in self._image_spans if span.offset < end < span.end), None)

    def plan_prompt_extension(self, token_budget: int) -> int:
        if self._checks:
            if not self.is_admitted:
                raise RuntimeError(f"request {self._uid} has not been admitted")
            if self.prompt_fully_scheduled:
                raise RuntimeError(f"request {self._uid} has completed prefill")
            if type(token_budget) is not int or token_budget <= 0:
                raise ValueError("token_budget must be a positive integer")

        num_tokens = min(token_budget, self._prefill_len - self._device_len)
        self._device_len += num_tokens
        return num_tokens

    def mark_submitted(self) -> None:
        if self._checks and self.extend_len <= 0:
            raise RuntimeError(f"request {self._uid} has no planned work to submit")
        self._cached_len = self._device_len
        if self.prompt_fully_scheduled and self.remain_len > 0:
            self._device_len += 1

    # ============================= Terminal state ==============================

    @property
    def is_finished(self) -> bool:
        return self._is_finished

    def mark_finished(self) -> bool:
        if self._is_finished:
            return False
        self._is_finished = True
        self._pixel_values = None
        return True

    def retract(self) -> None:
        """Give up its settled, closed slot: its next admission prefills every token it
        committed, which the prefix cache may hold."""

        self._slot = self._spec = None
        self._prefill_len = self.num_committed_tokens
        self._cached_len = self._device_len = 0

    def __repr__(self) -> str:
        return (
            f"Request(uid={self._uid}, priority={self._priority.value}, slot={self._slot}, "
            f"cached_len={self._cached_len}, device_len={self._device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass(frozen=True, slots=True)
class RequestBatch:
    """One scheduler plan with prompt extensions ordered before decode queries."""

    extend_requests: tuple[Request, ...] = ()
    decode_requests: tuple[Request, ...] = ()

    @property
    def requests(self) -> tuple[Request, ...]:
        return self.extend_requests + self.decode_requests


@dataclass(frozen=True, slots=True, eq=False)
class RowClaim:
    """One row of a planned step: its request, its cache use, and what settles it. A
    forward row has its ``query_kind`` and samples when ``max_outputs`` is 1; a round row
    (``query_kind`` None) may output up to ``max_outputs`` tokens."""

    request: Request
    use: CacheUse
    query_kind: QueryKind | None
    max_outputs: int

    @property
    def slot(self) -> RequestSlot:
        return self.use.slot


@dataclass(frozen=True, slots=True)
class DecodeStep:
    """A planned step: a forward over ``forward``'s requests, a round and its rows, or
    both, the round's target forward then running the forward's prompt extensions."""

    forward: RequestBatch | None
    rounds: tuple[Round, tuple[RowClaim, ...]] | None
