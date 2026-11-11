# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors

"""What a caller asks of one request: how to sample it, the validated policy it carries
to the engine, and its admission priority; and the error a rejected request raises, with
whom the rejection blames."""

from __future__ import annotations

from collections.abc import Sequence, Sized
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from cti.sampling import LogprobMode, SamplingError, TokenSampling


class RejectionKind(StrEnum):
    """Who a rejection blames: the server (unavailable) or the request."""

    UNAVAILABLE = "unavailable"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"


class RequestRejectedError(ValueError):
    """A request was rejected before model execution began: ``request_id`` is its index in
    the call, ``kind`` whom the rejection blames."""

    def __init__(self, request_id: int, reason: str, kind: RejectionKind) -> None:
        self.request_id = request_id
        self.reason = reason
        self.kind = kind
        super().__init__(f"request {request_id} was rejected: {reason}")


@dataclass(kw_only=True)
class SamplingParams:
    """``max_new_tokens`` is required: the most tokens sampled, or None for up to the end
    of the model's context, which the engine resolves at admission. ``temperature`` 0 is
    greedy and takes no ``top_k`` or ``top_p``; ``top_k`` None keeps the whole vocabulary.
    ``stop`` strings and ``stop_token_ids`` end a request's output besides the model's
    stop tokens, which ``ignore_eos`` alone turns off; stop strings are matched in the
    output decoded with ``skip_special_tokens``.

    ``min_p`` keeps the tokens at least that fraction as likely as the likeliest, after
    top-k and top-p. The penalties follow vLLM: ``repetition_penalty`` scales the logits of
    tokens in the prompt or output, ``frequency_penalty`` subtracts per output occurrence
    and ``presence_penalty`` once per output token seen.

    ``logprobs`` returns each sampled token's log-probability and its ``top_logprobs``
    likeliest alternatives, read from the model's distribution (the log-softmax of its
    logits); with ``logprobs_processed``, from the distribution the token is drawn from,
    after penalties, temperature, top-k, top-p and min-p (a greedy token's after penalties
    only), as RL needs. ``logit_bias`` adds to the logits of its tokens (OpenAI's, in [-100, 100]);
    until ``min_tokens`` tokens are out, every stop token is masked (vLLM's), the model's
    and ``stop_token_ids``. ``prompt_logprobs`` returns each prompt token's raw
    log-probability and that many alternatives (the first token has none); such a prompt
    is computed in full, never restored from the prefix cache."""

    max_new_tokens: int | None
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    ignore_eos: bool = False
    seed: int | None = None
    logprobs: bool = False
    top_logprobs: int = 0
    logprobs_processed: bool = False
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    prompt_logprobs: int | None = None
    stop: Sequence[str] | None = None
    stop_token_ids: Sequence[int] | None = None
    skip_special_tokens: bool = True
    logit_bias: dict[int, float] | None = None
    min_tokens: int = 0

    @classmethod
    def specs(
        cls, prompts: Sized, sampling_params: SamplingParams | Sequence[SamplingParams]
    ) -> list[SamplingSpec]:
        """Each of a batch's ``prompts``' validated policy, from one ``sampling_params`` for
        all of them or one each; checked before anything is sent or preprocessed."""

        if isinstance(sampling_params, cls):
            sampling_params = [sampling_params] * len(prompts)
        if len(prompts) != len(sampling_params):
            raise ValueError("prompts and sampling_params must have the same length")
        if not prompts:
            raise ValueError("prompts must not be empty")
        if any(not isinstance(params, cls) for params in sampling_params):
            raise TypeError("sampling_params must contain SamplingParams values")
        return [params.snapshot() for params in sampling_params]

    def snapshot(self) -> SamplingSpec:
        """Freeze the request-local sampling policy carried into execution."""

        if isinstance(self.stop, str):
            raise SamplingError("stop", "must be a sequence of strings, not a string")
        for name in ("logprobs", "logprobs_processed"):
            if type(getattr(self, name)) is not bool:
                raise SamplingError(name, "must be a boolean")
        if self.logprobs_processed and not self.logprobs:
            raise SamplingError("logprobs_processed", "requires logprobs")
        logprobs = None
        if self.logprobs:
            logprobs = LogprobMode.PROCESSED if self.logprobs_processed else LogprobMode.RAW
        return SamplingSpec(
            TokenSampling(
                self.temperature,
                self.top_k,
                self.top_p,
                self.min_p,
                self.presence_penalty,
                self.frequency_penalty,
                self.repetition_penalty,
                logprobs,
                self.top_logprobs,
                self.prompt_logprobs,
                tuple(sorted((self.logit_bias or {}).items())),
                self.min_tokens,
                tuple(self.stop_token_ids or ()),
                self.ignore_eos,
            ),
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
            stop=tuple(self.stop or ()),
            skip_special_tokens=self.skip_special_tokens,
        )


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    """Immutable request-local sampling policy preserved as result provenance: the law its
    tokens are sampled under, and when its output ends (the law's stop tokens too)."""

    law: TokenSampling
    # None runs to the end of the context; admission resolves it.
    max_new_tokens: int | None
    seed: int | None
    stop: tuple[str, ...] = ()
    skip_special_tokens: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.law, TokenSampling):
            raise SamplingError("law", "must be a TokenSampling")
        if self.max_new_tokens is not None and (
            type(self.max_new_tokens) is not int or self.max_new_tokens <= 0
        ):
            raise SamplingError("max_new_tokens", "must be a positive integer or None")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise SamplingError("seed", "must be an integer in [0, 2**63) or None")
        if type(self.stop) is not tuple or any(
            type(stop) is not str or not stop for stop in self.stop
        ):
            raise SamplingError("stop", "must be a tuple of non-empty strings")
        if type(self.skip_special_tokens) is not bool:
            raise SamplingError("skip_special_tokens", "must be a boolean")
        if self.max_new_tokens is not None and self.law.min_tokens > self.max_new_tokens:
            raise SamplingError("min_tokens", "must not exceed max_new_tokens")


class AdmissionPriority(StrEnum):
    """Public priority tiers for waiting-request admission."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

    @classmethod
    def parse(cls, value: object) -> Self:
        if isinstance(value, cls):
            return value
        if type(value) is not str:
            raise TypeError("priority must be high, normal, or low")
        try:
            return cls(value)
        except ValueError as error:
            raise ValueError("priority must be high, normal, or low") from error


__all__ = [
    "AdmissionPriority",
    "RejectionKind",
    "RequestRejectedError",
    "SamplingParams",
    "SamplingSpec",
]
