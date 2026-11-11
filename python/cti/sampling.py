"""A request's sampling law: what every path that samples it reads. Light: no torch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import ClassVar


class SamplingError(ValueError):
    """A sampling field that breaks its rule, ``field`` named first in the message."""

    def __init__(self, field: str, rule: str) -> None:
        super().__init__(f"{field} {rule}")
        self.field = field


def finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SamplingError(name, "must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise SamplingError(name, "must be finite")
    return result


class LogprobMode(StrEnum):
    """Which distribution a returned log-probability reads (``SampledLogprobs``)."""

    RAW = "raw"
    PROCESSED = "processed"


_PENALTIES = ("presence_penalty", "frequency_penalty")


@dataclass(frozen=True, slots=True)
class TokenSampling:
    """One request's sampling law, which every path that samples it reads: ordinary steps,
    DFlash rounds and the training objective; validated once, here. ``top_k`` None and
    ``top_p`` 1 are identity, the only values greedy ``temperature`` 0 takes. ``logprobs``
    (None: none) returns each token's log-probability with ``top_logprobs`` alternatives;
    ``prompt_logprobs`` (None: none) scores the prompt with that many alternatives.
    ``logit_bias`` adds to the logits of its ``(token, bias)`` pairs, ascending by token,
    each inside the model's vocabulary (the engine rejects others at admission); until
    ``min_tokens`` tokens are out, the model's stop tokens and ``stop_token_ids``, the
    request's own, are masked, ``ignore_eos`` notwithstanding (vLLM's)."""

    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logprobs: LogprobMode | None = None
    top_logprobs: int = 0
    prompt_logprobs: int | None = None
    logit_bias: tuple[tuple[int, float], ...] = ()
    min_tokens: int = 0
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    # Alternatives a returned log-probability may carry (OpenAI's limit).
    MAX_TOP: ClassVar[int] = 20
    # OpenAI's range of a logit bias.
    MAX_BIAS: ClassVar[float] = 100.0
    # The least temperature a stochastic draw, and a policy gradient through it, may use.
    MIN_STOCHASTIC_TEMPERATURE: ClassVar[float] = 1e-6

    def __post_init__(self) -> None:
        for name in ("temperature", "top_p", "min_p", "repetition_penalty", *_PENALTIES):
            object.__setattr__(self, name, finite_real(name, getattr(self, name)))
        if self.temperature < 0:
            raise SamplingError("temperature", "must be >= 0")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise SamplingError("top_k", "must be None or a positive integer")
        if not 0.0 < self.top_p <= 1.0:
            raise SamplingError("top_p", "must be in (0, 1]")
        if self.temperature == 0.0 and self.top_k is not None:
            raise SamplingError("top_k", "must be None when temperature is 0 (greedy)")
        if self.temperature == 0.0 and self.top_p != 1.0:
            raise SamplingError("top_p", "must be 1 when temperature is 0 (greedy)")
        if not 0.0 <= self.min_p <= 1.0:
            raise SamplingError("min_p", "must be in [0, 1]")
        for name in _PENALTIES:
            if not -2.0 <= getattr(self, name) <= 2.0:
                raise SamplingError(name, "must be in [-2, 2]")
        if self.repetition_penalty <= 0:
            raise SamplingError("repetition_penalty", "must be positive")
        if self.logprobs is not None:
            # The wire codec carries the mode as its value, which this restores.
            if not isinstance(self.logprobs, str) or self.logprobs not in tuple(LogprobMode):
                raise SamplingError("logprobs", "must be a LogprobMode or None")
            object.__setattr__(self, "logprobs", LogprobMode(self.logprobs))
        top = f"an integer in [0, {self.MAX_TOP}]"
        if type(self.top_logprobs) is not int or not 0 <= self.top_logprobs <= self.MAX_TOP:
            raise SamplingError("top_logprobs", f"must be {top}")
        if self.top_logprobs and self.logprobs is None:
            raise SamplingError("top_logprobs", "requires logprobs")
        if self.prompt_logprobs is not None and (
            type(self.prompt_logprobs) is not int or not 0 <= self.prompt_logprobs <= self.MAX_TOP
        ):
            raise SamplingError("prompt_logprobs", f"must be None or {top}")
        tokens = [token for token, _ in self.logit_bias]
        if any(type(token) is not int or token < 0 for token in tokens) or tokens != sorted(
            set(tokens)
        ):
            raise SamplingError("logit_bias", "tokens must be distinct non-negative integers")
        bias = tuple((token, finite_real("logit_bias", b)) for token, b in self.logit_bias)
        if any(abs(b) > self.MAX_BIAS for _, b in bias):
            raise SamplingError(
                "logit_bias", f"values must be in [-{self.MAX_BIAS:g}, {self.MAX_BIAS:g}]"
            )
        object.__setattr__(self, "logit_bias", bias)
        if type(self.min_tokens) is not int or self.min_tokens < 0:
            raise SamplingError("min_tokens", "must be a non-negative integer")
        if any(type(token) is not int or token < 0 for token in self.stop_token_ids):
            raise SamplingError("stop_token_ids", "must be non-negative integers")
        # One spelling on every rank: the lockstep digest compares reprs.
        object.__setattr__(self, "stop_token_ids", tuple(sorted(set(self.stop_token_ids))))
        if type(self.ignore_eos) is not bool:
            raise SamplingError("ignore_eos", "must be a boolean")

    @property
    def is_greedy(self) -> bool:
        # Top-p cannot widen a point mass.
        return self.temperature == 0.0 or self.top_k == 1

    def top_k_within(self, vocab_size: int) -> int:
        """How many likeliest tokens top-k keeps of ``vocab_size``: all when it is None."""

        return vocab_size if self.top_k is None else min(self.top_k, vocab_size)

    @classmethod
    def stochastic_temperature(cls, value: object) -> float:
        """``value`` as the temperature of a stochastic draw: finite and at least
        ``MIN_STOCHASTIC_TEMPERATURE``."""

        temperature = finite_real("temperature", value)
        if temperature < cls.MIN_STOCHASTIC_TEMPERATURE:
            raise ValueError(
                f"temperature must be at least {cls.MIN_STOCHASTIC_TEMPERATURE} for stochastic "
                "sampling"
            )
        return temperature

    @property
    def has_penalties(self) -> bool:
        return (self.presence_penalty, self.frequency_penalty, self.repetition_penalty) != (
            0.0,
            0.0,
            1.0,
        )

    @property
    def steers(self) -> bool:
        """Whether a logit bias or a min_tokens mask changes its logits."""

        return bool(self.logit_bias) or self.min_tokens > 0

    @property
    def processes_logits(self) -> bool:
        """Whether its logits change before temperature: steering or penalties."""

        return self.steers or self.has_penalties

    @property
    def reads_processed(self) -> bool:
        """Whether its log-probabilities read the distribution it draws from."""

        return self.logprobs is LogprobMode.PROCESSED

    @property
    def has_min_p(self) -> bool:
        return self.min_p > 0 and not self.is_greedy

    @property
    def needs_processing(self) -> bool:
        """Whether its logits need more than temperature, top-k and top-p."""

        return self.logprobs is not None or self.processes_logits or self.has_min_p

    @property
    def is_plain_categorical(self) -> bool:
        """Whether it draws from the temperature softmax alone, as a policy gradient's
        log-probabilities assume."""

        return (
            not self.is_greedy
            and self.top_k is None
            and self.top_p == 1.0
            and not self.has_min_p
            and not self.processes_logits
        )


@dataclass(frozen=True, slots=True)
class TokenLogprobs:
    """A token's log-probability and its likeliest ``(token, log-probability)``
    alternatives, best first: what every layer returns of a sampled or prompt token."""

    token_id: int
    logprob: float
    top: tuple[tuple[int, float], ...] = ()

    @property
    def token_ids(self) -> tuple[int, ...]:
        """The token's id and its alternatives'."""
        return (self.token_id, *(token_id for token_id, _ in self.top))


__all__ = ["LogprobMode", "SamplingError", "TokenLogprobs", "TokenSampling", "finite_real"]
