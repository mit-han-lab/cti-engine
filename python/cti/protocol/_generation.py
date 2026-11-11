from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, List, Literal, Protocol, TypeAlias, TypedDict

from cti.multimodal import ProcessedPrompt
from cti.sampling import TokenLogprobs

from ._request import SamplingSpec

if TYPE_CHECKING:
    from ._usage import TokenUsage

# A bounded sampling command's prompt: text, its token ids, or a prompt the model's input
# processor expanded, the form a prompt holding images takes.
SamplingPrompt: TypeAlias = str | list[int] | ProcessedPrompt


class _Ending(Protocol):
    """How a request ended: offline its ``RequestEnd``, online its last ``UserReply``."""

    usage: TokenUsage | None
    stop_str: str | None
    max_new_tokens: int | None

    @property
    def finish_reason(self) -> Literal["stop", "length"] | None: ...


@dataclass(frozen=True, slots=True)
class SampledSequence:
    """Exact terminal sampling evidence before serving-specific projection.

    ``cached_prompt_tokens`` counts the prompt tokens served from the prefix cache, the
    count the OpenAI server reports as ``usage.prompt_tokens_details.cached_tokens``.
    ``stop_str`` is the stop string whose completing token ends the sample, if one did.
    """

    request_id: int
    prompt_token_ids: tuple[int, ...]
    sampled_token_ids: tuple[int, ...]
    sampling_logprobs: tuple[float, ...] | None
    cached_prompt_tokens: int
    sampling: SamplingSpec
    finish_reason: Literal["stop", "length"]
    stop_str: str | None = None
    # Each sampled token's ``top_logprobs`` likeliest (token, log-probability), best first.
    sampling_top_logprobs: tuple[tuple[tuple[int, float], ...], ...] | None = None
    # Each prompt token's raw log-probability and alternatives; None for the first token.
    prompt_logprobs: tuple[TokenLogprobs | None, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.request_id) is not int or self.request_id < 0:
            raise ValueError("request_id must be a non-negative integer")
        for name, token_ids in (
            ("prompt_token_ids", self.prompt_token_ids),
            ("sampled_token_ids", self.sampled_token_ids),
        ):
            if type(token_ids) is not tuple:
                raise TypeError(f"{name} must be a tuple")
            if not token_ids:
                raise ValueError(f"{name} must not be empty")
            if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
                raise ValueError(f"{name} must contain non-negative integers")
        if self.sampling_logprobs is not None:
            if type(self.sampling_logprobs) is not tuple or len(self.sampling_logprobs) != len(
                self.sampled_token_ids
            ):
                raise ValueError(
                    "sampling_logprobs must contain one value per sampled token or be None"
                )
            if any(
                type(logprob) is not float or not math.isfinite(logprob)
                for logprob in self.sampling_logprobs
            ):
                raise ValueError("sampling_logprobs must contain finite native floats")
        if type(self.cached_prompt_tokens) is not int or not 0 <= self.cached_prompt_tokens <= len(
            self.prompt_token_ids
        ):
            raise ValueError("cached_prompt_tokens must be between zero and the prompt token count")
        if not isinstance(self.sampling, SamplingSpec):
            raise TypeError("sampling must be a SamplingSpec")
        top = self.sampling_top_logprobs
        if (top is not None) != bool(self.sampling.law.top_logprobs) or (
            top is not None
            and (
                len(top) != len(self.sampled_token_ids)
                or any(len(alternatives) != self.sampling.law.top_logprobs for alternatives in top)
            )
        ):
            raise ValueError(
                "sampling_top_logprobs must hold top_logprobs alternatives per sampled token "
                "exactly when they were requested"
            )
        if (self.prompt_logprobs is not None) != (
            self.sampling.law.prompt_logprobs is not None
        ) or (
            self.prompt_logprobs is not None
            and len(self.prompt_logprobs) != len(self.prompt_token_ids)
        ):
            raise ValueError("prompt_logprobs must cover the prompt exactly when requested")
        if (self.sampling.law.logprobs is not None) != (self.sampling_logprobs is not None):
            raise ValueError(
                "sampling_logprobs must be present exactly when selected log-probabilities "
                "were requested"
            )
        if self.finish_reason not in ("stop", "length"):
            raise ValueError("finish_reason must be 'stop' or 'length'")
        if self.stop_str is not None and (
            self.finish_reason != "stop" or self.stop_str not in self.sampling.stop
        ):
            raise ValueError("stop_str must be one of the sampling's stop strings that ended it")

    @classmethod
    def assemble(
        cls,
        request_id: int,
        prompt_token_ids: tuple[int, ...],
        sampling: SamplingSpec,
        *,
        token_ids: Sequence[int],
        logprobs: Sequence[TokenLogprobs],
        prompt_logprobs: tuple[TokenLogprobs | None, ...] | None,
        ending: _Ending,
    ) -> SampledSequence:
        """A request's sample from its replies, offline or online alike: each token's id and,
        when asked, log-probabilities, its prompt's, and its last reply's ending. The
        sampling's limit is the one admission applied, as up to the end of the context
        resolves there."""

        law = sampling.law
        if law.logprobs is not None and len(logprobs) != len(token_ids):
            raise RuntimeError(f"request {request_id} has no selected-token log-probability")
        return cls(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampled_token_ids=tuple(token_ids),
            sampling_logprobs=None if law.logprobs is None else tuple(t.logprob for t in logprobs),
            cached_prompt_tokens=ending.usage.cached_prompt_tokens,
            sampling=replace(sampling, max_new_tokens=ending.max_new_tokens),
            finish_reason=ending.finish_reason,
            stop_str=ending.stop_str,
            prompt_logprobs=prompt_logprobs,
            sampling_top_logprobs=tuple(t.top for t in logprobs) if law.top_logprobs else None,
        )

    def visible_text(self, text: str) -> str:
        """The output ``text``, cut where the stop string that ended the sample begins."""

        if self.stop_str is not None and (end := text.find(self.stop_str)) >= 0:
            return text[:end]
        return text

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        """Return the serving projection, excluding a terminal stop token."""

        if self.finish_reason == "stop" and self.stop_str is None:
            return self.sampled_token_ids[:-1]
        return self.sampled_token_ids


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """One bounded inference result with exact sampling and revision evidence."""

    samples: tuple[SampledSequence, ...]
    texts: tuple[str, ...]
    model_revision: int

    def __post_init__(self) -> None:
        if type(self.samples) is not tuple or not self.samples:
            raise TypeError("samples must be a non-empty tuple")
        if any(not isinstance(sample, SampledSequence) for sample in self.samples):
            raise TypeError("samples must contain SampledSequence values")
        request_ids = tuple(sample.request_id for sample in self.samples)
        if request_ids != tuple(range(len(self.samples))):
            raise ValueError("samples must be in contiguous request-ID order")
        if type(self.texts) is not tuple or len(self.texts) != len(self.samples):
            raise TypeError("texts must contain one string per sample")
        if any(type(text) is not str for text in self.texts):
            raise TypeError("texts must contain strings")
        if type(self.model_revision) is not int or self.model_revision < 0:
            raise ValueError("model_revision must be a non-negative integer")

    def require_revision(self, model_revision: int) -> None:
        if type(model_revision) is not int or model_revision < 0:
            raise ValueError("model_revision must be a non-negative integer")
        if self.model_revision != model_revision:
            raise ValueError(
                f"sampling revision {self.model_revision} does not match policy revision "
                f"{model_revision}"
            )

    @classmethod
    def at_revision(
        cls, samples: tuple[SampledSequence, ...], texts: tuple[str, ...], revisions: Iterable[int]
    ) -> SamplingResult:
        """The result of a command whose requests ended at ``revisions``, the revision each
        ran at as the engine stamped it; an optimizer step between them fails the command."""

        (revision, *others) = set(revisions)
        if others:
            raise RuntimeError("the model changed while sampling; resample")
        return cls(samples, texts, revision)

    def generation_outputs(self) -> List[GenerationOutput]:
        """Project exact samples into the existing serving-compatible result."""

        return [
            {"text": text, "token_ids": list(sample.output_token_ids)}
            for sample, text in zip(self.samples, self.texts, strict=True)
        ]


class GenerationOutput(TypedDict):
    text: str
    token_ids: List[int]
