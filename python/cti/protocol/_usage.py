from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TokenUsage:
    """Token counts sealed when a generation request finishes.

    ``completion_tokens`` includes a terminal stop token. ``cached_prompt_tokens`` counts
    the prompt tokens served from the prefix cache, which the OpenAI server reports as
    ``usage.prompt_tokens_details.cached_tokens``.
    """

    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int = 0

    def __post_init__(self) -> None:
        counts = (self.prompt_tokens, self.completion_tokens, self.cached_prompt_tokens)
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Token counts must be non-negative integers")
        if self.cached_prompt_tokens > self.prompt_tokens:
            raise ValueError("Cached prompt tokens cannot exceed prompt tokens")

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Field-wise sum, for totals over requests."""
        return TokenUsage(
            **{
                field.name: getattr(self, field.name) + getattr(other, field.name)
                for field in fields(self)
            }
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
