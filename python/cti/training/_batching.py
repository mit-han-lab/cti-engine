"""CPU-only planning for training microbatches: a batching policy splits one logical
batch, the same way for every objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from cti.config import BatchingPolicy, DynamicBatching, StaticBatching

MicrobatchIndices: TypeAlias = tuple[tuple[int, ...], ...]


def plan_microbatches(policy: BatchingPolicy, sample_lengths: Sequence[int]) -> MicrobatchIndices:
    """The input-order index groups ``policy`` splits samples of ``sample_lengths`` into."""

    if isinstance(policy, StaticBatching):
        return _plan_static(policy, sample_lengths)
    return _plan_dynamic(policy, sample_lengths)


def _plan_static(policy: StaticBatching, sample_lengths: Sequence[int]) -> MicrobatchIndices:
    """Input-order groups of ``microbatch_size`` samples."""

    for index, length in enumerate(sample_lengths):
        if length > policy.max_tokens_per_sample:
            raise ValueError(
                f"sample_lengths[{index}]={length} exceeds "
                f"max_tokens_per_sample={policy.max_tokens_per_sample}"
            )
    return tuple(
        tuple(range(start, min(start + policy.microbatch_size, len(sample_lengths))))
        for start in range(0, len(sample_lengths), policy.microbatch_size)
    )


def _plan_dynamic(policy: DynamicBatching, sample_lengths: Sequence[int]) -> MicrobatchIndices:
    """Input-order groups within the token and sequence budgets."""

    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    current_tokens = 0
    for index, length in enumerate(sample_lengths):
        if length > policy.max_tokens_per_microbatch:
            raise ValueError(
                f"sample_lengths[{index}]={length} exceeds "
                f"max_tokens_per_microbatch={policy.max_tokens_per_microbatch}"
            )
        if current and (
            current_tokens + length > policy.max_tokens_per_microbatch
            or len(current) == policy.max_sequences_per_microbatch
        ):
            groups.append(tuple(current))
            current = []
            current_tokens = 0
        current.append(index)
        current_tokens += length
    if current:
        groups.append(tuple(current))
    return tuple(groups)


__all__ = ["MicrobatchIndices", "plan_microbatches"]
