"""Training objectives and the metrics a forward/backward reports: SFT and GRPO, each a
value a batch carries, over any token layout."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import TYPE_CHECKING

import torch

from cti.protocol import SamplingResult, SamplingSpec
from cti.sampling import LogprobMode, TokenSampling

from ._errors import InvalidGradientError, TrainingInputError
from ._protocol import ForwardBackwardCompletion

if TYPE_CHECKING:
    from cti.models.training_contract import TrainingModel

    from ._batch import DeviceRows, TrainingBatch

_GRPO_REWARD_EPSILON = 1e-6


class Reduction(StrEnum):
    """How a metric folds over microbatches: summed, averaged by target tokens, or the
    largest."""

    SUM = "sum"
    MEAN = "mean"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class Metrics:
    """One submission's device metrics: its loss sum first, each with its reduction, the
    count its gradients normalize by and the target tokens its means weigh."""

    values: tuple[tuple[str, Reduction, torch.Tensor], ...]
    normalization_count: int
    num_targets: int

    @property
    def loss_sum(self) -> torch.Tensor:
        return self.values[0][2]

    def fold(self, later: Metrics) -> Metrics:
        """This microbatch's metrics followed by ``later``'s."""

        tokens = self.num_targets + later.num_targets
        fold = {
            Reduction.SUM: torch.add,
            Reduction.MEAN: lambda a, b: (a * self.num_targets + b * later.num_targets) / tokens,
            Reduction.MAX: torch.maximum,
        }
        values = tuple(
            (name, reduction, fold[reduction](value, other))
            for (name, reduction, value), (_, _, other) in zip(
                self.values, later.values, strict=True
            )
        )
        return Metrics(values, self.normalization_count + later.normalization_count, tokens)

    def relocate(self, device: torch.device | str) -> Metrics:
        """Detached copies on ``device``."""

        values = tuple(
            (name, reduction, value.detach().to(device, copy=True))
            for name, reduction, value in self.values
        )
        return Metrics(values, self.normalization_count, self.num_targets)

    @staticmethod
    def complete(results: Sequence[Metrics]) -> tuple[ForwardBackwardCompletion, ...]:
        """Read ``results`` back together; a non-finite metric raises InvalidGradientError."""

        flat = [value.detach().float() for result in results for _, _, value in result.values]
        read = iter(torch.stack(flat).tolist() if flat else ())
        completions = []
        for result in results:
            metrics = {name: next(read) for name, _, _ in result.values}
            if not all(map(math.isfinite, metrics.values())):
                raise InvalidGradientError("Training forward produced a non-finite loss or metric")
            loss_sum = metrics.pop(result.values[0][0])
            completions.append(
                ForwardBackwardCompletion(loss_sum, result.normalization_count, metrics)
            )
        return tuple(completions)


@dataclass(frozen=True, slots=True)
class SFT:
    """Summed target-token cross-entropy, normalized by the target-token count."""

    def validate(self, batch: TrainingBatch) -> None:
        """Any targets train."""

    def admit(self, batch: TrainingBatch, model_revision: int) -> None:
        """Any model revision may train on supervised data."""

    def normalization_count(self, batch: TrainingBatch) -> int:
        return batch.num_targets

    def supplied_token_ids(self, batch: TrainingBatch) -> torch.Tensor:
        return batch.tokens.token_ids

    def metrics(
        self,
        model: TrainingModel,
        batch: TrainingBatch,
        indices: tuple[int, ...],
        hidden_states: torch.Tensor,
        rows: DeviceRows,
    ) -> Metrics:
        num_targets = rows.target_ids.numel()
        loss_sum = model.training_loss(hidden_states, rows.target_ids)
        return Metrics((("loss_sum", Reduction.SUM, loss_sum),), num_targets, num_targets)


@dataclass(frozen=True)
class GRPO:
    """Reference-free clipped GRPO over a rollout's response tokens, its targets,
    normalized by sequences.

    ``sampling_logprobs`` holds each target's sampling log-probability in target order;
    the rest hold one value per sequence. The training runner recomputes the
    same-revision old-policy denominator.
    """

    sampling_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    group_size: int
    rollout_revision: int
    temperature: float
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2

    @classmethod
    def of(
        cls,
        sampling: SamplingResult,
        rewards: Sequence[float],
        *,
        group_size: int,
    ) -> tuple[GRPO, list[tuple[int, ...]], list[tuple[int, ...]]]:
        """The objective of a grouped rollout, and its samples' prompts and responses."""

        if not isinstance(sampling, SamplingResult):
            raise TypeError("sampling must be a SamplingResult")
        if len(rewards) != len(sampling.samples):
            raise ValueError("rewards must contain one value per sample")
        temperatures = {_require_grpo_sampling(sample.sampling) for sample in sampling.samples}
        if len(temperatures) != 1:
            raise ValueError("one GRPO batch requires a uniform sampling temperature")
        if any(sample.sampling_logprobs is None for sample in sampling.samples):
            raise ValueError("rollout samples must preserve sampling provenance")
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        objective = cls(
            sampling_logprobs=torch.tensor(
                [logprob for sample in sampling.samples for logprob in sample.sampling_logprobs],
                dtype=torch.float32,
            ),
            rewards=reward_tensor,
            advantages=group_relative_advantages(reward_tensor, group_size),
            group_size=group_size,
            rollout_revision=sampling.model_revision,
            temperature=temperatures.pop(),
        )
        return (
            objective,
            [sample.prompt_token_ids for sample in sampling.samples],
            [sample.sampled_token_ids for sample in sampling.samples],
        )

    def __post_init__(self) -> None:
        floats = {
            "sampling_logprobs": self.sampling_logprobs,
            "rewards": self.rewards,
            "advantages": self.advantages,
        }
        for name, tensor in floats.items():
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise TypeError(f"{name} must be a CPU tensor")
            if tensor.ndim != 1:
                raise TypeError(f"{name} must be a vector")
            if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be a finite float32 vector")
        sequences = self.rewards.numel()
        if self.advantages.numel() != sequences:
            raise ValueError("rewards and advantages need one value per row")
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError("group_size must be an integer of at least two")
        if sequences % self.group_size != 0:
            raise ValueError("batch size must be a multiple of group_size")
        if type(self.rollout_revision) is not int or self.rollout_revision < 0:
            raise ValueError("rollout_revision must be a non-negative integer")
        object.__setattr__(
            self, "temperature", TokenSampling.stochastic_temperature(self.temperature)
        )
        for name, value in (("epsilon_low", self.epsilon_low), ("epsilon_high", self.epsilon_high)):
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value < 1:
                raise ValueError(f"{name} must be a real number in [0, 1)")

    def validate(self, batch: TrainingBatch) -> None:
        """Each row's targets must be exactly its sampled response, after a prompt its group
        shares, with one sampling log-probability per target."""

        tokens = batch.tokens
        if self.rewards.numel() != tokens.batch_size:
            raise ValueError("rewards must contain one value per sequence")
        if self.sampling_logprobs.numel() != batch.num_targets:
            raise ValueError("sampling_logprobs must contain one value per target")
        lengths = torch.tensor(tokens.sequence_lengths(), dtype=torch.int64)
        first = tokens.row_first_targets(batch.targets)
        if not torch.equal(first + batch.row_target_counts, lengths - 1):
            raise ValueError("targets must select exactly each row's sampled response")
        for start in range(0, tokens.batch_size, self.group_size):
            prompt = tokens.row_tokens(start)[: int(first[start]) + 1]
            for row in range(start + 1, start + self.group_size):
                if int(first[row]) != int(first[start]) or not torch.equal(
                    tokens.row_tokens(row)[: prompt.numel()], prompt
                ):
                    raise ValueError("each rollout group must share one exact prompt")

    def admit(self, batch: TrainingBatch, model_revision: int) -> None:
        """Require the batch to have been sampled from ``model_revision``."""

        if self.rollout_revision != model_revision:
            raise TrainingInputError(
                f"rollout revision {self.rollout_revision} does not match policy revision "
                f"{model_revision}"
            )

    def normalization_count(self, batch: TrainingBatch) -> int:
        return batch.batch_size

    def supplied_token_ids(self, batch: TrainingBatch) -> torch.Tensor:
        # Sampled responses are the policy's own tokens.
        tokens, first = batch.tokens, batch.tokens.row_first_targets(batch.targets)
        return torch.cat(
            [tokens.row_tokens(row)[: int(first[row]) + 1] for row in range(tokens.batch_size)]
        )

    def metrics(
        self,
        model: TrainingModel,
        batch: TrainingBatch,
        indices: tuple[int, ...],
        hidden_states: torch.Tensor,
        rows: DeviceRows,
    ) -> Metrics:
        device = hidden_states.device
        selected = torch.tensor(indices, dtype=torch.int64)
        counts = batch.row_target_counts
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        response_lengths = counts.index_select(0, selected)
        sampling = torch.cat(
            [self.sampling_logprobs[int(offsets[row]) : int(offsets[row + 1])] for row in indices]
        ).to(device)
        logprobs = model.training_logprobs(
            hidden_states, rows.target_ids, temperature=self.temperature
        )
        # Malformed log-probabilities fail this comparison before advantages leave the host.
        sampling_delta = (logprobs.detach() - sampling).abs().max()
        surrogate = clipped_surrogate(
            logprobs,
            logprobs.detach(),
            self.advantages.index_select(0, selected).to(device),
            response_lengths,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
        )
        return Metrics(
            (*surrogate, ("max_abs_sampling_logprob_delta", Reduction.MAX, sampling_delta)),
            len(indices),
            logprobs.numel(),
        )


def clipped_surrogate(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_lengths: torch.Tensor,
    *,
    epsilon_low: float,
    epsilon_high: float,
) -> tuple[tuple[str, Reduction, torch.Tensor], ...]:
    """The negated clipped surrogate over response tokens, each sequence's averaged, summed,
    with the clip fraction and the largest log-ratio. ``logprobs`` and ``old_logprobs`` are
    compact per response token; ``response_lengths`` stays on the host."""

    device = logprobs.device
    log_ratio = logprobs.float() - old_logprobs.float()
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, min=1.0 - epsilon_low, max=1.0 + epsilon_high)
    sequence_ids = torch.repeat_interleave(
        torch.arange(advantages.numel()), response_lengths, output_size=logprobs.numel()
    ).to(device)
    expanded = advantages.float().index_select(0, sequence_ids)
    surrogate = torch.minimum(ratio * expanded, clipped * expanded)
    per_sequence = torch.zeros(advantages.numel(), dtype=surrogate.dtype, device=device)
    per_sequence.scatter_add_(0, sequence_ids, surrogate).div_(response_lengths.to(device))
    return (
        ("loss_sum", Reduction.SUM, -per_sequence.sum()),
        ("clip_fraction", Reduction.MEAN, (ratio != clipped).float().mean().detach()),
        ("max_abs_log_ratio", Reduction.MAX, log_ratio.abs().max().detach()),
    )


def group_relative_advantages(
    rewards: torch.Tensor, group_size: int, *, epsilon: float = _GRPO_REWARD_EPSILON
) -> torch.Tensor:
    """Normalize FP32 rewards within each complete rollout group.

    Groups whose sample standard deviation is no larger than ``epsilon`` have no reliable
    relative signal and receive exact zero advantages.
    """

    if type(group_size) is not int or group_size < 2:
        raise ValueError("group_size must be an integer of at least two")
    if rewards.numel() == 0 or rewards.numel() % group_size != 0:
        raise ValueError("rewards must contain one or more complete groups")

    groups = rewards.to(torch.float32).view(-1, group_size)
    variance, mean = torch.var_mean(groups, dim=1, correction=1, keepdim=True)
    standard_deviation = torch.sqrt(variance)
    denominator = standard_deviation + epsilon
    denominator.masked_fill_(standard_deviation <= epsilon, math.inf)
    return ((groups - mean) / denominator).reshape(-1)


def _require_grpo_sampling(sampling: SamplingSpec) -> float:
    """The temperature of sampling GRPO can train on."""

    temperature = TokenSampling.stochastic_temperature(sampling.law.temperature)
    if sampling.seed is None:
        raise ValueError("GRPO sampling must preserve a request-local seed")
    # The policy's log-probabilities are the temperature softmax's: the rollout's must be.
    if sampling.law.logprobs is not LogprobMode.PROCESSED or not sampling.law.is_plain_categorical:
        raise ValueError(
            "GRPO sampling must be plain categorical and return processed log-probabilities"
        )
    return temperature


__all__ = [
    "GRPO",
    "SFT",
    "Metrics",
    "Reduction",
    "clipped_surrogate",
    "group_relative_advantages",
]
