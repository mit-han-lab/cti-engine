"""Training's selected-token logprobs and cross-entropy over a vocabulary sharded across the
ranks, until cti-kernels' training operations take a shard."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .linear import ParallelEntry
    from .linear_method import FrozenProjection


class ChunkedLogprobs:
    """FP32 ``log_softmax((x @ W.T + b) / T)`` at each row's target, from the frozen
    projection of this rank's vocabulary rows ``W`` (all of them on one rank).

    Each rank reduces a chunk of rows to (maximum, sum of exponentials, target logit or
    zero); one gather of those triples gives every rank the log-normalizer. The backward
    recomputes each chunk's logits, so no rank holds more than one chunk of them, and
    returns this rank's partial input gradient, which the entry sums over the ranks.
    """

    CHUNK_ROWS = 256

    def __init__(
        self,
        enter: Callable[[torch.Tensor], torch.Tensor],
        vocab_range: tuple[int, int],
        gather: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """The vocabulary's ``enter`` of the hidden states, its rank's ``vocab_range``
        (start, count) and the ``gather`` that stacks every rank's tensor."""

        self._enter = enter
        self._vocab_range = vocab_range
        self._gather = gather

    def forward(
        self,
        hidden_states: torch.Tensor,
        projection: FrozenProjection,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        temperature: float,
    ) -> torch.Tensor:
        if bias is not None and bias.requires_grad:
            raise ValueError("a chunked LM head trains frozen")
        return _ChunkedLogprobs.apply(
            self._enter(hidden_states), bias, target_ids, temperature, projection, self
        )


class ShardedLogprobs(ChunkedLogprobs):
    """``LinearSelectedTokenLogprobs`` over a vocabulary sharded across the ranks, from this
    rank's dense shard ``W``."""

    def __init__(
        self,
        entry: ParallelEntry,
        vocab_range: tuple[int, int],
        gather: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__(entry.enter, vocab_range, gather)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        temperature: float,
    ) -> torch.Tensor:
        if weight.requires_grad:
            raise ValueError("a vocabulary-sharded LM head trains frozen")
        return super().forward(
            hidden_states, _DenseProjection(weight), target_ids, bias=bias, temperature=temperature
        )


class ShardedCrossEntropy(ShardedLogprobs):
    """``LinearCrossEntropyLoss`` over a sharded vocabulary: the summed negative
    log-probabilities of the targets."""

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return -super().forward(hidden_states, weight, target_ids, bias=bias, temperature=1.0).sum()


class _ChunkedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        target_ids: torch.Tensor,
        temperature: float,
        projection: FrozenProjection,
        owner: ChunkedLogprobs,
    ) -> torch.Tensor:
        start, count = owner._vocab_range
        local_targets = target_ids - start
        owned = (local_targets >= 0) & (local_targets < count)
        # A rank without vocabulary rows still gathers a valid column, never one it owns.
        local_targets = local_targets.clamp(0, max(count, 1) - 1)
        triples = x.new_empty(x.shape[0], 3, dtype=torch.float32)
        for rows in _chunks(x.shape[0], owner.CHUNK_ROWS):
            logits = _shard_logits(x[rows], projection, bias, temperature, count)
            maximum = logits.amax(-1)
            # A rank without vocabulary rows contributes an empty sum, never a NaN.
            finite = torch.where(maximum.isfinite(), maximum, 0.0)
            triples[rows, 0] = maximum
            triples[rows, 1] = (logits - finite.unsqueeze(-1)).exp().sum(-1)
            target = logits.gather(-1, local_targets[rows].unsqueeze(-1)).squeeze(-1)
            triples[rows, 2] = torch.where(owned[rows], target, 0.0)
        ranks = owner._gather(triples)
        maximum = ranks[..., 0].amax(0)
        rank_maximum = torch.where(ranks[..., 0].isfinite(), ranks[..., 0], 0.0)
        log_normalizer = maximum + (ranks[..., 1] * (rank_maximum - maximum).exp()).sum(0).log()
        ctx.save_for_backward(x, bias, local_targets, owned, log_normalizer)
        ctx.projection = projection
        ctx.temperature, ctx.count, ctx.chunk = temperature, count, owner.CHUNK_ROWS
        return ranks[..., 2].sum(0) - log_normalizer

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x, bias, local_targets, owned, log_normalizer = ctx.saved_tensors
        x_gradient = torch.empty_like(x)
        gradient = gradient.float() / ctx.temperature
        for rows in _chunks(x.shape[0], ctx.chunk):
            logits = _shard_logits(x[rows], ctx.projection, bias, ctx.temperature, ctx.count)
            # d logprob / d logits = onehot(target) - softmax.
            logits_gradient = -(logits - log_normalizer[rows].unsqueeze(-1)).exp()
            logits_gradient.scatter_add_(
                -1, local_targets[rows].unsqueeze(-1), owned[rows].float().unsqueeze(-1)
            )
            logits_gradient *= gradient[rows].unsqueeze(-1)
            x_gradient[rows] = ctx.projection.input_gradient(logits_gradient).to(x.dtype)
        return x_gradient, None, None, None, None, None


def _chunks(rows: int, step: int) -> Iterator[slice]:
    return (slice(row, min(row + step, rows)) for row in range(0, rows, step))


def _shard_logits(
    x: torch.Tensor,
    projection: FrozenProjection,
    bias: torch.Tensor | None,
    temperature: float,
    count: int,
) -> torch.Tensor:
    """One chunk's FP32 logits over a rank's shard; its padding rows score -inf."""

    logits = projection.logits(x, bias) / temperature
    logits[:, count:] = float("-inf")
    return logits


class _DenseProjection:
    """A dense frozen weight as a chunked training loss reads it."""

    def __init__(self, weight: torch.Tensor) -> None:
        self._weight = weight

    def logits(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return F.linear(x, self._weight, bias).float()

    def input_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        return gradient.to(self._weight.dtype) @ self._weight


__all__ = ["ChunkedLogprobs", "ShardedCrossEntropy", "ShardedLogprobs"]
