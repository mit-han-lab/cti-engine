from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from cti.models.training_contract import TrainingActivations

if TYPE_CHECKING:
    from cti.layers import OPList, ParallelLMHead
    from cti.models.training_contract import Packing

# FlashAttention 2.8's backward (behind torch SDPA's flash kernel and cti-kernels' varlen
# attention) keeps its FP32 query-gradient accumulator and softmax row
# sums for up to this many rows per sequence beyond the sequence's tokens: padded batches
# round each row's length up to it, packed ones add it to every sequence.
_FLASH_BACKWARD_TILE_ROWS = 128


@dataclass(frozen=True, slots=True)
class TrainingLossChunks:
    """Token rows per logits chunk of the LM head's loss operations."""

    # SFT's fused linear cross entropy.
    cross_entropy: int
    # GRPO's selected-token log-probabilities.
    logprobs: int


def loss_logits_blocks(
    tokens: int, vocab_size: int, chunks: TrainingLossChunks
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The logits blocks each objective holds at once over ``tokens`` targets: SFT's two
    consecutive BF16 chunks, GRPO's four FP32 chunks."""

    first = min(tokens, chunks.cross_entropy)
    second = min(tokens - first, chunks.cross_entropy)
    cross_entropy = tuple(2 * vocab_size * rows for rows in (first, second) if rows)
    logprobs = (4 * min(tokens, chunks.logprobs) * vocab_size,) * 4
    return cross_entropy, logprobs


def loss_working_bytes(
    tokens: int, hidden_size: int, vocab_size: int, chunks: TrainingLossChunks
) -> int:
    """The loss's working set over ``tokens`` targets, the larger of the two objectives.

    SFT's fused cross entropy holds the BF16 hidden input and FP32 token losses, then two
    BF16 logits chunks across a chunk's matmul, or one with its hidden-gradient product.
    GRPO's log-probabilities recompute in backward with the BF16 hidden gradient and, across
    a chunk's recomputation, the previous FP32 logits and their gradient besides two more.
    """

    rows = min(tokens, chunks.cross_entropy)
    two_chunks, four_chunks = loss_logits_blocks(tokens, vocab_size, chunks)
    cross_entropy = tokens * (2 * hidden_size + 4) + max(
        sum(two_chunks), 2 * rows * (vocab_size + hidden_size)
    )
    logprobs = tokens * 2 * hidden_size + sum(four_chunks)
    return max(cross_entropy, logprobs)


def attention_backward_tile_bytes(sequences: int, num_heads: int, head_dim: int) -> int:
    """FlashAttention backward's FP32 bytes beyond what its tokens take: the query-gradient
    accumulator, at the kernel's rounded head dimension, and the softmax row sums over the
    tile rows each of ``sequences`` sequences may add."""

    rounded_head_dim = -(-head_dim // 32) * 32 if head_dim <= 192 else 256
    return 4 * _FLASH_BACKWARD_TILE_ROWS * sequences * num_heads * (rounded_head_dim + 1)


class _TrainedLayer(Protocol):
    def training_bytes(
        self, tokens: int, sequences: int, *, rank: int, stream: int, row_tile: int | None
    ) -> tuple[tuple[int, int], ...]: ...


class _FinalNorm(Protocol):
    # Bytes per token the norm saves beside its input.
    TRAINING_SAVED_BYTES: int


class _TrainingHiddenStates(Protocol):
    layers: OPList
    norm: _FinalNorm

    def forward_training(
        self, input_ids: torch.Tensor, packing: Packing | None
    ) -> torch.Tensor: ...


class TrainingOutputMixin:
    """Training hidden states, losses and log-probabilities from a model and its LM head, and the
    paths of its decoder layers."""

    # The token mixer, then the MLP or sparse block.
    TRAINING_LAYER_SEGMENTS = 2

    model: _TrainingHiddenStates
    lm_head: ParallelLMHead

    def decoder_layer_paths(self) -> tuple[str, ...]:
        """Each decoder layer's state-dict path, in execution order."""

        return tuple(f"model.layers.{index}" for index in range(len(self.model.layers.op_list)))

    def training_activations(
        self, tokens: int, sequences: int, *, rank: int, row_tile: int | None = None
    ) -> TrainingActivations:
        """Each decoder layer's activation bytes as it accounts them, every LoRA target
        adapted at ``rank`` (an upper bound for fewer; 0 without an adapter), and the head's:
        the final stream, its norm's saves and the loss's precomputed hidden gradient, the
        loss's working set in its logits chunks. With ``row_tile`` the layers' row-wise
        segments train in tiles of that many rows."""

        hidden, dtype = self.residual_stream_layout()
        stream = dtype.itemsize
        layers: Sequence[_TrainedLayer] = self.model.layers.op_list
        per_layer = tuple(
            layer.training_bytes(tokens, sequences, rank=rank, stream=stream, row_tile=row_tile)
            for layer in layers
        )
        vocab = self.lm_head.num_embeddings
        loss_chunks = TrainingLossChunks(*self.lm_head.training_chunk_rows)
        widest = max(
            (
                getattr(op, "local_output_size", 0)
                for layer in layers
                for _, op in layer.named_ops()
            ),
            default=0,
        )
        return TrainingActivations(
            layers=per_layer,
            head_saved=tokens * ((stream + 2) * hidden + self.model.norm.TRAINING_SAVED_BYTES),
            head_working=loss_working_bytes(tokens, hidden, vocab, loss_chunks),
            head_blocks=max(loss_logits_blocks(tokens, vocab, loss_chunks), key=sum),
            # Projections write BF16 rows, as each layer's training_bytes counts them.
            largest_block=2 * tokens * widest,
        )

    def validate_training_targets(self, target_ids: torch.Tensor) -> None:
        self.lm_head.validate_training_targets(target_ids)

    def training_hidden_states(
        self, input_ids: torch.Tensor, packing: Packing | None
    ) -> torch.Tensor:
        return self.model.forward_training(input_ids, packing)

    def training_loss(self, hidden_states: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head.forward_training_loss(hidden_states, target_ids)

    def training_logprobs(
        self, hidden_states: torch.Tensor, target_ids: torch.Tensor, *, temperature: float
    ) -> torch.Tensor:
        return self.lm_head.forward_training_logprobs(
            hidden_states, target_ids, temperature=temperature
        )
