"""What training needs from a model: one Protocol every training model implements, and the
values it exchanges with training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from .activation_checkpointing import ReentrantActivationCheckpoint


@dataclass(frozen=True, slots=True)
class Packing:
    """Packed sequences' positions and input boundaries, and the longest input."""

    positions: torch.Tensor
    cu_seqlens: torch.Tensor
    max_sequence_length: int

    def to(self, device: torch.device) -> Packing:
        return Packing(
            self.positions.to(device), self.cu_seqlens.to(device), self.max_sequence_length
        )


@dataclass(frozen=True, slots=True)
class TrainingActivations:
    """Bytes a model's training ops hold for one microbatch."""

    # Per decoder layer, its training segments in forward order (its token mixer, then its
    # MLP or sparse block), each ``(saved, backward)``: what the segment's forward keeps
    # until its backward runs, and what its backward allocates beside that.
    layers: tuple[tuple[tuple[int, int], ...], ...]
    # The final norm and loss: what they keep until backward.
    head_saved: int
    # The loss's working set beside that, the larger of its forward and backward.
    head_working: int
    # The loss's logits blocks live at once, each of its own allocation.
    head_blocks: tuple[int, ...] = ()
    # The widest single block a decoder layer allocates: a projection's output.
    largest_block: int = 0


class TrainingOpLayer(Protocol):
    """A layer holding training ops that compile on first use."""

    def warm_up_training(self, max_tokens: int) -> None: ...


@runtime_checkable
class TrainingModel(Protocol):
    """What every training model implements: hidden states of padded rows or packed
    sequences, the loss and log-probabilities of selected hidden states, activation
    checkpointing, input validation, and what planning and warming training up read: its
    activation bytes, its decoder layers' paths and the layers whose training ops compile
    ahead."""

    # How many training segments each decoder layer runs as, in forward order.
    TRAINING_LAYER_SEGMENTS: int

    def validate_training_targets(self, target_ids: torch.Tensor) -> None: ...

    def validate_training_inputs(self, token_ids: torch.Tensor) -> None:
        """Refuse input tokens training cannot embed, such as tokens only a vision tower
        embeds.

        The runner passes the CPU tokens the caller supplied before any microbatch runs, so
        a refusal costs no device synchronization and leaves accumulation untouched: an SFT
        batch's tokens, a GRPO batch's prompts. Sampled responses are the policy's own
        tokens, embedded in training as they were in sampling.
        """

        ...

    def residual_stream_layout(self) -> tuple[int, torch.dtype]:
        """The width and dtype of the residual stream, every decoder layer's input."""

        ...

    def enable_activation_checkpointing(self, policy: ReentrantActivationCheckpoint) -> None:
        """Run the decoder layers under the runner's checkpoint policy in training."""

        ...

    def training_hidden_states(
        self, input_ids: torch.Tensor, packing: Packing | None
    ) -> torch.Tensor:
        """Final hidden states of every input position, flat: of padded rows, or with
        ``packing`` of packed sequences."""

        ...

    def training_loss(self, hidden_states: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """One summed 0-D cross-entropy over ``hidden_states``' rows."""

        ...

    def training_logprobs(
        self, hidden_states: torch.Tensor, target_ids: torch.Tensor, *, temperature: float
    ) -> torch.Tensor: ...

    def training_activations(
        self, tokens: int, sequences: int, *, rank: int, row_tile: int | None = None
    ) -> TrainingActivations:
        """Account a microbatch of ``tokens`` tokens in up to ``sequences`` sequences, every
        LoRA target adapted at ``rank``; with ``row_tile`` its row-wise segments train in
        tiles of that many rows."""

        ...

    def decoder_layer_paths(self) -> tuple[str, ...]:
        """Each decoder layer's state-dict path, in execution order."""

        ...

    def iter_recurrent_layers(self) -> Sequence[TrainingOpLayer]:
        """The layers whose training ops compile ahead of training."""

        ...


__all__ = ["Packing", "TrainingActivations", "TrainingModel", "TrainingOpLayer"]
