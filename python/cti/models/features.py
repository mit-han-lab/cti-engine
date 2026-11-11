"""What a speculative proposer reads of its target: the residual stream after chosen layers
(``feature_export``) and the token embedding and LM head (``token_head``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from cti.layers import LayerServices, ParallelLMHead, VocabParallelEmbedding


@dataclass(frozen=True, slots=True)
class FeatureRequest:
    """The residual stream after each of ``layer_ids``, before the final norm."""

    layer_ids: tuple[int, ...]


class FeatureExport(Protocol):
    """A target that writes the features a proposer requests during its forwards."""

    def bind_features(self, request: FeatureRequest, buffer: torch.Tensor) -> None:
        """Have every inference forward write ``request``'s features of its tokens into
        ``buffer`` ``[rows, len(layer_ids) * hidden]``; bound once, before graph capture."""
        ...


class LayerTapExport:
    """Features the decoder writes through its layer services' taps."""

    def __init__(self, services: LayerServices) -> None:
        self._services = services

    def bind_features(self, request: FeatureRequest, buffer: torch.Tensor) -> None:
        from cti.layers import FeatureTaps

        self._services.taps = FeatureTaps(request.layer_ids, buffer)


@dataclass(frozen=True, slots=True, eq=False)
class TokenHead:
    """A model's token embedding and LM head, borrowed rather than copied: the same rows
    on every rank. ``local_logits`` reads this rank's vocabulary shard once; the
    ``gather_*`` views of it then agree across ranks."""

    embedding: VocabParallelEmbedding
    lm_head: ParallelLMHead

    @property
    def hidden_size(self) -> int:
        return self.embedding.weight.shape[1]

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings, as the decoder's first layer reads them."""
        return self.embedding.forward(input_ids)

    def argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        """Greedy token ID of every row; ties go to the smallest ID."""
        return self.lm_head.argmax(hidden)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits of every row, contiguous."""
        return self.lm_head.project(hidden)

    def local_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head.local_logits(hidden)

    def gather_argmax(self, local: torch.Tensor) -> torch.Tensor:
        return self.lm_head.gather_argmax(local)

    def gather_logits(self, local: torch.Tensor) -> torch.Tensor:
        return self.lm_head.gather_logits(local)

    def gather_topk(self, local: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lm_head.gather_topk(local, k)

    def topk(self, logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``k`` largest of full-vocabulary ``logits`` (FP32) and their token IDs
        (int64), by ascending ID; ties go to the smaller ID."""
        return self.lm_head.topk(logits, k)


__all__ = ["FeatureExport", "FeatureRequest", "LayerTapExport", "TokenHead"]
