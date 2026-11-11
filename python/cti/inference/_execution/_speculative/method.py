"""The speculative method the composition root loads before any device state: the draft
checkpoint, its class from the seam's draft table, and its checks against the target. It
implements ``plan.SpeculativeMethod``, which resolution and the model host read."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import torch

from cti.attention import AttentionRequirement
from cti.forward import RoundShape
from cti.models import ModelCapability, ModelConfig, dflash, read_checkpoint
from cti.utils import init_logger

if TYPE_CHECKING:
    from cti.config import ModelHub
    from cti.forward import ForwardInputs
    from cti.models import CheckpointSource
    from cti.models.dflash import DFlashConfig

logger = init_logger(__name__)

# The draft classes, by their checkpoint's architecture, each named so in ``models.dflash``.
_DRAFTS = frozenset({"DFlashDraftModel", "DFlash2DraftModel"})


class DraftModel(Protocol):
    """What the rounds read of a draft, which the model host builds from its ``CONFIG`` and
    loads: its attention layers and state layers, the K/V it stores of target tap rows, and
    the hidden rows of a block's inputs."""

    CONFIG: ClassVar[type]

    def iter_attention_layers(self) -> Iterator[Any]: ...

    def iter_state_layers(self) -> tuple[Any, ...]: ...

    def store_context(
        self, features: torch.Tensor, positions: torch.Tensor, batch: ForwardInputs
    ) -> None: ...

    def forward(
        self, batches: Sequence[ForwardInputs], embeddings: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class DFlash:
    """A DFlash v1 or DFlash2 draft checkpoint checked against its target, drafting
    ``draft_tokens`` tokens a round: its round rows are the anchor and as many mask rows,
    whatever block it was trained on (``config.block_size``), as SGLang and vLLM run it."""

    # Engine features rounds run beside (a LoRA target verifies with its adapter and a
    # training engine serves its rollouts, the draft frozen).
    capabilities: ClassVar = ModelCapability.ALL
    source: CheckpointSource
    model_class: type[DraftModel]
    config: DFlashConfig
    draft_tokens: int

    @property
    def round_width(self) -> int:
        """Q: a round's rows per request, the anchor and its drafts."""

        return self.draft_tokens + 1

    def attention_requirements(self) -> tuple[AttentionRequirement, ...]:
        """The draft layers' KV slot, once per distinct window."""

        model = self.config.model
        return tuple(
            dict.fromkeys(
                AttentionRequirement.mha(
                    model.num_qo_heads, model.num_kv_heads, model.head_dim, window=window
                )
                for window in self.config.layer_windows
            )
        )

    def check(self, *, max_seq_len: int) -> None:
        if max_seq_len > (limit := self.config.model.rotary_config.max_position):
            raise ValueError(
                f"max_seq_len {max_seq_len} exceeds the draft's max_position_embeddings={limit}"
            )

    def round_shape(self, *, max_running_req: int, max_forward_len: int) -> RoundShape:
        """Q-token rounds, each request with up to two unsettled, of at most four rows."""

        q = self.round_width
        if (rows := min(4, max_running_req, max_forward_len // q)) < 1:
            raise ValueError(f"a DFlash block of {q} tokens exceeds max_batch_tokens")
        return RoundShape(q, 2, rows)


def load_speculative(
    model_path: str, *, target: ModelConfig, hub: ModelHub, speculative_tokens: int | None
) -> DFlash:
    """The draft checkpoint at ``model_path``, read from ``hub`` as the target is, its class
    chosen by its architecture, checked against the ``target`` model, drafting
    ``speculative_tokens`` a round (None: its trained block's, ``block_size - 1``)."""

    draft = read_checkpoint(model_path, hub=hub)
    architectures = draft.document.get("architectures")
    architecture = architectures[0] if isinstance(architectures, list) and architectures else None
    if architecture not in _DRAFTS:
        raise ValueError(f"Model architecture {architecture} not supported")
    draft_class = getattr(dflash, architecture)
    config = draft_class.CONFIG.from_hf(dict(draft.document), draft.hf_config, target)
    drafts = config.block_size - 1 if speculative_tokens is None else speculative_tokens
    logger.info(
        "DFlash drafts %d tokens a round (trained block %d)%s",
        drafts,
        config.block_size,
        "" if speculative_tokens is None else ", as set",
    )
    return DFlash(draft.source, draft_class, config, drafts)
