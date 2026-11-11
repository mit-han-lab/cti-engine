# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/models/base.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from enum import Flag, auto
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, Self, TypeVar

import torch

from cti.layers import BaseOP

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from cti.attention import AttentionRequirement
    from cti.config import LoRATarget
    from cti.forward import ForwardInputs, VisionTower
    from cti.layers import AttentionLayer, ParallelLMHead
    from cti.recurrent import RecurrentLayer

    from .config import ModelConfig
    from .features import FeatureExport, TokenHead
    from .mapping import CheckpointMapping


class PeerConfig(Protocol):
    """A model class's own configuration: read from a Hugging Face config, and folded into
    what the engine reads."""

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> Self: ...

    def engine(self, architectures: tuple[str, ...]) -> ModelConfig: ...


ConfigT = TypeVar("ConfigT", bound=PeerConfig)


class ModelCapability(Flag):
    """Optional execution features a model class declares support for.

    A member is a feature the configuration requests that some model class may not
    support and that can be refused on its own. Features every model must support,
    such as decode CUDA graphs and residency switching, are release gates, not
    members. Structural traits that need model methods, such as multimodal input,
    M-RoPE positions and recurrent state, are hooks, not members. Do not add
    ``can_*``/``supports_*`` model properties or
    architecture-name whitelists beside this flag.
    """

    LORA = auto()
    # The whole TrainingRunner contract: SFT and GRPO in padded and packed layouts,
    # plus activation checkpointing.
    TRAINING = auto()
    PREFIX_REUSE = auto()
    # Head- and vocabulary-sharded inference on one node, over its site's ranks.
    TENSOR_PARALLEL = auto()
    # A speculative draft's target: ``feature_export`` returns the residual stream after the
    # layers a proposer requests.
    SPECULATIVE_TARGET = auto()
    # Every member above, listed: what a component that refuses no feature declares. A new
    # member joins it only when each such component is checked to run beside it.
    ALL = LORA | TRAINING | PREFIX_REUSE | TENSOR_PARALLEL | SPECULATIVE_TARGET


class BaseLLMModel(ABC, BaseOP, Generic[ConfigT]):
    """A registered model, constructed from ``CONFIG.from_hf(hf_config)``."""

    CAPABILITIES: ClassVar[ModelCapability]
    # The LoRA targets that adapt at least one projection; non-empty exactly with LORA.
    # The worker rejects any other requested target before it builds anything.
    LORA_TARGETS: ClassVar[tuple[LoRATarget, ...]]
    # The model's topology configuration type; its ``from_hf`` reads a Hugging Face config.
    CONFIG: ClassVar[type[ConfigT]]
    # The token multiples its recurrent layers' prefills export states after
    # (``SlotStateSpec.export_granularity``); None without recurrent layers.
    RECURRENT_EXPORT_GRANULARITY: ClassVar[int | None] = None
    # How the model's checkpoint tensors become its parameters. A mapping that depends
    # on the model's geometry is set by the constructor; a fixed one may be a class
    # attribute.
    checkpoint_mapping: CheckpointMapping

    model: BaseOP
    lm_head: ParallelLMHead

    @classmethod
    @abstractmethod
    def validate_hf_config(cls, hf_config: PretrainedConfig) -> None:
        """Reject Hugging Face text-config semantics this class does not implement.

        Raise ``ValueError`` naming the field and its value. Every class states its own
        rules, against the Hugging Face modeling code of its architecture; there is no
        default that accepts every config. ``GPUWorker`` calls this before any device
        state or model exists.
        """
        ...

    @classmethod
    def engine_config(cls, hf_config: PretrainedConfig) -> ModelConfig:
        """What the engine reads of the model, folded from its own ``CONFIG``."""

        return cls.CONFIG.from_hf(hf_config).engine(tuple(hf_config.architectures))

    @classmethod
    @abstractmethod
    def attention_requirements(
        cls, hf_config: PretrainedConfig
    ) -> tuple[AttentionRequirement, ...]:
        """What the attention layers need of an attention family, given at resolution
        before any model exists."""

    def validate_training_inputs(self, token_ids: torch.Tensor) -> None:
        """Refuse input tokens training cannot embed; by default every token trains."""

        _ = token_ids

    @classmethod
    def require_capabilities(
        cls,
        requested: ModelCapability,
        *,
        model_architecture: str,
        lora_targets: Sequence[str] = (),
    ) -> None:
        """Reject a configuration that needs capabilities or LoRA targets this class lacks."""

        missing = requested & ~cls.CAPABILITIES
        if missing:
            names = ", ".join(member.name for member in ModelCapability if member in missing)
            raise ValueError(
                f"Model architecture {model_architecture} does not support the requested "
                f"capabilities: {names}"
            )
        undeclared = [target for target in lora_targets if target not in cls.LORA_TARGETS]
        if undeclared:
            declared = ", ".join(cls.LORA_TARGETS) or "none"
            raise ValueError(
                f"Model architecture {model_architecture} does not declare the requested LoRA "
                f"targets: {', '.join(undeclared)}; it declares {declared}"
            )

    @classmethod
    def exports_prefill_states(
        cls, hf_config: PretrainedConfig, device: torch.device, *, num_ranks: int
    ) -> bool:
        """Whether every recurrent layer's prefill on ``device``, at ``num_ranks`` ranks,
        exports the states prefix reuse restores; a model without recurrent layers has none
        to export."""

        _ = hf_config, device, num_ranks
        return True

    @abstractmethod
    def iter_attention_layers(self) -> Iterator[AttentionLayer]:
        """Yield the current paged-KV attention layers in execution order."""
        ...

    def iter_recurrent_layers(self) -> tuple[RecurrentLayer, ...]:
        """Return the recurrent layers in execution order."""

        return ()

    def iter_state_layers(self) -> tuple[AttentionLayer | RecurrentLayer, ...]:
        """The layers that hold inference state, which the runner groups and binds: the
        attention layers, then the recurrent ones, each in execution order."""

        return (*self.iter_attention_layers(), *self.iter_recurrent_layers())

    def rope_position_axes(self) -> int:
        """Return how many position rows each token's rotary embedding reads.

        One axis rotates by the sequence positions themselves; multi-axis rotary (M-RoPE)
        reads one int64 row per axis where image prompts make the axes differ, and one row
        that every axis reads otherwise.
        """

        return 1

    def warm_up_image_encoder(
        self,
        *,
        max_tokens: int,
        max_image_tokens: int,
        device: torch.device,
    ) -> bool:
        """Encode the largest image batch one step may carry, if the model encodes images.

        A step's images fill at most ``max_tokens`` tokens, none more than
        ``max_image_tokens``. The warm-up also holds what a step keeps alive beside the
        encoder. InferenceRunner measures the activation peak around this call and takes from
        the KV cache whatever the peak reserve cannot hold. Returns whether anything ran.
        """

        return False

    def vision_tower(self) -> VisionTower | None:
        """The image encoder vision graphs capture; None for a model without one."""

        return None

    def forward_hidden(self, batch: ForwardInputs) -> torch.Tensor:
        return self.model.forward(batch)

    def compute_logits(self, hidden_states: torch.Tensor, batch: ForwardInputs) -> torch.Tensor:
        """Logits of the rows that take them: a ragged forward's ``logit_rows``, selected
        before the head projects, else every row."""

        if batch.ragged:
            if batch.logit_rows is None:
                raise ValueError("a ragged forward that takes logits names its logit_rows")
            hidden_states = hidden_states[batch.logit_rows]
        return self.lm_head.project(hidden_states)

    @abstractmethod
    def token_head(self) -> TokenHead:
        """The token embedding and LM head, for the readers of rows outside the forward:
        prompt log-probabilities and a speculative proposer."""
        ...

    def feature_export(self) -> FeatureExport | None:
        """The features a SPECULATIVE_TARGET exports to a proposer; None for other models."""

        return None

    @abstractmethod
    def forward(self, batch: ForwardInputs) -> torch.Tensor: ...
