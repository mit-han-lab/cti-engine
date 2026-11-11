# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Z Lab
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from z-lab/dflash 07ebd93db9f472af339b644bb70221ad8428328a (`dflash/model.py`)
# and Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/models/qwen3.py`); see THIRD_PARTY_NOTICES.

"""DFlash v1 draft: a block of mask tokens attends to keys projected from target taps."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import torch

from cti.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    PrologueFactory,
    RMSNorm,
    RMSNormFused,
    RotaryEmbedding,
    SiluAndMul,
)

from ..mapping import CheckpointMapping, MergeRule
from .config import DFlashConfig, DraftDecoderConfig

if TYPE_CHECKING:
    from cti.forward import ForwardInputs
    from cti.layers import Site


class DFlashAttention(BaseOP):
    def __init__(
        self,
        config: DraftDecoderConfig,
        *,
        rotary: RotaryEmbedding,
        site: Site,
        deferred: bool = False,
    ) -> None:
        """``deferred``: the residual norm after the attention sums inference's shares."""

        self.qkv_proj = LinearQKVMerged(
            config.hidden_size,
            config.head_dim,
            config.num_qo_heads,
            config.num_kv_heads,
            has_bias=False,
            site=site,
            projects_kv=True,
        )
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            qkv=self.qkv_proj,
            rotary=rotary,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            prologue=PrologueFactory(),
        )
        self.o_proj = LinearRowParallel(
            config.num_qo_heads * config.head_dim,
            config.hidden_size,
            has_bias=False,
            site=site,
            deferred=deferred,
        )

    def forward(self, x: torch.Tensor, batch: ForwardInputs) -> torch.Tensor:
        return self.o_proj.forward(self.attn.forward(self.qkv_proj.forward(x), batch))

    def store_context(
        self, context: torch.Tensor, positions: torch.Tensor, batch: ForwardInputs
    ) -> None:
        """Store context rows' K and V at ``batch``'s write slots; Q is not computed."""

        self.attn.store_context(self.qkv_proj.project_kv(context), positions, batch)


class DFlashMLP(BaseOP):
    def __init__(self, config: DraftDecoderConfig, *, site: Site, deferred: bool = False) -> None:
        """``deferred``: the residual norm after the MLP sums inference's shares."""

        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
            output_names=("gate", "up"),
            site=site,
        )
        self.activation = SiluAndMul(self.gate_up_proj.local_output_size // 2)
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
            site=site,
            deferred=deferred,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(self.activation.forward(self.gate_up_proj.forward(x)))


class DFlashDecoderLayer(BaseOP):
    def __init__(
        self,
        config: DraftDecoderConfig,
        *,
        rotary: RotaryEmbedding,
        site: Site,
    ) -> None:
        reduction = site.residual_reduction(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, zero_centered=False
        )
        deferred = reduction is not None
        self.self_attn = DFlashAttention(config, rotary=rotary, site=site, deferred=deferred)
        self.mlp = DFlashMLP(config, site=site, deferred=deferred)
        self.input_layernorm = RMSNormFused(
            config.hidden_size, config.rms_norm_eps, reduction=reduction
        )
        self.post_attention_layernorm = RMSNormFused(
            config.hidden_size, config.rms_norm_eps, reduction=reduction
        )

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None, batch: ForwardInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x, batch)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        return self.mlp.forward(x), residual


class DFlashDraftModel(BaseOP):
    """Inference-only draft that borrows the target's embedding and LM head.

    Context rows (target taps) skip the decoder: each layer stores only their K/V. Block
    rows run the whole decoder; each layer attends under its ``DFlashConfig.layer_masks``
    entry: a full layer to all context and block keys of their request, a sliding layer
    causally to the keys within its window. Its layers split heads and MLP channels at the
    runner's ``site``, as the target's are; every process projects the whole taps.
    """

    CONFIG = DFlashConfig
    checkpoint_mapping = CheckpointMapping(
        merges=(
            MergeRule("qkv_proj", ("q_proj", "k_proj", "v_proj")),
            MergeRule("gate_up_proj", ("gate_proj", "up_proj")),
        ),
    )

    def __init__(self, config: DFlashConfig, *, round_width: int, site: Site) -> None:
        """``round_width``: a round's rows per request, the anchor and its mask rows."""

        model = config.model
        rotary_config = model.rotary_config
        rotary = RotaryEmbedding.create(
            rotary_config.head_dim,
            rotary_config.rotary_dim,
            rotary_config.max_position,
            rotary_config.base,
            rotary_config.scaling,
            site=site,
        )
        self.fc = LinearReplicated(
            len(config.target_layer_ids) * model.hidden_size,
            model.hidden_size,
            has_bias=False,
            site=site,
        )
        self.hidden_norm = RMSNorm(model.hidden_size, eps=model.rms_norm_eps)
        self.layers = OPList(
            [
                self._decoder_layer(config, round_width=round_width, rotary=rotary, site=site)
                for _ in range(model.num_layers)
            ]
        )
        self.norm = RMSNormFused(
            model.hidden_size,
            model.rms_norm_eps,
            reduction=site.residual_reduction(
                hidden_size=model.hidden_size, eps=model.rms_norm_eps, zero_centered=False
            ),
        )

    def _decoder_layer(
        self, config: DFlashConfig, *, round_width: int, rotary: RotaryEmbedding, site: Site
    ) -> DFlashDecoderLayer:
        return DFlashDecoderLayer(config.model, rotary=rotary, site=site)

    def iter_attention_layers(self) -> Iterator[AttentionLayer]:
        return (layer.self_attn.attn for layer in self.layers.op_list)

    def iter_state_layers(self) -> tuple[AttentionLayer, ...]:
        """The layers that hold inference state: the attention layers, in order."""

        return tuple(self.iter_attention_layers())

    def store_context(
        self, features: torch.Tensor, positions: torch.Tensor, batch: ForwardInputs
    ) -> None:
        """Store every layer's K and V of the context rows of concatenated target taps
        ``features`` at ``positions``, at ``batch``'s first write slots."""

        context = self.hidden_norm.forward(self.fc.forward(features))
        for layer in self.layers.op_list:
            layer.self_attn.store_context(context, positions, batch)

    def forward(self, batches: Sequence[ForwardInputs], embeddings: torch.Tensor) -> torch.Tensor:
        """Final-normed hidden rows; ``embeddings`` is consumed as residual scratch. Layer
        ``i`` attends through ``batches[i]``, which carries that layer's mask."""

        x, residual = embeddings, None
        for layer, batch in zip(self.layers.op_list, batches, strict=True):
            x, residual = layer.forward(x, residual, batch)
        return self.norm.forward(x, residual)[0]


__all__ = ["DFlashDraftModel"]
