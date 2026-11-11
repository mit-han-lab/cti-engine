# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/models/qwen3_moe.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING, Tuple

import torch

from cti.attention import AttentionRequirement
from cti.layers import (
    AttentionLayer,
    BaseOP,
    LinearQKVMerged,
    LinearRowParallel,
    MoELayer,
    OPList,
    ParallelLMHead,
    PrologueFactory,
    RMSNorm,
    RMSNormFused,
    RotaryEmbedding,
    VocabParallelEmbedding,
)
from cti.utils import nvtx_annotate

from ..activation_checkpointing import run_segments
from ..base import BaseLLMModel, ModelCapability
from ..features import FeatureExport, LayerTapExport, TokenHead
from ..mapping import CheckpointMapping, MergeRule
from ..training_output import (
    TrainingOutputMixin,
    attention_backward_tile_bytes,
)
from .config import Qwen3MoeConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from cti.forward import ForwardInputs
    from cti.layers import Site

    from ..activation_checkpointing import ReentrantActivationCheckpoint
    from ..training_contract import Packing


class Qwen3MoeAttention(BaseOP):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        *,
        rotary: RotaryEmbedding,
        site: Site,
        deferred: bool = False,
    ):
        """``deferred``: the residual norm after the attention sums inference's shares."""

        self._config = config
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=config.attention_bias,
            lora="attention",
            site=site,
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
            config.head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=config.attention_bias,
            lora="attention",
            site=site,
            deferred=deferred,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor, batch: ForwardInputs) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        return self.o_proj.forward(self.attn.forward(qkv, batch))

    def forward_training(
        self, x: torch.Tensor, positions: torch.Tensor, packing: Packing | None
    ) -> torch.Tensor:
        qkv = self.qkv_proj.forward_training(x)
        return self.o_proj.forward_training(self.attn.forward_training(qkv, positions, packing))

    def training_bytes(
        self, tokens: int, sequences: int, *, rank: int, stream: int
    ) -> tuple[int, int]:
        """``(saved, backward)`` of ``tokens`` rows in ``sequences`` sequences: the input
        stream with the FP32 inverse RMS of it and of the post-attention sum, which the MLP's
        norm saves, the normalized input;
        merged q/k/v with FP32 q/k inverse RMS, the attention output; SDPA's rotated q and k
        and FP32 log-sum-exp; four low-rank inputs. Its backward holds the stream's gradient
        and FlashAttention's tiles."""

        config = self._config
        hidden, heads, kv_heads = config.hidden_size, config.num_qo_heads, config.num_kv_heads
        query, kv = heads * config.head_dim, kv_heads * config.head_dim
        saved = (
            stream * hidden
            + 2 * hidden
            + 8
            + 6 * query
            + 6 * kv
            + 4 * (2 * heads + kv_heads)
            + 4 * 2 * rank
        )
        backward = tokens * (stream * hidden + 12 * query + 8 * kv)
        return tokens * saved, backward + attention_backward_tile_bytes(
            sequences, heads, config.head_dim
        )


class Qwen3MoeDecoderLayer(BaseOP):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_id: int,
        *,
        rotary: RotaryEmbedding,
        site: Site,
    ):
        reduction = site.residual_reduction(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=False,
        )
        deferred = reduction is not None
        self.self_attn = Qwen3MoeAttention(config, rotary=rotary, site=site, deferred=deferred)
        self.mlp = MoELayer(
            config.num_experts,
            config.num_experts_per_tok,
            config.hidden_size,
            config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            site=site,
            lora="experts",
            deferred=deferred,
        )
        self._hidden_size = config.hidden_size
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
            reduction=reduction,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
            reduction=reduction,
        )

        self._layer_id = layer_id

    def enter(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The input norm; the residual it returns is the stream after the previous layer."""

        return self.input_layernorm.forward(x, residual)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        batch: ForwardInputs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The layer after ``enter``."""

        x = self.self_attn.forward(x, batch)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual

    def mixer_training(
        self,
        stream: torch.Tensor,
        positions: torch.Tensor,
        packing: Packing | None,
    ) -> torch.Tensor:
        """The stream after the input norm and attention, of padded rows or with
        ``packing`` of packed sequences."""

        x, stream = self.input_layernorm.forward_training(stream)
        return stream + self.self_attn.forward_training(x, positions, packing)

    def mlp_training(self, stream: torch.Tensor) -> torch.Tensor:
        """The stream after the post-attention norm and MLP."""

        x, stream = self.post_attention_layernorm.forward_training(stream)
        return stream + self.mlp.forward_training(x)

    def training_bytes(
        self, tokens: int, sequences: int, *, rank: int, stream: int, row_tile: int | None
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """The mixer's and the sparse block's ``(saved, backward)``; the sparse block is not
        row-wise, so ``row_tile`` leaves it whole."""

        # The MLP's norm saves its input, the post-attention sum.
        stream_bytes = tokens * stream * self._hidden_size
        mlp_saved, mlp_backward = self.mlp.training_bytes(tokens, rank=rank)
        return (
            self.self_attn.training_bytes(tokens, sequences, rank=rank, stream=stream),
            (mlp_saved + stream_bytes, mlp_backward + stream_bytes),
        )


class Qwen3MoeModel(BaseOP):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        *,
        rotary: RotaryEmbedding,
        site: Site,
    ):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            site=site,
        )
        self.layers = OPList(
            [
                Qwen3MoeDecoderLayer(config, layer_id, rotary=rotary, site=site)
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
            reduction=site.residual_reduction(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                zero_centered=False,
            ),
        )
        self._activation_checkpoint: ReentrantActivationCheckpoint | None = None
        self._services = site.services

    def enable_activation_checkpointing(self, policy: ReentrantActivationCheckpoint) -> None:
        """Run the training blocks under the runner's checkpoint policy."""

        self._activation_checkpoint = policy

    def forward(self, batch: ForwardInputs) -> torch.Tensor:
        x = self.embed_tokens.forward(batch.input_ids)
        residual: torch.Tensor | None = None
        taps = self._services.taps
        for layer_id, layer in enumerate(self.layers.op_list):
            x, residual = layer.enter(x, residual)
            if taps is not None and layer_id > 0:
                taps.store(layer_id - 1, residual)
            x, residual = layer.forward(x, residual, batch)
        x, residual = self.norm.forward(x, residual)
        if taps is not None:
            taps.store(len(self.layers.op_list) - 1, residual)
        return x

    def forward_training(self, input_ids: torch.Tensor, packing: Packing | None) -> torch.Tensor:
        """Final hidden states of every input position, flat: of padded ``[batch, length]``
        rows, or with ``packing`` of packed sequences."""

        x = self.embed_tokens.forward_training(input_ids)
        if packing is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        else:
            positions = packing.positions
        return self._forward_blocks(x, positions, packing)

    def _forward_blocks(
        self, x: torch.Tensor, positions: torch.Tensor, packing: Packing | None
    ) -> torch.Tensor:
        """Run every decoder layer's attention and MLP, under the checkpoint policy if one
        is set, then normalize the stream and flatten its rows."""

        mixer = partial(Qwen3MoeDecoderLayer.mixer_training, positions=positions, packing=packing)
        layers = tuple((partial(mixer, layer), layer.mlp_training) for layer in self.layers.op_list)
        policy = self._activation_checkpoint
        x = run_segments(layers, x) if policy is None else policy.run_blocks(layers, x)
        return self.norm.forward_training(x)[0].flatten(0, -2)


class Qwen3MoeForCausalLM(TrainingOutputMixin, BaseLLMModel[Qwen3MoeConfig]):
    CAPABILITIES = (
        ModelCapability.LORA
        | ModelCapability.TRAINING
        | ModelCapability.PREFIX_REUSE
        | ModelCapability.TENSOR_PARALLEL
        | ModelCapability.SPECULATIVE_TARGET
    )
    LORA_TARGETS = ("attention", "experts")
    CONFIG = Qwen3MoeConfig
    checkpoint_mapping = CheckpointMapping(
        prefix_renames=(("language_model.", ""),),
        skip_prefixes=("vision_tower.", "multi_modal_projector."),
        merges=(
            MergeRule("qkv_proj", ("q_proj", "k_proj", "v_proj")),
            MergeRule("gate_up_proj", ("gate_proj", "up_proj")),
        ),
        # Published checkpoints store each expert's projections apart; they load stacked.
        stack_experts=True,
        optional_names=("lm_head.weight",),
    )

    @classmethod
    def validate_hf_config(cls, hf_config: PretrainedConfig) -> None:
        # Reading the config refuses what the layers would run differently.
        Qwen3MoeConfig.from_hf(hf_config)

    @classmethod
    def attention_requirements(
        cls, hf_config: PretrainedConfig
    ) -> tuple[AttentionRequirement, ...]:
        config = Qwen3MoeConfig.from_hf(hf_config)
        return (
            AttentionRequirement.mha(config.num_qo_heads, config.num_kv_heads, config.head_dim),
        )

    def __init__(self, config: Qwen3MoeConfig, *, site: Site):
        if site.training:
            config.refuse_training_only_fields()
        rotary_config = config.rotary_config
        rotary = RotaryEmbedding.create(
            config.head_dim,
            rotary_config.rotary_dim,
            rotary_config.max_position,
            rotary_config.base,
            rotary_config.scaling,
            site=site,
        )
        self.model = Qwen3MoeModel(config, rotary=rotary, site=site)
        self._services = site.services
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            site=site,
        )
        super().__init__()

    def iter_attention_layers(self) -> Iterator[AttentionLayer]:
        return (layer.self_attn.attn for layer in self.model.layers.op_list)

    def token_head(self) -> TokenHead:
        return TokenHead(self.model.embed_tokens, self.lm_head)

    def feature_export(self) -> FeatureExport:
        return LayerTapExport(self._services)

    def enable_activation_checkpointing(self, policy: ReentrantActivationCheckpoint) -> None:
        self.model.enable_activation_checkpointing(policy)

    def residual_stream_layout(self) -> tuple[int, torch.dtype]:
        # The decoder layers carry the embedding output's width and dtype.
        weight = self.model.embed_tokens.weight
        return weight.shape[1], weight.dtype

    def forward_hidden(self, batch: ForwardInputs) -> torch.Tensor:
        return self.model.forward(batch)

    def forward(self, batch: ForwardInputs) -> torch.Tensor:
        return self.compute_logits(self.forward_hidden(batch), batch)


__all__ = ["Qwen3MoeForCausalLM"]
