# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`); see THIRD_PARTY_NOTICES.

"""Qwen3.5 dense model: Gated DeltaNet and gated full-attention layers, and the vision tower.

Rewritten from transformers 5.16.1 ``models/qwen3_5/modeling_qwen3_5.py``
(``Qwen3_5Attention``, ``Qwen3_5GatedDeltaNet``, ``Qwen3_5MLP``, ``Qwen3_5DecoderLayer``,
``Qwen3_5TextModel`` and ``Qwen3_5ForConditionalGeneration``). An image-text checkpoint
also builds its frozen vision tower (``vision.py``), whose tokens replace the image
placeholders' embeddings in a prefill step; a text-only export has none. The
multi-token-prediction head is not built, and its checkpoint tensors are skipped.

LoRA adapts the text projections the way PEFT adapts the checkpoint's modules: one A per
checkpoint module and one B per output slice. A module that the engine splits or merges
into several slices of one projection (``q_proj``'s query and gate, ``in_proj_qkv``'s q,
k and v) shares its A among them.

Training updates only the text adapters and accepts padded or packed text sequences.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING

import torch

from cti.attention import AttentionRequirement
from cti.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearQKVMerged,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    PrologueFactory,
    ResidualStreamRMSNorm,
    RMSNorm,
    RotaryEmbedding,
    SiluAndMul,
    VocabParallelEmbedding,
)
from cti.layers.gated_delta_net import GatedDeltaNet, prefill_exports_states
from cti.utils import nvtx_annotate

from ..activation_checkpointing import RowWise, run_segments
from ..base import BaseLLMModel, ModelCapability
from ..features import FeatureExport, LayerTapExport, TokenHead
from ..mapping import CheckpointMapping, CheckpointRule, MergeRule, Reshape, RowSplit
from ..training_output import (
    TrainingOutputMixin,
    attention_backward_tile_bytes,
)
from .config import (
    LayerKind,
    Qwen3_5AttentionConfig,
    Qwen3_5Config,
    Qwen3_5LinearAttentionConfig,
)
from .vision import Qwen3_5VisionModel

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from cti.forward import ForwardInputs, VisionTower
    from cti.layers import Site
    from cti.multimodal import ImagePreprocessing

    from ..activation_checkpointing import ReentrantActivationCheckpoint
    from ..training_contract import Packing
    from .config import Qwen3_5TextConfig


def _flatten_patch_conv(conv: torch.Tensor) -> tuple[torch.Tensor]:
    """The patch embedding's Conv3d weight ``[D, C, T, p, p]`` as a linear ``[D, C*T*p*p]``.

    transformers 5.16.1 ``Qwen3_5VisionPatchEmbed`` convolves each patch with a stride
    equal to its kernel, which is one linear map of the patch flattened channel, frame,
    row, column first, the order of the weight's trailing dimensions.
    """

    return (conv.flatten(1),)


def _checkpoint_mapping(
    linear: Qwen3_5LinearAttentionConfig,
    *,
    has_vision: bool,
) -> CheckpointMapping:
    """How a checkpoint with this Gated DeltaNet geometry becomes the model's parameters.

    One mapping reads both key layouts: image-text checkpoints prefix the text model
    with ``model.language_model.`` and the vision tower with ``model.visual.``, text-only
    exports the text model with ``model.``. A model without a vision tower skips its
    tensors. The Gated DeltaNet projection ``in_proj_qkv`` and convolution ``conv1d``
    store all query, then all key, then all value rows or channels. Split by this
    geometry, each of q, k and v becomes its own merge slot, read through the layer
    owner's checkpoint window.
    """

    key_size = linear.num_key_heads * linear.key_head_dim
    split_qkv = RowSplit((key_size, key_size, linear.num_value_heads * linear.value_head_dim))
    vision_renames = (("model.visual.", "visual."),) if has_vision else ()
    return CheckpointMapping(
        prefix_renames=(("model.language_model.", "model."), *vision_renames),
        skip_prefixes=("mtp.",) if has_vision else ("model.visual.", "mtp."),
        rules=(
            CheckpointRule(
                "visual.patch_embed.proj.weight",
                ("visual.patch_embed.proj.weight",),
                Reshape(_flatten_patch_conv),
            ),
            # Per head: query rows, then gate rows (transformers 5.16.1 Qwen3_5Attention).
            CheckpointRule(
                "self_attn.q_proj.weight",
                ("self_attn.q_proj.weight", "self_attn.output_gate_proj.weight"),
                RowSplit((Qwen3_5AttentionConfig.HEAD_DIM,) * 2),
            ),
            CheckpointRule(
                "linear_attn.in_proj_qkv.weight",
                tuple(f"linear_attn.in_proj_{part}.weight" for part in "qkv"),
                split_qkv,
            ),
            # The convolution's q, k and v channel groups are slots 0-2 of conv_weight.
            CheckpointRule(
                "linear_attn.conv1d.weight", ("linear_attn.gdn.conv_weight",) * 3, split_qkv
            ),
            CheckpointRule("linear_attn.A_log", ("linear_attn.gdn.A_log",)),
            CheckpointRule("linear_attn.dt_bias", ("linear_attn.gdn.dt_bias",)),
            # Qualified so that it never matches the final model.norm.weight.
            CheckpointRule("linear_attn.norm.weight", ("linear_attn.gdn.norm_weight",)),
        ),
        # Merges match module leaf names anywhere in the model, so the split gate must
        # not be named like the MLP's gate_proj.
        merges=(
            MergeRule("qkv_gate_proj", ("q_proj", "k_proj", "v_proj", "output_gate_proj")),
            MergeRule(
                "in_proj",
                tuple(f"in_proj_{part}" for part in ("q", "k", "v", "z", "b", "a")),
            ),
            MergeRule("gate_up_proj", ("gate_proj", "up_proj")),
        ),
        optional_names=("lm_head.weight",),
    )


class Qwen3_5MLP(BaseOP):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        *,
        site: Site,
        deferred: bool = False,
    ) -> None:
        """``deferred``: the residual norm after the MLP sums inference's shares."""

        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
            output_names=("gate", "up"),
            lora="mlp",
            site=site,
        )
        self.activation = SiluAndMul(self.gate_up_proj.local_output_size // 2)
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
            lora="mlp",
            site=site,
            deferred=deferred,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.activation.forward(gate_up)
        del gate_up
        return self.down_proj.forward(y)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation.mlp_training(self.gate_up_proj, self.down_proj, x)

    def training_bytes(self, tokens: int, *, rank: int) -> tuple[int, int]:
        return self.activation.mlp_training_bytes(
            self.gate_up_proj, self.down_proj, tokens, rank=rank
        )


class Qwen3_5Attention(BaseOP):
    """Paged attention between zero-centered Q/K norms and a sigmoid output gate; inference
    runs the norms, the rotary and the K/V write as one fused prologue."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        *,
        rotary: RotaryEmbedding,
        site: Site,
        deferred: bool = False,
    ) -> None:
        """``deferred``: the residual norm after the attention sums inference's shares."""

        attention = config.attention
        self._hidden_size, self._attention = config.hidden_size, attention
        # Each rank holds whole query heads with their gates, and its KV heads (shared by
        # consecutive ranks when there are fewer KV heads than ranks). q_proj holds every
        # head's query and gate, so one LoRA A serves both slices.
        self.qkv_gate_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=attention.head_dim,
            num_qo_heads=attention.num_qo_heads,
            num_kv_heads=attention.num_kv_heads,
            has_bias=False,
            with_output_gate=True,
            lora="attention",
            lora_shared_a={"q_gate": ("q", "gate")},
            site=site,
        )
        # Q/K norms act within one head, so every rank holds their whole weights.
        self.q_norm = RMSNorm(
            attention.head_dim,
            eps=config.rms_norm_eps,
            zero_centered=True,
        )
        self.k_norm = RMSNorm(
            attention.head_dim,
            eps=config.rms_norm_eps,
            zero_centered=True,
        )
        self.attn = AttentionLayer(
            qkv=self.qkv_gate_proj,
            rotary=rotary,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            prologue=PrologueFactory(required=True),
        )
        self.o_proj = LinearRowParallel(
            attention.head_dim * attention.num_qo_heads,
            config.hidden_size,
            has_bias=False,
            # The gate is a column slice of the qkv_gate output rows.
            gate_of=self.qkv_gate_proj,
            lora="attention",
            site=site,
            deferred=deferred,
        )
        # The output gate follows q | k | v, which the attention layer splits exactly.
        gate = self.qkv_gate_proj.output_slices[-1]
        self._split_sizes = (gate.offset, gate.size)

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor, batch: ForwardInputs) -> torch.Tensor:
        qkv_gate = self.qkv_gate_proj.forward(x)
        del x
        qkv, gate = qkv_gate.split(self._split_sizes, dim=-1)
        output = self.attn.forward(qkv, batch)
        return self.o_proj.forward_gated(output, gate)

    def forward_training(
        self, x: torch.Tensor, positions: torch.Tensor, packing: Packing | None
    ) -> torch.Tensor:
        qkv, gate = self.qkv_gate_proj.forward_training(x).split(self._split_sizes, dim=-1)
        output = self.attn.forward_training(qkv, positions, packing)
        return self.o_proj.forward_training(output * torch.sigmoid(gate))

    def training_bytes(
        self, tokens: int, sequences: int, *, rank: int, stream: int
    ) -> tuple[int, int]:
        """``(saved, backward)`` of ``tokens`` rows in ``sequences`` sequences: the input
        stream, the mixer's output and its normalized input; the merged q/k/v/gate
        projection, sigmoid gate, attention and gated outputs; SDPA's rotated q and k and
        FP32 log-sum-exp; five low-rank inputs. Its backward holds the stream's gradient and
        FlashAttention's tiles."""

        hidden, attention = self._hidden_size, self._attention
        heads = attention.num_qo_heads
        query, kv = heads * attention.head_dim, attention.num_kv_heads * attention.head_dim
        saved = stream * hidden + 4 * hidden + 12 * query + 6 * kv + 4 * heads + 5 * 2 * rank
        backward = tokens * (stream * hidden + 12 * query + 8 * kv)
        return tokens * saved, backward + attention_backward_tile_bytes(
            sequences, heads, attention.head_dim
        )


class Qwen3_5LinearAttention(BaseOP):
    """A Gated DeltaNet mixer between a merged input projection and an output projection."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        *,
        site: Site,
        deferred: bool = False,
    ) -> None:
        """``deferred``: the residual norm after the mixer sums inference's shares."""

        linear = config.linear_attention
        # The mixer checks that the ranks split whole key heads before any projection
        # divides its rows.
        gdn = GatedDeltaNet(
            num_key_heads=linear.num_key_heads,
            num_value_heads=linear.num_value_heads,
            key_head_dim=linear.key_head_dim,
            value_head_dim=linear.value_head_dim,
            conv_kernel_size=linear.conv_kernel_size,
            site=site,
        )
        key_size = linear.num_key_heads * linear.key_head_dim
        value_size = linear.num_value_heads * linear.value_head_dim
        heads = linear.num_value_heads
        # The column order q | k | v | z | b | a is the projected rows ctikernels.gdn reads.
        # Each part is column-parallel by head, so a rank's rows are its own heads in
        # that order.
        # in_proj_qkv holds q, k and v, so one LoRA A serves the three slices.
        self.in_proj = LinearColParallelMerged(
            config.hidden_size,
            [key_size, key_size, value_size, value_size, heads, heads],
            has_bias=False,
            output_names=("q", "k", "v", "z", "b", "a"),
            lora="linear_attention",
            lora_shared_a={"qkv": ("q", "k", "v")},
            site=site,
        )
        self.gdn = gdn
        self.out_proj = LinearRowParallel(
            value_size,
            config.hidden_size,
            has_bias=False,
            lora="linear_attention",
            site=site,
            deferred=deferred,
        )

    @nvtx_annotate("GDN")
    def forward(self, x: torch.Tensor, batch: ForwardInputs) -> torch.Tensor:
        projected = self.in_proj.forward(x)
        del x
        return self.out_proj.forward(self.gdn.forward(projected, batch))

    def forward_training(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_sequence_length: int,
    ) -> torch.Tensor:
        """Train padded ``[batch, length, hidden]`` or packed ``[tokens, hidden]`` rows;
        ``cu_seqlens`` bounds the sequences of the flattened rows, none longer than
        ``max_sequence_length``."""

        projected = self.in_proj.forward_training(x)
        y = self.gdn.forward_training(projected.flatten(0, -2), cu_seqlens, max_sequence_length)
        return self.out_proj.forward_training(y.view(*projected.shape[:-1], -1))

    def training_bytes(
        self, tokens: int, sequences: int, *, rank: int, stream: int
    ) -> tuple[int, int]:
        """``(saved, backward)`` of ``tokens`` rows in ``sequences`` sequences: the Gated
        DeltaNet op's, the input stream and its normalized input, the gated-norm output the
        output projection saves and seven low-rank inputs; the backward also holds the
        stream's gradient. The mixer's output only joins the stream: nothing saves it."""

        hidden = self.out_proj.local_output_size
        values = self.gdn.num_value_heads * GatedDeltaNet.HEAD_DIM
        op_saved, op_backward = self.gdn.training_bytes(tokens, sequences)
        saved = op_saved + tokens * (stream * hidden + 2 * hidden + 2 * values + 7 * 2 * rank)
        return saved, op_backward + tokens * stream * hidden


class Qwen3_5DecoderLayer(BaseOP):
    """A pre-norm block: one token mixer, then the MLP.

    The mixer keeps its checkpoint name, ``self_attn`` or ``linear_attn``. Every forward, in
    inference and in training, carries the residual stream as a pair: ``x``, the last branch output not
    yet added, and ``residual``, ``None`` where ``x`` holds the whole stream.
    """

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        *,
        rotary: RotaryEmbedding,
        site: Site,
    ) -> None:
        self.kind = config.layer_types[layer_id]
        reduction = site.residual_reduction(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            zero_centered=True,
        )
        deferred = reduction is not None
        mixer: Qwen3_5Attention | Qwen3_5LinearAttention
        if self.kind is LayerKind.FULL_ATTENTION:
            mixer = self.self_attn = Qwen3_5Attention(
                config, rotary=rotary, site=site, deferred=deferred
            )
        else:
            mixer = self.linear_attn = Qwen3_5LinearAttention(config, site=site, deferred=deferred)
        self.mlp = Qwen3_5MLP(config, site=site, deferred=deferred)
        self.input_layernorm = ResidualStreamRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, reduction=reduction
        )
        self.post_attention_layernorm = ResidualStreamRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, reduction=reduction
        )

        self._hidden_size = config.hidden_size
        self._mixer = mixer
        self._layer_id = layer_id

    def enter(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The input norm; the residual it returns is the stream after the previous layer."""

        return self.input_layernorm.forward(x, residual)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        batch: ForwardInputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer after ``enter``."""

        x = self._mixer.forward(x, batch)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual

    def mixer_training(
        self,
        stream: torch.Tensor,
        positions: torch.Tensor,
        packing: Packing | None,
        cu_seqlens: torch.Tensor,
        max_sequence_length: int,
    ) -> torch.Tensor:
        """The stream after the input norm and token mixer: attention reads ``positions``
        and, for packed sequences, ``packing``; the Gated DeltaNet the sequence bounds of
        the flattened rows, ``cu_seqlens``, none longer than ``max_sequence_length``."""

        x, stream = self.input_layernorm.forward_training(stream)
        if self.kind is LayerKind.FULL_ATTENTION:
            x = self.self_attn.forward_training(x, positions, packing)
        else:
            x = self.linear_attn.forward_training(x, cu_seqlens, max_sequence_length)
        return stream + x

    def mlp_training(self, stream: torch.Tensor) -> torch.Tensor:
        """The stream after the post-attention norm and MLP."""

        x, stream = self.post_attention_layernorm.forward_training(stream)
        return stream + self.mlp.forward_training(x)

    def training_bytes(
        self, tokens: int, sequences: int, *, rank: int, stream: int, row_tile: int | None
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """The mixer's and the MLP's ``(saved, backward)``; with ``row_tile`` the MLP trains
        in tiles of that many rows: it keeps only its input, and its backward holds one
        tile's recomputation beside the stream's incoming and outgoing gradients."""

        stream_bytes = tokens * stream * self._hidden_size
        mixer = self._mixer.training_bytes(tokens, sequences, rank=rank, stream=stream)
        if row_tile is None:
            # The MLP's norm saves its input, the post-mixer sum.
            mlp_saved, mlp_backward = self.mlp.training_bytes(tokens, rank=rank)
            return mixer, (mlp_saved + stream_bytes, mlp_backward + stream_bytes)
        rows = min(tokens, row_tile)
        tile_saved, tile_backward = self.mlp.training_bytes(rows, rank=rank)
        tile_stream = rows * stream * self._hidden_size
        return mixer, (0, tile_saved + tile_backward + tile_stream + 2 * stream_bytes)


class Qwen3_5TextModel(BaseOP):
    """Qwen3.5 embeddings, decoder layers and final norm; the stream is in the model dtype."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        *,
        rotary: RotaryEmbedding,
        site: Site,
    ) -> None:
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            site=site,
        )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, rotary=rotary, site=site)
                for layer_id in range(len(config.layer_types))
            ]
        )
        self.norm = ResidualStreamRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            reduction=site.residual_reduction(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                zero_centered=True,
            ),
        )
        self._activation_checkpoint: ReentrantActivationCheckpoint | None = None
        self._services = site.services

    def enable_activation_checkpointing(self, policy: ReentrantActivationCheckpoint) -> None:
        """Run the training blocks under the runner's checkpoint policy."""

        self._activation_checkpoint = policy

    def forward(self, embeddings: torch.Tensor, batch: ForwardInputs) -> torch.Tensor:
        """Run the layers on the step's input embeddings and normalize the last rows."""

        x = embeddings
        residual: torch.Tensor | None = None
        taps = self._services.taps
        for layer_id, layer in enumerate(self.layers.op_list):
            x, residual = layer.enter(x, residual)
            if taps is not None and layer_id > 0:
                taps.store(layer_id - 1, residual)
            x, residual = layer.forward(x, residual, batch)
        if taps is None:
            return self.norm.forward_last(x, residual)
        x, residual = self.norm.forward(x, residual)
        taps.store(len(self.layers.op_list) - 1, residual)
        return x

    def forward_training(self, input_ids: torch.Tensor, packing: Packing | None) -> torch.Tensor:
        """Final hidden states of every input position, flat: of padded ``[batch, length]``
        rows, or with ``packing`` of packed sequences."""

        if packing is None:
            batch_size, length = input_ids.shape
            device = input_ids.device
            positions = torch.arange(length, device=device)
            # Training rows are right-padded, so row b is one sequence: flattened tokens
            # [b * length, (b + 1) * length). Its padding follows every target it holds.
            cu_seqlens = torch.arange(
                0, batch_size * length + 1, length, dtype=torch.int32, device=device
            )
            max_sequence_length = length
        else:
            positions, cu_seqlens = packing.positions, packing.cu_seqlens
            max_sequence_length = packing.max_sequence_length
        x = self.embed_tokens.forward_training(input_ids)
        return self._forward_blocks(x, positions, packing, cu_seqlens, max_sequence_length)

    def _forward_blocks(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        packing: Packing | None,
        cu_seqlens: torch.Tensor,
        max_sequence_length: int,
    ) -> torch.Tensor:
        """Run every decoder layer's mixer and MLP, under the checkpoint policy if one is
        set, then normalize the stream and flatten its rows."""

        mixer = partial(
            Qwen3_5DecoderLayer.mixer_training,
            positions=positions,
            packing=packing,
            cu_seqlens=cu_seqlens,
            max_sequence_length=max_sequence_length,
        )
        layers = tuple(
            (partial(mixer, layer), RowWise(layer.mlp_training)) for layer in self.layers.op_list
        )
        policy = self._activation_checkpoint
        x = run_segments(layers, x) if policy is None else policy.run_blocks(layers, x)
        return self.norm.forward_training(x)[0].flatten(0, -2)


class Qwen3_5ForConditionalGeneration(TrainingOutputMixin, BaseLLMModel[Qwen3_5Config]):
    """Qwen3.5 text generation from an image-text checkpoint or a text-only export."""

    CAPABILITIES = (
        ModelCapability.LORA
        | ModelCapability.TRAINING
        | ModelCapability.TENSOR_PARALLEL
        | ModelCapability.PREFIX_REUSE
        | ModelCapability.SPECULATIVE_TARGET
    )
    LORA_TARGETS = ("attention", "linear_attention", "mlp")
    CONFIG = Qwen3_5Config
    RECURRENT_EXPORT_GRANULARITY = GatedDeltaNet.EXPORT_GRANULARITY

    @classmethod
    def validate_hf_config(cls, hf_config: PretrainedConfig) -> None:
        # Reading the config rejects what the layers would run differently (Qwen3_5TextConfig).
        Qwen3_5Config.from_hf(hf_config)
        # Hugging Face Qwen3.5 ignores both soft-capping fields; a checkpoint that
        # declares them still expects capped logits.
        for name in ("attn_logit_softcapping", "final_logit_softcapping"):
            if value := getattr(hf_config, name, None):
                raise ValueError(f"{cls.__name__} does not implement {name}={value}")

    @classmethod
    def attention_requirements(
        cls, hf_config: PretrainedConfig
    ) -> tuple[AttentionRequirement, ...]:
        attention = Qwen3_5AttentionConfig.from_hf(hf_config.get_text_config())
        return (
            AttentionRequirement.mha(
                attention.num_qo_heads, attention.num_kv_heads, attention.head_dim
            ),
        )

    @classmethod
    def exports_prefill_states(
        cls, hf_config: PretrainedConfig, device: torch.device, *, num_ranks: int
    ) -> bool:
        linear = Qwen3_5LinearAttentionConfig.from_hf(hf_config.get_text_config())
        num_key_heads, num_value_heads = GatedDeltaNet.rank_heads(
            linear.num_key_heads, linear.num_value_heads, num_ranks
        )
        return prefill_exports_states(
            device, num_key_heads=num_key_heads, num_value_heads=num_value_heads
        )

    def __init__(
        self,
        config: Qwen3_5Config,
        *,
        site: Site,
        image_preprocessing: ImagePreprocessing | None = None,
    ) -> None:
        text = self._text_config = config.text
        self.checkpoint_mapping = _checkpoint_mapping(
            text.linear_attention, has_vision=config.vision is not None
        )
        attention = text.attention
        rotary = RotaryEmbedding.create(
            attention.head_dim,
            attention.rotary_dim,
            attention.max_position,
            attention.rope_theta,
            mrope_sections=attention.mrope_section,
            site=site,
        )
        self.model = Qwen3_5TextModel(text, rotary=rotary, site=site)
        self.lm_head = ParallelLMHead(
            num_embeddings=text.vocab_size,
            embedding_dim=text.hidden_size,
            tied_embedding=self.model.embed_tokens if text.tie_word_embeddings else None,
            site=site,
        )
        self.visual = (
            None
            if config.vision is None
            else Qwen3_5VisionModel(config.vision, image_preprocessing, site=site)
        )
        self._vision_token_ids = tuple(
            token_id
            for token_id in (config.image_token_id, config.video_token_id)
            if token_id is not None
        )
        self._services = site.services
        super().__init__()

    def iter_attention_layers(self) -> Iterator[AttentionLayer]:
        return (
            layer.self_attn.attn
            for layer in self.model.layers.op_list
            if layer.kind is LayerKind.FULL_ATTENTION
        )

    def iter_recurrent_layers(self) -> tuple[GatedDeltaNet, ...]:
        return tuple(
            layer.linear_attn.gdn
            for layer in self.model.layers.op_list
            if layer.kind is LayerKind.LINEAR_ATTENTION
        )

    def rope_position_axes(self) -> int:
        # Interleaved M-RoPE reads temporal, height and width rows, even for text.
        return 3

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
        """Return hidden states with image embeddings at their placeholder rows.

        Text-only and decode steps carry no images and never reach the tower.
        """

        embeddings = self.model.embed_tokens.forward(batch.input_ids)
        if batch.images is not None:
            self._write_image_tokens(embeddings, batch)
        return self.model.forward(embeddings, batch)

    def warm_up_image_encoder(
        self,
        *,
        max_tokens: int,
        max_image_tokens: int,
        device: torch.device,
    ) -> bool:
        if self.visual is None:
            return False
        # forward_hidden holds the step's token embeddings while the tower runs.
        weight = self.model.embed_tokens.weight
        embeddings = torch.empty((max_tokens, weight.shape[1]), dtype=weight.dtype, device=device)
        images = self.visual.largest_image_batch(
            max_tokens=max_tokens, max_image_tokens=max_image_tokens, device=device
        )
        embeddings[images.dst] = self.visual.forward(images).to(embeddings.dtype)
        return True

    def vision_tower(self) -> VisionTower | None:
        return self.visual

    def forward(self, batch: ForwardInputs) -> torch.Tensor:
        return self.compute_logits(self.forward_hidden(batch), batch)

    def validate_training_inputs(self, token_ids: torch.Tensor) -> None:
        """Reject image and video placeholder tokens in training inputs."""

        placeholders = torch.tensor(self._vision_token_ids, dtype=token_ids.dtype)
        found = token_ids[torch.isin(token_ids, placeholders)].unique()
        if found.numel():
            raise ValueError(
                f"Qwen3.5 training takes text only, but the inputs hold vision placeholder "
                f"tokens {found.tolist()}, whose patches only the vision tower embeds"
            )

    def _write_image_tokens(self, embeddings: torch.Tensor, batch: ForwardInputs) -> None:
        """Load the step's cached image rows, encode its other images into the encoder-output
        rows as the batch's image encoding runs the tower, then load theirs."""

        images = batch.images
        rows, cached = self._services.image_rows, images.cached
        if self.visual is None or rows is None:
            raise ValueError("this Qwen3.5 model holds no vision tower or encoder-output rows")
        embeddings[images.dst[:cached]] = rows[images.src_rows[:cached]]
        if images.grids:
            encoded = batch.image_encoding.encode(self.visual, images)
            rows[images.store_rows] = encoded.to(rows.dtype)
        embeddings[images.dst[cached:]] = rows[images.src_rows[cached:]]


__all__ = ["Qwen3_5ForConditionalGeneration"]
