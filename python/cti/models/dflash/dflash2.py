# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Draft semantics follow SGLang 0.5.22 (`python/sglang/srt/models/dflash.py`) and vLLM 0.31.0
# (`vllm/model_executor/models/qwen3_dflash2.py`); written for cti, no code reused.

"""DFlash2 draft: the v1 decoder with dynamic block convolutions and a candidate selector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cti.layers import BaseOP, BlockConv, LinearReplicated, RotaryEmbedding

from .config import DFlash2Extension, DFlashConfig
from .model import DFlashDecoderLayer, DFlashDraftModel

if TYPE_CHECKING:
    from cti.forward import ForwardInputs
    from cti.layers import Site


class DFlash2DecoderLayer(DFlashDecoderLayer):
    """A v1 layer whose attention and MLP branches are each wrapped in a ``BlockConv``."""

    def __init__(
        self, config: DFlashConfig, round_width: int, *, rotary: RotaryEmbedding, site: Site
    ) -> None:
        super().__init__(config.model, rotary=rotary, site=site)
        assert config.dflash2 is not None
        hidden, q = config.model.hidden_size, round_width
        conv = config.dflash2
        self.attention_conv = BlockConv(
            hidden, conv.conv_taps, conv.conv_segment_size, q, site=site
        )
        self.mlp_conv = BlockConv(hidden, conv.conv_taps, conv.conv_segment_size, q, site=site)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None, batch: ForwardInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A branch's convolution is linear in its output, so it convolves a partial output
        # alike when the norm after it completes the sum.
        x, residual = self.input_layernorm.forward(x, residual)
        x, deltas = self.attention_conv.prepare(x)
        x = self.attention_conv.finish(self.self_attn.forward(x, batch), deltas)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x, deltas = self.mlp_conv.prepare(x)
        return self.mlp_conv.finish(self.mlp.forward(x), deltas), residual


class CandidateSelector(BaseOP):
    """Scores each proposal position's candidates given the chosen predecessor, as vLLM and
    SGLang do: ``unary[c] + <A[pred] * P h, B[c]>`` with ``[V, C]`` codebooks A
    (predecessor) and B (successor) and the hidden projection P, whole in every process,
    in the model dtype (FP32 accumulation); the FP32 unary logits add in FP32."""

    def __init__(
        self, hidden_size: int, vocab_size: int, extension: DFlash2Extension, *, site: Site
    ) -> None:
        self.top_k = extension.candidates
        self.predecessor_codebook = torch.empty(vocab_size, extension.code_size)
        self.successor_codebook = torch.empty(vocab_size, extension.code_size)
        self.hidden_projection = LinearReplicated(
            hidden_size, extension.code_size, has_bias=False, site=site
        )

    def lattice(
        self, hidden: torch.Tensor, ids: torch.Tensor, unary: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        """FP32 ``[B, G, K, K]`` scores of each position's candidates (last axis) after each
        predecessor (third axis): the previous position's candidates, or at position 0 the
        anchor for every predecessor. ``hidden`` is ``[B, G, H]``, ``ids`` (ascending) and
        their logits ``unary`` ``[B, G, K]``, ``anchors`` ``[B]``."""

        projected = self.hidden_projection.forward(hidden.flatten(0, 1))
        predecessors = torch.cat(
            (anchors[:, None, None].expand(-1, 1, self.top_k), ids[:, :-1]), dim=1
        )
        edges = torch.einsum(
            "bgpr,bgcr->bgpc",
            self.predecessor_codebook[predecessors.long()]
            * projected.view(*hidden.shape[:2], 1, -1),
            self.successor_codebook[ids.long()],
        )
        return unary.float()[:, :, None] + edges


class DFlash2DraftModel(DFlashDraftModel):
    """The v1 draft with DFlash2 layers (non-causal, each branch convolved along its block)
    and a ``candidate_selector``, which the round's chain walk scores with. Its convolution
    and selector weights are whole in every process."""

    def __init__(self, config: DFlashConfig, *, round_width: int, site: Site) -> None:
        super().__init__(config, round_width=round_width, site=site)
        assert config.dflash2 is not None
        model = config.model
        self.candidate_selector = CandidateSelector(
            model.hidden_size, model.vocab_size, config.dflash2, site=site
        )

    def _decoder_layer(
        self, config: DFlashConfig, *, round_width: int, rotary: RotaryEmbedding, site: Site
    ) -> DFlash2DecoderLayer:
        return DFlash2DecoderLayer(config, round_width, rotary=rotary, site=site)


__all__ = ["CandidateSelector", "DFlash2DraftModel"]
