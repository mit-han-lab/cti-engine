"""Qwen3-MoE's topology: the shared text geometry and the fields only its layers read.

Checked against transformers 5.16.1 ``configuration_qwen3_moe`` and ``modeling_qwen3_moe``:
reading a config implements each field the modeling code reads or refuses it by name.

- Implemented: the attention and expert geometry, ``attention_bias``, ``norm_topk_prob``,
  ``tie_word_embeddings`` and the RoPE types the rotary embedding implements.
- Training only, so inference reads nothing from them: ``attention_dropout`` (Transformers
  drops attention weights only in training) and ``output_router_logits`` with
  ``router_aux_loss_coef`` (the load-balancing term of a training loss). Training refuses a
  non-zero dropout and the router loss (``refuse_training_only_fields``).
- Refused: dense MLP layers (``mlp_only_layers``, ``decoder_sparse_step``), an expert
  activation other than SiLU, sliding-window attention, dual-chunk attention
  (``dual_chunk_attention_config``, which Qwen's ``config_1m.json`` adds) and logit
  soft-capping, which Transformers ignores but a checkpoint declaring it expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import ModelConfig, RotaryConfig, UnimplementedConfigError, num_key_value_heads

if TYPE_CHECKING:
    from collections.abc import Mapping

    from transformers import PretrainedConfig

_ARCHITECTURE = "Qwen3MoeForCausalLM"


@dataclass(frozen=True)
class Qwen3MoeConfig:
    """A Qwen3-MoE decoder's geometry: every layer full attention and a sparse MLP."""

    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    # Whether the query, key, value and output projections add a bias.
    attention_bias: bool = False
    # Read by training alone: Transformers drops attention weights at this rate, and with
    # output_router_logits adds the router's load-balancing loss to the training loss.
    attention_dropout: float = 0.0
    output_router_logits: bool = False

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> Qwen3MoeConfig:
        text = config.get_text_config()
        declared = text.to_dict()
        _refuse_unimplemented(declared)
        head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
        num_experts = getattr(text, "num_local_experts", getattr(text, "num_experts", 0))
        num_experts_per_tok = getattr(text, "num_experts_per_tok", 0)
        if not 0 < num_experts_per_tok <= num_experts:
            raise ValueError(
                f"num_experts_per_tok={num_experts_per_tok} must be in "
                f"[1, num_experts={num_experts}]"
            )
        return cls(
            num_layers=text.num_hidden_layers,
            num_qo_heads=text.num_attention_heads,
            num_kv_heads=num_key_value_heads(text),
            head_dim=head_dim,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
            rms_norm_eps=text.rms_norm_eps,
            rotary_config=RotaryConfig.from_hf(text, head_dim),
            tie_word_embeddings=getattr(text, "tie_word_embeddings", False),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
            norm_topk_prob=getattr(text, "norm_topk_prob", False),
            attention_bias=bool(declared.get("attention_bias", False)),
            attention_dropout=float(declared.get("attention_dropout") or 0.0),
            output_router_logits=bool(declared.get("output_router_logits", False)),
        )

    def engine(self, architectures: tuple[str, ...]) -> ModelConfig:
        return ModelConfig(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            max_position=self.rotary_config.max_position,
            tie_word_embeddings=self.tie_word_embeddings,
            architectures=architectures,
        )

    def refuse_training_only_fields(self) -> None:
        """Raise ``UnimplementedConfigError`` for a training-only field that training would
        ignore: its attention drops nothing and its loss has no router term."""

        if self.attention_dropout:
            _refuse("attention_dropout", self.attention_dropout, "training attention drops nothing")
        if self.output_router_logits:
            _refuse("output_router_logits", True, "the training loss has no router term")


def _refuse_unimplemented(fields: Mapping[str, Any]) -> None:
    """Raise ``UnimplementedConfigError`` for the first field these layers would run
    differently from Transformers."""

    # Layers listed here, and all but every decoder_sparse_step-th layer, use a dense MLP.
    if mlp_only_layers := fields.get("mlp_only_layers"):
        _refuse("mlp_only_layers", mlp_only_layers, "every layer is sparse")
    if (decoder_sparse_step := fields.get("decoder_sparse_step", 1)) != 1:
        _refuse("decoder_sparse_step", decoder_sparse_step, "every layer is sparse")
    # Transformers applies hidden_act inside the experts.
    if (hidden_act := fields.get("hidden_act", "silu")) != "silu":
        _refuse("hidden_act", hidden_act, "the experts apply SiLU")
    # Transformers' config keeps sliding_window only with use_sliding_window, and then windows
    # every layer, whatever max_window_layers says.
    if (sliding_window := fields.get("sliding_window")) is not None:
        _refuse("sliding_window", sliding_window, "attention is full")
    if dual_chunk := fields.get("dual_chunk_attention_config"):
        _refuse("dual_chunk_attention_config", dual_chunk, "attention is one causal pass")
    for name in ("attn_logit_softcapping", "final_logit_softcapping"):
        if capping := fields.get(name):
            _refuse(name, capping, "logits are not soft-capped")


def _refuse(field: str, value: Any, implemented: str) -> None:
    raise UnimplementedConfigError(_ARCHITECTURE, field, value, implemented)


__all__ = ["Qwen3MoeConfig"]
