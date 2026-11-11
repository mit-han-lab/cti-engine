from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..config import RotaryConfig, num_key_value_heads

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ..config import ModelConfig

_DFLASH2_FIELDS = ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k")
# Upstream draft scalars; only their identity values are implemented.
_DFLASH_IDENTITY_SCALARS = (("input_embedding_scale", 1.0), ("output_multiplier", 1.0))
# Each draft class's verify width Q.
_BLOCK_SIZES = MappingProxyType({"DFlashDraftModel": 16, "DFlash2DraftModel": 8})


@dataclass(frozen=True)
class DraftDecoderConfig:
    """A draft's decoder layers: Qwen3-style full or sliding attention and a dense MLP."""

    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> DraftDecoderConfig:
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_key_value_heads(config),
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
            rotary_config=RotaryConfig.from_hf(config, head_dim),
            hidden_act=config.hidden_act,
        )


@dataclass(frozen=True)
class DFlash2Extension:
    """DFlash2's per-layer convolutions, of ``conv_taps`` taps (``conv_kernel_size``) whose
    coefficients each segment of ``conv_segment_size`` channels shares (``conv_group_size``),
    and its candidate selector, which scores ``candidates`` target-head candidates per
    position (``selector_top_k``) through codebooks of width ``code_size``
    (``selector_rank``)."""

    conv_taps: int
    conv_segment_size: int
    code_size: int
    candidates: int


@dataclass(frozen=True)
class DFlashConfig:
    """A v1 or DFlash2 draft checkpoint checked against its target (docs/dflash.md).

    ``block_size`` Q counts the anchor and Q - 1 mask positions; ``target_layer_ids`` are
    the zero-based target layers whose output streams the draft reads. ``layer_windows``
    holds each draft layer's sliding window, or ``None`` for a full layer that sees all
    context and block keys. ``dflash2`` is set for a DFlash2 draft, whose layers are
    non-causal; a v1 sliding layer is causal.
    """

    model: DraftDecoderConfig
    target_layer_ids: tuple[int, ...]
    block_size: int
    mask_token_id: int
    layer_windows: tuple[int | None, ...]
    dflash2: DFlash2Extension | None = None

    @classmethod
    def from_hf(
        cls, document: Mapping[str, Any], hf_config: PretrainedConfig, target: ModelConfig
    ) -> DFlashConfig:
        """``document`` is the draft's ``config.json``, ``hf_config`` Transformers' reading
        of it."""

        model = DraftDecoderConfig.from_hf(hf_config)

        architectures = document.get("architectures")
        if not isinstance(architectures, list) or len(architectures) != 1:
            raise ValueError(f"unsupported draft architectures {architectures!r}")
        architecture = architectures[0]
        if architecture not in _BLOCK_SIZES:
            raise ValueError(
                f"unsupported draft architecture {architecture!r}; "
                f"{', '.join(_BLOCK_SIZES)} are implemented"
            )
        spec = document.get("dflash_config") or {}
        if not isinstance(spec, Mapping):
            raise ValueError("dflash_config must be a mapping")

        def value(name: str, default: Any = None) -> Any:
            return spec.get(name, document.get(name, default))

        dflash2 = None
        if architecture == "DFlashDraftModel":
            if "is_causal" in document:
                raise ValueError("a DFlash v1 config must not declare is_causal")
            if extensions := [name for name in _DFLASH2_FIELDS if name in spec or name in document]:
                raise ValueError(f"DFlash2 fields {extensions} are invalid for a v1 draft")
        else:
            if document.get("is_causal") is not False:
                raise ValueError("a DFlash2 config must declare is_causal false")
            dflash2 = _dflash2_extension(spec, model)
        for name, identity in _DFLASH_IDENTITY_SCALARS:
            if value(name, identity) != identity:
                raise ValueError(f"DFlash {name} is not implemented")
        if value("final_logit_softcapping"):
            raise ValueError("DFlash final_logit_softcapping is not implemented")
        layer_types = document.get("layer_types") or ["full_attention"] * model.num_layers
        window = document.get("sliding_window")
        kinds = {"full_attention": None, "sliding_attention": window}
        if (
            not isinstance(layer_types, list)
            or len(layer_types) != model.num_layers
            or any(kind not in kinds for kind in layer_types)
        ):
            raise ValueError(f"draft layer_types {layer_types} are not implemented")
        if "sliding_attention" in layer_types and (type(window) is not int or window < 1):
            raise ValueError(f"invalid draft sliding_window {window!r}")
        if document.get("attention_bias") or document.get("mlp_bias"):
            raise ValueError("draft projection bias is not implemented")
        experts = document.get("num_experts") or document.get("num_local_experts")
        if model.hidden_act != "silu" or experts:
            raise ValueError("the draft must be a dense SiLU decoder")
        if document.get("num_target_layers") != target.num_layers:
            raise ValueError(
                f"draft num_target_layers {document.get('num_target_layers')} does not match "
                f"the target's {target.num_layers} layers"
            )
        if (model.hidden_size, model.vocab_size) != (target.hidden_size, target.vocab_size):
            raise ValueError("the draft must share the target's hidden size and vocabulary")
        block_size, pinned = value("block_size"), _BLOCK_SIZES[architecture]
        if block_size != pinned or type(block_size) is not int:
            raise ValueError(
                f"{architecture} verifies blocks of {pinned} tokens, not {block_size!r}"
            )
        mask_token_id = value("mask_token_id")
        if type(mask_token_id) is not int or not 0 <= mask_token_id < target.vocab_size:
            raise ValueError(f"invalid DFlash mask_token_id {mask_token_id!r}")
        layer_ids = value("target_layer_ids")
        # Taps read the pre-norm stream, which upstream's last-layer entry is not.
        if (
            not isinstance(layer_ids, list)
            or not layer_ids
            or any(type(i) is not int or not 0 <= i < target.num_layers - 1 for i in layer_ids)
            or len(set(layer_ids)) != len(layer_ids)
        ):
            raise ValueError(f"invalid DFlash target_layer_ids {layer_ids!r}")
        layer_windows = tuple(kinds[kind] for kind in layer_types)
        return cls(model, tuple(layer_ids), block_size, mask_token_id, layer_windows, dflash2)

    @property
    def layer_masks(self) -> tuple[tuple[bool, int | None], ...]:
        """Each layer's ``(causal, window)`` attention mask over context and block keys: a
        v1 sliding layer is causal, a DFlash2 one sees every block key."""

        causal = self.dflash2 is None
        return tuple((causal and window is not None, window) for window in self.layer_windows)


def _dflash2_extension(spec: Mapping[str, Any], model: DraftDecoderConfig) -> DFlash2Extension:
    """The DFlash2 fields of ``dflash_config``, checked against the draft's geometry."""

    fields = [spec.get(name) for name in _DFLASH2_FIELDS]
    if invalid := [
        name
        for name, field in zip(_DFLASH2_FIELDS, fields, strict=True)
        if type(field) is not int or field < 1
    ]:
        raise ValueError(f"DFlash2 fields {invalid} must be positive integers")
    extension = DFlash2Extension(*fields)
    if model.hidden_size % extension.conv_segment_size:
        raise ValueError(
            f"conv_group_size {extension.conv_segment_size} does not divide the hidden size "
            f"{model.hidden_size}"
        )
    # K1 samples and verifies over exactly 16 candidates.
    if extension.candidates != 16:
        raise ValueError(f"selector_top_k {extension.candidates} is not implemented")
    return extension
