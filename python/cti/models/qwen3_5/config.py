"""Qwen3.5 configuration: the hybrid text layer schedule, both mixers and the vision tower.

Read from a Hugging Face ``qwen3_5`` (image-text) or ``qwen3_5_text`` config with the
semantics of transformers 5.16.1 ``configuration_qwen3_5.py`` and ``modeling_qwen3_5.py``.
A setting the engine would execute differently from that reference is rejected, never
ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar

from cti.layers.gated_delta_net import GatedDeltaNet

from ..config import ModelConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig


class LayerKind(Enum):
    """The token mixer of one decoder layer, valued as in ``layer_types``."""

    FULL_ATTENTION = "full_attention"
    LINEAR_ATTENTION = "linear_attention"


@dataclass(frozen=True)
class Qwen3_5AttentionConfig:
    """Gated full attention with partial, interleaved multi-axis rotary (M-RoPE).

    The leading ``rotary_dim`` channels of each head rotate; ``mrope_section`` counts
    the rotary frequencies that read the temporal, height and width positions.
    """

    # q_proj interleaves a 256-wide query and output gate per head; the model's
    # checkpoint mapping splits them at this width.
    HEAD_DIM: ClassVar[int] = 256
    # The fused attention prologue (ctikernels.attention.inference.Prologue) rotates the
    # leading 64 channels.
    ROTARY_DIM: ClassVar[int] = 64

    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    max_position: int
    mrope_section: tuple[int, int, int]

    def __post_init__(self) -> None:
        _require_positive_integers(
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            max_position=self.max_position,
        )
        if self.num_qo_heads % self.num_kv_heads:
            raise ValueError(
                f"query heads ({self.num_qo_heads}) must be divisible by KV heads "
                f"({self.num_kv_heads})"
            )
        if self.head_dim != self.HEAD_DIM:
            raise ValueError(f"Qwen3.5 attention head_dim must be {self.HEAD_DIM}: {self.head_dim}")
        if self.rotary_dim != self.ROTARY_DIM:
            raise ValueError(
                f"Qwen3.5 attention rotary_dim must be {self.ROTARY_DIM}: {self.rotary_dim}"
            )
        if (
            isinstance(self.rope_theta, bool)
            or not isinstance(self.rope_theta, Real)
            or not math.isfinite(self.rope_theta)
            or self.rope_theta <= 0
        ):
            raise ValueError(f"rope_theta must be positive and finite: {self.rope_theta}")
        sections = self.mrope_section
        if (
            not isinstance(sections, tuple)
            or len(sections) != 3
            or any(type(size) is not int or size < 0 for size in sections)
        ):
            raise ValueError(f"mrope_section must be a tuple of three sizes: {sections}")
        if 2 * sum(sections) != self.rotary_dim:
            raise ValueError(
                f"mrope_section {sections} must count half of rotary_dim {self.rotary_dim}"
            )

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> Qwen3_5AttentionConfig:
        if config.attention_bias:
            raise ValueError("Qwen3.5 does not support attention bias")
        # The reference always gates the attention output and never reads this field.
        if getattr(config, "attn_output_gate", True) is not True:
            raise ValueError("Qwen3.5 requires the attention output gate")
        rope: dict[str, Any] = config.rope_parameters
        if rope.get("rope_type") != "default":
            raise ValueError(f"Qwen3.5 supports only default RoPE: {rope.get('rope_type')!r}")
        # The reference always interleaves the sections and never reads
        # mrope_interleaved, so a config declaring chunked sections is ambiguous.
        if rope.get("mrope_interleaved", True) is not True:
            raise ValueError("Qwen3.5 requires interleaved M-RoPE sections")
        rotary_dim = config.head_dim * rope.get("partial_rotary_factor", 1.0)
        if not float(rotary_dim).is_integer():
            raise ValueError(
                f"partial_rotary_factor must give an integral rotary_dim, got {rotary_dim}"
            )
        return cls(
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_dim=int(rotary_dim),
            rope_theta=rope.get("rope_theta"),
            max_position=config.max_position_embeddings,
            # The reference's default sections (Qwen3_5TextRotaryEmbedding).
            mrope_section=tuple(rope.get("mrope_section", (11, 11, 10))),
        )


@dataclass(frozen=True)
class Qwen3_5LinearAttentionConfig:
    """Gated DeltaNet geometry; ``ctikernels.gdn`` fixes the head width, the convolution
    and the SiLU gate of the output norm."""

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int

    def __post_init__(self) -> None:
        _require_positive_integers(
            num_key_heads=self.num_key_heads,
            num_value_heads=self.num_value_heads,
        )
        if self.num_value_heads % self.num_key_heads:
            raise ValueError(
                f"linear-attention value heads ({self.num_value_heads}) must be a multiple "
                f"of key heads ({self.num_key_heads})"
            )
        if self.key_head_dim != GatedDeltaNet.HEAD_DIM or (
            self.value_head_dim != GatedDeltaNet.HEAD_DIM
        ):
            raise ValueError(
                f"linear-attention head dimensions must be {GatedDeltaNet.HEAD_DIM}, got key "
                f"{self.key_head_dim} and value {self.value_head_dim}"
            )
        if self.conv_kernel_size != GatedDeltaNet.CONV_KERNEL_SIZE:
            raise ValueError(
                "linear-attention convolution kernel size must be "
                f"{GatedDeltaNet.CONV_KERNEL_SIZE}: {self.conv_kernel_size}"
            )

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> Qwen3_5LinearAttentionConfig:
        # The reference never reads output_gate_type and gates the output norm with SiLU;
        # vLLM reads it as that gate's activation, where swish is SiLU.
        gate = getattr(config, "output_gate_type", "silu")
        if gate not in ("silu", "swish"):
            raise ValueError(
                f"Qwen3.5 gates the Gated DeltaNet output norm with SiLU: output_gate_type {gate!r}"
            )
        return cls(
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
        )


@dataclass(frozen=True)
class Qwen3_5TextConfig:
    """The dense hybrid text model.

    Every RMSNorm except the Gated DeltaNet output norm scales by ``1 + weight``; the
    MLP and the Gated DeltaNet convolution use SiLU.
    """

    # ctikernels.gdn fixes its gated-norm epsilon, and the reference passes one epsilon
    # to all of the model's norms.
    RMS_NORM_EPS: ClassVar[float] = 1e-6

    hidden_size: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    layer_types: tuple[LayerKind, ...]
    attention: Qwen3_5AttentionConfig
    linear_attention: Qwen3_5LinearAttentionConfig

    def __post_init__(self) -> None:
        _require_positive_integers(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
        )
        if self.rms_norm_eps != self.RMS_NORM_EPS:
            raise ValueError(
                f"Qwen3.5 requires RMSNorm epsilon {self.RMS_NORM_EPS}: {self.rms_norm_eps}"
            )
        if not isinstance(self.tie_word_embeddings, bool):
            raise ValueError("tie_word_embeddings must be a bool")
        if (
            not isinstance(self.layer_types, tuple)
            or not self.layer_types
            or not all(isinstance(kind, LayerKind) for kind in self.layer_types)
        ):
            raise ValueError("layer_types must be a non-empty tuple of LayerKind")

    @classmethod
    def from_hf(cls, config: PretrainedConfig, *, tie_word_embeddings: bool) -> Qwen3_5TextConfig:
        """Read a ``qwen3_5_text`` config; the caller resolves which field ties the head."""

        if config.hidden_act != "silu":
            raise ValueError(f"Qwen3.5 requires SiLU activation: {config.hidden_act!r}")
        # transformers derives layer_types from full_attention_interval when absent and
        # checks that they name num_hidden_layers layers.
        layer_types = config.layer_types
        try:
            kinds = tuple(LayerKind(kind) for kind in layer_types)
        except ValueError:
            raise ValueError(f"unsupported Qwen3.5 layer type in {layer_types}") from None
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            layer_types=kinds,
            attention=Qwen3_5AttentionConfig.from_hf(config),
            linear_attention=Qwen3_5LinearAttentionConfig.from_hf(config),
        )


@dataclass(frozen=True)
class Qwen3_5VisionConfig:
    """The vision tower: a ViT over image patches and a projector into the text model.

    A patch is ``temporal_patch_size`` frames of ``in_channels`` channels and
    ``patch_size`` square pixels. Patches embed to ``hidden_size``, pass ``depth`` pre-norm
    blocks of ``num_heads`` bidirectional heads with 2-D rotary positions, and every
    ``spatial_merge_size`` square of them becomes one ``out_hidden_size`` token. A learned
    square grid of ``num_position_embeddings`` positions is resampled to each image.
    """

    # Constants of transformers 5.16.1 Qwen3_5VisionModel, never read from the config:
    # every LayerNorm's epsilon and the rotary base of the patch rows and columns.
    LAYER_NORM_EPS: ClassVar[float] = 1e-6
    ROPE_THETA: ClassVar[float] = 10_000.0

    depth: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int

    def __post_init__(self) -> None:
        _require_positive_integers(
            depth=self.depth,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size,
            num_position_embeddings=self.num_position_embeddings,
            out_hidden_size=self.out_hidden_size,
        )
        if self.hidden_size % self.num_heads:
            raise ValueError(
                f"vision hidden_size ({self.hidden_size}) must be divisible by num_heads "
                f"({self.num_heads})"
            )
        # The rotary embeds the patch row and column in one half of the head each.
        if self.head_dim % 4:
            raise ValueError(f"vision head dimension must be a multiple of 4: {self.head_dim}")
        if math.isqrt(self.num_position_embeddings) ** 2 != self.num_position_embeddings:
            raise ValueError(
                f"num_position_embeddings must be a square grid: {self.num_position_embeddings}"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        """Values in one flattened patch: channels, then frames, then rows and columns."""

        return self.in_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def position_grid_side(self) -> int:
        return math.isqrt(self.num_position_embeddings)

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> Qwen3_5VisionConfig:
        if config.hidden_act != "gelu_pytorch_tanh":
            raise ValueError(
                f"Qwen3.5 vision blocks use tanh-approximated GELU: {config.hidden_act!r}"
            )
        # A field of the published configs that transformers 5.16.1 never reads: Qwen3.5
        # feeds no intermediate vision layer to the text model.
        if list(getattr(config, "deepstack_visual_indexes", [])):
            raise ValueError("Qwen3.5 does not support deepstack_visual_indexes")
        return cls(
            depth=config.depth,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            num_position_embeddings=config.num_position_embeddings,
            out_hidden_size=config.out_hidden_size,
        )


@dataclass(frozen=True)
class Qwen3_5Config:
    """The parts of a Qwen3.5 checkpoint the engine runs: its text model and vision tower.

    ``image_token_id`` and ``video_token_id`` are the placeholders whose embeddings an
    image-text checkpoint replaces with the vision tower's tokens; a text-only export has
    neither, nor a vision tower, and embeds every id as text.
    """

    text: Qwen3_5TextConfig
    image_token_id: int | None = None
    video_token_id: int | None = None
    vision: Qwen3_5VisionConfig | None = None

    def __post_init__(self) -> None:
        for name in ("image_token_id", "video_token_id"):
            token_id = getattr(self, name)
            if token_id is not None and (type(token_id) is not int or token_id < 0):
                raise ValueError(f"{name} must be a non-negative integer: {token_id!r}")
        if self.vision is None:
            return
        if self.image_token_id is None:
            raise ValueError("a vision tower needs the image_token_id its tokens replace")
        if self.vision.out_hidden_size != self.text.hidden_size:
            raise ValueError(
                f"vision out_hidden_size ({self.vision.out_hidden_size}) must equal the text "
                f"hidden_size ({self.text.hidden_size})"
            )

    def engine(self, architectures: tuple[str, ...]) -> ModelConfig:
        text = self.text
        return ModelConfig(
            num_layers=len(text.layer_types),
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
            max_position=text.attention.max_position,
            tie_word_embeddings=text.tie_word_embeddings,
            architectures=architectures,
        )

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> Qwen3_5Config:
        """Read an image-text ``qwen3_5`` config or a text-only ``qwen3_5_text`` export."""

        match config.model_type:
            case "qwen3_5":
                text_config = config.text_config
                placeholders = (config.image_token_id, config.video_token_id)
                vision = Qwen3_5VisionConfig.from_hf(config.vision_config)
            case "qwen3_5_text":
                text_config = config
                placeholders = (None, None)
                vision = None
            case model_type:
                raise ValueError(f"expected a qwen3_5 or qwen3_5_text config, got {model_type!r}")
        # transformers ties the image-text model's head from the top-level field, which
        # text_config does not inherit.
        return cls(
            Qwen3_5TextConfig.from_hf(text_config, tie_word_embeddings=config.tie_word_embeddings),
            *placeholders,
            vision=vision,
        )


def _require_positive_integers(**values: object) -> None:
    for name, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer: {value!r}")


__all__ = [
    "LayerKind",
    "Qwen3_5AttentionConfig",
    "Qwen3_5Config",
    "Qwen3_5LinearAttentionConfig",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
]
