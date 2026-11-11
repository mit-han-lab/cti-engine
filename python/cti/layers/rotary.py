# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2024 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/layers/rotary.py`); see THIRD_PARTY_NOTICES.
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`); see THIRD_PARTY_NOTICES.
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/modeling_rope_utils.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Tuple

import torch

from .base import StateLessOP

if TYPE_CHECKING:
    from .site import Site


class RotaryEmbedding(StateLessOP):
    """NeoX rotary embedding of the leading ``rotary_dim`` channels of each head.

    The provider infers the rotated width from the ``cos | sin`` cache rows and leaves
    the remaining channels unchanged, so partial rotary (Qwen3.5 rotates 64 of 256)
    uses the same operation as full-head rotary. Training does the same from separate
    ``[max_position, rotary_dim / 2]`` caches.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        inv_freq: torch.Tensor,
        *,
        attention_factor: float = 1.0,
        training_enabled: bool = False,
    ) -> None:
        super().__init__()
        if head_dim not in (64, 128, 256, 512):
            raise ValueError(f"head_dim {head_dim} must be 64, 128, 256 or 512")
        if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_dim:
            raise ValueError(
                f"rotary_dim {rotary_dim} must be positive, even and at most head_dim {head_dim}"
            )
        self.head_dim = head_dim
        t = torch.arange(max_position, dtype=torch.float, device=inv_freq.device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # Hugging Face scales cos and sin, not the frequencies, by the attention factor.
        cos = freqs.cos() * attention_factor
        sin = freqs.sin() * attention_factor
        # FlashAttention training RoPE requires independent contiguous caches.
        # Only a training-enabled model keeps them; pure serving retains the
        # original FlashInfer concat cache and pays no persistent-memory tax.
        self._training_cos_cache = cos if training_enabled else None
        self._training_sin_cache = sin if training_enabled else None
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)

        from ctikernels.rotary.inference import apply_rotary_

        self._apply_rotary = apply_rotary_

    @classmethod
    def create(
        cls,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        rope_scaling: Dict[str, Any] | None = None,
        *,
        mrope_sections: tuple[int, ...] | None = None,
        site: Site,
    ) -> RotaryEmbedding:
        """The rotary of one spec, its tables on ``site.device``; ``mrope_sections``
        selects multi-axis positions. A model builds one and shares it across layers."""
        rope_type = "default" if rope_scaling is None else rope_scaling["rope_type"]
        # Each RoPE type follows transformers 5.16.1. default and yarn also keep its
        # operation order, so their CPU caches match bit for bit; llama3 evaluates the
        # same formula in another order and differs by fp32 rounding.
        pos_freqs = base ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float, device=site.device) / rotary_dim
        )
        match rope_type:
            case "default":
                inv_freq, attention_factor = 1.0 / pos_freqs, 1.0
            case "llama3":
                inv_freq = cls._llama3_inv_freq(1.0 / pos_freqs, rope_scaling)
                attention_factor = 1.0
            case "yarn":
                inv_freq, attention_factor = cls._yarn_frequencies(
                    pos_freqs, rope_scaling, rotary_dim, max_position, base
                )
            case _:
                raise ValueError(
                    f"unsupported rope_type {rope_type!r}: only default, llama3 and yarn "
                    "RoPE are implemented"
                )
        options = {"attention_factor": attention_factor, "training_enabled": site.training}
        if mrope_sections is None:
            return RotaryEmbedding(head_dim, rotary_dim, max_position, inv_freq, **options)
        return MultiAxisRotaryEmbedding(
            head_dim,
            rotary_dim,
            max_position,
            inv_freq,
            sections=tuple(mrope_sections),
            **options,
        )

    @staticmethod
    def _llama3_inv_freq(inv_freq: torch.Tensor, rope_scaling: Dict[str, Any]) -> torch.Tensor:
        scaling_factor: float = rope_scaling["factor"]
        low_freq_factor: float = rope_scaling["low_freq_factor"]
        high_freq_factor: float = rope_scaling["high_freq_factor"]
        original_max_position: int = rope_scaling["original_max_position_embeddings"]

        wave_len = 2 * math.pi / inv_freq
        if low_freq_factor == high_freq_factor:
            return torch.where(
                wave_len < original_max_position / high_freq_factor,
                inv_freq,
                inv_freq / scaling_factor,
            )

        delta = high_freq_factor - low_freq_factor
        smooth = (original_max_position / wave_len - low_freq_factor) / delta
        smooth = torch.clamp(smooth, 0, 1)
        factor = (1 - smooth) / scaling_factor + smooth
        return factor * inv_freq

    @staticmethod
    def _yarn_frequencies(
        pos_freqs: torch.Tensor,
        rope_scaling: Dict[str, Any],
        rotary_dim: int,
        max_position: int,
        base: float,
    ) -> tuple[torch.Tensor, float]:
        """Return YaRN inverse frequencies and the attention factor for cos and sin.

        Rewritten from `_compute_yarn_parameters` in transformers 5.16.1.
        """

        original_max_position: int = rope_scaling["original_max_position_embeddings"]
        factor = rope_scaling["factor"]
        if factor is None:
            factor = max_position / original_max_position

        def magnitude_scale(multiplier: float) -> float:
            return 1.0 if factor <= 1 else 0.1 * multiplier * math.log(factor) + 1.0

        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is None:
            mscale = rope_scaling.get("mscale")
            mscale_all_dim = rope_scaling.get("mscale_all_dim")
            # A lone or zero mscale is ignored; only a nonzero pair changes the default.
            if mscale and mscale_all_dim:
                attention_factor = magnitude_scale(mscale) / magnitude_scale(mscale_all_dim)
            else:
                attention_factor = magnitude_scale(1)

        def correction_dim(num_rotations: float) -> float:
            # The dimension whose wavelength fits num_rotations times into the
            # original context.
            return (
                rotary_dim
                * math.log(original_max_position / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        # Unset or zero betas fall back to the paper's 32 and 1.
        low = correction_dim(rope_scaling.get("beta_fast") or 32)
        high = correction_dim(rope_scaling.get("beta_slow") or 1)
        if rope_scaling.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, rotary_dim - 1)
        if low == high:
            high += 0.001  # keeps the ramp finite
        ramp = torch.arange(rotary_dim // 2, dtype=torch.float32, device=pos_freqs.device)
        ramp = torch.clamp((ramp - low) / (high - low), 0, 1)
        # Dimensions below the ramp keep their frequency; those above it are
        # interpolated, i.e. divided by factor.
        interpolation = 1.0 / (factor * pos_freqs)
        extrapolation = 1.0 / pos_freqs
        extrapolation_weight = 1 - ramp
        inv_freq = interpolation * (1 - extrapolation_weight) + extrapolation * extrapolation_weight
        return inv_freq, attention_factor

    @property
    def rotary_dim(self) -> int:
        """Number of leading head channels rotated by the cache's NeoX cos/sin rows."""
        return self._cos_sin_cache.shape[-1]

    @property
    def cos_sin_cache(self) -> torch.Tensor:
        """The static FP32 ``[max_position, rotary_dim]`` ``cos | sin`` table; read-only."""
        return self._cos_sin_cache

    @property
    def position_axes(self) -> tuple[int, ...] | None:
        """The position axis each frequency reads, or None: one axis (1-D positions)."""
        return None

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate by int64 ``positions``: ``[tokens]``, or ``[3, tokens]`` under
        ``position_axes``."""
        return self._apply_rotary(
            query,
            key,
            self._cos_sin_cache,
            positions,
            head_dim=self.head_dim,
            position_axes=self.position_axes,
        )

    def forward_key(self, positions: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Rotate ``key`` alone, in place; full-head rotary only."""

        from ctikernels.rotary.inference import apply_rotary_key_

        return apply_rotary_key_(key, self._cos_sin_cache, positions, head_dim=self.head_dim)

    def forward_training(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_sequence_length: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate padded ``[batch, sequence, heads, head_dim]`` Q/K, or packed
        ``[tokens, heads, head_dim]`` ones with ``cu_seqlens`` and ``max_sequence_length``."""
        from ctikernels.rotary import training as training_rotary

        cos, sin = self._training_cos_cache, self._training_sin_cache
        if cos is None or sin is None:
            raise RuntimeError("rotary embedding was not constructed for training")
        return training_rotary.apply_rotary(
            query,
            key,
            cos,
            sin,
            positions,
            cu_seqlens=cu_seqlens,
            max_sequence_length=max_sequence_length,
        )


class MultiAxisRotaryEmbedding(RotaryEmbedding):
    """Rotary whose frequencies read interleaved temporal, height and width positions.

    ``sections`` counts the frequencies of each axis (Qwen3.5 M-RoPE). Rewritten from
    transformers 5.16.1 ``Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope``:
    frequency ``j`` reads the height row when ``j % 3 == 1`` and ``j < 3 * sections[1]``,
    the width row when ``j % 3 == 2`` and ``j < 3 * sections[2]``, and the temporal row
    otherwise; ``position_axes`` is that map, which the kernels read. ``forward`` and the
    fused prologue take 1-D positions exactly like :class:`RotaryEmbedding`, or
    ``[3, tokens]`` positions, each frequency at its own axis's position; three equal rows
    reproduce the 1-D result. Training takes 1-D positions only: text, whose three rows are
    equal, until image training.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        inv_freq: torch.Tensor,
        *,
        sections: tuple[int, ...],
        attention_factor: float = 1.0,
        training_enabled: bool = False,
    ) -> None:
        super().__init__(
            head_dim,
            rotary_dim,
            max_position,
            inv_freq,
            attention_factor=attention_factor,
            training_enabled=training_enabled,
        )
        num_frequencies = rotary_dim // 2
        frequency_axis = [0] * num_frequencies
        for axis, size in enumerate(sections[1:], start=1):
            for frequency in range(axis, min(3 * size, num_frequencies), 3):
                frequency_axis[frequency] = axis
        if tuple(frequency_axis.count(axis) for axis in range(3)) != tuple(sections):
            raise ValueError(
                f"M-RoPE sections {sections} do not interleave {num_frequencies} frequencies"
            )
        self._position_axes = tuple(frequency_axis)

    @property
    def position_axes(self) -> tuple[int, ...]:
        return self._position_axes


class VisionRotaryEmbedding(StateLessOP):
    """NeoX rotary of vision patches by their (row, column) position in the image grid.

    For head dimension D, D/4 frequencies ``base ** (-4i / D)`` rotate by the patch row
    and the next D/4 by the patch column; the second half of the head repeats them.
    """

    def __init__(self, head_dim: int, base: float, *, site: Site) -> None:
        """The frequencies live on ``site.device``, as the text rotary's tables do."""
        super().__init__()
        if head_dim % 4:
            raise ValueError(f"vision rotary head_dim {head_dim} must be a multiple of 4")
        axis_dim = head_dim // 2
        arange = torch.arange(0, axis_dim, 2, dtype=torch.float, device=site.device)
        self._inv_freq = 1.0 / (base ** (arange / axis_dim))

        from ctikernels.rotary.vision import apply_rotary

        self._apply_rotary = apply_rotary

    def cos_sin(self, patch_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FP32 ``(cos, sin)`` ``[patches, head_dim]`` for (row, column) ``[patches, 2]``."""
        frequencies = (patch_positions.unsqueeze(-1) * self._inv_freq).flatten(1)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos(), angles.sin()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``[patches, heads, head_dim]`` query and key into fresh tensors."""
        return self._apply_rotary(query, key, cos, sin)


__all__ = ["MultiAxisRotaryEmbedding", "RotaryEmbedding", "VisionRotaryEmbedding"]
