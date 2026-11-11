"""What a model's attention needs of an attention family, given before any model exists."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class AttentionRequirement:
    """What one attention slot needs of an attention family, which the model class gives at
    resolution, before construction: the ``family``, the ``slot`` its layers' state names,
    the tokens one cache entry holds, how many trailing keys a query reads (None: all), and
    the global logical ``geometry``, ``(qo_heads, kv_heads, head_dim)`` for ``mha``."""

    family: str
    slot: str
    tokens_per_entry: int = 1
    window: int | None = None
    geometry: tuple[tuple[str, int], ...]

    @classmethod
    def mha(
        cls, qo_heads: int, kv_heads: int, head_dim: int, *, window: int | None = None
    ) -> AttentionRequirement:
        """Multi-head attention's KV slot over all of a model's query and KV heads."""

        geometry = (("qo_heads", qo_heads), ("kv_heads", kv_heads), ("head_dim", head_dim))
        return cls(family="mha", slot="kv", window=window, geometry=geometry)

    def size(self, name: str) -> int:
        """The geometry's ``name`` extent."""

        return dict(self.geometry)[name]


def qo_head_window(num_qo_heads: int, rank: int, size: int) -> tuple[int, int]:
    """Return (first query head, local query heads) of ``rank`` among ``size`` ranks, which
    split the query heads evenly."""
    if num_qo_heads % size:
        raise ValueError(f"query heads ({num_qo_heads}) must be divisible by {size} ranks")
    local = num_qo_heads // size
    return rank * local, local


def kv_head_window(num_kv_heads: int, rank: int, size: int) -> tuple[int, int]:
    """Return (first KV head, local KV heads) of ``rank`` among ``size`` ranks.

    Ranks split the KV heads evenly; with fewer heads than ranks, consecutive ranks share one
    replicated head.
    """
    if num_kv_heads >= size:
        if num_kv_heads % size:
            raise ValueError(f"KV heads ({num_kv_heads}) must be divisible by {size} ranks")
        local = num_kv_heads // size
        return rank * local, local
    if size % num_kv_heads:
        raise ValueError(f"{size} ranks must be divisible by the KV heads ({num_kv_heads})")
    return rank * num_kv_heads // size, 1
