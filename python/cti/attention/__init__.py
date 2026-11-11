from __future__ import annotations

from typing import TYPE_CHECKING

from cti.utils import lazy_getattr

from .base import BaseAttnBackend, BaseAttnMetadata
from .family import AttentionRequest, ResolvedFamily, resolve_attention_config
from .requirement import AttentionRequirement

if TYPE_CHECKING:
    from .fixed_query import (
        FixedQueryGeometry,
        FixedQueryShape,
        MixedQuery,
        SupportsFixedQuery,
        create_fixed_query_backends,
        fixed_query_bytes,
    )

# Selection and contract imports must not load the fixed-query kernels (PEP 562). The
# runner builds the speculative rounds' SupportsFixedQuery backends from these.
__getattr__ = lazy_getattr(
    __name__,
    {
        "FixedQueryGeometry": ".fixed_query",
        "FixedQueryShape": ".fixed_query",
        "MixedQuery": ".fixed_query",
        "SupportsFixedQuery": ".fixed_query",
        "create_fixed_query_backends": ".fixed_query",
        "fixed_query_bytes": ".fixed_query",
    },
    globals(),
)

__all__ = [
    "AttentionRequest",
    "AttentionRequirement",
    "BaseAttnBackend",
    "BaseAttnMetadata",
    "FixedQueryGeometry",
    "FixedQueryShape",
    "MixedQuery",
    "ResolvedFamily",
    "SupportsFixedQuery",
    "create_fixed_query_backends",
    "fixed_query_bytes",
    "resolve_attention_config",
]
