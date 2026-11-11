# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/attention/base.py`, `python/minisgl/attention/utils.py`);
# see THIRD_PARTY_NOTICES.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, ClassVar, List, Protocol

if TYPE_CHECKING:
    import torch

    from cti.forward import ForwardInputs, ForwardRows
    from cti.utils import DeviceCapability


@dataclass
class BaseAttnMetadata(ABC):
    # The cache slot each query row's key and value go to.
    cache_write_indices: torch.Tensor


class SupportsKVPrologue(Protocol):
    """A backend that runs a layer's fused KV prologue (``SUPPORTS_KV_PROLOGUE``)."""

    def forward_prologue(
        self,
        prologue: Callable[..., torch.Tensor],
        layer_id: int,
        batch: ForwardInputs,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend the Q ``prologue(k_cache, v_cache, cache_indices)`` returns once it writes the
        step's K and V there, in layer ``layer_id``'s ``[slots, kv_heads, head_dim]`` caches.
        The result goes to, and is, ``out`` when given: a contiguous tensor of Q's shape, dtype
        and device that shares no storage with Q or the caches."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class BackendSetup:
    """What one engine's MHA backends are sized and built for: this rank's heads, the KV
    ``dtype``, the ``device`` and its ``capability``, the most running requests, and a
    tensor-core override (None: the backend's choice)."""

    head_dim: int
    num_qo_heads: int
    num_kv_heads: int
    dtype: torch.dtype
    device: torch.device
    capability: DeviceCapability | None
    max_running_req: int
    tensor_cores: bool | None


@dataclass(frozen=True, slots=True)
class PageConstraint:
    """The page sizes a backend takes: those in ``allowed`` (None: any), multiples of
    ``multiple``."""

    allowed: frozenset[int] | None = None
    multiple: int = 1

    def accepts(self, page_size: int) -> bool:
        return page_size % self.multiple == 0 and (
            self.allowed is None or page_size in self.allowed
        )


class BaseAttnBackend(ABC):
    # The page sizes the backend's provider takes; kvcache.choose_page_size decides.
    PAGES: ClassVar[PageConstraint] = PageConstraint()
    # Head dimensions without an accepted paged-attention operation test on this
    # backend. Resolution never selects the backend for them and rejects an explicit
    # request.
    UNQUALIFIED_HEAD_DIMS: ClassVar[frozenset[int]] = frozenset()
    # Qualified for head-sharded inputs: the setup's head counts are the rank's.
    SUPPORTS_TENSOR_PARALLEL: ClassVar[bool] = False
    # Implements SupportsKVPrologue.
    SUPPORTS_KV_PROLOGUE: ClassVar[bool] = False
    # Serves a pool's other batches beside fixed-query rounds over it.
    SUPPORTS_FIXED_QUERY: ClassVar[bool] = False

    @classmethod
    def validate_environment(
        cls,
        capability: DeviceCapability | None,
    ) -> tuple[str, ...]:
        if capability is None:
            return ("CUDA device is unavailable",)
        return ()

    @staticmethod
    def _check_provider(load: Callable[[], object], label: str) -> tuple[str, ...]:
        """Return why ``load`` cannot load the provider ``label`` names, if it cannot."""
        try:
            load()
        except (ImportError, OSError) as error:
            return (f"{label} could not be loaded: {type(error).__name__}: {error}",)
        return ()

    @classmethod
    def workspace_bytes(cls, setup: BackendSetup) -> int:
        """The device bytes construction for ``setup`` allocates."""
        return 0

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardInputs,
    ) -> torch.Tensor: ...

    def forward_into(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardInputs,
        out: torch.Tensor,
    ) -> None:
        out.copy_(self.forward(q, k, v, layer_id, batch))

    @abstractmethod
    def prepare_metadata(self, rows: ForwardRows) -> BaseAttnMetadata: ...

    def prepare_inputs(self, rows: ForwardRows) -> dict[str, BaseAttnMetadata]:
        """``rows``' metadata, prepared on the preparation stream, as the forward input it
        fills."""

        return {"attn_metadata": self.prepare_metadata(rows)}

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def capture_metadata(self, rows: ForwardRows) -> BaseAttnMetadata:
        """The metadata a decode graph of ``rows`` captures, over the static buffers."""

    def prepare_for_capture(self, rows: ForwardRows, inputs: ForwardInputs) -> ForwardInputs:
        return replace(inputs, attn_metadata=self.capture_metadata(rows))

    @abstractmethod
    def prepare_for_replay(self, batch: ForwardInputs, bs: int) -> None:
        """Copy ``batch``'s metadata, prepared for ``bs`` padded rows, into the captured
        graph's."""
