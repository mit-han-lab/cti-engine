# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/attention/__init__.py`, `python/minisgl/attention/base.py`);
# see THIRD_PARTY_NOTICES.

"""The MHA attention family: paged multi-head attention over K/V row pairs, served by the
FlashAttention, FlashInfer or TRT-LLM backend (or a ragged and decode pair of them)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, List

from cti.utils import DeviceCapability, init_logger

from .base import BackendSetup, BaseAttnBackend, BaseAttnMetadata, PageConstraint
from .paged import KVPair, kv_pages
from .requirement import kv_head_window, qo_head_window

if TYPE_CHECKING:
    import torch

    from cti.forward import ForwardInputs, ForwardRows
    from cti.kvcache import GroupRole, PageGroup, PageTable, RowLayout, SlotKey

    from .family import AttentionRequest
    from .requirement import AttentionRequirement

logger = init_logger(__name__)


class AttentionBackend(StrEnum):
    """A paged attention backend by the name a request gives it. Its class, and the kernels
    behind it, load on first use."""

    @staticmethod
    def _flash_attention() -> type[BaseAttnBackend]:
        from .flash_attention import FlashAttentionBackend

        return FlashAttentionBackend

    @staticmethod
    def _flashinfer() -> type[BaseAttnBackend]:
        from .flashinfer import FlashInferBackend

        return FlashInferBackend

    @staticmethod
    def _tensorrt_llm() -> type[BaseAttnBackend]:
        from .tensorrt_llm import TensorRTLLMBackend

        return TensorRTLLMBackend

    FLASH_ATTENTION = "fa", _flash_attention
    FLASHINFER = "fi", _flashinfer
    TENSORRT_LLM = "trtllm", _tensorrt_llm

    _load: Callable[[], type[BaseAttnBackend]]

    def __new__(cls, name: str, load: Callable[[], type[BaseAttnBackend]]) -> AttentionBackend:
        backend = str.__new__(cls, name)
        backend._value_ = name
        backend._load = load
        return backend

    @property
    def backend_class(self) -> type[BaseAttnBackend]:
        return self._load()


class HybridBackend(BaseAttnBackend):
    """A ragged and a decode backend over one KV group: batches with prompt rows go to the
    ragged one; decodes and their graphs to the decode one."""

    def __init__(
        self,
        ragged_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.ragged_backend = ragged_backend
        self.decode_backend = decode_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardInputs,
    ) -> torch.Tensor:
        backend = self.ragged_backend if batch.ragged else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def forward_into(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardInputs,
        out: torch.Tensor,
    ) -> None:
        backend = self.ragged_backend if batch.ragged else self.decode_backend
        backend.forward_into(q, k, v, layer_id, batch, out)

    def prepare_metadata(self, rows: ForwardRows) -> BaseAttnMetadata:
        backend = self.ragged_backend if rows.uses_ragged_attention else self.decode_backend
        return backend.prepare_metadata(rows)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def capture_metadata(self, rows: ForwardRows) -> BaseAttnMetadata:
        return self.decode_backend.capture_metadata(rows)

    def prepare_for_replay(self, batch: ForwardInputs, bs: int) -> None:
        self.decode_backend.prepare_for_replay(batch, bs)


@dataclass(frozen=True)
class ResolvedMha:
    """The MHA family resolved for one engine: the backends it selected and why, and this
    rank's heads of the target's attention, which its backend serves."""

    ragged_backend: AttentionBackend
    decode_backend: AttentionBackend
    reason: str
    # Each paged slot's row layout on this rank, by role and requirement name.
    layouts: Mapping[SlotKey, RowLayout]
    head_dim: int
    num_qo_heads: int
    num_kv_heads: int
    dtype: torch.dtype
    capability: DeviceCapability | None
    # Overrides a backend's own tensor-core choice; None lets it choose.
    tensor_cores: bool | None

    @property
    def selected(self) -> str:
        if not self.is_hybrid:
            return self.ragged_backend
        return f"{self.ragged_backend},{self.decode_backend}"

    @property
    def is_hybrid(self) -> bool:
        return self.ragged_backend != self.decode_backend

    @property
    def pages(self) -> tuple[PageConstraint, ...]:
        """The page sizes each selected backend takes, for kvcache.choose_page_size."""
        return tuple(backend.backend_class.PAGES for backend in self._pair)

    def workspace_bytes(self, *, device: torch.device, max_running_req: int) -> int:
        """The device scratch that building the selected backends allocates."""

        setup = self._setup(device, max_running_req, self.capability)
        return sum(backend.backend_class.workspace_bytes(setup) for backend in self._pair)

    def build(
        self,
        groups: Sequence[PageGroup],
        page_table: PageTable,
        *,
        max_running_req: int,
        capability: DeviceCapability | None,
    ) -> BaseAttnBackend:
        """The runtime backend serving the target's one KV group of ``groups`` on
        ``page_table`` with this rank's heads."""

        if len(groups) != 1:
            raise ValueError(f"MHA serves one target KV group, got {len(groups)}")
        (group,) = groups
        setup = self._setup(group.layers[0].device, max_running_req, capability)
        kv = kv_pages(group)

        def construct(backend: AttentionBackend) -> BaseAttnBackend:
            return backend.backend_class(setup, kv=kv, table=page_table)

        if not self.is_hybrid:
            return construct(self.ragged_backend)
        logger.info(
            "Using hybrid attention backend: ragged=%s, decode=%s",
            self.ragged_backend,
            self.decode_backend,
        )
        return HybridBackend(
            ragged_backend=construct(self.ragged_backend),
            decode_backend=construct(self.decode_backend),
        )

    def _setup(
        self, device: torch.device, max_running_req: int, capability: DeviceCapability | None
    ) -> BackendSetup:
        # Resolution admits other backends only on one rank, where all heads are local.
        return BackendSetup(
            head_dim=self.head_dim,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            dtype=self.dtype,
            device=device,
            capability=capability,
            max_running_req=max_running_req,
            tensor_cores=self.tensor_cores,
        )

    @property
    def _pair(self) -> tuple[AttentionBackend, ...]:
        """The distinct selected backends."""
        return _distinct((self.ragged_backend, self.decode_backend))


def _distinct(pair: tuple[AttentionBackend, AttentionBackend]) -> tuple[AttentionBackend, ...]:
    return tuple(dict.fromkeys(pair))


def _parse(backend: str) -> tuple[AttentionBackend, AttentionBackend]:
    """The ``ragged[,decode]`` pair a request names; one name serves both."""
    names = backend.split(",")
    if len(names) > 2:
        raise ValueError("at most one ragged and one decode attention backend are allowed")
    supported = set(AttentionBackend)
    if unsupported := sorted({name for name in names if name not in supported}):
        raise ValueError(f"unsupported attention backend: {', '.join(unsupported)}")
    return AttentionBackend(names[0]), AttentionBackend(names[-1])


def _head_dims(requirements: Sequence[AttentionRequirement]) -> tuple[int, ...]:
    """The head dimensions ``requirements`` ask for; MHA holds one token per cache entry."""

    for requirement in requirements:
        if requirement.tokens_per_entry != 1:
            raise ValueError(f"MHA holds one token per cache entry, not {requirement}")
    return tuple(dict.fromkeys(requirement.size("head_dim") for requirement in requirements))


def _target_heads(requirements: Sequence[AttentionRequirement], ranks: int) -> tuple[int, ...]:
    """``(head_dim, qo_heads, kv_heads)`` of one of ``ranks`` ranks over the target's slots,
    which the paged backend serves alike."""

    if not requirements:
        raise ValueError("the target role has no attention requirement")
    heads = {
        (
            requirement.size("head_dim"),
            qo_head_window(requirement.size("qo_heads"), 0, ranks)[1],
            kv_head_window(requirement.size("kv_heads"), 0, ranks)[1],
        )
        for requirement in requirements
    }
    if len(heads) != 1:
        raise ValueError(f"the target's attention needs one head geometry, got {sorted(heads)}")
    return heads.pop()


def _layouts(
    requirements: Mapping[GroupRole, Sequence[AttentionRequirement]],
    *,
    ranks: int,
    dtype: torch.dtype,
) -> dict[SlotKey, RowLayout]:
    """The MHA row layout of each role's slots on one of ``ranks`` ranks; a slot's
    requirements share one."""

    layouts: dict[SlotKey, RowLayout] = {}
    for role, role_requirements in requirements.items():
        for requirement in role_requirements:
            _, num_kv_heads = kv_head_window(requirement.size("kv_heads"), 0, ranks)
            layout = KVPair(num_kv_heads, requirement.size("head_dim"), dtype)
            if layouts.setdefault((role, requirement.slot), layout) != layout:
                raise ValueError(f"{role} slot {requirement.slot!r} needs two row layouts")
    return layouts


def _invalid_reasons(
    pair: tuple[AttentionBackend, AttentionBackend],
    capability: DeviceCapability | None,
    head_dims: tuple[int, ...],
    *,
    ranks: int,
    rounds: bool,
) -> tuple[str, ...]:
    backends = _distinct(pair)
    # Reject before any provider environment check can load a library.
    if ranks > 1:
        unqualified = [b for b in backends if not b.backend_class.SUPPORTS_TENSOR_PARALLEL]
        if unqualified:
            return (f"{', '.join(unqualified)} is not qualified for tensor-parallel execution",)
    if rounds:
        unqualified = [b for b in backends if not b.backend_class.SUPPORTS_FIXED_QUERY]
        if unqualified:
            return (f"{', '.join(unqualified)} does not run beside fixed-query rounds",)
    reasons: list[str] = []
    for backend in backends:
        backend_class = backend.backend_class
        reasons.extend(backend_class.validate_environment(capability))
        reasons.extend(
            f"{backend} has no accepted operation test for head_dim {head_dim}"
            for head_dim in head_dims
            if head_dim in backend_class.UNQUALIFIED_HEAD_DIMS
        )
    return tuple(reasons)


def _page_reasons(
    pair: tuple[AttentionBackend, AttentionBackend], page_size: int | None
) -> tuple[str, ...]:
    """Why ``pair`` cannot run on pages of an explicit ``page_size``."""
    if page_size is None:
        return ()
    refused = [b for b in _distinct(pair) if not b.backend_class.PAGES.accepts(page_size)]
    if refused:
        return (f"{', '.join(refused)} does not take page size {page_size}",)
    return ()


def _auto_candidates(capability: DeviceCapability | None) -> tuple[str, ...]:
    if capability is not None and capability.is_family(10):
        return "trtllm", "fi", "fa"
    if capability is not None and capability.is_family(9):
        return "fa,fi", "fi", "fa"
    return ("fi",)


_SELECTION_REASONS = MappingProxyType(
    {
        "trtllm": "supports FlashInfer TRTLLM attention",
        "fa,fi": "uses SGLang FlashAttention ragged and FlashInfer decode",
        "fa": "uses SGLang FlashAttention",
        "fi": "uses the portable FlashInfer backend",
    }
)


def _selection_reason(
    selected: str,
    capability: DeviceCapability,
    rejected: list[str],
) -> str:
    reason = f"{capability} {_SELECTION_REASONS[selected]}"
    if rejected:
        reason += f"; rejected {'; '.join(rejected)}"
    return reason


def resolve(
    requirements: Mapping[GroupRole, Sequence[AttentionRequirement]], request: AttentionRequest
) -> ResolvedMha:
    """Resolve one validated MHA plan that serves every role's requirements, the model's
    and its speculative method's, on each of the ranks, and with the request's rounds the
    fixed-query rounds beside it. An explicit page size must suit the selected backends:
    auto passes over a backend that does not take it, and an explicit backend that does not
    is refused."""
    requested, page_size, capability = request.backend, request.page_size, request.capability
    rounds, dtype, ranks = request.rounds, request.dtype, request.tensor_parallel_size
    pair = None if requested == "auto" else _parse(requested)
    head_dims = _head_dims([r for role in requirements.values() for r in role])
    layouts = _layouts(requirements, ranks=ranks, dtype=dtype)
    head_dim, num_qo_heads, num_kv_heads = _target_heads(
        requirements.get(request.target, ()), ranks
    )
    invalid = partial(_invalid_reasons, ranks=ranks, rounds=rounds is not None)

    if pair is not None:
        reasons = _page_reasons(pair, page_size) or invalid(pair, capability, head_dims)
        if reasons:
            raise ValueError(
                f"Attention backend '{requested}' is unavailable: {'; '.join(reasons)}"
            )
        reason = "explicit attention backend"
    else:
        rejected: list[str] = []
        for candidate in _auto_candidates(capability):
            pair = _parse(candidate)
            reasons = _page_reasons(pair, page_size) or invalid(pair, capability, head_dims)
            if not reasons:
                assert capability is not None
                reason = _selection_reason(candidate, capability, rejected)
                break
            rejected.append(f"{candidate}: {', '.join(reasons)}")
        else:
            raise ValueError(f"No attention backend is available: {'; '.join(rejected)}")

    return ResolvedMha(
        ragged_backend=pair[0],
        decode_backend=pair[1],
        reason=reason,
        layouts=layouts,
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        capability=capability,
        tensor_cores=request.tensor_cores,
    )


__all__ = ["AttentionBackend", "HybridBackend", "ResolvedMha", "resolve"]
