# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/attention/fi.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Dict, List, Literal

import torch

from cti.utils import align_ceil, div_ceil, init_logger
from ctikernels.attention.inference import (
    FlashInferPagedAttention,
    check_flashinfer_available,
    paged_kv_metadata,
    select_flashinfer_prefill_implementation,
)

from .base import BackendSetup, BaseAttnBackend, BaseAttnMetadata
from .paged import ragged_lengths, store_kv_pair, write_slots

if TYPE_CHECKING:
    from cti.forward import ForwardInputs, ForwardRows
    from cti.kvcache import PageTable
    from cti.utils import DeviceCapability

    from .paged import KVPages


logger = init_logger(__name__)

_INT_WORKSPACE_BYTES = 8 * 1024 * 1024


# TODO: drop this planner copy once FlashInferPagedAttention declares its workspace.
def _split_kv_scratch_bytes(
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_sms: int,
    max_running_req: int,
) -> int:
    """Bound the FP32 ``tmp_v``/``tmp_s`` rows one rank's split-KV plans carve out, per
    FlashInfer 0.7.0.post1 ``scheduler.cuh``/``utils.cuh`` on SM80+ (recheck on upgrade):
    eager FA2 plans take at most ``2 * SMs // kv_heads`` tiles of 128 rows (64 from head_dim
    256); decode graphs pad to ``ceil(batch * group / tile) + batch - 1`` tiles; CUDA-core
    decode takes one row per CTA, at most 16 CTAs per SM. Loose for SM90 FA3 prefill."""
    group = num_qo_heads // num_kv_heads
    largest_tile = 128 if head_dim < 256 else 64
    decode_tile = 16 if group <= 16 else 64 if group <= 64 else largest_tile
    graph_tiles = div_ceil(max_running_req * group, decode_tile) + max_running_req - 1
    rows = max(
        2 * num_sms // num_kv_heads * largest_tile,
        graph_tiles * decode_tile,
        16 * num_sms // num_kv_heads,
    )
    fp32_bytes = 4
    tmp_v = align_ceil(num_qo_heads * rows * head_dim * fp32_bytes, 16)
    return tmp_v + num_qo_heads * rows * fp32_bytes


# FlashInfer's non-quantized paged decode runs FA2 even with tensor-core decode selected.
_DECODE_IMPLEMENTATION = "fa2"


def _bucket(requests: int) -> int:
    """The power of two a batch of ``requests`` builds its KV indices at."""
    return 1 << (requests - 1).bit_length()


class KVIndices:
    """FlashInfer's page-size-1 KV metadata (each request's token slots as a CSR) of up to
    ``max_requests`` requests over ``page_table``, which ``paged_kv_metadata`` builds on the
    device into fixed buffers: decode graphs borrow them, and each plan reads the latest
    build."""

    def __init__(self, page_table: PageTable, max_requests: int) -> None:
        pages = page_table.pages
        self._pages, self._page_size = pages, page_table.page_size
        # A request's tokens fit its table row.
        self._max_kv_len = pages.shape[1] * self._page_size
        max_batch = _bucket(max_requests)

        def int32(count: int) -> torch.Tensor:
            return torch.zeros(count, dtype=torch.int32, device=pages.device)

        self.indptr, self.last_page_len = int32(max_batch + 1), int32(max_batch)
        self.indices = int32(max_requests * self._max_kv_len)
        # The operation compiles per batch size: build every bucket once before serving.
        for batch in (1 << shift for shift in range(max_batch.bit_length())):
            self.build(torch.zeros((2, batch), dtype=torch.int32, device=pages.device))

    def requests(self, rows: ForwardRows) -> torch.Tensor:
        """``rows``' table rows and KV lengths, int32 ``[2, B]`` on the device: the padded
        rows, then empty requests up to a power of two, so batches take few compiled sizes."""
        padding = (0,) * (_bucket(rows.padded_size) - rows.padded_size)
        host = torch.tensor(
            (rows.slots + padding, rows.sequence_lens + padding),
            dtype=torch.int32,
            pin_memory=True,
        )
        return host.to(self._pages.device, non_blocking=True)

    def build(self, requests: torch.Tensor) -> None:
        """Write ``requests``' metadata into the buffers, on the current stream."""
        batch = requests.shape[1]
        paged_kv_metadata(
            self._pages,
            requests[0],
            requests[1],
            page_size=self._page_size,
            granularity="token",
            max_kv_len=self._max_kv_len,
            kv_indptr=self.indptr[: batch + 1],
            kv_indices=self.indices,
            kv_last_page_len=self.last_page_len[:batch],
        )


@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    requests:           torch.Tensor  # on gpu, KVIndices.requests
    indices:            torch.Tensor  # on gpu, built at the plan
    last_page_len_cpu:  torch.Tensor  # on cpu
    seq_lens_cpu:       torch.Tensor  # on cpu
    wrapper:            FlashInferPagedAttention
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.requests.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )


class FlashInferBackend(BaseAttnBackend):
    SUPPORTS_TENSOR_PARALLEL = True
    SUPPORTS_KV_PROLOGUE = True
    SUPPORTS_FIXED_QUERY = True

    @classmethod
    def validate_environment(
        cls,
        capability: DeviceCapability | None,
    ) -> tuple[str, ...]:
        if reasons := super().validate_environment(capability):
            return reasons
        assert capability is not None
        if not capability.is_at_least(8):
            # _split_kv_scratch_bytes follows the SM80+ tile sizes; SM75 tiles are larger.
            return (
                f"FlashInfer attention does not support {capability}; "
                "it needs compute capability 8.0 or newer",
            )
        return cls._check_provider(check_flashinfer_available, "FlashInfer attention functions")

    @classmethod
    def workspace_bytes(cls, setup: BackendSetup) -> int:
        float_bytes = _split_kv_scratch_bytes(
            num_qo_heads=setup.num_qo_heads,
            num_kv_heads=setup.num_kv_heads,
            head_dim=setup.head_dim,
            num_sms=torch.cuda.get_device_properties(setup.device).multi_processor_count,
            max_running_req=setup.max_running_req,
        )
        return float_bytes + _INT_WORKSPACE_BYTES

    def __init__(self, setup: BackendSetup, *, kv: tuple[KVPages, ...], table: PageTable) -> None:
        # The provider picks its kernels by device at run time.
        self.kv = kv
        self.dtype = setup.dtype
        # FlashInfer consumes flat token indices irrespective of the physical cache
        # allocation page size: this backend expands the page table's page IDs, and
        # provider metadata always uses page 1.
        self.page_table = table
        self.device = setup.device
        self.qo_head_local = setup.num_qo_heads
        self.kv_head_local = setup.num_kv_heads
        self.head_dim = setup.head_dim
        self.prefill_implementation, self.use_tensor_cores, self.dispatch = (
            self._resolve_operations(
                num_qo_heads=setup.num_qo_heads,
                num_kv_heads=setup.num_kv_heads,
                device=self.device,
                dtype=self.dtype,
                tensor_cores=setup.tensor_cores,
            )
        )
        logger.info(
            "FlashInfer attention: prefill=%s, decode=%s, tensor_cores=%s, dispatch=%s",
            self.prefill_implementation,
            _DECODE_IMPLEMENTATION,
            self.use_tensor_cores,
            self.dispatch,
        )
        # The operation requires zeroed float scratch on first use. Eager and
        # graph wrappers borrow this allocation; do not clear it again on reuse.
        self.float_workspace_buffer = torch.zeros(
            self.workspace_bytes(setup) - _INT_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=self.device,
        )
        self.int_workspace_buffer = torch.empty(
            _INT_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=self.device,
        )
        self.prefill_wrapper = FlashInferPagedAttention(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            mode="prefill",
            backend=self.prefill_implementation,
            dispatch=self.dispatch,
        )
        self.decode_wrapper = FlashInferPagedAttention(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            mode="decode",
            use_tensor_cores=self.use_tensor_cores,
            backend=_DECODE_IMPLEMENTATION,
            dispatch=self.dispatch,
        )

        self.kv_indices = KVIndices(table, setup.max_running_req)
        # Page size 1 makes every last page length 1.
        self._ones_cpu = torch.ones(setup.max_running_req, dtype=torch.int32, pin_memory=True)
        # for cuda graph
        self.capture_bs: List[int] = []
        self.graph_wrappers: Dict[int, FlashInferPagedAttention] = {}
        self._graph_write_slots: torch.Tensor | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    @staticmethod
    def _resolve_operations(
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        device: torch.device,
        dtype: torch.dtype,
        tensor_cores: bool | None,
    ) -> tuple[Literal["fa2", "fa3"], bool, Literal["device", "provider"]]:
        """``(prefill implementation, tensor cores, dispatch)``: dispatch is ``"device"``
        (cti-kernels picks per device and batch; tensor cores is then the native default)
        unless an explicit tensor-core choice keeps the native ``"provider"`` routes."""

        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"FlashInfer paged attention requires FP16 or BF16 KV cache dtype, got {dtype}"
            )
        if tensor_cores is not None:
            logger.warning("Overriding FlashInfer tensor core usage to %s", tensor_cores)
        prefill_implementation = select_flashinfer_prefill_implementation(device, dtype)
        if prefill_implementation not in ("fa2", "fa3"):
            raise RuntimeError(
                "FlashInfer selected an unsupported paged-prefill implementation: "
                f"{prefill_implementation}"
            )
        if tensor_cores is None:
            return prefill_implementation, num_qo_heads // num_kv_heads >= 4, "device"
        return prefill_implementation, tensor_cores, "provider"

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        metadata.initialized = True
        self.kv_indices.build(metadata.requests)
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        self.last_event.synchronize()
        metadata.wrapper.plan(
            cu_seqlens_q=metadata.cu_seqlens_q_cpu,
            cu_seqlens_k=metadata.cu_seqlens_k_cpu,
            indices=metadata.indices,
            last_page_len=metadata.last_page_len_cpu,
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens=metadata.seq_lens_cpu,
            dtype=self.dtype,
        )
        self.last_event.record()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardInputs,
    ) -> torch.Tensor:
        # FlashInfer's prefill run() zero-fills an output it allocates (a memset node per layer
        # in a graph); it writes every row, so this backend passes an uninitialized one.
        out = q.new_empty(q.shape)
        self.forward_into(q, k, v, layer_id, batch, out)
        return out

    def forward_into(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: ForwardInputs,
        out: torch.Tensor,
    ) -> None:
        run, kv_cache = self._prepare_forward(layer_id, batch.attn_metadata)
        store_kv_pair(self.kv[layer_id], k, v, batch.attn_metadata.cache_write_indices)
        run(q=q, paged_kv_cache=kv_cache, out=out)

    def attend_into(
        self,
        q: torch.Tensor,
        layer_id: int,
        metadata: BaseAttnMetadata,
        out: torch.Tensor,
    ) -> None:
        """Attend ``q`` as ``metadata`` planned it, into ``out``, over layer ``layer_id``'s
        cache, which already holds the step's keys and values."""

        run, kv_cache = self._prepare_forward(layer_id, metadata)
        run(q=q, paged_kv_cache=kv_cache, out=out)

    def forward_prologue(
        self,
        prologue: Callable[..., torch.Tensor],
        layer_id: int,
        batch: ForwardInputs,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        run, (k_cache, v_cache) = self._prepare_forward(layer_id, batch.attn_metadata)
        # Without their page axis, the page-size-1 views are the prologue's token rows.
        q = prologue(
            k_cache.squeeze(1), v_cache.squeeze(1), batch.attn_metadata.cache_write_indices
        )
        out = q.new_empty(q.shape) if out is None else out
        return run(q=q, paged_kv_cache=(k_cache, v_cache), out=out)

    def _prepare_forward(
        self,
        layer_id: int,
        metadata: BaseAttnMetadata,
    ) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        k_cache, v_cache = self.kv[layer_id]
        # FlashInfer consumes flat token indices, so expose physical pages as page size 1.
        return metadata.wrapper.run, (
            k_cache.view(-1, 1, k_cache.shape[2], k_cache.shape[3]),
            v_cache.view(-1, 1, v_cache.shape[2], v_cache.shape[3]),
        )

    def prepare_metadata(self, rows: ForwardRows) -> FIMetadata:
        """A uniform decode of a captured size plans its graph's wrapper: the decode graph
        replays exactly those."""

        seq_len_cpu, cu_seqlens_k_cpu, cu_seqlens_q_cpu = ragged_lengths(rows)
        if rows.is_uniform_decode:
            wrapper = self.graph_wrappers.get(rows.padded_size, self.decode_wrapper)
        else:
            wrapper = self.prefill_wrapper
        return FIMetadata(
            cache_write_indices=write_slots(self.page_table, rows),
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            requests=self.kv_indices.requests(rows),
            indices=self.kv_indices.indices[: sum(rows.sequence_lens)],
            last_page_len_cpu=self._ones_cpu[: rows.padded_size],
            seq_lens_cpu=seq_len_cpu,
            wrapper=wrapper,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        del max_seq_len  # the KV indices cover the page table
        assert self._graph_write_slots is None, "Capture already initialized."
        self._graph_write_slots = torch.zeros(max(bs_list), dtype=torch.int32, device=self.device)
        self.capture_bs = sorted(bs_list)

    def capture_metadata(self, rows: ForwardRows) -> FIMetadata:
        bs = rows.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers
        assert self._graph_write_slots is not None
        kv_indices = self.kv_indices
        wrapper = FlashInferPagedAttention(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            mode="decode",
            use_cuda_graph=True,
            use_tensor_cores=self.use_tensor_cores,
            paged_kv_indptr_buffer=kv_indices.indptr[: bs + 1],
            paged_kv_indices_buffer=kv_indices.indices,
            paged_kv_last_page_len_buffer=kv_indices.last_page_len[:bs],
            backend=_DECODE_IMPLEMENTATION,
            dispatch=self.dispatch,
        )
        self.graph_wrappers[bs] = wrapper
        metadata = replace(
            self.prepare_metadata(rows), cache_write_indices=self._graph_write_slots[:bs]
        )
        self._initialize_metadata_once(metadata)
        return metadata

    def prepare_for_replay(self, batch: ForwardInputs, bs: int) -> None:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self._graph_write_slots is not None and metadata.wrapper is self.graph_wrappers[bs]
        self._graph_write_slots[:bs].copy_(metadata.cache_write_indices)
        self._initialize_metadata_once(metadata)
