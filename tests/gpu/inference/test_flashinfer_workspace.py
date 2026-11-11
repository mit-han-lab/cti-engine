"""FlashInfer plans that overflow a fixed 128 MiB float workspace fit the sized one and
match FP32 attention."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.checkpoints()

if TYPE_CHECKING:
    import torch

    from cti.attention.flashinfer import FIMetadata, FlashInferBackend
    from cti.kvcache import PageGroup

RECEIPT_PREFIX = "CTI_FLASHINFER_WORKSPACE="
# Qwen3-32B attention at TP1.
NUM_QO_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_RUNNING_REQ = 256
DECODE_LEN = 32
FIXED_FLOAT_WORKSPACE_BYTES = 128 * 1024 * 1024


def _reference(
    q: torch.Tensor,
    group: PageGroup,
    indices: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    """Causal GQA attention in FP32 over the cached tokens at ``indices``."""
    import torch

    heads_per_kv = NUM_QO_HEADS // NUM_KV_HEADS
    keys, values = (
        cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[indices.long()]
        .float()
        .repeat_interleave(heads_per_kv, dim=1)
        for cache in group.layers[0]
    )
    scores = torch.einsum("qhd,khd->hqk", q.float(), keys) * HEAD_DIM**-0.5
    query_positions = prefix_len + torch.arange(q.shape[0], device=q.device)
    key_positions = torch.arange(len(indices), device=q.device)
    scores.masked_fill_(key_positions[None, :] > query_positions[:, None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), values)


def _relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    return ((output.float() - reference).norm() / reference.norm()).item()


def _random_qkv(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch

    # As the attention layer hands them over: q per head, k and v as flat [tokens, kv_dim]
    # columns of the fused projection.
    q = torch.randn(num_tokens, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn(
        2, num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    ).unbind()
    return q, k, v


def _fixed_plan_overflows(
    backend: FlashInferBackend,
    metadata: FIMetadata,
    /,
    **operation: object,
) -> bool:
    """Whether planning ``metadata`` on a fixed 128 MiB float workspace overflows."""
    import torch

    from ctikernels.attention.inference import FlashInferPagedAttention

    fixed = FlashInferPagedAttention(
        torch.zeros(FIXED_FLOAT_WORKSPACE_BYTES, dtype=torch.uint8, device=backend.device),
        torch.empty_like(backend.int_workspace_buffer),
        **operation,
    )
    try:
        fixed.plan(
            cu_seqlens_q=metadata.cu_seqlens_q_cpu,
            cu_seqlens_k=metadata.cu_seqlens_k_cpu,
            indices=metadata.indices,
            last_page_len=metadata.last_page_len_cpu,
            num_qo_heads=backend.qo_head_local,
            num_kv_heads=backend.kv_head_local,
            head_dim=backend.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens=metadata.seq_lens_cpu,
            dtype=backend.dtype,
        )
    except RuntimeError as error:
        if "Buffer overflow" not in str(error):
            raise
        return True
    torch.cuda.synchronize()
    return False


def _run_cases() -> dict[str, object]:
    import torch

    from cti.attention.base import BackendSetup
    from cti.attention.flashinfer import FlashInferBackend
    from cti.attention.paged import kv_pages
    from cti.forward import ForwardInputs
    from tests.support.forward_rows import decode_entry, extend_entry, forward_rows
    from tests.support.paged_attention import kv_group

    torch.manual_seed(20261003)
    device = torch.device("cuda")
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    split_tiles = 2 * num_sms // NUM_KV_HEADS
    # (new tokens, KV tokens): the reported crash, then one full 128-row query tile
    # over the most KV chunks this GPU splits into (the largest prefill split).
    prefill_cases = {"prefill_reported": (128, 2176), "prefill_worst": (16, split_tiles * 128)}
    prefill_len = max(kv_len for _, kv_len in prefill_cases.values())
    group = kv_group(
        1,
        NUM_KV_HEADS,
        HEAD_DIM,
        prefill_len + MAX_RUNNING_REQ * DECODE_LEN,
        dtype=torch.bfloat16,
        device=device,
    )
    group.layers[0].normal_()
    # Decode requests own rows [0, MAX_RUNNING_REQ); the prefill request owns the last row.
    cache_index_table = torch.zeros(
        (MAX_RUNNING_REQ + 1, prefill_len), dtype=torch.int32, device=device
    )
    cache_index_table[:MAX_RUNNING_REQ, :DECODE_LEN] = prefill_len + torch.arange(
        MAX_RUNNING_REQ * DECODE_LEN, dtype=torch.int32, device=device
    ).view(MAX_RUNNING_REQ, DECODE_LEN)
    cache_index_table[MAX_RUNNING_REQ] = torch.arange(prefill_len, dtype=torch.int32, device=device)
    setup = BackendSetup(
        head_dim=HEAD_DIM,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        dtype=torch.bfloat16,
        device=device,
        capability=None,
        max_running_req=MAX_RUNNING_REQ,
        tensor_cores=None,
    )
    backend = FlashInferBackend(
        setup, kv=kv_pages(group), table=SimpleNamespace(page_size=1, pages=cache_index_table)
    )
    receipt: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "float_workspace_bytes": backend.float_workspace_buffer.numel(),
        "workspace_bytes": backend.float_workspace_buffer.numel()
        + backend.int_workspace_buffer.numel(),
        "reserved_workspace_bytes": FlashInferBackend.workspace_bytes(setup),
        "use_tensor_cores": backend.use_tensor_cores,
    }

    for name, (new_tokens, kv_len) in prefill_cases.items():
        prefix_len = kv_len - new_tokens
        rows = forward_rows(extend_entry(MAX_RUNNING_REQ, prefix_len, kv_len))
        q, k, v = _random_qkv(new_tokens)
        metadata = backend.prepare_metadata(rows)
        batch = ForwardInputs(None, None, metadata, rows.size, rows.uses_ragged_attention)
        output = backend.forward(q, k, v, 0, batch)
        reference = _reference(q, group, cache_index_table[MAX_RUNNING_REQ, :kv_len], prefix_len)
        receipt[name] = {
            # From 128 SMs on, both shapes split into at least 32 tiles of 4 MiB.
            "expected_fixed_overflow": backend.prefill_implementation == "fa2" and num_sms >= 128,
            "fixed_overflow": _fixed_plan_overflows(
                backend,
                batch.attn_metadata,
                mode="prefill",
                backend=backend.prefill_implementation,
            ),
            "relative_error": _relative_error(output, reference),
        }

    backend.init_capture_graph(max_seq_len=prefill_len, bs_list=[MAX_RUNNING_REQ])
    entries = tuple(decode_entry(row, DECODE_LEN - 1) for row in range(MAX_RUNNING_REQ))
    rows = forward_rows(*entries)
    q, k, v = _random_qkv(MAX_RUNNING_REQ)
    metadata = backend.capture_metadata(rows)
    batch = ForwardInputs(None, None, metadata, rows.size, ragged=False)
    # The graph writes through the capture's static slots; the decode rows' own go there.
    metadata.cache_write_indices.copy_(cache_index_table[:MAX_RUNNING_REQ, DECODE_LEN - 1])
    eager = backend.forward(q, k, v, 0, batch).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend.forward(q, k, v, 0, batch)
    graph.replay()
    torch.cuda.synchronize()
    reference = torch.cat(
        [
            _reference(q[row, None], group, indices[:DECODE_LEN], DECODE_LEN - 1)
            for row, indices in enumerate(cache_index_table[:MAX_RUNNING_REQ])
        ]
    )
    receipt["decode_graph"] = {
        # 256 requests pad to at least 383 tiles of 16 rows: 202 MB on every GPU.
        "expected_fixed_overflow": backend.use_tensor_cores,
        "fixed_overflow": _fixed_plan_overflows(
            backend,
            batch.attn_metadata,
            mode="decode",
            backend="fa2",
            use_tensor_cores=backend.use_tensor_cores,
            use_cuda_graph=True,
            paged_kv_indptr_buffer=torch.empty_like(
                backend.kv_indices.indptr[: MAX_RUNNING_REQ + 1]
            ),
            paged_kv_indices_buffer=torch.empty_like(backend.kv_indices.indices),
            paged_kv_last_page_len_buffer=torch.empty_like(
                backend.kv_indices.last_page_len[:MAX_RUNNING_REQ]
            ),
        ),
        "relative_error": _relative_error(eager, reference),
        "replay_matches_eager": bool(torch.equal(captured, eager)),
    }
    return receipt


def _run_child() -> None:
    receipt = _run_cases()
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def test_flashinfer_workspace_covers_qwen3_32b_overflow_shapes(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "flashinfer-workspace")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_flashinfer_workspace"),
        log_root / "flashinfer-workspace.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )
    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert_cuda_device_metadata(receipt)

    assert receipt["workspace_bytes"] == receipt["reserved_workspace_bytes"]
    assert receipt["float_workspace_bytes"] > FIXED_FLOAT_WORKSPACE_BYTES
    for name in ("prefill_reported", "prefill_worst", "decode_graph"):
        case = receipt[name]
        assert case["fixed_overflow"] is case["expected_fixed_overflow"], name
        assert case["relative_error"] < 2e-2, name
    # The decode-graph overflow reproduces on every GPU with tensor-core decode.
    assert receipt["use_tensor_cores"] is True
    assert receipt["decode_graph"]["replay_matches_eager"] is True


if __name__ == "__main__":
    _run_child()
