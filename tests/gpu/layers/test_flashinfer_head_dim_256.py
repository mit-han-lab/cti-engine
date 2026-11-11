"""FlashInfer paged attention at head_dim 256 (Qwen3.5-9B 16x4, 27B 24x4 heads) against FP32 SDPA
over the pool's rows, within the head_dim 72 bound of ``test_layer_operations.py``; each test
asserts on one child's receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import PYTHON_ROOT, REPO_ROOT
from tests.support.qwen3_5_gpu import write_receipt

pytestmark = pytest.mark.checkpoints()

if TYPE_CHECKING:
    import torch

    from cti.attention.flashinfer import FlashInferBackend
    from cti.kvcache import PageGroup

RECEIPT_PREFIX = "CTI_FLASHINFER_HEAD_DIM_256="
HEAD_DIM = 256
GEOMETRIES = ("16x4", "24x4")
# (cached prefix, new tokens) per request of each prefill batch. A plan that outgrows
# FlashInfer's float workspace fails the batch.
PREFILL_BATCHES = {
    "fresh": ((0, 1), (0, 7), (0, 64), (0, 300), (0, 1025), (0, 4097)),
    "chunk-8192-onto-8192": ((8192, 8192),),
    "mixed-prefixes": ((100, 1), (0, 33), (4096, 513), (17, 64)),
}
DECODE_BATCH_SIZES = (1, 3, 8, 32, 128, 256)
NUM_PAGES = 600_000
TABLE_WIDTH = 16384
RTOL = ATOL = 0.02
# Query rows of the FP32 reference per block, to bound its score matrix.
_REFERENCE_BLOCK_ROWS = 512
_CHILD_TIMEOUT_SECONDS = 900


def _decode_requests(batch_size: int) -> tuple[tuple[int, int], ...]:
    return tuple((1 + (977 * request) % 4096, 1) for request in range(batch_size))


def _model_config(geometry: str, workdir: Path) -> Any:
    from cti.models.qwen3_5.config import Qwen3_5Config
    from tests.support.qwen3_5 import QWEN3_5_9B_CONFIG, QWEN3_8_27B_CONFIG, load_hf_config

    config = {"16x4": QWEN3_5_9B_CONFIG, "24x4": QWEN3_8_27B_CONFIG}[geometry]
    # The full-attention layers' heads.
    return Qwen3_5Config.from_hf(load_hf_config(config, workdir / geometry)).text.attention


def _reference(
    q: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    prefix: int,
) -> torch.Tensor:
    """FP32 causal attention of ``q [L, Hq, D]`` over ``prefix + L`` cached rows."""

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    group = q.shape[1] // k_rows.shape[1]
    keys = k_rows.float().repeat_interleave(group, dim=1).transpose(0, 1)
    values = v_rows.float().repeat_interleave(group, dim=1).transpose(0, 1)
    key_positions = torch.arange(len(k_rows), device=q.device)
    blocks = []
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, len(q), _REFERENCE_BLOCK_ROWS):
            rows = q[start : start + _REFERENCE_BLOCK_ROWS].float().transpose(0, 1)
            query_positions = prefix + start + torch.arange(rows.shape[1], device=q.device)
            visible = key_positions <= query_positions[:, None]
            output = F.scaled_dot_product_attention(rows, keys, values, attn_mask=visible)
            blocks.append(output.transpose(0, 1))
    return torch.cat(blocks)


def _run_batch(
    backend: FlashInferBackend,
    group: PageGroup,
    cache_index_table: torch.Tensor,
    pages: torch.Tensor,
    requests: tuple[tuple[int, int], ...],
    *,
    decode: bool,
    seed: int,
) -> dict[str, Any]:
    """One batch through the backend against the FP32 reference; prefixes are random."""

    import torch

    from cti.forward import ForwardInputs
    from tests.support.forward_rows import Entry, forward_rows

    device = cache_index_table.device
    generator = torch.Generator(device=device).manual_seed(seed)
    k_pool, v_pool = (cache.flatten(0, 1) for cache in group.layers[0])
    num_kv_heads = k_pool.shape[1]
    num_qo_heads = backend.qo_head_local
    entries, rows, offset = [], [], 0
    for request, (prefix, extend) in enumerate(requests):
        length = prefix + extend
        cache_index_table[request, :length] = pages[offset : offset + length]
        offset += length
        rows.append(cache_index_table[request, :length].long())
        entries.append(Entry(request, prefix, length, decode=decode))
    # Every row the batch reads; the backend overwrites the new tokens' rows.
    read_rows = torch.cat(rows)
    for pool in (k_pool, v_pool):
        pool[read_rows] = torch.randn(
            (len(read_rows), *pool.shape[1:]), generator=generator, device=device, dtype=pool.dtype
        )

    batch_rows = forward_rows(*entries)
    num_tokens = sum(extend for _, extend in requests)
    widths = [num_qo_heads * HEAD_DIM, num_kv_heads * HEAD_DIM, num_kv_heads * HEAD_DIM]
    qkv = torch.randn(num_tokens, sum(widths), generator=generator, device=device)
    q, k, v = qkv.to(torch.bfloat16).split(widths, dim=-1)
    q = q.view(-1, num_qo_heads, HEAD_DIM)
    with torch.inference_mode():
        metadata = backend.prepare_metadata(batch_rows)
        batch = ForwardInputs(
            None, None, metadata, batch_rows.size, batch_rows.uses_ragged_attention
        )
        output = backend.forward(q, k, v, 0, batch).view(-1, num_qo_heads, HEAD_DIM)
    torch.cuda.synchronize(device)
    # The wrapper the batch ran on, by identity, and the pool rows of its new tokens.
    ran_on = metadata.wrapper
    wrappers = {"prefill": backend.prefill_wrapper, "decode": backend.decode_wrapper}
    wrapper = next((name for name, known in wrappers.items() if known is ran_on), None)
    written = metadata.cache_write_indices.long()
    cache_rows_written = {
        name: torch.equal(pool[written], new.view(-1, num_kv_heads, HEAD_DIM))
        for name, pool, new in (("k", k_pool, k), ("v", v_pool, v))
    }

    max_error = max_violation = reference_scale = 0.0
    start = 0
    for (prefix, extend), request_rows in zip(requests, rows, strict=True):
        expected = _reference(
            q[start : start + extend], k_pool[request_rows], v_pool[request_rows], prefix
        )
        error = (output[start : start + extend].float() - expected).abs()
        max_error = max(max_error, float(error.max()))
        max_violation = max(max_violation, float((error - RTOL * expected.abs() - ATOL).max()))
        reference_scale = max(reference_scale, float(expected.abs().max()))
        start += extend
    return {
        "requests": len(requests),
        "tokens": num_tokens,
        "max_cached_tokens": max(prefix for prefix, _ in requests),
        "wrapper": wrapper,
        "cache_rows_written": cache_rows_written,
        "max_abs_error": max_error,
        # Largest |actual - expected| - (RTOL |expected| + ATOL); within tolerance if <= 0.
        "max_violation": max_violation,
        "reference_max_abs": reference_scale,
        "finite": bool(torch.isfinite(output).all()),
    }


def _operations(geometry: str, workdir: Path) -> dict[str, Any]:
    import torch

    from cti.attention import (
        AttentionRequest,
        AttentionRequirement,
        resolve_attention_config,
    )
    from cti.kvcache import GroupRole
    from cti.utils import get_device_capability
    from tests.support.paged_attention import kv_group

    config = _model_config(geometry, workdir)
    assert config.head_dim == HEAD_DIM
    device = torch.device("cuda", torch.cuda.current_device())
    resolved = resolve_attention_config(
        {
            GroupRole.TARGET: (
                AttentionRequirement.mha(config.num_qo_heads, config.num_kv_heads, config.head_dim),
            )
        },
        AttentionRequest(
            target=GroupRole.TARGET,
            backend="auto",
            page_size=1,
            capability=get_device_capability(),
            tensor_parallel_size=1,
            dtype=torch.bfloat16,
        ),
    )
    assert resolved.selected == "fi", resolved
    group = kv_group(
        1, config.num_kv_heads, config.head_dim, NUM_PAGES, dtype=torch.bfloat16, device=device
    )
    cache_index_table = torch.zeros(
        (max(DECODE_BATCH_SIZES) + 1, TABLE_WIDTH), dtype=torch.int32, device=device
    )
    backend = resolved.build(
        (group,),
        SimpleNamespace(page_size=1, pages=cache_index_table),
        # The table holds a row per running request and the graph-padding row.
        max_running_req=max(DECODE_BATCH_SIZES),
        capability=None,
    )
    pages = torch.randperm(
        NUM_PAGES, generator=torch.Generator(device=device).manual_seed(0), device=device
    ).to(torch.int32)
    prefill = {
        name: _run_batch(
            backend, group, cache_index_table, pages, requests, decode=False, seed=seed
        )
        for seed, (name, requests) in enumerate(PREFILL_BATCHES.items())
    }
    decode = {
        str(batch_size): _run_batch(
            backend,
            group,
            cache_index_table,
            pages,
            _decode_requests(batch_size),
            decode=True,
            seed=100 + batch_size,
        )
        for batch_size in DECODE_BATCH_SIZES
    }
    return {
        "backend": type(backend).__name__,
        "num_qo_heads": config.num_qo_heads,
        "num_kv_heads": config.num_kv_heads,
        "execution": {
            "prefill": backend.prefill_implementation,
            "tensor_cores": backend.use_tensor_cores,
            "dispatch": backend.dispatch,
        },
        "prefill": prefill,
        "decode": decode,
    }


def _run_child(component: str, workdir: Path) -> None:
    import torch

    import cti

    receipt: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "component": component,
        "engine_root": str(Path(cti.__file__).resolve().parent),
    }
    receipt.update(_operations(component, workdir))
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def _receipt(tmp_path: Path, component: str) -> dict[str, Any]:
    env, log_root = isolated_runtime_env(tmp_path, f"flashinfer-head-dim-256-{component}")
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.layers.test_flashinfer_head_dim_256",
            "--component",
            component,
            "--workdir",
            str(tmp_path),
        ),
        log_root / f"{component}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )
    receipt = child_receipt(output, RECEIPT_PREFIX)
    write_receipt(f"flashinfer_head_dim_256-{component}", receipt)
    assert_cuda_device_metadata(receipt)
    assert receipt["component"] == component
    assert Path(receipt["engine_root"]).is_relative_to(PYTHON_ROOT)
    return receipt


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_flashinfer_paged_attention_serves_head_dim_256(tmp_path: Path, geometry: str) -> None:
    receipt = _receipt(tmp_path, geometry)

    assert receipt["backend"] == "FlashInferBackend"
    num_qo_heads, num_kv_heads = map(int, geometry.split("x"))
    assert (receipt["num_qo_heads"], receipt["num_kv_heads"]) == (num_qo_heads, num_kv_heads)
    assert set(receipt["prefill"]) == set(PREFILL_BATCHES)
    assert set(receipt["decode"]) == {str(batch_size) for batch_size in DECODE_BATCH_SIZES}
    for name, batch in (*receipt["prefill"].items(), *receipt["decode"].items()):
        assert batch["finite"] is True, (name, batch)
        assert batch["cache_rows_written"] == {"k": True, "v": True}, (name, batch)
        assert batch["max_violation"] <= 0, (name, batch)
    assert {batch["wrapper"] for batch in receipt["prefill"].values()} == {"prefill"}
    assert {batch["wrapper"] for batch in receipt["decode"].values()} == {"decode"}
    assert receipt["prefill"]["chunk-8192-onto-8192"]["tokens"] == 8192


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", required=True, choices=GEOMETRIES)
    parser.add_argument("--workdir", type=Path, required=True)
    arguments = parser.parse_args()
    _run_child(arguments.component, arguments.workdir)


if __name__ == "__main__":
    main()
