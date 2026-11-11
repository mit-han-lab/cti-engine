"""A mixed query's attention on the GPU: prompt rows that FlashInfer planned and fixed-query
rows after them, in one layer call, against the two groups called apart.

Each case writes the keys and values of both groups once and attends prompts of 1, 17 and
129 new tokens (one of them behind a cached prefix) beside one to three fixed rows of 16
queries, through ``forward_into`` and through a fused prologue; queries are a strided slice of
one projection. Outputs and the pool rows written must equal the separate calls' bitwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.checkpoints()

RECEIPT_PREFIX = "CTI_MIXED_QUERY_ATTENTION="
_Q = 16
_MAX_SEQ_LEN = 1024
_SLOTS = 8
# (cached prefix, new tokens) of each prompt; the fixed rows' frontiers.
_CASES = {
    "one-token": (((0, 1),), (40,)),
    "prompts": (((0, 17), (64, 129)), (33, 200)),
    "three-rows": (((0, 129),), (1, 17, 300)),
}
# Query and KV heads with their head dim: Qwen3-8B's and Qwen3.5-9B's attention.
GEOMETRIES = {"32x8x128": (32, 8, 128), "16x4x256": (16, 4, 256)}


def _case(
    backends: tuple[Any, Any, Any],
    group: Any,
    table: Any,
    prompts: tuple[tuple[int, int], ...],
    frontiers: tuple[int, ...],
    *,
    prologue: bool,
    seed: int,
) -> dict[str, bool]:
    import torch

    from cti.attention import MixedQuery
    from tests.support.forward_rows import extend_entry, forward_rows

    _, draft, addresses = backends
    device = table.device
    generator = torch.Generator(device=device).manual_seed(seed)
    k_pool, v_pool = (cache.flatten(0, 1) for cache in group.layers[0])
    for pool in (k_pool, v_pool):
        pool.copy_(torch.randn(pool.shape, generator=generator, device=device))
    heads, kv_heads, dim = backends[0].qo_head_local, k_pool.shape[1], k_pool.shape[2]
    # Prompts in slots 0.., fixed rows after them; each slot owns a disjoint page range.
    table.copy_(
        torch.randperm(table.numel(), generator=generator, device=device)
        .view(table.shape)
        .to(torch.int32)
    )
    extension = forward_rows(
        *(
            extend_entry(slot, prefix, prefix + extend)
            for slot, (prefix, extend) in enumerate(prompts)
        )
    )
    tokens = sum(extend for _, extend in prompts)
    rows = len(frontiers)
    first_row = len(prompts)
    request_indices = torch.arange(first_row, first_row + rows, dtype=torch.int32, device=device)
    positions = torch.cat([torch.arange(f, f + _Q, device=device) for f in frontiers]).to(
        torch.int64
    )
    kv_tokens = _MAX_SEQ_LEN + _Q
    kv_indptr = torch.tensor(
        [0, *torch.tensor(frontiers).add(_Q).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    kv_indices = torch.zeros(rows * kv_tokens, dtype=torch.int32, device=device)
    row_writes = []
    for row, frontier in enumerate(frontiers):
        slot_rows = table[first_row + row, : frontier + _Q]
        kv_indices[kv_indptr[row] : kv_indptr[row + 1]] = slot_rows
        row_writes.append(slot_rows[frontier:])
    fixed_writes = torch.cat(row_writes)
    addresses.indptr[: rows + 1] = kv_indptr
    addresses.indices[: rows * kv_tokens] = kv_indices
    addresses.write[: rows * _Q] = fixed_writes
    fixed = draft.create_fixed_query(request_indices, positions, causal=True)

    width = (heads + 2 * kv_heads) * dim
    qkv = torch.randn(tokens + rows * _Q, width, generator=generator, device=device)
    q, k, v = qkv.to(torch.bfloat16).split([heads * dim, kv_heads * dim, kv_heads * dim], -1)
    q = q.view(-1, heads, dim)
    pools = (k_pool.clone(), v_pool.clone())

    def attend(batch, q_rows, k_rows, v_rows):
        out = q_rows.new_empty(q_rows.shape)
        with torch.inference_mode():
            if not prologue:
                draft.forward_into(q_rows, k_rows, v_rows, 0, batch, out)
                return out

            # A fused prologue writes the rows' keys and values and yields their queries.
            def run(k_cache, v_cache, cache_write_indices):
                written = cache_write_indices.long()
                k_cache[written] = k_rows.reshape(-1, kv_heads, dim)
                v_cache[written] = v_rows.reshape(-1, kv_heads, dim)
                return q_rows

            return draft.forward_prologue(run, 0, batch, out)

    with torch.inference_mode():
        planned = draft.prepare_metadata(extension)
        fixed.prepare()
    # Apart: the prompts through the paged backend, the rows through DraftAttention.
    prompts_batch = SimpleNamespace(attn_metadata=planned)
    rows_batch = SimpleNamespace(attn_metadata=fixed)
    apart = torch.cat(
        (
            attend(prompts_batch, q[:tokens], k[:tokens], v[:tokens]),
            attend(rows_batch, q[tokens:], k[tokens:], v[tokens:]),
        )
    )
    apart_pools = (k_pool.clone(), v_pool.clone())
    # Together: the pool as it was, one call over both groups.
    k_pool.copy_(pools[0])
    v_pool.copy_(pools[1])
    joint = SimpleNamespace(attn_metadata=MixedQuery(planned, fixed, tokens))
    together = attend(joint, q, k, v)
    torch.cuda.synchronize(device)
    return {
        "outputs_equal": torch.equal(together, apart),
        "finite": bool(torch.isfinite(together).all()),
        "pools_equal": torch.equal(k_pool, apart_pools[0]) and torch.equal(v_pool, apart_pools[1]),
    }


def _operations(geometry: str) -> dict[str, Any]:
    import torch

    from cti.attention import (
        FixedQueryGeometry,
        FixedQueryShape,
        create_fixed_query_backends,
    )
    from cti.attention.base import BackendSetup
    from cti.attention.flashinfer import FlashInferBackend
    from cti.attention.paged import kv_pages
    from tests.support.paged_attention import kv_group

    heads, kv_heads, dim = GEOMETRIES[geometry]
    device = torch.device("cuda", torch.cuda.current_device())
    num_pages = (_SLOTS + 1) * (_MAX_SEQ_LEN + _Q)
    group = kv_group(1, kv_heads, dim, num_pages, dtype=torch.bfloat16, device=device)
    table = torch.zeros((_SLOTS + 1, _MAX_SEQ_LEN + _Q), dtype=torch.int32, device=device)
    page_table = SimpleNamespace(page_size=1, pages=table, tokens=table)
    setup = BackendSetup(
        head_dim=dim,
        num_qo_heads=heads,
        num_kv_heads=kv_heads,
        dtype=torch.bfloat16,
        device=device,
        capability=None,
        max_running_req=_SLOTS,
        tensor_cores=None,
    )
    paged = FlashInferBackend(setup, kv=kv_pages(group), table=page_table)
    shape = FixedQueryShape(4, _Q, _MAX_SEQ_LEN + _Q, ((True, None),), torch.bfloat16)
    (draft,) = create_fixed_query_backends(
        (FixedQueryGeometry(dim, heads, kv_heads, shape),),
        SimpleNamespace(
            groups=(group,),
            capacity=SimpleNamespace(max_seq_len=_MAX_SEQ_LEN, guard_start=num_pages),
        ),
        (paged,),
        page_table,
    )
    # The cases write their own round's addresses.
    addresses = draft._addresses
    receipt = {}
    for seed, (name, (prompts, frontiers)) in enumerate(_CASES.items()):
        for prologue in (False, True):
            receipt[f"{name}-{'prologue' if prologue else 'forward'}"] = _case(
                (paged, draft, addresses),
                group,
                table,
                prompts,
                frontiers,
                prologue=prologue,
                seed=seed,
            )
    return receipt


def _run_child(geometry: str) -> None:
    import torch

    receipt: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "cases": _operations(geometry),
    }
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_a_mixed_query_attends_like_its_groups_apart(tmp_path: Path, geometry: str) -> None:
    env, log_root = isolated_runtime_env(tmp_path, f"mixed-query-{geometry}")
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.layers.test_mixed_query_attention",
            "--geometry",
            geometry,
        ),
        log_root / f"{geometry}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=600,
    )
    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert_cuda_device_metadata(receipt)
    assert len(receipt["cases"]) == 2 * len(_CASES)
    for name, case in receipt["cases"].items():
        assert case == {"outputs_equal": True, "finite": True, "pools_equal": True}, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    _run_child(parser.parse_args().geometry)


if __name__ == "__main__":
    main()
