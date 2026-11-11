from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

import cti.attention.flashinfer as flashinfer_module
import cti.attention.tensorrt_llm as tensorrt_llm_module
from cti.attention.base import BackendSetup
from cti.attention.flash_attention import FlashAttentionBackend
from cti.attention.tensorrt_llm import TensorRTLLMBackend
from cti.utils import DeviceCapability, align_ceil, div_ceil
from tests.support.attention import CPUAttention

MiB = 1024 * 1024
# Attention heads per rank: (query heads, KV heads, head_dim).
QWEN3_32B = (64, 8, 128)
QWEN3_32B_TP2 = (32, 4, 128)
QWEN3_5_27B = (24, 4, 256)
QWEN3_5_27B_TP2 = (12, 2, 256)
RTX_PRO_6000_SMS = 188
_SETUP = BackendSetup(
    head_dim=128,
    num_qo_heads=8,
    num_kv_heads=8,
    dtype=torch.bfloat16,
    device=torch.device("cpu"),
    capability=DeviceCapability(10, 0),
    max_running_req=256,
    tensor_cores=None,
)


def _tile_rows(packed_qo_len: int, head_dim: int) -> int:
    if packed_qo_len > 64 and head_dim < 256:
        return 128
    return 64 if packed_qo_len > 16 else 16


def _planned_scratch(
    *,
    num_sms: int,
    heads: tuple[int, int, int],
    requests: list[tuple[int, int]],
    graph: bool = False,
) -> tuple[int, int]:
    """Model one FlashInfer 0.6.11 FA2 plan: (tmp_v, tmp_s) bytes, (0, 0) unsplit.

    Re-derived on the CPU from ``attention/scheduler.cuh`` (``PrefillPlan``,
    ``PrefillSplitQOKVIndptr``, ``PrefillBinarySearchKVChunkSize``) and ``utils.cuh``
    (``FA2DetermineCtaTileQ``) at page size 1 without a sliding window. ``requests``
    holds (new query tokens, KV tokens). ``graph`` models a tensor-core decode graph,
    where each request has one query token.
    """
    num_qo_heads, num_kv_heads, head_dim = heads
    group = num_qo_heads // num_kv_heads
    max_tiles = 2 * num_sms // num_kv_heads
    packed = [qo_len * group for qo_len, _ in requests]
    kv_lens = [max(kv_len, 1) for _, kv_len in requests]
    tile = _tile_rows(group if graph else sum(packed) // len(packed), head_dim)

    def split_tiles(chunk: int) -> int:
        return sum(
            div_ceil(p, tile) * div_ceil(k, chunk) for p, k in zip(packed, kv_lens, strict=True)
        )

    chunk, high = 128, max(kv_lens)
    while chunk < high:
        middle = (chunk + high) // 2
        if split_tiles(middle) > max_tiles:
            chunk = middle + 1
        else:
            high = middle
    if graph:
        padded = max(max_tiles, div_ceil(len(requests) * group, tile) + len(requests) - 1)
    elif chunk < max(kv_lens):
        padded = split_tiles(chunk)
    else:
        return 0, 0
    return num_qo_heads * padded * tile * head_dim * 4, num_qo_heads * padded * tile * 4


def _worst_prefill(num_sms: int, heads: tuple[int, int, int]) -> list[tuple[int, int]]:
    """One request filling one largest query tile (at least one token) over the most
    KV chunks."""
    num_qo_heads, num_kv_heads, head_dim = heads
    largest_tile = 128 if head_dim < 256 else 64
    qo_len = max(1, largest_tile * num_kv_heads // num_qo_heads)
    return [(qo_len, 2 * num_sms // num_kv_heads * 128)]


def _scratch_bound(
    monkeypatch: pytest.MonkeyPatch,
    num_sms: int,
    heads: tuple[int, int, int],
    max_running_req: int,
) -> int:
    """The float scratch FlashInfer reserves on ``num_sms`` SMs: the reservation less the
    integer workspace a backend allocates."""
    attention = CPUAttention(monkeypatch, num_sms=num_sms)
    attention.build()
    int_bytes = attention.operations[0].int_workspace.nbytes
    return attention.workspace_bytes(heads=heads, max_running_req=max_running_req) - int_bytes


# tmp_v FlashInfer 0.6.11 reported on an RTX PRO 6000 with a 128 MiB float workspace. The
# Qwen3.5-27B shapes are synthetic matches; at TP2, half the heads take twice the tiles.
@pytest.mark.parametrize(
    ("heads", "shape", "tmp_v"),
    [
        pytest.param(QWEN3_32B, (128, 2176), 167_772_160, id="qwen3-32b-prefix"),
        pytest.param(QWEN3_32B_TP2, (512, 512), 128 * MiB, id="qwen3-32b-tp2-prompt"),
        pytest.param(QWEN3_5_27B, (130, 2178), 143_130_624, id="qwen3.5-27b-prefix"),
        pytest.param(QWEN3_5_27B, (466, 466), 138_412_032, id="qwen3.5-27b-prompt"),
        pytest.param(QWEN3_5_27B_TP2, (130, 2178), 143_130_624, id="qwen3.5-27b-tp2-prefix"),
        pytest.param(QWEN3_5_27B_TP2, (466, 466), 138_412_032, id="qwen3.5-27b-tp2-prompt"),
    ],
)
def test_observed_overflow_fits_the_bound(
    monkeypatch: pytest.MonkeyPatch, heads: tuple[int, int, int], shape: tuple[int, int], tmp_v: int
) -> None:
    planned_v, planned_s = _planned_scratch(num_sms=RTX_PRO_6000_SMS, heads=heads, requests=[shape])

    assert planned_v == tmp_v
    assert (
        128 * MiB
        < planned_v + planned_s
        <= _scratch_bound(monkeypatch, RTX_PRO_6000_SMS, heads, 256)
    )


@pytest.mark.parametrize(
    "heads",
    [
        pytest.param((16, 8, 128), id="qwen3-0.6b"),
        pytest.param((32, 8, 128), id="qwen3-8b"),
        pytest.param(QWEN3_32B, id="qwen3-32b"),
        pytest.param(QWEN3_32B_TP2, id="qwen3-32b-tp2"),
        pytest.param(QWEN3_5_27B, id="qwen3.5-27b"),
        pytest.param((16, 2, 256), id="qwen3.5-35b-a3b"),
        # Higher GQA groups reach the larger tensor-core decode tiles.
        pytest.param((64, 4, 128), id="group16"),
        pytest.param((64, 2, 128), id="group32"),
        pytest.param((96, 1, 128), id="group96"),
        pytest.param((128, 1, 256), id="group128-d256"),
    ],
)
@pytest.mark.parametrize("num_sms", [48, 132, 170, RTX_PRO_6000_SMS])
@pytest.mark.parametrize("max_running_req", [64, 256])
def test_flashinfer_float_workspace_is_the_largest_planned_split(
    monkeypatch: pytest.MonkeyPatch,
    heads: tuple[int, int, int],
    num_sms: int,
    max_running_req: int,
) -> None:
    generator = random.Random(f"{heads}-{num_sms}-{max_running_req}")
    prefill_batches = [
        _worst_prefill(num_sms, heads),
        [(128, 2176)],
        [(512, 512)],
        [(1, 4096)] * 8,
    ]
    for _ in range(200):
        batch = []
        for _ in range(generator.choice((1, 1, 2, 3, 8, 32))):
            kv_len = generator.choice((1, 17, 128, 300, 2048, 6016, 32768))
            batch.append((generator.randint(1, min(kv_len, 2048)), kv_len))
        prefill_batches.append(batch)
    planned = [_planned_scratch(num_sms=num_sms, heads=heads, requests=b) for b in prefill_batches]
    planned += [
        _planned_scratch(num_sms=num_sms, heads=heads, requests=[(1, 4096)] * size, graph=True)
        for size in (1, 7, 160, max_running_req)
        if size <= max_running_req
    ]

    workspace = _scratch_bound(monkeypatch, num_sms, heads, max_running_req)

    # Never exceeded, and attained by the worst prefill or the largest decode graph.
    assert max(align_ceil(tmp_v, 16) + tmp_s for tmp_v, tmp_s in planned) == workspace


def test_flashinfer_prefill_scratch_does_not_grow_with_tokens(monkeypatch) -> None:
    worst = _scratch_bound(monkeypatch, RTX_PRO_6000_SMS, QWEN3_32B, max_running_req=1)
    for tokens in (512, 8192, 131072):
        tmp_v, tmp_s = _planned_scratch(
            num_sms=RTX_PRO_6000_SMS, heads=QWEN3_32B, requests=[(16, tokens)]
        )
        assert tmp_v + tmp_s <= worst
    # A 2048-token prompt fills the GPU with query tiles and never splits.
    assert _planned_scratch(
        num_sms=RTX_PRO_6000_SMS, heads=QWEN3_32B_TP2, requests=[(2048, 2048)]
    ) == (0, 0)


def test_flashinfer_decode_graph_scratch_grows_with_running_requests(monkeypatch) -> None:
    workspaces = [
        _scratch_bound(monkeypatch, RTX_PRO_6000_SMS, QWEN3_32B, max_running_req=size)
        for size in (256, 512, 1024)
    ]

    # No fixed size covers every decode graph: 1024 requests need more than 512 MiB.
    assert workspaces == sorted(workspaces)
    assert workspaces[-1] > 512 * MiB


def test_flashinfer_rejects_devices_below_sm80(monkeypatch) -> None:
    def fail_on_load() -> None:
        raise AssertionError("the capability check must come before loading FlashInfer")

    monkeypatch.setattr(flashinfer_module, "check_flashinfer_available", fail_on_load)

    reasons = flashinfer_module.FlashInferBackend.validate_environment(DeviceCapability(7, 5))

    assert len(reasons) == 1
    assert reasons[0] == (
        "FlashInfer attention does not support SM75; it needs compute capability 8.0 or newer"
    )
    monkeypatch.setattr(flashinfer_module, "check_flashinfer_available", lambda: None)
    assert flashinfer_module.FlashInferBackend.validate_environment(DeviceCapability(8, 0)) == ()


def test_tensorrt_llm_reserves_the_workspace_it_allocates(monkeypatch) -> None:
    allocations = []
    workspace = torch.empty((0, 0), dtype=torch.int32)

    def allocate(size, *, dtype, device):
        allocations.append(size)
        return workspace

    monkeypatch.setattr(
        tensorrt_llm_module, "torch", SimpleNamespace(empty=allocate, uint8=torch.uint8)
    )

    TensorRTLLMBackend(_SETUP, kv=(), table=SimpleNamespace(page_size=64))

    assert allocations == [TensorRTLLMBackend.workspace_bytes(_SETUP)]


def test_flash_attention_reserves_no_workspace() -> None:
    assert FlashAttentionBackend.workspace_bytes(_SETUP) == 0
