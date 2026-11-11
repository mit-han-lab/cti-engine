from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

import cti.layers.prepare
from cti.attention.requirement import kv_head_window
from cti.config import LoRAConfig
from cti.distributed import Topology
from cti.layers import (
    Adapter,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    ParallelLMHead,
    prepare_inference,
)
from cti.layers.lora import LoRAProjection
from cti.layers.placement import WeightShard
from tests.support.gemm import GemmRecorder
from tests.support.tp_shards import site, windows

HIDDEN = 32
HEAD_DIM = 8
NUM_QO_HEADS = 4
NUM_KV_HEADS = 2


def _gated_projection(topology: Topology | None = None) -> LinearQKVMerged:
    return LinearQKVMerged(
        HIDDEN,
        HEAD_DIM,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        has_bias=False,
        with_output_gate=True,
        site=site(topology),
    )


def test_gated_qkv_projection_matches_hf_query_gate_key_value() -> None:
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        hidden_size=HIDDEN,
        num_attention_heads=NUM_QO_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
    )
    attention = Qwen3_5Attention(config, layer_idx=3).to(torch.float64)
    # HF interleaves each head's query and gate rows; the merged layout is q|k|v|gate.
    query_rows, gate_rows = (
        attention.q_proj.weight.detach().view(NUM_QO_HEADS, 2, HEAD_DIM, HIDDEN).unbind(1)
    )
    weight = torch.cat(
        (
            query_rows.reshape(-1, HIDDEN),
            attention.k_proj.weight.detach(),
            attention.v_proj.weight.detach(),
            gate_rows.reshape(-1, HIDDEN),
        )
    )
    projection = _gated_projection()
    projection.weight = torch.empty_like(weight)
    projection.load_state_dict({"weight": weight})
    x = torch.randn(5, HIDDEN, dtype=torch.float64)

    output = projection.forward(x)
    actual = {
        item.name: output[:, item.offset : item.offset + item.size]
        for item in projection.output_slices
    }

    with torch.no_grad():
        query, gate = attention.q_proj(x).view(5, NUM_QO_HEADS, 2 * HEAD_DIM).chunk(2, dim=-1)
        expected = {
            "q": query.reshape(5, -1),
            "k": attention.k_proj(x),
            "v": attention.v_proj(x),
            "gate": gate.reshape(5, -1),
        }
    assert list(actual) == ["q", "k", "v", "gate"]
    for name, reference in expected.items():
        torch.testing.assert_close(actual[name], reference)


def test_gated_qkv_projection_splits_the_gate_with_query_heads() -> None:
    projection = _gated_projection(Topology(1, 2))
    query_size = NUM_QO_HEADS // 2 * HEAD_DIM
    kv_size = NUM_KV_HEADS // 2 * HEAD_DIM

    assert projection.full_output_size == (2 * NUM_QO_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    assert projection.weight.shape == (2 * query_size + 2 * kv_size, HIDDEN)
    assert [(item.name, item.offset, item.size) for item in projection.output_slices] == [
        ("q", 0, query_size),
        ("k", query_size, kv_size),
        ("v", query_size + kv_size, kv_size),
        ("gate", query_size + 2 * kv_size, query_size),
    ]
    # The checkpoint's gate rows follow the query heads this rank owns.
    query = WeightShard(0, query_size, 2 * query_size)
    kv = WeightShard(0, kv_size, 2 * kv_size)
    assert windows(projection) == {"weight": (query, kv, kv, query)}


def test_inference_runs_the_kernel_gemm_and_training_runs_f_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = GemmRecorder(monkeypatch)
    layer = LinearRowParallel(HIDDEN, 6, has_bias=False, site=site())
    biased = LinearReplicated(HIDDEN, 6, has_bias=True, site=site())
    head = ParallelLMHead(5, HIDDEN, site=site())
    layer.weight, biased.weight, head.weight = (torch.randn(n, HIDDEN) for n in (6, 6, 5))
    biased.bias = torch.randn(6)
    x = torch.randn(3, HIDDEN)

    torch.testing.assert_close(layer.forward(x), F.linear(x, layer.weight))
    torch.testing.assert_close(head.project(x), F.linear(x, head.weight))
    assert [(call.rows, call.shape) for call in recorder.calls] == [
        (3, (6, HIDDEN)),
        (3, (5, HIDDEN)),
    ]
    # Training and biased projections stay on F.linear.
    torch.testing.assert_close(layer.forward_training(x), F.linear(x, layer.weight))
    torch.testing.assert_close(biased.forward(x), F.linear(x, biased.weight, biased.bias))
    assert len(recorder.calls) == 2


def test_project_kv_runs_only_the_key_and_value_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = GemmRecorder(monkeypatch)
    projection = LinearQKVMerged(
        HIDDEN,
        HEAD_DIM,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        has_bias=False,
        with_output_gate=True,
        projects_kv=True,
        site=site(),
    )
    projection.weight = torch.randn(projection.local_output_size, HIDDEN)
    x = torch.randn(3, HIDDEN)

    k, v = projection.project_kv(x)

    _, key, value, _ = projection.output_slices
    full = F.linear(x, projection.weight)
    torch.testing.assert_close(k, full[:, key.offset : key.offset + key.size])
    torch.testing.assert_close(v, full[:, value.offset : value.offset + value.size])
    assert [call.shape for call in recorder.calls] == [(2 * NUM_KV_HEADS * HEAD_DIM, HIDDEN)]
    assert len(list(projection.inference_gemms())) == 2
    with pytest.raises(RuntimeError, match="projects_kv"):
        _gated_projection().project_kv(x)


def test_one_workspace_serves_every_inference_gemm(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = GemmRecorder(monkeypatch)
    projection = LinearQKVMerged(
        HIDDEN, HEAD_DIM, NUM_QO_HEADS, NUM_KV_HEADS, has_bias=False, site=site()
    )
    head = ParallelLMHead(6, HIDDEN, site=site())
    draft = LinearReplicated(HIDDEN, 8, has_bias=False, site=site())
    adapted = LinearReplicated(HIDDEN, 4 * HIDDEN, has_bias=False, site=site(), lora="mlp")
    Adapter(LoRAConfig(targets=("mlp",))).attach(adapted)
    projection.weight = torch.randn(projection.local_output_size, HIDDEN)
    head.weight, draft.weight = torch.randn(6, HIDDEN), torch.randn(8, HIDDEN)
    x = torch.randn(3, HIDDEN)

    biased = LinearReplicated(HIDDEN, 16 * HIDDEN, has_bias=True, site=site())
    size = prepare_inference(
        (projection, head, adapted, biased, draft),
        max_rows=0,
        max_sequences=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    projection.forward(x)
    head.project(x)
    draft.forward(x)

    # The largest call site's scratch; the LoRA host runs its adapter's kernel
    # and the biased one F.linear.
    assert size == 4 * projection.local_output_size
    workspaces = [call.workspace for call in recorder.calls]
    assert len(workspaces) == 3 and all(workspace is workspaces[0] for workspace in workspaces)
    assert workspaces[0].numel() * workspaces[0].element_size() == size


def test_an_adapted_call_site_warms_up_its_lora_plan_over_the_layer_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapted = LinearReplicated(HIDDEN, 4 * HIDDEN, has_bias=False, site=site(), lora="mlp")
    Adapter(LoRAConfig(targets=("mlp",))).attach(adapted)
    adapted.weight = torch.randn(4 * HIDDEN, HIDDEN)
    warmed = []
    route = {"op": "delta", "m": (1, 5), "provider": "cute", "config": None}

    def warm_up(lora: LoRAProjection, rows: int, weight: torch.Tensor) -> list[dict]:
        warmed.append((rows, weight))
        return [route]

    monkeypatch.setattr(LoRAProjection, "warm_up", warm_up)

    assert (
        prepare_inference(
            (adapted,),
            max_rows=5,
            max_sequences=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        == 0
    )
    # The plan compiles the delta and the base product it reads the weight for.
    assert len(warmed) == 1 and warmed[0][0] == 5 and warmed[0][1] is adapted.weight


def test_the_workspace_covers_every_call_site_of_a_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = GemmRecorder(monkeypatch)
    layers = [LinearReplicated(HIDDEN, 6, has_bias=False, site=site()) for _ in range(2)]
    # A call site may keep a route its sibling of the same shape did not (decisions are per
    # operation), here one that needs more split-K scratch.
    monkeypatch.setattr(recorder.operations[1], "workspace_bytes", lambda *_a, **_k: 1000)

    assert (
        prepare_inference(
            layers,
            max_rows=0,
            max_sequences=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        == 1000
    )


def test_preparing_reports_each_call_site_and_each_bucket_it_compiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cold compiles outlast the engine's startup stall bound unless each one reports.
    GemmRecorder(monkeypatch, cute_rows=4)
    layers = [LinearReplicated(HIDDEN, n, has_bias=False, site=site()) for n in (6, 8)]
    for layer in layers:
        layer.weight = torch.randn(layer.local_output_size, HIDDEN)
    heard: list[str] = []

    prepare_inference(
        layers,
        max_rows=8,
        max_sequences=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        progress=heard.append,
    )

    assert heard == [
        f"GEMM {site}{bucket}"
        for site, n in (("1/2", 6), ("2/2", 8))
        for bucket in (
            "",
            f": {n}x{HIDDEN} rows 1-1",
            f": {n}x{HIDDEN} rows 2-2",
            f": {n}x{HIDDEN} rows 3-4",
        )
    ]


@pytest.mark.parametrize(
    ("max_rows", "rows"), [(12, [1, 2, 4, 8]), (3, [1, 2, 3]), (32, [1, 2, 4, 8])]
)
def test_warm_up_compiles_each_cute_bucket_once_per_call_site(
    monkeypatch: pytest.MonkeyPatch, max_rows: int, rows: list[int]
) -> None:
    recorder = GemmRecorder(monkeypatch, cute_rows=8)
    layers = [LinearReplicated(HIDDEN, n, has_bias=False, site=site()) for n in (6, 6, 8)]
    for layer in layers:
        layer.weight = torch.randn(layer.local_output_size, HIDDEN)

    prepare_inference(
        layers,
        max_rows=max_rows,
        max_sequences=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # One call per bucket up to max_rows, at the bucket's last row count: every row count of
    # a bucket runs its plan. Buckets the dispatch gives cuBLAS (9-16 here) and rows past
    # the buckets compile nothing.
    for operation in recorder.operations:
        assert [(call.rows, call.gate_stride) for call in operation.calls] == [(m, 0) for m in rows]


def _o_proj() -> LinearRowParallel:
    """A BF16 output projection gated by columns of one-head QKV rows, ``3 * HIDDEN``
    wide (the kernel takes a gate with BF16 inputs only)."""
    qkv = LinearQKVMerged(HIDDEN, HIDDEN, 1, 1, has_bias=False, site=site())
    o_proj = LinearRowParallel(HIDDEN, 6, has_bias=False, site=site(), gate_of=qkv)
    o_proj.weight = torch.randn(6, HIDDEN).bfloat16()
    return o_proj


def test_o_proj_fuses_the_output_gate_where_the_kernel_accepts_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = GemmRecorder(monkeypatch, fused_rows=2)
    stride = 3 * HIDDEN
    o_proj = _o_proj()
    for rows in (2, 3):
        x = torch.randn(rows, HIDDEN).bfloat16()
        # The gate is a column slice of wider projection rows.
        gate = torch.randn(rows, stride).bfloat16()[:, -HIDDEN:]

        gated = (x.float() * torch.sigmoid(gate.float())).bfloat16()
        expected = F.linear(gated, o_proj.weight)
        torch.testing.assert_close(o_proj.forward_gated(x, gate), expected)
        torch.testing.assert_close(o_proj.forward_gated(x, gate.contiguous()), expected)

    # Every gated call hands its gate to the kernel, which fuses where it accepts the call.
    assert [(call.rows, call.gate_stride) for call in recorder.calls] == [
        (2, stride),
        (2, HIDDEN),
        (3, stride),
        (3, HIDDEN),
    ]


def test_a_gated_call_site_warms_up_both_routes_and_sizes_for_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = GemmRecorder(monkeypatch, fused_rows=2)
    o_proj = _o_proj()
    logged = []
    monkeypatch.setattr(
        cti.layers.prepare,
        "logger",
        SimpleNamespace(info_rank0=lambda message, *args: logged.append(message % args)),
    )

    size = prepare_inference(
        (o_proj,),
        max_rows=4,
        max_sequences=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # The plain route in buckets 1, 2 and 3-4; the fused gated one where it is admitted.
    assert [(call.rows, call.gate_stride) for call in recorder.calls] == [
        (1, 0),
        (2, 0),
        (4, 0),
        (1, 3 * HIDDEN),
        (2, 3 * HIDDEN),
    ]
    assert size == 4 * (6 + HIDDEN)
    assert logged == [
        "Inference GEMM routes by rows: 1: cute 1, fused 1; 2: cute 1, fused 1; "
        "3-4: cute 1, unfused 1"
    ]


@pytest.mark.parametrize(
    ("num_kv_heads", "size", "windows"),
    [
        (8, 1, [(0, 8)]),
        (8, 4, [(0, 2), (2, 2), (4, 2), (6, 2)]),
        (2, 4, [(0, 1), (0, 1), (1, 1), (1, 1)]),
    ],
    ids=["one-rank", "split", "replica"],
)
def test_kv_head_window_splits_kv_heads_or_replicates_them(
    num_kv_heads: int, size: int, windows: list[tuple[int, int]]
) -> None:
    assert [kv_head_window(num_kv_heads, rank, size) for rank in range(size)] == windows


@pytest.mark.parametrize(("num_kv_heads", "size"), [(6, 4), (3, 4)], ids=["split", "replica"])
def test_kv_head_window_refuses_uneven_layouts(num_kv_heads: int, size: int) -> None:
    with pytest.raises(ValueError, match="divisible"):
        kv_head_window(num_kv_heads, 0, size)


class _Summing:
    """A rank of two whose ``all_reduce`` records its input and returns it."""

    def __init__(self, rank: int) -> None:
        self.rank, self.size = rank, 2
        self.reduced: list[torch.Tensor] = []

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.reduced.append(tensor.clone())
        return tensor


@pytest.mark.parametrize("rank", [0, 1])
def test_a_deferred_exit_leaves_inference_its_share_and_sums_training(rank: int) -> None:
    group = _Summing(rank)
    layer = LinearRowParallel(2 * HIDDEN, 6, has_bias=True, site=site(group=group), deferred=True)
    layer.weight, layer.bias = torch.randn(6, HIDDEN), torch.randn(6)
    x = torch.randn(3, HIDDEN)

    # Rank 0's share carries the bias, so the ranks' sum holds it once.
    share = F.linear(x, layer.weight, layer.bias if rank == 0 else None)
    torch.testing.assert_close(layer.forward(x), share)
    assert group.reduced == []
    torch.testing.assert_close(layer.forward_training(x), F.linear(x, layer.weight, layer.bias))
    assert len(group.reduced) == 1
