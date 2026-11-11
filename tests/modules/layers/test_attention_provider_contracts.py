from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import cti.attention.flash_attention as fa_module
import cti.attention.tensorrt_llm as trt_module
from cti.attention.flashinfer import FlashInferBackend
from cti.attention.mha import HybridBackend
from tests.support.attention import CPUAttention
from tests.support.forward_rows import decode_entry, extend_entry, forward_rows

# Two decodes over 2 and 4 tokens: at page size 1, slot 0 writes page 1 and slot 1 page 11.
_DECODES = (decode_entry(0, 1), decode_entry(1, 3))


def test_flashinfer_plans_once_then_stores_and_runs_each_layer(monkeypatch):
    attention = CPUAttention(monkeypatch)
    backend = attention.build()
    _, decode = attention.operations
    rows = forward_rows(*_DECODES)
    batch = SimpleNamespace(attn_metadata=backend.prepare_metadata(rows))
    q = torch.empty(2, 4, 8, dtype=torch.bfloat16)
    k, v = (torch.randn(2, 2, 8, dtype=torch.bfloat16) for _ in range(2))
    attention.log.clear()

    forward_out = backend.forward(q, k, v, 0, batch)
    out = torch.empty_like(q)
    backend.forward_into(q, k, v, 1, batch, out)

    # The KV indices build first; a plan waits out the previous plan's host staging copy,
    # then marks its own.
    assert attention.log == ["kv_metadata", "synchronize", "plan", "record"] + ["store", "run"] * 2
    (plan,) = decode.plans
    assert {name: value.tolist() for name, value in plan.items() if torch.is_tensor(value)} == {
        "cu_seqlens_q": [0, 1, 2],
        "cu_seqlens_k": [0, 2, 6],
        "indices": [0, 1, 8, 9, 10, 11],
        "last_page_len": [1, 1],
        "seq_lens": [2, 4],
    }
    assert (plan["num_qo_heads"], plan["num_kv_heads"], plan["head_dim"]) == (4, 2, 8)
    assert (plan["page_size"], plan["dtype"]) == (1, torch.bfloat16)
    for k_cache, v_cache in attention.group.layers:
        assert torch.equal(k_cache.view(-1, 2, 8)[[1, 11]], k)
        assert torch.equal(v_cache.view(-1, 2, 8)[[1, 11]], v)
    first, second = decode.runs
    assert first["q"] is q and second["q"] is q
    # FlashInfer would zero-fill an output it allocated; forward hands it a fresh one.
    assert first["out"] is forward_out and forward_out.shape == q.shape
    assert forward_out.data_ptr() != q.data_ptr()
    assert second["out"] is out
    for native, original in zip(second["paged_kv_cache"], attention.group.layers[1], strict=True):
        assert native.shape == (16, 1, 2, 8)
        assert native.data_ptr() == original.data_ptr()

    # Fresh metadata plans again.
    backend.forward(q, k, v, 0, SimpleNamespace(attn_metadata=backend.prepare_metadata(rows)))
    assert attention.log[-6:] == ["kv_metadata", "synchronize", "plan", "record", "store", "run"]
    assert len(decode.plans) == 2


def test_flashinfer_attends_over_written_kv_without_storing(monkeypatch):
    attention = CPUAttention(monkeypatch)
    backend = attention.build()
    metadata = backend.prepare_metadata(forward_rows(*_DECODES))
    q, out = (torch.empty(2, 4, 8, dtype=torch.bfloat16) for _ in range(2))
    attention.log.clear()

    backend.attend_into(q, 0, metadata, out)
    backend.attend_into(q, 1, metadata, out)

    # The first layer plans once; no layer writes keys or values.
    assert attention.log == ["kv_metadata", "synchronize", "plan", "record", "run", "run"]
    assert all(run["q"] is q and run["out"] is out for run in attention.operations[1].runs)


def test_flashinfer_runs_a_kv_prologue_on_the_layer_cache_then_attends_its_q(monkeypatch):
    attention = CPUAttention(monkeypatch)
    backend = attention.build()
    _, decode = attention.operations
    batch = SimpleNamespace(attn_metadata=backend.prepare_metadata(forward_rows(*_DECODES)))
    q = torch.empty(2, 4, 8, dtype=torch.bfloat16)
    calls = []

    def prologue(*args):
        calls.append(args)
        return q

    first = backend.forward_prologue(prologue, 0, batch)
    second = backend.forward_prologue(prologue, 1, batch)
    assert first is decode.runs[0]["out"] and second is decode.runs[1]["out"]
    assert bool((first == 3.0).all() and (second == 3.0).all())
    # A given output buffer is where the run writes, and what the call returns.
    out = torch.zeros_like(q)
    assert backend.forward_prologue(prologue, 0, batch, out) is out
    assert decode.runs[-1]["out"] is out and bool((out == 3.0).all())
    assert "store" not in attention.log  # the prologue writes K and V

    for layer, (k_cache, v_cache, indices), run in zip((0, 1, 0), calls, decode.runs, strict=True):
        assert indices.tolist() == [1, 11]
        assert run.keys() == {"q", "paged_kv_cache", "out"}
        assert run["q"] is q and run["out"].shape == q.shape
        assert run["out"].data_ptr() != q.data_ptr()
        for view, native, original in zip(
            (k_cache, v_cache), run["paged_kv_cache"], attention.group.layers[layer], strict=True
        ):
            assert (view.shape, view.stride()) == ((16, 2, 8), (16, 8, 1))
            assert (native.shape, native.stride()) == ((16, 1, 2, 8), (16, 16, 8, 1))
            assert view.data_ptr() == native.data_ptr() == original.data_ptr()


@pytest.mark.parametrize(
    ("backend_type", "supported"),
    [
        (FlashInferBackend, True),
        (fa_module.FlashAttentionBackend, False),
        (trt_module.TensorRTLLMBackend, False),
        (HybridBackend, False),
    ],
)
def test_only_flashinfer_runs_kv_prologues(backend_type, supported):
    # The flag says whether the backend implements SupportsKVPrologue; no hook only raises.
    assert backend_type.SUPPORTS_KV_PROLOGUE is supported
    assert hasattr(backend_type, "forward_prologue") is supported


def test_flashinfer_decode_graphs_plan_their_own_operation_on_the_shared_workspace(monkeypatch):
    attention = CPUAttention(monkeypatch)
    backend = attention.build(tensor_cores=True)
    prefill, decode = attention.operations
    backend.init_capture_graph(max_seq_len=7, bs_list=[1, 2])
    # Scratch the provider left: a later borrower must not see it cleared.
    prefill.float_workspace.fill_(7)
    attention.log.clear()

    capture = backend.capture_metadata(forward_rows(*(decode_entry(0, 0),) * 2))

    graph = attention.operations[2]
    assert graph.float_workspace is prefill.float_workspace is decode.float_workspace
    assert graph.int_workspace is prefill.int_workspace is decode.int_workspace
    assert prefill.zeroed and not graph.zeroed
    options = graph.options
    assert (options["mode"], options["backend"], options["use_cuda_graph"]) == (
        "decode",
        "fa2",
        True,
    )
    assert (options["use_tensor_cores"], options["dispatch"]) == (True, "provider")
    # The graph borrows the KV indices' buffers: every request's tokens fit them.
    kv_indices = backend.kv_indices
    assert options["paged_kv_indptr_buffer"].data_ptr() == kv_indices.indptr.data_ptr()
    assert options["paged_kv_indptr_buffer"].shape == (3,)
    assert options["paged_kv_indices_buffer"] is kv_indices.indices
    assert kv_indices.indices.shape == (4 * 8,)
    # Page size 1 makes every last page length 1.
    assert options["paged_kv_last_page_len_buffer"].tolist() == [1, 1]
    assert attention.log == ["kv_metadata", "synchronize", "plan", "record"]

    # A real-size one batch replays the padded-size two capture: its metadata plans the
    # captured operation, whose static write slots it fills; running plans nothing more.
    replay = backend.prepare_metadata(forward_rows(decode_entry(1, 2), padding=1))
    backend.prepare_for_replay(SimpleNamespace(attn_metadata=replay), 2)
    assert capture.cache_write_indices.tolist() == [10, 0]
    # The replay's indices are built into the borrowed buffers: slot 1's three tokens, then
    # the padding row's one.
    assert options["paged_kv_indptr_buffer"].tolist() == [0, 3, 4]
    assert kv_indices.indices[:4].tolist() == [8, 9, 10, 0]
    q = torch.empty(2, 4, 8, dtype=torch.bfloat16)
    backend.attend_into(q, 0, replay, torch.empty_like(q))
    assert len(graph.plans) == 2 and len(graph.runs) == 1
    assert attention.log == ["kv_metadata", "synchronize", "plan", "record"] * 2 + ["run"]

    # Only a captured size's uniform decode plans a graph's operation.
    for rows, operation in (
        (forward_rows(decode_entry(1, 2)), decode),
        (forward_rows(extend_entry(0, 0, 2), extend_entry(1, 0, 2)), prefill),
    ):
        backend.attend_into(q, 0, backend.prepare_metadata(rows), torch.empty_like(q))
        assert len(operation.plans) == len(operation.runs) == 1
    assert len(attention.operations) == 3 and len(graph.plans) == 2

    # A replay whose metadata planned another operation is refused.
    uncaptured = backend.prepare_metadata(forward_rows(decode_entry(1, 2)))
    with pytest.raises(AssertionError):
        backend.prepare_for_replay(SimpleNamespace(attn_metadata=uncaptured), 2)


@pytest.mark.parametrize("kind", ["fa", "trt_context", "trt_decode"])
def test_paged_engine_stores_kv_before_tensor_only_provider_call(monkeypatch, kind):
    attention = CPUAttention(monkeypatch)
    received = []
    output = torch.empty(2, 4, 16)

    def numerical_call(**kwargs):
        attention.log.append("provider")
        received.append(kwargs)
        return output

    if kind == "fa":
        monkeypatch.setattr(fa_module, "flash_attention_with_kv_cache", numerical_call)
    else:
        selected = (
            "trtllm_context_attention" if kind == "trt_context" else "trtllm_decode_attention"
        )
        other = "trtllm_decode_attention" if kind == "trt_context" else "trtllm_context_attention"
        monkeypatch.setattr(trt_module, selected, numerical_call)
        monkeypatch.setattr(
            trt_module, other, lambda **_: pytest.fail("wrong physical attention route")
        )
    name = "fa" if kind == "fa" else "trtllm"
    backend = attention.build(name, heads=(4, 2, 16), page_size=16, num_pages=2)
    metadata = backend.prepare_metadata(forward_rows(*_DECODES))
    batch = SimpleNamespace(attn_metadata=metadata, ragged=kind == "trt_context", size=1)
    q = torch.empty(2, 4, 16, dtype=torch.bfloat16)
    k, v = (torch.randn(2, 2, 16, dtype=torch.bfloat16) for _ in range(2))
    k_cache, v_cache = attention.group.layers[1]
    attention.log.clear()

    assert backend.forward(q, k, v, 1, batch) is output

    assert attention.log == ["store", "provider"]
    # Slot 0's token 1 on page 0 and slot 1's token 3 on page 1.
    assert torch.equal(k_cache.view(-1, 2, 16)[[1, 19]], k)
    assert torch.equal(v_cache.view(-1, 2, 16)[[1, 19]], v)
    arguments = received[0]
    assert arguments["q"] is q
    assert arguments["page_table"].tolist() == [[0], [1]]
    assert arguments["cache_seqlens"].tolist() == [2, 4]
    assert arguments["softmax_scale"] == 0.25
    if kind == "fa":
        assert arguments["k_cache"].data_ptr() == k_cache.data_ptr()
        assert arguments["v_cache"].data_ptr() == v_cache.data_ptr()
        assert arguments["version"] == 4  # SM100
    else:
        assert [cache.data_ptr() for cache in arguments["kv_cache"]] == [
            k_cache.data_ptr(),
            v_cache.data_ptr(),
        ]
        assert arguments["workspace_buffer"].numel() == attention.workspace_bytes(
            "trtllm", heads=(4, 2, 16), page_size=16, max_running_req=4
        )
        assert arguments["max_seqlen_k"] == 4
        if kind == "trt_context":
            assert arguments["batch_size"] == 1  # real batch, not the padded bucket
    assert all(value is not batch and value is not metadata for value in arguments.values())
