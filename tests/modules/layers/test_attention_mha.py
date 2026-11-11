from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cti.attention import AttentionRequest, AttentionRequirement, resolve_attention_config
from cti.attention.base import BackendSetup
from cti.attention.flashinfer import FlashInferBackend
from cti.attention.mha import AttentionBackend, HybridBackend, ResolvedMha
from cti.kvcache import GroupRole
from cti.utils import DeviceCapability
from tests.support.attention import CPUAttention
from tests.support.forward_rows import decode_entry, forward_rows

# The capability the plan resolved for.
_CAPABILITY = DeviceCapability(9, 0)


def _load_backends(monkeypatch: pytest.MonkeyPatch, load: Callable[[str], object]) -> None:
    """Give every AttentionBackend the class ``load`` returns for its name."""

    monkeypatch.setattr(AttentionBackend, "backend_class", property(lambda self: load(self.value)))


def _resolved(
    ragged: str, decode: str | None = None, *, num_qo_heads: int = 4, num_kv_heads: int = 1
) -> ResolvedMha:
    """A resolved MHA plan serving 128-dim heads, ``num_qo_heads``/``num_kv_heads`` of
    them on this rank."""
    return ResolvedMha(
        ragged_backend=AttentionBackend(ragged),
        decode_backend=AttentionBackend(decode or ragged),
        reason="test plan",
        layouts={},
        head_dim=128,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=torch.bfloat16,
        capability=_CAPABILITY,
        tensor_cores=None,
    )


def _group() -> SimpleNamespace:
    """A KV group of two CPU layers."""
    return SimpleNamespace(layers=tuple(torch.empty(2, 1, 1, 1, 1) for _ in range(2)))


def _setup(
    *, num_qo_heads: int = 4, capability: DeviceCapability | None = _CAPABILITY
) -> BackendSetup:
    """The setup ``_resolved`` gives its backends for 9 running requests."""
    return BackendSetup(
        head_dim=128,
        num_qo_heads=num_qo_heads,
        num_kv_heads=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        capability=capability,
        max_running_req=9,
        tensor_cores=None,
    )


def _assert_constructed(call: tuple, setup: BackendSetup, group, page_table) -> None:
    """``call`` built a backend from ``setup`` over ``group``'s pages and ``page_table``."""
    args, kwargs = call
    assert args == (setup,) and kwargs.keys() == {"kv", "table"}
    assert kwargs["table"] is page_table
    assert [(k.data_ptr(), v.data_ptr()) for k, v in kwargs["kv"]] == [
        (layer[0].data_ptr(), layer[1].data_ptr()) for layer in group.layers
    ]


@pytest.mark.parametrize("name", ["fa", "trtllm"])
def test_paged_backend_capture_covers_partial_final_page(monkeypatch, name: str) -> None:
    backend = CPUAttention(monkeypatch).build(name, page_size=64, num_pages=2)

    backend.init_capture_graph(max_seq_len=66, bs_list=[1])

    assert backend.capture_metadata(forward_rows(decode_entry(0, 0))).page_table.shape == (1, 2)


@pytest.mark.parametrize(
    ("name", "module_name", "class_name"),
    [
        ("fa", "cti.attention.flash_attention", "FlashAttentionBackend"),
        ("fi", "cti.attention.flashinfer", "FlashInferBackend"),
        ("trtllm", "cti.attention.tensorrt_llm", "TensorRTLLMBackend"),
    ],
)
def test_factory_constructs_selected_single_backend(
    monkeypatch,
    name,
    module_name,
    class_name,
):
    group = _group()
    page_table = object()
    constructed = object()
    module = import_module(module_name)
    capability = DeviceCapability(10, 0)
    create = Mock(return_value=constructed)
    monkeypatch.setattr(module, class_name, create)

    backend = _resolved(name).build((group,), page_table, max_running_req=9, capability=capability)

    assert backend is constructed
    (call,) = create.call_args_list
    _assert_constructed(call, _setup(capability=capability), group, page_table)


def test_factory_preserves_hybrid_ragged_decode_order(monkeypatch):
    group = _group()
    page_table = object()
    created = []

    def load(name):
        def create(*args, **kwargs):
            backend = object()
            created.append((name, (args, kwargs), backend))
            return backend

        return create

    _load_backends(monkeypatch, load)

    backend = _resolved("fa", "fi").build((group,), page_table, max_running_req=9, capability=None)

    assert isinstance(backend, HybridBackend)
    assert [name for name, _, _ in created] == ["fa", "fi"]
    for _, call, _ in created:
        _assert_constructed(call, _setup(capability=None), group, page_table)
    assert backend.ragged_backend is created[0][2]
    assert backend.decode_backend is created[1][2]


@pytest.mark.parametrize(
    ("ragged", "decode", "expected"),
    [("fi", "fi", 3), ("fa", "fi", 4), ("trtllm", "trtllm", 5)],
    ids=["fi", "hybrid", "trtllm"],
)
def test_attention_reservation_counts_each_constructed_backend_once(
    monkeypatch, ragged: str, decode: str, expected: int
) -> None:
    calls = []
    reservations = {"fa": 1, "fi": 3, "trtllm": 5}

    def load(name):
        def workspace_bytes(setup):
            calls.append((name, setup))
            return reservations[name]

        return SimpleNamespace(workspace_bytes=workspace_bytes)

    _load_backends(monkeypatch, load)
    device = torch.device("cpu")

    reserved = _resolved(ragged, decode, num_qo_heads=2).workspace_bytes(
        device=device, max_running_req=9
    )

    assert reserved == expected
    # The setup construction takes.
    assert calls == [(name, _setup(num_qo_heads=2)) for name in dict.fromkeys((ragged, decode))]


def test_flashinfer_initializes_shared_workspace_once(monkeypatch) -> None:
    attention = CPUAttention(monkeypatch)

    attention.build(max_running_req=256)

    prefill, decode = attention.operations
    assert [prefill.options["mode"], decode.options["mode"]] == ["prefill", "decode"]
    # Both borrow one workspace, zeroed before the first.
    assert decode.float_workspace is prefill.float_workspace
    assert decode.int_workspace is prefill.int_workspace
    assert prefill.zeroed
    # The KV budget reserves exactly what construction allocates.
    allocated = prefill.float_workspace.nbytes + prefill.int_workspace.nbytes
    assert allocated == attention.workspace_bytes(max_running_req=256)


@pytest.mark.parametrize(
    ("ragged", "expected"),
    ((True, "ragged"), (False, "decode")),
)
def test_hybrid_backend_routes_by_physical_attention_shape(
    ragged: bool,
    expected: str,
) -> None:
    calls: list[tuple[str, str, tuple[object, ...]]] = []

    class _Backend:
        def __init__(self, name: str) -> None:
            self.name = name

        def prepare_metadata(self, *args: object) -> str:
            calls.append((self.name, "prepare_metadata", args))
            return f"{self.name} metadata"

        def forward(self, *args: object) -> str:
            calls.append((self.name, "forward", args))
            return self.name

        def forward_into(self, *args: object) -> None:
            calls.append((self.name, "forward_into", args))

    backend = HybridBackend(_Backend("ragged"), _Backend("decode"))
    rows = SimpleNamespace(uses_ragged_attention=ragged)
    batch = SimpleNamespace(ragged=ragged)
    q, k, v, out = (object() for _ in range(4))

    assert backend.prepare_metadata(rows) == f"{expected} metadata"  # type: ignore[arg-type]
    assert backend.forward(q, k, v, 3, batch) == expected  # type: ignore[arg-type]
    backend.forward_into(q, k, v, 3, batch, out)  # type: ignore[arg-type]

    assert calls == [
        (expected, "prepare_metadata", (rows,)),
        (expected, "forward", (q, k, v, 3, batch)),
        (expected, "forward_into", (q, k, v, 3, batch, out)),
    ]


def test_normalized_duplicate_constructs_one_backend(monkeypatch):
    monkeypatch.setattr(
        FlashInferBackend,
        "validate_environment",
        classmethod(lambda cls, capability: ()),
    )
    resolved = resolve_attention_config(
        {GroupRole.TARGET: (AttentionRequirement.mha(4, 1, 128),)},
        AttentionRequest(
            target=GroupRole.TARGET,
            backend="fi,fi",
            page_size=1,
            capability=DeviceCapability(12, 0),
            tensor_parallel_size=1,
            dtype=torch.bfloat16,
        ),
    )
    calls = []
    constructed = object()

    def load(name):
        calls.append(name)
        return Mock(return_value=constructed, SUPPORTS_TENSOR_PARALLEL=True)

    _load_backends(monkeypatch, load)

    backend = resolved.build((_group(),), object(), max_running_req=9, capability=None)

    assert resolved.selected == "fi"
    assert backend is constructed
    assert calls == ["fi"]
