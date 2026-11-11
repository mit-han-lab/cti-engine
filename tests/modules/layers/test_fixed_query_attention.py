from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cti.attention import fixed_query as fixed_query_module
from cti.attention.fixed_query import (
    FixedQueryBackend,
    FixedQueryGeometry,
    FixedQueryShape,
    RoundAddresses,
    fixed_query_bytes,
)

_SHAPE = FixedQueryShape(2, 4, 40, ((False, None), (True, 16)), torch.bfloat16)


def _addresses(max_kv_tokens: int = 40) -> RoundAddresses:
    """A round's addresses: two rows of four queries over the pools' KV bound."""

    shape = FixedQueryShape(2, 4, max_kv_tokens, _SHAPE.masks, torch.bfloat16)
    table = torch.zeros(2, 32, dtype=torch.int32)
    return RoundAddresses(table, shape, max_seq_len=32, guard_start=64)


_ADDRESSES = _addresses()


def _geometry(heads: int) -> FixedQueryGeometry:
    return FixedQueryGeometry(64, heads, 1, _SHAPE)


def _group(num_tokens: int = 1, num_layers: int = 1) -> SimpleNamespace:
    """A KV group of ``num_layers`` zeroed one-head layers of ``num_tokens`` token pages."""
    return SimpleNamespace(layers=(torch.zeros(2, num_tokens, 1, 1, 64),) * num_layers)


def test_operations_share_one_float_scratch_and_keep_their_own_integer_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def workspace_size(*, batch_size, num_qo_heads, causal, max_kv_tokens, **_):
        assert max_kv_tokens == 40
        return 1000 * num_qo_heads * batch_size, 10 * batch_size + causal

    monkeypatch.setattr(
        fixed_query_module.DraftAttention, "workspace_size", staticmethod(workspace_size)
    )
    geometries = (_geometry(2), _geometry(4))

    # Two pools x (2 sizes x 2 masks) operations; each mask keeps its CSR of the largest batch.
    csr = 4 * (2 * 3 + 2 * 40 + 2) * 2
    ints = 2 * (10 + 11 + 20 + 21) + 2 * csr
    assert fixed_query_bytes(geometries, torch.device("cpu")) == 1000 * 4 * 2 + ints


def test_other_batches_attend_through_the_paged_backend_and_a_draft_pool_has_none() -> None:
    group = _group()
    paged = Mock()
    backend = FixedQueryBackend(
        _geometry(2),
        group=group,
        float_workspace=torch.empty(0),
        paged=paged,
        addresses=_ADDRESSES,
    )
    batch = SimpleNamespace(attn_metadata=object())
    q = torch.zeros(1, 2, 64)

    backend.forward_into(q, q, q, 3, batch, q)
    paged.forward_into.assert_called_once_with(q, q, q, 3, batch, q)
    rows = object()
    assert backend.prepare_metadata(rows) is paged.prepare_metadata.return_value
    paged.prepare_metadata.assert_called_once_with(rows)

    draft = FixedQueryBackend(
        _geometry(2),
        group=group,
        float_workspace=torch.empty(0),
        paged=None,
        addresses=_ADDRESSES,
    )
    with pytest.raises(RuntimeError, match="only in rounds"):
        draft.forward(q, q, q, 0, batch)


def test_an_operation_outside_the_pool_shape_is_refused() -> None:
    def backend(max_kv_tokens: int) -> FixedQueryBackend:
        return FixedQueryBackend(
            _geometry(2),
            group=_group(),
            float_workspace=torch.empty(0),
            paged=None,
            addresses=_addresses(max_kv_tokens),
        )

    rows = torch.zeros(2, dtype=torch.int32)
    positions = torch.zeros(8, dtype=torch.int64)

    with pytest.raises(ValueError, match="outside this pool's shape"):
        backend(41).create_fixed_query(rows, positions, causal=True, window=8)
    with pytest.raises(ValueError, match="outside this pool's shape"):
        backend(40).create_fixed_query(rows, positions, causal=True, window=9)


@pytest.mark.parametrize("path", ["forward", "prologue"])
def test_round_queries_reach_the_operation_as_dense_rows(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    from cti.attention.fixed_query import FixedQueryMetadata

    monkeypatch.setattr(fixed_query_module, "store_kv_pair", lambda *_args: None)
    seen = []

    def run(*, q, paged_kv_cache, out):
        seen.append((q.is_contiguous(), out))
        out.fill_(5.0)

    operation = SimpleNamespace(run=run)
    metadata = FixedQueryMetadata(
        torch.arange(4), operation, torch.zeros(3, dtype=torch.int32), torch.empty(0)
    )
    backend = FixedQueryBackend(
        _geometry(2),
        group=_group(8),
        float_workspace=torch.empty(0),
        paged=None,
        addresses=_ADDRESSES,
    )
    batch = SimpleNamespace(attn_metadata=metadata)
    # Q as a fused QKV projection or prologue leaves it: a view with a wider row stride.
    q = torch.zeros(4, 4 * 64)[:, : 2 * 64].view(4, 2, 64)
    assert not q.is_contiguous()

    if path == "forward":
        result = backend.forward(q, q, q, 0, batch)
    else:
        out = torch.zeros(4, 2, 64)
        result = backend.forward_prologue(lambda *_args: q, 0, batch, out)
        assert result is out
    # The call returns the buffer the operation wrote.
    ((contiguous, written),) = seen
    assert contiguous and result is written and bool((result == 5.0).all())


@pytest.mark.parametrize("path", ["forward", "prologue"])
def test_a_mixed_query_writes_every_row_once_then_attends_each_group(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    from cti.attention.fixed_query import FixedQueryMetadata, MixedQuery

    calls = []

    def run(*, q, paged_kv_cache, out):
        calls.append(("fixed", q.shape[0], q.is_contiguous()))
        out.fill_(2.0)

    paged = Mock()
    paged.attend_into.side_effect = lambda q, layer, metadata, out: (
        calls.append(("paged", q.shape[0], layer, metadata)),
        out.fill_(1.0),
    )
    rows = torch.arange(11)
    fixed = FixedQueryMetadata(
        rows[3:], SimpleNamespace(run=run), torch.zeros(3, dtype=torch.int32), torch.empty(0)
    )
    planned = SimpleNamespace(cache_write_indices=rows[:3])
    metadata = MixedQuery(planned, fixed, paged_tokens=3)
    # The joint rows write the leading group's slots, then the fixed rows'.
    rows = metadata.cache_write_indices
    assert rows.tolist() == list(range(11))
    stored = []
    monkeypatch.setattr(
        fixed_query_module,
        "store_kv_pair",
        lambda pages, k, v, rows: stored.append((rows, pages)),
    )
    backend = FixedQueryBackend(
        _geometry(2),
        group=_group(16, 6),
        float_workspace=torch.empty(0),
        paged=paged,
        addresses=_ADDRESSES,
    )
    batch = SimpleNamespace(attn_metadata=metadata)
    # 3 prompt rows, then a block of 8: Q as a fused projection leaves it, strided.
    q = torch.zeros(11, 4 * 64)[:, : 2 * 64].view(11, 2, 64)

    if path == "forward":
        result = backend.forward(q, q, q, 5, batch)
        assert stored == [(rows, backend.kv[5])]
    else:
        prologues = []
        result = backend.forward_prologue(lambda *args: prologues.append(args[2]) or q, 5, batch)
        assert len(prologues) == 1 and prologues[0] is rows and stored == []
    assert calls == [("paged", 3, 5, planned), ("fixed", 8, True)]
    assert bool((result[:3] == 1.0).all()) and bool((result[3:] == 2.0).all())
