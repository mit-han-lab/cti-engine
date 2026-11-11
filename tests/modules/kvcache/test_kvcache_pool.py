from __future__ import annotations

import pytest
import torch

from cti.attention.paged import KVPair, kv_pages, store_kv_pair
from cti.kvcache import (
    GroupLayout,
    GroupRole,
    InferenceState,
    KVLayout,
    RecurrentRows,
    Spec,
    TableGeometry,
)

_SPEC = Spec(KVPair(num_kv_heads=2, head_dim=4, dtype=torch.float32))
_LAYOUT = KVLayout((GroupLayout(GroupRole.TARGET, 0, _SPEC, 3),), RecurrentRows(()))
# 3 usable pages of 2 tokens, plus the graph sink page.
_CAPACITY = TableGeometry(2, 1, 64).capacity(_LAYOUT, 2, 0)


def _group():
    state = InferenceState.allocate(_LAYOUT, _CAPACITY, device=torch.device("cpu"))
    (group,) = state.groups
    return group


def test_kv_pair_group_stripes_stable_views_by_layer() -> None:
    group = _group()

    assert (group.role, group.index, group.spec) == (GroupRole.TARGET, 0, _SPEC)
    layer_storage_ptrs = set()
    for layer, (k_cache, v_cache) in zip(group.layers, kv_pages(group), strict=True):
        assert layer.shape == (2, 3, 2, 2, 4)
        assert layer.device == torch.device("cpu") and layer.dtype == torch.float32
        k_storage = k_cache.untyped_storage()
        assert k_cache.shape == v_cache.shape == (3, 2, 2, 4)
        assert k_cache.stride() == v_cache.stride() == (16, 8, 4, 1)
        assert k_cache.is_contiguous() and v_cache.is_contiguous()
        assert k_storage.data_ptr() == v_cache.untyped_storage().data_ptr()
        assert k_storage.nbytes() == 2 * k_cache.nbytes
        assert (k_cache.storage_offset(), v_cache.storage_offset()) == (0, k_cache.numel())
        layer_storage_ptrs.add(k_storage.data_ptr())

    assert len(layer_storage_ptrs) == 3
    # One page's K and V of one layer, as allocated.
    assert _SPEC.page_bytes(2) == 2 * group.layers[0][0][0].nbytes
    assert _LAYOUT.bytes_per_page(2) == 3 * _SPEC.page_bytes(2)


def test_store_kv_pair_writes_the_selected_layer_views(monkeypatch) -> None:
    group = _group()
    recorded: dict[str, torch.Tensor] = {}

    def record_store(**arguments: torch.Tensor) -> None:
        recorded.update(arguments)

    monkeypatch.setattr("ctikernels.store_cache", record_store)
    k = torch.empty((2, 2, 4))
    v = torch.empty((2, 2, 4))
    indices = torch.tensor([1, 4], dtype=torch.int32)

    store_kv_pair(kv_pages(group)[1], k, v, indices)

    assert recorded["k_cache"].shape == recorded["v_cache"].shape == (6, 2, 4)
    assert recorded["k_cache"].data_ptr() == group.layers[1][0].data_ptr()
    assert recorded["v_cache"].data_ptr() == group.layers[1][1].data_ptr()
    assert recorded["indices"] is indices
    assert recorded["k"] is k
    assert recorded["v"] is v


def test_spec_entries_divide_the_page() -> None:
    pair = KVPair(1, 8, torch.bfloat16)
    assert Spec(pair, tokens_per_entry=4).entries(16) == 4
    assert Spec(pair, tokens_per_entry=4).page_bytes(16) == pair.page_bytes(4) == 2 * 4 * 8 * 2
    with pytest.raises(ValueError, match="not a multiple of 3 tokens"):
        Spec(pair, tokens_per_entry=3).entries(16)


def test_one_row_layout_with_other_entries_or_windows_is_another_spec() -> None:
    pair = KVPair(1, 8, torch.bfloat16)
    specs = {Spec(pair), Spec(pair, tokens_per_entry=2), Spec(pair, window=128)}
    assert len(specs) == 3
    assert Spec(pair) == Spec(KVPair(1, 8, torch.bfloat16))


def test_kv_layout_indexes_groups_in_order() -> None:
    with pytest.raises(ValueError, match="indexed 0..n-1"):
        KVLayout((GroupLayout(GroupRole.DRAFT, 1, _SPEC, 1),), RecurrentRows(()))
