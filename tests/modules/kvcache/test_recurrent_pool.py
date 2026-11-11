from __future__ import annotations

import pytest
import torch

from cti.kvcache import RecurrentRows, RecurrentStatePool
from cti.kvcache.state import PAYLOAD
from cti.layers.gated_delta_net.layer import gated_delta_net_state

HEAD_DIM = 128
NUM_VALUE_HEADS = 2
CONV_CHANNELS = 512
_SPEC = gated_delta_net_state(NUM_VALUE_HEADS, HEAD_DIM, HEAD_DIM, CONV_CHANNELS)


def test_state_spec_prices_the_qwen3_5_9b_request() -> None:
    spec = gated_delta_net_state(32, 128, 128, 8192)

    assert spec.tensors == (
        ("state", (32, 128, 128), torch.float32),
        ("conv", (3, 8192), torch.bfloat16),
    )
    assert spec.tensor_bytes == (2 * 1024 * 1024, 3 * 8192 * 2)
    assert spec.bytes_per_slot == 2 * 1024 * 1024 + 3 * 8192 * 2
    # 24 recurrent layers: 48 MiB of FP32 state plus 1.125 MiB of history.
    assert RecurrentRows((spec,) * 24).row_bytes == 48 * 2**20 + 24 * 3 * 8192 * 2


def test_a_pool_views_the_rows_a_caller_allocated() -> None:
    # The caller allocates the rows in a region a suspension releases, then builds the pool,
    # and its copy table, outside it.
    specs = (_SPEC, _SPEC)
    rows = RecurrentStatePool.allocate_rows(specs, 4, device=torch.device("cpu"))

    pool = RecurrentStatePool(
        specs, num_slots=3, num_snapshot_rows=1, device=torch.device("cpu"), rows=rows
    )

    for layer_index, buffer in enumerate(rows):
        state, conv_state = pool.layer_views(layer_index)
        assert state.data_ptr() == buffer.data_ptr()
        assert conv_state.untyped_storage().data_ptr() == buffer.data_ptr()


def test_pool_lays_out_stable_per_layer_views_over_one_buffer() -> None:
    specs = (_SPEC, _SPEC)
    pool = RecurrentStatePool(specs, num_slots=3, device=torch.device("cpu"))

    assert pool.num_layers == 2
    assert pool.num_slots == 3
    assert pool.specs == specs
    storages = set()
    for layer_index in range(2):
        state, conv_state = pool.layer_views(layer_index)
        assert pool.layer_views(layer_index)[0] is state
        assert state.shape == (3, NUM_VALUE_HEADS, HEAD_DIM, HEAD_DIM)
        assert state.dtype == torch.float32
        assert conv_state.shape == (3, 3, CONV_CHANNELS)
        assert conv_state.dtype == torch.bfloat16
        assert state.is_contiguous() and conv_state.is_contiguous()
        storage = state.untyped_storage()
        assert conv_state.untyped_storage().data_ptr() == storage.data_ptr()
        assert storage.nbytes() == 3 * specs[layer_index].bytes_per_slot
        storages.add(storage.data_ptr())
    assert len(storages) == 2
    assert (
        RecurrentRows(specs).allocation(3, 0, PAYLOAD).payload_bytes
        == 2 * 3 * specs[0].bytes_per_slot
    )


@pytest.mark.parametrize(
    ("specs", "num_slots", "message"),
    [
        pytest.param(0, 2, "at least one layer", id="no-layers"),
        pytest.param(1, 0, "num_slots", id="no-slots"),
    ],
)
def test_pool_rejects_inconsistent_construction(specs: int, num_slots: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RecurrentStatePool((_SPEC,) * specs, num_slots=num_slots, device=torch.device("cpu"))


def test_snapshot_rows_share_the_same_layer_allocations_and_exact_budget():
    pool = RecurrentStatePool(
        (_SPEC,) * 2, num_slots=3, num_snapshot_rows=5, device=torch.device("cpu")
    )
    assert pool.num_slots == 3 and pool.snapshot_rows == range(3, 8)
    total = 0
    for layer in range(2):
        state, history = pool.layer_views(layer)
        assert state.shape[0] == history.shape[0] == 8
        assert state.untyped_storage().data_ptr() == history.untyped_storage().data_ptr()
        assert state.dtype == torch.float32
        total += state.untyped_storage().nbytes()
    assert total == RecurrentRows(pool.specs).allocation(3, 5, PAYLOAD).payload_bytes
