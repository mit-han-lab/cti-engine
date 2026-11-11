"""Layer-owned checkpoint windows: every rank's shards rebuild the TP1 parameters exactly."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import safetensors.torch
import torch
from transformers import Qwen3Config

from cti.config import LoRAConfig
from cti.distributed import Topology
from cti.layers import (
    Adapter,
    LinearColParallelMerged,
    LinearQKVMerged,
    LinearRowParallel,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from cti.layers.placement import WeightShard
from tests.support.tp_shards import (
    assert_tiles,
    build_model,
    load_sharded,
    qwen3_checkpoint_shapes,
    rebuild_parameter,
    site,
    windows,
)

pytestmark = pytest.mark.usefixtures("stub_flashinfer")


def _config(**overrides: object) -> Qwen3Config:
    geometry = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "hidden_size": 32,
        "vocab_size": 259,
        "intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "architectures": ["Qwen3ForCausalLM"],
    }
    return Qwen3Config(**{**geometry, **overrides})


def _published_config(
    *, hidden_size: int, num_attention_heads: int, intermediate_size: int
) -> Qwen3Config:
    """A one-layer Qwen3 geometry of a published size; layers repeat it."""

    return _config(
        num_hidden_layers=1,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=hidden_size,
        vocab_size=151936,
        intermediate_size=intermediate_size,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
    )


def _write_checkpoint(config: Qwen3Config, folder: Path) -> None:
    """Write Hugging Face names whose every element is a distinct FP32-exact integer."""

    tensors = {}
    offset = 0
    for name, shape in qwen3_checkpoint_shapes(config).items():
        count = math.prod(shape)
        tensors[name] = torch.arange(offset, offset + count, dtype=torch.float32).view(shape)
        offset += count
    assert offset < 2**24, "every value must be exact in FP32"
    safetensors.torch.save_file(tensors, folder / "model.safetensors")


@pytest.mark.parametrize("tie_word_embeddings", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("vocab_size", [259, 256])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4, 8])
def test_every_rank_shard_rebuilds_the_single_rank_parameters(
    tmp_path: Path,
    num_kv_heads: int,
    vocab_size: int,
    tie_word_embeddings: bool,
) -> None:
    config = _config(
        num_key_value_heads=num_kv_heads,
        vocab_size=vocab_size,
        tie_word_embeddings=tie_word_embeddings,
    )
    _write_checkpoint(config, tmp_path)
    _, reference, reference_shards = load_sharded(config, tmp_path)
    assert reference_shards == {}

    for size in (2, 4, 8):
        ranks = [
            load_sharded(config, tmp_path, topology=Topology(rank, size))[1:]
            for rank in range(size)
        ]
        assert all(local.keys() == reference.keys() for local, _ in ranks)
        # A window always names a parameter its rank loads.
        assert all(shards.keys() <= local.keys() for local, shards in ranks)
        for name, tensor in reference.items():
            torch.testing.assert_close(
                rebuild_parameter(name, tensor, ranks),
                tensor,
                rtol=0,
                atol=0,
                msg=f"{name} at TP{size}",
            )


@pytest.mark.parametrize(
    "config",
    [
        _published_config(hidden_size=4096, num_attention_heads=32, intermediate_size=12288),
        _published_config(hidden_size=5120, num_attention_heads=64, intermediate_size=25600),
    ],
    ids=["qwen3-8b", "qwen3-32b"],
)
@pytest.mark.parametrize("size", [2, 4, 8])
def test_published_geometry_windows_tile_every_parameter(config: Qwen3Config, size: int) -> None:
    full = {param.path: param.shape for param in build_model(config).params()}
    ranks = [
        {param.path: param for param in build_model(config, topology=Topology(rank, size)).params()}
        for rank in range(size)
    ]

    for name, full_shape in full.items():
        params = [rank[name] for rank in ranks]
        if not params[0].windows:
            assert all(param.shape == full_shape for param in params)
            continue
        axis = params[0].windows[0].axis
        slot_sizes = []
        for slot in range(len(params[0].windows)):
            slot_windows = [param.windows[slot] for param in params]
            slot_sizes.append(max(window.stop for window in slot_windows))
            assert_tiles({(window.start, window.stop) for window in slot_windows}, slot_sizes[-1])
        assert sum(slot_sizes) == full_shape[axis]
        for param in params:
            local_shape = list(full_shape)
            local_shape[axis] = sum(window.local_size for window in param.windows)
            assert list(param.shape) == local_shape, name


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4, 8])
def test_each_rank_keeps_the_kv_heads_its_query_heads_read(num_kv_heads: int, size: int) -> None:
    config = _config(num_key_value_heads=num_kv_heads)
    group_size = config.num_attention_heads // num_kv_heads
    for rank in range(size):
        model = build_model(config, topology=Topology(rank, size))
        q, k, v = windows(model)["model.layers.0.self_attn.qkv_proj.weight"]
        attention = next(model.iter_attention_layers())

        query_heads = range(q.start // config.head_dim, q.stop // config.head_dim)
        kv_heads = range(k.start // config.head_dim, k.stop // config.head_dim)
        assert k == v
        assert len(query_heads) == attention.num_qo_heads
        assert len(kv_heads) == attention.num_kv_heads
        assert {head // group_size for head in query_heads} == set(kv_heads)


@pytest.mark.parametrize("size", [2, 4, 8])
def test_row_parallel_inputs_are_the_rank_local_column_outputs(size: int) -> None:
    for rank in range(size):
        shards = windows(build_model(_config(), topology=Topology(rank, size)))
        prefix = "model.layers.1"
        q, _, _ = shards[f"{prefix}.self_attn.qkv_proj.weight"]
        (o_proj,) = shards[f"{prefix}.self_attn.o_proj.weight"]
        gate, up = shards[f"{prefix}.mlp.gate_up_proj.weight"]
        (down_proj,) = shards[f"{prefix}.mlp.down_proj.weight"]

        assert (o_proj.axis, o_proj.start, o_proj.stop) == (1, q.start, q.stop)
        assert gate == up
        assert (down_proj.axis, down_proj.start, down_proj.stop) == (1, gate.start, gate.stop)


def test_bias_follows_column_windows_and_stays_whole_after_a_row_reduction() -> None:
    at = site(Topology(1, 2))
    with torch.device("meta"):
        column = LinearColParallelMerged(4, [6, 2], has_bias=True, site=at)
        qkv = LinearQKVMerged(4, 2, 4, 2, has_bias=True, site=at)
        rows = LinearRowParallel(8, 4, has_bias=True, site=at)
        head = ParallelLMHead(7, 4, bias=True, site=at)

    for layer in (column, qkv):
        shards = windows(layer)
        assert shards["bias"] == shards["weight"]
    assert windows(rows) == {"weight": (WeightShard(1, 4, 8),)}
    head_shards = windows(head)
    assert head_shards["bias"] == head_shards["weight"] == (WeightShard(0, 4, 7, padded_size=4),)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"num_attention_heads": 6}, ValueError),
        ({"num_attention_heads": 12, "num_key_value_heads": 3}, ValueError),
        ({"intermediate_size": 63}, AssertionError),
    ],
    ids=["query-heads", "kv-heads", "intermediate"],
)
def test_indivisible_geometry_fails_at_model_construction(
    overrides: dict[str, int], error: type[Exception]
) -> None:
    with pytest.raises(error, match="divisible"):
        build_model(_config(**overrides), topology=Topology(0, 4))


def test_lora_hosts_keep_their_base_params() -> None:
    topology = Topology(1, 2)
    plain = build_model(_config(), topology=topology).params()
    adapters = Adapter(LoRAConfig(targets=("attention", "mlp")))

    assert build_model(_config(), topology=topology, lora_adapters=adapters).params() == plain
    assert len(adapters.modules()) == 4 * _config().num_hidden_layers


def test_tied_head_declares_no_window_of_a_redundant_checkpoint_copy(tmp_path: Path) -> None:
    # Qwen3-0.6B and 1.7B tie the head and still ship lm_head.weight.
    config = _config(tie_word_embeddings=True)
    _write_checkpoint(config, tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = safetensors.torch.load_file(path)
    tensors["lm_head.weight"] = torch.zeros(config.vocab_size, config.hidden_size)
    safetensors.torch.save_file(tensors, path)

    _, loaded, shards = load_sharded(config, tmp_path, topology=Topology(1, 2))

    # The head reads the embedding's rows; the mapping skips the copy unread.
    assert "model.embed_tokens.weight" in shards
    assert "lm_head.weight" not in shards
    assert "lm_head.weight" not in loaded


def test_empty_vocabulary_rank_reads_a_zero_padded_window() -> None:
    embedding = VocabParallelEmbedding(5, 32, site=site(Topology(3, 4)))
    (window,) = windows(embedding)["weight"]

    assert embedding.vocab_range == (5, 0)
    assert window == WeightShard(0, 5, 5, padded_size=2)
    selected = window.select(torch.ones(5, 32))
    assert selected.shape == embedding.weight.shape == (2, 32)
    assert not selected.count_nonzero()


def test_weight_window_is_a_view_unless_it_pads() -> None:
    checkpoint = torch.arange(24, dtype=torch.float32).view(4, 6)

    rows = WeightShard(0, 1, 3).select(checkpoint)
    columns = WeightShard(1, 2, 5).select(checkpoint)
    padded = WeightShard(0, 2, 4, padded_size=3).select(checkpoint)

    assert rows.untyped_storage().data_ptr() == checkpoint.untyped_storage().data_ptr()
    assert columns.untyped_storage().data_ptr() == checkpoint.untyped_storage().data_ptr()
    assert torch.equal(rows, checkpoint[1:3])
    assert torch.equal(columns, checkpoint[:, 2:5])
    assert torch.equal(padded, torch.cat((checkpoint[2:4], torch.zeros(1, 6))))


@pytest.mark.parametrize(
    "arguments",
    [(-1, 0, 1), (0, 2, 1), (0, -1, 1), (0, 0, 4, 3)],
    ids=["axis", "reversed", "negative", "truncating-padding"],
)
def test_weight_window_rejects_invalid_coordinates(arguments: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="checkpoint shard"):
        WeightShard(*arguments)


def test_weight_window_rejects_a_tensor_that_does_not_contain_it() -> None:
    with pytest.raises(ValueError, match=r"does not contain its window \[2, 5\) on axis 1"):
        WeightShard(1, 2, 5).select(torch.ones(4, 4))
