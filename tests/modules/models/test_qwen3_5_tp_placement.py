"""Qwen3.5 checkpoint windows at the published head geometries: every rank's shards rebuild
the single-rank parameters, and its Gated DeltaNet rows are exactly its heads in kernel order."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import safetensors.torch
import torch
import transformers
from transformers import PretrainedConfig

from cti.config import LoRAConfig
from cti.distributed import Topology
from cti.layers import Adapter
from cti.layers.gated_delta_net.layer import gated_delta_net_state
from cti.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from tests.support.qwen3_5 import QWEN3_5_9B_CONFIG, QWEN3_8_27B_CONFIG, load_hf_config
from tests.support.tp_shards import build_model, load_sharded, rebuild_parameter

# Gated DeltaNet key and value heads are 128 wide.
_HEAD_DIM = 128
# One block of the published schedule: three Gated DeltaNet layers, then full attention.
_LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]


def _small(published: Mapping[str, Any], **text: object) -> dict[str, Any]:
    """The published head geometry with a small hidden size, vocabulary and depth, and a
    one-block vision tower."""

    return {
        **published,
        "text_config": {
            **published["text_config"],
            "hidden_size": 32,
            "intermediate_size": 64,
            "vocab_size": 259,
            "num_hidden_layers": len(_LAYER_TYPES),
            "layer_types": _LAYER_TYPES,
            "max_position_embeddings": 64,
            **text,
        },
        "vision_config": {
            **published["vision_config"],
            "depth": 1,
            "hidden_size": 16,
            "num_heads": 2,
            "intermediate_size": 16,
            "out_hidden_size": 32,
            "num_position_embeddings": 4,
        },
    }


def _rows(heads: range) -> slice:
    """The rows of ``heads`` in a head-major tensor of 128-wide heads."""

    return slice(heads.start * _HEAD_DIM, heads.stop * _HEAD_DIM)


@pytest.fixture(
    scope="module",
    params=[QWEN3_5_9B_CONFIG, QWEN3_8_27B_CONFIG],
    ids=["9b-heads", "27b-heads"],
)
def checkpoint(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[PretrainedConfig, Path, dict[str, torch.Tensor]]:
    """An image-text checkpoint with Hugging Face names and seeded random tensors.

    Random values make a misplaced window fail even after the convolution is rounded to
    its BF16 declaration. Every rank loads the vision tower whole; one
    multi-token-prediction tensor must be skipped.
    """

    folder = tmp_path_factory.mktemp("qwen3_5-tp")
    hf_config = load_hf_config(_small(request.param), folder)
    with torch.device("meta"):
        shapes = {
            name: tensor.shape
            for name, tensor in transformers.Qwen3_5ForConditionalGeneration(hf_config)
            .state_dict()
            .items()
        }
    generator = torch.Generator().manual_seed(0)
    tensors = {name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}
    assert any(name.startswith("model.language_model.") for name in tensors)
    assert any(name.startswith("model.visual.") for name in tensors)
    extra = {"mtp.fc.weight": torch.ones(2)}
    safetensors.torch.save_file(tensors | extra, folder / "model.safetensors")
    return hf_config, folder, tensors


@pytest.mark.parametrize("size", [2, 4, 8])
def test_every_rank_shard_rebuilds_the_single_rank_parameters(
    checkpoint: tuple[PretrainedConfig, Path, dict[str, torch.Tensor]],
    stub_flashinfer: None,
    size: int,
) -> None:
    hf_config, folder, _ = checkpoint
    _, reference, reference_shards = load_sharded(hf_config, folder)
    assert reference_shards == {}

    ranks = [
        load_sharded(hf_config, folder, topology=Topology(rank, size))[1:] for rank in range(size)
    ]

    for local, shards in ranks:
        assert local.keys() == reference.keys()
        # A window always names a parameter its rank loads.
        assert shards.keys() <= local.keys()
    for name, tensor in reference.items():
        torch.testing.assert_close(
            rebuild_parameter(name, tensor, ranks), tensor, rtol=0, atol=0, msg=name
        )


@pytest.mark.parametrize("size", [2, 4, 8])
def test_each_rank_holds_its_own_gated_delta_net_heads_in_kernel_order(
    checkpoint: tuple[PretrainedConfig, Path, dict[str, torch.Tensor]],
    stub_flashinfer: None,
    size: int,
) -> None:
    hf_config, folder, tensors = checkpoint
    text = hf_config.text_config
    num_key_heads, num_value_heads = text.linear_num_key_heads, text.linear_num_value_heads
    key_size = num_key_heads * _HEAD_DIM
    group = num_value_heads // num_key_heads

    for rank in range(size):
        model, loaded, _ = load_sharded(hf_config, folder, topology=Topology(rank, size))
        keys = range(rank * num_key_heads // size, (rank + 1) * num_key_heads // size)
        values = range(rank * num_value_heads // size, (rank + 1) * num_value_heads // size)
        # Value head j reads key head j // (Hv / Hk): each rank owns every key head its
        # value heads read.
        assert {head // group for head in values} == set(keys)
        recurrent = model.iter_recurrent_layers()
        assert {(layer.num_key_heads, layer.num_value_heads) for layer in recurrent} == {
            (len(keys), len(values))
        }
        # The runner sizes this rank's recurrent-state pool from these specs.
        specs = tuple(layer.state_spec for layer in recurrent)
        channels = (2 * len(keys) + len(values)) * _HEAD_DIM
        assert set(specs) == {gated_delta_net_state(len(values), _HEAD_DIM, _HEAD_DIM, channels)}
        for index in range(len(recurrent)):
            hf = f"model.language_model.layers.{index}.linear_attn."
            engine = f"model.layers.{index}.linear_attn."
            key_rows, value_rows = _rows(keys), _rows(values)
            # This rank's q | k | v rows of in_proj_qkv, which are also its conv1d channels.
            channels = (
                key_rows,
                slice(key_size + key_rows.start, key_size + key_rows.stop),
                slice(2 * key_size + value_rows.start, 2 * key_size + value_rows.stop),
            )
            qkv = tensors[hf + "in_proj_qkv.weight"]
            conv = tensors[hf + "conv1d.weight"]

            expected_in_proj = torch.cat(
                [
                    *(qkv[part] for part in channels),
                    tensors[hf + "in_proj_z.weight"][value_rows],
                    tensors[hf + "in_proj_b.weight"][values.start : values.stop],
                    tensors[hf + "in_proj_a.weight"][values.start : values.stop],
                ]
            )
            expected_conv = torch.cat([conv[part] for part in channels]).bfloat16()
            assert torch.equal(loaded[engine + "in_proj.weight"], expected_in_proj)
            assert torch.equal(loaded[engine + "gdn.conv_weight"], expected_conv)
            for name in ("A_log", "dt_bias"):
                expected = tensors[hf + name][values.start : values.stop]
                assert torch.equal(loaded[engine + "gdn." + name], expected)
            assert torch.equal(loaded[engine + "gdn.norm_weight"], tensors[hf + "norm.weight"])
            assert torch.equal(
                loaded[engine + "out_proj.weight"], tensors[hf + "out_proj.weight"][:, value_rows]
            )


def test_key_heads_the_tensor_parallel_size_does_not_divide_fail_at_construction(
    tmp_path: Path,
    stub_flashinfer: None,
) -> None:
    published = _small(QWEN3_5_9B_CONFIG, linear_num_key_heads=6, linear_num_value_heads=12)
    hf_config = load_hf_config(published, tmp_path)

    with pytest.raises(
        ValueError,
        match=r"key heads \(6\) must be divisible by the tensor-parallel size \(4\)",
    ):
        build_model(hf_config, topology=Topology(0, 4))


def test_lora_hosts_keep_their_base_params(
    tmp_path: Path,
    stub_flashinfer: None,
) -> None:
    hf_config = load_hf_config(_small(QWEN3_5_9B_CONFIG), tmp_path)
    topology = Topology(1, 2)
    plain = build_model(hf_config, topology=topology).params()
    adapters = Adapter(LoRAConfig(targets=Qwen3_5ForConditionalGeneration.LORA_TARGETS))

    # Including in_proj, whose shared-A adapter spans three of its six windows.
    assert build_model(hf_config, topology=topology, lora_adapters=adapters).params() == plain
    # Two mixer projections and two MLP projections in each layer.
    assert len(adapters.modules()) == 4 * len(_LAYER_TYPES)
