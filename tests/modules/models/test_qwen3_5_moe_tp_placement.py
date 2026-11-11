"""Qwen3.5-MoE checkpoint windows at the published 35B-A3B head and expert geometry: every
rank's shards rebuild the single-rank parameters, each rank loading its own experts whole, a
window of the shared expert and the whole vision tower."""

from __future__ import annotations

from pathlib import Path

import pytest
import safetensors.torch
import torch
from transformers import AutoModelForImageTextToText, PretrainedConfig

from cti.distributed import Topology
from tests.support.moe import ReferenceMoe
from tests.support.qwen3_5 import load_hf_config
from tests.support.qwen3_5_moe import QWEN3_5_35B_A3B_CONFIG
from tests.support.tp_shards import load_sharded, rebuild_parameter

_TEXT = {
    **QWEN3_5_35B_A3B_CONFIG["text_config"],
    "hidden_size": 32,
    "vocab_size": 259,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "moe_intermediate_size": 8,
    "shared_expert_intermediate_size": 16,
    "max_position_embeddings": 64,
}
# A one-block vision tower that projects into the small text model.
_VISION = {
    **QWEN3_5_35B_A3B_CONFIG["vision_config"],
    "depth": 1,
    "hidden_size": 16,
    "num_heads": 2,
    "intermediate_size": 16,
    "out_hidden_size": _TEXT["hidden_size"],
    "num_position_embeddings": 4,
}
_NUM_EXPERTS = _TEXT["num_experts"]


@pytest.fixture(scope="module")
def checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[PretrainedConfig, Path]:
    """An image-text checkpoint, routed experts stacked as published, with seeded random
    tensors, so a misplaced window fails."""

    folder = tmp_path_factory.mktemp("qwen3_5_moe-tp")
    published = {**QWEN3_5_35B_A3B_CONFIG, "text_config": _TEXT, "vision_config": _VISION}
    hf_config = load_hf_config(published, folder)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(hf_config)
        shapes = {name: tensor.shape for name, tensor in model.state_dict().items()}
    generator = torch.Generator().manual_seed(0)
    tensors = {name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in tensors
    assert any(name.startswith("model.visual.") for name in tensors)
    safetensors.torch.save_file(tensors, folder / "model.safetensors")
    return hf_config, folder


@pytest.mark.parametrize("size", [2, 4])
def test_every_rank_shard_rebuilds_the_single_rank_parameters(
    checkpoint: tuple[PretrainedConfig, Path], stub_flashinfer: None, size: int
) -> None:
    hf_config, folder = checkpoint
    _, reference, reference_shards = load_sharded(hf_config, folder, moe_backend=ReferenceMoe())
    assert reference_shards == {}

    ranks = [
        load_sharded(hf_config, folder, topology=Topology(rank, size), moe_backend=ReferenceMoe())[
            1:
        ]
        for rank in range(size)
    ]

    vision = [name for name in reference if name.startswith("visual.")]
    assert vision
    for local, shards in ranks:
        assert local.keys() == reference.keys() and shards.keys() <= local.keys()
        # Every rank runs the whole tower on its own images.
        assert shards.keys().isdisjoint(vision)
    for name, tensor in reference.items():
        torch.testing.assert_close(
            rebuild_parameter(name, tensor, ranks), tensor, rtol=0, atol=0, msg=name
        )
    count = _NUM_EXPERTS // size
    shared = _TEXT["shared_expert_intermediate_size"] // size
    for rank, (local, _) in enumerate(ranks):
        experts = slice(rank * count, (rank + 1) * count)
        for layer in range(_TEXT["num_hidden_layers"]):
            prefix = f"model.layers.{layer}.mlp"
            for name in ("experts.gate_up_proj", "experts.down_proj"):
                expected = reference[f"{prefix}.{name}"][experts]
                assert torch.equal(local[f"{prefix}.{name}"], expected), (rank, layer, name)
            assert local[f"{prefix}.shared_expert.gate_up_proj.weight"].shape[0] == 2 * shared
            assert local[f"{prefix}.shared_expert.down_proj.weight"].shape[1] == shared
            for name in ("gate.weight", "shared_expert_gate.weight"):
                assert torch.equal(local[f"{prefix}.{name}"], reference[f"{prefix}.{name}"])
