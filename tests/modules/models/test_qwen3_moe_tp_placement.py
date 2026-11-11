"""Qwen3-MoE checkpoint windows at the published 30B-A3B and 235B-A22B head and expert geometry,
read from the published layout (one expert's projections apart): every rank's shards rebuild the
single-rank parameters, and each rank loads only its own experts, whole. Beyond four ranks the
four KV heads are replicated."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import safetensors.torch
import torch
from transformers import AutoConfig, PretrainedConfig

from cti.distributed import Topology
from cti.models.weight import load_weight
from tests.support.moe import ReferenceMoe
from tests.support.qwen3_moe import PUBLISHED, checkpoint_shapes
from tests.support.tp_shards import CPU, build_model, load_sharded, rebuild_parameter

pytestmark = pytest.mark.usefixtures("stub_flashinfer")

# The published heads and experts over a narrow hidden width, two layers and a small vocabulary
# that leaves padded rows at every size.
_NARROW = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "moe_intermediate_size": 8,
    "vocab_size": 259,
    "max_position_embeddings": 64,
}
_SIZES = {"30b-a3b": (2, 4, 8), "235b-a22b": (4, 8)}


def _checkpoint(variant: str, folder: Path) -> tuple[PretrainedConfig, Mapping[str, Any]]:
    """Save seeded random tensors under the published names; return the config and its
    document."""

    document = {**PUBLISHED[variant].config, **_NARROW}
    folder.mkdir(parents=True, exist_ok=True)
    hf_config = AutoConfig.for_model(**document)
    hf_config.save_pretrained(folder)
    generator = torch.Generator().manual_seed(0)
    tensors = {
        name: torch.randn(shape, generator=generator)
        for name, shape in checkpoint_shapes(document).items()
    }
    safetensors.torch.save_file(tensors, folder / "model.safetensors")
    return AutoConfig.from_pretrained(folder), document


@pytest.mark.parametrize(
    ("variant", "size"),
    [(variant, size) for variant, sizes in _SIZES.items() for size in sizes],
)
def test_every_rank_shard_rebuilds_the_single_rank_parameters(
    tmp_path: Path, variant: str, size: int
) -> None:
    hf_config, document = _checkpoint(variant, tmp_path)
    _, reference, reference_shards = load_sharded(hf_config, tmp_path, moe_backend=ReferenceMoe())
    assert reference_shards == {}

    ranks = [
        load_sharded(
            hf_config, tmp_path, topology=Topology(rank, size), moe_backend=ReferenceMoe()
        )[1:]
        for rank in range(size)
    ]

    for local, shards in ranks:
        assert local.keys() == reference.keys() and shards.keys() <= local.keys()
    for name, tensor in reference.items():
        torch.testing.assert_close(
            rebuild_parameter(name, tensor, ranks), tensor, rtol=0, atol=0, msg=name
        )
    num_experts = document["num_experts"]
    count = num_experts // size
    for rank, (local, _) in enumerate(ranks):
        experts = slice(rank * count, (rank + 1) * count)
        for layer in range(document["num_hidden_layers"]):
            prefix = f"model.layers.{layer}.mlp"
            for name in ("experts.gate_up_proj", "experts.down_proj"):
                expected = reference[f"{prefix}.{name}"][experts]
                assert torch.equal(local[f"{prefix}.{name}"], expected), (rank, layer, name)
            # Every rank routes every token.
            assert torch.equal(local[f"{prefix}.gate.weight"], reference[f"{prefix}.gate.weight"])
    # Each rank holds whole query heads and its share of the four KV heads, at least one:
    # beyond four ranks they are replicated.
    head_dim, kv_heads = document["head_dim"], document["num_key_value_heads"]
    queries = document["num_attention_heads"] // size * head_dim
    qkv = ranks[0][0]["model.layers.0.self_attn.qkv_proj.weight"]
    assert qkv.shape[0] == queries + 2 * max(kv_heads // size, 1) * head_dim


def test_a_rank_reads_no_tensor_of_another_ranks_experts(tmp_path: Path) -> None:
    """Rank 1 of 4 needs only its own experts' tensors: a checkpoint without every other
    expert still loads there, and its stack holds exactly its experts."""

    hf_config, document = _checkpoint("30b-a3b", tmp_path)
    (file,) = tmp_path.glob("*.safetensors")
    state = safetensors.torch.load_file(file)
    count = document["num_experts"] // 4
    held = range(count, 2 * count)
    others = [
        name
        for name in state
        if ".experts." in name and int(name.split(".experts.")[1].split(".")[0]) not in held
    ]
    assert others
    safetensors.torch.save_file({k: v for k, v in state.items() if k not in others}, file)

    model = build_model(hf_config, topology=Topology(1, 4), moe_backend=ReferenceMoe())
    loaded = dict(
        load_weight(
            str(tmp_path),
            CPU,
            mapping=model.checkpoint_mapping,
            params=model.params(),
        )
    )
    model.load_state_dict(dict(loaded))

    gate_up = loaded["model.layers.0.mlp.experts.gate_up_proj"]
    assert gate_up.shape[0] == count
    expected = state[f"model.layers.0.mlp.experts.{count}.gate_proj.weight"]
    assert torch.equal(gate_up[0, : expected.shape[0]], expected)
