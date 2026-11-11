"""Registered models load Hugging Face checkpoints through their own mapping."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import safetensors
import safetensors.torch
import torch
from transformers import (
    AutoConfig,
    PreTrainedModel,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from cti.distributed import Topology
from cti.layers import Param
from cti.layers.gated_delta_net import GatedDeltaNet
from cti.models import get_model_class
from cti.models.base import BaseLLMModel
from cti.models.mapping import (
    CheckpointMapping,
    CheckpointPlacement,
    CheckpointRule,
    MergeRule,
    Reshape,
)
from cti.models.register import MODELS
from cti.models.weight import load_weight
from cti.moe import BaseMoeBackend
from tests.support.tiny_configs import TINY_GEOMETRY, tiny_hf_config
from tests.support.tp_shards import build_model, load_sharded, site


class _UnusedMoeBackend(BaseMoeBackend):
    def route(self, *args: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("the layer routes inside its backend's forward")

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("loading must not execute the MoE backend")


def _save_hf_checkpoint(
    hf_model: PreTrainedModel,
    folder: Path,
    *,
    extra: dict[str, torch.Tensor],
    wrapper_prefix: str,
) -> None:
    """Save like Hugging Face, then optionally wrap names as a LLaVA-style export would."""

    hf_model.save_pretrained(folder, max_shard_size="40KB")
    for file in sorted(folder.glob("*.safetensors")):
        tensors = safetensors.torch.load_file(file)
        safetensors.torch.save_file({wrapper_prefix + k: v for k, v in tensors.items()}, file)
    safetensors.torch.save_file(extra, folder / "extra.safetensors")


def _load_with_mapping(folder: Path, dtype: torch.dtype, **options: object) -> BaseLLMModel:
    return load_sharded(AutoConfig.from_pretrained(folder), folder, dtype=dtype, **options)[0]


def _expected_dense_tensor(hf_state: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Reproduce main's merge rules from the Hugging Face tensors of one runtime name."""

    for merged, parts in (
        ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
        ("gate_up_proj", ("gate_proj", "up_proj")),
    ):
        if f".{merged}." in name:
            return torch.cat([hf_state[name.replace(merged, part)] for part in parts])
    return hf_state[name]


@pytest.mark.parametrize("architecture", sorted(MODELS))
def test_registered_model_declares_its_own_checkpoint_mapping(
    tmp_path: Path,
    stub_flashinfer: None,
    architecture: str,
) -> None:
    hf_config = tiny_hf_config(architecture, tmp_path)
    # A dense model never asks its services for the MoE backend.
    model = build_model(hf_config, moe_backend=_UnusedMoeBackend())

    # The base class only annotates the mapping, so a model that declares none fails here.
    assert isinstance(model.checkpoint_mapping, CheckpointMapping)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("wrapper_prefix", ["", "language_model."])
def test_qwen3_loads_a_hugging_face_checkpoint_like_main(
    tmp_path: Path,
    stub_flashinfer: None,
    dtype: torch.dtype,
    tied: bool,
    wrapper_prefix: str,
) -> None:
    torch.manual_seed(0)
    hf_config = Qwen3Config(
        **TINY_GEOMETRY,
        tie_word_embeddings=tied,
        architectures=["Qwen3ForCausalLM"],
    )
    hf_model = Qwen3ForCausalLM(hf_config)
    hf_state = {k: v.detach() for k, v in hf_model.state_dict().items()}
    extra = {
        "vision_tower.patch_embed.weight": torch.ones(2, 2),
        "multi_modal_projector.linear.weight": torch.ones(2, 2),
    }
    if tied:
        # Released tied Qwen3 checkpoints, such as Qwen3-0.6B, still store the head.
        extra["lm_head.weight"] = torch.ones_like(hf_state["lm_head.weight"])
    _save_hf_checkpoint(hf_model, tmp_path, extra=extra, wrapper_prefix=wrapper_prefix)
    assert len(list(tmp_path.glob("model-*.safetensors"))) > 1

    model = _load_with_mapping(tmp_path, dtype)

    loaded = model.state_dict()
    assert ("lm_head.weight" in loaded) is not tied
    for name, tensor in loaded.items():
        assert tensor.dtype == dtype
        assert torch.equal(tensor, _expected_dense_tensor(hf_state, name).to(dtype)), name


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_qwen3_moe_loads_a_hugging_face_checkpoint_like_main(
    tmp_path: Path,
    stub_flashinfer: None,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    hf_config = Qwen3MoeConfig(
        **TINY_GEOMETRY,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        architectures=["Qwen3MoeForCausalLM"],
    )
    hf_model = Qwen3MoeForCausalLM(hf_config)
    # Transformers keeps experts fused as [E, 2I, H] with gate rows first and saves
    # them per expert, so its in-memory tensors are an independent stacking oracle.
    hf_state = {k: v.detach() for k, v in hf_model.state_dict().items()}
    _save_hf_checkpoint(hf_model, tmp_path, extra={}, wrapper_prefix="")
    saved_names = set()
    for file in tmp_path.glob("*.safetensors"):
        with safetensors.safe_open(file, "pt") as f:
            saved_names.update(f.keys())
    assert "model.layers.0.mlp.experts.3.up_proj.weight" in saved_names

    model = _load_with_mapping(tmp_path, dtype, moe_backend=_UnusedMoeBackend())

    loaded = model.state_dict()
    assert loaded["model.layers.1.mlp.experts.gate_up_proj"].shape == (4, 32, 32)
    for name, tensor in loaded.items():
        assert tensor.dtype == dtype
        assert torch.equal(tensor, _expected_dense_tensor(hf_state, name).to(dtype)), name


def test_qwen3_moe_mapping_places_an_expert_projection_in_its_stacked_parameter() -> None:
    mapping = get_model_class("Qwen3MoeForCausalLM").checkpoint_mapping
    part = "model.layers.0.mlp.experts.3.up_proj.weight"

    placements = mapping.placements(part, {"model.layers.0.mlp.experts.gate_up_proj"})

    assert placements == (
        CheckpointPlacement(
            name="model.layers.0.mlp.experts.gate_up_proj",
            slot=1,
            num_slots=2,
            expert=3,
        ),
    )


def _block_amax(weight: torch.Tensor) -> torch.Tensor:
    """One value per 128 x 128 block of ``weight``, laid out as a block-FP8 scale grid."""
    return weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).abs().amax(dim=(1, 3))


@pytest.mark.parametrize(
    ("module", "rows"),
    [
        ("layers.0.linear_attn.in_proj_qkv", 512),  # q, k, v rows
        ("layers.1.self_attn.q_proj", 1024),  # per-head query, gate rows
        ("layers.1.self_attn.k_proj", 256),  # merge slot only
    ],
)
def test_a_qwen3_5_scale_follows_its_weights_rule_and_merge_block_by_block(
    tmp_path: Path, stub_flashinfer: None, module: str, rows: int
) -> None:
    model = build_model(tiny_hf_config("Qwen3_5ForCausalLM", tmp_path))
    mapping = model.checkpoint_mapping
    declared = {
        name.removesuffix("weight") + leaf
        for name in model.state_dict()
        for leaf in ("weight", "weight_scale_inv")
    }
    weight_name = f"model.{module}.weight"
    weight = torch.randn(rows, 256)

    weight_placements = mapping.placements(weight_name, declared)
    scale_placements = mapping.placements(weight_name + "_scale_inv", declared)
    weight_parts = mapping.split(weight_name, weight)
    scale_parts = mapping.split(weight_name + "_scale_inv", _block_amax(weight), rows)

    assert scale_placements == tuple(
        replace(placement, name=placement.name + "_scale_inv") for placement in weight_placements
    )
    assert len(scale_parts) == len(weight_parts)
    for weight_part, scale_part in zip(weight_parts, scale_parts, strict=True):
        assert torch.equal(scale_part, _block_amax(weight_part))


@pytest.mark.parametrize("stack_experts", [False, True], ids=["dense", "experts"])
def test_a_repeated_rule_target_takes_each_part_as_its_next_slot(stack_experts: bool) -> None:
    # In order: the rule, the repeated target's slots, then expert stacking.
    mapping = CheckpointMapping(
        rules=(CheckpointRule("fused.weight", ("w.weight",) * 3, Reshape(lambda t: t.chunk(3))),),
        stack_experts=stack_experts,
    )
    layer = "model.experts.2" if stack_experts else "model.layers.0"
    name = "model.experts.w" if stack_experts else f"{layer}.w.weight"

    placements = mapping.placements(f"{layer}.fused.weight", {name})

    expert = 2 if stack_experts else None
    assert placements == tuple(
        CheckpointPlacement(name=name, slot=slot, num_slots=3, expert=expert) for slot in range(3)
    )


def test_the_loader_concatenates_a_repeated_targets_parts_in_rule_order(tmp_path: Path) -> None:
    fused = torch.arange(12, dtype=torch.float32).view(6, 2)
    safetensors.torch.save_file({"model.fused.weight": fused}, tmp_path / "model.safetensors")
    # The parts reach the slots in the transform's order, not the checkpoint's.
    mapping = CheckpointMapping(
        rules=(
            CheckpointRule("fused.weight", ("w.weight",) * 3, Reshape(lambda t: t.chunk(3)[::-1])),
        )
    )

    loaded = dict(
        load_weight(
            str(tmp_path),
            torch.device("cpu"),
            mapping=mapping,
            params=(Param("model.w.weight", (6, 2), torch.float32),),
        )
    )

    assert torch.equal(loaded["model.w.weight"], torch.cat(fused.chunk(3)[::-1]))


def test_each_slot_of_a_repeated_target_reads_its_own_window(tmp_path: Path) -> None:
    # Rank 1 of 2 owns key head 1 and value heads 2-3 of a convolution whose channels
    # are all query, then all key, then all value heads, 128 channels each.
    layer = GatedDeltaNet(
        num_key_heads=2,
        num_value_heads=4,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_size=4,
        site=site(Topology(1, 2)),
    )
    conv = torch.arange(1024 * 4, dtype=torch.float32).view(1024, 1, 4)
    safetensors.torch.save_file({"model.conv1d.weight": conv}, tmp_path / "model.safetensors")
    mapping = CheckpointMapping(
        rules=(
            CheckpointRule(
                "conv1d.weight",
                ("gdn.conv_weight",) * 3,
                Reshape(lambda t: t.split((256, 256, 512))),
            ),
        )
    )

    params = {param.path: param for param in layer.params(prefix="model.gdn")}
    conv_weight = params["model.gdn.conv_weight"]
    loaded = dict(
        load_weight(
            str(tmp_path),
            torch.device("cpu"),
            mapping=mapping,
            # FP32, so every element stays exact.
            params=[replace(conv_weight, dtype=torch.float32)],
        )
    )

    expected = torch.cat((conv[128:256], conv[384:512], conv[768:1024]))
    assert torch.equal(loaded["model.gdn.conv_weight"], expected)


def test_a_repeated_target_that_a_merge_also_matches_is_rejected() -> None:
    mapping = CheckpointMapping(
        rules=(
            CheckpointRule("fused.weight", ("q_proj.weight",) * 2, Reshape(lambda t: t.chunk(2))),
        ),
        merges=(MergeRule("qkv_proj", ("q_proj", "k_proj", "v_proj")),),
    )

    with pytest.raises(
        ValueError,
        match=r"target 'model\.q_proj\.weight' is repeated and also merged into 'qkv_proj'",
    ):
        mapping.placements("model.fused.weight", {"model.qkv_proj.weight"})
