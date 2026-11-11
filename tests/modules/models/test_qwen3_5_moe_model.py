"""Qwen3.5-MoE text model: configuration, topology, registration and checkpoint mapping on CPU."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import safetensors.torch
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, PretrainedConfig

from cti.models import ModelCapability, get_model_class
from cti.models.base import BaseLLMModel
from cti.models.dflash import DFlashConfig
from cti.models.qwen3_5 import config as dense_config
from cti.models.qwen3_5_moe import config as moe_config
from cti.models.qwen3_5_moe.config import Qwen3_5MoeConfig
from cti.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration
from cti.models.weight import load_weight
from cti.moe.fused import FusedMoe
from tests.support.moe import BlockReferenceMoe, ReferenceMoe
from tests.support.paths import PYTHON_ROOT
from tests.support.qwen3_5 import load_hf_config, randomize_hf_weights
from tests.support.qwen3_5_moe import (
    QWEN3_5_35B_A3B_CONFIG,
    QWEN3_5_35B_A3B_DFLASH_CONFIG,
    QWEN3_5_35B_A3B_MTP_NAMES,
    QWEN3_5_35B_A3B_VISION_NAMES,
    TINY_QWEN3_5_MOE_CONFIG,
    hf_state,
    text_only_config,
)
from tests.support.tp_shards import build_model

CPU = torch.device("cpu")


def _meta_model(hf_config: PretrainedConfig, dtype: torch.dtype = torch.bfloat16) -> BaseLLMModel:
    return build_model(hf_config, dtype=dtype, moe_backend=ReferenceMoe())


def _load(model: BaseLLMModel, folder: Path) -> dict[str, torch.Tensor]:
    return dict(
        load_weight(str(folder), CPU, mapping=model.checkpoint_mapping, params=model.params())
    )


def _without_imports(path: Path) -> list[str]:
    """The lines of a module that no import statement spans."""

    source = path.read_text(encoding="utf-8")
    imports = {
        number
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for number in range(node.lineno, node.end_lineno + 1)
    }
    return [line for number, line in enumerate(source.splitlines(), 1) if number not in imports]


def _text_config(**changes: Any) -> dict[str, Any]:
    text = {**TINY_QWEN3_5_MOE_CONFIG["text_config"], **changes}
    return text_only_config({**TINY_QWEN3_5_MOE_CONFIG, "text_config": text})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"hidden_act": "gelu"}, "SiLU activation"),
        ({"layer_types": ["linear_attention", "sliding_attention"] * 2}, "layer type"),
        ({"num_experts_per_tok": 9}, "must not exceed num_experts"),
        ({"moe_intermediate_size": 0}, "moe_intermediate_size must be a positive integer"),
        ({"rms_norm_eps": 1e-5}, "RMSNorm epsilon"),
        ({"attention_bias": True}, "attention bias"),
        ({"linear_key_head_dim": 64}, "head dimensions must be 128"),
    ],
)
def test_config_rejects_what_the_layers_would_run_differently(
    tmp_path: Path, changes: Mapping[str, Any], message: str
) -> None:
    hf_config = load_hf_config(_text_config(**changes), tmp_path)

    with pytest.raises(ValueError, match=message):
        Qwen3_5MoeForConditionalGeneration.validate_hf_config(hf_config)


def test_config_reads_the_published_35b_geometry(tmp_path: Path) -> None:
    config = Qwen3_5MoeConfig.from_hf(load_hf_config(QWEN3_5_35B_A3B_CONFIG, tmp_path / "35b"))
    text, vision = config.text, config.vision

    assert (text.hidden_size, text.num_experts, text.num_experts_per_tok) == (2048, 256, 8)
    assert (text.moe_intermediate_size, text.shared_expert_intermediate_size) == (512, 512)
    assert (text.attention.num_qo_heads, text.attention.num_kv_heads) == (16, 2)
    assert (text.linear_attention.num_key_heads, text.linear_attention.num_value_heads) == (16, 32)
    assert [kind.value for kind in text.layer_types].count("full_attention") == 10
    assert not text.tie_word_embeddings
    assert (config.image_token_id, config.video_token_id) == (248056, 248057)
    assert (vision.depth, vision.hidden_size, vision.num_heads, vision.out_hidden_size) == (
        27,
        1152,
        16,
        2048,
    )
    assert (vision.patch_dim, vision.spatial_merge_size, vision.position_grid_side) == (1536, 2, 48)
    text_only = text_only_config(QWEN3_5_35B_A3B_CONFIG)
    exported = Qwen3_5MoeConfig.from_hf(load_hf_config(text_only, tmp_path / "text-only"))
    assert (exported.image_token_id, exported.vision) == (None, None)


@pytest.mark.parametrize(
    ("changes", "renormalize"), [({}, True), ({"norm_topk_prob": False}, False)]
)
def test_the_blocks_renormalize_as_norm_topk_prob_says(
    tmp_path: Path, stub_flashinfer: None, changes: Mapping[str, Any], renormalize: bool
) -> None:
    """Absent (the published 35B's config) it is True, as the reference always renormalizes;
    false reaches every block, which then gets no whole-block op from the fused backend."""

    hf_config = load_hf_config(_text_config(**changes), tmp_path)
    assert Qwen3_5MoeConfig.from_hf(hf_config).text.norm_topk_prob is renormalize

    backend = BlockReferenceMoe()
    model = build_model(hf_config, moe_backend=backend)

    blocks = [layer.mlp for layer in model.model.layers.op_list]
    assert {block.experts.renormalize for block in blocks} == {renormalize}
    assert {request[1] for request in backend.requests} == {renormalize}
    assert (FusedMoe().sparse_block(8, renormalize, torch.bfloat16) is None) is not renormalize


def test_unrenormalized_routes_keep_their_router_probabilities() -> None:
    """The composition ``FusedMoe`` falls back to without renormalization weights each
    selected expert by its own router probability."""

    generator = torch.Generator().manual_seed(0)
    experts, hidden, intermediate = 4, 8, 6
    x = torch.randn(3, hidden, generator=generator)
    w1 = torch.randn(experts, 2 * intermediate, hidden, generator=generator)
    w2 = torch.randn(experts, hidden, intermediate, generator=generator)
    router = torch.randn(experts, hidden, generator=generator)

    output = ReferenceMoe().forward(x.clone(), w1, w2, router, 2, renormalize=False)

    weights, ids = (x @ router.T).softmax(-1).topk(2, dim=-1)
    assert float(weights.sum(-1).max()) < 1
    expected = torch.zeros_like(x)
    for row in range(len(x)):
        for weight, expert in zip(weights[row], ids[row], strict=True):
            gate, up = (w1[expert] @ x[row]).chunk(2)
            expected[row] += weight * (w2[expert] @ (torch.nn.functional.silu(gate) * up))
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"hidden_act": "gelu"}, "tanh-approximated GELU"),
        ({"deepstack_visual_indexes": [8]}, "deepstack_visual_indexes"),
        ({"out_hidden_size": 16}, "must equal the text hidden_size"),
    ],
)
def test_config_rejects_a_vision_tower_the_engine_would_run_differently(
    tmp_path: Path, changes: Mapping[str, Any], message: str
) -> None:
    vision = {**TINY_QWEN3_5_MOE_CONFIG["vision_config"], **changes}
    hf_config = load_hf_config({**TINY_QWEN3_5_MOE_CONFIG, "vision_config": vision}, tmp_path)

    with pytest.raises(ValueError, match=message):
        Qwen3_5MoeForConditionalGeneration.validate_hf_config(hf_config)


def test_the_vision_tower_stays_dense_qwen3_5s_verbatim() -> None:
    """Structure debt: a peer model imports no other model, so qwen3_5_moe holds a copy of
    dense Qwen3.5's vision tower and its config until a model-neutral module shares them.
    The copy differs in imports alone, which keeps that move mechanical."""

    models = PYTHON_ROOT / "cti" / "models"

    assert _without_imports(models / "qwen3_5_moe" / "vision.py") == _without_imports(
        models / "qwen3_5" / "vision.py"
    )
    assert inspect.getsource(moe_config.Qwen3_5VisionConfig) == inspect.getsource(
        dense_config.Qwen3_5VisionConfig
    )


@pytest.mark.parametrize("layout", ["image-text", "text-only"])
def test_mapping_fills_every_tensor_from_the_published_key_table(
    tmp_path: Path, stub_flashinfer: None, layout: str
) -> None:
    image_text = layout == "image-text"
    published = QWEN3_5_35B_A3B_CONFIG
    hf_config = load_hf_config(published if image_text else text_only_config(published), tmp_path)
    model = _meta_model(hf_config)
    mapping, declared = model.checkpoint_mapping, model.state_dict()
    checkpoint = hf_state(hf_config)
    if image_text:
        assert {name for name in checkpoint if name.startswith("model.visual.")} == set(
            QWEN3_5_35B_A3B_VISION_NAMES
        )
        checkpoint |= {name: torch.empty(1, device="meta") for name in QWEN3_5_35B_A3B_MTP_NAMES}
        # The published index lists 1811 tensors: 693 text, 333 vision and 785 MTP.
        assert len(checkpoint) == 1811

    parts: dict[str, dict[int, torch.Tensor]] = {}
    skipped = set()
    for name, tensor in checkpoint.items():
        if not (placements := mapping.placements(name, declared)):
            skipped.add(name)
            continue
        for placement, part in zip(placements, mapping.split(name, tensor), strict=True):
            assert placement.expert is None, name
            slots = parts.setdefault(placement.name, {})
            assert slots.setdefault(placement.slot, part) is part, name
            assert len(slots) <= placement.num_slots

    # Only the multi-token-prediction head, numbered experts and all, is skipped: the vision
    # tower loads whole, and the text model's experts as the stacks they are stored as.
    assert skipped == {name for name in checkpoint if name.startswith("mtp.")}
    assert parts.keys() == declared.keys()
    vision = {name for name in declared if name.startswith("visual.")}
    assert vision == {
        name.removeprefix("model.") for name in checkpoint if name.startswith("model.visual.")
    }
    assert bool(vision) is image_text
    for name, slots in parts.items():
        merged = torch.cat([slots[slot] for slot in range(len(slots))])
        assert merged.shape == declared[name].shape, name
    if image_text:
        # The patch embedding's Conv3d weight [1152, 3, 2, 16, 16] loads as one linear map.
        assert declared["visual.patch_embed.proj.weight"].shape == (1152, 1536)
    layer = "model.layers.0.mlp."
    assert declared[layer + "experts.gate_up_proj"].shape == (256, 1024, 2048)
    assert declared[layer + "experts.down_proj"].shape == (256, 2048, 512)
    assert declared[layer + "shared_expert.gate_up_proj.weight"].shape == (1024, 2048)
    assert declared[layer + "shared_expert_gate.weight"].shape == (1, 2048)


@pytest.mark.parametrize("layout", ["image-text", "text-only"])
def test_published_stacked_experts_load_like_a_per_expert_export(
    tmp_path: Path, stub_flashinfer: None, layout: str
) -> None:
    image_text = layout == "image-text"
    hf_config = load_hf_config(
        TINY_QWEN3_5_MOE_CONFIG if image_text else text_only_config(TINY_QWEN3_5_MOE_CONFIG),
        tmp_path / "config",
    )
    auto_class = AutoModelForImageTextToText if image_text else AutoModelForCausalLM
    reference = auto_class.from_config(hf_config)
    randomize_hf_weights(reference)
    # Transformers saves the experts one by one, as fine-tuned exports store them.
    reference.save_pretrained(tmp_path / "per-expert")
    published = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
    if image_text:
        published["mtp.layers.0.mlp.experts.0.gate_proj.weight"] = torch.ones(2)
    (tmp_path / "published").mkdir()
    safetensors.torch.save_file(published, tmp_path / "published" / "model.safetensors")
    model = _meta_model(hf_config, torch.float32)

    stacked = _load(model, tmp_path / "published")
    numbered = _load(model, tmp_path / "per-expert")

    assert stacked.keys() == numbered.keys() == model.state_dict().keys()
    for name, tensor in stacked.items():
        assert torch.equal(tensor, numbered[name]), name
    mlp = "model.layers.1.mlp."
    text_model = reference.model.language_model if image_text else reference.model
    hf_mlp = text_model.layers[1].mlp
    assert torch.equal(stacked[mlp + "experts.gate_up_proj"], hf_mlp.experts.gate_up_proj)
    assert torch.equal(stacked[mlp + "experts.down_proj"], hf_mlp.experts.down_proj)
    assert torch.equal(stacked[mlp + "gate.weight"], hf_mlp.gate.weight)
    shared = hf_mlp.shared_expert
    assert torch.equal(
        stacked[mlp + "shared_expert.gate_up_proj.weight"],
        torch.cat([shared.gate_proj.weight, shared.up_proj.weight]),
    )
    assert torch.equal(stacked[mlp + "shared_expert.down_proj.weight"], shared.down_proj.weight)
    assert torch.equal(stacked[mlp + "shared_expert_gate.weight"], hf_mlp.shared_expert_gate.weight)


def test_35b_numbers_paged_and_recurrent_layers_and_builds_a_sparse_block_in_each(
    tmp_path: Path, stub_flashinfer: None
) -> None:
    model = _meta_model(load_hf_config(QWEN3_5_35B_A3B_CONFIG, tmp_path))
    layers = model.model.layers.op_list
    attention = list(model.iter_attention_layers())
    recurrent = model.iter_recurrent_layers()

    assert type(recurrent) is tuple
    assert (len(attention), len(recurrent)) == (10, 30)
    assert [i for i, layer in enumerate(layers) if hasattr(layer, "self_attn")] == list(
        range(3, 40, 4)
    )
    assert {(layer.num_qo_heads, layer.num_kv_heads, layer.head_dim) for layer in attention} == {
        (16, 2, 256)
    }
    assert {(layer.mlp.experts.top_k, layer.mlp.experts.renormalize) for layer in layers} == {
        (8, True)
    }
    assert model.rope_position_axes() == 3


def test_both_architectures_register_the_text_model_with_its_capabilities() -> None:
    for architecture in ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"):
        assert get_model_class(architecture) is Qwen3_5MoeForConditionalGeneration
    assert Qwen3_5MoeForConditionalGeneration.CAPABILITIES == (
        ModelCapability.LORA
        | ModelCapability.TRAINING
        | ModelCapability.TENSOR_PARALLEL
        | ModelCapability.PREFIX_REUSE
        | ModelCapability.SPECULATIVE_TARGET
    )
    assert Qwen3_5MoeForConditionalGeneration.LORA_TARGETS == (
        "attention",
        "linear_attention",
        "experts",
    )


def test_the_published_dflash_draft_fits_the_35b(tmp_path: Path) -> None:
    target_hf_config = load_hf_config(QWEN3_5_35B_A3B_CONFIG, tmp_path / "target")
    target = Qwen3_5MoeForConditionalGeneration.engine_config(target_hf_config)
    document = QWEN3_5_35B_A3B_DFLASH_CONFIG
    draft = load_hf_config(document, tmp_path / "draft")

    config = DFlashConfig.from_hf(document, draft, target)

    assert (config.block_size, config.mask_token_id) == (16, 248077)
    assert config.target_layer_ids == (1, 6, 11, 16, 22, 27, 32, 37)
    assert config.layer_windows == (4096,) * 5 + (None,)
