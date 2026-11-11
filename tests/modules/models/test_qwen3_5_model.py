# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`); see THIRD_PARTY_NOTICES.

"""Qwen3.5 dense text model: topology, registration and checkpoint mapping on CPU."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import safetensors.torch
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PretrainedConfig

from cti.distributed import Topology
from cti.forward import RecurrentExport
from cti.layers import LayerServices
from cti.layers.rotary import MultiAxisRotaryEmbedding
from cti.models import ModelCapability, create_model, get_model_class
from cti.models.base import BaseLLMModel
from cti.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from cti.models.weight import load_weight
from tests.support import reference_ops
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.forward_rows import decode_entry, extend_entry
from tests.support.gated_delta_net import ReferenceGatedDeltaNetPrefill, reference_kernel_module
from tests.support.qwen3_5 import (
    INTERLEAVED_QWEN3_5_CONFIG,
    QWEN3_5_9B_CONFIG,
    QWEN3_8_27B_CONFIG,
    TINY_QWEN3_5_CONFIG,
    hf_state,
    load_hf_config,
    randomize_hf_weights,
    text_only_config,
)
from tests.support.qwen3_5_engine import EagerEngine, bind_reference_kernels, load_model
from tests.support.tp_shards import default_dtype, site
from tests.support.training_ops import bind_training_references

_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_Q_PROJ = "model.language_model.layers.3.self_attn.q_proj.weight"
# Fifteen mtp.* tensors stand in for the released checkpoints' multi-token-prediction
# head, which the engine skips by prefix.
_MTP_NAMES = (
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    *(
        f"mtp.layers.0.{module}.weight"
        for module in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    ),
)


def _meta_model(hf_config: PretrainedConfig) -> BaseLLMModel:
    with torch.device("meta"), default_dtype(torch.bfloat16):
        return create_model(
            Qwen3_5ForConditionalGeneration, checkpoint_facts(hf_config), site=site()
        )


def _merged_runtime_tensors(
    model: BaseLLMModel,
    checkpoint: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Place and merge ``checkpoint`` as the loader does, checking every merge completes."""

    mapping = model.checkpoint_mapping
    declared = model.state_dict()
    slots: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    num_slots: dict[str, int] = {}
    for name, tensor in checkpoint.items():
        if not (placements := mapping.placements(name, declared)):
            continue
        for placement, part in zip(placements, mapping.split(name, tensor), strict=True):
            assert num_slots.setdefault(placement.name, placement.num_slots) == placement.num_slots
            assert slots[placement.name].setdefault(placement.slot, part) is part, name
    assert {name: len(parts) for name, parts in slots.items()} == num_slots
    return {
        name: torch.cat([parts[slot] for slot in range(len(parts))])
        for name, parts in slots.items()
    }


def _save_checkpoint(checkpoint: Mapping[str, torch.Tensor], folder: Path) -> Path:
    """Write ``checkpoint`` as two safetensors shards in ``folder``."""

    folder.mkdir()
    names = sorted(checkpoint)
    for index, shard in enumerate((names[::2], names[1::2])):
        safetensors.torch.save_file(
            {name: checkpoint[name] for name in shard}, folder / f"model-{index}.safetensors"
        )
    return folder


def _load(model: BaseLLMModel, folder: Path) -> dict[str, torch.Tensor]:
    return dict(
        load_weight(
            str(folder),
            torch.device("cpu"),
            mapping=model.checkpoint_mapping,
            params=model.params(),
        )
    )


def _cpu_tiny_model(tmp_path: Path) -> BaseLLMModel:
    torch.manual_seed(0)
    with default_dtype(torch.float32):
        model = create_model(
            Qwen3_5ForConditionalGeneration,
            checkpoint_facts(load_hf_config(TINY_QWEN3_5_CONFIG, tmp_path)),
            site=site(),
        )
    for tensor in model.state_dict().values():
        tensor.normal_()
    return model


def _reference_runtime_tensor(checkpoint: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    """Build one runtime tensor from Hugging Face text-model tensors, independently."""

    if name.startswith("visual."):
        tensor = checkpoint[f"model.{name}"]
        # A Conv3d whose stride is its kernel is one linear map of each flattened patch.
        return (
            tensor.reshape(len(tensor), -1) if name.endswith("patch_embed.proj.weight") else tensor
        )
    hf_name = (
        name.replace("linear_attn.gdn.", "linear_attn.")
        .replace(".conv_weight", ".conv1d.weight")
        .replace(".norm_weight", ".norm.weight")
    )
    if ".qkv_gate_proj." in name:
        # transformers 5.16.1 Qwen3_5Attention chunks each 2 * 256 q_proj rows of a head
        # into its query and its gate.
        q_proj = checkpoint[hf_name.replace("qkv_gate_proj", "q_proj")]
        query, gate = torch.chunk(q_proj.unflatten(0, (-1, 2 * 256)), 2, dim=1)
        key = checkpoint[hf_name.replace("qkv_gate_proj", "k_proj")]
        value = checkpoint[hf_name.replace("qkv_gate_proj", "v_proj")]
        return torch.cat([query.flatten(0, 1), key, value, gate.flatten(0, 1)])
    for merged, parts in (
        ("in_proj", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")),
        ("gate_up_proj", ("gate_proj", "up_proj")),
    ):
        if f".{merged}." in name:
            return torch.cat([checkpoint[hf_name.replace(merged, part)] for part in parts])
    return checkpoint[hf_name]


@pytest.mark.parametrize(
    ("published", "checkpoint_size"),
    # The published 9B index lists 775 tensors: 427 text, 333 vision and 15 MTP.
    [(QWEN3_5_9B_CONFIG, 775), (QWEN3_8_27B_CONFIG, None)],
    ids=["9b", "27b"],
)
@pytest.mark.parametrize("layout", ["image-text", "text-only"])
def test_mapping_fills_every_tensor_from_a_published_key_table(
    tmp_path: Path,
    stub_flashinfer: None,
    published: Mapping[str, Any],
    checkpoint_size: int | None,
    layout: str,
) -> None:
    image_text = layout == "image-text"
    hf_config = load_hf_config(published if image_text else text_only_config(published), tmp_path)
    model = _meta_model(hf_config)
    declared = model.state_dict()
    checkpoint = hf_state(hf_config)
    if image_text:
        checkpoint |= {name: torch.empty(1, device="meta") for name in _MTP_NAMES}
        if checkpoint_size is not None:
            assert len(checkpoint) == checkpoint_size

    skipped = {
        name for name in checkpoint if not model.checkpoint_mapping.placements(name, declared)
    }
    merged = _merged_runtime_tensors(model, checkpoint)

    # Only the multi-token-prediction head is skipped: the vision tower loads whole.
    expected_skips = {name for name in checkpoint if name.startswith("mtp.")}
    assert skipped == expected_skips
    assert bool(expected_skips) is image_text
    assert merged.keys() == declared.keys()
    vision = {name for name in declared if name.startswith("visual.")}
    assert vision == {
        name.removeprefix("model.") for name in checkpoint if name.startswith("model.visual.")
    }
    assert bool(vision) is image_text
    for name, tensor in merged.items():
        assert tensor.shape == declared[name].shape, name


@pytest.mark.parametrize("num_heads", [1, 3])
def test_q_proj_splits_into_the_reference_query_and_gate_rows(
    tmp_path: Path,
    stub_flashinfer: None,
    num_heads: int,
) -> None:
    head_dim, hidden_size = 256, 5
    mapping = _meta_model(load_hf_config(TINY_QWEN3_5_CONFIG, tmp_path)).checkpoint_mapping
    q_proj = torch.randn(num_heads * 2 * head_dim, hidden_size)

    query, gate = mapping.split(_Q_PROJ, q_proj)

    # Rewritten from transformers 5.16.1 Qwen3_5Attention.forward: the projection is
    # viewed as [heads, 2 * head_dim] and chunked into query and gate.
    reference_query, reference_gate = torch.chunk(
        q_proj.view(num_heads, 2 * head_dim, hidden_size), 2, dim=1
    )
    assert torch.equal(query, reference_query.reshape(-1, hidden_size))
    assert torch.equal(gate, reference_gate.reshape(-1, hidden_size))


@pytest.mark.parametrize("layout", ["image-text", "text-only"])
def test_loads_a_checkpoint_in_its_stored_dtypes(
    tmp_path: Path,
    stub_flashinfer: None,
    layout: str,
) -> None:
    torch.manual_seed(0)
    image_text = layout == "image-text"
    hf_config = load_hf_config(
        TINY_QWEN3_5_CONFIG if image_text else text_only_config(TINY_QWEN3_5_CONFIG),
        tmp_path / "config",
    )
    # Released checkpoints store BF16 tensors except FP32 A_log and gated-norm weights.
    checkpoint = {
        name: torch.randn(
            tensor.shape,
            dtype=torch.float32
            if name.endswith(("linear_attn.A_log", "linear_attn.norm.weight"))
            else torch.bfloat16,
        )
        for name, tensor in hf_state(hf_config).items()
    }
    if image_text:
        checkpoint |= {name: torch.ones(2, dtype=torch.bfloat16) for name in _MTP_NAMES}
    folder = _save_checkpoint(checkpoint, tmp_path / "checkpoint")
    with torch.device("meta"), default_dtype(torch.bfloat16):
        model = create_model(
            Qwen3_5ForConditionalGeneration, checkpoint_facts(hf_config), site=site()
        )
    targets = model.state_dict()

    loaded = _load(model, folder)

    prefix = "model.language_model." if image_text else "model."
    hf_text = {
        "model." + name.removeprefix(prefix) if name.startswith(prefix) else name: tensor
        for name, tensor in checkpoint.items()
    }
    assert loaded.keys() == targets.keys()
    for name, tensor in loaded.items():
        assert tensor.dtype == targets[name].dtype, name
        expected = _reference_runtime_tensor(hf_text, name).to(tensor.dtype)
        assert torch.equal(tensor, expected), name
    gdn = "model.layers.0.linear_attn.gdn."
    assert {loaded[gdn + name].dtype for name in ("A_log", "dt_bias", "norm_weight")} == {
        torch.float32
    }
    assert torch.equal(loaded[gdn + "A_log"], checkpoint[prefix + "layers.0.linear_attn.A_log"])
    assert loaded[gdn + "conv_weight"].dtype == torch.bfloat16


@pytest.mark.parametrize("stores_lm_head", [True, False], ids=["lm-head", "no-lm-head"])
def test_a_tied_head_reads_the_embedding_whether_or_not_lm_head_is_stored(
    tmp_path: Path,
    stub_flashinfer: None,
    stores_lm_head: bool,
) -> None:
    torch.manual_seed(0)
    tied = {**TINY_QWEN3_5_CONFIG, "tie_word_embeddings": True}
    hf_config = load_hf_config(tied, tmp_path / "config")
    checkpoint = {
        name: torch.randn(tensor.shape, dtype=torch.bfloat16)
        for name, tensor in hf_state(hf_config).items()
        if name != "lm_head.weight"
    }
    embedding = checkpoint["model.language_model.embed_tokens.weight"]
    if stores_lm_head:
        # A stored head that differs from the embedding is dropped, never read.
        checkpoint["lm_head.weight"] = torch.randn(embedding.shape, dtype=torch.bfloat16)
    folder = _save_checkpoint(checkpoint, tmp_path / "checkpoint")
    with torch.device("meta"), default_dtype(torch.bfloat16):
        model = create_model(
            Qwen3_5ForConditionalGeneration, checkpoint_facts(hf_config), site=site()
        )

    loaded = _load(model, folder)
    model.load_state_dict(dict(loaded))

    assert "lm_head.weight" not in loaded
    assert loaded.keys() == model.state_dict().keys()
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    assert torch.equal(model.lm_head.forward_training(x), F.linear(x, embedding))


def test_9b_numbers_paged_and_recurrent_layers_in_execution_order(
    tmp_path: Path,
    stub_flashinfer: None,
) -> None:
    model = _meta_model(load_hf_config(QWEN3_5_9B_CONFIG, tmp_path))
    layers = model.model.layers.op_list
    attention = list(model.iter_attention_layers())
    recurrent = model.iter_recurrent_layers()

    # The runner tests the recurrent layers for truth and walks them twice.
    assert type(recurrent) is tuple
    assert attention == list(model.iter_attention_layers())
    # The runner groups the state layers in this order: KV layers, then recurrent rows.
    assert model.iter_state_layers() == (*attention, *recurrent)
    assert (len(attention), len(recurrent)) == (8, 24)
    assert {(layer.num_key_heads, layer.num_value_heads) for layer in recurrent} == {(16, 32)}
    assert [i for i, layer in enumerate(layers) if hasattr(layer, "self_attn")] == list(
        range(3, 32, 4)
    )
    assert all(hasattr(layer, "self_attn") != hasattr(layer, "linear_attn") for layer in layers)
    assert len({id(layer.rotary) for layer in attention}) == 1
    assert isinstance(attention[0].rotary, MultiAxisRotaryEmbedding)
    assert model.rope_position_axes() == 3


def test_9b_declares_merged_projections_and_fp32_recurrence_weights(
    tmp_path: Path,
    stub_flashinfer: None,
) -> None:
    state = _meta_model(load_hf_config(QWEN3_5_9B_CONFIG, tmp_path)).state_dict()

    assert state["model.layers.3.self_attn.qkv_gate_proj.weight"].shape == (10240, 4096)
    assert state["model.layers.3.self_attn.q_norm.weight"].shape == (256,)
    assert state["model.layers.0.linear_attn.in_proj.weight"].shape == (12352, 4096)
    assert state["model.layers.0.linear_attn.gdn.conv_weight"].shape == (8192, 1, 4)
    assert state["model.layers.0.mlp.gate_up_proj.weight"].shape == (24576, 4096)
    assert {name for name, tensor in state.items() if tensor.dtype == torch.float32} == {
        f"model.layers.{index}.linear_attn.gdn.{name}"
        for index in range(32)
        if (index + 1) % 4
        for name in ("A_log", "dt_bias", "norm_weight")
    }
    assert {tensor.dtype for tensor in state.values()} == {torch.bfloat16, torch.float32}


def test_attention_attends_to_q_k_v_and_gates_the_output(
    tmp_path: Path,
    stub_flashinfer: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = _cpu_tiny_model(tmp_path).model.layers.op_list[1].self_attn
    attended = torch.randn(3, 512)
    received: list[torch.Tensor] = []

    def attend(qkv: torch.Tensor, _batch: object) -> torch.Tensor:
        received.append(qkv)
        return attended

    monkeypatch.setattr(attention.attn, "forward", attend)
    x = torch.randn(3, 32)

    output = attention.forward(x, SimpleNamespace())

    # Two 256-wide query heads and one KV head: q | k | v is 1024 wide, then the gate.
    projected = F.linear(x, attention.qkv_gate_proj.weight)
    assert torch.equal(received[0], projected[:, :1024])
    expected = F.linear(attended * torch.sigmoid(projected[:, 1024:]), attention.o_proj.weight)
    assert torch.equal(output, expected)


def test_linear_attention_runs_the_mixer_on_the_merged_projection(
    tmp_path: Path,
    stub_flashinfer: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear_attention = _cpu_tiny_model(tmp_path).model.layers.op_list[0].linear_attn
    mixed = torch.randn(3, 256)
    received: list[torch.Tensor] = []

    def mix(projected: torch.Tensor, _batch: object) -> torch.Tensor:
        received.append(projected)
        return mixed

    monkeypatch.setattr(linear_attention.gdn, "forward", mix)
    x = torch.randn(3, 32)

    output = linear_attention.forward(x, SimpleNamespace())

    # q | k | v | z | b | a = 128 + 128 + 256 + 256 + 2 + 2 columns.
    assert received[0].shape == (3, 772)
    assert torch.equal(received[0], F.linear(x, linear_attention.in_proj.weight))
    assert torch.equal(output, F.linear(mixed, linear_attention.out_proj.weight))


@pytest.mark.parametrize("num_ranks", [1, 2])
@pytest.mark.parametrize("exports", [True, False])
def test_the_class_answers_whether_its_built_layers_export_prefill_states(
    tmp_path: Path,
    stub_flashinfer: None,
    monkeypatch: pytest.MonkeyPatch,
    exports: bool,
    num_ranks: int,
) -> None:
    built: list[tuple[int, int, torch.device]] = []

    class GatedDeltaNetPrefill(ReferenceGatedDeltaNetPrefill):
        supports_state_capture = exports

        def __init__(self, **geometry: Any) -> None:
            super().__init__(**geometry)
            built.append((self.num_k_heads, self.num_v_heads, self.device))

    module = reference_kernel_module()
    module.GatedDeltaNetPrefill = GatedDeltaNetPrefill  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", module)
    monkeypatch.setattr("ctikernels.attention.inference.Prologue", reference_ops.Prologue)
    config = deepcopy(dict(INTERLEAVED_QWEN3_5_CONFIG))
    config["text_config"] |= {"linear_num_key_heads": 2, "linear_num_value_heads": 4}
    hf_config = load_hf_config(config, tmp_path)
    rank_site = site(Topology(0, num_ranks), services=LayerServices())

    answer = Qwen3_5ForConditionalGeneration.exports_prefill_states(
        hf_config, rank_site.device, num_ranks=num_ranks
    )
    with torch.device("meta"), default_dtype(torch.bfloat16):
        model = create_model(
            Qwen3_5ForConditionalGeneration, checkpoint_facts(hf_config), site=rank_site
        )

    assert answer is exports
    assert [layer.exports_prefill_states for layer in model.iter_recurrent_layers()] == [answer] * 2
    # The query built the prefill of the layers' device key.
    assert built == [(2 // num_ranks, 4 // num_ranks, rank_site.device)] * 2


def test_layers_share_their_geometry_operations(
    tmp_path: Path,
    stub_flashinfer: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())
    monkeypatch.setattr("ctikernels.attention.inference.Prologue", reference_ops.Prologue)
    hf_config = load_hf_config(INTERLEAVED_QWEN3_5_CONFIG, tmp_path)
    with torch.device("meta"), default_dtype(torch.bfloat16):
        model = create_model(
            Qwen3_5ForConditionalGeneration,
            checkpoint_facts(hf_config),
            site=site(services=LayerServices()),
        )
    attention = tuple(model.iter_attention_layers())
    recurrent = model.iter_recurrent_layers()

    prologue = attention[0].prologue
    assert (prologue.num_q_heads, prologue.num_kv_heads, prologue.eps) == (2, 1, 1e-6)
    assert len(attention) == 2
    assert all(layer.prologue is prologue for layer in attention)
    assert len(recurrent) == 2 and all(layer.exports_prefill_states for layer in recurrent)
    with pytest.raises(ValueError, match="BF16"):
        with torch.device("meta"), default_dtype(torch.float32):
            create_model(
                Qwen3_5ForConditionalGeneration,
                checkpoint_facts(hf_config),
                site=site(services=LayerServices()),
            )


def test_both_architectures_register_the_dense_model_with_its_capabilities(
    tmp_path: Path,
    stub_flashinfer: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_class = get_model_class(_ARCHITECTURE)
    declared = (
        ModelCapability.LORA
        | ModelCapability.TRAINING
        | ModelCapability.TENSOR_PARALLEL
        | ModelCapability.PREFIX_REUSE
        | ModelCapability.SPECULATIVE_TARGET
    )

    assert get_model_class("Qwen3_5ForCausalLM") is model_class
    assert model_class.CAPABILITIES == declared
    model_class.require_capabilities(declared, model_architecture="Qwen3_5ForCausalLM")
    for capability in ModelCapability:
        if capability in declared:
            continue
        with pytest.raises(ValueError, match=capability.name):
            model_class.require_capabilities(capability, model_architecture="Qwen3_5ForCausalLM")

    # A training model keeps the rotary caches the training operation reads.
    bind_training_references(monkeypatch)
    with torch.device("meta"):
        model = create_model(
            model_class,
            checkpoint_facts(load_hf_config(TINY_QWEN3_5_CONFIG, tmp_path)),
            site=site(device=torch.device("meta"), training=True),
        )
    (attention,) = model.iter_attention_layers()
    assert attention.rotary._training_cos_cache is not None


def _prefix_checkpoint(tmp_path_factory, length):
    folder = tmp_path_factory.mktemp("prefix-checkpoint")
    published = deepcopy(TINY_QWEN3_5_CONFIG)
    published["text_config"]["max_position_embeddings"] = 512
    config = load_hf_config(text_only_config(published), folder)
    reference = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    randomize_hf_weights(reference, conv_dtype=torch.bfloat16)
    reference.save_pretrained(folder)
    tokens = torch.randint(
        0,
        reference.config.vocab_size,
        (1, length),
        generator=torch.Generator().manual_seed(5),
    ).repeat(2, 1)
    return folder, tokens


@pytest.mark.parametrize("ends", [(64, 128, 192, 193), (128, 193), (193,)])
def test_joint_hit_matches_cold_logits_and_exact_snapshot(monkeypatch, tmp_path_factory, ends):
    bind_reference_kernels(monkeypatch)
    folder, tokens = _prefix_checkpoint(tmp_path_factory, 193)
    engine = EagerEngine(load_model(folder), tokens, snapshot_rows=2)
    # The forward whose whole 64-token multiples end at 192 exports it, at its end or inside.
    for frontier, end in pairwise((0, *ends)):
        exports = frontier < 192 == frontier + (end - frontier) // 64 * 64
        captures = (RecurrentExport(192, 2),) if exports else ()
        leader = engine.execute(extend_entry(0, frontier, end, captures=captures))
    # Canonical KV belongs to the same frontier as the recurrent snapshot.
    engine._cache_index_table[1, :192] = engine._cache_index_table[0, :192]
    hit = engine.execute(extend_entry(1, 192, 193, restore_row=2))
    cold_engine = EagerEngine(load_model(folder), tokens)
    cold = cold_engine.execute(extend_entry(1, 0, 193))
    torch.testing.assert_close(hit, cold, rtol=1e-4, atol=5e-5)
    if 192 in ends:
        # The leader's last chunk is the hit's step from the state it exported.
        assert torch.equal(hit, leader)


def test_a_decode_snapshot_matches_the_prefill_state_and_restores_like_one(
    monkeypatch, tmp_path_factory
):
    bind_reference_kernels(monkeypatch)
    folder, tokens = _prefix_checkpoint(tmp_path_factory, 130)
    engine = EagerEngine(load_model(folder), tokens, snapshot_rows=2)
    # Request 0 prefills 60 tokens, then decodes one at a time; the step reaching 128
    # copies its row out. Request 1 prefills the same 128 tokens and exports inside.
    engine.execute(extend_entry(0, 0, 60))
    for position in range(60, 128):
        captures = (RecurrentExport(128, 2),) if position == 127 else ()
        engine.execute(decode_entry(0, position, captures=captures))
    engine.execute(extend_entry(1, 0, 128, captures=(RecurrentExport(128, 3),)))
    pool = engine.recurrent_state
    for layer in range(pool.num_layers):
        for view in pool.layer_views(layer):
            torch.testing.assert_close(view[2], view[3], rtol=1e-4, atol=5e-5)
    # Request 1 restores the decode snapshot over request 0's KV and matches a cold run.
    engine._cache_index_table[1, :128] = engine._cache_index_table[0, :128]
    hit = engine.execute(extend_entry(1, 128, 130, restore_row=2))
    cold = EagerEngine(load_model(folder), tokens).execute(extend_entry(1, 0, 130))
    torch.testing.assert_close(hit, cold, rtol=1e-4, atol=5e-5)


def test_a_text_only_load_builds_the_text_export_without_the_vision_tower(
    tmp_path: Path, stub_flashinfer: None
) -> None:
    """Qwen3.5-9B loaded text only builds what its text-only export builds: every text
    tensor of the image-text model and none of its vision tower's 0.85 GiB."""

    from cti.resolution import ResolvedModel

    load_hf_config(QWEN3_5_9B_CONFIG, tmp_path / "image-text")
    text_only = ResolvedModel.load(str(tmp_path / "image-text"), text_only=True)
    with torch.device("meta"), default_dtype(torch.bfloat16):
        model = create_model(text_only.model_class, text_only.facts, site=site())
    export = _meta_model(load_hf_config(text_only_config(QWEN3_5_9B_CONFIG), tmp_path / "text"))
    image_text = _meta_model(load_hf_config(QWEN3_5_9B_CONFIG, tmp_path / "full"))

    def tensors(built: BaseLLMModel) -> dict[str, tuple[torch.Size, int]]:
        return {name: (t.shape, t.nbytes) for name, t in built.state_dict().items()}

    assert model.visual is None
    assert tensors(model) == tensors(export)
    vision = {
        name: size for name, size in tensors(image_text).items() if name not in tensors(model)
    }
    assert vision and all(name.startswith("visual.") for name in vision)
    assert 0.8 < sum(nbytes for _, nbytes in vision.values()) / (1 << 30) < 0.9
