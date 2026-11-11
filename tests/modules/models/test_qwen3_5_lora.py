"""Qwen3.5 dense LoRA on the CPU: adapter layout, PEFT round trip, merged-weight equality."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
import safetensors.torch
import torch
from transformers import PretrainedConfig

from cti.config import LoRAConfig
from cti.layers import Adapter
from cti.models import create_model
from cti.models.base import BaseLLMModel
from cti.models.qwen3_5.config import LayerKind, Qwen3_5Config
from cti.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from cti.models.weight import load_weight
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.qwen3_5 import (
    QWEN3_5_9B_CONFIG,
    TINY_QWEN3_5_CONFIG,
    hf_state,
    load_hf_config,
    text_only_config,
)
from tests.support.qwen3_5_lora import (
    IMAGE_TEXT_PREFIX,
    TEXT_ONLY_PREFIX,
    Qwen3_5LoRAMapping,
    merged_checkpoint,
    random_peft_state,
)
from tests.support.tp_shards import CPU, default_dtype, site

_TARGETS = Qwen3_5ForConditionalGeneration.LORA_TARGETS
_RANK = 4
_ALPHA = 2.0
# Each projection's name inside a decoder layer, by the target that adapts it.
_PROJECTIONS = {
    "attention": ("self_attn.qkv_gate_proj", "self_attn.o_proj"),
    "linear_attention": ("linear_attn.in_proj", "linear_attn.out_proj"),
    "mlp": ("mlp.gate_up_proj", "mlp.down_proj"),
}
# One A per checkpoint module, then the B of each output slice it spans.
_ATTENTION_ADAPTER = {
    "self_attn.qkv_gate_proj": (
        "q_gate.lora_A",
        "q.lora_B",
        "gate.lora_B",
        "k.lora_A",
        "k.lora_B",
        "v.lora_A",
        "v.lora_B",
    ),
    "self_attn.o_proj": ("full.lora_A", "full.lora_B"),
}
_LINEAR_ATTENTION_ADAPTER = {
    "linear_attn.in_proj": (
        "qkv.lora_A",
        "q.lora_B",
        "k.lora_B",
        "v.lora_B",
        "z.lora_A",
        "z.lora_B",
        "b.lora_A",
        "b.lora_B",
        "a.lora_A",
        "a.lora_B",
    ),
    "linear_attn.out_proj": ("full.lora_A", "full.lora_B"),
}
_MLP_ADAPTER = {
    "mlp.gate_up_proj": ("gate.lora_A", "gate.lora_B", "up.lora_A", "up.lora_B"),
    "mlp.down_proj": ("full.lora_A", "full.lora_B"),
}


def _mapping(hf_config: PretrainedConfig, prefix: str) -> Qwen3_5LoRAMapping:
    return Qwen3_5LoRAMapping(Qwen3_5Config.from_hf(hf_config).text, prefix=prefix)


def _adapted_model(
    hf_config: PretrainedConfig,
    targets: tuple[str, ...] = _TARGETS,
    *,
    device: torch.device = CPU,
    dtype: torch.dtype = torch.float32,
) -> tuple[BaseLLMModel, Adapter]:
    adapters = Adapter(LoRAConfig(targets=targets, rank=_RANK, alpha=_ALPHA))
    with torch.device("meta"), default_dtype(dtype):
        model = create_model(
            Qwen3_5ForConditionalGeneration, checkpoint_facts(hf_config), site=site(device=device)
        )
    adapters.attach(model)
    return model, adapters


def _load_checkpoint(
    model: BaseLLMModel,
    checkpoint: Mapping[str, torch.Tensor],
    folder: Path,
) -> None:
    """Save ``checkpoint`` to ``folder`` and load it into ``model`` as the runner does."""

    folder.mkdir()
    safetensors.torch.save_file(dict(checkpoint), folder / "model.safetensors")
    model.load_state_dict(
        dict(
            load_weight(
                str(folder),
                CPU,
                mapping=model.checkpoint_mapping,
                params=model.params(),
            )
        )
    )


def _random_checkpoint(hf_config: PretrainedConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        name: torch.randn(tensor.shape, generator=generator) * tensor.shape[-1] ** -0.5
        for name, tensor in hf_state(hf_config).items()
    }


def _layer_projections(model: BaseLLMModel) -> Iterator[tuple[str, object]]:
    """Yield each decoder layer's adaptable projections by their state_dict stem."""

    for index, layer in enumerate(model.model.layers.op_list):
        if layer.kind is LayerKind.FULL_ATTENTION:
            mixer, names = layer.self_attn, ("qkv_gate_proj", "o_proj")
            stem = f"model.layers.{index}.self_attn"
        else:
            mixer, names = layer.linear_attn, ("in_proj", "out_proj")
            stem = f"model.layers.{index}.linear_attn"
        for name in names:
            yield f"{stem}.{name}", getattr(mixer, name)
        for name in ("gate_up_proj", "down_proj"):
            yield f"model.layers.{index}.mlp.{name}", getattr(layer.mlp, name)


def test_9b_adapter_keeps_one_a_per_checkpoint_module_and_one_b_per_slice(
    tmp_path: Path,
    stub_flashinfer: None,
) -> None:
    hf_config = load_hf_config(QWEN3_5_9B_CONFIG, tmp_path)
    model, adapters = _adapted_model(hf_config, device=torch.device("meta"), dtype=torch.bfloat16)
    adapters.materialize(device=torch.device("meta"), dtype=torch.bfloat16)
    state = adapters.state_dict()
    base_state = model.state_dict()

    expected = []
    for index, kind in enumerate(Qwen3_5Config.from_hf(hf_config).text.layer_types):
        mixer = (
            _ATTENTION_ADAPTER if kind is LayerKind.FULL_ATTENTION else _LINEAR_ATTENTION_ADAPTER
        )
        for projection, names in (*mixer.items(), *_MLP_ADAPTER.items()):
            expected.extend(f"model.layers.{index}.{projection}.{name}" for name in names)
    # 24 Gated DeltaNet layers, 8 full-attention layers and 32 MLPs.
    assert len(adapters.modules()) == 24 * 2 + 8 * 2 + 32 * 2
    assert tuple(state) == tuple(expected)
    assert len(state) == 552
    qkv_gate = "self_attn.qkv_gate_proj"
    assert {
        name: tuple(state[f"model.layers.3.{qkv_gate}.{name}"].shape)
        for name in _ATTENTION_ADAPTER[qkv_gate]
    } == {
        "q_gate.lora_A": (_RANK, 4096),
        "q.lora_B": (4096, _RANK),
        "gate.lora_B": (4096, _RANK),
        "k.lora_A": (_RANK, 4096),
        "k.lora_B": (1024, _RANK),
        "v.lora_A": (_RANK, 4096),
        "v.lora_B": (1024, _RANK),
    }
    in_proj = "model.layers.0.linear_attn.in_proj"
    assert {
        name: state[f"{in_proj}.{name}.lora_B"].shape[0] for name in ("q", "k", "v", "z", "b", "a")
    } == {"q": 2048, "k": 2048, "v": 4096, "z": 4096, "b": 32, "a": 32}
    # Every adapter lives beside the base weight it updates, and only in text layers:
    # the vision tower and the MTP head are never built, so never adapted.
    stems = {name.rsplit(".", 2)[0] for name in state}
    assert all(f"{stem}.weight" in base_state for stem in stems)
    assert all(re.match(r"^model\.layers\.\d+\.", name) for name in state)


@pytest.mark.parametrize("layout", ["image-text", "text-only"])
def test_9b_peft_adapter_names_and_shapes_map_onto_the_engine_adapter(
    tmp_path: Path,
    stub_flashinfer: None,
    layout: str,
) -> None:
    image_text = layout == "image-text"
    hf_config = load_hf_config(
        QWEN3_5_9B_CONFIG if image_text else text_only_config(QWEN3_5_9B_CONFIG), tmp_path
    )
    prefix = IMAGE_TEXT_PREFIX if image_text else TEXT_ONLY_PREFIX
    _, adapters = _adapted_model(hf_config, device=torch.device("meta"), dtype=torch.bfloat16)
    adapters.materialize(device=torch.device("meta"), dtype=torch.bfloat16)
    # PEFT's adapter for every adapted module of the published checkpoint's key table.
    peft_state = random_peft_state(
        hf_state(hf_config),
        prefix=prefix,
        targets=_TARGETS,
        rank=_RANK,
        generator=torch.Generator(),
    )

    engine_state = _mapping(hf_config, prefix).to_engine(peft_state)

    # 24 x 5 Gated DeltaNet, 8 x 4 attention and 32 x 3 MLP modules.
    assert len(peft_state) == 2 * (24 * 5 + 8 * 4 + 32 * 3)
    expected = adapters.state_dict()
    assert engine_state.keys() == expected.keys()
    for name, tensor in engine_state.items():
        assert tensor.shape == expected[name].shape, name


@pytest.mark.parametrize("layout", ["image-text", "text-only"])
def test_peft_adapter_round_trips_through_the_engine_exactly(
    tmp_path: Path,
    stub_flashinfer: None,
    layout: str,
) -> None:
    image_text = layout == "image-text"
    hf_config = load_hf_config(
        TINY_QWEN3_5_CONFIG if image_text else text_only_config(TINY_QWEN3_5_CONFIG), tmp_path
    )
    prefix = IMAGE_TEXT_PREFIX if image_text else TEXT_ONLY_PREFIX
    mapping = _mapping(hf_config, prefix)
    _, adapters = _adapted_model(hf_config)
    adapters.materialize(device=CPU, dtype=torch.float32)
    generator = torch.Generator().manual_seed(1)
    peft_state = random_peft_state(
        _random_checkpoint(hf_config),
        prefix=prefix,
        targets=_TARGETS,
        rank=_RANK,
        generator=generator,
    )
    pointers = {name: tensor.data_ptr() for name, tensor in adapters.state_dict().items()}

    adapters.load_state_dict(mapping.to_engine(peft_state))
    exported = mapping.to_peft(adapters.state_dict())

    # One layer of each kind: 5 + 4 Transformers modules in the mixers, 3 in each MLP.
    assert len(peft_state) == 2 * (5 + 4 + 3 * 2)
    assert exported.keys() == peft_state.keys()
    for name, tensor in peft_state.items():
        assert torch.equal(exported[name], tensor), name
    assert {name: tensor.data_ptr() for name, tensor in adapters.state_dict().items()} == pointers

    # And the other way: any engine adapter is one PEFT adapter, so one A serves q and
    # the gate, and one A serves the Gated DeltaNet's q, k and v.
    engine_state = {
        name: torch.randn(tensor.shape, generator=generator)
        for name, tensor in adapters.state_dict().items()
    }
    restored = mapping.to_engine(mapping.to_peft(engine_state))
    assert restored.keys() == engine_state.keys()
    for name, tensor in engine_state.items():
        assert torch.equal(restored[name], tensor), name


@pytest.mark.parametrize(
    "key",
    [
        "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight",
        "base_model.model.model.visual.merger.linear_fc1.lora_A.weight",
        "base_model.model.mtp.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.0.linear_attn.conv1d.lora_A.weight",
    ],
)
def test_peft_adapters_outside_the_text_projections_have_no_engine_place(
    tmp_path: Path,
    key: str,
) -> None:
    hf_config = load_hf_config(TINY_QWEN3_5_CONFIG, tmp_path)

    with pytest.raises(ValueError, match="no module of the engine|no LoRA target wraps"):
        _mapping(hf_config, IMAGE_TEXT_PREFIX).to_engine({key: torch.zeros(_RANK, 32)})


@pytest.mark.parametrize("target", _TARGETS)
def test_each_target_adapts_exactly_its_projections(
    tmp_path: Path,
    stub_flashinfer: None,
    target: str,
) -> None:
    hf_config = load_hf_config(TINY_QWEN3_5_CONFIG, tmp_path)
    _, adapters = _adapted_model(hf_config, (target,))
    adapters.materialize(device=CPU, dtype=torch.float32)

    stems = {name.rsplit(".", 2)[0] for name in adapters.state_dict()}

    # The tiny model's layer 0 is a Gated DeltaNet layer and layer 1 full attention.
    layers = (0, 1) if target == "mlp" else (1,) if target == "attention" else (0,)
    assert stems == {
        f"model.layers.{index}.{projection}"
        for index in layers
        for projection in _PROJECTIONS[target]
    }


def test_non_zero_adapter_equals_merged_weights_in_every_projection(
    tmp_path: Path,
    stub_flashinfer: None,
) -> None:
    hf_config = load_hf_config(TINY_QWEN3_5_CONFIG, tmp_path / "config")
    checkpoint = _random_checkpoint(hf_config)
    peft_state = random_peft_state(
        checkpoint,
        prefix=IMAGE_TEXT_PREFIX,
        targets=_TARGETS,
        rank=_RANK,
        generator=torch.Generator().manual_seed(1),
    )
    adapted, adapters = _adapted_model(hf_config)
    _load_checkpoint(adapted, checkpoint, tmp_path / "base")
    adapters.materialize(device=CPU, dtype=torch.float32)
    adapters.load_state_dict(_mapping(hf_config, IMAGE_TEXT_PREFIX).to_engine(peft_state))
    with torch.device("meta"), default_dtype(torch.float32):
        merged = create_model(
            Qwen3_5ForConditionalGeneration, checkpoint_facts(hf_config), site=site()
        )
        base = create_model(
            Qwen3_5ForConditionalGeneration, checkpoint_facts(hf_config), site=site()
        )
    _load_checkpoint(
        merged,
        merged_checkpoint(checkpoint, peft_state, scaling=_ALPHA / _RANK),
        tmp_path / "merged",
    )
    _load_checkpoint(base, checkpoint, tmp_path / "unadapted")
    generator = torch.Generator().manual_seed(2)

    compared = []
    for (name, projection), (_, merged_projection), (_, base_projection) in zip(
        _layer_projections(adapted),
        _layer_projections(merged),
        _layer_projections(base),
        strict=True,
    ):
        x = torch.randn(5, merged_projection.local_input_size, generator=generator)
        with torch.inference_mode():
            actual = projection.forward(x)
            expected = merged_projection.forward(x)
            unadapted = base_projection.forward(x)
        # The adapter changes the projection, and by exactly PEFT's merge: the q and
        # gate rows of q_proj land on their own slices, and in_proj_qkv's B on q, k, v.
        assert (actual - unadapted).abs().max() > 0.1, name
        # The low-rank path rounds differently from the merged FP32 weight.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5, msg=name)
        compared.append(name)
    # Two mixer projections and two MLP projections in each of the two layers.
    assert len(compared) == len(adapters.modules()) == 8


def test_adapter_gradients_never_reach_the_frozen_base_weights(
    tmp_path: Path,
    stub_flashinfer: None,
) -> None:
    hf_config = load_hf_config(TINY_QWEN3_5_CONFIG, tmp_path / "config")
    model, adapters = _adapted_model(hf_config)
    _load_checkpoint(model, _random_checkpoint(hf_config), tmp_path / "checkpoint")
    adapters.materialize(device=CPU, dtype=torch.float32)
    adapters.load_state_dict(
        {name: torch.randn_like(tensor) for name, tensor in adapters.state_dict().items()}
    )

    base_state = model.state_dict()
    loss = sum(
        projection.forward(torch.randn(3, base_state[f"{name}.weight"].shape[1])).square().sum()
        for name, projection in _layer_projections(model)
    )
    loss.backward()

    assert not any(tensor.requires_grad for tensor in base_state.values())
    assert all(tensor.grad is None for tensor in base_state.values())
    assert all(
        parameter.grad is not None for parameter in tuple(t for _, t in adapters.named_parameters())
    )
