"""PEFT adapters loaded at launch, on the CPU: the slot their config sizes, each refusal, and
their factors through each model's checkpoint mapping, whole or as a tensor-parallel rank's
windows."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
import safetensors.torch
import torch

from cti.config import ExpertLoRALayout, LoRAConfig
from cti.distributed import Topology
from cti.layers import Adapter
from cti.models import load_peft, peft_lora
from cti.models.base import BaseLLMModel
from cti.models.peft import PeftLayout
from cti.models.qwen3_5.config import Qwen3_5Config
from cti.models.qwen3_5_moe.config import Qwen3_5MoeConfig
from tests.support.peft_fixtures import QWEN3_5_MOE_ADAPTER
from tests.support.qwen3_5 import hf_state as qwen3_5_hf_state
from tests.support.qwen3_5_lora import (
    TEXT_ONLY_PREFIX,
    Qwen3_5LoRAMapping,
    merged_checkpoint,
    random_peft_state,
)
from tests.support.qwen3_5_moe import hf_state as moe_hf_state
from tests.support.tiny_configs import tiny_hf_config
from tests.support.tp_shards import CPU, build_model, load_sharded, qwen3_checkpoint_shapes

_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _adapter(folder: Path, tensors: dict[str, torch.Tensor], **config: object) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    document = {"r": 4, "lora_alpha": 8, "target_modules": _MODULES, "bias": "none", **config}
    (folder / "adapter_config.json").write_text(json.dumps(document))
    safetensors.torch.save_file(tensors, folder / "adapter_model.safetensors")
    return str(folder)


def _slot(
    architecture: str, folder: Path, source: str, **site: object
) -> tuple[BaseLLMModel, Adapter]:
    """The engine model of ``architecture`` and ``source``'s slot, attached to it and
    materialized."""

    hf_config = tiny_hf_config(architecture, folder)
    model_class = type(build_model(hf_config))
    adapters = Adapter(peft_lora(LoRAConfig(adapter=source), model_class.LORA_TARGETS))
    model = build_model(hf_config, lora_adapters=adapters, **site)
    adapters.materialize(device=CPU, dtype=torch.float32)
    return model, adapters


def _random(
    shapes: dict[str, tuple[int, ...]],
    rank: int = 4,
    targets: tuple[str, ...] = ("attention", "mlp"),
) -> dict[str, torch.Tensor]:
    return random_peft_state(
        {name: torch.empty(shape) for name, shape in shapes.items()},
        prefix=TEXT_ONLY_PREFIX,
        targets=targets,
        rank=rank,
        generator=torch.Generator().manual_seed(1),
    )


def test_the_slot_takes_the_adapters_targets_rank_and_scaling(tmp_path: Path) -> None:
    source = _adapter(tmp_path, {}, r=12, lora_alpha=6, target_modules=["q_proj", "up_proj"])

    dense = peft_lora(LoRAConfig(adapter=source), ("attention", "mlp"))
    sparse = peft_lora(LoRAConfig(adapter=source), ("attention", "experts"))

    assert (dense.targets, dense.rank, dense.alpha / dense.rank) == (("attention", "mlp"), 12, 0.5)
    # A per-expert rank pads with zeros to a multiple of 8; the scaling stays the adapter's.
    assert (sparse.targets, sparse.rank, sparse.alpha / sparse.rank) == (
        ("attention", "experts"),
        16,
        0.5,
    )


@pytest.mark.parametrize(
    ("setting", "named"),
    [
        ({"use_dora": True}, "use_dora"),
        ({"use_rslora": True}, "use_rslora"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
        ({"rank_pattern": {"q_proj": 8}}, "rank_pattern"),
        ({"alpha_pattern": {"q_proj": 8}}, "alpha_pattern"),
        ({"layers_to_transform": [0]}, "layers_to_transform"),
        ({"fan_in_fan_out": True}, "fan_in_fan_out"),
        ({"bias": "lora_only"}, "bias 'lora_only'"),
        ({"target_modules": r".*\.q_proj"}, "target_modules other than a list of names"),
    ],
)
def test_each_setting_beyond_b_times_a_is_refused(
    tmp_path: Path, setting: dict, named: str
) -> None:
    with pytest.raises(ValueError, match=rf"unsupported {named}$"):
        peft_lora(LoRAConfig(adapter=_adapter(tmp_path, {}, **setting)), ("attention", "mlp"))


@pytest.mark.parametrize("module", ["lm_head", "embed_tokens", "in_proj_qkv", "gate"])
def test_a_module_no_target_of_the_model_adapts_is_refused(tmp_path: Path, module: str) -> None:
    source = _adapter(tmp_path, {}, target_modules=["q_proj", module])

    with pytest.raises(ValueError, match=f"no LoRA target adapts {module}$"):
        peft_lora(LoRAConfig(adapter=source), ("attention", "mlp"))


def test_a_loaded_adapter_computes_pefts_merge(tmp_path: Path, stub_flashinfer: None) -> None:
    hf_config = tiny_hf_config("Qwen3ForCausalLM", tmp_path)
    shapes = qwen3_checkpoint_shapes(hf_config)
    generator = torch.Generator().manual_seed(0)
    base = {n: torch.randn(s, generator=generator) * s[-1] ** -0.5 for n, s in shapes.items()}
    peft = _random(shapes)
    source = _adapter(tmp_path / "adapter", peft)
    folders = {}
    for name, state in (("adapted", base), ("merged", merged_checkpoint(base, peft, scaling=2.0))):
        folders[name] = tmp_path / name
        folders[name].mkdir()
        safetensors.torch.save_file(state, folders[name] / "model.safetensors")
    adapters = Adapter(peft_lora(LoRAConfig(adapter=source), ("attention", "mlp")))
    adapted, _, _ = load_sharded(hf_config, folders["adapted"], lora_adapters=adapters)
    adapters.materialize(device=CPU, dtype=torch.float32)
    merged, _, _ = load_sharded(hf_config, folders["merged"])

    load_peft(adapted, adapters, source)

    compared = 0
    with torch.inference_mode():
        for (path, op), (_, reference) in zip(adapted.named_ops(), merged.named_ops(), strict=True):
            if op.lora_target is not None:
                x = torch.randn(5, op.local_input_size, generator=generator)
                actual, expected = op.forward(x), reference.forward(x)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5, msg=path)
                compared += 1
    assert compared == 2 * 4


@pytest.mark.parametrize(
    ("architecture", "targets"),
    [
        ("Qwen3_5ForCausalLM", ("attention", "linear_attention", "mlp")),
        ("Qwen3_5MoeForCausalLM", ("attention", "linear_attention", "experts")),
    ],
)
def test_qwen3_5_factors_split_as_the_reference_layout_splits_them(
    tmp_path: Path, stub_flashinfer: None, architecture: str, targets: tuple[str, ...]
) -> None:
    hf_config = tiny_hf_config(architecture, tmp_path)
    state = (qwen3_5_hf_state if architecture == "Qwen3_5ForCausalLM" else moe_hf_state)(hf_config)
    peft = _random({n: t.shape for n, t in state.items()}, targets=targets)
    modules = sorted({key.rsplit(".", 3)[-3] for key in peft})
    source = _adapter(tmp_path / "adapter", peft, target_modules=modules)
    model, adapter = _slot(architecture, tmp_path, source)
    config_class = Qwen3_5Config if architecture == "Qwen3_5ForCausalLM" else Qwen3_5MoeConfig
    reference = Qwen3_5LoRAMapping(config_class.from_hf(hf_config).text, prefix=TEXT_ONLY_PREFIX)

    load_peft(model, adapter, source)

    expected = reference.to_engine(peft)
    state = adapter.state_dict()
    for name, tensor in state.items():
        if name in expected:  # the adapter's rank 4 of the slot's 8 in Qwen3.5-MoE
            rank = tensor[..., :4] if name.endswith("B") else tensor[..., :4, :]
            assert torch.equal(rank, expected[name]), name
        else:  # the routed experts this adapter leaves out: B zero, A as initialized
            assert ".mlp.experts." in name, name
            assert name.endswith("lora_A") or not tensor.any(), name


def test_slices_and_experts_an_adapter_leaves_out_are_zero_and_logged(
    tmp_path: Path, stub_flashinfer: None, caplog: pytest.LogCaptureFixture
) -> None:
    stem = "base_model.model.model.layers.1"
    peft = {
        f"{stem}.self_attn.q_proj.lora_A.weight": torch.randn(4, 32),
        f"{stem}.self_attn.q_proj.lora_B.weight": torch.randn(128, 4),
        f"{stem}.mlp.experts.2.up_proj.lora_A.weight": torch.randn(4, 32),
        f"{stem}.mlp.experts.2.up_proj.lora_B.weight": torch.randn(16, 4),
    }
    source = _adapter(tmp_path / "adapter", peft, target_modules=["q_proj", "up_proj"])
    model, adapter = _slot("Qwen3MoeForCausalLM", tmp_path, source)
    # Transformers 4's per-expert modules: their gate and up have their own A.
    assert adapter.config.expert_layout is ExpertLoRALayout.PER_PROJECTION

    with caplog.at_level(logging.INFO):
        load_peft(model, adapter, source)

    state = adapter.state_dict()
    up = state["model.layers.1.mlp.experts.gate_up_proj.up.lora_B"]
    assert up.shape == (4, 16, 8) and torch.equal(
        up[2, :, :4], peft[f"{stem}.mlp.experts.2.up_proj.lora_B.weight"]
    )
    assert not up[[0, 1, 3]].any() and not up[2, :, 4:].any()
    assert not state["model.layers.0.self_attn.qkv_proj.q.lora_B"].any()
    # A left-out slice keeps its initial A, so it still trains.
    assert state["model.layers.0.self_attn.qkv_proj.q.lora_A"].any()
    (message,) = [r.getMessage() for r in caplog.records if "leaves" in r.getMessage()]
    assert message.endswith(
        "leaves model.layers.*.mlp.experts.down_proj.full.lora_B, "
        "model.layers.*.mlp.experts.gate_up_proj.gate.lora_B, "
        "model.layers.*.mlp.experts.gate_up_proj.up.lora_B, "
        "model.layers.*.self_attn.o_proj.full.lora_B, model.layers.*.self_attn.qkv_proj.k.lora_B, "
        "model.layers.*.self_attn.qkv_proj.q.lora_B, model.layers.*.self_attn.qkv_proj.v.lora_B zero"
    )


def test_a_module_outside_the_slot_is_refused(tmp_path: Path, stub_flashinfer: None) -> None:
    peft = {"base_model.model.model.embed_tokens.lora_A.weight": torch.zeros(4, 32)}
    source = _adapter(tmp_path / "adapter", peft, target_modules=["q_proj"])
    model, adapter = _slot("Qwen3ForCausalLM", tmp_path, source)

    with pytest.raises(ValueError, match="model.embed_tokens.weight"):
        load_peft(model, adapter, source)


@pytest.mark.parametrize("architecture", ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"])
def test_a_tensor_parallel_rank_loads_its_windows(
    tmp_path: Path, stub_flashinfer: None, architecture: str
) -> None:
    hf_config = tiny_hf_config(architecture, tmp_path)
    moe = architecture == "Qwen3MoeForCausalLM"
    peft = _random(
        qwen3_checkpoint_shapes(hf_config), 8, ("attention",) if moe else ("attention", "mlp")
    )
    for layer in range(2 if moe else 0):
        for expert in range(4):
            stem = f"base_model.model.model.layers.{layer}.mlp.experts.{expert}.down_proj"
            peft[f"{stem}.lora_A.weight"] = torch.randn(8, 16)
            peft[f"{stem}.lora_B.weight"] = torch.randn(32, 8)
    source = _adapter(tmp_path / "adapter", peft, r=8)
    whole, whole_adapter = _slot(architecture, tmp_path, source)
    held, held_adapter = _slot(architecture, tmp_path, source, topology=Topology(1, 2))

    load_peft(whole, whole_adapter, source)
    load_peft(held, held_adapter, source)

    state, local = whole_adapter.state_dict(), held_adapter.state_dict()
    attention = "model.layers.1.self_attn"
    # Rank 1 holds the second query head's B rows and o_proj's A columns; the one KV head and
    # each A of a column-parallel projection whole.
    assert torch.equal(
        local[f"{attention}.qkv_proj.q.lora_B"], state[f"{attention}.qkv_proj.q.lora_B"][64:]
    )
    assert torch.equal(
        local[f"{attention}.o_proj.full.lora_A"], state[f"{attention}.o_proj.full.lora_A"][:, 64:]
    )
    for name in ("qkv_proj.k.lora_B", "qkv_proj.q.lora_A", "o_proj.full.lora_B"):
        assert torch.equal(local[f"{attention}.{name}"], state[f"{attention}.{name}"]), name
    if moe:
        name = "model.layers.1.mlp.experts.down_proj.full.lora_A"  # experts 2 and 3
        assert torch.equal(local[name], state[name][2:])
    else:
        name = "model.layers.1.mlp.gate_up_proj.up.lora_B"
        assert torch.equal(local[name], state[name][24:])


def test_a_transformers_5_expert_adapter_loads_pefts_deltas_and_saves_back_exactly(
    tmp_path: Path, stub_flashinfer: None
) -> None:
    """PEFT's own adapter on Transformers 5's fused expert parameters (``target_parameters``):
    each routed expert's shared-A gate/up and its down projection take PEFT's deltas, every
    module the adapter's, and ``PeftLayout.save`` writes the adapter's tensors back."""

    source = str(QWEN3_5_MOE_ADAPTER)
    model, adapter = _slot("Qwen3_5MoeForCausalLM", tmp_path, source)
    assert adapter.config.expert_layout is ExpertLoRALayout.SHARED_A

    load_peft(model, adapter, source)

    state, modules = adapter.state_dict(), adapter.modules()
    deltas = safetensors.torch.load_file(QWEN3_5_MOE_ADAPTER / "peft_deltas.safetensors")
    scaling = adapter.config.alpha / adapter.config.rank
    compared = 0
    for path, expected in deltas.items():
        prefix = path.removeprefix("base_model.model.")
        if not prefix.endswith(("experts", "experts.base_layer")):
            continue
        projection = "gate_up_proj" if prefix.endswith("base_layer") else "down_proj"
        module = modules[f"{prefix.removesuffix('.base_layer')}.{projection}"]
        a, b = state[module.a[0]], state[module.b[0]]
        actual = (torch.cat((b[0] @ a, b[1] @ a), 1) if module.slot_major_b else b @ a) * scaling
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6, msg=path)
        compared += 1
    assert compared == 2 * len(model.model.layers.op_list)

    peft = safetensors.torch.load_file(QWEN3_5_MOE_ADAPTER / "adapter_model.safetensors")
    names = moe_hf_state(tiny_hf_config("Qwen3_5MoeForCausalLM", tmp_path)).keys()
    tensors, target_modules, target_parameters = PeftLayout(model, adapter).save(state, names)
    assert tensors.keys() == peft.keys()
    for key, tensor in peft.items():
        assert torch.equal(tensors[key], tensor), key
    with open(QWEN3_5_MOE_ADAPTER / "adapter_config.json", encoding="utf-8") as file:
        config = json.load(file)
    assert sorted(config["target_modules"]) == list(target_modules)
    assert sorted(config["target_parameters"]) == list(target_parameters)


def test_the_expert_parameters_nest_as_peft_wraps_them_in_any_listed_order(
    tmp_path: Path, stub_flashinfer: None
) -> None:
    """PEFT wraps ``gate_up_proj`` innermost whatever order ``target_parameters`` lists (and
    saves, sorted) them in, so a reversed list loads the same factors."""

    source = tmp_path / "adapter"
    shutil.copytree(QWEN3_5_MOE_ADAPTER, source)
    config = json.loads((source / "adapter_config.json").read_text())
    config["target_parameters"] = sorted(config["target_parameters"])  # down_proj first
    (source / "adapter_config.json").write_text(json.dumps(config))
    loaded = []
    for path in (QWEN3_5_MOE_ADAPTER, source):
        model, adapter = _slot("Qwen3_5MoeForCausalLM", tmp_path / path.name, str(path))
        load_peft(model, adapter, str(path))
        loaded.append(adapter.state_dict())
    assert loaded[0].keys() == loaded[1].keys()
    for name, tensor in loaded[0].items():
        assert torch.equal(tensor, loaded[1][name]), name
