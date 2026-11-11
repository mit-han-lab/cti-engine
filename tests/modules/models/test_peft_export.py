"""Trained adapters saved as PEFT adapters (``models.save_peft``), on the CPU: a PEFT 0.19.1
adapter of each form loads and saves back to its very keys and tensors, an adapter PEFT has no
form for is refused, and an engine adapter saved and run by PEFT on Transformers gives the
engine's logits."""

from __future__ import annotations

import json
from pathlib import Path

import peft
import pytest
import safetensors.torch
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from cti.config import ExpertLoRALayout, LoRAConfig
from cti.layers import Adapter
from cti.models import load_peft, peft_lora, save_peft
from cti.models.weight import CheckpointWeights
from tests.support.forward_rows import decode_entry, extend_entry
from tests.support.moe import ReferenceMoe
from tests.support.qwen3_5_engine import EagerEngine, bind_reference_kernels
from tests.support.tiny_configs import tiny_hf_config
from tests.support.tp_shards import CPU, build_model, load_sharded

_ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
# Each form's model and PEFT adapter: Transformers 5's linear modules (the dense MLP's
# gate_proj and up_proj land in the engine's merged gate_up), and its fused routed experts.
_FORMS = {
    "Qwen3ForCausalLM": {"target_modules": [*_ATTENTION, "gate_proj", "up_proj", "down_proj"]},
    "Qwen3MoeForCausalLM": {
        "target_modules": _ATTENTION,
        "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    },
}


def _base(architecture: str, folder: Path) -> PreTrainedModel:
    """A tiny random FP32 checkpoint saved by Transformers in ``folder``."""

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        tiny_hf_config(architecture, folder),
        attn_implementation="eager",
        experts_implementation="eager",
    ).eval()
    model.save_pretrained(folder)
    return model


def _read(folder: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    config = json.loads((folder / "adapter_config.json").read_text())
    return config, safetensors.torch.load_file(folder / "adapter_model.safetensors")


@pytest.mark.parametrize("architecture", list(_FORMS))
def test_a_peft_adapter_saves_back_to_its_keys_and_tensors(
    tmp_path: Path, stub_flashinfer: None, architecture: str
) -> None:
    base = _base(architecture, tmp_path / "base")
    # Non-zero B, rank 8 (a routed expert's rank pads to a multiple of 8).
    config = peft.LoraConfig(r=8, lora_alpha=16, init_lora_weights=False, **_FORMS[architecture])
    peft.get_peft_model(base, config).save_pretrained(tmp_path / "peft")
    targets = type(build_model(base.config)).LORA_TARGETS
    adapters = Adapter(peft_lora(LoRAConfig(adapter=str(tmp_path / "peft")), targets))
    model = build_model(base.config, lora_adapters=adapters)
    adapters.materialize(device=CPU, dtype=torch.float32)
    load_peft(model, adapters, str(tmp_path / "peft"))

    names = CheckpointWeights(str(tmp_path / "base")).names()
    save_peft(str(tmp_path / "saved"), model, adapters, adapters.state_dict(), names)

    expected_config, expected = _read(tmp_path / "peft")
    saved_config, saved = _read(tmp_path / "saved")
    assert saved.keys() == expected.keys()
    for key, tensor in expected.items():
        assert torch.equal(saved[key], tensor), key
    for field in ("r", "lora_alpha", "peft_type", "bias"):
        assert saved_config[field] == expected_config[field], field
    assert sorted(saved_config["target_modules"]) == sorted(expected_config["target_modules"])
    # PEFT keeps its user's module path and lists the parameters in set order (it varies
    # with the hash seed); they nest the same way whatever the order.
    assert {
        name.rsplit(".", 2)[-2] + "." + name.rsplit(".", 1)[-1]
        for name in saved_config["target_parameters"] or ()
    } == {
        name.rsplit(".", 2)[-2] + "." + name.rsplit(".", 1)[-1]
        for name in expected_config.get("target_parameters") or ()
    }


def test_an_adapter_with_its_own_gate_and_up_a_has_no_peft_form(
    tmp_path: Path, stub_flashinfer: None
) -> None:
    base = _base("Qwen3MoeForCausalLM", tmp_path / "base")
    adapters = Adapter(
        LoRAConfig(targets=("experts",), rank=8, expert_layout=ExpertLoRALayout.PER_PROJECTION)
    )
    model = build_model(base.config, lora_adapters=adapters)
    adapters.materialize(device=CPU, dtype=torch.float32)

    with pytest.raises(ValueError, match="have their own A"):
        save_peft(
            str(tmp_path / "saved"),
            model,
            adapters,
            adapters.state_dict(),
            CheckpointWeights(str(tmp_path / "base")).names(),
        )
    assert not (tmp_path / "saved" / "adapter_model.safetensors").exists()


def test_peft_runs_a_saved_engine_adapter_to_the_engines_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind_reference_kernels(monkeypatch)
    base = _base("Qwen3MoeForCausalLM", tmp_path / "base")
    adapters = Adapter(LoRAConfig(targets=("attention", "experts"), rank=8, alpha=16.0))
    model = load_sharded(
        base.config, tmp_path / "base", lora_adapters=adapters, moe_backend=ReferenceMoe()
    )[0]
    adapters.materialize(device=CPU, dtype=torch.float32)
    # An adapter the engine trained, as far as PEFT can tell: every factor non-zero.
    generator = torch.Generator().manual_seed(3)
    adapters.load_state_dict(
        {
            name: torch.randn(tensor.shape, generator=generator) * 0.1
            for name, tensor in adapters.state_dict().items()
        }
    )
    save_peft(
        str(tmp_path / "saved"),
        model,
        adapters,
        adapters.state_dict(),
        CheckpointWeights(str(tmp_path / "base")).names(),
    )

    # PEFT 0.19.1's from_pretrained cannot load target_parameters under Transformers 5.16.1
    # (its weight conversion passes WeightConverter a distributed_operation it no longer
    # takes): PEFT builds the adapter from the saved config and takes the saved factors by
    # key, and its ParamWrapper runs them.
    token_ids = torch.randint(
        base.config.vocab_size, (1, 12), generator=torch.Generator().manual_seed(4)
    )
    with torch.no_grad():
        plain = base(input_ids=token_ids.long()).logits[0]
    config, tensors = _read(tmp_path / "saved")
    # PEFT adapts ``base`` in place.
    adapted = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=config["r"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["target_modules"],
            target_parameters=config["target_parameters"],
        ),
    )
    factors = {key.replace(".weight", ".default.weight"): tensor for key, tensor in tensors.items()}
    missing, unexpected = adapted.load_state_dict(factors, strict=False)
    assert not unexpected and not [key for key in missing if "lora_" in key]

    with torch.no_grad():
        reference = adapted(input_ids=token_ids.long()).logits[0]
    with torch.inference_mode():
        engine = EagerEngine(model, token_ids.int())
        logits = [
            engine.execute(extend_entry(0, 0, 8)),
            *(engine.execute(decode_entry(0, position)) for position in range(8, 12)),
        ]
    # The adapter moves the logits; both run FP32 end to end.
    assert not torch.allclose(plain, reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(torch.cat(logits), reference[7:], rtol=1e-4, atol=5e-5)
