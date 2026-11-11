"""Writes tests/modules/models/peft_adapters/qwen3_5_moe: a PEFT 0.19.1 adapter on Transformers
5.16.1's tiny text-only Qwen3.5-MoE (``TINY_QWEN3_5_MOE_CONFIG``) adapting every target, the
routed experts through ``target_parameters``, with non-zero B; beside it peft_deltas.safetensors,
PEFT's own ``get_delta_weight`` of every adapted module and expert parameter by module path.

    python -m tests.support.peft_fixtures  # in an env with peft==0.19.1
"""

from __future__ import annotations

from pathlib import Path

import safetensors.torch
import torch

from tests.support.qwen3_5_moe import TINY_QWEN3_5_MOE_CONFIG, text_only_config

QWEN3_5_MOE_ADAPTER = (
    Path(__file__).parents[1] / "modules" / "models" / "peft_adapters" / "qwen3_5_moe"
)
_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b",
            "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj")  # fmt: skip


def main() -> None:
    import peft
    import transformers

    assert peft.__version__ == "0.19.1" and transformers.__version__ == "5.16.1"
    config = transformers.Qwen3_5MoeTextConfig(**text_only_config(TINY_QWEN3_5_MOE_CONFIG))
    torch.manual_seed(0)
    lora = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=list(_MODULES),
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )
    adapted = peft.get_peft_model(transformers.Qwen3_5MoeForCausalLM(config), lora)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    adapted.save_pretrained(QWEN3_5_MOE_ADAPTER)
    # PEFT ends the config without a newline, which the repository's end-of-file check wants.
    config = QWEN3_5_MOE_ADAPTER / "adapter_config.json"
    config.write_text(config.read_text().rstrip("\n") + "\n")
    deltas = {
        name: module.get_delta_weight("default").contiguous()
        for name, module in adapted.named_modules()
        if isinstance(module, peft.tuners.lora.LoraLayer) and "default" in module.lora_A
    }
    safetensors.torch.save_file(deltas, QWEN3_5_MOE_ADAPTER / "peft_deltas.safetensors")


if __name__ == "__main__":
    main()
