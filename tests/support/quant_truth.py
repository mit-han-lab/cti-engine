"""The quantized truth child: ``tests/support/hf_truth.py`` on a dequantized folder
(``quant_gates.dequantized_folder``) whose quantized modules also quantize their inputs as the
engine does, by the numerics contract (``cti/layers/quant/numerics.py``): the
FP32 quantized truth of G-C, or with ``--dtype bfloat16`` its BF16 floor.

Each linear the folder's ``activation_quant.json`` names quantizes and dequantizes its input
before its product, in FP32. A MoE block's routed experts (Transformers' fused module) run this
file's per-expert loop, which does the same to the input of each expert's gate/up and down
products. Takes ``hf_truth``'s arguments; ``--checkpoint`` is the dequantized folder. After the
truth's manifest, writes ``quant_truth.json``: what was hooked and the sources it ran."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.support import hf_truth
from tests.support.paths import PYTHON_ROOT

NUMERICS_PATH = PYTHON_ROOT / "cti" / "layers" / "quant" / "numerics.py"


def load_numerics() -> types.ModuleType:
    """The numerics contract, loaded from its file alone: it imports only torch, so the truth
    child imports nothing of the engine's (no kernels, nothing of what it blocks)."""

    spec = importlib.util.spec_from_file_location("quant_numerics_contract", NUMERICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActivationQuantizer:
    """Quantizes and dequantizes a module's input by the numerics contract, in FP32, and
    hooks a Transformers model's modules to do so (``install``)."""

    def __init__(self) -> None:
        self._contract = load_numerics()

    def fake_quantize(self, x: Any, spec: Mapping[str, Any]) -> Any:
        """``x`` quantized and dequantized by ``spec``'s format, returned in its dtype."""

        import torch

        contract, kind = self._contract, spec["format"]
        if kind == "fp8_block":
            y = contract.dequantize_activation(*contract.quantize_activation(x))
        else:
            scale = torch.tensor(spec["input_scale"], dtype=torch.float32, device=x.device)
            if kind == "fp8_static":
                q = contract.quantize_static_activation(x, scale)
                y = contract.dequantize_static(q, scale)
            elif kind == "nvfp4":
                codes, block = contract.quantize_nvfp4_activation(x, scale)
                y = contract.dequantize_nvfp4_activation(codes, block, scale)
            else:
                raise ValueError(f"unknown activation format {kind!r}")
        return y.to(x.dtype)

    def experts_forward(self, spec: Mapping[str, Any]) -> Any:
        """A routed-experts forward over Transformers' fused ``gate_up_proj`` / ``down_proj``
        whose every product reads its input quantized by ``spec``."""

        import torch
        import torch.nn.functional as F

        quantize = self.fake_quantize

        def forward(self: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
            out = torch.zeros_like(hidden_states)
            # Per-token block scales: quantizing every row once quantizes each expert's rows.
            quantized = quantize(hidden_states, spec)
            for expert in top_k_index.unique().tolist():
                if expert >= self.num_experts:
                    continue
                rows, slots = torch.where(top_k_index == expert)
                gate, up = F.linear(quantized[rows], self.gate_up_proj[expert]).chunk(2, dim=-1)
                inner = quantize(self.act_fn(gate) * up, spec)
                y = F.linear(inner, self.down_proj[expert]) * top_k_weights[rows, slots, None]
                out.index_add_(0, rows, y.to(out.dtype))
            return out

        return forward

    def install(self, model: Any, plan: Mapping[str, Any]) -> dict[str, int]:
        """Hook ``model``'s modules by ``plan``; the counts hooked, or raise on a missing one."""

        modules = dict(model.named_modules())
        missing = [name for name in [*plan["linears"], *plan["experts"]] if name not in modules]
        if missing:
            raise KeyError(f"modules of activation_quant.json not in the model: {missing[:8]}")
        for name, spec in plan["linears"].items():
            modules[name].register_forward_pre_hook(
                lambda _module, args, spec=spec: (self.fake_quantize(args[0], spec), *args[1:])
            )
        for name, kind in plan["experts"].items():
            experts = modules[name]
            forward = types.MethodType(self.experts_forward({"format": kind}), experts)
            # accelerate's offloading hook calls ``_old_forward`` once it has placed the weights.
            if hasattr(experts, "_old_forward"):
                experts._old_forward = forward
            else:
                experts.forward = forward
        return {"linears": len(plan["linears"]), "experts": len(plan["experts"])}


def check_loaded(model: Any, folder: Path, plan: Mapping[str, Any]) -> list[str]:
    """The first hooked linear's and the first routed experts' resident weights equal the
    folder's dequantized tensors (gate before up in Transformers' fused ``gate_up_proj``): the
    class loaded what the folder holds, in its layout. Returns what it compared; an offloaded
    weight is not compared."""

    import torch
    from safetensors import safe_open

    index = json.loads((folder / "model.safetensors.index.json").read_text(encoding="utf-8"))

    def stored(name: str) -> Any:
        with safe_open(folder / index["weight_map"][name], framework="pt") as handle:
            return handle.get_tensor(name)

    modules = dict(model.named_modules())
    checks = []
    for name in list(plan["linears"])[:1]:
        weight = modules[name].weight
        if not weight.is_meta:
            expected = stored(f"{name}.weight").to(weight.dtype)
            if not torch.equal(weight.detach().cpu(), expected):
                raise RuntimeError(f"{name}.weight is not the dequantized folder's")
            checks.append(name)
    for name in list(plan["experts"])[:1]:
        fused = modules[name].gate_up_proj
        if not fused.is_meta:
            parts = [stored(f"{name}.0.{part}_proj.weight") for part in ("gate", "up")]
            if not torch.equal(fused[0].detach().cpu(), torch.cat(parts).to(fused.dtype)):
                raise RuntimeError(f"{name}.gate_up_proj[0] is not the folder's gate and up")
            checks.append(name)
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    checkpoint = Path(argv[argv.index("--checkpoint") + 1])
    output = Path(argv[argv.index("--output") + 1])
    plan = json.loads((checkpoint / "activation_quant.json").read_text(encoding="utf-8"))
    hooked: dict[str, int] = {}
    compared: list[str] = []
    load_reference = hf_truth.load_reference
    quantizer = ActivationQuantizer()

    def load_quantized(*args: Any, **kwargs: Any) -> Any:
        model = load_reference(*args, **kwargs)
        compared.extend(check_loaded(model, checkpoint, plan))
        hooked.update(quantizer.install(model, plan))
        return model

    with patch.object(hf_truth, "load_reference", load_quantized):
        status = hf_truth.main(argv)
    if hooked != {"linears": len(plan["linears"]), "experts": len(plan["experts"])}:
        raise RuntimeError(f"hooked {hooked}, not the plan's modules")
    from tests.support.quant_gates import code_identity

    source = json.loads((checkpoint / "dequantized.json").read_text(encoding="utf-8"))["source"]
    record = {
        "quantized_checkpoint": source,
        "hooked": hooked,
        "compared": compared,
        "plan": plan,
        **code_identity(),
    }
    (output / "quant_truth.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
