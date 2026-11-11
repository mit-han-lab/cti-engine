"""A quantized checkpoint gives the built model's ops their methods before anything loads,
and its weights and scales then load through the generic loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch
import torch.nn.functional as F
from transformers import AutoConfig, Qwen3Config, Qwen3ForCausalLM

from cti.layers import LinearColParallelMerged, prepare_inference
from cti.layers import activation as activation_module
from cti.layers.linear_method import Unquantized
from cti.layers.lora import LoRAProjection
from cti.layers.quant import numerics
from cti.layers.quant.fp8_block import Fp8Block128Linear
from cti.models.mapping import CheckpointMapping, MergeRule
from cti.models.weight import CheckpointWeights, load_weight
from tests.support import quant_checkpoints, reference_ops
from tests.support.fp8_block import contract_product, quantized_weight
from tests.support.lora_serving import lora_delta
from tests.support.tp_shards import build_model, default_dtype, site

CPU = torch.device("cpu")
_GEOMETRY = {
    "vocab_size": 64,
    "hidden_size": 256,
    "intermediate_size": 384,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "max_position_embeddings": 64,
}
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _fp8_checkpoint(folder: Path, *, quantization: bool = True) -> Qwen3Config:
    """A tiny Qwen3 checkpoint whose attention and MLP projections are block FP8."""

    config = Qwen3Config(**_GEOMETRY, architectures=["Qwen3ForCausalLM"])
    tensors = dict(Qwen3ForCausalLM(config).to(torch.bfloat16).state_dict())
    for seed, name in enumerate(sorted(tensors)):
        if name.rpartition(".")[0].rpartition(".")[2] in _PROJECTIONS:
            rows, cols = tensors[name].shape
            weight, scale = quantized_weight(rows, cols, seed=seed)
            tensors[name], tensors[f"{name}_scale_inv"] = weight, scale.bfloat16()
    safetensors.torch.save_file(tensors, folder / "model.safetensors")
    document = config.to_dict()
    if quantization:
        document["quantization_config"] = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "activation_scheme": "dynamic",
        }
    (folder / "config.json").write_text(json.dumps(document))
    return config


def test_an_fp8_checkpoint_gives_its_projections_their_method_and_loads(
    tmp_path: Path, stub_flashinfer: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Warm-up runs the norms and the SiLU, which have no CPU kernels.
    reference_ops.bind_norm_references(monkeypatch)
    monkeypatch.setattr(
        activation_module.inference_activation, "silu_and_mul", reference_ops.silu_and_mul
    )
    config = _fp8_checkpoint(tmp_path)
    model = build_model(config, dtype=torch.bfloat16)
    weights = CheckpointWeights(str(tmp_path))

    methods = weights.methods(CPU)
    # As the runner adopts: on the meta device, in the model dtype.
    with torch.device("meta"), default_dtype(torch.bfloat16):
        summary = methods.adopt(model)
    qkv = dict(model.named_ops())["model.layers.0.self_attn.qkv_proj"]
    assert qkv.weight.is_meta and qkv.method._gemm._dtype == torch.bfloat16
    model.load_state_dict(weights.state_dict(model, CPU, show_progress=False))
    prepare_inference([model], max_rows=4, max_sequences=1, dtype=torch.float32, device=CPU)

    # Per layer: qkv, o, gate|up and down; the embedding and the head stay BF16.
    assert summary == {"Fp8Block128Linear on Fp8BlockReference": 8}
    methods = {path: type(op.method) for path, op in model.named_ops() if hasattr(op, "method")}
    assert {path for path, method in methods.items() if method is Unquantized} == {"lm_head"}
    assert isinstance(qkv.method, Fp8Block128Linear)
    assert qkv.weight_scale_inv.shape == (4, 2)  # q 2, k 1, v 1 blocks of rows
    x = torch.randn(3, 256).bfloat16()
    expected, bound = contract_product(x, qkv.weight, qkv.weight_scale_inv)
    assert ((qkv.forward(x).double() - expected).abs() <= bound).all()


def test_quantized_tensors_without_a_declaration_are_refused_by_module(
    tmp_path: Path, stub_flashinfer: None
) -> None:
    _fp8_checkpoint(tmp_path, quantization=False)

    # Before any model is built.
    with pytest.raises(ValueError, match=r"model\.layers\.0\.\w+\.\w+_proj holds .* no recipe"):
        CheckpointWeights(str(tmp_path)).methods(CPU)


def _split_in_proj(folder: Path) -> LinearColParallelMerged:
    """A GDN-style ``in_proj`` whose q and k slots are block FP8 and b and a BF16, loaded."""

    sizes = {"q": 128, "k": 128, "b": 8, "a": 8}
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = LinearColParallelMerged(
            256, list(sizes.values()), has_bias=False, site=site(), output_names=tuple(sizes)
        )
        layer.split([Fp8Block128Linear, Fp8Block128Linear, Unquantized, Unquantized])
    tensors = {}
    for seed, (name, rows) in enumerate(sizes.items()):
        if name in "qk":
            weight, scale = quantized_weight(rows, 256, seed=seed)
            tensors[f"in_proj_{name}.weight"] = weight
            tensors[f"in_proj_{name}.weight_scale_inv"] = scale.bfloat16()
        else:
            tensors[f"in_proj_{name}.weight"] = torch.randn(rows, 256).bfloat16()
    safetensors.torch.save_file(tensors, folder / "model.safetensors")
    mapping = CheckpointMapping(
        merges=(MergeRule("in_proj", tuple(f"in_proj_{n}" for n in sizes)),)
    )

    params = layer.params(prefix="in_proj")
    layer.load_state_dict(
        dict(load_weight(str(folder), CPU, mapping=mapping, params=params)), prefix="in_proj"
    )
    return layer


def test_a_projection_whose_slots_differ_in_method_splits_and_loads_each_run(
    tmp_path: Path,
) -> None:
    layer = _split_in_proj(tmp_path)
    prepare_inference([layer], max_rows=4, max_sequences=1, dtype=torch.float32, device=CPU)

    # One FP8 part, one BF16 part, each holding its slots of the merged checkpoint tensors.
    assert {param.path: param.merge for param in layer.params(prefix="in_proj")} == {
        "in_proj.qk.weight": ("in_proj.weight", (0, 1)),
        "in_proj.qk.weight_scale_inv": ("in_proj.weight_scale_inv", (0, 1)),
        "in_proj.ba.weight": ("in_proj.weight", (2, 3)),
    }
    x = torch.randn(3, 256).bfloat16()
    y = layer.forward(x)
    qk = layer.qk
    expected, bound = contract_product(x, qk.weight, qk.weight_scale_inv)
    assert ((y[:, :256].double() - expected).abs() <= bound).all()
    torch.testing.assert_close(y[:, 256:], F.linear(x, layer.ba.weight))


def test_a_split_projections_parts_write_their_columns_of_one_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = _split_in_proj(tmp_path)
    prepare_inference([layer], max_rows=4, max_sequences=1, dtype=torch.float32, device=CPU)
    outs = []
    for part in layer.parts:
        apply = part.method.apply

        def recorded(x, bias=None, gate=None, out=None, apply=apply):
            outs.append(out)
            return apply(x, bias, gate, out)

        monkeypatch.setattr(part.method, "apply", recorded)
    x = torch.randn(3, 256).bfloat16()

    y = layer.forward(x)

    # Each part writes its column window of the whole's rows: no concatenation follows.
    assert [(out.data_ptr() - y.data_ptr(), out.stride()) for out in outs] == [
        (0, (272, 1)),
        (256 * y.element_size(), (272, 1)),
    ]
    assert layer.qk.output_row_stride == layer.ba.output_row_stride == 272


def test_a_split_projection_trains_lora_over_its_parts_frozen_bases(tmp_path: Path) -> None:
    layer = _split_in_proj(tmp_path)
    base = layer.method.frozen_product()
    assert (base.out_features, base.in_features) == (272, 256)
    x = torch.randn(3, 256).bfloat16().requires_grad_(True)
    y = layer.method.training(x)
    gradient = torch.randn_like(y)
    (x_gradient,) = torch.autograd.grad(y, x, gradient)

    # The parts' products side by side, then the caller's epilogue once.
    out, seen = torch.empty_like(y), []
    product = base.forward(x.detach(), out=out, epilogue=lambda rows, z: seen.append(rows))
    assert product is out and seen == [slice(0, 3)]
    assert torch.equal(product, y.detach())
    # dX sums the parts' dy @ W_hat, each rounded to BF16 at most twice (2^-8 relative).
    qk = layer.qk
    w_hat = torch.cat(
        [numerics.dequantize_block_weight(qk.weight, qk.weight_scale_inv), layer.ba.weight.float()]
    ).double()
    wanted = gradient.double() @ w_hat
    bound = 2**-7 * (gradient.double().abs() @ w_hat.abs()) + 1e-6
    for x_grad in (base.grad_input(gradient), x_gradient):
        assert ((x_grad.double() - wanted).abs() <= bound).all()
    into = torch.randn(3, 256).bfloat16()
    accumulated = base.grad_input(gradient, out=into.clone(), accumulate=True).double()
    assert ((accumulated - into.double() - wanted).abs() <= bound + 2**-8 * into.abs()).all()

    # LoRA trains through the kernels' op over that base, as the base's product plus the delta.
    generator = torch.Generator().manual_seed(7)
    lora = LoRAProjection(
        input_size=256, output_slices=layer.output_slices, rank=4, scaling=0.5, owns_whole=True
    )
    lora.materialize(device=CPU, dtype=torch.bfloat16, generator=generator)
    with torch.no_grad():
        for factor in lora.factors[1::2]:
            factor.copy_(torch.randn(factor.shape, generator=generator))
    trained = layer.method.lora_training_product(lora)(x)
    reference = lora_delta(lora, layer.method.training(x), x)
    assert torch.equal(trained, reference)
    inputs = (x, *lora.factors)
    (x_grad, *factor_grads) = torch.autograd.grad(trained, inputs, gradient)
    (x_wanted, *factor_wanted) = torch.autograd.grad(reference, inputs, gradient)
    assert all(map(torch.equal, factor_grads, factor_wanted))
    # The op adds the low-rank terms before the base's dX: both within BF16 rounding.
    factors = lora.factors
    delta = 0.5 * torch.cat(
        [b.double() @ a.double() for a, b in zip(factors[::2], factors[1::2], strict=True)]
    )
    wanted = gradient.double() @ (w_hat + delta)
    bound = 2**-6 * (gradient.double().abs() @ (w_hat.abs() + delta.abs())) + 1e-6
    for candidate in (x_grad, x_wanted):
        assert ((candidate.double() - wanted).abs() <= bound).all()


_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_DTYPES |= {"F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}
_FLOATS = {torch.bfloat16, torch.float16, torch.float32}


@pytest.mark.parametrize(
    "stem", ["qwen3_8b_fp8", "qwen3_8_27b_fp8", "qwen3_8_27b_nvfp4", "qwen3_5_35b_a3b_fp8"]
)
def test_each_format_declares_what_its_checkpoint_stores(
    tmp_path: Path, stub_flashinfer: None, stem: str
) -> None:
    documents, headers = quant_checkpoints.load(stem)
    quant_checkpoints.write_headers_only(stem, tmp_path)
    model = build_model(AutoConfig.from_pretrained(tmp_path), dtype=torch.bfloat16)
    with torch.device("meta"), default_dtype(torch.bfloat16):
        CheckpointWeights(str(tmp_path)).methods(CPU).adopt(model)

    # Every checkpoint tensor lands on a declared tensor of its dtype (floats convert), and
    # the parts of each fill its shape exactly, as the loader would place them.
    params = {param.path: param for param in model.params()}
    # A split projection's parts hold slots of its merged tensors.
    held = {
        (param.merge[0], slot): (param.path, local)
        for param in params.values()
        if param.merge is not None
        for local, slot in enumerate(param.merge[1])
    }
    declared = params.keys() | {source for source, _ in held}
    mapping = model.checkpoint_mapping
    rows: dict[str, dict[tuple[int | None, int], torch.Size]] = {}
    for name, header in headers.items():
        placements = mapping.placements(name, declared)
        if not placements:
            continue
        weight = mapping.weight_of(name)
        tensor = torch.empty(header["shape"], dtype=_DTYPES[header["dtype"]], device="meta")
        parts = mapping.split(name, tensor, weight and headers[weight]["shape"][0])
        for placement, part in zip(placements, parts, strict=True):
            path, slot = held.get(
                (placement.name, placement.slot), (placement.name, placement.slot)
            )
            target = params[path]
            assert part.dtype == target.dtype or {part.dtype, target.dtype} <= _FLOATS, name
            rows.setdefault(path, {})[placement.expert, slot] = part.shape
    assert rows.keys() == params.keys()
    for path, parts in rows.items():
        experts = {expert for expert, _ in parts}
        stacked = experts != {None}
        slot_rows = [parts[next(iter(experts)), slot] for slot in sorted({s for _, s in parts})]
        shape = (sum(s[0] if s else 1 for s in slot_rows), *slot_rows[0][1:])
        assert ((len(experts), *shape) if stacked else shape) == params[path].shape, path
