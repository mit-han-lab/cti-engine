"""Per-expert LoRA of a sparse block on CPU.

The torch provider adds each route's delta over the grouped GEMMs' slots as merged FP32
weights would, nothing at a zero weight; the layer adapts its routed and shared experts and
composes its block (never the backend's whole-block op)
bitwise the base with a zero adapter and as the merged weights with a non-zero one; two
expert-parallel ranks over gloo, each with its factors' window, sum to one rank. Training
passes FP64 gradcheck in the factors and matches Transformers' block, frozen or with PEFT's
adapters, in FP32; a dropped router derivative, expert delta or shared-expert delta fails the
comparison.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

import cti.layers.activation as activation_module
import cti.moe.base as base_module
from cti.config import LoRAConfig
from cti.distributed.runtime import ParallelGroup
from cti.layers import Adapter, MoELayer
from cti.layers.lora import LoRAProjection
from cti.moe import training
from tests.support import reference_ops
from tests.support.expert_lora import (
    FACTOR_NAMES,
    LoRAExperts,
    NamedExpertLoRA,
    layer_adapter_window,
    random_expert_factors,
    random_layer_adapter,
)
from tests.support.gloo_ranks import gloo_ranks
from tests.support.lora_serving import merge_adapter
from tests.support.moe import BlockReferenceMoe, ReferenceMoe, rank_state, routed_experts
from tests.support.qwen3_5_lora import attach_peft_lora
from tests.support.tp_shards import CPU, site
from tests.support.training_ops import bind_training_references

_EXPERTS, _TOP_K, _HIDDEN, _INTERMEDIATE, _SHARED = 8, 2, 16, 4, 6
_RANK, _ALPHA = 8, 4.0
_SCALING = _ALPHA / _RANK
# Another CPU host's BLAS sums FP32 partials in another order (test_moe.py's bound).
_FP32_REORDERED = {"rtol": 1e-5, "atol": 1e-5}


def _bind_reference_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        activation_module.inference_activation, "silu_and_mul", reference_ops.silu_and_mul
    )


@pytest.fixture(autouse=True)
def _reference_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    _bind_reference_activation(monkeypatch)


def _weights(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    gate_up = torch.randn(_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN, generator=generator)
    down = torch.randn(_EXPERTS, _HIDDEN, _INTERMEDIATE, generator=generator)
    return gate_up * _HIDDEN**-0.5, down * _INTERMEDIATE**-0.5


def _routes(kind: str, tokens: int, generator: torch.Generator) -> torch.Tensor:
    if kind == "one-expert":
        return torch.stack([torch.full((tokens,), 3), torch.full((tokens,), 5)], 1)
    if kind == "distinct":
        return torch.arange(tokens * _TOP_K).remainder(_EXPERTS).view(tokens, _TOP_K)
    return torch.rand(tokens, _EXPERTS, generator=generator).topk(_TOP_K, -1).indices


def _infer(layer: MoELayer, x: torch.Tensor) -> torch.Tensor:
    """The inference forward as the runner calls it: settled, in inference mode, over a
    copy."""

    layer.settle_inference()
    with torch.inference_mode():
        return layer.forward(x.clone())


def _layer(group: ParallelGroup | None = None, backend: ReferenceMoe | None = None) -> MoELayer:
    return MoELayer(
        _EXPERTS,
        _TOP_K,
        _HIDDEN,
        _INTERMEDIATE,
        renormalize=True,
        site=site(group=group, moe_backend=ReferenceMoe() if backend is None else backend),
        shared_expert_intermediate_size=_SHARED,
        lora="experts",
    )


def _adapted(layer: MoELayer) -> Adapter:
    adapters = Adapter(LoRAConfig(targets=("experts",), rank=_RANK, alpha=_ALPHA))
    adapters.attach(layer)
    adapters.materialize(device=CPU, dtype=torch.float32)
    return adapters


def _whole_state(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(t.shape, generator=generator) * t.shape[-1] ** -0.5
        for name, t in _layer().state_dict().items()
    }


def _whole_adapter(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return random_layer_adapter(
        _EXPERTS, _HIDDEN, _INTERMEDIATE, _SHARED, _RANK, generator=generator
    )


def _loaded(state: Mapping[str, torch.Tensor], backend: ReferenceMoe | None = None) -> MoELayer:
    layer = _layer(backend=backend)
    layer.load_state_dict(dict(state))
    return layer


def _merged_layer(
    state: Mapping[str, torch.Tensor], adapter: Mapping[str, torch.Tensor]
) -> MoELayer:
    """A layer whose weights merge ``adapter`` in FP32, its factors then zero."""

    layer = _loaded(state)
    adapters = _adapted(layer)
    adapters.load_state_dict(adapter)
    merge_adapter(adapters, layer.state_dict())
    return layer


def test_the_routed_experts_lora_trains_whole_or_not_at_all() -> None:
    adapters = _adapted(_layer())
    names = tuple(name for name, _ in adapters.named_parameters())
    routed, shared = names[:4], names[4:]
    assert all(name.startswith("experts.") for name in routed)

    adapters.check_training_selection(routed)
    # The shared expert's factors each cast at use: any of them may train alone.
    adapters.check_training_selection(shared[:1])
    with pytest.raises(ValueError, match="train all together or not at all"):
        adapters.check_training_selection((*routed[2:], *shared))


def test_an_adapted_layer_composes_its_block_as_the_base_and_merged_weights() -> None:
    state, x = _whole_state(0), torch.randn(7, _HIDDEN, generator=torch.Generator().manual_seed(1))
    backend = BlockReferenceMoe()
    blocked, adapted = _loaded(state, backend), _loaded(state, backend)
    adapters = _adapted(adapted)
    assert len(backend.requests) == 2
    base = _infer(_loaded(state), x)

    _infer(blocked, x)
    assert backend.calls == 1
    # The op reads no adapter, so the adapted layer composes its block: with new (zero)
    # factors, bitwise the unadapted composition.
    assert torch.equal(_infer(adapted, x), base)
    adapters.load_state_dict(_whole_adapter(2))
    output = _infer(adapted, x)
    merged = _infer(_merged_layer(state, _whole_adapter(2)), x)
    torch.testing.assert_close(output, merged, **_FP32_REORDERED)
    assert float((output - base).abs().max()) > 0.1
    assert backend.calls == 1


def test_adapted_experts_pass_fp64_gradcheck_in_the_input_and_every_factor() -> None:
    generator = torch.Generator().manual_seed(6)
    gate_up, down = (w.double() for w in _weights(generator))
    named = random_expert_factors(_EXPERTS, _HIDDEN, _INTERMEDIATE, 2, generator=generator)
    factors = tuple(named[name].double().requires_grad_() for name in FACTOR_NAMES)
    x = torch.randn(5, _HIDDEN, generator=generator, dtype=torch.float64, requires_grad=True)
    selected = _routes("random", 5, generator)
    weights = torch.rand(5, _TOP_K, generator=generator, dtype=torch.float64)

    def routed(x: torch.Tensor, *factors: torch.Tensor) -> torch.Tensor:
        lora = NamedExpertLoRA(dict(zip(FACTOR_NAMES, factors, strict=True)), _SCALING)
        return routed_experts(x, gate_up, down, weights, selected, lora)

    assert torch.autograd.gradcheck(routed, (x, *factors))


_SHARED_MODULES = (
    ("gate_proj", "gate_up_proj.gate"),
    ("up_proj", "gate_up_proj.up"),
    ("down_proj", "down_proj.full"),
)


def _transformers_block(
    state: Mapping[str, torch.Tensor], adapter: Mapping[str, torch.Tensor] | None
) -> tuple[Qwen3_5MoeSparseMoeBlock, dict[str, torch.Tensor]]:
    """Transformers' block with ``state`` and the PEFT form of ``adapter``, if any, and its
    factors by the engine's names."""

    config = Qwen3_5MoeTextConfig(
        hidden_size=_HIDDEN,
        num_experts=_EXPERTS,
        num_experts_per_tok=_TOP_K,
        moe_intermediate_size=_INTERMEDIATE,
        shared_expert_intermediate_size=_SHARED,
    )
    config._experts_implementation = "eager"
    block = Qwen3_5MoeSparseMoeBlock(config)
    shared_gate, shared_up = state["shared_expert.gate_up_proj.weight"].chunk(2)
    with torch.no_grad():
        for parameter, value in (
            (block.gate.weight, state["gate.weight"]),
            (block.shared_expert.gate_proj.weight, shared_gate),
            (block.shared_expert.up_proj.weight, shared_up),
            (block.shared_expert.down_proj.weight, state["shared_expert.down_proj.weight"]),
            (block.shared_expert_gate.weight, state["shared_expert_gate.weight"]),
            (block.experts.gate_up_proj, state["experts.gate_up_proj"]),
            (block.experts.down_proj, state["experts.down_proj"]),
        ):
            parameter.copy_(value)
    if adapter is None:
        return block, {}

    def peft_name(module: str, factor: str) -> str:
        return f"base_model.model.shared_expert.{module}.lora_{factor}.weight"

    peft = {
        peft_name(module, factor): adapter[f"shared_expert.{name}.lora_{factor}"]
        for module, name in _SHARED_MODULES
        for factor in "AB"
    }
    shared = attach_peft_lora(block, peft, scaling=_SCALING)
    routed = {name: adapter[f"experts.{name}"] for name in FACTOR_NAMES}
    block.experts = LoRAExperts(
        state["experts.gate_up_proj"], state["experts.down_proj"], routed, scaling=_SCALING
    )
    factors = {f"experts.{n}": f for n, f in block.experts.named_factors.items()}
    for module, name in _SHARED_MODULES:
        for factor in "AB":
            factors[f"shared_expert.{name}.lora_{factor}"] = shared[peft_name(module, factor)]
    return block, factors


def _assert_trains_as_transformers(adapted: bool) -> None:
    """The engine's and Transformers' training output, input gradient and factor gradients
    of one FP32 block, frozen or with a non-zero adapter, agree by name."""

    state, adapter = _whole_state(0), _whole_adapter(2) if adapted else None
    layer, engine_factors = _loaded(state), {}
    if adapted:
        adapters = _adapted(layer)
        adapters.load_state_dict(adapter)
        engine_factors = dict(adapters.named_parameters())
    block, reference_factors = _transformers_block(state, adapter)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(1, 10, _HIDDEN, generator=generator)
    upstream = torch.randn(x.shape, generator=generator)
    results = []
    for forward, factors in (
        (layer.forward_training, engine_factors),
        (block, reference_factors),
    ):
        inputs = x.clone().requires_grad_()
        output = forward(inputs)
        tensors = (inputs, *factors.values())
        # A factor a defect leaves out of the graph gets a zero gradient.
        gradients = torch.autograd.grad(output, tensors, upstream, allow_unused=True)
        gradients = [
            g if g is not None else t.new_zeros(t.shape)
            for g, t in zip(gradients, tensors, strict=True)
        ]
        names = ("output", "input", *factors)
        results.append(dict(zip(names, (output.detach(), *gradients), strict=True)))
    engine, reference = results
    assert engine.keys() == reference.keys()
    for name, expected in reference.items():
        assert torch.count_nonzero(expected) > 0, name
        torch.testing.assert_close(engine[name], expected, rtol=1e-4, atol=1e-5, msg=name)


def _detach_router_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """The selected weights no longer differentiate the router logits."""

    route = training.route

    def detached(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        weights, selected = route(*args, **kwargs)
        return weights.detach(), selected

    monkeypatch.setattr(training, "route", detached)
    monkeypatch.setattr(base_module, "route", detached)


def _without_shared_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every adapted linear, here the shared expert's, trains as its base: its MLP through
    the separate products, each without its delta."""

    monkeypatch.setattr(activation_module, "_fused_products", lambda gate_up, down: None)
    monkeypatch.setattr(
        LoRAProjection,
        "training_product",
        lambda self, x, base, bias: base.forward(x) if bias is None else base.forward(x) + bias,
    )


@pytest.mark.parametrize(
    ("adapted", "plant"),
    [
        pytest.param(False, None, id="frozen"),
        pytest.param(False, _detach_router_weights, id="frozen-detached-router-weights"),
        pytest.param(True, None, id="adapted"),
        pytest.param(True, _without_shared_deltas, id="adapted-dropped-shared-delta"),
    ],
)
def test_a_block_trains_as_transformers_with_peft_adapters_in_fp32(
    monkeypatch: pytest.MonkeyPatch, adapted: bool, plant
) -> None:
    bind_training_references(monkeypatch)
    if plant is None:
        _assert_trains_as_transformers(adapted)
        return
    plant(monkeypatch)
    with pytest.raises(AssertionError):
        _assert_trains_as_transformers(adapted)


def _expert_parallel_rank(group: ParallelGroup, argument: tuple) -> list[float]:
    # The rank process exits after the scenario, so its binding is never undone.
    _bind_reference_activation(pytest.MonkeyPatch())
    state, adapter, x = argument
    layer = _layer(group)
    layer.load_state_dict(rank_state(state, layer))
    local = layer_adapter_window(adapter, layer.experts.placement, group.rank, group.size)
    _adapted(layer).load_state_dict(local)
    return _infer(layer, x).flatten().tolist()


@pytest.mark.xdist_group("gloo")
def test_expert_parallel_ranks_with_their_factor_windows_sum_to_one_rank(tmp_path: Path) -> None:
    state, adapter = _whole_state(0), _whole_adapter(2)
    x = torch.randn(6, _HIDDEN, generator=torch.Generator().manual_seed(1))
    layer = _loaded(state)
    _adapted(layer).load_state_dict(adapter)
    expected = _infer(layer, x)

    with gloo_ranks(_expert_parallel_rank, 2, (state, adapter, x), tmp_path) as outputs:
        pass

    for output in outputs:
        torch.testing.assert_close(torch.tensor(output).view_as(x), expected, **_FP32_REORDERED)
