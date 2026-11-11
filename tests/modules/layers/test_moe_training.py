"""The frozen MoE block's training forward on CPU.

Training routes through the experts the backend's inference router selects and weighs them
as transformers 5.16.1's routers do, keeping the FP32 weights; the input gradient passes
FP64 gradcheck; test_moe_lora.py compares the block with Transformers'. Rank partials of the
routed experts sum to the whole block.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3_5MoeTextConfig, Qwen3MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTopKRouter,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from cti.layers import MoELayer, Placement
from cti.moe import training
from tests.support.expert_lora import (
    FACTOR_NAMES,
    NamedExpertLoRA,
    factor_windows,
    random_expert_factors,
)
from tests.support.moe import ReferenceMoe, routed_experts
from tests.support.tp_shards import default_dtype, site
from tests.support.training_ops import bind_training_references

_EXPERTS, _TOP_K, _HIDDEN, _INTERMEDIATE, _SHARED = 8, 2, 16, 8, 12
_TOKENS = 10


@pytest.fixture(autouse=True)
def _reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_training_references(monkeypatch)


def _layer(dtype: torch.dtype, *, seed: int = 0) -> MoELayer:
    with default_dtype(dtype):
        layer = MoELayer(
            _EXPERTS,
            _TOP_K,
            _HIDDEN,
            _INTERMEDIATE,
            renormalize=True,
            site=site(moe_backend=ReferenceMoe()),
            shared_expert_intermediate_size=_SHARED,
        )
    generator = torch.Generator().manual_seed(seed)
    layer.load_state_dict(
        {
            name: (torch.randn(t.shape, generator=generator) * t.shape[-1] ** -0.5).to(dtype)
            for name, t in layer.state_dict().items()
        }
    )
    return layer


def _input(dtype: torch.dtype, *, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(_TOKENS, _HIDDEN, generator=generator).to(dtype)


@pytest.mark.parametrize(
    ("router_class", "renormalize"),
    [(Qwen3_5MoeTopKRouter, True), (Qwen3MoeTopKRouter, True), (Qwen3MoeTopKRouter, False)],
    ids=["qwen3_5_moe", "qwen3_moe", "qwen3_moe-raw"],
)
def test_routing_selects_as_transformers_and_keeps_the_fp32_weights(
    router_class: type[torch.nn.Module], renormalize: bool
) -> None:
    """BF16 logits: FP32 softmax, top-k and renormalization. Transformers then rounds the
    weights to BF16; training keeps them FP32, as the inference top-k gives them."""

    geometry = {"hidden_size": _HIDDEN, "num_experts": _EXPERTS, "num_experts_per_tok": 4}
    if router_class is Qwen3_5MoeTopKRouter:
        config = Qwen3_5MoeTextConfig(**geometry)
    else:
        config = Qwen3MoeConfig(**geometry, norm_topk_prob=renormalize)
    router = router_class(config).to(torch.bfloat16)
    with torch.no_grad():
        router.weight.normal_(generator=torch.Generator().manual_seed(2))
        logits, expected_weights, expected_selected = router(_input(torch.bfloat16))

    weights, selected = training.route(logits, 4, renormalize=renormalize)

    assert weights.dtype == torch.float32
    probabilities = logits.float().softmax(-1).gather(-1, selected)
    expected = probabilities / probabilities.sum(-1, keepdim=True) if renormalize else probabilities
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    torch.testing.assert_close(weights.bfloat16(), expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(selected, expected_selected, rtol=0, atol=0)


class _FixedRoutes(ReferenceMoe):
    """``ReferenceMoe`` whose inference router selects ``ids`` for every row."""

    def __init__(self, ids: torch.Tensor) -> None:
        self.ids = ids

    def route(self, hidden_states, router, topk, renormalize):
        return torch.full(self.ids.shape, float("nan")), self.ids.int()


def test_training_routes_through_the_experts_the_inference_router_selects() -> None:
    layer = _layer(torch.float32)
    experts = layer.experts
    x = _input(torch.float32)
    # Not the top-k: the selection's order flipped and its rows rolled.
    selected = training.route(F.linear(x, layer.gate.weight), _TOP_K, renormalize=True)[1]
    selected = selected.flip(-1).roll(1, dims=0)
    weights, _ = training.route(
        F.linear(x, layer.gate.weight), _TOP_K, renormalize=True, selected=selected
    )
    expected = routed_experts(x, experts.gate_up_proj, experts.down_proj, weights, selected)

    actual = _FixedRoutes(selected).forward_training(
        x, experts.gate_up_proj, experts.down_proj, layer.gate.weight, _TOP_K, True
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _fixed_route_function(layer: MoELayer, selected: torch.Tensor) -> Callable[..., torch.Tensor]:
    experts = layer.experts

    def routed(x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x, layer.gate.weight)
        weights, _ = training.route(logits, _TOP_K, renormalize=True, selected=selected)
        return routed_experts(x, experts.gate_up_proj, experts.down_proj, weights, selected)

    return routed


def _selection(layer: MoELayer, x: torch.Tensor) -> torch.Tensor:
    logits = F.linear(x, layer.gate.weight)
    return training.route(logits, _TOP_K, renormalize=True)[1]


def test_the_block_passes_fp64_gradcheck_through_its_router_and_shared_expert() -> None:
    layer = _layer(torch.float64)
    x = _input(torch.float64).requires_grad_()
    # Gradcheck's perturbations never reorder a token's top-k: its routes are fixed.
    with torch.no_grad():
        logits = F.linear(x, layer.gate.weight)
    probabilities = logits.softmax(-1).sort(-1, descending=True).values
    assert float((probabilities[:, _TOP_K - 1] - probabilities[:, _TOP_K]).min()) > 1e-4
    # A top-k that differs from the selection still routes through the selected experts.
    selected = _selection(layer, x).flip(-1).roll(1, dims=0)

    assert torch.autograd.gradcheck(layer.forward_training, (x,))
    assert torch.autograd.gradcheck(_fixed_route_function(layer, selected), (x,))


def test_bf16_output_sums_the_routes_in_fp32() -> None:
    layer = _layer(torch.bfloat16)
    x = _input(torch.bfloat16)
    experts = layer.experts
    logits = F.linear(x, layer.gate.weight)
    weights, selected = training.route(logits, _TOP_K, renormalize=True)

    routed = routed_experts(x, experts.gate_up_proj, experts.down_proj, weights, selected)

    expected = torch.zeros(_TOKENS, _HIDDEN)
    for token in range(_TOKENS):
        for slot in range(_TOP_K):
            expert = int(selected[token, slot])
            gate, up = F.linear(x[token], experts.gate_up_proj[expert]).chunk(2)
            value = F.linear(F.silu(gate) * up, experts.down_proj[expert])
            expected[token] += value.float() * weights[token, slot]
    torch.testing.assert_close(routed, expected.bfloat16(), rtol=0, atol=0)


def test_an_empty_step_backpropagates_an_empty_gradient() -> None:
    layer = _layer(torch.float32)
    x = torch.empty(0, _HIDDEN, requires_grad=True)

    layer.forward_training(x).sum().backward()

    assert x.grad is not None and x.grad.shape == (0, _HIDDEN)


@pytest.mark.parametrize("name", ["experts.gate_up_proj", "experts.down_proj"])
def test_a_trainable_base_weight_fails_the_training_forward(name: str) -> None:
    layer = _layer(torch.float32)
    layer.state_dict()[name].requires_grad_(True)

    with pytest.raises(ValueError, match="frozen"):
        layer.forward_training(_input(torch.float32))


def _placement(split: str, rank: int, ranks: int) -> Placement:
    """Rank ``rank``'s range of whole experts, or its window of every expert's width."""

    if split == "experts":
        count = _EXPERTS // ranks
        return Placement(
            _EXPERTS, _INTERMEDIATE, range(rank * count, (rank + 1) * count), range(_INTERMEDIATE)
        )
    width = _INTERMEDIATE // ranks
    return Placement(
        _EXPERTS, _INTERMEDIATE, range(_EXPERTS), range(rank * width, (rank + 1) * width)
    )


@pytest.mark.parametrize("split", ["experts", "intermediate"])
def test_rank_partials_sum_to_the_whole_block_and_its_gradients(split: str) -> None:
    """Each rank's stacks, expert map and factor windows give its partial output; the
    partials, their input and router-logit gradients, and their factor gradients placed at
    their windows sum to the whole block's."""

    generator = torch.Generator().manual_seed(4)
    experts = _layer(torch.float64).experts
    whole_factors = {
        name: tensor.double()
        for name, tensor in random_expert_factors(
            _EXPERTS, _HIDDEN, _INTERMEDIATE, 8, generator=generator
        ).items()
    }
    x = _input(torch.float64).requires_grad_()
    router = torch.randn(_EXPERTS, _HIDDEN, generator=generator, dtype=torch.float64)
    logits = F.linear(x, router).detach().requires_grad_()
    upstream = torch.randn(_TOKENS, _HIDDEN, generator=generator, dtype=torch.float64)

    def output_and_gradients(placement: Placement) -> list[torch.Tensor]:
        held = slice(placement.experts.start, placement.experts.stop)
        window = slice(placement.intermediate.start, placement.intermediate.stop)
        gate, up = experts.gate_up_proj[held].chunk(2, dim=1)
        gate_up = torch.cat((gate[:, window], up[:, window]), 1)
        down = experts.down_proj[held][..., window]
        expert_map = None if placement.holds_every_expert else placement.expert_map()
        windows = factor_windows(placement)
        factors = {
            name: tensor[windows[name]].clone().requires_grad_()
            for name, tensor in whole_factors.items()
        }
        weights, selected = training.route(logits, _TOP_K, renormalize=True)
        lora = NamedExpertLoRA(factors, 0.5)
        y = routed_experts(x, gate_up, down, weights, selected, lora, expert_map)
        gradients = torch.autograd.grad(y, (x, logits, *factors.values()), upstream)
        placed = []
        for name, gradient in zip(FACTOR_NAMES, gradients[2:], strict=True):
            whole = gradient.new_zeros(whole_factors[name].shape)
            whole[windows[name]] = gradient
            placed.append(whole)
        return [y.detach(), *gradients[:2], *placed]

    whole = output_and_gradients(_placement(split, 0, 1))
    parts = [output_and_gradients(_placement(split, rank, 2)) for rank in range(2)]
    for expected, *partials in zip(whole, *parts, strict=True):
        assert torch.count_nonzero(expected) > 0
        torch.testing.assert_close(sum(partials), expected)
