"""MoE training through cti-kernels' fused ops on CUDA, at Qwen3.5-35B-A3B's routing (256
experts, top 8) and widths (hidden 2048, intermediate and shared 512).

A frozen block trains through the whole training op (``moe_block``) and an adapted one's
routed experts through ``fused_experts_lora``; their outputs, input gradients and factor
gradients stay near an FP64 reference over the experts the inference router selects, no
further from it than the torch oracle in BF16 (``tests.support.moe.routed_experts``); two
expert-parallel ranks' partials sum to one rank's; and each op holds no more than the
training memory plan counts for it.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from cti.layers import MoELayer
from cti.moe import training
from cti.moe.fused import FusedMoe
from tests.support.expert_lora import NamedExpertLoRA, random_expert_factors
from tests.support.moe import routed_experts
from tests.support.tp_shards import default_dtype, site, windows

pytestmark = pytest.mark.checkpoints()

DEVICE = torch.device("cuda")
NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, SHARED = 256, 8, 2048, 512, 512
RANK, ALPHA = 16, 32.0
RECEIPT_PREFIX = "CTI_MOE_TRAINING="
# A fused result's relative error from the FP64 reference, over the torch oracle's in BF16:
# measured at most 1.5 on an RTX 5090 and an RTX PRO 6000.
ORACLE_MULTIPLE = 2.0


@dataclass(frozen=True)
class _Rank:
    """A rank whose exit all-reduce leaves the partial sum as it is."""

    rank: int
    size: int

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def _draw(generator: torch.Generator, *shape: int) -> torch.Tensor:
    scale = shape[-1] ** -0.5
    return (scale * torch.randn(*shape, generator=generator, device=DEVICE)).bfloat16()


def _state() -> dict[str, torch.Tensor]:
    """A whole layer's BF16 weights."""

    generator = torch.Generator(device=DEVICE).manual_seed(0)
    return {
        "experts.gate_up_proj": _draw(generator, NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        "experts.down_proj": _draw(generator, NUM_EXPERTS, HIDDEN, INTERMEDIATE),
        "gate.weight": _draw(generator, NUM_EXPERTS, HIDDEN),
        "shared_expert.gate_up_proj.weight": _draw(generator, 2 * SHARED, HIDDEN),
        "shared_expert.down_proj.weight": _draw(generator, HIDDEN, SHARED),
        "shared_expert_gate.weight": _draw(generator, 1, HIDDEN),
    }


def _layer(state: dict[str, torch.Tensor], rank: int | None = None) -> MoELayer:
    group = _Rank(0, 1) if rank is None else _Rank(rank, 2)
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = MoELayer(
            NUM_EXPERTS,
            TOP_K,
            HIDDEN,
            INTERMEDIATE,
            renormalize=True,
            site=site(group=group, device=DEVICE, moe_backend=FusedMoe()),
            shared_expert_intermediate_size=SHARED,
        )
    local, shards = {}, windows(layer)
    for name, tensor in state.items():
        held = shards.get(name, ())
        parts = tensor.chunk(len(held), dim=held[0].axis) if held else ()
        local[name] = (
            torch.cat([w.select(p) for w, p in zip(held, parts, strict=True)], held[0].axis)
            if held
            else tensor
        ).contiguous()
    layer.load_state_dict(local)
    layer.settle_inference()
    return layer


def _inputs(rows: int, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 rows and an output gradient."""

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=generator, device=DEVICE).bfloat16()
    return x, torch.randn(rows, HIDDEN, generator=generator, device=DEVICE).bfloat16()


def _selected(x: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
    """The experts the backend's inference router selects."""

    with torch.no_grad():
        return FusedMoe().route(x, router, TOP_K, True)[1].long()


def _relative(actual: torch.Tensor, reference: torch.Tensor) -> float:
    reference = reference.double()
    return float((actual.double() - reference).abs().max() / reference.abs().max())


def _block(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    selected: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The frozen block in ``dtype`` through the torch oracle, routed as ``selected``."""

    w = {name: tensor.to(dtype) for name, tensor in state.items()}
    logits = F.linear(x, w["gate.weight"])
    weights, _ = training.route(logits, TOP_K, renormalize=True, selected=selected)
    routed = routed_experts(x, w["experts.gate_up_proj"], w["experts.down_proj"], weights, selected)
    gate, up = F.linear(x, w["shared_expert.gate_up_proj.weight"]).chunk(2, -1)
    shared = F.linear(F.silu(gate) * up, w["shared_expert.down_proj.weight"])
    return routed + shared * torch.sigmoid(F.linear(x, w["shared_expert_gate.weight"]))


def _trained(fn, x: torch.Tensor, upstream: torch.Tensor, *leaves: torch.Tensor):
    """``fn(x)`` and the gradients of ``(fn(x) * upstream).sum()`` in ``x`` and ``leaves``."""

    x = x.detach().clone().requires_grad_(True)
    output = fn(x)
    gradients = torch.autograd.grad(output, (x, *leaves), upstream.to(output.dtype))
    return output.detach(), gradients


def _receipt(case: str, errors: dict[str, float]) -> None:
    print(RECEIPT_PREFIX + json.dumps({"case": case, **errors}, sort_keys=True))


@pytest.mark.parametrize("rows", [64, 1024])
def test_a_frozen_block_trains_through_the_whole_op_near_fp64(rows: int) -> None:
    state = _state()
    layer = _layer(state)
    assert layer._whole_training() is not None
    x, upstream = _inputs(rows)
    selected = _selected(x, state["gate.weight"])

    fused = _trained(layer.forward_training, x, upstream)
    oracle = _trained(lambda v: _block(v, state, selected, torch.bfloat16), x, upstream)
    exact = _trained(lambda v: _block(v.double(), state, selected, torch.float64), x, upstream)

    errors = {}
    for name, index in (("output", 0), ("dx", 1)):
        pick = (lambda r: r[0]) if index == 0 else (lambda r: r[1][0])
        errors[f"{name}_fused"] = _relative(pick(fused), pick(exact))
        errors[f"{name}_oracle"] = _relative(pick(oracle), pick(exact))
    _receipt(f"frozen-{rows}", errors)
    for name in ("output", "dx"):
        assert errors[f"{name}_fused"] <= ORACLE_MULTIPLE * errors[f"{name}_oracle"], errors


def _adapted_factors() -> NamedExpertLoRA:
    """BF16 trainable factors by name; ``B @ A * scaling`` about a fifth of a weight."""

    named = random_expert_factors(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE, RANK, generator=torch.Generator().manual_seed(RANK)
    )
    leaves = {}
    for name, drawn in named.items():
        factor = drawn * (0.2 * RANK / ALPHA) if name.endswith("lora_B") else drawn
        leaves[name] = factor.to(DEVICE, torch.bfloat16).contiguous().requires_grad_(True)
    return NamedExpertLoRA(leaves, ALPHA / RANK)


@pytest.mark.parametrize("rows", [64, 1024])
def test_adapted_experts_train_their_factors_through_the_fused_experts_near_fp64(
    rows: int,
) -> None:
    state = _state()
    gate_up, down, router = (
        state["experts.gate_up_proj"],
        state["experts.down_proj"],
        state["gate.weight"],
    )
    lora = _adapted_factors()
    leaves = tuple(lora.factors.values())
    x, upstream = _inputs(rows)
    selected = _selected(x, router)

    def fused(v: torch.Tensor) -> torch.Tensor:
        return FusedMoe().forward_training(v, gate_up, down, router, TOP_K, True, lora=lora)

    def through_oracle(dtype: torch.dtype):
        def run(v: torch.Tensor) -> torch.Tensor:
            factors = NamedExpertLoRA(
                {name: leaf.to(dtype) for name, leaf in lora.factors.items()}, lora.scaling
            )
            weights, _ = training.route(
                F.linear(v, router.to(dtype)), TOP_K, renormalize=True, selected=selected
            )
            return routed_experts(v, gate_up.to(dtype), down.to(dtype), weights, selected, factors)

        return run

    results = {
        "fused": _trained(fused, x, upstream, *leaves),
        "oracle": _trained(through_oracle(torch.bfloat16), x, upstream, *leaves),
        "exact": _trained(
            lambda v: through_oracle(torch.float64)(v.double()), x, upstream, *leaves
        ),
    }
    errors = {}
    names = ("output", "dx", *lora.factors)
    for path in ("fused", "oracle"):
        output, gradients = results[path]
        exact_output, exact_gradients = results["exact"]
        errors[f"output_{path}"] = _relative(output, exact_output)
        for name, actual, reference in zip(names[1:], gradients, exact_gradients, strict=True):
            errors[f"{name}_{path}"] = _relative(actual, reference)
    _receipt(f"adapted-{rows}", errors)
    for name in names:
        assert errors[f"{name}_fused"] <= ORACLE_MULTIPLE * errors[f"{name}_oracle"], (
            name,
            errors,
        )


@pytest.mark.parametrize("adapted", [False, True], ids=["frozen", "adapted"])
def test_two_expert_parallel_ranks_sum_to_one_rank(adapted: bool) -> None:
    state = _state()
    x, upstream = _inputs(512)
    whole = _layer(state)
    ranks = [_layer(state, rank) for rank in range(2)]
    if adapted:
        lora = _adapted_factors()
        frozen = NamedExpertLoRA({n: t.detach() for n, t in lora.factors.items()}, lora.scaling)
        for layer in (whole, *ranks):
            held = frozen.window(layer.experts.placement)
            factors = {name: t.contiguous() for name, t in held.factors.items()}
            layer.experts.lora = NamedExpertLoRA(factors, lora.scaling)
    one = _trained(whole.forward_training, x, upstream)
    partials = [_trained(layer.forward_training, x, upstream) for layer in ranks]
    summed_output = sum(p[0].float() for p in partials)
    summed_dx = sum(p[1][0].float() for p in partials)
    errors = {
        "output": _relative(summed_output, one[0]),
        "dx": _relative(summed_dx, one[1][0]),
    }
    _receipt(f"ranks-{'adapted' if adapted else 'frozen'}", errors)
    # BF16 partials, each rounded once before the sum.
    assert errors["output"] <= 2e-2 and errors["dx"] <= 2e-2, errors


@pytest.mark.parametrize("adapted", [False, True], ids=["frozen", "adapted"])
def test_the_plan_bounds_what_each_op_holds(adapted: bool) -> None:
    state = _state()
    layer = _layer(state)
    if adapted:
        layer.experts.lora = _adapted_factors()
    rows = 4096
    x, upstream = _inputs(rows)
    saved_plan, backward_plan = layer.training_bytes(rows, rank=RANK)
    x = x.requires_grad_(True)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = layer.forward_training(x)
    torch.cuda.synchronize()
    saved = torch.cuda.memory_allocated() - base - output.nbytes
    torch.cuda.reset_peak_memory_stats()
    before_backward = torch.cuda.memory_allocated()
    output.backward(upstream)
    torch.cuda.synchronize()
    backward = torch.cuda.max_memory_allocated() - before_backward
    measured = {"saved": saved, "backward": backward}
    planned = {"saved": saved_plan, "backward": backward_plan}
    _receipt(
        f"plan-{'adapted' if adapted else 'frozen'}",
        {**measured, **{f"{k}_plan": v for k, v in planned.items()}},
    )
    assert saved <= saved_plan and backward <= backward_plan, (measured, planned)
