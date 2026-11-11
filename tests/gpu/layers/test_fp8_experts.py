"""Block-FP8 routed experts on CUDA.

cti-kernels' ``BlockFP8MoE`` agrees with the reference provider (the tests' oracle) within the
two quantized projections' rounding, router ties included; splits over ranks by the expert
map; keeps padding rows at zero; and a captured CUDA graph replays its eager product bit for
bit. Adapted, the fused backend runs the E4M3 stacks as stored through cti-kernels'
``fused_experts_lora``: a zero adapter is bitwise the unadapted experts, a non-zero one
follows the same adapter over the dequantized stacks within the activation quantization,
and training runs straight through the quantizers, repeatably, as the dequantized torch
route trains.
"""

from __future__ import annotations

import gc

import pytest
import torch

from cti.config import LoRAConfig
from cti.distributed import Topology
from cti.layers import prepare_inference
from cti.layers.moe import Experts
from cti.layers.quant import fp8_block, numerics
from cti.layers.quant.fp8_block import Fp8Block128Experts, Fp8ExpertsReference
from cti.moe.fused import FusedMoe
from tests.support.fp8_block import quantized_weight
from tests.support.moe import ReferenceMoe
from tests.support.tp_shards import default_dtype, site

# Synthetic weights: no checkpoint to cache.
pytestmark = pytest.mark.checkpoints()

DEVICE = torch.device("cuda")
_EXPERTS, _TOP_K, _HIDDEN, _INTERMEDIATE = 8, 4, 256, 256


def _stacks() -> dict[str, torch.Tensor]:
    stacks: dict[str, list[torch.Tensor]] = {}
    for expert in range(_EXPERTS):
        for name, rows, cols in (
            ("gate_up_proj", 2 * _INTERMEDIATE, _HIDDEN),
            ("down_proj", _HIDDEN, _INTERMEDIATE),
        ):
            weight, scale = quantized_weight(rows, cols, seed=100 * expert + rows, device=DEVICE)
            stacks.setdefault(name, []).append(weight)
            stacks.setdefault(f"{name}_weight_scale_inv", []).append(scale)
    return {name: torch.stack(parts) for name, parts in stacks.items()}


def _experts(
    stacks: dict[str, torch.Tensor],
    topology: Topology | None = None,
    *,
    max_rows: int = 64,
    stream: torch.cuda.Stream | None = None,
    backend: object | None = None,
) -> Experts:
    layer_site = site(
        topology, device=DEVICE, moe_backend=ReferenceMoe() if backend is None else backend
    )
    with torch.device("meta"), default_dtype(torch.bfloat16):
        experts = Experts(
            _EXPERTS, _TOP_K, _HIDDEN, _INTERMEDIATE, renormalize=True, site=layer_site
        )
        experts.use_method(Fp8Block128Experts)
    held = experts.placement.experts
    experts.load_state_dict({name: t[held.start : held.stop] for name, t in stacks.items()})
    with torch.cuda.stream(stream or torch.cuda.current_stream(DEVICE)):
        prepare_inference(
            [experts],
            max_rows=max_rows,
            max_sequences=1,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
    return experts


def _inputs(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(DEVICE).manual_seed(rows)
    x = torch.randn(rows, _HIDDEN, device=DEVICE, generator=generator) * 3
    router = torch.randn(_EXPERTS, _HIDDEN, device=DEVICE, generator=generator)
    return x.bfloat16(), router.bfloat16()


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.double() - expected.double()).norm() / expected.double().norm())


@pytest.mark.parametrize(
    ("rows", "ties"),
    [(1, False), (3, False), (8, False), (64, False), (300, False), (3, True)],
    ids=["1", "3", "8", "64", "300", "ties"],
)
def test_the_kernel_agrees_with_the_reference(
    monkeypatch: pytest.MonkeyPatch, rows: int, ties: bool
) -> None:
    stacks = _stacks()
    x, router = _inputs(rows)
    if ties:
        # Every logit ties: both route each row to the lowest experts.
        router = torch.zeros_like(router)
    with torch.inference_mode():
        kernel = _experts(stacks, max_rows=rows).forward(x, router)
        monkeypatch.setattr(
            fp8_block,
            "EXPERTS_PROVIDERS",
            {**fp8_block.EXPERTS_PROVIDERS, "cuda": Fp8ExpertsReference},
        )
        reference = _experts(stacks, max_rows=rows).forward(x, router)

    assert kernel.dtype == torch.bfloat16 and kernel.shape == x.shape
    # Both round each projection's output to BF16 before the second quantizer, apart.
    assert _relative(kernel, reference) < 4e-2


def test_each_rank_adds_only_its_experts_routes() -> None:
    stacks = _stacks()
    x, router = _inputs(16)
    with torch.inference_mode():
        whole = _experts(stacks).forward(x, router)
        shares = [_experts(stacks, Topology(rank, 2)).forward(x, router) for rank in range(2)]

    total = shares[0].float() + shares[1].float()
    torch.testing.assert_close(total, whole.float(), rtol=2**-7, atol=2**-7 * whole.abs().max())


def test_padding_rows_give_zero_and_never_nan() -> None:
    experts = _experts(_stacks())
    x, router = _inputs(4)
    x[1:] = 0

    with torch.inference_mode():
        y = experts.forward(x, router)

    assert torch.isfinite(y).all()
    assert torch.equal(y[1:], torch.zeros_like(y[1:]))


# 37 rows is no row count the warm-up visits: the eager forward before capture readies it.
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 16, 37])
def test_a_captured_graph_replays_the_kernels_eager_product(rows: int) -> None:
    stream = torch.cuda.Stream(DEVICE)
    stream.wait_stream(torch.cuda.current_stream(DEVICE))
    experts = _experts(_stacks(), stream=stream)
    x, router = _inputs(rows)
    with torch.cuda.stream(stream), torch.inference_mode():
        eager = experts.forward(x, router)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = experts.forward(x, router)
        # Earlier tests' layers hold reference cycles: free them before measuring.
        gc.collect()
        allocated = torch.cuda.memory_allocated(DEVICE)
        graph.replay()
    stream.synchronize()

    assert torch.equal(captured, eager)
    assert torch.cuda.memory_allocated(DEVICE) == allocated


def _straight_through(x: torch.Tensor) -> torch.Tensor:
    """``x`` quantized and dequantized per token and 128 columns, the gradient passed through."""

    q = numerics.dequantize_activation(*numerics.quantize_activation(x)).to(x.dtype)
    return x + (q - x).detach()


def _contract_training(
    experts: Experts, x: torch.Tensor, router: torch.Tensor, stacks: dict[str, torch.Tensor]
) -> torch.Tensor:
    """The experts' forward by the contract in torch, both quantizers passed straight through:
    each route's projections in FP64 from the dequantized stacks, rounded to BF16 where the
    kernel rounds them, the weighted routes summed in FP32 and rounded once."""

    weights, selected = experts.moe_backend.training_route(
        x, router, experts.top_k, experts.renormalize
    )
    dequantized = {
        name: [
            numerics.dequantize_block_weight(w, s).double()
            for w, s in zip(stacks[name], stacks[f"{name}_weight_scale_inv"], strict=True)
        ]
        for name in ("gate_up_proj", "down_proj")
    }
    rows = []
    for token in range(x.shape[0]):
        x_q = _straight_through(x[token : token + 1]).double()
        total = torch.zeros(1, x.shape[1], dtype=torch.float32, device=x.device)
        for k in range(selected.shape[1]):
            expert = int(selected[token, k])
            gate, up = (x_q @ dequantized["gate_up_proj"][expert].t()).bfloat16().chunk(2, -1)
            act = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
            down = _straight_through(act).double() @ dequantized["down_proj"][expert].t()
            total = total + down.bfloat16().float() * weights[token, k]
        rows.append(total)
    return torch.cat(rows).bfloat16()


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("rows", [1, 8, 64])
def test_frozen_experts_train_through_the_kernel_by_the_contract(rows: int, seed: int) -> None:
    stacks = _stacks()
    x, router = _inputs(rows)
    generator = torch.Generator(DEVICE).manual_seed(seed)
    gradient = torch.randn(rows, _HIDDEN, device=DEVICE, generator=generator).bfloat16()
    experts = _experts(stacks, max_rows=rows)

    def train(forward) -> tuple[torch.Tensor, ...]:
        inputs = (x.clone().requires_grad_(True), router.clone().requires_grad_(True))
        y = forward(*inputs)
        return (y, *torch.autograd.grad(y, inputs, gradient))

    kernel = train(experts.forward_training)
    contract = train(lambda x, router: _contract_training(experts, x, router, stacks))

    # The same quantized operands, the gradient through both quantizers. A BF16 rounding
    # apart before the second quantizer flips its codes, as against the reference product
    # (test_the_kernel_agrees_with_the_reference); the routing weights' gradients are dot
    # products whose cancellation magnifies those flips at a few rows.
    (y, x_gradient, router_gradient), (y_c, x_gradient_c, router_gradient_c) = kernel, contract
    assert _relative(y, y_c) < 4e-2
    assert _relative(x_gradient, x_gradient_c) < 4e-2
    assert _relative(router_gradient, router_gradient_c) < 1e-1


def _adapted(experts: Experts, *, zero: bool = False) -> None:
    """Attach rank-16 factors whose delta is well above the E4M3 rows' rounding, or zero B (a
    new adapter)."""

    lora = experts.adapt(LoRAConfig(targets=("experts",), rank=16, alpha=32.0))
    lora.materialize(device=DEVICE, dtype=torch.bfloat16, generator=torch.Generator())
    generator = torch.Generator(DEVICE).manual_seed(7)
    with torch.no_grad():
        for index, factor in enumerate(lora.parameters()):
            factor.normal_(generator=generator).mul_(factor.shape[-1] ** -0.5)
            if index % 2:  # each projection's B
                factor.mul_(0.0 if zero else 0.4)


def _dequantized_experts(stacks: dict[str, torch.Tensor], experts: Experts) -> Experts:
    """BF16 experts over ``stacks`` dequantized, on the fused backend, sharing ``experts``' LoRA."""

    layer_site = site(None, device=DEVICE, moe_backend=FusedMoe())
    with torch.device("meta"), default_dtype(torch.bfloat16):
        whole = Experts(_EXPERTS, _TOP_K, _HIDDEN, _INTERMEDIATE, renormalize=True, site=layer_site)
    whole.load_state_dict(
        {
            name: torch.stack(
                [
                    numerics.dequantize_block_weight(w, s).bfloat16()
                    for w, s in zip(stacks[name], stacks[f"{name}_weight_scale_inv"], strict=True)
                ]
            )
            for name in ("gate_up_proj", "down_proj")
        }
    )
    whole.lora = experts.lora
    return whole


@pytest.mark.parametrize("rows", [1, 4, 64, 300])
def test_a_zero_adapter_on_e4m3_stacks_is_bitwise_the_unadapted_experts(rows: int) -> None:
    stacks = _stacks()
    x, router = _inputs(rows)
    with torch.inference_mode():
        unadapted = _experts(stacks, max_rows=rows, backend=FusedMoe()).forward(x.clone(), router)
    experts = _experts(stacks, max_rows=rows, backend=FusedMoe())
    _adapted(experts, zero=True)
    with torch.inference_mode():
        adapted = experts.forward(x.clone(), router)

    assert torch.equal(adapted, unadapted)


@pytest.mark.parametrize("rows", [1, 4, 64, 300])
def test_adapted_e4m3_stacks_run_as_stored_with_their_lora(rows: int) -> None:
    stacks = _stacks()
    x, router = _inputs(rows)
    experts = _experts(stacks, max_rows=rows, backend=FusedMoe())
    _adapted(experts)
    whole = _dequantized_experts(stacks, experts)
    with torch.inference_mode():
        output = experts.forward(x.clone(), router)
        expected = whole.forward(x.clone(), router)
        base = _experts(stacks, max_rows=rows, backend=FusedMoe()).forward(x.clone(), router)
        whole.lora = None
        base_expected = whole.forward(x.clone(), router)

    # The E4M3 product quantizes its rows 1x128, where the dequantized route keeps them in
    # BF16: within twice the unadapted experts' own error (cti-kernels' bound); a dropped or
    # misplaced delta misses by far more.
    floor = _relative(base, base_expected)
    error = _relative(output, expected)
    assert error <= 2 * floor, (error, floor)
    assert _relative(base, expected) > 4 * error


@pytest.mark.parametrize("rows", [4, 64])
def test_e4m3_lora_trains_straight_through_as_the_dequantized_route(rows: int) -> None:
    stacks = _stacks()
    x, router = _inputs(rows)
    experts = _experts(stacks, max_rows=rows, backend=FusedMoe())
    _adapted(experts)
    whole = _dequantized_experts(stacks, experts)
    factors = experts.lora.parameters()

    def train(layer: Experts) -> tuple[torch.Tensor, ...]:
        for factor in factors:
            factor.grad = None
        rows_in = x.clone().requires_grad_()
        y = layer.forward_training(rows_in, router)
        y.float().square().sum().backward()
        return (y.detach(), rows_in.grad, *(factor.grad.clone() for factor in factors))

    first, second = train(experts), train(experts)
    reference = train(whole)

    # Deterministic Triton steps: repeat runs agree bit for bit.
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    # Straight through the activation quantizers: the 1x128 E4M3 rows carry 3-4% per element
    # into every gradient, which a few rows do not average (cti-kernels' fixed 6e-2 was
    # empirical), so each gradient (dx, which carries the router weights' too, and every
    # factor's) stays within twice the forward's own error against the BF16 route.
    floor = _relative(first[0], reference[0])
    names = ("dx", *range(len(factors)))
    for name, actual, expected in zip(names, first[1:], reference[1:], strict=True):
        assert _relative(actual, expected) <= 2 * floor, (name, _relative(actual, expected), floor)


@pytest.mark.parametrize("rows", [1, 16, 64, 65, 300])
def test_adapted_e4m3_experts_join_the_shared_expert_as_the_torch_join(rows: int) -> None:
    # cti-kernels joins in its combine (the fused down-expand at decode, the route sum above):
    # r(r(r(sigmoid(gate)) * shared) + routed), the order of the layer's torch join.
    stacks = _stacks()
    x, router = _inputs(rows)
    generator = torch.Generator(DEVICE).manual_seed(rows + 1)
    shared = (torch.randn(rows, _HIDDEN, device=DEVICE, generator=generator) * 2).bfloat16()
    gate = torch.randn(rows, 1, device=DEVICE, generator=generator).bfloat16()
    stream = torch.cuda.Stream(DEVICE)
    stream.wait_stream(torch.cuda.current_stream(DEVICE))
    experts = _experts(stacks, max_rows=rows, backend=FusedMoe(), stream=stream)
    _adapted(experts)
    assert experts.method.joins_shared()
    with torch.cuda.stream(stream), torch.inference_mode():
        routed = experts.forward(x.clone(), router)
        expected = routed.add_(shared * torch.sigmoid(gate))
        joined = experts.forward(x.clone(), router, shared, gate)
        static_x = x.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = experts.forward(static_x, router, shared, gate)
        static_x.copy_(x)
        graph.replay()
    stream.synchronize()

    assert torch.equal(joined, expected)
    assert torch.equal(captured, joined)
