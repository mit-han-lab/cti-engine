"""Block-FP8 routed experts: the method's stacks under each placement and the reference
provider that serves the CPU against the contract."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cti.config import LoRAConfig
from cti.distributed import Topology
from cti.layers import prepare_inference
from cti.layers.moe import Experts
from cti.layers.placement import ExpertTensorParallel
from cti.layers.quant import numerics
from cti.layers.quant.fp8_block import Fp8Block128Experts
from tests.support.fp8_block import quantized_weight
from tests.support.moe import ReferenceMoe, routed_experts
from tests.support.tp_shards import default_dtype, site

CPU = torch.device("cpu")
_EXPERTS, _TOP_K, _HIDDEN, _INTERMEDIATE = 4, 2, 256, 128


def _stacks() -> dict[str, torch.Tensor]:
    """Every expert's E4M3 projections and scale grids, stacked as the method holds them."""

    gate_up = [quantized_weight(2 * _INTERMEDIATE, _HIDDEN, seed=e) for e in range(_EXPERTS)]
    down = [quantized_weight(_HIDDEN, _INTERMEDIATE, seed=50 + e) for e in range(_EXPERTS)]
    return {
        "gate_up_proj": torch.stack([w for w, _ in gate_up]),
        "gate_up_proj_weight_scale_inv": torch.stack([s for _, s in gate_up]),
        "down_proj": torch.stack([w for w, _ in down]),
        "down_proj_weight_scale_inv": torch.stack([s for _, s in down]),
    }


def _experts(stacks: dict[str, torch.Tensor], topology: Topology) -> Experts:
    with torch.device("meta"), default_dtype(torch.bfloat16):
        experts = Experts(
            _EXPERTS,
            _TOP_K,
            _HIDDEN,
            _INTERMEDIATE,
            renormalize=True,
            site=site(topology, moe_backend=ReferenceMoe()),
        )
        experts.use_method(Fp8Block128Experts)
    held = experts.placement.experts
    for param in experts.params():
        assert param.experts == (None if experts.placement.holds_every_expert else held)
    experts.load_state_dict({name: t[held.start : held.stop] for name, t in stacks.items()})
    prepare_inference([experts], max_rows=8, max_sequences=1, dtype=torch.float32, device=CPU)
    return experts


def _contract(
    x: torch.Tensor, router: torch.Tensor, stacks: dict[str, torch.Tensor]
) -> torch.Tensor:
    """The routed experts in FP64 from the contract's operands and the shared router."""

    logits = F.linear(x, router).float()
    ids = logits.sort(dim=-1, descending=True, stable=True).indices[:, :_TOP_K]
    weights = torch.softmax(logits.gather(-1, ids), dim=-1)
    output = torch.zeros(x.shape, dtype=torch.float64)
    for t in range(x.shape[0]):
        for k in range(_TOP_K):
            expert, h = int(ids[t, k]), x[t]
            for name in ("gate_up_proj", "down_proj"):
                q, s = numerics.quantize_activation(h.to(x.dtype))
                h_hat = numerics.dequantize_activation(q, s).double()
                w_hat = numerics.dequantize_block_weight(
                    stacks[name][expert], stacks[f"{name}_weight_scale_inv"][expert]
                ).double()
                h = h_hat @ w_hat.t()
                if name == "gate_up_proj":
                    gate, up = h.chunk(2)
                    h = F.silu(gate) * up
            output[t] += weights[t, k].double() * h
    return output


@pytest.mark.parametrize("size", [1, 2], ids=["whole", "expert-parallel"])
def test_the_reference_experts_follow_the_contract_under_each_placement(size: int) -> None:
    stacks = _stacks()
    x = torch.randn(5, _HIDDEN).bfloat16()
    router = torch.randn(_EXPERTS, _HIDDEN).bfloat16()

    ranks = [_experts(stacks, Topology(rank, size)) for rank in range(size)]
    y = sum(experts.forward(x.clone(), router).double() for experts in ranks)

    # The down projection's input quantizes to E4M3 (3 mantissa bits) from a BF16 activation
    # here and an FP64 one in the contract: one step of 2^-4 of the largest output bounds it.
    expected = _contract(x, router, stacks)
    torch.testing.assert_close(y, expected, rtol=0, atol=2**-4 * expected.abs().max().item())


def test_an_expert_window_that_cuts_a_scale_block_is_refused() -> None:
    with torch.device("meta"), pytest.raises(ValueError, match="96 intermediate columns cuts"):
        Experts(
            _EXPERTS, _TOP_K, _HIDDEN, 96, renormalize=True, site=site(moe_backend=ReferenceMoe())
        ).use_method(Fp8Block128Experts)


def test_adapted_experts_run_their_lora_over_the_dequantized_stacks() -> None:
    stacks = _stacks()
    experts = _experts(stacks, Topology(0, 1))
    with torch.device("meta"), default_dtype(torch.bfloat16):
        whole = Experts(
            _EXPERTS,
            _TOP_K,
            _HIDDEN,
            _INTERMEDIATE,
            renormalize=True,
            site=site(moe_backend=ReferenceMoe()),
        )
    whole.load_state_dict(
        {
            name: torch.stack(
                [
                    numerics.dequantize_block_weight(weight, scale).bfloat16()
                    for weight, scale in zip(
                        stacks[name], stacks[f"{name}_weight_scale_inv"], strict=True
                    )
                ]
            )
            for name in ("gate_up_proj", "down_proj")
        }
    )
    lora = experts.adapt(LoRAConfig(targets=("experts",), rank=8, alpha=16.0))
    lora.materialize(device=CPU, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        for factor in lora.parameters():
            factor.normal_()
    whole.lora = lora
    x = torch.randn(5, _HIDDEN).bfloat16()
    router = torch.randn(_EXPERTS, _HIDDEN).bfloat16()

    y = experts.forward_training(x, router)
    y.float().sum().backward()

    assert torch.equal(y, whole.forward_training(x, router))
    assert all(factor.grad is not None for factor in lora.parameters())


class _TorchExperts:
    """Stands in for ``BlockFP8MoE``: its forward and ``dx`` by torch over dequantized stacks."""

    def __init__(self, stacks: dict[str, torch.Tensor]) -> None:
        self._stacks = [
            torch.stack(
                [
                    numerics.dequantize_block_weight(weight, scale).bfloat16()
                    for weight, scale in zip(
                        stacks[name], stacks[f"{name}_weight_scale_inv"], strict=True
                    )
                ]
            )
            for name in ("gate_up_proj", "down_proj")
        ]

    def __call__(self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return routed_experts(x, *self._stacks, weights, ids.long())

    def dx(
        self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, dy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, weights = x.detach().requires_grad_(True), weights.detach().requires_grad_(True)
        with torch.enable_grad():
            y = routed_experts(x, *self._stacks, weights, ids.long())
        return torch.autograd.grad(y, (x, weights), dy)


def test_frozen_experts_train_through_the_kernels_forward_and_dx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stacks = _stacks()
    dequantizing, through_kernel = (_experts(stacks, Topology(0, 1)) for _ in range(2))
    monkeypatch.setattr(
        through_kernel.method._call, "frozen_experts", lambda: _TorchExperts(stacks)
    )
    x = torch.randn(5, _HIDDEN).bfloat16()
    router = torch.randn(_EXPERTS, _HIDDEN).bfloat16()
    gradient = torch.randn(5, _HIDDEN).bfloat16()

    grads = []
    for experts in (dequantizing, through_kernel):
        inputs = (x.clone().requires_grad_(True), router.clone().requires_grad_(True))
        y = experts.forward_training(*inputs)
        grads.append((y, *torch.autograd.grad(y, inputs, gradient)))

    assert all(torch.equal(a, b) for a, b in zip(*grads, strict=True))


def test_expert_tensor_parallel_scale_windows_follow_the_intermediate_rows() -> None:
    with torch.device("meta"), default_dtype(torch.bfloat16):
        experts = Experts(
            _EXPERTS,
            _TOP_K,
            _HIDDEN,
            2 * _INTERMEDIATE,
            renormalize=True,
            site=site(Topology(1, 2), experts=ExpertTensorParallel(), moe_backend=ReferenceMoe()),
        )
        experts.use_method(Fp8Block128Experts)

    windows = {
        param.path: [(w.axis, w.start, w.stop) for w in param.windows] for param in experts.params()
    }

    assert windows == {
        "gate_up_proj": [(0, 128, 256)] * 2,
        "down_proj": [(1, 128, 256)],
        "gate_up_proj_weight_scale_inv": [(0, 1, 2)] * 2,
        "down_proj_weight_scale_inv": [(1, 1, 2)],
    }
