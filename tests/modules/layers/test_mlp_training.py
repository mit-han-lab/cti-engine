from __future__ import annotations

import pytest
import torch

import ctikernels.mlp.training as mlp_kernels
from cti.config import LoRAConfig
from cti.layers import (
    LinearColParallelMerged,
    LinearRowParallel,
    SiluAndMul,
)
from tests.support.tp_shards import site
from tests.support.training_ops import bind_training_references

HIDDEN, WIDTH, RANK = 32, 48, 4


def _mlp(*, adapted: bool) -> tuple[LinearColParallelMerged, SiluAndMul, LinearRowParallel]:
    gate_up = LinearColParallelMerged(
        HIDDEN, [WIDTH, WIDTH], has_bias=False, output_names=("gate", "up"), site=site()
    )
    down = LinearRowParallel(WIDTH, HIDDEN, has_bias=False, site=site())
    generator = torch.Generator().manual_seed(0)
    gate_up.weight = torch.randn(2 * WIDTH, HIDDEN, generator=generator)
    down.weight = torch.randn(HIDDEN, WIDTH, generator=generator)
    if adapted:
        for layer in (gate_up, down):
            lora = layer.attach_lora(LoRAConfig(targets=("mlp",), rank=RANK, alpha=2.0 * RANK))
            lora.materialize(device=torch.device("cpu"), dtype=torch.float32, generator=generator)
            with torch.no_grad():
                for factor in lora.factors:
                    factor.normal_(generator=generator)
    return gate_up, SiluAndMul(WIDTH), down


def test_an_adapted_unquantized_mlp_trains_through_the_one_kernel_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    bind_training_references(monkeypatch)
    calls = []
    fused = mlp_kernels.gated_mlp_lora

    def spy(x, gate_up, down, *factors, scaling):
        calls.append((gate_up, down))
        return fused(x, gate_up, down, *factors, scaling=scaling)

    monkeypatch.setattr(mlp_kernels, "gated_mlp_lora", spy)
    gate_up, activation, down = _mlp(adapted=True)
    x = torch.randn(5, HIDDEN, requires_grad=True)

    y = activation.mlp_training(gate_up, down, x)
    y.sum().backward()

    # The engine's own frozen bases reach the kernel through forward / grad_input alone.
    ((gate_up_base, down_base),) = calls
    assert type(gate_up_base).__name__ == "_DenseBase" == type(down_base).__name__
    fused_grads = (x.grad, *(f.grad for f in (*gate_up.lora.factors, *down.lora.factors)))
    x_nodes = x.detach().clone().requires_grad_()
    for factor in (*gate_up.lora.factors, *down.lora.factors):
        factor.grad = None
    nodes = down.forward_training(activation.forward_training(gate_up.forward_training(x_nodes)))
    nodes.sum().backward()
    torch.testing.assert_close(y, nodes)
    node_grads = (x_nodes.grad, *(f.grad for f in (*gate_up.lora.factors, *down.lora.factors)))
    # The fused backward sums the factor gradients in other FP32 orders (its FP32 partials).
    for fused_grad, node_grad in zip(fused_grads, node_grads, strict=True):
        torch.testing.assert_close(fused_grad, node_grad, rtol=1e-4, atol=1e-5)


def test_an_mlp_without_an_adapter_keeps_the_separate_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind_training_references(monkeypatch)
    monkeypatch.setattr(mlp_kernels, "gated_mlp_lora", pytest.fail)
    gate_up, activation, down = _mlp(adapted=False)
    x = torch.randn(5, HIDDEN, requires_grad=True)

    activation.mlp_training(gate_up, down, x).sum().backward()

    assert x.grad is not None
    # The fused op keeps no SwiGLU output and holds only its gradient in the backward.
    assert SiluAndMul.mlp_training_bytes(gate_up, down, 10, rank=RANK) == (
        10 * (2 * HIDDEN + 6 * WIDTH + 6 * RANK),
        10 * 4 * WIDTH,
    )
    adapted_up, _, adapted_down = _mlp(adapted=True)
    assert SiluAndMul.mlp_training_bytes(adapted_up, adapted_down, 10, rank=RANK) == (
        10 * (2 * HIDDEN + 4 * WIDTH + 6 * RANK),
        10 * 2 * WIDTH,
    )
