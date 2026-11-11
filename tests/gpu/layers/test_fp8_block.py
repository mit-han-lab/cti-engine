"""The block-FP8 method on CUDA.

cti-kernels' ``BlockFP8Linear`` serves CUDA without scratch and stays within the contract's
BF16 bound, the method adding the bias once; the reference provider, the tests' oracle, shares
one scratch across layers. A captured CUDA graph replays either provider's eager product bit for bit.
"""

from __future__ import annotations

import gc

import pytest
import torch

from cti.config import LoRAConfig
from cti.layers import LinearColParallelMerged, LinearReplicated, prepare_inference
from cti.layers.linear_method import Unquantized
from cti.layers.quant import fp8_block, numerics
from cti.layers.quant.fp8_block import Fp8Block128Linear, Fp8BlockReference
from ctikernels.linear.block_fp8 import BlockFP8Linear, quantize_per_token_group_fp8
from tests.support.fp8_block import contract_product, quantized_weight
from tests.support.tp_shards import default_dtype, site

# Synthetic weights: no checkpoint to cache.
pytestmark = pytest.mark.checkpoints()

DEVICE = torch.device("cuda")


def _stack(hidden: int = 256, inner: int = 384) -> tuple[LinearReplicated, LinearReplicated]:
    """Two block-FP8 layers, the second with a bias; built in BF16 like a served model."""

    with torch.device("meta"), default_dtype(torch.bfloat16):
        up = LinearReplicated(hidden, inner, has_bias=False, site=site(device=DEVICE))
        down = LinearReplicated(inner, hidden, has_bias=True, site=site(device=DEVICE))
        for layer in (up, down):
            layer.use_method(Fp8Block128Linear)
    for seed, layer in enumerate((up, down)):
        rows, cols = layer.local_output_size, layer.local_input_size
        layer.weight, layer.weight_scale_inv = quantized_weight(
            rows, cols, seed=seed, device=DEVICE
        )
    down.bias = torch.randn(hidden, device=DEVICE).bfloat16()
    return up, down


def _prepared(layers, stream: torch.cuda.Stream, *, max_rows: int = 16) -> int:
    stream.wait_stream(torch.cuda.current_stream(DEVICE))
    with torch.cuda.stream(stream):
        return prepare_inference(
            layers,
            max_rows=max_rows,
            max_sequences=1,
            dtype=torch.bfloat16,
            device=DEVICE,
        )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("shape", [(256, 384), (300, 520)], ids=["aligned", "ragged"])
def test_the_kernel_serves_cuda_within_the_contract_bound(
    shape: tuple[int, int], seed: int
) -> None:
    torch.manual_seed(seed)
    stream = torch.cuda.Stream(DEVICE)
    up, down = _stack(*shape)
    size = _prepared((up, down), stream)
    x = (torch.randn(5, shape[0], device=DEVICE) * 4).bfloat16()

    with torch.cuda.stream(stream), torch.inference_mode():
        hidden = up.forward(x)
        y = down.forward(hidden)
        unbiased = down.method._gemm(hidden)
    stream.synchronize()

    # The shared GEMM workspace holds the largest any layer's kernel stages for the rows it
    # warmed; no layer keeps its own.
    assert size == max(
        BlockFP8Linear(layer.weight, layer.weight_scale_inv).workspace_bytes(16)
        for layer in (up, down)
    )
    for layer, inputs, output in ((up, x, hidden), (down, hidden, y)):
        bias = None if layer.bias is None else layer.bias.cpu()
        # The kernel's activation quantizer is the contract's, bit for bit.
        kernel_codes, kernel_scales = quantize_per_token_group_fp8(inputs)
        codes, scales = numerics.quantize_activation(inputs.cpu())
        assert torch.equal(kernel_codes.cpu().view(torch.uint8), codes.view(torch.uint8))
        assert torch.equal(kernel_scales.cpu(), scales)
        expected, bound = contract_product(
            inputs.cpu(), layer.weight.cpu(), layer.weight_scale_inv.cpu(), bias
        )
        assert ((output.double().cpu() - expected).abs() <= bound).all()
    # The method adds the bias once, after the kernel's product.
    assert torch.equal(y, unbiased + down.bias)


@pytest.mark.parametrize("provider", ["kernel", "reference"])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 16])
def test_a_captured_graph_replays_the_eager_product(
    monkeypatch: pytest.MonkeyPatch, provider: str, rows: int
) -> None:
    if provider == "reference":
        monkeypatch.setattr(
            fp8_block, "PROVIDERS", {**fp8_block.PROVIDERS, "cuda": Fp8BlockReference}
        )
    stream = torch.cuda.Stream(DEVICE)
    up, down = _stack()
    size = _prepared((up, down), stream)
    x = torch.randn(rows, 256, device=DEVICE).bfloat16()
    with torch.cuda.stream(stream), torch.inference_mode():
        eager = down.forward(up.forward(x))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = down.forward(up.forward(x))
        # Earlier tests' layers hold reference cycles: free them before measuring.
        gc.collect()
        allocated = torch.cuda.memory_allocated(DEVICE)
        graph.replay()
    stream.synchronize()

    assert torch.equal(captured, eager)
    assert torch.cuda.memory_allocated(DEVICE) == allocated
    if provider == "reference":
        # Both layers dequantize into one scratch, sized for the larger weight in BF16.
        assert size == 384 * 256 * 2
        assert len({layer.services.gemm_workspace.data_ptr() for layer in (up, down)}) == 1


@pytest.mark.parametrize("rows", [1, 7, 300])
def test_training_runs_the_kernels_frozen_product_by_the_contract(rows: int) -> None:
    up, _ = _stack()
    generator = torch.Generator(DEVICE).manual_seed(rows)
    x = (torch.randn(rows, 256, device=DEVICE, generator=generator) * 4).bfloat16()
    gradient = torch.randn(rows, 384, device=DEVICE, generator=generator).bfloat16()
    inputs = x.clone().requires_grad_(True)

    y = up.forward_training(inputs)
    (x_gradient,) = torch.autograd.grad(y, inputs, gradient)

    expected, bound = contract_product(x, up.weight, up.weight_scale_inv)
    assert ((y.double() - expected).abs() <= bound).all()
    # dY @ W_hat, the activation's quantizer passed straight through: each block of W decoded
    # exactly, summed in FP32 and rounded once to BF16.
    w_hat = numerics.dequantize_block_weight(up.weight, up.weight_scale_inv).double()
    reference = gradient.double() @ w_hat
    tolerance = 2**-8 * (gradient.double().abs() @ w_hat.abs()) + 1e-6
    assert ((x_gradient.double() - reference).abs() <= tolerance).all()


@pytest.mark.parametrize("rows", [1, 2048])
def test_an_adapted_block_fp8_projection_adds_the_kernels_delta_within_its_contract(
    rows: int,
) -> None:
    # 35B-A3B-FP8's attention q+gate|k|v projection, adapted at rank 16; its warmed delta
    # replays in a captured graph as eagerly.
    slots, cols = (8192, 512, 512), 2048
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = LinearColParallelMerged(cols, list(slots), has_bias=False, site=site(device=DEVICE))
        layer.use_method(Fp8Block128Linear)
        lora = layer.attach_lora(LoRAConfig(targets=("attention",), rank=16, alpha=32.0))
    layer.weight, layer.weight_scale_inv = quantized_weight(sum(slots), cols, device=DEVICE)
    generator = torch.Generator(DEVICE).manual_seed(rows)
    x = torch.randn(rows, cols, device=DEVICE, generator=generator).bfloat16()
    lora.materialize(device=DEVICE, dtype=torch.bfloat16, generator=torch.Generator())
    with torch.no_grad():
        for factor in lora.factors[1::2]:
            factor.copy_(torch.randn(factor.shape, device=DEVICE, generator=generator) * 0.1)
    stream = torch.cuda.Stream(DEVICE)
    _prepared([layer], stream, max_rows=rows)
    with torch.cuda.stream(stream), torch.inference_mode():
        eager = layer.forward(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = layer.forward(x)
        graph.replay()
    stream.synchronize()

    assert torch.equal(captured, eager)
    a, b = lora.factors[0::2], lora.factors[1::2]
    low = torch.cat([lora.scaling * x.double() @ ai.double().t() for ai in a], -1)
    blocks = torch.block_diag(*(bi.double() for bi in b)).t()
    product, bound = contract_product(x, layer.weight, layer.weight_scale_inv)
    expected = product + low @ blocks
    # The block-FP8 contract's bound; the base output rounds to BF16, then the delta (its
    # scaled low rank rounded to BF16 once) adds onto it with one more rounding.
    bound = bound + 2**-8 * (expected.abs() + low.abs() @ blocks.abs())
    assert ((eager.double() - expected).abs() <= bound).all()


@pytest.mark.parametrize("rows", [1, 16, 2048])
@pytest.mark.parametrize(
    ("quantized", "bf16", "cols"),
    [((10240, 6144), (48, 48), 5120), ((8192, 4096), (32, 32), 2048)],
    ids=["27b-fp8", "35b-a3b-fp8"],
)
def test_a_split_gdn_in_proj_writes_its_parts_columns_as_their_own_products(
    quantized: tuple[int, ...], bf16: tuple[int, ...], cols: int, rows: int
) -> None:
    # Block-FP8 qkv | z beside BF16 b | a, as the FP8 checkpoints store a GDN in_proj.
    sizes = (*quantized, *bf16)
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = LinearColParallelMerged(cols, list(sizes), has_bias=False, site=site(device=DEVICE))
        layer.split([Fp8Block128Linear] * len(quantized) + [Unquantized] * len(bf16))
    fp8, plain = layer.parts
    fp8.weight, fp8.weight_scale_inv = quantized_weight(sum(quantized), cols, device=DEVICE)
    plain.weight = (torch.randn(sum(bf16), cols, device=DEVICE) * cols**-0.5).bfloat16()
    x = torch.randn(rows, cols, device=DEVICE).bfloat16()
    stream = torch.cuda.Stream(DEVICE)
    _prepared([layer], stream, max_rows=rows)
    with torch.cuda.stream(stream), torch.inference_mode():
        parts = torch.cat([fp8.method.apply(x), plain.method.apply(x)], -1)
        eager = layer.forward(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = layer.forward(x)
        graph.replay()
    stream.synchronize()

    assert torch.equal(eager, parts)
    assert torch.equal(captured, eager)
