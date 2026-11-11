"""ModelOpt's static FP8 and NVFP4 methods on CUDA.

cti-kernels' NVFP4 quantizer is within one step of the numerics contract (its reciprocals are
approximate); ``NVFP4Linear`` and the static-FP8 cuBLASLt provider stay within the contract's
BF16 bound, each merge slot by its own scale; and a captured CUDA graph replays the eager
products bit for bit.
"""

from __future__ import annotations

import gc

import pytest
import torch
import torch.nn.functional as F

from cti.config import LoRAConfig
from cti.layers import (
    LinearColParallelMerged,
    LinearRowParallel,
    prepare_inference,
)
from cti.layers.linear_method import Unquantized
from cti.layers.quant import numerics
from cti.layers.quant.fp8_static import Fp8StaticLinear
from cti.layers.quant.nvfp4 import Nvfp4Linear
from ctikernels.linear.nvfp4 import quantize_nvfp4, unblocked_scale
from tests.support.modelopt import (
    input_scale,
    nvfp4_product,
    nvfp4_weight,
    static_fp8_product,
    static_fp8_weight,
)
from tests.support.tp_shards import default_dtype, site

# Synthetic weights: no checkpoint to cache.
pytestmark = pytest.mark.checkpoints()

DEVICE = torch.device("cuda")
_HIDDEN, _INTERMEDIATE = 512, 256
_MIXER = (256, 128, 128)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("rows", [1, 7, 64, 300])
def test_the_kernels_quantizer_is_within_one_step_of_the_contract(rows: int, seed: int) -> None:
    # The kernel's reciprocals are rcp.approx, the contract's IEEE (numerics docstring).
    generator = torch.Generator(DEVICE).manual_seed(seed)
    x = (torch.randn(rows, _HIDDEN, device=DEVICE, generator=generator) * 4).bfloat16()
    x[0, :32] = 0
    scale = input_scale(x, 448 * 6).reshape(1)

    activation = quantize_nvfp4(x, scale)
    codes, block = numerics.quantize_nvfp4_activation(x, scale)

    kernel_scales = unblocked_scale(activation.scale, rows, _HIDDEN)
    scale_steps = kernel_scales.view(torch.uint8).int() - block.view(torch.uint8).int()
    assert scale_steps.abs().max() <= 1
    kernel_codes = numerics.unpack_e2m1(activation.data).int()
    magnitude_steps = (kernel_codes & 7) - (codes.int() & 7)
    nonzero = (kernel_codes & 7 != 0) & (codes.int() & 7 != 0)
    signs_differ = (kernel_codes >> 3 != codes.int() >> 3) & nonzero
    assert magnitude_steps.abs().max() <= 1 and not signs_differ.any()
    assert torch.equal(kernel_codes[0, :32], codes.int()[0, :32])


class _Layers:
    """An NVFP4 gated MLP (merged gate|up, row-parallel down) and a merged static-FP8
    projection beside it, built in BF16 and loaded on CUDA with their FP64 oracles."""

    def __init__(self) -> None:
        layer_site = site(device=DEVICE)
        with torch.device("meta"), default_dtype(torch.bfloat16):
            self.gate_up = LinearColParallelMerged(
                _HIDDEN, [_INTERMEDIATE] * 2, has_bias=False, site=layer_site
            )
            self.down = LinearRowParallel(_INTERMEDIATE, _HIDDEN, has_bias=False, site=layer_site)
            self.mixer = LinearColParallelMerged(
                _HIDDEN, list(_MIXER), has_bias=False, site=layer_site
            )
            for layer in (self.gate_up, self.down):
                layer.use_method(Nvfp4Linear)
            self.mixer.use_method(Fp8StaticLinear)
        x = torch.randn(64, _HIDDEN, device=DEVICE) * 4
        act = F.silu(torch.randn(64, _INTERMEDIATE, device=DEVICE)) * 4
        self.gate_up_weights = self._nvfp4(self.gate_up, (1, 2), input_scale(x, 448 * 6))
        self.down_weights = self._nvfp4(self.down, (3,), input_scale(act, 448 * 6))
        static = [
            static_fp8_weight(n, _HIDDEN, seed=s, device=DEVICE) for s, n in enumerate(_MIXER)
        ]
        self.mixer.weight = torch.cat([w for w, _ in static])
        self.mixer.weight_scale = torch.stack([s for _, s in static])
        self.mixer.input_scale = input_scale(x, 448).expand(len(_MIXER)).clone()
        for layer in self.layers:
            layer.method.bind()

    @property
    def layers(self) -> tuple:
        return self.gate_up, self.down, self.mixer

    @staticmethod
    def _nvfp4(layer, seeds: tuple[int, ...], scale: torch.Tensor) -> list:
        rows = layer.local_output_size // len(seeds)
        weights = [nvfp4_weight(rows, layer.local_input_size, seed=s, device=DEVICE) for s in seeds]
        layer.weight = torch.cat([w for w, _, _ in weights])
        layer.weight_scale = torch.cat([s for _, s, _ in weights])
        layer.weight_scale_2 = torch.stack([g for _, _, g in weights])
        layer.input_scale = scale.expand(len(weights)).clone()
        return [tuple(t.cpu() for t in weight) for weight in weights]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        gate_up = self.gate_up.forward(x)
        gate, up = gate_up.chunk(2, dim=-1)
        activation = F.silu(gate) * up
        return x, gate_up, activation, self.down.forward(activation), self.mixer.forward(x)


def _prepared(layers: _Layers, stream: torch.cuda.Stream, max_rows: int) -> int:
    stream.wait_stream(torch.cuda.current_stream(DEVICE))
    with torch.cuda.stream(stream):
        return prepare_inference(
            layers.layers,
            max_rows=max_rows,
            max_sequences=1,
            dtype=torch.bfloat16,
            device=DEVICE,
        )


@pytest.mark.parametrize("rows", [1, 3, 16, 64])
def test_the_products_stay_within_the_contract_bound(rows: int) -> None:
    stream = torch.cuda.Stream(DEVICE)
    layers = _Layers()
    # The shared workspace is the static-FP8 product's scratch; the NVFP4 products need none.
    assert _prepared(layers, stream, max_rows=64) == layers.mixer.method.workspace_bytes(DEVICE)
    x = (torch.randn(rows, _HIDDEN, device=DEVICE) * 4).bfloat16()
    with torch.cuda.stream(stream), torch.inference_mode():
        x, gate_up, activation, down, mixed = (t.cpu() for t in layers.forward(x))
    stream.synchronize()

    # ``bind`` merged each layer's input scales into one; the static-FP8 weight scales stay
    # one per slot, which each slot's columns take.
    products = (
        (gate_up, nvfp4_product(x, layers.gate_up_weights, layers.gate_up.input_scale[0].cpu())),
        (down, nvfp4_product(activation, layers.down_weights, layers.down.input_scale[0].cpu())),
        (
            mixed,
            static_fp8_product(
                x,
                layers.mixer.weight.cpu(),
                layers.mixer.weight_scale.cpu().repeat_interleave(torch.tensor(_MIXER)),
                layers.mixer.input_scale[0].cpu(),
            ),
        ),
    )
    for output, (expected, bound) in products:
        assert output.dtype == torch.bfloat16
        assert ((output.double() - expected).abs() <= bound).all()


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 16])
def test_a_captured_graph_replays_the_eager_products(rows: int) -> None:
    stream = torch.cuda.Stream(DEVICE)
    layers = _Layers()
    _prepared(layers, stream, max_rows=16)
    x = torch.randn(rows, _HIDDEN, device=DEVICE).bfloat16()
    with torch.cuda.stream(stream), torch.inference_mode():
        eager = layers.forward(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = layers.forward(x)
        # Earlier tests' layers hold reference cycles: free them before measuring.
        gc.collect()
        allocated = torch.cuda.memory_allocated(DEVICE)
        graph.replay()
    stream.synchronize()

    assert all(torch.equal(a, b) for a, b in zip(captured, eager, strict=True))
    assert torch.cuda.memory_allocated(DEVICE) == allocated


# Qwen3.8-27B-NVFP4's static-FP8 projections: (slot rows, input width).
_STATIC_27B = {
    "gdn-in-proj": ((2048, 2048, 6144, 6144), 5120),
    "gdn-out-proj": ((5120,), 6144),
    "attention-qkv-gate": ((12288, 1024, 1024), 5120),
    "attention-o-proj": ((5120,), 6144),
}


@pytest.mark.parametrize("rows", [1, 2048])
@pytest.mark.parametrize("name", list(_STATIC_27B))
def test_static_fp8_at_the_27b_shapes_stays_within_the_contract(name: str, rows: int) -> None:
    slots, cols = _STATIC_27B[name]
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = LinearColParallelMerged(cols, list(slots), has_bias=False, site=site(device=DEVICE))
        layer.use_method(Fp8StaticLinear)
    x = (torch.randn(rows, cols, device=DEVICE) * 4).bfloat16()
    weights = [static_fp8_weight(n, cols, seed=s, device=DEVICE) for s, n in enumerate(slots)]
    layer.weight = torch.cat([w for w, _ in weights])
    layer.weight_scale = torch.stack([scale for _, scale in weights])
    layer.input_scale = input_scale(x, 448).expand(len(slots)).clone()
    stream = torch.cuda.Stream(DEVICE)
    stream.wait_stream(torch.cuda.current_stream(DEVICE))
    with torch.cuda.stream(stream):
        prepare_inference(
            [layer],
            max_rows=rows,
            max_sequences=1,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        with torch.inference_mode():
            y = layer.forward(x)
    stream.synchronize()

    alpha = torch.cat([scale.expand(n) for (_, scale), n in zip(weights, slots, strict=True)])
    expected, bound = static_fp8_product(x, layer.weight, alpha, layer.input_scale[0])
    assert y.dtype == torch.bfloat16 and ((y.double() - expected).abs() <= bound).all()


@pytest.mark.parametrize("rows", [1, 2048])
def test_an_adapted_static_fp8_projection_adds_the_kernels_delta_within_its_contract(
    rows: int,
) -> None:
    # The 27B-NVFP4 GDN in_proj's q|k|v|z on static FP8, adapted at rank 16; its warmed delta
    # replays in a captured graph as eagerly.
    slots, cols = _STATIC_27B["gdn-in-proj"]
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = LinearColParallelMerged(cols, list(slots), has_bias=False, site=site(device=DEVICE))
        layer.use_method(Fp8StaticLinear)
        lora = layer.attach_lora(LoRAConfig(targets=("linear_attention",), rank=16, alpha=32.0))
    generator = torch.Generator(DEVICE).manual_seed(rows)
    x = (torch.randn(rows, cols, device=DEVICE, generator=generator) * 4).bfloat16()
    weights = [static_fp8_weight(n, cols, seed=s, device=DEVICE) for s, n in enumerate(slots)]
    layer.weight = torch.cat([w for w, _ in weights])
    layer.weight_scale = torch.stack([scale for _, scale in weights])
    layer.input_scale = input_scale(x, 448).expand(len(slots)).clone()
    lora.materialize(device=DEVICE, dtype=torch.bfloat16, generator=torch.Generator())
    with torch.no_grad():
        for factor in lora.factors[1::2]:
            factor.copy_(torch.randn(factor.shape, device=DEVICE, generator=generator) * 0.1)
    stream = torch.cuda.Stream(DEVICE)
    stream.wait_stream(torch.cuda.current_stream(DEVICE))
    with torch.cuda.stream(stream):
        prepare_inference(
            [layer],
            max_rows=rows,
            max_sequences=1,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        with torch.inference_mode():
            base = layer.method.apply(x)
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
    expected = base.double() + low @ blocks
    # The kernel rounds the scaled low rank to BF16 once and the sum to BF16 once.
    bound = 2**-8 * expected.abs() + 2**-8 * (low.abs() @ blocks.abs()) + 1e-6
    assert ((eager.double() - expected).abs() <= bound).all()


@pytest.mark.parametrize("rows", [1, 16, 2048])
def test_a_split_static_fp8_in_proj_writes_its_parts_columns_as_their_own_products(
    rows: int,
) -> None:
    # 27B-NVFP4's GDN in_proj: static-FP8 q | k | v | z beside BF16 b | a.
    slots, cols = _STATIC_27B["gdn-in-proj"]
    sizes = (*slots, 48, 48)
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = LinearColParallelMerged(cols, list(sizes), has_bias=False, site=site(device=DEVICE))
        layer.split([Fp8StaticLinear] * len(slots) + [Unquantized] * 2)
    fp8, plain = layer.parts
    x = (torch.randn(rows, cols, device=DEVICE) * 4).bfloat16()
    weights = [static_fp8_weight(n, cols, seed=s, device=DEVICE) for s, n in enumerate(slots)]
    fp8.weight = torch.cat([w for w, _ in weights])
    fp8.weight_scale = torch.stack([scale for _, scale in weights])
    fp8.input_scale = input_scale(x, 448).expand(len(slots)).clone()
    plain.weight = (torch.randn(96, cols, device=DEVICE) * cols**-0.5).bfloat16()
    stream = torch.cuda.Stream(DEVICE)
    stream.wait_stream(torch.cuda.current_stream(DEVICE))
    with torch.cuda.stream(stream):
        prepare_inference(
            [layer],
            max_rows=rows,
            max_sequences=1,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        with torch.inference_mode():
            parts = torch.cat([fp8.method.apply(x), plain.method.apply(x)], -1)
            eager = layer.forward(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = layer.forward(x)
            graph.replay()
    stream.synchronize()

    assert torch.equal(eager, parts)
    assert torch.equal(captured, eager)
