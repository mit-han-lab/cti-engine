"""ModelOpt's formats: the static FP8 and NVFP4 numerics contract, their methods' storage and
windows, the reference providers that serve the CPU against the contract, and the recipe's
claims."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from cti.distributed import Topology
from cti.layers import (
    LinearColParallelMerged,
    LinearRowParallel,
    ParallelLMHead,
)
from cti.layers.linear import LinearBase
from cti.layers.lora import LoRAProjection
from cti.layers.placement import WeightShard
from cti.layers.quant import CheckpointPart, TensorHeader, fp8_static, numerics, nvfp4
from cti.layers.quant.fp8_block import Fp8Block128Linear
from cti.layers.quant.fp8_static import Fp8StaticKernel, Fp8StaticLinear, Fp8StaticReference
from cti.layers.quant.modelopt_mixed import ModelOptMixedRecipe
from cti.layers.quant.nvfp4 import Nvfp4Kernel, Nvfp4Linear, Nvfp4Reference
from tests.support.lora_serving import lora_delta
from tests.support.modelopt import (
    input_scale,
    nvfp4_product,
    nvfp4_weight,
    static_fp8_product,
    static_fp8_weight,
)
from tests.support.tp_shards import default_dtype, site

# The contract's FP64 product of a loaded layer's input and its bound for a BF16 computation.
Oracle = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def test_e2m1_rounds_to_nearest_even_saturates_and_packs_the_even_column_low() -> None:
    v = torch.tensor([0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 100.0, -0.75, -0.0])
    codes = torch.tensor([[1, 9, 7, 15]], dtype=torch.uint8)

    values = numerics.e2m1_values(numerics.e2m1_codes(v))
    packed = numerics.pack_e2m1(codes)

    # Ties go to the even code: 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4.
    assert values.tolist() == [0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -1.0, 0.0]
    assert packed.tolist() == [[1 | 9 << 4, 7 | 15 << 4]]
    assert torch.equal(numerics.unpack_e2m1(packed), codes)


def test_the_nvfp4_weight_scales_multiply_their_blocks() -> None:
    weight, scale, global_scale = nvfp4_weight(4, 64)

    w_hat = numerics.dequantize_nvfp4_weight(weight, scale, global_scale)

    codes = numerics.unpack_e2m1(weight)
    for row, col in ((0, 0), (3, 17), (2, 63)):
        value = numerics.e2m1_values(codes[row, col]) * scale[row, col // 16].float()
        assert w_hat[row, col] == value * global_scale


def test_the_nvfp4_activation_quantizes_each_block_by_its_amax() -> None:
    x = (torch.randn(3, 64) * torch.tensor([1.0, 0.01, 30.0]).unsqueeze(-1)).bfloat16()
    x[1, 16:32] = 0
    scale = input_scale(x, 448 * 6)

    codes, block = numerics.quantize_nvfp4_activation(x, scale)
    x_hat = numerics.dequantize_nvfp4_activation(codes, block, scale)

    assert codes.shape == x.shape and block.shape == (3, 4) and block.dtype == numerics.E4M3
    # A block's largest magnitude maps to about 6 (its scale rounds to E4M3).
    amax = x.float().unflatten(-1, (4, 16)).abs().amax(-1)
    expected = (amax * (1 / scale * (1 / 6))).clamp_max(448).to(numerics.E4M3)
    assert torch.equal(block.float(), expected.float())
    # An all-zero block stays zero, with a zero scale.
    assert block[1, 1] == 0 and not x_hat[1, 16:32].any()
    # E2M1 steps at most a third of the block's amax apart, which maps to 6.
    assert ((x_hat - x.float()).abs() <= amax.repeat_interleave(16, -1) / 5).all()


def test_static_fp8_activations_saturate_instead_of_turning_nan() -> None:
    x = torch.tensor([[1.0, -1e6, 3e5, 0.0]])

    q = numerics.quantize_static_activation(x, torch.tensor(2.0))

    assert q.float().tolist() == [[0.5, -448.0, 448.0, 0.0]]


def _merged(
    sizes: tuple[int, ...], cols: int, method: type, topology: Topology | None = None
) -> LinearColParallelMerged:
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = LinearColParallelMerged(cols, list(sizes), has_bias=False, site=site(topology))
        layer.use_method(method)
    return layer


def _static_fp8(layer: LinearBase, x: torch.Tensor, factors: tuple[float, ...]) -> Oracle:
    """Load ``layer`` with one static-FP8 weight per slot, the slots' input scales
    calibrated on ``x`` times ``factors``; bind it; return its oracle."""

    sizes = [piece.size for piece in layer.output_slices]
    weights = [
        static_fp8_weight(rows, layer.local_input_size, seed=s) for s, rows in enumerate(sizes)
    ]
    scales = torch.stack([input_scale(x, 448) * factor for factor in factors])
    layer.weight = torch.cat([w for w, _ in weights])
    layer.weight_scale = torch.stack([s for _, s in weights])
    layer.input_scale = scales
    weight = layer.weight
    layer.method.bind()
    alpha = torch.cat([s.expand(rows) for (_, s), rows in zip(weights, sizes, strict=True)])
    return lambda x: static_fp8_product(x, weight, alpha, scales.max())


def _nvfp4(layer: LinearBase, x: torch.Tensor, factors: tuple[float, ...]) -> Oracle:
    """Load ``layer`` with one NVFP4 weight per slot, the slots' input scales calibrated on
    ``x`` times ``factors``; bind it; return its oracle."""

    sizes = [piece.size for piece in layer.output_slices]
    weights = [nvfp4_weight(rows, layer.local_input_size, seed=s) for s, rows in enumerate(sizes)]
    scales = torch.stack([input_scale(x, 448 * 6) * factor for factor in factors])
    layer.weight = torch.cat([w for w, _, _ in weights])
    layer.weight_scale = torch.cat([s for _, s, _ in weights])
    layer.weight_scale_2 = torch.stack([g for _, _, g in weights])
    layer.input_scale = scales
    layer.method.bind()
    return lambda x: nvfp4_product(x, weights, scales.max())


FORMATS = pytest.mark.parametrize(
    ("method", "load"),
    [(Fp8StaticLinear, _static_fp8), (Nvfp4Linear, _nvfp4)],
    ids=["fp8-static", "nvfp4"],
)


def test_the_methods_declare_their_storage_by_merge_slot() -> None:
    static = _merged((128, 64, 64, 128), 256, Fp8StaticLinear)
    packed = _merged((256, 256), 512, Nvfp4Linear)

    assert {name: (t.dtype, tuple(t.shape)) for name, t in static.state_dict().items()} == {
        "weight": (numerics.E4M3, (384, 256)),
        "weight_scale": (torch.float32, (4,)),
        "input_scale": (torch.float32, (4,)),
    }
    assert {name: (t.dtype, tuple(t.shape)) for name, t in packed.state_dict().items()} == {
        "weight": (torch.uint8, (512, 256)),
        "weight_scale": (numerics.E4M3, (512, 32)),
        "weight_scale_2": (torch.float32, (2,)),
        "input_scale": (torch.float32, (2,)),
    }
    assert fp8_static.PROVIDERS == {"cpu": Fp8StaticReference, "cuda": Fp8StaticKernel}
    assert nvfp4.PROVIDERS == {"cpu": Nvfp4Reference, "cuda": Nvfp4Kernel}


@FORMATS
def test_each_slot_takes_its_own_weight_scales_and_all_the_largest_input_scale(
    method: type, load: Callable[..., Oracle]
) -> None:
    layer = _merged((128, 64, 64, 128), 256, method)
    x = (torch.randn(5, 256) * 3).bfloat16()
    oracle = load(layer, x, (1.0, 0.5, 1.0, 0.25))

    y = layer.forward(x)

    expected, bound = oracle(x)
    assert y.dtype == torch.bfloat16 and ((y.double() - expected).abs() <= bound).all()
    # ``bind`` merged the input scales in place.
    assert layer.input_scale.shape == (1,)


def test_swapping_two_nvfp4_slots_global_scales_moves_their_outputs() -> None:
    layer = _merged((128, 128), 256, Nvfp4Linear)
    x = (torch.randn(5, 256) * 3).bfloat16()
    oracle = _nvfp4(layer, x, (1.0, 1.0))
    layer.weight_scale_2 = layer.weight_scale_2.flip(0)
    layer.method.bind()

    y = layer.forward(x)

    expected, bound = oracle(x)
    assert not ((y.double() - expected).abs() <= bound).all()


@FORMATS
def test_a_gated_product_quantizes_the_gated_input_and_adds_the_bias(
    method: type, load: Callable[..., Oracle]
) -> None:
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = LinearRowParallel(256, 128, has_bias=True, site=site())
        layer.use_method(method)
    layer.bias = torch.randn(128).bfloat16()
    x, gate = torch.randn(2, 5, 256).bfloat16()
    # The gate applies in FP32, rounded once to BF16.
    gated = (x.float() * torch.sigmoid(gate.float())).bfloat16()
    oracle = load(layer, gated, (1.0,))

    y = layer.forward_gated(x, gate)

    expected, bound = oracle(gated)
    expected += layer.bias.double()
    bound += 2**-8 * expected.abs()
    assert ((y.double() - expected).abs() <= bound).all()


@pytest.mark.parametrize("rank", [0, 1])
def test_tensor_parallel_windows_halve_packed_columns_and_divide_block_scales(rank: int) -> None:
    column = _merged((256, 512), 512, Nvfp4Linear, Topology(rank, 2))
    with torch.device("meta"), default_dtype(torch.bfloat16):
        row = LinearRowParallel(512, 256, has_bias=False, site=site(Topology(rank, 2)))
        row.use_method(Nvfp4Linear)

    def windows(layer: LinearBase) -> dict[str, list[tuple[int, int, int]]]:
        return {
            param.path: [(w.axis, w.start, w.stop) for w in param.windows]
            for param in layer.params()
        }

    # Each slot's rows (128 of 256, 256 of 512) for every leaf but the per-slot scalars.
    rows = [(0, 128 * rank, 128 * (rank + 1)), (0, 256 * rank, 256 * (rank + 1))]
    assert windows(column) == {
        "weight": rows,
        "weight_scale": rows,
        "weight_scale_2": [],
        "input_scale": [],
    }
    # The row-sharded input columns: 256 of 512 elements, 128 packed bytes, 16 block scales.
    assert windows(row) == {
        "weight": [(1, 128 * rank, 128 * (rank + 1))],
        "weight_scale": [(1, 16 * rank, 16 * (rank + 1))],
        "weight_scale_2": [],
        "input_scale": [],
    }


def test_nvfp4_windows_or_local_columns_that_cut_a_block_are_refused() -> None:
    layer = _merged((128,), 64, Nvfp4Linear)

    with pytest.raises(ValueError, match=r"the window \[8, 40\) on axis 1 cuts its 16-element"):
        layer.method.windows("weight_scale", (WeightShard(1, 8, 40),))
    with pytest.raises(ValueError, match=r"the window \[1, 5\) on axis 1 cuts its 2-element"):
        layer.method.windows("weight", (WeightShard(1, 1, 5),))
    with pytest.raises(ValueError, match="24 local columns cut its 16-column blocks"):
        with torch.device("meta"):
            LinearRowParallel(48, 64, has_bias=False, site=site(Topology(0, 2))).use_method(
                Nvfp4Linear
            )


def _head(method: type, *, training: bool = True) -> ParallelLMHead:
    with torch.device("meta"), default_dtype(torch.bfloat16):
        head = ParallelLMHead(48, 64, site=site(training=training))
        head.use_method(method)
    return head


def test_a_training_head_refuses_a_quantized_method() -> None:
    # A training head multiplies the dense matrix; quantized heads serve.
    for method in (Fp8Block128Linear, Fp8StaticLinear, Nvfp4Linear):
        _head(method, training=False)
        with pytest.raises(ValueError, match=f"{method.__name__} has no training path"):
            _head(method)


def _part(module: str, **tensors: tuple[str, tuple[int, ...]]) -> CheckpointPart:
    return CheckpointPart(
        module, {leaf: TensorHeader(dtype, shape) for leaf, (dtype, shape) in tensors.items()}
    )


def test_the_recipe_refuses_routed_experts_by_name() -> None:
    recipe = ModelOptMixedRecipe({"layers.0.mlp.experts.0.up_proj": "NVFP4"}, excluded=())
    part = _part(
        "layers.0.mlp.experts.0.up_proj",
        weight=("U8", (64, 16)),
        weight_scale=("F8_E4M3", (64, 2)),
        weight_scale_2=("F32", ()),
        input_scale=("F32", ()),
    )

    with pytest.raises(ValueError, match=r"NVFP4 routed experts are not supported yet"):
        recipe.experts_method_for(part)


def test_the_recipe_claims_each_listed_module_by_its_format() -> None:
    recipe = ModelOptMixedRecipe({"a.q_proj": "FP8", "a.up_proj": "NVFP4"}, excluded=())
    static = _part(
        "a.q_proj", weight=("F8_E4M3", (64, 32)), weight_scale=("F32", ()), input_scale=("F32", ())
    )
    packed = _part(
        "a.up_proj",
        weight=("U8", (64, 16)),
        weight_scale=("F8_E4M3", (64, 2)),
        weight_scale_2=("F32", ()),
        input_scale=("F32", ()),
    )

    assert recipe.method_for(static) is Fp8StaticLinear
    assert recipe.method_for(packed) is Nvfp4Linear
    with pytest.raises(ValueError, match=r"^a\.up_proj: NVFP4 stores"):
        recipe.method_for(_part("a.up_proj", weight=("U8", (64, 16))))
    with pytest.raises(ValueError, match=r"^a\.k_proj holds quantized tensors but no declaration"):
        recipe.method_for(_part("a.k_proj", weight=("F8_E4M3", (64, 32))))


@FORMATS
@pytest.mark.parametrize("gated", [False, True], ids=["ungated", "gated"])
def test_an_adapted_quantized_product_adds_the_delta_in_place_after_its_own(
    method: type, load: Callable[..., Oracle], gated: bool
) -> None:
    layer = _merged((64, 64), 256, method)
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(2, 5, 256, generator=generator).bfloat16()
    gate = torch.randn(2, 5, 256, generator=generator).bfloat16() if gated else None
    gated_x = x if gate is None else layer.services.gated_input()(x, gate)
    load(layer, gated_x, (1.0, 1.0))
    lora = LoRAProjection(
        input_size=256, output_slices=layer.output_slices, rank=4, scaling=0.5, owns_whole=True
    )
    lora.materialize(device=torch.device("cpu"), dtype=torch.bfloat16, generator=generator)
    with torch.no_grad():
        for factor in lora.factors[1::2]:
            factor.copy_(torch.randn(factor.shape, generator=generator))
        expected = lora_delta(lora, layer.method.apply(gated_x), gated_x)

        actual = layer.method.lora_product(lora)(x, None, gate)

    # The base product gains each slice's delta in its own storage, one addmm per slice.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
