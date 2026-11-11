"""The Fp8Block128 numerics contract, its method and the reference provider that serves the
CPU."""

from __future__ import annotations

import pytest
import torch

from cti.config import LoRAConfig
from cti.distributed import Topology
from cti.layers import (
    Adapter,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    prepare_inference,
)
from cti.layers.quant import fp8_block, numerics
from cti.layers.quant.fp8_block import Fp8Block128Linear, warm_rows
from tests.support.fp8_block import contract_product, quantized_weight
from tests.support.tp_shards import site

CPU = torch.device("cpu")


def test_activation_scales_are_the_group_amax_times_one_over_448() -> None:
    x = torch.randn(3, 300) * torch.tensor([1.0, 0.01, 100.0]).unsqueeze(-1)
    x[1, 128:256] = 0

    q, s = numerics.quantize_activation(x)

    # Multiplications in FP32, as the kernels compute them: s = amax * (1/448), q = x * (448/amax).
    groups = torch.nn.functional.pad(x, (0, 84)).unflatten(-1, (3, 128))
    amax = groups.abs().amax(-1).clamp_min(numerics.EPS)
    assert q.dtype == numerics.E4M3 and q.shape == x.shape
    assert torch.equal(s, amax * (1.0 / 448))
    expected = (x * (448 / amax).repeat_interleave(128, -1)[:, :300]).to(q.dtype)
    assert torch.equal(q.float(), expected.float())
    # An all-zero group stays zero rather than dividing zero by zero.
    assert torch.equal(numerics.dequantize_activation(q, s)[1, 128:256], torch.zeros(128))
    assert q.float().abs().amax() <= 448


def test_the_weight_scale_multiplies_its_block() -> None:
    weight, scale = quantized_weight(200, 260)

    w_hat = numerics.dequantize_block_weight(weight, scale)

    for row, col in ((0, 0), (127, 129), (199, 259), (130, 5)):
        assert w_hat[row, col] == weight[row, col].float() * scale[row // 128, col // 128]


def _unprepared(rows: int, cols: int, *, bias: bool, lora: str | None = None) -> LinearReplicated:
    with torch.device("meta"):
        layer = LinearReplicated(cols, rows, has_bias=bias, site=site(), lora=lora)
        layer.use_method(Fp8Block128Linear)
    layer.weight, layer.weight_scale_inv = quantized_weight(rows, cols)
    if bias:
        layer.bias = torch.randn(rows).bfloat16()
    return layer


def _layer(rows: int, cols: int, *, bias: bool) -> LinearReplicated:
    layer = _unprepared(rows, cols, bias=bias)
    prepare_inference([layer], max_rows=8, max_sequences=1, dtype=torch.float32, device=CPU)
    return layer


@pytest.mark.parametrize("chunked", [False, True], ids=["whole", "chunked"])
@pytest.mark.parametrize("bias", [False, True])
def test_the_reference_provider_follows_the_contract(
    monkeypatch: pytest.MonkeyPatch, chunked: bool, bias: bool
) -> None:
    if chunked:
        # 256-row scratch chunks, each dequantized 128 rows at a time; 520 rows end ragged.
        monkeypatch.setattr(fp8_block, "_SCRATCH_BYTES", 256 * 300 * 2)
        monkeypatch.setattr(fp8_block, "_DEQUANTIZE_BYTES", 128 * 300 * 4)
    layer = _layer(520, 300, bias=bias)
    x = (torch.randn(2, 3, 300) * 4).bfloat16()

    y = layer.forward(x)

    assert y.dtype == torch.bfloat16 and y.shape == (2, 3, 520)
    expected, bound = contract_product(x, layer.weight, layer.weight_scale_inv, layer.bias)
    assert ((y.double() - expected).abs() <= bound).all()


def test_padding_rows_give_zero_and_never_nan() -> None:
    layer = _layer(256, 128, bias=False)
    x = torch.zeros(4, 128, dtype=torch.bfloat16)
    x[0] = torch.randn(128).bfloat16()

    y = layer.forward(x)

    assert torch.isfinite(y).all()
    assert torch.equal(y[1:], torch.zeros(3, 256, dtype=torch.bfloat16))


def test_the_method_declares_its_scale_grid_beside_the_weight() -> None:
    with torch.device("meta"):
        layer = LinearReplicated(300, 520, has_bias=False, site=site())
        layer.use_method(Fp8Block128Linear)

    shapes = {name: (t.dtype, tuple(t.shape)) for name, t in layer.state_dict().items()}

    assert shapes == {
        "weight": (numerics.E4M3, (520, 300)),
        "weight_scale_inv": (torch.float32, (5, 3)),
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_tensor_parallel_scale_windows_follow_the_weight_in_whole_blocks(rank: int) -> None:
    with torch.device("meta"):
        column = LinearColParallelMerged(
            512, [256, 512], has_bias=False, site=site(Topology(rank, 2))
        )
        row = LinearRowParallel(512, 256, has_bias=False, site=site(Topology(rank, 2)))
        for layer in (column, row):
            layer.use_method(Fp8Block128Linear)

    # Each slot's rows (128 of 256, 256 of 512) and the row-sharded input columns.
    assert [(w.axis, w.start, w.stop) for w in column.params()[1].windows] == [
        (0, rank, rank + 1),
        (0, 2 * rank, 2 * rank + 2),
    ]
    assert [(w.axis, w.start, w.stop) for w in row.params()[1].windows] == [
        (1, 2 * rank, 2 * rank + 2)
    ]


def test_a_tensor_parallel_window_that_cuts_a_scale_block_is_refused_by_name() -> None:
    with torch.device("meta"):
        layer = LinearRowParallel(384, 128, has_bias=False, site=site(Topology(1, 2)))
        layer.use_method(Fp8Block128Linear)

    with pytest.raises(ValueError, match=r"o\.weight_scale_inv: the window \[192, 384\) on axis 1"):
        layer.params(prefix="o")


def test_training_runs_the_serving_numerics_and_passes_the_quantizer_straight_through() -> None:
    layer = _unprepared(256, 300, bias=False)
    x = (torch.randn(3, 300) * 4).bfloat16().requires_grad_(True)
    y = layer.forward_training(x)
    gradient = torch.randn_like(y)

    (x_gradient,) = torch.autograd.grad(y, x, gradient)

    expected, bound = contract_product(x.detach(), layer.weight, layer.weight_scale_inv)
    assert ((y.detach().double() - expected).abs() <= bound).all()
    w_hat = numerics.dequantize_block_weight(layer.weight, layer.weight_scale_inv).double()
    reference = gradient.double() @ w_hat
    bound = 2**-7 * (gradient.double().abs() @ w_hat.abs()) + 1e-6
    assert ((x_gradient.double() - reference).abs() <= bound).all()


def test_the_frozen_product_writes_out_accumulates_and_reduces_over_no_group() -> None:
    torch.manual_seed(0)
    product = _unprepared(256, 300, bias=False).method.frozen_product()
    x = (torch.randn(3, 300) * 4).bfloat16()
    dy = torch.randn(3, 256).bfloat16()
    y, dx = product.forward(x), product.grad_input(dy)

    def halve(rows: slice, y: torch.Tensor) -> None:
        y[rows] *= 0.5

    out = torch.empty(3, 256, dtype=torch.bfloat16)
    assert product.forward(x, out=out, epilogue=halve) is out
    assert torch.equal(out, y * 0.5)
    previous = torch.randn(3, 300).bfloat16()
    accumulated = product.grad_input(dy, out=previous.clone(), accumulate=True).float()
    expected = dx.float() + previous.float()
    assert ((accumulated - expected).abs() <= 2**-7 * expected.abs() + 2**-7).all()
    assert (product.out_features, product.in_features) == (256, 300)
    with pytest.raises(ValueError, match="reduces over no group"):
        product.grad_input(dy, exchange=object())


def test_a_power_of_two_scale_grid_stored_as_ue8m0_serves_as_its_fp32_twin() -> None:
    fp32 = _unprepared(256, 300, bias=False)
    ue8m0 = _unprepared(256, 300, bias=False)
    # Powers of two, which UE8M0 holds exactly.
    grid = torch.exp2(torch.randint(-12, -6, fp32.weight_scale_inv.shape).float())
    fp32.weight_scale_inv = grid
    ue8m0.weight_scale_inv = grid.to(torch.float8_e8m0fnu)
    x = (torch.randn(3, 300) * 4).bfloat16()

    assert torch.equal(
        fp32.method.frozen_product().forward(x), ue8m0.method.frozen_product().forward(x)
    )


def test_the_contract_bound_catches_a_scale_used_as_a_divisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer(256, 256, bias=False)
    x = torch.randn(4, 256).bfloat16()
    expected, bound = contract_product(x, layer.weight, layer.weight_scale_inv)
    dequantize = numerics.dequantize_block_weight
    monkeypatch.setattr(
        numerics, "dequantize_block_weight", lambda w, s: dequantize(w, s.reciprocal())
    )

    assert not ((layer.forward(x).double() - expected).abs() <= bound).all()


def test_a_product_before_its_provider_is_bound_is_refused() -> None:
    layer = _unprepared(256, 128, bias=False)

    with pytest.raises(RuntimeError, match="prepare_inference binds a block-FP8 weight's provider"):
        layer.forward(torch.randn(2, 128).bfloat16())


def test_the_method_adds_the_bias_to_its_product() -> None:
    plain, biased = _layer(256, 384, bias=False), _layer(256, 384, bias=True)
    x = torch.randn(3, 384).bfloat16()

    # Both layers hold the same weight; the bias is added once, after the product.
    assert torch.equal(biased.forward(x), plain.forward(x) + biased.bias)


def test_layers_share_the_largest_reference_scratch_and_an_adapted_one_is_prepared() -> None:
    small = _unprepared(256, 128, bias=False)
    large = _unprepared(384, 256, bias=False, lora="mlp")
    Adapter(LoRAConfig(targets=("mlp",), rank=4)).attach(large)

    size = prepare_inference(
        [small, large], max_rows=8, max_sequences=1, dtype=torch.float32, device=CPU
    )

    # Built in FP32 (the default dtype), the larger weight's rows fit one scratch chunk.
    assert size == 384 * 256 * 4
    scratch = {layer.services.gemm_workspace.data_ptr() for layer in (small, large)}
    assert len(scratch) == 1


def test_a_dequantization_step_keeps_its_padded_scale_grid_within_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two 128-row blocks of 300 FP32 columns fit the budget, but the scale grid spans 384
    # columns: only one block does.
    budget = 2 * 128 * 300 * 4
    monkeypatch.setattr(fp8_block, "_DEQUANTIZE_BYTES", budget)
    steps = []
    dequantize = numerics.dequantize_block_weight

    def record(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        steps.append((weight.shape[0], scale.shape[1] * numerics.BLOCK))
        return dequantize(weight, scale)

    monkeypatch.setattr(numerics, "dequantize_block_weight", record)
    layer = _layer(520, 300, bias=False)
    x = (torch.randn(3, 300) * 4).bfloat16()

    y = layer.forward(x)

    assert steps == [(128, 384)] * 4 + [(8, 384)]
    assert all(rows * columns * 4 <= budget for rows, columns in steps)
    expected, bound = contract_product(x, layer.weight, layer.weight_scale_inv)
    assert ((y.double() - expected).abs() <= bound).all()


def test_the_kernel_warms_every_split_k_row_and_each_other_runs_last() -> None:
    # An RTX 5090's runs for a (3072, 2048) weight on a cti-kernels prerelease: FlashInfer
    # between Triton runs.
    runs = ((512, "triton-fp8"), (896, "flashinfer-cutlass"), (1024, "triton-fp8"))
    runs += ((1536, "flashinfer-cutlass"), (2048, "triton-fp8"), (4096, "flashinfer-cutlass"))

    def provider_for(count: int) -> str:
        return next(provider for last, provider in runs if count <= last)

    rows, found = warm_rows(provider_for, 1200)

    assert found == [
        (1, 512, "triton-fp8"),
        (513, 896, "flashinfer-cutlass"),
        (897, 1024, "triton-fp8"),
        (1025, 1200, "flashinfer-cutlass"),
    ]
    # Every Triton row; each other run's last and, of another divisibility, the one before it.
    triton = [count for count in range(1, 1201) if provider_for(count) == "triton-fp8"]
    assert rows == sorted({*triton, 895, 896, 1199, 1200})
    assert warm_rows(provider_for, 0) == ([], [])


class _RecordedOperation:
    """``BlockFP8Linear``'s surface the provider uses, recording what it is given."""

    in_features = out_features = 256

    def __init__(self) -> None:
        self.warm_ups: list[tuple[int, dict]] = []
        self.inputs: list[torch.Tensor] = []

    def warm_up(self, max_m: int, **options) -> list[dict]:
        self.warm_ups.append((max_m, options))
        return []

    def provider_for(self, rows: int) -> str:
        return "triton-fp8"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x)
        return x


def test_the_kernel_is_warmed_for_the_only_layout_it_is_handed() -> None:
    # cti-kernels' graph capture accepts only the layouts warm_up registered.
    provider = object.__new__(fp8_block.Fp8BlockKernel)
    provider._operation, provider._dtype = _RecordedOperation(), torch.bfloat16

    provider.warm_up(64, out_row_stride=256)
    whole = torch.zeros(8, 512, dtype=torch.bfloat16)
    flat = torch.zeros(4 * 256 + 1, dtype=torch.bfloat16)
    for x in (whole[:, :256], flat[1:].view(4, 256), whole[:4, :256].contiguous()):
        provider(x, None)

    assert provider._operation.warm_ups == [
        (64, {"dtype": torch.bfloat16, "input_layouts": "packed", "out_row_stride": None})
    ]
    for x in provider._operation.inputs:
        assert x.is_contiguous() and x.data_ptr() % 16 == 0
