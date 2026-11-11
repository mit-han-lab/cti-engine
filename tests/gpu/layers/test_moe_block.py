"""A sparse block with a shared expert on CUDA at Qwen3.5-35B-A3B's widths, through the fused
backend's whole-block op (cti-kernels' ``MoEBlock``) and the layer's own composition: the op
routes as an FP64 router over its BF16 logits would, and as the composition does; the op's
fused kernel for few rows, its composition for more, and two expert-parallel ranks' partials
summed in BF16 each stay near an FP64 reference and no further from it than the composed
block; and a captured CUDA graph replays the eager output, also when captured after the layer's
warm-up and followed by eager batches up to the warmed rows; without that warm-up, a larger
eager batch frees what the composition's graph reads (a known cti-kernels issue, an expected
failure); a graph padded past its real rows keeps them apart from the padding."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.nn.functional import silu

from cti.layers import MoELayer
from cti.moe import BaseMoeBackend
from cti.moe.fused import FusedMoe
from ctikernels.moe import MoEBlock, route
from tests.support.tp_shards import default_dtype, site, windows

pytestmark = pytest.mark.checkpoints()

DEVICE = torch.device("cuda")
NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, SHARED = 256, 8, 2048, 512, 512
# Relative error from the FP64 reference, in multiples of rounding that reference to BF16
# once: measured at most 3.1 on an RTX 5090 for every path.
FLOOR_MULTIPLE = 4
# The op's error over the composed block's: measured at most 0.92.
COMPOSED_MULTIPLE = 1.5

# Every MoE forward runs eagerly in these tests: no graph breaks.


@dataclass(frozen=True)
class _Rank:
    """A rank whose exit all-reduce leaves the partial sum as it is."""

    rank: int
    size: int

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _Composed(FusedMoe):
    """The fused backend without its whole-block op: the layer composes the block around the
    routed experts."""

    def sparse_block(self, top_k: int, renormalize: bool, dtype: torch.dtype) -> None:
        return None


def _state() -> dict[str, torch.Tensor]:
    """A whole layer's BF16 weights."""

    generator = torch.Generator(device=DEVICE).manual_seed(0)

    def draw(*shape: int) -> torch.Tensor:
        scale = shape[-1] ** -0.5
        return (scale * torch.randn(*shape, generator=generator, device=DEVICE)).bfloat16()

    return {
        "experts.gate_up_proj": draw(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        "experts.down_proj": draw(NUM_EXPERTS, HIDDEN, INTERMEDIATE),
        "gate.weight": draw(NUM_EXPERTS, HIDDEN),
        "shared_expert.gate_up_proj.weight": draw(2 * SHARED, HIDDEN),
        "shared_expert.down_proj.weight": draw(HIDDEN, SHARED),
        "shared_expert_gate.weight": draw(1, HIDDEN),
    }


def _inputs(rows: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(rows, HIDDEN, generator=generator, device=DEVICE).bfloat16()


def _layer(backend: BaseMoeBackend, state: dict[str, torch.Tensor], rank: int | None = None):
    group = _Rank(0, 1) if rank is None else _Rank(rank, 2)
    layer_site = site(group=group, device=DEVICE, moe_backend=backend)
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = MoELayer(
            NUM_EXPERTS,
            TOP_K,
            HIDDEN,
            INTERMEDIATE,
            renormalize=True,
            site=layer_site,
            shared_expert_intermediate_size=SHARED,
        )
    local = {}
    shards = windows(layer)
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


def _forward(layer: MoELayer, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return layer.forward(x.clone())


def _rows_by_provider() -> dict[str, int]:
    """Row counts for which the op runs its fused kernel on this GPU, and its composition."""

    block = MoEBlock()
    fused = [rows for rows in range(1, 1025) if block.decision(rows, DEVICE) == "triton_block"]
    if not fused:
        pytest.skip("cti-kernels runs no fused MoE block on this GPU")
    return {"triton_block": fused[-1], "composed": fused[-1] + 1}


def _reference(state: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """The block in FP64 over the op's routes."""

    weights, ids = route(x, state["gate.weight"], TOP_K, renormalize=True)
    x64 = x.double()
    out = torch.zeros_like(x64)
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        gate, up = (x64[rows] @ state["experts.gate_up_proj"][expert].double().T).chunk(2, -1)
        down = (silu(gate) * up) @ state["experts.down_proj"][expert].double().T
        out.index_add_(0, rows, down * weights[rows, slots].double()[:, None])
    gate, up = (x64 @ state["shared_expert.gate_up_proj.weight"].double().T).chunk(2, -1)
    shared = (silu(gate) * up) @ state["shared_expert.down_proj.weight"].double().T
    return out + torch.sigmoid(x64 @ state["shared_expert_gate.weight"].double().T) * shared


def _error(output: torch.Tensor, reference: torch.Tensor) -> float:
    return ((output.double() - reference).norm() / reference.norm()).item()


def _check_near_reference(
    output: torch.Tensor, composed: torch.Tensor, reference: torch.Tensor
) -> None:
    floor = _error(reference.bfloat16(), reference)
    error = _error(output, reference)
    assert error <= FLOOR_MULTIPLE * floor, (error, floor)
    assert error <= COMPOSED_MULTIPLE * _error(composed, reference), error


@pytest.fixture(scope="module")
def state() -> dict[str, torch.Tensor]:
    return _state()


def test_the_whole_block_op_returns_an_empty_batch(state: dict[str, torch.Tensor]) -> None:
    assert _forward(_layer(FusedMoe(), state), _inputs(0)).shape == (0, HIDDEN)


@pytest.mark.parametrize("provider", ["triton_block", "composed", "one-row"])
def test_the_whole_block_op_agrees_with_the_composed_block(
    state: dict[str, torch.Tensor], provider: str
) -> None:
    rows = {"one-row": 1, **_rows_by_provider()}[provider]
    backend = FusedMoe()
    blocked, composed = _layer(backend, state), _layer(_Composed(), state)
    x = _inputs(rows)

    output = _forward(blocked, x)

    expected = _forward(composed, x)
    assert output.shape == (rows, HIDDEN) and torch.isfinite(output).all()
    op = backend.sparse_block(TOP_K, True, torch.bfloat16)
    assert op.provider == ("composed" if provider == "composed" else "triton_block")
    # The composition routes by the op's own router arithmetic.
    _, ids = route(x, state["gate.weight"], TOP_K, renormalize=True)
    assert torch.equal(op.routing_ids[:rows].long().sort(-1).values, ids.long().sort(-1).values)
    _check_near_reference(output, expected, _reference(state, x))


@pytest.mark.parametrize("provider", ["triton_block", "composed"])
def test_two_ranks_partials_sum_to_the_whole_block(
    state: dict[str, torch.Tensor], provider: str
) -> None:
    rows = _rows_by_provider()[provider]
    x = _inputs(rows)

    def summed(backend: type[FusedMoe]) -> torch.Tensor:
        # The exit all-reduce sums the ranks' BF16 partials.
        partials = [_forward(_layer(backend(), state, rank), x) for rank in range(2)]
        return (partials[0].float() + partials[1].float()).bfloat16()

    _check_near_reference(summed(FusedMoe), summed(_Composed), _reference(state, x))


def test_a_captured_graph_replays_the_eager_block(state: dict[str, torch.Tensor]) -> None:
    rows = min(4, _rows_by_provider()["triton_block"])
    layer = _layer(FusedMoe(), state)
    static_x = _inputs(rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        # Warm-up compiles and allocates outside the capture.
        layer.forward(static_x.clone())
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        output = layer.forward(static_x)

    for seed in (2, 3):
        x = _inputs(rows, seed)
        static_x.copy_(x)
        graph.replay()
        torch.testing.assert_close(output, _forward(layer, x), rtol=0, atol=0)


def test_the_router_picks_the_reference_experts_and_weights(state: dict[str, torch.Tensor]) -> None:
    """The op's routes against an FP64 softmax top-8 with renormalized weights over the same
    logits rounded once to BF16, as the op stores them: the same experts for every token whose
    eighth and ninth logits differ, weights within FP32 rounding."""

    x = _inputs(256)
    weights, ids = route(x, state["gate.weight"], TOP_K, renormalize=True)
    logits = (x.double() @ state["gate.weight"].double().T).bfloat16().double()
    top, reference_ids = logits.softmax(-1).topk(TOP_K + 1, dim=-1)
    decided = top[:, TOP_K - 1] > top[:, TOP_K]
    # BF16 logits tie the eighth and ninth experts of 34 of these 256 tokens; scaling the
    # router keeps the same ties, as BF16 rounds relative to each logit. The assertions below
    # compare the decided tokens only, of which at least 200 remain.
    assert int(decided.sum()) >= 200
    experts = ids.long().sort(-1).values
    assert torch.equal(experts[decided], reference_ids[:, :TOP_K].sort(-1).values[decided])
    reference = top[:, :TOP_K] / top[:, :TOP_K].sum(-1, keepdim=True)
    order = ids.long().argsort(-1)
    reference_order = reference_ids[:, :TOP_K].argsort(-1)
    torch.testing.assert_close(
        weights.double().gather(1, order)[decided],
        reference.gather(1, reference_order)[decided],
        rtol=1e-4,
        atol=1e-6,
    )


@pytest.mark.parametrize("provider", ["triton_block", "composed"])
def test_a_graph_captured_after_the_warm_up_survives_any_eager_block_on_its_stream(
    state: dict[str, torch.Tensor], provider: str
) -> None:
    """The runner's order on one stream: the layer's warm-up at the most rows, capture, then
    an eager batch of those rows, whose storage the op keeps; freed storage is taken by other
    tensors. Replays still compute the captured rows exactly as an eager call does."""

    limits = _rows_by_provider()
    rows = min(4, limits["triton_block"]) if provider == "triton_block" else limits["composed"]
    max_rows = 1024
    layer = _layer(FusedMoe(), state)
    static_x = _inputs(rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        layer.warm_up_rows(max_rows, dtype=torch.bfloat16, device=DEVICE, warmed=set())
        layer.forward(static_x.clone())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = layer.forward(static_x)
        layer.forward(_inputs(max_rows, 7))
        filler = torch.full((1 << 26,), float("nan"), device=DEVICE)
        for seed in (2, 3):
            x = _inputs(rows, seed)
            static_x.copy_(x)
            graph.replay()
            expected = layer.forward(x.clone())
            torch.cuda.current_stream().synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    del filler


@pytest.mark.parametrize("provider", ["triton_block", "composed"])
def test_a_graph_captured_before_a_larger_eager_block_keeps_what_it_reads(
    state: dict[str, torch.Tensor], provider: str
) -> None:
    """The op's own contract, without the runner's warm-up: capture, then an eager batch of
    more rows on the stream, then freed storage taken by other tensors. Replays must still
    compute the captured rows exactly as an eager call does."""

    limits = _rows_by_provider()
    if provider == "triton_block":
        rows, larger = min(4, limits["triton_block"]), limits["triton_block"]
    else:
        rows, larger = limits["composed"], 1024
    layer = _layer(FusedMoe(), state)
    static_x = _inputs(rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        layer.forward(static_x.clone())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = layer.forward(static_x)
        layer.forward(_inputs(larger, 7))
        filler = torch.full((1 << 26,), float("nan"), device=DEVICE)
        for seed in (2, 3):
            x = _inputs(rows, seed)
            static_x.copy_(x)
            graph.replay()
            expected = layer.forward(x.clone())
            torch.cuda.current_stream().synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    del filler


def _block_args(state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (
        state["gate.weight"],
        state["experts.gate_up_proj"],
        state["experts.down_proj"],
        state["shared_expert.gate_up_proj.weight"],
        state["shared_expert.down_proj.weight"],
        state["shared_expert_gate.weight"],
    )


@pytest.mark.parametrize("masked", [True, False], ids=["valid_tokens", "unmasked"])
@pytest.mark.parametrize("provider", ["triton_block", "composed"])
def test_a_padded_graph_keeps_its_real_rows_apart_from_the_padding(
    state: dict[str, torch.Tensor], provider: str, masked: bool
) -> None:
    """A graph captured at a bucket of rows, replayed with fewer real rows: the real rows do
    not depend on what the padding rows hold, and with ``valid_tokens`` (the real row count, a
    device scalar the replay updates) the padding rows are zero. The fused kernel's real rows
    are also bitwise those of an eager call on them alone; the composition's GEMMs pick their
    tiles by the row count, so it matches an eager call of the bucket's rows instead."""

    limits = _rows_by_provider()
    if provider == "triton_block":
        bucket = min(8, limits["triton_block"])
        counts = sorted({1, max(1, bucket // 2), bucket - 1})
    else:
        bucket = 2 * limits["composed"]
        counts = [limits["composed"], bucket - 1]
    args = _block_args(state)
    block, eager = MoEBlock(), MoEBlock()
    static_x = _inputs(bucket)
    valid = torch.full((1,), bucket, dtype=torch.int32, device=DEVICE)
    mask = {"valid_tokens": valid} if masked else {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        block(static_x, *args, **mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = block(static_x, *args, **mask)
        assert block.provider == provider
        for count in counts:
            x = _inputs(count, seed=10 + count)
            replays = []
            for padding in (_inputs(bucket, seed=99), torch.zeros_like(static_x)):
                static_x.copy_(padding)
                static_x[:count].copy_(x)
                valid.fill_(count)
                graph.replay()
                replays.append(output.clone())
            alone = eager(x, *args)
            assert eager.provider == provider, count
            bucketed = eager(static_x.clone(), *args, **mask)
            torch.cuda.current_stream().synchronize()
            assert torch.equal(replays[0][:count], replays[1][:count]), count
            assert torch.equal(replays[1][:count], bucketed[:count]), count
            if provider == "triton_block":
                assert torch.equal(replays[0][:count], alone), count
            if masked:
                assert not replays[0][count:].any(), count
