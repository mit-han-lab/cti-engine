"""The fused MoE backend with an expert map on CUDA: two ranks' partial sums, each over its own
half of the experts, give the single-rank output, through cti-kernels' expert-map kernel and
through the zero-weight remap that the backend's composition runs; a route to the other
half adds exactly zero, and a captured CUDA graph replays the eager result."""

from __future__ import annotations

import pytest
import torch

from cti.distributed import Topology
from cti.layers import ExpertParallel, Placement
from cti.layers.moe import Experts
from cti.moe import BaseMoeBackend
from cti.moe.fused import FusedMoe, compose
from ctikernels.moe import route
from tests.support.moe import ReferenceMoe
from tests.support.tp_shards import default_dtype, site

pytestmark = pytest.mark.checkpoints()

DEVICE = torch.device("cuda")
# Qwen3.5-35B-A3B routing (256 experts, top 8) at smaller widths.
NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE = 256, 8, 512, 128
SIZE = 2


class _ZeroWeight(BaseMoeBackend):
    """The fused backend's composition at any dtype: a route to another rank's expert runs on
    local expert 0 at weight exactly 0."""

    def route(self, hidden_states, router, topk, renormalize):
        return route(hidden_states, router, topk, renormalize=renormalize)

    def forward(
        self,
        hidden_states,
        w1,
        w2,
        router,
        topk,
        renormalize,
        expert_map=None,
        lora=None,
        shared=None,
        shared_gate=None,
    ):
        # An unadapted block joins no shared expert here.
        assert lora is None and shared is None
        weights, ids = self.route(hidden_states, router, topk, renormalize)
        return compose(hidden_states, w1, w2, weights, ids, expert_map)


@pytest.fixture(params=["kernel", "zero-weight"])
def backend(request: pytest.FixtureRequest) -> BaseMoeBackend:
    return _ZeroWeight() if request.param == "zero-weight" else FusedMoe()


def _placement(rank: int) -> Placement:
    return ExpertParallel().shard(NUM_EXPERTS, INTERMEDIATE, Topology(rank, SIZE))


def _weights() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(0)

    def draw(*shape: int) -> torch.Tensor:
        scale = shape[-1] ** -0.5
        return (scale * torch.randn(*shape, generator=generator, device=DEVICE)).bfloat16()

    return draw(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN), draw(NUM_EXPERTS, HIDDEN, INTERMEDIATE)


# The router reads each expert's logit from its own input column: a row's first NUM_EXPERTS
# columns are its logits, exact in any arithmetic.
def _router(device: torch.device = DEVICE) -> torch.Tensor:
    return torch.eye(NUM_EXPERTS, HIDDEN, device=device).bfloat16()


def _inputs(num_tokens: int, seed: int = 1) -> torch.Tensor:
    """Rows whose router logits are distinct within each row (k / 256, exact in BF16 and at
    the other columns' scale): a tie at the top-k boundary would break by expert id in the
    kernel and by sort order in the reference."""

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, generator=generator, device=DEVICE)
    order = torch.rand(num_tokens, NUM_EXPERTS, generator=generator, device=DEVICE).argsort(-1)
    x[:, :NUM_EXPERTS] = order.float() / 256
    return x.bfloat16()


def _partial(backend: BaseMoeBackend, rank: int, x, w1, w2) -> torch.Tensor:
    placement = _placement(rank)
    held = slice(placement.experts.start, placement.experts.stop)
    expert_map = placement.expert_map().to(DEVICE)
    return backend.forward(
        x.clone(), w1[held], w2[held], _router(), TOP_K, True, expert_map=expert_map
    )


@pytest.mark.parametrize("num_tokens", [0, 1, 4, 64, 300])
def test_two_ranks_partials_sum_to_the_whole(backend: BaseMoeBackend, num_tokens: int) -> None:
    w1, w2 = _weights()
    x = _inputs(num_tokens)

    whole = backend.forward(x.clone(), w1, w2, _router(), TOP_K, True)
    partials = [_partial(backend, rank, x, w1, w2) for rank in range(SIZE)]

    total = partials[0].float() + partials[1].float()
    assert total.shape == (num_tokens, HIDDEN) and torch.isfinite(total).all()
    # Each partial rounds to BF16 once more than the whole.
    torch.testing.assert_close(total, whole.float(), rtol=2e-2, atol=2e-3)
    reference = ReferenceMoe().forward(
        x.cpu().float(),
        w1.cpu().float(),
        w2.cpu().float(),
        _router(torch.device("cpu")),
        TOP_K,
        True,
    )
    torch.testing.assert_close(total.cpu(), reference, rtol=2e-2, atol=4e-3)


def test_a_route_to_the_other_half_adds_exactly_zero(backend: BaseMoeBackend) -> None:
    w1, w2 = _weights()
    x = _inputs(16)
    # Rows 0-7 pick only rank 1's experts; rows 8-15 one of rank 0's among them.
    x[:, : NUM_EXPERTS // 2] = -30.0
    x[8:, 3] = 30.0

    first = _partial(backend, 0, x, w1, w2)

    assert torch.equal(first[:8], torch.zeros_like(first[:8]))
    assert (first[8:] != 0).any(dim=-1).all()


def test_a_captured_graph_replays_the_eager_partial(backend: BaseMoeBackend) -> None:
    w1, w2 = _weights()
    placement = _placement(1)
    held = slice(placement.experts.start, placement.experts.stop)
    w1, w2 = w1[held].contiguous(), w2[held].contiguous()
    expert_map, router = placement.expert_map().to(DEVICE), _router()

    def run(hidden: torch.Tensor) -> torch.Tensor:
        return backend.forward(hidden, w1, w2, router, TOP_K, True, expert_map=expert_map)

    static_x = _inputs(4)
    # The engine warms up, captures and runs eagerly on one stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(static_x.clone())  # warm-up compiles outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run(static_x)
        # A larger eager batch after the capture must leave the graph's intermediates alone:
        # live NaN sentinels of their sizes would take any it freed.
        run(_inputs(300, seed=4))
        routes = 4 * TOP_K
        sentinels = [
            torch.full((size,), float("nan"), device=DEVICE, dtype=torch.bfloat16)
            for size in (routes * 2 * INTERMEDIATE, routes * HIDDEN, routes * INTERMEDIATE)
            for _ in range(32)
        ]
    torch.cuda.current_stream().wait_stream(stream)

    for seed in (2, 3):
        new_x = _inputs(4, seed)
        static_x.copy_(new_x)
        graph.replay()
        torch.testing.assert_close(output, run(new_x.clone()), rtol=0, atol=0)
    assert all(sentinel.isnan().all() for sentinel in sentinels)


def test_partial_experts_hold_their_map_on_their_sites_device(backend: BaseMoeBackend) -> None:
    w1, w2 = _weights()
    placement = _placement(1)
    with torch.device("meta"), default_dtype(torch.bfloat16):
        experts = Experts(
            NUM_EXPERTS,
            TOP_K,
            HIDDEN,
            INTERMEDIATE,
            renormalize=True,
            site=site(Topology(1, SIZE), device=DEVICE, moe_backend=backend),
        )
    held = slice(placement.experts.start, placement.experts.stop)
    experts.load_state_dict(
        {"gate_up_proj": w1[held].contiguous(), "down_proj": w2[held].contiguous()}
    )
    x = _inputs(16)

    output = experts.forward(x.clone(), _router())

    expected = _partial(backend, 1, x, w1, w2)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
