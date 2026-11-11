"""Per-expert LoRA through the fused MoE backend on CUDA, at Qwen3.5-35B-A3B's routing (256
experts, top 8) and widths (hidden 2048, intermediate 512).

An adapted block equals its merged weights in FP32 within BF16 rounding at every rank the
"experts" target takes; a zero adapter changes nothing bitwise; a captured graph replays the
eager result; two expert-parallel ranks with their factor windows sum to one rank, as
partials on one GPU and through the layer's NCCL all-reduce over two.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from pathlib import Path

import pytest
import torch

from cti.config import LoRAConfig
from cti.distributed import CollectiveGroup, Topology
from cti.distributed.runtime import ParallelGroup
from cti.layers import Adapter, ExpertParallel, LayerServices, MoELayer, Placement, Site
from cti.moe.fused import FusedMoe
from tests.support.expert_lora import (
    NamedExpertLoRA,
    layer_adapter_window,
    merged_experts,
    random_expert_factors,
    random_layer_adapter,
)
from tests.support.moe import ComposedFusedMoe, ReferenceMoe, rank_state
from tests.support.tp_shards import DryGroup, default_dtype

pytestmark = pytest.mark.checkpoints()

# Every MoE forward runs eagerly in these tests: no graph breaks.
DEVICE = torch.device("cuda")
NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, SHARED = 256, 8, 2048, 512, 512
ALPHA = 16.0
SIZE = 2


def _placement(rank: int) -> Placement:
    """Rank ``rank``'s experts of ``SIZE`` expert-parallel ranks."""

    return ExpertParallel().shard(NUM_EXPERTS, INTERMEDIATE, Topology(rank, SIZE))


def _weights() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(0)

    def draw(*shape: int) -> torch.Tensor:
        scale = shape[-1] ** -0.5
        return (scale * torch.randn(*shape, generator=generator, device=DEVICE)).bfloat16()

    return draw(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN), draw(NUM_EXPERTS, HIDDEN, INTERMEDIATE)


def _factors(
    rank: int, *, zero: bool = False, per_projection: bool = False
) -> tuple[dict[str, torch.Tensor], NamedExpertLoRA]:
    """FP32 factors by name, and their BF16 ``moe.ExpertLoRA``; ``B @ A * scaling`` is about
    a fifth of a weight."""

    named = random_expert_factors(
        NUM_EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        rank,
        generator=torch.Generator().manual_seed(rank),
        per_projection=per_projection,
    )
    for name in named:
        if name.endswith("lora_B"):
            named[name] = named[name] * (0.0 if zero else 0.2 * rank / ALPHA)
    device = {name: t.to(DEVICE, torch.bfloat16) for name, t in named.items()}
    # The merge reads the factors the kernels do.
    return {n: t.float() for n, t in device.items()}, NamedExpertLoRA(device, ALPHA / rank)


# The router reads each expert's logit from its own input column.
def _router(device: torch.device = DEVICE) -> torch.Tensor:
    return torch.eye(NUM_EXPERTS, HIDDEN, device=device).bfloat16()


def _inputs(num_tokens: int, seed: int = 1, device: torch.device = DEVICE) -> torch.Tensor:
    """Rows whose router logits, their first NUM_EXPERTS columns, are distinct within each row
    (k / 256, exact in BF16 and at the other columns' scale)."""

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, generator=generator)
    order = torch.rand(num_tokens, NUM_EXPERTS, generator=generator).argsort(-1)
    x[:, :NUM_EXPERTS] = order.float() / 256
    return x.bfloat16().to(device)


@pytest.mark.parametrize("per_projection", [False, True], ids=["shared-a", "per-projection"])
@pytest.mark.parametrize("rank", [8, 64, 128])
@pytest.mark.parametrize("num_tokens", [1, 4, 64, 300, 2048])
def test_an_adapted_block_equals_its_merged_weights(
    rank: int, num_tokens: int, per_projection: bool
) -> None:
    w1, w2 = _weights()
    named, lora = _factors(rank, per_projection=per_projection)
    x = _inputs(num_tokens)
    backend = FusedMoe()

    output = backend.forward(x.clone(), w1, w2, _router(), TOP_K, True, lora=lora).float()

    def fp32(gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        return ReferenceMoe().forward(x.float(), gate_up, down, _router(), TOP_K, True)

    error = (
        output - fp32(*merged_experts(w1.float(), w2.float(), named, scaling=lora.scaling))
    ).abs()
    base = backend.forward(x.clone(), w1, w2, _router(), TOP_K, True).float()
    # The unadapted block's BF16 floor on the same rows. Each delta rounds the cache once
    # more: twice the floor bounds that, and a dropped delta misses by far more.
    floor = (base - fp32(w1.float(), w2.float())).abs()
    assert torch.isfinite(output).all()
    assert float(error.max()) <= 2 * float(floor.max()), (float(error.max()), float(floor.max()))
    assert float(error.mean()) <= 2 * float(floor.mean()), (
        float(error.mean()),
        float(floor.mean()),
    )
    assert float((output - base).abs().max()) > 10 * float(floor.max())


@pytest.mark.parametrize("num_tokens", [1, 300])
def test_a_zero_adapter_is_bitwise_the_base(num_tokens: int) -> None:
    w1, w2 = _weights()
    _, lora = _factors(16, zero=True)
    x = _inputs(num_tokens)
    placement = _placement(1)
    held = slice(placement.experts.start, placement.experts.stop)
    expert_map = placement.expert_map().to(DEVICE)
    backend = ComposedFusedMoe()

    whole = backend.forward(x.clone(), w1, w2, _router(), TOP_K, True, lora=lora)
    partial = backend.forward(
        x.clone(),
        w1[held],
        w2[held],
        _router(),
        TOP_K,
        True,
        expert_map=expert_map,
        lora=lora.window(placement),
    )

    assert torch.equal(whole, backend.forward(x.clone(), w1, w2, _router(), TOP_K, True))
    expected = backend.forward(
        x.clone(), w1[held], w2[held], _router(), TOP_K, True, expert_map=expert_map
    )
    assert torch.equal(partial, expected)


@pytest.mark.parametrize("num_tokens", [0, 1, 4, 300])
def test_two_ranks_adapted_partials_sum_to_the_whole(num_tokens: int) -> None:
    w1, w2 = _weights()
    _, lora = _factors(32)
    x = _inputs(num_tokens)
    backend = FusedMoe()

    whole = backend.forward(x.clone(), w1, w2, _router(), TOP_K, True, lora=lora).float()
    total = torch.zeros_like(whole)
    for rank in range(SIZE):
        placement = _placement(rank)
        held = slice(placement.experts.start, placement.experts.stop)
        total += backend.forward(
            x.clone(),
            w1[held],
            w2[held],
            _router(),
            TOP_K,
            True,
            expert_map=placement.expert_map().to(DEVICE),
            lora=lora.window(placement),
        ).float()

    # Each partial rounds to BF16 once more than the whole.
    torch.testing.assert_close(total, whole, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("mapped", [False, True], ids=["whole", "expert-map"])
def test_a_captured_graph_replays_the_eager_adapted_block(mapped: bool) -> None:
    w1, w2 = _weights()
    _, lora = _factors(64)
    expert_map = None
    if mapped:
        placement = _placement(1)
        held = slice(placement.experts.start, placement.experts.stop)
        w1, w2 = w1[held].contiguous(), w2[held].contiguous()
        expert_map, lora = placement.expert_map().to(DEVICE), lora.window(placement)
    backend = FusedMoe()
    x = _inputs(4)

    router = _router()

    def run(hidden: torch.Tensor) -> torch.Tensor:
        return backend.forward(
            hidden, w1, w2, router, TOP_K, True, expert_map=expert_map, lora=lora
        )

    static_x = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(static_x.clone())  # warm-up compiles outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(static_x)

    for seed in (2, 3):
        new_x = _inputs(4, seed)
        static_x.copy_(new_x)
        graph.replay()
        torch.testing.assert_close(output, run(new_x.clone()), rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 16, 64, 65, 300])
def test_an_adapted_block_joins_the_shared_expert_as_the_torch_join(num_tokens: int) -> None:
    # cti-kernels joins it in its combine on BF16 stacks, in the order of the layer's torch
    # join: r(r(r(sigmoid(gate)) * shared) + routed), eagerly and replayed.
    w1, w2 = _weights()
    _, lora = _factors(64)
    backend, router, x = FusedMoe(), _router(), _inputs(num_tokens)
    generator = torch.Generator(device=DEVICE).manual_seed(num_tokens)
    shared = (2 * torch.randn(num_tokens, HIDDEN, generator=generator, device=DEVICE)).bfloat16()
    gate = torch.randn(num_tokens, 1, generator=generator, device=DEVICE).bfloat16()

    def run(hidden: torch.Tensor, **join: torch.Tensor) -> torch.Tensor:
        return backend.forward(hidden, w1, w2, router, TOP_K, True, lora=lora, **join)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        expected = run(x.clone()).add_(shared * torch.sigmoid(gate))
        joined = run(x.clone(), shared=shared, shared_gate=gate)
        static_x = x.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run(static_x, shared=shared, shared_gate=gate)
        static_x.copy_(x)
        graph.replay()
    stream.synchronize()

    assert torch.equal(joined, expected)
    assert torch.equal(captured, joined)


def _layer(group: CollectiveGroup, device: torch.device, *, lora: str | None) -> MoELayer:
    site = Site(group, ExpertParallel(), device, services=LayerServices())
    with torch.device("meta"), default_dtype(torch.bfloat16):
        return MoELayer(
            NUM_EXPERTS,
            TOP_K,
            HIDDEN,
            INTERMEDIATE,
            renormalize=True,
            site=site,
            shared_expert_intermediate_size=SHARED,
            lora=lora,
        )


def _whole_state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    layer = _layer(DryGroup(0, 1), DEVICE, lora=None)
    return {
        name: (torch.randn(t.shape, generator=generator) * t.shape[-1] ** -0.5).bfloat16()
        for name, t in layer.state_dict().items()
    }


def _whole_adapter(rank: int) -> dict[str, torch.Tensor]:
    """A non-zero BF16 adapter of the whole layer by ``Adapter`` name."""

    generator = torch.Generator().manual_seed(5)
    adapter = random_layer_adapter(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE, SHARED, rank, generator=generator
    )
    scale = 0.2 * rank / ALPHA
    return {n: (t * scale if n.endswith("lora_B") else t).bfloat16() for n, t in adapter.items()}


def _loaded_layer(
    group: CollectiveGroup, device: torch.device = DEVICE, *, lora: str | None = None
) -> MoELayer:
    """The layer with this rank's share of ``_whole_state``."""

    layer = _layer(group, device, lora=lora)
    state = rank_state(_whole_state(), layer)
    layer.load_state_dict({name: tensor.to(device) for name, tensor in state.items()})
    if lora is None:
        layer.settle_inference()
    return layer


def _adapted_layer(
    group: CollectiveGroup, rank: int, size: int, device: torch.device = DEVICE
) -> MoELayer:
    layer = _loaded_layer(group, device, lora="experts")
    adapters = Adapter(LoRAConfig(targets=("experts",), rank=16, alpha=ALPHA))
    adapters.attach(layer)
    adapters.materialize(device=device, dtype=torch.bfloat16)
    local = layer_adapter_window(_whole_adapter(16), layer.experts.placement, rank, size)
    adapters.load_state_dict({name: t.to(device) for name, t in local.items()})
    layer.settle_inference()
    return layer


def _nccl_rank(rank: int, rendezvous: str, connection) -> None:
    try:
        torch.cuda.set_device(rank)
        runtime = ParallelGroup(
            Topology(rank, SIZE),
            device=torch.device("cuda", rank),
            timeout_seconds=300,
            init_method=rendezvous,
            use_pynccl=False,
        )
        try:
            device = torch.device("cuda", rank)
            layer = _adapted_layer(runtime, rank, SIZE, device)
            x = _inputs(8, device=device)
            with torch.inference_mode():
                output = layer.forward(x).float().cpu()
            connection.send(("ok", output.tolist()))
        finally:
            runtime.close()
    except BaseException:  # noqa: BLE001 - the parent asserts on the reported failure
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def test_two_gpus_reduce_their_adapted_partials_to_one_gpus_block(tmp_path: Path) -> None:
    if torch.cuda.device_count() < SIZE:
        pytest.skip(f"needs {SIZE} visible GPUs")
    x = _inputs(8)
    with torch.inference_mode():
        expected = _adapted_layer(DryGroup(0, 1), 0, 1).forward(x.clone()).float().cpu()
        unadapted = _loaded_layer(DryGroup(0, 1)).forward(x.clone()).float().cpu()

    context = mp.get_context("spawn")
    outputs = []
    processes = []
    for rank in range(SIZE):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_nccl_rank, args=(rank, f"file://{tmp_path}/group", child), daemon=True
        )
        process.start()
        child.close()
        processes.append((process, parent))
    for rank, (process, parent) in enumerate(processes):
        assert parent.poll(600), f"rank {rank} reported nothing"
        status, value = parent.recv()
        assert status == "ok", f"rank {rank} failed:\n{value}"
        outputs.append(torch.tensor(value))
        process.join(60)

    # Each rank rounds its partial to BF16 at the partial's own size, which the BF16
    # reduction can cancel: up to an ULP of the partials (2^-6 below 4).
    tolerance = {"rtol": 2e-2, "atol": 2**-6}
    for output in outputs:
        torch.testing.assert_close(output, expected, **tolerance)
    # A dropped partial or delta misses by far more.
    assert float((expected - unadapted).abs().max()) > 4 * tolerance["atol"]
