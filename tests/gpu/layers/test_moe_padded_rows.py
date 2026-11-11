"""A sparse block over a breakable graph's padded row bucket, on CUDA at Qwen3.5-35B-A3B's
routing (256 experts, top 8) and widths: padding rows route and run the experts like real
ones and are dropped, as vLLM and SGLang capture MoE (``MoELayer.forward``).

Over every placement (whole, an expert-parallel and an expert tensor-parallel rank's
partial) and with and without per-expert LoRA: the real rows route to the same experts as
alone; their outputs follow the unpadded block within BF16 rounding (the grouped GEMMs pick
their tiles by row count, which cti-kernels leaves batch-variant); a captured bucket replays
the eager padded block bitwise; and the largest bucket, all padding but one row of zeros,
completes.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator

import pytest
import torch

from cti.config import LoRAConfig
from cti.layers import (
    Adapter,
    ExpertParallel,
    ExpertTensorParallel,
    LayerServices,
    MoELayer,
    Site,
)
from cti.moe.fused import FusedMoe
from tests.support.expert_lora import layer_adapter_window, random_layer_adapter
from tests.support.tp_shards import default_dtype

pytestmark = pytest.mark.checkpoints()

DEVICE = torch.device("cuda")
NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, SHARED = 256, 8, 2048, 512, 512
RANK, ALPHA = 8, 16.0
# The breakable graph's largest default row bucket.
LARGEST_BUCKET = 512
# Real rows against the unpadded block, relative to its largest output. The floor is batch
# composition alone: the same rows beside other real rows, unpadded, move by at most 1.08e-2
# (ETP with LoRA; 0.72-0.96e-2 elsewhere) on an RTX PRO 6000 with a cti-kernels prerelease,
# as the grouped GEMMs and the torch LoRA deltas tile by row count; padding moved them by at
# most 0.76e-2 (bitwise for the whole block without LoRA). 1.5x the floor.
TOLERANCE = 1.6e-2


class _Rank:
    """A rank whose exit all-reduce leaves its partial sum as it is."""

    def __init__(self, rank: int, size: int) -> None:
        self.rank, self.size = rank, size

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


_PLACEMENTS = {
    "whole": (ExpertParallel(), _Rank(0, 1)),
    "ep": (ExpertParallel(), _Rank(1, 2)),
    "etp": (ExpertTensorParallel(), _Rank(1, 2)),
}


@functools.cache
def _layer(placement: str, lora: bool) -> tuple[MoELayer, FusedMoe]:
    """The block at ``placement``, built once per module, and its backend."""

    experts, group = _PLACEMENTS[placement]
    services = LayerServices()
    backend = services.moe(DEVICE)
    site = Site(group, experts, DEVICE, services=services)
    with torch.device("meta"), default_dtype(torch.bfloat16):
        layer = MoELayer(
            NUM_EXPERTS,
            TOP_K,
            HIDDEN,
            INTERMEDIATE,
            renormalize=True,
            site=site,
            shared_expert_intermediate_size=SHARED,
            lora="experts" if lora else None,
        )
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    layer.load_state_dict(
        {
            param.path: (
                param.shape[-1] ** -0.5
                * torch.randn(param.shape, generator=generator, device=DEVICE)
            ).bfloat16()
            for param in layer.params()
        }
    )
    if lora:
        adapters = Adapter(LoRAConfig(targets=("experts",), rank=RANK, alpha=ALPHA))
        adapters.attach(layer)
        adapters.materialize(device=DEVICE, dtype=torch.bfloat16)
        whole = random_layer_adapter(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            SHARED,
            RANK,
            generator=torch.Generator().manual_seed(1),
        )
        local = layer_adapter_window(whole, layer.experts.placement, group.rank, group.size)
        adapters.load_state_dict({n: t.to(DEVICE, torch.bfloat16) for n, t in local.items()})
    layer.settle_inference()
    return layer, backend


@pytest.fixture(scope="module", autouse=True)
def _module_layers() -> Iterator[None]:
    """Drop the built blocks with the module: the cache would hold them for the session."""

    yield
    _layer.cache_clear()


def _rows(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(count, HIDDEN, generator=generator, device=DEVICE).bfloat16()


def _padded(real: torch.Tensor, bucket: int, padding: str) -> torch.Tensor:
    """``real`` in a ``bucket``-row buffer whose tail holds zeros (the graph's zeroed bridges)
    or a previous step's rows (its stale token buffer)."""

    tail = bucket - real.shape[0]
    rest = torch.zeros(tail, HIDDEN) if padding == "zeros" else _rows(tail, seed=7).cpu()
    return torch.cat((real, rest.to(DEVICE, real.dtype)))


def _forward(layer: MoELayer, x: torch.Tensor) -> torch.Tensor:
    # The block may write its output over its input.
    with torch.inference_mode():
        return layer.forward(x.clone())


@pytest.mark.parametrize("placement", sorted(_PLACEMENTS))
@pytest.mark.parametrize("lora", [False, True], ids=["base", "lora"])
@pytest.mark.parametrize(("rows", "bucket"), [(1, 16), (50, 64), (127, 128), (300, 512)])
@pytest.mark.parametrize("padding", ["zeros", "stale"])
def test_padding_rows_leave_the_real_rows_routes_and_outputs(
    placement: str, lora: bool, rows: int, bucket: int, padding: str
) -> None:
    layer, backend = _layer(placement, lora)
    real = _rows(rows, seed=1)
    padded = _padded(real, bucket, padding)
    router = layer.gate.weight

    with torch.inference_mode():
        _, alone = backend.route(real, router, TOP_K, True)
        _, among = backend.route(padded, router, TOP_K, True)
    assert torch.equal(among[:rows], alone)
    alone_output = _forward(layer, real).float()
    gap = (_forward(layer, padded)[:rows].float() - alone_output).abs().max()
    assert float(gap) <= TOLERANCE * float(alone_output.abs().max())


@pytest.mark.parametrize("placement", sorted(_PLACEMENTS))
@pytest.mark.parametrize("lora", [False, True], ids=["base", "lora"])
@pytest.mark.parametrize(("rows", "bucket"), [(50, 64), (1, LARGEST_BUCKET)])
def test_a_captured_bucket_replays_the_eager_padded_block(
    placement: str, lora: bool, rows: int, bucket: int
) -> None:
    layer, _ = _layer(placement, lora)
    static = _padded(_rows(rows, seed=1), bucket, "zeros")
    source = torch.empty_like(static)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        # Warm at the bucket's rows before capture, as the breakable graph does.
        source.copy_(static)
        layer.forward(source)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            source.copy_(static)
            output = layer.forward(source)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (2, 3):
        static.copy_(_padded(_rows(rows, seed=seed), bucket, "zeros"))
        graph.replay()
        torch.testing.assert_close(output, _forward(layer, static), rtol=0, atol=0)
    assert bool(torch.isfinite(output[:rows]).all())
