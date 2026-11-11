from __future__ import annotations

from pathlib import Path

import pytest
import safetensors.torch
import torch
import torch.nn.functional as F

import cti.layers.activation as activation_module
import cti.moe.fused as fused_module
from cti.distributed import Topology
from cti.layers import (
    ExpertParallel,
    ExpertPlacement,
    ExpertTensorParallel,
    MoELayer,
    Placement,
    prepare_inference,
)
from cti.layers.moe import Experts
from cti.models.mapping import CheckpointMapping, MergeRule
from cti.models.weight import load_weight
from cti.moe import BaseMoeBackend
from cti.moe.fused import FusedMoe, local_routes
from ctikernels.moe import MoEBlock
from tests.support import reference_ops
from tests.support.moe import BlockReferenceMoe, ReferenceMoe
from tests.support.tp_shards import site

CPU = torch.device("cpu")
# Another CPU host's BLAS sums FP32 partials in another order: measured worst 1.4e-6.
_FP32_REORDERED = {"rtol": 1e-5, "atol": 1e-5}
_PLACEMENTS = [ExpertParallel(), ExpertTensorParallel()]
_IDS = ["ep", "etp"]


class _InPlaceBackend(BaseMoeBackend):
    """Record each call and, like the fused backend, write the output over the input."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def route(self, hidden_states, router, topk, renormalize):
        raise AssertionError("the layer routes inside its backend's forward")

    def forward(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        hidden_states = kwargs["hidden_states"]
        assert isinstance(hidden_states, torch.Tensor)
        return hidden_states.mul_(2)


class _Group:
    """A rank of a group that keeps its partial sums: ``all_reduce`` records its input."""

    def __init__(self, rank: int, size: int) -> None:
        self.rank, self.size = rank, size
        self.reduced: list[torch.Tensor] = []

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.reduced.append(tensor.clone())
        return tensor


@pytest.fixture(autouse=True)
def _reference_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        activation_module.inference_activation, "silu_and_mul", reference_ops.silu_and_mul
    )


def _randomized(layer: MoELayer) -> MoELayer:
    """``layer`` with random weights, settled for inference as prepare_inference settles it."""

    layer.load_state_dict({name: torch.randn_like(t) for name, t in layer.state_dict().items()})
    layer.settle_inference()
    return layer


@pytest.mark.parametrize(
    ("experts", "intermediate", "message"),
    [
        (range(-1, 2), range(4), r"^expert range \[-1, 2\) is not inside 4$"),
        (range(2, 2), range(4), r"^expert range \[2, 2\) is not inside 4$"),
        (range(3, 5), range(4), r"^expert range \[3, 5\) is not inside 4$"),
        (range(4), range(2, 6), r"^intermediate range \[2, 6\) is not inside 4$"),
    ],
)
def test_placement_rejects_a_range_outside_its_experts(
    experts: range, intermediate: range, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Placement(4, 4, experts, intermediate)


def test_expert_parallel_splits_whole_experts_evenly_and_maps_global_to_local() -> None:
    placement = ExpertParallel().shard(8, 4, Topology(1, 2))

    assert placement == Placement(8, 4, range(4, 8), range(4))
    assert not placement.holds_every_expert
    assert placement.expert_map().tolist() == [-1, -1, -1, -1, 0, 1, 2, 3]
    assert ExpertParallel().shard(8, 4, Topology(0, 1)).is_whole
    with pytest.raises(ValueError, match="^8 experts do not divide evenly over 3 ranks$"):
        ExpertParallel().shard(8, 4, Topology(0, 3))


def test_expert_tensor_parallel_cuts_every_experts_width_evenly() -> None:
    placement = ExpertTensorParallel().shard(8, 6, Topology(1, 2))

    assert placement == Placement(8, 6, range(8), range(3, 6))
    assert placement.holds_every_expert and not placement.is_whole
    assert ExpertTensorParallel().shard(8, 6, Topology(0, 1)).is_whole
    with pytest.raises(ValueError, match="^6 expert intermediate rows do not divide evenly"):
        ExpertTensorParallel().shard(8, 6, Topology(0, 4))


def test_moe_layer_hands_the_backend_its_router_in_one_call() -> None:
    torch.manual_seed(0)
    backend = _InPlaceBackend()
    layer = _randomized(MoELayer(4, 2, 8, 4, renormalize=False, site=site(moe_backend=backend)))
    x = torch.randn(3, 8)

    output = layer.forward(x)

    (call,) = backend.calls
    assert call["hidden_states"] is x and output is x
    assert call["w1"] is layer.experts.gate_up_proj and call["w2"] is layer.experts.down_proj
    assert call["router"] is layer.gate.weight
    assert (call["topk"], call["renormalize"], call["expert_map"]) == (2, False, None)
    # The checkpoint names and the order dummy weights are drawn in.
    assert list(layer.state_dict()) == ["experts.gate_up_proj", "experts.down_proj", "gate.weight"]
    assert all(param.windows == () and param.experts is None for param in layer.params())


def test_the_shared_expert_reads_the_input_before_the_backend_overwrites_it() -> None:
    torch.manual_seed(0)
    layer = _randomized(
        MoELayer(
            4,
            2,
            8,
            4,
            renormalize=True,
            site=site(moe_backend=_InPlaceBackend()),
            shared_expert_intermediate_size=6,
        )
    )
    x = torch.randn(3, 8)
    shared = layer.shared_expert
    gate, up = F.linear(x, shared.gate_up_proj.weight).chunk(2, dim=-1)
    expected_shared = F.linear(F.silu(gate) * up, shared.down_proj.weight) * torch.sigmoid(
        F.linear(x, layer.shared_expert_gate.weight)
    )
    expected = 2 * x + expected_shared

    torch.testing.assert_close(layer.forward(x), expected)
    assert list(layer.state_dict()) == [
        "experts.gate_up_proj",
        "experts.down_proj",
        "gate.weight",
        "shared_expert.gate_up_proj.weight",
        "shared_expert.down_proj.weight",
        "shared_expert_gate.weight",
    ]


@pytest.mark.parametrize("stacked", [False, True], ids=["per-expert", "stacked"])
@pytest.mark.parametrize("placement", _PLACEMENTS, ids=_IDS)
def test_each_ranks_share_loads_its_window_and_together_the_whole(
    tmp_path: Path, placement: ExpertPlacement, stacked: bool
) -> None:
    torch.manual_seed(0)
    num_experts, hidden_size, intermediate_size = 4, 8, 4
    gate_up = torch.randn(num_experts, 2 * intermediate_size, hidden_size)
    down = torch.randn(num_experts, hidden_size, intermediate_size)
    if stacked:
        tensors = {"mlp.experts.gate_up_proj": gate_up, "mlp.experts.down_proj": down}
    else:
        tensors = {
            f"mlp.experts.{expert}.{name}.weight": tensor[expert]
            for expert in range(num_experts)
            for name, tensor in zip(
                ("gate_proj", "up_proj", "down_proj"), (*gate_up.chunk(2, dim=1), down), strict=True
            )
        }
    safetensors.torch.save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        tmp_path / "model.safetensors",
    )
    mapping = CheckpointMapping(
        merges=(MergeRule("gate_up_proj", ("gate_proj", "up_proj")),), stack_experts=True
    )

    def load(rank: int) -> dict[str, torch.Tensor]:
        with torch.device("meta"):
            experts = Experts(
                num_experts,
                2,
                hidden_size,
                intermediate_size,
                renormalize=True,
                site=site(Topology(rank, 2), experts=placement),
            )
        params = experts.params(prefix="mlp.experts")
        return dict(load_weight(str(tmp_path), CPU, mapping=mapping, params=params))

    ranks = [load(rank) for rank in range(2)]
    if isinstance(placement, ExpertParallel):
        loaded_gate_up = torch.cat([rank["mlp.experts.gate_up_proj"] for rank in ranks])
        loaded_down = torch.cat([rank["mlp.experts.down_proj"] for rank in ranks])
    else:
        gates, ups = zip(
            *(rank["mlp.experts.gate_up_proj"].chunk(2, dim=1) for rank in ranks), strict=True
        )
        loaded_gate_up = torch.cat([*gates, *ups], dim=1)
        loaded_down = torch.cat([rank["mlp.experts.down_proj"] for rank in ranks], dim=2)

    assert torch.equal(loaded_gate_up, gate_up) and torch.equal(loaded_down, down)


def _layer(
    moe_backend: BaseMoeBackend,
    group: _Group | None = None,
    placement: ExpertPlacement = _PLACEMENTS[0],
    *,
    settled: bool = True,
) -> MoELayer:
    # Eight experts, two per token, and a shared expert of another width than the routed ones,
    # settled for inference as prepare_inference settles it.
    layer = MoELayer(
        8,
        2,
        16,
        4,
        renormalize=True,
        site=site(group=group, moe_backend=moe_backend, experts=placement),
        shared_expert_intermediate_size=6,
    )
    if settled:
        layer.settle_inference()
    return layer


def _rank_state(state: dict[str, torch.Tensor], layer: MoELayer) -> dict[str, torch.Tensor]:
    """``layer``'s windows of a whole layer's ``state``, as the loader would read them."""

    def select(held: tuple, tensor: torch.Tensor) -> torch.Tensor:
        if not held:
            return tensor
        parts = tensor.chunk(len(held), dim=held[0].axis)
        return torch.cat([w.select(p) for w, p in zip(held, parts, strict=True)], held[0].axis)

    local = {}
    for param in layer.params():
        tensor = state[param.path]
        if param.experts is None:
            local[param.path] = select(param.windows, tensor)
        else:
            held = tensor[param.experts.start : param.experts.stop]
            local[param.path] = torch.stack([select(param.windows, e) for e in held])
    return local


@pytest.fixture
def whole_layer() -> MoELayer:
    torch.manual_seed(0)
    return _randomized(_layer(ReferenceMoe()))


def _ranks(
    whole: MoELayer,
    backend: type[ReferenceMoe] = ReferenceMoe,
    placement: ExpertPlacement = _PLACEMENTS[0],
) -> list[tuple[MoELayer, list[torch.Tensor]]]:
    """Each of two ranks' layer, loaded with its windows of ``whole``, and its reductions."""

    ranks = []
    for rank in range(2):
        group = _Group(rank, 2)
        layer = _layer(backend(), group, placement)
        layer.load_state_dict(_rank_state(whole.state_dict(), layer))
        ranks.append((layer, group.reduced))
    return ranks


@pytest.mark.parametrize(
    ("placement", "gate_up", "down"),
    [(_PLACEMENTS[0], (4, 8, 16), (4, 16, 4)), (_PLACEMENTS[1], (8, 4, 16), (8, 16, 2))],
    ids=_IDS,
)
def test_each_rank_holds_its_expert_share_and_a_window_of_the_shared_expert(
    whole_layer: MoELayer, placement: ExpertPlacement, gate_up: tuple, down: tuple
) -> None:
    for rank, (layer, _) in enumerate(_ranks(whole_layer, placement=placement)):
        experts, shared = layer.experts, layer.shared_expert
        assert experts.placement == placement.shard(8, 4, Topology(rank, 2))
        assert experts.gate_up_proj.shape == gate_up and experts.down_proj.shape == down
        assert (experts.expert_map is None) == experts.placement.holds_every_expert
        assert shared.gate_up_proj.weight.shape == (6, 16)
        assert shared.down_proj.weight.shape == (16, 3)
        assert torch.equal(layer.gate.weight, whole_layer.gate.weight)
        whole_gate, whole_up = whole_layer.shared_expert.gate_up_proj.weight.chunk(2)
        rows = slice(3 * rank, 3 * rank + 3)
        assert torch.equal(
            shared.gate_up_proj.weight, torch.cat([whole_gate[rows], whole_up[rows]])
        )
        assert torch.equal(
            shared.down_proj.weight, whole_layer.shared_expert.down_proj.weight[:, rows]
        )


@pytest.mark.parametrize("placement", _PLACEMENTS, ids=_IDS)
@pytest.mark.parametrize("backend", [ReferenceMoe, BlockReferenceMoe], ids=["composed", "block"])
def test_rank_partials_meet_in_one_reduction_that_gives_the_whole_layer(
    whole_layer: MoELayer, backend: type[ReferenceMoe], placement: ExpertPlacement
) -> None:
    x = torch.randn(5, 16)
    expected = whole_layer.forward(x.clone())
    partials = []
    for layer, reductions in _ranks(whole_layer, backend, placement):
        output = layer.forward(x.clone())
        (reduced,) = reductions
        assert torch.equal(output, reduced)
        partials.append(reduced)

    torch.testing.assert_close(partials[0] + partials[1], expected, **_FP32_REORDERED)
    # A dropped partial misses by far more than reordering: the tolerance still catches it.
    assert all(float((p - expected).abs().max()) > 1e3 * _FP32_REORDERED["atol"] for p in partials)


def test_a_row_routed_only_to_another_ranks_experts_adds_only_its_shared_part(
    whole_layer: MoELayer,
) -> None:
    # Positive inputs and a router that scores rank 0's experts far below rank 1's.
    gate = whole_layer.gate.weight
    gate[:4] = -gate[:4].abs() - 1
    gate[4:] = gate[4:].abs() + 1
    x = torch.rand(3, 16) + 0.5
    expected = whole_layer.forward(x.clone())
    (first, first_reduced), (second, second_reduced) = _ranks(whole_layer)

    routed = first.experts.forward(x.clone(), first.gate.weight)
    assert torch.equal(routed, torch.zeros_like(x))
    first.forward(x.clone())
    second.forward(x.clone())

    shared = first.shared_expert.forward(x) * torch.sigmoid(first.shared_expert_gate.forward(x))
    torch.testing.assert_close(first_reduced[0], shared, rtol=0, atol=0)
    total = first_reduced[0] + second_reduced[0]
    torch.testing.assert_close(total, expected, **_FP32_REORDERED)


def test_inference_runs_the_backends_whole_block_on_the_layers_weights(
    whole_layer: MoELayer,
) -> None:
    backend = BlockReferenceMoe()
    layer = _layer(backend)
    layer.load_state_dict(whole_layer.state_dict())
    x = torch.randn(5, 16)

    torch.testing.assert_close(layer.forward(x), whole_layer.forward(x.clone()))
    assert backend.requests == [(2, True, torch.float32)] and backend.calls == 1
    # A layer without a shared expert never asks for the op: it composes its block.
    plain = _randomized(MoELayer(4, 2, 8, 4, renormalize=True, site=site(moe_backend=backend)))
    plain.forward(torch.randn(3, 8))
    assert len(backend.requests) == 1 and backend.calls == 1


@pytest.mark.parametrize("placement", _PLACEMENTS, ids=_IDS)
def test_an_empty_or_padded_batch_reduces_on_every_rank(
    whole_layer: MoELayer, placement: ExpertPlacement
) -> None:
    x = torch.randn(3, 16)
    padded = torch.cat([x, torch.zeros(5, 16)])
    for layer, reduced in _ranks(whole_layer, placement=placement):
        assert layer.forward(torch.empty(0, 16)).shape == (0, 16)
        output = layer.forward(padded.clone())
        assert len(reduced) == 2 and reduced[0].shape == (0, 16)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output[:3], layer.forward(x.clone()), **_FP32_REORDERED)


def test_the_fused_backend_offers_the_kernels_block_for_bf16_top8_renormalized_routes() -> None:
    backend = FusedMoe()

    block = backend.sparse_block(8, True, torch.bfloat16)
    assert isinstance(block, MoEBlock)
    # Every layer shares one op, so its persistent workspaces exist once.
    assert backend.sparse_block(8, True, torch.bfloat16) is block
    for request in (
        (4, True, torch.bfloat16),
        (8, False, torch.bfloat16),
        (8, True, torch.float16),
    ):
        assert backend.sparse_block(*request) is None


def test_the_zero_weight_remap_runs_a_remote_route_on_local_expert_zero_at_weight_zero() -> None:
    expert_map = Placement(8, 4, range(4, 8), range(4)).expert_map()
    weights = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
    ids = torch.tensor([[5, 2], [0, 7]], dtype=torch.int32)

    local_weights, local_ids = local_routes(weights, ids, expert_map)

    assert local_weights.tolist() == [[0.75, 0.0], [0.0, 0.5]]
    assert local_ids.dtype == torch.int32 and local_ids.tolist() == [[1, 0], [0, 3]]


@pytest.mark.parametrize("mapped", [False, True], ids=["whole", "expert-map"])
@pytest.mark.parametrize(
    ("dtype", "path"), [(torch.bfloat16, "fused_experts"), (torch.float16, "compose")]
)
def test_the_fused_backend_routes_then_runs_the_kernel_or_composes_a_dtype_it_does_not_take(
    monkeypatch: pytest.MonkeyPatch, mapped: bool, dtype: torch.dtype, path: str
) -> None:
    weights, ids = torch.rand(3, 2), torch.zeros(3, 2, dtype=torch.int32)
    routes, calls = [], []

    def route(x, router, top_k, *, renormalize):
        routes.append((x, router, top_k, renormalize))
        return weights, ids

    def fused_experts(x, w1, w2, topk_weights, topk_ids, *, expert_map, out):
        calls.append(("fused_experts", topk_weights, topk_ids, expert_map))
        assert out is x
        return x

    def compose(x, w1, w2, topk_weights, topk_ids, expert_map):
        calls.append(("compose", topk_weights, topk_ids, expert_map))
        return x

    monkeypatch.setattr(fused_module, "route", route)
    monkeypatch.setattr(fused_module, "fused_experts", fused_experts)
    monkeypatch.setattr(fused_module, "compose", compose)
    expert_map = Placement(8, 4, range(4, 8), range(4)).expert_map() if mapped else None
    x, router = torch.randn(3, 16, dtype=dtype), torch.randn(8, 16, dtype=dtype)

    assert FusedMoe().forward(x, x, x, router, 2, True, expert_map=expert_map) is x
    assert routes == [(x, router, 2, True)]
    assert calls == [(path, weights, ids, expert_map)]


def test_the_expert_map_is_built_on_the_sites_device() -> None:
    meta = torch.device("meta")
    layer = MoELayer(8, 2, 16, 4, renormalize=True, site=site(Topology(1, 2), device=meta))

    assert layer.experts.expert_map.device == meta


def test_an_unsettled_block_refuses_inference_until_prepare_inference_settles_it() -> None:
    backend = BlockReferenceMoe()
    layer = _layer(backend, settled=False)
    layer.load_state_dict({name: torch.randn_like(t) for name, t in layer.state_dict().items()})
    x = torch.randn(3, 16)

    with pytest.raises(RuntimeError, match="prepare_inference settles the MoE block"):
        layer.forward(x.clone())
    prepare_inference([layer], max_rows=2, max_sequences=1, dtype=torch.float32, device=CPU)
    warmed = backend.calls
    layer.forward(x.clone())
    assert warmed == 1 and backend.calls == 2


def test_the_warm_up_runs_each_whole_block_op_once_at_the_most_rows() -> None:
    torch.manual_seed(0)
    backend = BlockReferenceMoe()
    layer = _randomized(_layer(backend))

    warmed: set = set()
    for _ in range(2):
        layer.warm_up_rows(4, dtype=torch.float32, device=torch.device("cpu"), warmed=warmed)

    assert backend.calls == 1
