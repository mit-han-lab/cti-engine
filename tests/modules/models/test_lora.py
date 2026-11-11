from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
import torch
import torch.nn.functional as F

from cti.config import ExpertLoRALayout, LoRAConfig
from cti.layers import Adapter, BaseOP, Placement
from cti.layers.linear import (
    LinearBase,
    LinearColParallelMerged,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    ProjectionSlice,
)
from cti.layers.lora import ExpertLoRAFactors, LoRAProjection
from tests.support.expert_lora import factor_windows
from tests.support.lora_serving import lora_delta
from tests.support.tp_shards import CPU, site

KEY = "projection"


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _assert_projection_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Merged projections accumulate each semantic slice with addmm_, while the
    # stock reference concatenates the slices before adding them to the base
    # projection. CPU BLAS implementations may round those equivalent FP32
    # expressions differently, so bitwise equality is not portable to Arm64.
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def _assert_lora_input_gradient_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    # The custom backward accumulates one contribution per output slice into
    # dX. Its addition order differs from stock autograd for merged projections.
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


class _Model(BaseOP):
    """A model whose one LoRA host sits at ``KEY``."""

    def __init__(self, projection: LinearBase) -> None:
        self.projection = projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection.forward(x)


class _FakeCommunicator:
    rank, size = 0, 2

    def __init__(self, peer_output: torch.Tensor) -> None:
        self.peer_output = peer_output
        self.inputs: list[torch.Tensor] = []

    def all_reduce(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value.detach().clone())
        return value + self.peer_output


def _attach(
    base: LinearBase, targets: tuple[str, ...], *, rank: int = 2, dtype=torch.float32
) -> Adapter:
    adapters = Adapter(LoRAConfig(targets=targets, rank=rank, alpha=2.0 * rank))
    adapters.attach(_Model(base))
    adapters.materialize(device=CPU, dtype=dtype)
    return adapters


def _projection_case(kind: str) -> tuple[LinearBase, Adapter, dict[str, str]]:
    """A LoRA host of ``kind`` at ``KEY`` with its adapter and each output slice's A name."""

    at = site()
    shared_a: dict[str, tuple[str, ...]] = {}
    if kind == "qkv":
        base = LinearQKVMerged(4, 2, 2, 1, has_bias=False, site=at, lora="attention")
    elif kind == "gate_up":
        base = LinearColParallelMerged(
            4, [6, 6], has_bias=False, output_names=("gate", "up"), site=at, lora="mlp"
        )
    elif kind == "single":
        base = LinearReplicated(4, 6, has_bias=False, site=at, lora="attention")
    elif kind == "qkv_gate":
        # Like Qwen3.5's q_proj, one checkpoint module holds the query and the gate.
        shared_a = {"q_gate": ("q", "gate")}
        base = LinearQKVMerged(
            4,
            2,
            2,
            1,
            has_bias=False,
            with_output_gate=True,
            site=at,
            lora="attention",
            lora_shared_a=shared_a,
        )
    elif kind == "in_proj":
        # Like Qwen3.5's in_proj_qkv, one checkpoint module holds q, k and v.
        shared_a = {"qkv": ("q", "k", "v")}
        base = LinearColParallelMerged(
            4,
            [2, 2, 3, 3, 1, 1],
            has_bias=False,
            output_names=("q", "k", "v", "z", "b", "a"),
            site=at,
            lora="linear_attention",
            lora_shared_a=shared_a,
        )
    else:
        raise ValueError(f"Unsupported projection test case: {kind}")

    torch.manual_seed(7)
    base.weight.normal_()
    adapters = _attach(base, (base.lora_target,))
    return base, adapters, _a_names(base, shared_a)


def _a_names(base: LinearBase, shared_a: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Each output slice's A name, in output order."""

    shared = {name: a_name for a_name, names in shared_a.items() for name in names}
    return {s.name: shared.get(s.name, s.name) for s in base.output_slices}


def _reference_output(
    base: LinearBase,
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    a_names: Mapping[str, str],
) -> torch.Tensor:
    base_output = F.linear(x, base.weight, base.bias)
    delta = torch.cat(
        [
            F.linear(
                F.linear(x, state[f"{KEY}.{a_name}.lora_A"]),
                state[f"{KEY}.{name}.lora_B"],
            )
            * 2.0
            for name, a_name in a_names.items()
        ],
        dim=-1,
    )
    return base_output + delta


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"targets": ()}, "must not be empty"),
        ({"targets": ("attention", "attention")}, "must be unique"),
        ({"targets": ("unsupported",)}, "Unsupported"),
        ({"targets": ("attention",), "rank": 0}, r"rank must be in \[1, 128\]"),
        ({"targets": ("attention",), "alpha": math.nan}, "finite and positive"),
    ],
)
def test_lora_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LoRAConfig(**kwargs)


def test_lora_config_canonicalizes_targets_to_immutable_storage() -> None:
    targets = ["attention"]
    config = LoRAConfig(targets=targets)

    targets.append("mlp")

    assert config.targets == ("attention",)
    assert isinstance(config.targets, tuple)


def test_inactive_lora_target_attaches_nothing() -> None:
    base = LinearQKVMerged(4, 2, 2, 1, has_bias=False, site=site(), lora="attention")
    adapters = Adapter(LoRAConfig(targets=("mlp",)))
    adapters.attach(_Model(base))

    assert len(adapters.modules()) == 0
    with pytest.raises(RuntimeError, match="has no projections"):
        adapters.materialize(device=CPU, dtype=torch.float32)


def test_an_adapter_attaches_once_and_a_host_takes_one_adapter() -> None:
    base = LinearReplicated(4, 6, has_bias=False, site=site(), lora="attention")
    adapters = Adapter(LoRAConfig(targets=("attention",)))
    adapters.attach(_Model(base))

    with pytest.raises(RuntimeError, match="already attached"):
        adapters.attach(_Model(base))
    with pytest.raises(RuntimeError, match="already attached"):
        Adapter(LoRAConfig(targets=("attention",))).attach(_Model(base))


def test_lora_materializes_at_model_execution_dtype() -> None:
    base = LinearReplicated(4, 6, has_bias=False, site=site(), lora="attention")
    adapters = _attach(base, ("attention",), dtype=torch.bfloat16)

    state = adapters.state_dict()
    assert len(adapters.modules()) == 1
    assert state
    assert all(tensor.dtype == torch.bfloat16 for tensor in state.values())
    assert all(
        bool(torch.count_nonzero(tensor) > 0)
        for name, tensor in state.items()
        if name.endswith(".lora_A")
    )
    assert all(
        bool(torch.count_nonzero(tensor) == 0)
        for name, tensor in state.items()
        if name.endswith(".lora_B")
    )


def test_fused_projection_uses_independent_semantic_lora_slices() -> None:
    base, adapters, _ = _projection_case("qkv")
    x = torch.randn(3, 4)

    assert tuple(
        (output_slice.name, output_slice.offset, output_slice.size)
        for output_slice in base.output_slices
    ) == (
        ("q", 0, 4),
        ("k", 4, 2),
        ("v", 6, 2),
    )

    base_output = F.linear(x, base.weight)
    _assert_projection_close(base.forward(x), base_output)
    assert set(base.state_dict(prefix=KEY)) == {f"{KEY}.weight"}

    state = {name: torch.randn_like(tensor) for name, tensor in adapters.state_dict().items()}
    adapters.load_state_dict(state)

    expected_delta = torch.cat(
        [
            F.linear(
                F.linear(x, state[f"{KEY}.{name}.lora_A"]),
                state[f"{KEY}.{name}.lora_B"],
            )
            * 2.0
            for name in ("q", "k", "v")
        ],
        dim=-1,
    )
    _assert_projection_close(base.forward(x), base_output + expected_delta)


def test_gate_up_projection_declares_two_semantic_slices() -> None:
    projection = LinearColParallelMerged(
        4, [6, 6], has_bias=False, output_names=("gate", "up"), site=site()
    )

    assert tuple(
        (output_slice.name, output_slice.offset, output_slice.size)
        for output_slice in projection.output_slices
    ) == (
        ("gate", 0, 6),
        ("up", 6, 6),
    )


@pytest.mark.parametrize("use_lora", (False, True), ids=("plain", "lora"))
def test_row_sharded_projection_reduces_before_adding_bias(use_lora: bool) -> None:
    peer_output = torch.tensor(
        [[0.3, -0.2, 0.1], [-0.4, 0.5, -0.6]],
        dtype=torch.float32,
    )

    communicator = _FakeCommunicator(peer_output)
    linear = LinearRowParallel(4, 3, has_bias=True, site=site(group=communicator))
    weight = torch.tensor(
        [[0.2, -0.1], [0.4, 0.3], [-0.5, 0.6]],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.7, -0.8, 0.9], dtype=torch.float32)
    a = torch.tensor([[0.1, -0.3]], dtype=torch.float32)
    b = torch.tensor([[0.5], [-0.2], [0.4]], dtype=torch.float32)
    x = torch.tensor([[1.0, 2.0], [-1.5, 0.5]], dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(weight)
        assert linear.bias is not None
        linear.bias.copy_(bias)

    if use_lora:
        lora = linear.attach_lora(LoRAConfig(targets=("attention",), rank=1, alpha=0.5))
        lora.materialize(device=CPU, dtype=torch.float32, generator=torch.Generator())
        with torch.no_grad():
            for factor, value in zip(lora.factors, (a, b), strict=True):
                factor.copy_(value)
        actual = linear.forward(x)
        local_output = F.linear(x, weight) + 0.5 * F.linear(F.linear(x, a), b)
    else:
        actual = linear.forward(x)
        local_output = F.linear(x, weight)

    assert len(communicator.inputs) == 1
    _assert_exact(communicator.inputs[0], local_output)
    _assert_exact(actual, local_output + peer_output + bias)


@pytest.mark.parametrize("projection_kind", ["qkv", "gate_up", "single"])
def test_lora_forward_gradients_and_adamw_match_reference(
    projection_kind: str,
) -> None:
    base, adapters, a_names = _projection_case(projection_kind)
    loaded = {name: torch.randn_like(tensor) for name, tensor in adapters.state_dict().items()}
    adapters.load_state_dict(loaded)

    parameters = tuple(t for _, t in adapters.named_parameters())
    reference_state = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in adapters.state_dict().items()
    }
    reference_parameters = tuple(reference_state.values())
    pointers = tuple(parameter.data_ptr() for parameter in parameters)
    identities = tuple(id(parameter) for parameter in parameters)
    base_before = base.weight.clone()
    x = torch.randn(5, 4, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()

    output = base.forward(x)
    reference = _reference_output(base, reference_x, reference_state, a_names)
    _assert_projection_close(output, reference)
    upstream_gradient = torch.randn_like(output)
    output.backward(upstream_gradient)
    reference.backward(upstream_gradient.clone())

    assert x.grad is not None
    assert reference_x.grad is not None
    _assert_lora_input_gradient_close(x.grad, reference_x.grad)
    for parameter, reference_parameter in zip(
        parameters,
        reference_parameters,
        strict=True,
    ):
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None
        assert reference_parameter.grad is not None
        _assert_exact(parameter.grad, reference_parameter.grad)

    optimizer = torch.optim.AdamW(
        parameters,
        lr=0.05,
        weight_decay=0.0,
        foreach=False,
    )
    reference_optimizer = torch.optim.AdamW(
        reference_parameters,
        lr=0.05,
        weight_decay=0.0,
        foreach=False,
    )
    optimizer.step()
    reference_optimizer.step()

    assert tuple(parameter.data_ptr() for parameter in parameters) == pointers
    assert tuple(id(parameter) for parameter in parameters) == identities
    assert torch.equal(base.weight, base_before)
    for parameter, reference_parameter in zip(
        parameters,
        reference_parameters,
        strict=True,
    ):
        _assert_exact(parameter, reference_parameter)
    _assert_projection_close(
        base.forward(x.detach()),
        _reference_output(base, x.detach(), reference_state, a_names),
    )


# q | k | v | gate of two 2-wide query heads and one KV head.
_GATED_QKV_SLICES = (
    ProjectionSlice(name="q", offset=0, size=4),
    ProjectionSlice(name="k", offset=4, size=2),
    ProjectionSlice(name="v", offset=6, size=2),
    ProjectionSlice(name="gate", offset=8, size=4),
)


@pytest.mark.parametrize(
    ("kind", "expected_names"),
    [
        (
            "qkv_gate",
            ("q_gate.lora_A", "q.lora_B", "gate.lora_B")
            + ("k.lora_A", "k.lora_B", "v.lora_A", "v.lora_B"),
        ),
        (
            "in_proj",
            ("qkv.lora_A", "q.lora_B", "k.lora_B", "v.lora_B")
            + ("z.lora_A", "z.lora_B", "b.lora_A", "b.lora_B", "a.lora_A", "a.lora_B"),
        ),
    ],
)
def test_slices_of_one_checkpoint_module_share_its_a(
    kind: str,
    expected_names: tuple[str, ...],
) -> None:
    base, adapters, a_names = _projection_case(kind)

    state = adapters.state_dict()

    # Each checkpoint module's A comes first, then the B of each slice it spans.
    assert tuple(state) == tuple(f"{KEY}.{name}" for name in expected_names)
    for output_slice in base.output_slices:
        assert state[f"{KEY}.{a_names[output_slice.name]}.lora_A"].shape == (2, 4)
        assert state[f"{KEY}.{output_slice.name}.lora_B"].shape == (output_slice.size, 2)


def test_shared_a_reaches_the_projection_once_per_slice() -> None:
    projection = LoRAProjection(
        input_size=4,
        output_slices=_GATED_QKV_SLICES,
        rank=2,
        scaling=1.0,
        owns_whole=True,
        shared_a={"q_gate": ("q", "gate")},
    )
    projection.materialize(device=CPU, dtype=torch.float32, generator=torch.Generator())

    a_q_gate, b_q, b_gate, a_k, b_k, a_v, b_v = projection.parameters()

    # The linear owner reads an A/B pair per output slice, in output order.
    expected = (a_q_gate, b_q, a_k, b_k, a_v, b_v, a_q_gate, b_gate)
    assert len(projection.factors) == len(expected)
    assert all(
        factor is tensor for factor, tensor in zip(projection.factors, expected, strict=True)
    )
    assert len({id(parameter) for parameter in projection.parameters()}) == 7


def test_the_in_place_delta_adds_one_addmm_per_slice_bitwise() -> None:
    projection = LoRAProjection(
        input_size=4,
        output_slices=_GATED_QKV_SLICES,
        rank=2,
        scaling=0.5,
        owns_whole=True,
        shared_a={"q_gate": ("q", "gate")},
    )
    projection.materialize(device=CPU, dtype=torch.float32, generator=torch.Generator())
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for factor in projection.factors[1::2]:
            factor.copy_(torch.randn(factor.shape, generator=generator))
    width = sum(output_slice.size for output_slice in _GATED_QKV_SLICES)
    x = torch.randn(2, 3, 4, generator=generator)
    y = torch.randn(2, 3, width, generator=generator)
    expected = lora_delta(projection, y, x)

    with torch.no_grad():
        actual = projection.add_delta_(y, x)

    # Each slice's addmm accumulates as an out-of-place one would, into y's own storage.
    _assert_exact(actual, expected)
    assert actual.data_ptr() == y.data_ptr()


def test_unshared_slices_draw_their_a_in_slice_order() -> None:
    # One Kaiming-uniform A per slice, in slice order, from the projection's generator.
    generator = torch.Generator().manual_seed(11)
    expected = []
    for _ in range(3):
        a = torch.empty(2, 4)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        expected.append(a)
    projection = LoRAProjection(
        input_size=4,
        output_slices=_GATED_QKV_SLICES[:3],
        rank=2,
        scaling=1.0,
        owns_whole=True,
    )

    projection.materialize(
        device=CPU, dtype=torch.float32, generator=torch.Generator().manual_seed(11)
    )

    for actual, reference in zip(projection.factors[0::2], expected, strict=True):
        _assert_exact(actual.detach(), reference)


@pytest.mark.parametrize("kind", ["qkv_gate", "in_proj"])
def test_shared_a_forward_and_gradients_match_one_a_per_checkpoint_module(
    kind: str,
) -> None:
    base, adapters, a_names = _projection_case(kind)
    adapters.load_state_dict(
        {name: torch.randn_like(tensor) for name, tensor in adapters.state_dict().items()}
    )
    reference_state = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in adapters.state_dict().items()
    }
    x = torch.randn(5, 4, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()

    output = base.forward(x)
    # Each checkpoint module projects x through its one A, then each of its slices'
    # B expands that low-rank input.
    reference = _reference_output(base, reference_x, reference_state, a_names)
    _assert_projection_close(output, reference)
    upstream_gradient = torch.randn_like(output)
    output.backward(upstream_gradient)
    reference.backward(upstream_gradient.clone())

    assert x.grad is not None and reference_x.grad is not None
    _assert_lora_input_gradient_close(x.grad, reference_x.grad)
    for (name, parameter), reference_parameter in zip(
        adapters.named_parameters(),
        reference_state.values(),
        strict=True,
    ):
        assert parameter.grad is not None, name
        assert reference_parameter.grad is not None, name
        # A shared A sums its slices' gradients, in another order than stock autograd.
        _assert_lora_input_gradient_close(parameter.grad, reference_parameter.grad)
    assert not base.weight.requires_grad
    assert base.weight.grad is None


@pytest.mark.parametrize(
    ("shared_a", "message"),
    [
        ({"q_x": ("q", "x")}, "names unknown output slices"),
        ({"q_only": ("q",)}, "a tuple of at least two output slices"),
        # A bare string would iterate as the known slices q, k and v.
        ({"qkv": "qkv"}, "a tuple of at least two output slices"),
        ({"q_q": ("q", "q")}, "'q' is shared more than once"),
        ({"q_gate": ("q", "gate"), "q_k": ("q", "k")}, "'q' is shared more than once"),
        # Named like a slice, the A would also collect that slice: here k.
        ({"k": ("q", "gate")}, "must be non-empty and differ"),
    ],
)
def test_shared_a_must_name_disjoint_groups_of_known_slices(
    shared_a: dict[str, tuple[str, ...]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LoRAProjection(
            input_size=4,
            output_slices=_GATED_QKV_SLICES,
            rank=2,
            scaling=1.0,
            owns_whole=True,
            shared_a=shared_a,
        )


def test_lora_state_dict_load_is_atomic_strict_and_preserves_storage() -> None:
    _, adapters, _ = _projection_case("qkv")
    state = adapters.state_dict()
    pointers = {name: tensor.data_ptr() for name, tensor in state.items()}
    before = {name: tensor.clone() for name, tensor in state.items()}

    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing="):
        adapters.load_state_dict(missing)

    invalid = {
        name: torch.full_like(tensor, index + 1)
        for index, (name, tensor) in enumerate(state.items())
    }
    last_name = next(reversed(invalid))
    invalid[last_name] = torch.empty(1, dtype=state[last_name].dtype)
    with pytest.raises(ValueError, match="has shape"):
        adapters.load_state_dict(invalid)
    for name, tensor in adapters.state_dict().items():
        _assert_exact(tensor, before[name])

    replacement = {name: torch.randn_like(tensor) for name, tensor in state.items()}
    adapters.load_state_dict(replacement)
    for name, tensor in adapters.state_dict().items():
        _assert_exact(tensor, replacement[name])
    assert {name: tensor.data_ptr() for name, tensor in adapters.state_dict().items()} == pointers


@pytest.mark.parametrize("rank", [1, 4, 8, 12, 128])
def test_per_expert_lora_takes_multiples_of_eight_up_to_128(rank: int) -> None:
    # Other targets take any rank up to 128.
    assert LoRAConfig(targets=("attention", "linear_attention", "mlp"), rank=rank).rank == rank
    if rank % 8 == 0:
        assert LoRAConfig(targets=("attention", "experts"), rank=rank).rank == rank
        return
    with pytest.raises(
        ValueError, match=f"per-expert LoRA takes ranks 8 to 128 in steps of 8, not {rank}$"
    ):
        LoRAConfig(targets=("attention", "experts"), rank=rank)


@pytest.mark.parametrize("layout", list(ExpertLoRALayout))
@pytest.mark.parametrize(
    ("experts", "intermediate"), [(range(4, 8), range(6)), (range(8), range(3, 6))]
)
def test_placed_expert_factors_start_as_the_whole_adapters_window(
    experts: range, intermediate: range, layout: ExpertLoRALayout
) -> None:
    def named(held: range, window: range) -> dict[str, torch.Tensor]:
        factors = ExpertLoRAFactors(
            Placement(8, 6, held, window), 16, rank=8, scaling=0.5, owns_whole=True, layout=layout
        )
        generator = torch.Generator().manual_seed(3)
        factors.materialize(device=CPU, dtype=torch.float32, generator=generator)
        return dict(factors.named_parameters(""))

    whole, placed = named(range(8), range(6)), named(experts, intermediate)
    windows = factor_windows(Placement(8, 6, experts, intermediate))
    assert placed.keys() == whole.keys()
    for name, tensor in whole.items():
        window = windows[name.removeprefix("experts.")]
        assert placed[name].requires_grad and placed[name].is_leaf, name
        assert torch.equal(placed[name], tensor[window]), name
