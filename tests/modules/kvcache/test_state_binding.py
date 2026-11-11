"""Grouping state layers into KV groups and slot rows, and binding each layer to its slots."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from cti.attention.paged import KVPair
from cti.forward import InlineBreaks
from cti.kvcache import (
    GroupLayout,
    GroupRole,
    PagedSlot,
    RecurrentStatePool,
    ResolvedSlot,
    Spec,
    StateSlot,
    group_layers,
)
from cti.layers import (
    AttentionLayer,
    LayerServices,
    LinearQKVMerged,
    PrologueFactory,
    RMSNorm,
    RotaryEmbedding,
)
from cti.layers.gated_delta_net.layer import gated_delta_net_state
from tests.support import reference_ops
from tests.support.tp_shards import site

_PAIR = KVPair(1, 4, torch.bfloat16)
_ROW = gated_delta_net_state(1, 1, 1, 1, conv_history=1)
_TARGET, _DRAFT = GroupRole.TARGET, GroupRole.DRAFT


class _Layer:
    """A state layer of the given slots that records what it is bound to."""

    def __init__(self, *slots: StateSlot) -> None:
        self._slots = slots
        self.bound: dict[str, object] | None = None

    def state_slots(self) -> tuple[StateSlot, ...]:
        return self._slots

    def bind_state(self, bound):
        self.bound = dict(bound)


def _attention(requirement: str = "kv") -> _Layer:
    return _Layer(StateSlot("kv", requirement=requirement))


def _recurrent() -> _Layer:
    return _Layer(StateSlot("recurrent", state=_ROW))


def _slots(**specs: Spec) -> dict:
    return {(role, "kv"): ResolvedSlot("mha", spec) for role, spec in specs.items()}


def test_a_hybrid_model_makes_one_kv_group_and_numbers_each_kind_in_order() -> None:
    # Qwen3.5's interleave: three recurrent layers, then one attention layer, twice.
    layers = [factory() for factory in (_recurrent, _recurrent, _recurrent, _attention) * 2]

    layout = group_layers([(_TARGET, layer) for layer in layers], _slots(target=Spec(_PAIR)))

    assert layout.groups == (GroupLayout(_TARGET, 0, Spec(_PAIR), 2),)
    assert layout.rows.specs == (_ROW,) * 6
    places = [(p.layer, p.slot, p.group, p.index) for p in layout.slots]
    assert places == [
        (layers[0], "recurrent", None, 0),
        (layers[1], "recurrent", None, 1),
        (layers[2], "recurrent", None, 2),
        (layers[3], "kv", 0, 0),
        (layers[4], "recurrent", None, 3),
        (layers[5], "recurrent", None, 4),
        (layers[6], "recurrent", None, 5),
        (layers[7], "kv", 0, 1),
    ]


def test_target_and_draft_slots_of_one_name_are_groups_0_and_1() -> None:
    target, draft = [_attention() for _ in range(3)], [_attention() for _ in range(2)]
    slots = _slots(target=Spec(_PAIR), draft=Spec(_PAIR))

    layout = group_layers(
        [(_TARGET, layer) for layer in target] + [(_DRAFT, layer) for layer in draft], slots
    )

    assert layout.groups == (
        GroupLayout(_TARGET, 0, Spec(_PAIR), 3),
        GroupLayout(_DRAFT, 1, Spec(_PAIR), 2),
    )
    assert [(p.group, p.index) for p in layout.slots] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
    ]


@pytest.mark.parametrize("other", [Spec(_PAIR, window=64), Spec(_PAIR, tokens_per_entry=2)])
def test_a_spec_differing_only_in_window_or_entry_tokens_is_another_group(other: Spec) -> None:
    layers = [_attention("full"), _attention("other"), _attention("full")]
    slots = {
        (_TARGET, "full"): ResolvedSlot("mha", Spec(_PAIR)),
        (_TARGET, "other"): ResolvedSlot("mha", other),
    }

    layout = group_layers([(_TARGET, layer) for layer in layers], slots)

    assert layout.groups == (
        GroupLayout(_TARGET, 0, Spec(_PAIR), 2),
        GroupLayout(_TARGET, 1, other, 1),
    )
    assert [(p.group, p.index) for p in layout.slots] == [(0, 0), (1, 0), (0, 1)]


def test_bind_hands_each_layer_its_groups_backend_and_its_row_views() -> None:
    target, draft, recurrent = _attention(), _attention(), _recurrent()
    layout = group_layers(
        [(_TARGET, target), (_TARGET, recurrent), (_DRAFT, draft)],
        _slots(target=Spec(_PAIR), draft=Spec(_PAIR)),
    )
    pool = RecurrentStatePool(layout.rows.specs, num_slots=2, device=torch.device("cpu"))
    backends = (object(), object())

    layout.bind(backends, pool)

    assert target.bound == {"kv": PagedSlot(backends[0], 0)}
    assert draft.bound == {"kv": PagedSlot(backends[1], 0)}
    (views,) = recurrent.bound.values()
    assert list(views) == ["state", "conv"]
    assert all(
        view is pooled for view, pooled in zip(views.values(), pool.layer_views(0), strict=True)
    )


@pytest.fixture
def _references(monkeypatch: pytest.MonkeyPatch) -> None:
    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)
    reference_ops.bind_norm_references(monkeypatch)
    monkeypatch.setattr("ctikernels.attention.inference.Prologue", reference_ops.Prologue)


def _attention_layer(prologue: PrologueFactory | None = None) -> AttentionLayer:
    """A Qwen3-like layer of two 128-wide query heads over one KV head."""
    layer_site = site(services=LayerServices())
    norms = [RMSNorm(128, 1e-6) for _ in range(2)]
    for norm in norms:
        norm.weight = torch.ones(128, dtype=torch.bfloat16)
    return AttentionLayer(
        qkv=LinearQKVMerged(
            hidden_size=8,
            head_dim=128,
            num_qo_heads=2,
            num_kv_heads=1,
            has_bias=False,
            site=layer_site,
        ),
        rotary=RotaryEmbedding.create(128, 128, 64, 1e6, site=layer_site),
        q_norm=norms[0],
        k_norm=norms[1],
        prologue=prologue,
    )


@pytest.mark.usefixtures("_references")
def test_an_unbound_attention_layer_says_so() -> None:
    batch = SimpleNamespace(graph_breaks=InlineBreaks(), rope_positions=torch.arange(1))

    with pytest.raises(RuntimeError, match="not bound to its KV group"):
        _attention_layer().forward(torch.zeros(1, 4 * 128, dtype=torch.bfloat16), batch)


@pytest.mark.usefixtures("_references")
def test_a_required_prologue_refuses_a_backend_without_kv_prologues_at_binding() -> None:
    layer = _attention_layer(PrologueFactory(required=True))
    backend = SimpleNamespace(SUPPORTS_KV_PROLOGUE=False)

    with pytest.raises(ValueError, match="does not run the required KV prologue"):
        layer.bind_state({"kv": PagedSlot(backend, 0)})
