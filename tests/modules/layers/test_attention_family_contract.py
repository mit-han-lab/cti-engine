"""Every resolved family meets the ``ResolvedFamily`` contract, and a second family's
multi-slot layers resolve, group, build over all of the target's groups and bind."""

from __future__ import annotations

import inspect
from types import MappingProxyType

import pytest
import torch

import cti.attention.family as family_module
from cti.attention import (
    AttentionRequest,
    AttentionRequirement,
    ResolvedFamily,
    resolve_attention_config,
)
from cti.attention.mha import AttentionBackend, ResolvedMha
from cti.kvcache import (
    GroupRole,
    InferenceState,
    PagedSlot,
    PageTable,
    ResolvedSlot,
    SlotStateSpec,
    Spec,
    StateCapacity,
    StateSlot,
    choose_page_size,
    group_layers,
)
from tests.support.attention_family import FakeBackend, ResolvedFake

_TARGET = GroupRole.TARGET
# A sliding window of tokens and pools of 4- and 128-token entries, in rows of their bytes.
_SLOTS = (("window", 1, 8), ("c4", 4, 16), ("c128", 128, 32))
_REQUIREMENTS = tuple(
    AttentionRequirement(
        family="fake", slot=name, tokens_per_entry=tokens, geometry=(("row_bytes", row),)
    )
    for name, tokens, row in _SLOTS
)
_CARRY = SlotStateSpec((("carry", (2,), torch.float32),), export_granularity=4)


@pytest.mark.parametrize("resolved", [ResolvedMha, ResolvedFake])
def test_a_resolved_family_meets_the_contract(resolved: type) -> None:
    for name in ("build", "workspace_bytes"):
        ours = inspect.signature(getattr(resolved, name)).parameters
        contract = inspect.signature(getattr(ResolvedFamily, name)).parameters
        assert list(ours.values()) == list(contract.values()), name
    fields = set(getattr(resolved, "__dataclass_fields__", ()))
    for name in ("selected", "reason", "pages", "layouts"):
        assert name in fields or isinstance(getattr(resolved, name, None), property), name


def test_mha_builds_over_one_target_group_only() -> None:
    fi = AttentionBackend("fi")
    resolved = ResolvedMha(
        ragged_backend=fi,
        decode_backend=fi,
        reason="test plan",
        layouts={},
        head_dim=128,
        num_qo_heads=4,
        num_kv_heads=1,
        dtype=torch.bfloat16,
        capability=None,
        tensor_cores=None,
    )

    with pytest.raises(ValueError, match="MHA serves one target KV group, got 2"):
        resolved.build((object(), object()), object(), max_running_req=1, capability=None)  # type: ignore[arg-type]


class _Layer:
    """A layer of every fake slot and a carry row, which records what it is bound to."""

    def __init__(self) -> None:
        self.bound: dict[str, object] = {}

    def state_slots(self) -> tuple[StateSlot, ...]:
        paged = tuple(StateSlot(name, requirement=name) for name, _, _ in _SLOTS)
        return (*paged, StateSlot("carry", state=_CARRY))

    def bind_state(self, bound):
        self.bound = dict(bound)


def test_a_second_family_resolves_groups_builds_over_every_target_group_and_binds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        family_module,
        "FAMILIES",
        MappingProxyType(
            {**family_module.FAMILIES, "fake": ("tests.support.attention_family", "resolve")}
        ),
    )
    request = AttentionRequest(
        target=_TARGET,
        backend="auto",
        page_size=None,
        capability=None,
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
    )
    resolved = resolve_attention_config({_TARGET: _REQUIREMENTS}, request)
    page_size = choose_page_size(None, resolved.pages, [r.tokens_per_entry for r in _REQUIREMENTS])
    slots = {
        (_TARGET, r.slot): ResolvedSlot(
            "fake", Spec(resolved.layouts[_TARGET, r.slot], r.tokens_per_entry)
        )
        for r in _REQUIREMENTS
    }
    layers = [_Layer(), _Layer()]
    layout = group_layers([(_TARGET, layer) for layer in layers], slots)
    capacity = StateCapacity(
        page_size=page_size,
        num_slots=2,
        max_seq_len=512,
        table_width=2,
        num_pages=4,
        snapshot_rows=range(2, 2),
        sink_slot=4,
        guard_slots=range(5, 5),
        snapshot_grid=page_size,
        export_granularity=1,
    )
    state = InferenceState.allocate(layout, capacity, device=torch.device("cpu"))
    table = PageTable(capacity, 2, device=torch.device("cpu"))
    target = tuple(group for group in state.groups if group.role is _TARGET)

    backend = resolved.build(target, table, max_running_req=2, capability=None)
    layout.bind((backend,) * len(target), state.recurrent)

    assert page_size == 256
    # One page ID addresses a page of every group: 256 window rows, 64 and 2 pooled ones.
    pools = [(group.spec.tokens_per_entry, group.layers[0].shape) for group in state.groups]
    assert pools == [(1, (5, 256 * 8)), (4, (5, 64 * 16)), (128, (5, 2 * 32))]
    assert isinstance(backend, FakeBackend)
    assert len(backend.groups) == 3
    assert all(built is group for built, group in zip(backend.groups, target, strict=True))
    for index, layer in enumerate(layers):
        assert {name: layer.bound[name] for name, _, _ in _SLOTS} == {
            name: PagedSlot(backend, index) for name, _, _ in _SLOTS
        }
        assert list(layer.bound["carry"]) == ["carry"]
