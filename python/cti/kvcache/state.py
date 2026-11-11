"""Kinds of inference state: what each kind allocates, known before any pool exists, and
the pools of one inference state."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import lcm
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Protocol

import torch

from cti.utils import align_ceil, div_ceil

from .recurrent_pool import RecurrentStatePool, SlotStateSpec

if TYPE_CHECKING:
    from .budget import SnapshotPolicy


class RowLayout(Protocol):
    """How one layer of a KV group lays out a page's entries; an attention family implements
    it."""

    def page_bytes(self, entries: int) -> int: ...

    def allocate(
        self,
        num_pages: int,
        entries: int,
        *,
        device: torch.device,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class Spec:
    """One layer of a KV group: its row ``layout``, the tokens one entry holds, and how many
    trailing keys its queries read (None: all)."""

    layout: RowLayout
    tokens_per_entry: int = 1
    window: int | None = None

    def entries(self, page_size: int) -> int:
        """The entries of one page of ``page_size`` tokens."""
        if page_size % self.tokens_per_entry:
            raise ValueError(
                f"page size {page_size} is not a multiple of {self.tokens_per_entry} tokens"
            )
        return page_size // self.tokens_per_entry

    def page_bytes(self, page_size: int) -> int:
        """One layer's bytes of one page."""
        return self.layout.page_bytes(self.entries(page_size))


class GroupRole(StrEnum):
    """The model a KV group serves."""

    TARGET = "target"
    DRAFT = "draft"


# A paged slot of one role's layers, by its requirement's name.
SlotKey = tuple[GroupRole, str]


@dataclass(frozen=True, slots=True)
class ResolvedSlot:
    """A paged slot as resolution chose it: its attention ``family`` and per-layer spec."""

    family: str
    spec: Spec


@dataclass(frozen=True, slots=True)
class StateSlot:
    """One state a layer holds: paged rows of a ``requirement``, named, or a slot-row
    ``state``."""

    name: str
    requirement: str | None = None
    state: SlotStateSpec | None = None

    def __post_init__(self) -> None:
        if (self.requirement is None) == (self.state is None):
            raise ValueError(f"state slot {self.name!r} needs exactly one of requirement, state")


@dataclass(frozen=True, slots=True)
class PagedSlot:
    """A paged slot as bound: its KV group's ``backend``, which kvcache never reads, and the
    layer's ``index`` in the group."""

    backend: Any
    index: int


# A bound slot: a paged slot, or the layer's slot-row views by tensor name.
BoundSlot = PagedSlot | Mapping[str, "torch.Tensor"]


class StateLayer(Protocol):
    """A layer that holds inference state, in the slots it declares and is bound to once the
    state exists."""

    def state_slots(self) -> tuple[StateSlot, ...]: ...

    def bind_state(self, bound: Mapping[str, BoundSlot]) -> None: ...


@dataclass(frozen=True, slots=True, eq=False)
class LayerSlot:
    """Where one layer's slot lives: in KV group ``group`` at the layer's ``index`` there,
    or, with no group, in recurrent row ``index``."""

    layer: StateLayer
    slot: str
    group: int | None
    index: int


class AllocatorModel(Protocol):
    """The device allocator's backing for the pools' allocations, which planning injects:
    the segment bytes one allocation maps, and the request size from which an allocation
    takes a segment of its own."""

    @property
    def OWN_SEGMENT_REQUEST_BYTES(self) -> int: ...

    def segment(self, nbytes: int) -> int: ...


class Allocation(NamedTuple):
    """Device bytes of one kind's pools: tensor payload, and the allocator's backing of it
    and of the scratch its operations may hold besides the pools."""

    payload_bytes: int
    backing_bytes: int


@dataclass(frozen=True, slots=True)
class _Payload:
    """The allocator model of a device without one: every allocation backs its payload
    exactly."""

    OWN_SEGMENT_REQUEST_BYTES: ClassVar[int] = 1 << 62

    def segment(self, nbytes: int) -> int:
        return nbytes


PAYLOAD: AllocatorModel = _Payload()


@dataclass(frozen=True, slots=True)
class GroupLayout:
    """One KV group before allocation: ``num_layers`` layers of ``spec``, the ``index``-th
    group of the state."""

    role: GroupRole
    index: int
    spec: Spec
    num_layers: int

    def bytes_per_page(self, page_size: int) -> int:
        return self.num_layers * self.spec.page_bytes(page_size)

    def allocation(self, num_pages: int, page_size: int, backing: AllocatorModel) -> Allocation:
        """The group's pools of ``num_pages`` pages, one allocation per layer."""
        layer_bytes = num_pages * self.spec.page_bytes(page_size)
        payload = self.num_layers * layer_bytes
        return Allocation(payload, self.num_layers * backing.segment(layer_bytes))


class RecurrentRows:
    """Request-slot rows, then snapshot rows, of every slot-state layer. Snapshots export
    at multiples of every layer's ``export_granularity``."""

    def __init__(self, specs: Sequence[SlotStateSpec]) -> None:
        self.specs = tuple(specs)
        self.row_bytes = sum(spec.bytes_per_slot for spec in self.specs)
        self.export_granularity = lcm(*(spec.export_granularity for spec in self.specs))

    def snapshot_count(self, share: int, snapshots: SnapshotPolicy) -> int:
        """The policy's explicit rows, or its ratio of the recurrent ``share``."""
        rows, ratio = snapshots.rows, snapshots.ratio
        ratio = snapshots.MEMORY_RATIO if ratio is None else ratio
        return rows if rows is not None else int(share * ratio // self.row_bytes)

    def allocation(self, num_slots: int, num_snapshots: int, backing: AllocatorModel) -> Allocation:
        """Rows of ``num_slots`` requests and ``num_snapshots`` snapshots, one allocation per
        layer; snapshots add one allocation of packed copy indices. Row copies need no
        workspace."""
        num_rows = num_slots + num_snapshots
        backing_bytes = sum(backing.segment(num_rows * s.bytes_per_slot) for s in self.specs)
        if num_snapshots:
            backing_bytes += backing.segment(4 * num_slots * 8)
        return Allocation(num_rows * self.row_bytes, backing_bytes)


@dataclass(frozen=True, slots=True)
class KVLayout:
    """One inference state before allocation: KV groups on one page table, in their index
    order, recurrent rows, and where each grouped layer's slots lie in them."""

    groups: tuple[GroupLayout, ...]
    rows: RecurrentRows
    slots: tuple[LayerSlot, ...] = ()

    def __post_init__(self) -> None:
        if [group.index for group in self.groups] != list(range(len(self.groups))):
            raise ValueError("KV groups must be indexed 0..n-1 in order")

    def bind(self, backends: Sequence[Any], recurrent: RecurrentStatePool | None) -> None:
        """Bind each grouped layer to its slots: group ``i``'s paged slots to
        ``backends[i]``, its slot rows to their views in ``recurrent``."""

        bound: dict[int, tuple[StateLayer, dict[str, BoundSlot]]] = {}
        for where in self.slots:
            if where.group is None:
                assert recurrent is not None
                slot: BoundSlot = recurrent.layer_state(where.index)
            else:
                slot = PagedSlot(backends[where.group], where.index)
            layer = where.layer
            bound.setdefault(id(layer), (layer, {}))[1][where.slot] = slot
        for layer, slots in bound.values():
            layer.bind_state(slots)

    def bytes_per_page(self, page_size: int) -> int:
        """One page of every group's layers."""
        return sum(group.bytes_per_page(page_size) for group in self.groups)

    def allocation(self, num_pages: int, page_size: int, backing: AllocatorModel) -> Allocation:
        """Every group's pools of ``num_pages`` pages, one allocation per layer."""
        allocations = [group.allocation(num_pages, page_size, backing) for group in self.groups]
        return Allocation(
            sum(a.payload_bytes for a in allocations), sum(a.backing_bytes for a in allocations)
        )

    def fit(self, budget: int, page_size: int, backing: AllocatorModel) -> int:
        """The most pages whose backing fits ``budget``; -1 if ``budget`` is negative."""
        # Backing grows with pages except where a pool's layer tensor reaches its own segment's
        # request size: bisect each growing stretch, from the top.
        high = budget // self.bytes_per_page(page_size)
        drops = {
            -(-backing.OWN_SEGMENT_REQUEST_BYTES // group.spec.page_bytes(page_size))
            for group in self.groups
        }
        for low in sorted({0, *drops}, reverse=True):
            stretch = range(low, high + 1)
            n = bisect_right(
                stretch,
                budget,
                key=lambda p: self.allocation(p, page_size, backing).backing_bytes,
            )
            if n:
                return low + n - 1
            high = min(high, low - 1)
        return -1


def group_layers(
    layers: Sequence[tuple[GroupRole, StateLayer]], slots: Mapping[SlotKey, ResolvedSlot]
) -> KVLayout:
    """Group ``layers``' paged slots by role, family and spec, groups and their layers in
    order of first appearance; number their slot rows in order."""

    groups: dict[tuple[GroupRole, str, Spec], list[StateLayer]] = {}
    row_specs: list[SlotStateSpec] = []
    layer_slots = []
    for role, layer in layers:
        for slot in layer.state_slots():
            if slot.state is not None:
                layer_slots.append(LayerSlot(layer, slot.name, None, len(row_specs)))
                row_specs.append(slot.state)
                continue
            assert slot.requirement is not None
            resolved = slots[role, slot.requirement]
            key = (role, resolved.family, resolved.spec)
            members = groups.setdefault(key, [])
            layer_slots.append(LayerSlot(layer, slot.name, list(groups).index(key), len(members)))
            members.append(layer)
    layouts = tuple(
        GroupLayout(role, index, spec, len(members))
        for index, ((role, _, spec), members) in enumerate(groups.items())
    )
    return KVLayout(layouts, RecurrentRows(row_specs), tuple(layer_slots))


@dataclass(frozen=True, slots=True)
class StateCapacity:
    """What an inference state holds: ``num_pages`` usable pages, sequences of at most
    ``max_seq_len`` tokens in ``table_width`` page-table columns, ``num_slots`` request
    slots and the snapshot rows past them. Its KV pools hold the ``sink_slot`` page, which
    only graph padding writes, then the ``guard_slots`` pages, which only guard rows write.
    Snapshots lie on multiples of ``snapshot_grid`` tokens: whole pages, at which every
    slot-state layer exports. A prompt forward exports after the most whole multiples of
    ``export_granularity`` tokens it runs (``SlotStateSpec.export_granularity``)."""

    page_size: int
    num_slots: int
    max_seq_len: int
    table_width: int
    num_pages: int
    snapshot_rows: range
    sink_slot: int
    guard_slots: range
    snapshot_grid: int
    export_granularity: int

    @property
    def pool_pages(self) -> int:
        """The pages every KV pool holds: the usable ones, the sink and the guard pages."""
        return self.guard_slots.stop

    @property
    def guard_start(self) -> int:
        """The flat token index of the first guard page's first token."""
        return self.guard_slots.start * self.page_size


@dataclass(frozen=True, slots=True)
class TableGeometry:
    """A page table's request slots and longest sequence, and the pages its pools hold
    past the usable ones: a graph sink page, then the guard pages of ``guard_tokens``."""

    page_size: int
    num_slots: int
    max_seq_len: int
    guard_tokens: int = 0

    @property
    def guard_pages(self) -> int:
        return div_ceil(self.guard_tokens, self.page_size)

    def capacity(self, layout: KVLayout, num_pages: int, num_snapshot_rows: int) -> StateCapacity:
        page_size = self.page_size
        max_seq_len = min(self.max_seq_len, num_pages * page_size)
        guard_slots = range(num_pages + 1, num_pages + 1 + self.guard_pages)
        # Kernels address tokens of every page, the guard pages' too, in int32.
        if guard_slots.stop * page_size > torch.iinfo(torch.int32).max:
            raise ValueError("cache-index range exceeds torch.int32")
        return StateCapacity(
            page_size,
            self.num_slots,
            max_seq_len,
            # Rows align to 128 bytes.
            align_ceil(div_ceil(max_seq_len, page_size), 32),
            num_pages,
            range(self.num_slots, self.num_slots + num_snapshot_rows),
            num_pages,
            guard_slots,
            lcm(page_size, layout.rows.export_granularity),
            layout.rows.export_granularity,
        )


@dataclass(frozen=True, slots=True, eq=False)
class PageGroup:
    """One KV group's pools on the state's page table: a tensor per layer in its spec's row
    layout, which the group's backend views."""

    role: GroupRole
    index: int
    spec: Spec
    layers: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class StateData:
    """An inference state's data, which a caller may allocate in a region that releases its
    memory and restores it at the same addresses: the recurrent rows' buffers, one per
    layer, and a ``PageGroup`` per group of its ``KVLayout``, in their order."""

    layout: KVLayout
    capacity: StateCapacity
    recurrent_rows: tuple[torch.Tensor, ...]
    groups: tuple[PageGroup, ...]

    @classmethod
    def allocate(
        cls, layout: KVLayout, capacity: StateCapacity, *, device: torch.device
    ) -> StateData:
        """Allocate the recurrent rows, then each KV group's pools."""

        rows = RecurrentStatePool.allocate_rows(
            layout.rows.specs,
            capacity.num_slots + len(capacity.snapshot_rows),
            device=device,
        )
        num_pages = capacity.pool_pages

        def pools(group: GroupLayout) -> tuple[torch.Tensor, ...]:
            entries = group.spec.entries(capacity.page_size)
            return tuple(
                group.spec.layout.allocate(num_pages, entries, device=device)
                for _ in range(group.num_layers)
            )

        groups = tuple(
            PageGroup(group.role, group.index, group.spec, pools(group)) for group in layout.groups
        )
        return cls(layout, capacity, rows, groups)


@dataclass(frozen=True, slots=True)
class InferenceState:
    """One inference state's pools: a ``PageGroup`` per group of its ``KVLayout``, in their
    order, and the recurrent rows, if any."""

    capacity: StateCapacity
    groups: tuple[PageGroup, ...]
    recurrent: RecurrentStatePool | None

    @classmethod
    def allocate(
        cls, layout: KVLayout, capacity: StateCapacity, *, device: torch.device
    ) -> InferenceState:
        """Allocate the state's data, then the state over it."""

        return cls.over(StateData.allocate(layout, capacity, device=device), device=device)

    @classmethod
    def over(cls, data: StateData, *, device: torch.device) -> InferenceState:
        """The state over ``data``, with the metadata it needs beside it (the recurrent rows'
        copy table), allocated now: outside any region that may release ``data``'s memory."""

        specs, capacity = data.layout.rows.specs, data.capacity
        recurrent = (
            RecurrentStatePool(
                specs,
                num_slots=capacity.num_slots,
                num_snapshot_rows=len(capacity.snapshot_rows),
                device=device,
                rows=data.recurrent_rows,
            )
            if specs
            else None
        )
        return cls(capacity, data.groups, recurrent)
