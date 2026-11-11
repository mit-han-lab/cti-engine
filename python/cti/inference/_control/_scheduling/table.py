# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/scheduler/table.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .request import Request


@dataclass(frozen=True, slots=True)
class RequestSlot:
    """One allocation lifetime of a physical request-table row."""

    index: int
    generation: int


@dataclass(eq=False, slots=True)
class Slot:
    """One admitted request in its request-table row, until the row is reclaimed: the
    request, and its unsettled speculative rounds, which ``RoundLedger`` keeps here: each
    round's output credit, and the device row after the last settled one (its frontier and
    whether it is done) while more are unsettled."""

    id: RequestSlot
    request: Request
    round_credits: list[int] = field(default_factory=list)
    round_device_row: tuple[int, bool] | None = None


class SlotTable:
    """The request-table rows and the admitted request each holds (``Slot``), from
    admission until the row is reclaimed: by row and by uid, the running ones in the order
    they started running, which every rank sees alike, and the retired ones whose row
    awaits reclamation."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._free = list(range(size))
        self._generations = [0] * size
        self._slots: list[Slot | None] = [None] * size
        self._by_uid: dict[int, Slot] = {}
        self._running: dict[Slot, None] = {}  # an ordered set
        self._retired: dict[RequestSlot, Slot] = {}

    @property
    def available_size(self) -> int:
        return len(self._free)

    def allocate(self, request: Request) -> RequestSlot:
        """A free row for ``request``, admitted into it."""

        if not self._free:
            raise RuntimeError("no request table slots are available")
        index = self._free.pop()
        if self._slots[index] is not None:
            raise RuntimeError(f"request table slot {index} is already live")
        generation = self._generations[index]
        self._generations[index] += 1
        slot = Slot(RequestSlot(index=index, generation=generation), request)
        self._slots[index] = self._by_uid[request.uid] = slot
        return slot.id

    def slot(self, id: RequestSlot) -> Slot:
        """Live row ``id``'s slot."""

        if id.index >= self._size:
            raise RuntimeError(f"request table slot {id.index} is out of range")
        slot = self._slots[id.index]
        if slot is None:
            raise RuntimeError(f"request table slot {id.index} is not live")
        if slot.id != id:
            raise RuntimeError(
                f"request table slot {id.index} has stale generation "
                f"{id.generation}; current generation is {slot.id.generation}"
            )
        return slot

    def require_live(self, id: RequestSlot) -> None:
        self.slot(id)

    def release(self, id: RequestSlot) -> None:
        """End row ``id``'s lifetime, and its request's admission."""

        slot = self.slot(id)
        self._slots[id.index] = None
        self._free.append(id.index)
        del self._by_uid[slot.request.uid]

    def request(self, uid: int) -> Request | None:
        """The admitted request ``uid``, whose row is live."""

        slot = self._by_uid.get(uid)
        return None if slot is None else slot.request

    # ============================ Running requests ============================

    @property
    def running(self) -> tuple[Slot, ...]:
        """The running requests' slots, in the order they started running."""

        return tuple(self._running)

    @property
    def num_running(self) -> int:
        return len(self._running)

    def is_running(self, request: Request) -> bool:
        return self._slots[request.slot.index] in self._running

    def run(self, request: Request) -> None:
        self._running.setdefault(self.slot(request.slot), None)

    def stop(self, request: Request) -> None:
        if request.is_admitted:
            self._running.pop(self._slots[request.slot.index], None)

    # ============================ Retired requests ============================

    def retire(self, request: Request) -> None:
        """Retire ``request``'s row for reclamation, which it may already await (a retracted
        request finishing as its steps in flight settle)."""

        slot = self.slot(request.slot)
        if self._retired.setdefault(slot.id, slot) is not slot:
            raise RuntimeError(f"table slot {slot.id} is already pending reclamation")

    def retired(self, id: RequestSlot) -> Request | None:
        slot = self._retired.get(id)
        return None if slot is None else slot.request

    def reclaimed(self, id: RequestSlot) -> None:
        del self._retired[id]

    @property
    def has_retired(self) -> bool:
        return bool(self._retired)

    @property
    def retracting(self) -> tuple[Request, ...]:
        """The retired requests that have not finished: retracted, to resume."""

        return tuple(s.request for s in self._retired.values() if not s.request.is_finished)

    def require_all_released(self) -> None:
        """Require every row's lifetime to have ended exactly once."""

        if self._running:
            raise RuntimeError("inference is not quiescent: running decode requests")
        if self._retired:
            raise RuntimeError("inference is not quiescent: retired requests await reclamation")
        counts = Counter(self._free)
        expected = set(range(self._size))
        live = tuple(slot.id for slot in self._slots if slot is not None)
        duplicates = sorted(index for index, count in counts.items() if count > 1)
        missing, unexpected = sorted(expected - counts.keys()), sorted(counts.keys() - expected)
        if live or duplicates or missing or unexpected:
            raise RuntimeError(
                "request table slots are not fully released: "
                f"live={live}, duplicates={duplicates}, "
                f"missing={missing}, unexpected={unexpected}"
            )
