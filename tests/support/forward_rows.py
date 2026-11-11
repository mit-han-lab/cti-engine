"""Forward rows from plain values, for tests of the forward contract: models, layers,
attention, recurrent backends. They never reach the inference step's rows."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

from cti.forward import ForwardRows, RecurrentExport


@dataclass(frozen=True, slots=True)
class Entry:
    """Slot ``slot`` runs tokens ``[start, end)`` over ``start`` cached ones."""

    slot: int
    start: int
    end: int
    decode: bool = False
    restore_row: int | None = None
    captures: tuple[RecurrentExport, ...] = ()

    @property
    def length(self) -> int:
        return self.end - self.start


def extend_entry(
    slot: int,
    start: int,
    end: int,
    *,
    restore_row: int | None = None,
    captures: tuple[RecurrentExport, ...] = (),
) -> Entry:
    """Prefill of tokens ``[start, end)``, restoring snapshot ``restore_row`` first."""

    return Entry(slot, start, end, restore_row=restore_row, captures=captures)


def decode_entry(slot: int, position: int, *, captures: tuple[RecurrentExport, ...] = ()) -> Entry:
    """Decode of the token at ``position``."""

    return Entry(slot, position, position + 1, decode=True, captures=captures)


def forward_rows(*entries: Entry, padding: int = 0, padding_slot: int = 0) -> ForwardRows:
    """The rows of ``entries``, extensions first, then ``padding`` dummy decodes of
    ``padding_slot`` at position 0."""

    num_extend = sum(not entry.decode for entry in entries)
    if any(entry.decode for entry in entries[:num_extend]):
        raise ValueError("extend entries must precede decode entries")
    padded = (*entries, *(decode_entry(padding_slot, 0),) * padding)
    return ForwardRows(
        size=len(entries),
        num_extend=num_extend,
        slots=tuple(entry.slot for entry in padded),
        kv_prefix_lens=tuple(entry.start for entry in padded),
        sequence_lens=tuple(entry.end for entry in padded),
        rope_offsets=(0,) * len(padded),
        restores=tuple(
            (entry.restore_row, entry.slot)
            for entry in entries[:num_extend]
            if entry.restore_row is not None
        ),
        captures=tuple(entry.captures for entry in entries),
        logit_rows=tuple(end - 1 for end in accumulate(entry.length for entry in entries)),
    )
