"""The host's account of the runner's encoder-output rows."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .request import Request


class EncoderBudget:
    """One step's encoder rows: those no pinned image holds. Each image the step touches
    takes its rows once, unless pinned; an image whose rows are not left waits."""

    def __init__(self, free: int, pinned: set[bytes]) -> None:
        self._free = free
        self._taken = pinned

    def take(self, key: bytes, rows: int) -> bool:
        """Take image ``key``'s ``rows`` for the step; whether they were left."""

        if key not in self._taken:
            if rows > self._free:
                return False
            self._free -= rows
            self._taken.add(key)
        return True


class EncoderCache:
    """Which images' encoder output the runner's ``rows`` hold, by ``ImageSpan.key``; the
    least recently used image no request pins makes room first. A request pins the image its
    planned prompt extensions end inside, and a step the images it touches, which its
    ``EncoderBudget`` keeps within the rows. Every rank plans the same calls, so their rows
    agree. ``checks`` checks each step's packed loads and stores, and the pins."""

    def __init__(self, rows: int, *, checks: bool = False) -> None:
        self.rows = rows
        self._checks = checks
        self._free = list(range(rows))
        self._images: OrderedDict[bytes, list[int]] = OrderedDict()
        self._pins: dict[int, bytes] = {}  # request uid -> the image it pins

    def budget(self) -> EncoderBudget:
        pinned = set(self._pins.values())
        return EncoderBudget(self.rows - sum(len(self._images[key]) for key in pinned), pinned)

    def plan(
        self, images: Sequence[tuple[bytes, int]]
    ) -> tuple[dict[bytes, list[int]], dict[bytes, list[int]]]:
        """One step's ``(key, tokens)`` images: the rows of those held, all made the most
        recently used, and rows for each other image once, evicting no pinned or touched
        image."""

        held = {key: self._images[key] for key, _ in images if key in self._images}
        for key in held:
            self._images.move_to_end(key)
        kept = {*self._pins.values(), *(key for key, _ in images)}
        stored: dict[bytes, list[int]] = {}
        for key, num_rows in images:
            if key not in held and key not in stored:
                stored[key] = self._store(key, num_rows, kept)
        return held, stored

    def pin(self, uid: int, key: bytes | None) -> None:
        """Pin image ``key`` for request ``uid`` in place of the one it pinned; ``None``
        releases it."""

        if key is None:
            self._pins.pop(uid, None)
        else:
            self._pins[uid] = key

    def check(self, store_rows: Sequence[int], src_rows: Sequence[int], *, cached: int) -> None:
        """Under checks, before the step's pins: a step's ``ImageBatch`` rows write distinct
        pool rows no pinned image or ``cached`` load holds, its cached loads read held rows,
        and its loads after the encoder only rows it writes."""

        if not self._checks:
            return
        stores = set(store_rows)
        held = {row for rows in self._images.values() for row in rows}
        pinned = {row for key in self._pins.values() for row in self._images[key]}
        if (
            len(stores) != len(store_rows)
            or not stores.issubset(range(self.rows))
            or not stores.issuperset(src_rows[cached:])
            or not stores.isdisjoint(pinned.union(src_rows[:cached]))
            or not held.issuperset(src_rows[:cached])
        ):
            raise RuntimeError("a step's image stores repeat, leave the rows or miss a load")

    def check_pins(self, prefilling: Iterable[Request]) -> None:
        """Under checks: each pin is a ``prefilling`` request's that ends inside its image."""

        if not self._checks:
            return
        if self._pins != {r.uid: key for r in prefilling if (key := r.split_image) is not None}:
            raise RuntimeError("the encoder cache's pins are not the split images'")

    def _store(self, key: bytes, num_rows: int, kept: set[bytes]) -> list[int]:
        if num_rows > self.rows:
            raise ValueError(f"an image of {num_rows} tokens exceeds {self.rows} encoder rows")
        while len(self._free) < num_rows:
            self._free += self._images.pop(next(k for k in self._images if k not in kept))
        rows, self._free = self._free[:num_rows], self._free[num_rows:]
        self._images[key] = rows
        return rows

    def reset(self) -> None:
        """Forget every image: the rows' memory went to training."""

        if self._checks and self._pins:
            raise RuntimeError("a reset encoder cache still pins images")
        self._free = list(range(self.rows))
        self._images.clear()
