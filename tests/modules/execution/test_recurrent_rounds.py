from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cti.forward import RoundShape
from cti.recurrent import RecurrentMetadata, RecurrentPhase, RecurrentRounds


class _Transaction:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def begin(self, slot: int, sequence: int) -> None:
        self.calls.append(("begin", slot, sequence))

    def end(self, slot: int, size: int, sequence: int) -> SimpleNamespace:
        self.calls.append(("end", slot, size, sequence))
        return SimpleNamespace(copies=(), settle=lambda: self.calls.append(("settle", slot)))

    def capture_handoffs(self, pool: object, stream: object) -> None:
        self.calls.append(("capture",))


class _Replay:
    def __init__(self, name: str, transaction: _Transaction, calls: list) -> None:
        self.name, self.transaction, self.calls = name, transaction, calls

    def handoff(self, slot: int, rows, frontiers) -> None:
        self.calls.append(("handoff", self.name, slot, list(rows)))

    def commit(self, v, lengths, snapshot_rows=None, snapshot_offsets=None) -> None:
        self.calls.append(("commit", self.name, v.slot, lengths.tolist()))


def _rounds(depth: int = 2):
    calls: list = []
    family = _Transaction(calls)
    replays = [_Replay(name, family, calls) for name in ("a", "b")]
    return RecurrentRounds(replays, shape=RoundShape(4, depth, 3)), calls


def test_a_round_opens_each_family_once_and_hands_off_only_its_stale_rows() -> None:
    rounds, calls = _rounds()
    rounds.mark_stale([0, 2, 5])

    slot = rounds.begin([2, 1], torch.zeros(6, dtype=torch.int64))

    assert slot == 0
    assert calls == [("begin", 0, 1), ("handoff", "a", 0, [2]), ("handoff", "b", 0, [2])]
    rounds.end(slot, 2).settle()
    calls.clear()
    # Row 2 is fresh now; rows 0 and 5 stay stale for their own rounds.
    slot = rounds.begin([2, 0], torch.zeros(6, dtype=torch.int64))
    assert calls == [("begin", 1, 2), ("handoff", "a", 1, [0]), ("handoff", "b", 1, [0])]


def test_a_commit_reads_the_verify_rows_its_slot_prepared() -> None:
    rounds, calls = _rounds()
    slot_ids, positions = torch.tensor([3, -1], dtype=torch.int32), torch.zeros(2, 4)
    metadata = rounds.prepare_verify(1, slot_ids, positions)

    assert metadata.phase is RecurrentPhase.VERIFY and metadata.verify.slot == 1
    rounds.commit(1, torch.tensor([2, 0], dtype=torch.int32))
    assert calls == [("commit", "a", 1, [2, 0]), ("commit", "b", 1, [2, 0])]
    extension = RecurrentMetadata(torch.zeros(1, dtype=torch.int32), 1, 3)
    joint = rounds.verify_after(extension, 1, 2)
    assert joint.verify is metadata.verify and joint.phase is RecurrentPhase.VERIFY
    assert joint.num_extend_tokens == 3


def test_a_status_slot_is_reused_only_after_its_settlement() -> None:
    rounds, _ = _rounds(depth=1)
    frontiers = torch.zeros(6, dtype=torch.int64)
    status = rounds.end(rounds.begin([0], frontiers), 1)

    with pytest.raises(RuntimeError, match="not settled"):
        rounds.begin([0], frontiers)
    status.settle()
    assert rounds.begin([0], frontiers) == 0
