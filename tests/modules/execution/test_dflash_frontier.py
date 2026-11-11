"""DFlash device frontiers on CPU: block addressing, the capped commit, greedy acceptance."""

from __future__ import annotations

import itertools

import pytest
import torch

from tests.support.dflash_frontier import (
    BlockGeometry,
    commit_block,
    greedy_accepted,
    prepare_block,
)

_Q = 4
_M = 12
_EOS = 9
_GUARD = 1000


def _geometry() -> BlockGeometry:
    return BlockGeometry(block_size=_Q, max_seq_len=_M, guard_start=_GUARD)


def _tables(rows: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Token and cache-index tables; row r maps position p to cache row 100 * r + p."""

    cache = (
        torch.arange(_M, dtype=torch.int32) + 100 * torch.arange(rows, dtype=torch.int32)[:, None]
    )
    tokens = torch.arange((_M + _Q) * rows, dtype=torch.int32).view(rows, _M + _Q) % 50
    return tokens, cache


def _prepare(frontiers: list[int], done: list[bool]) -> dict[str, torch.Tensor]:
    size = len(frontiers)
    tokens, cache = _tables()
    out = {
        "input_ids": torch.empty(size * _Q, dtype=torch.int32),
        "positions": torch.empty(size * _Q, dtype=torch.int64),
        "rope_positions": torch.empty(size * _Q, dtype=torch.int64),
        "cache_write_indices": torch.empty(size * _Q, dtype=torch.int32),
        "kv_indptr": torch.empty(size + 1, dtype=torch.int32),
        "kv_indices": torch.empty(size * (_M + _Q), dtype=torch.int32),
    }
    prepare_block(
        _geometry(),
        rows=torch.arange(size, dtype=torch.int32),
        frontiers=torch.tensor(frontiers),
        done=torch.tensor(done),
        token_table=tokens,
        cache_index_table=cache,
        mask_token_id=49,
        **out,
    )
    return {**out, "tokens": tokens, "cache": cache}


def test_live_rows_address_their_mapping_then_guard_rows_and_done_rows_only_guard() -> None:
    # A row inside the capacity, one crossing it, and a done row.
    frontiers, done = [3, _M - 2, 5], [False, False, True]
    out = _prepare(frontiers, done)
    cache, tokens = out["cache"], out["tokens"]

    def address(row: int, position: int) -> int:
        if done[row]:
            return _GUARD + row * _Q + position
        if position < _M:
            return int(cache[row, position])
        return _GUARD + row * _Q + position - _M

    writes = out["cache_write_indices"].view(3, _Q).tolist()
    assert writes[0] == [address(0, 3 + i) for i in range(_Q)]
    assert writes[1] == [address(1, _M - 2 + i) for i in range(_Q)]
    assert writes[2] == [address(2, i) for i in range(_Q)]
    lengths = [3 + _Q, _M - 2 + _Q, _Q]
    assert out["kv_indptr"].tolist() == [0, *itertools.accumulate(lengths)]
    keys = out["kv_indices"].tolist()
    start = 0
    for row, length in enumerate(lengths):
        expected = [address(row, j) for j in range(length)]
        assert keys[start : start + length] == expected
        start += length
    # Padding reads a legal guard key.
    assert set(keys[start:]) == {_GUARD}
    assert out["positions"].view(3, _Q)[1].tolist() == [_M - 2 + i for i in range(_Q)]
    assert out["rope_positions"].view(3, _Q)[1].tolist() == [_M - 2, _M - 1, _M - 1, _M - 1]
    ids = out["input_ids"].view(3, _Q)
    assert ids[:, 0].tolist() == [int(tokens[r, f]) for r, f in enumerate(frontiers)]
    assert (ids[:, 1:] == 49).all()


def test_positions_keep_int64_frontiers_past_int32() -> None:
    big = 2**33 + 5
    out = _prepare([big], [False])

    assert out["positions"].dtype == torch.int64
    assert out["positions"].tolist() == [big + i for i in range(_Q)]
    assert out["rope_positions"].tolist() == [_M - 1] * _Q


def _host_committed(tokens: list[int], accepted: int, remaining: int, ignore_eos: bool) -> int:
    """The scheduler's visible prefix: accepted drafts then correction, clipped by the output
    budget, then cut after the first EOS."""

    visible = tokens[: accepted + 1][:remaining]
    if not ignore_eos and _EOS in visible:
        visible = visible[: visible.index(_EOS) + 1]
    return len(visible)


def _commit(frontier, done, accepted, tokens, output_end, ignore_eos=False):
    token_table, _ = _tables(1)
    before = token_table.clone()
    frontier_table = torch.tensor([frontier])
    done_table = torch.tensor([done])
    accepted_in = torch.tensor([accepted], dtype=torch.int32)
    committed, finished = commit_block(
        _geometry(),
        rows=torch.zeros(1, dtype=torch.int32),
        frontiers=torch.tensor([frontier]),
        done=torch.tensor([done]),
        accepted=accepted_in,
        tokens=torch.tensor([tokens], dtype=torch.int32),
        output_ends=torch.tensor([output_end]),
        ignore_eos=torch.tensor([ignore_eos]),
        stop_token_ids=torch.tensor([_EOS], dtype=torch.int32),
        token_table=token_table,
        frontier_table=frontier_table,
        done_table=done_table,
    )
    assert accepted_in.tolist() == [accepted], "raw a is never rewritten"
    return (
        int(committed),
        bool(finished),
        before,
        token_table,
        int(frontier_table),
        bool(done_table),
    )


@pytest.mark.parametrize("frontier", [0, _M - 6, _M - 2, _M - 1])
@pytest.mark.parametrize("remaining", [0, 1, _Q - 1, _Q, _Q + 1])
@pytest.mark.parametrize("accepted", range(_Q))
@pytest.mark.parametrize("stop_at", [None, 0, 2, _Q - 1])
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_the_commit_caps_by_budget_stop_and_capacity(
    frontier: int, remaining: int, accepted: int, stop_at: int | None, ignore_eos: bool
) -> None:
    tokens = [20, 21, 22, 23]
    if stop_at is not None:
        tokens[stop_at] = _EOS
    # Outputs may occupy positions up to output_end - 1, at most the capacity's last.
    output_end = frontier + 1 + remaining
    capacity_left = max(_M - 1 - frontier, 0)
    expected = _host_committed(tokens, accepted, min(remaining, capacity_left), ignore_eos)

    committed, finished, before, after, new_frontier, done = _commit(
        frontier, False, accepted, tokens, output_end, ignore_eos
    )

    assert committed == expected
    assert new_frontier == frontier + committed
    hit_stop = not ignore_eos and _EOS in tokens[:committed]
    assert finished == done == (hit_stop or frontier + committed >= min(output_end, _M) - 1)
    written = after[0, frontier + 1 : frontier + 1 + committed].tolist()
    assert written == tokens[:committed]
    # Everything else outside the scratch columns is untouched.
    untouched = torch.ones(_M + _Q, dtype=torch.bool)
    untouched[frontier + 1 : frontier + 1 + committed] = False
    untouched[_M:] = False
    assert torch.equal(after[0][untouched], before[0][untouched])


def test_a_done_row_commits_nothing_and_writes_only_scratch() -> None:
    committed, finished, before, after, frontier, done = _commit(5, True, _Q - 1, [1, 2, 3, 4], 40)

    assert (committed, finished, frontier, done) == (0, True, 5, True)
    assert torch.equal(after[0, :_M], before[0, :_M])
    assert after[0, _M:].tolist() == [1, 2, 3, 4]


def test_mixed_live_and_done_rows_commit_independently() -> None:
    token_table, _ = _tables(3)
    frontiers = torch.tensor([2, 7, 4])
    frontier_table, done_table = frontiers.clone(), torch.tensor([False, True, False])
    committed, finished = commit_block(
        _geometry(),
        rows=torch.tensor([2, 1, 0], dtype=torch.int32),
        frontiers=frontiers[[2, 1, 0]],
        done=done_table[[2, 1, 0]],
        accepted=torch.tensor([3, 3, 1], dtype=torch.int32),
        tokens=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [_EOS, 1, 1, 1]], dtype=torch.int32),
        output_ends=torch.tensor([40, 40, 40]),
        ignore_eos=torch.tensor([False, False, False]),
        stop_token_ids=torch.tensor([_EOS], dtype=torch.int32),
        token_table=token_table,
        frontier_table=frontier_table,
        done_table=done_table,
    )

    assert committed.tolist() == [4, 0, 1]
    assert finished.tolist() == [False, True, True]
    assert frontier_table.tolist() == [3, 7, 8]
    assert done_table.tolist() == [True, True, False]
    assert token_table[2, 5:9].tolist() == [1, 2, 3, 4]
    assert token_table[0, 3].item() == _EOS


def test_greedy_acceptance_is_the_matching_prefix() -> None:
    targets = torch.tensor([[5, 6, 7, 8], [5, 0, 7, 8], [0, 6, 7, 8], [5, 6, 7, 0]])
    proposals = torch.tensor([[5, 6, 7]] * 4)

    assert greedy_accepted(proposals, targets).tolist() == [3, 1, 0, 3]
