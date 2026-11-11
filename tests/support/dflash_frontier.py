"""Torch oracle of DFlash round bookkeeping, which ``ctikernels.speculative``'s
``prepare_block`` and ``commit_block`` replace in the runner: fixed-Q block inputs and KV
addresses, and the commit that caps each row by its output budget, stop tokens and the
sequence capacity.

A row's frontier F is its anchor's position: the last emitted token, whose KV is not yet
written. A round's block holds positions ``[F, F + Q)``; its row commits the first ``v`` of
them and emits ``v`` tokens at ``[F + 1, F + v + 1)``. A done row runs as Q dummy queries on
its execution row's guard KV and commits nothing.

Every function is a fixed-shape torch composition: capture-safe and free of host
synchronization, and every index is made legal before it is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BlockGeometry:
    """Static addressing of a runner's rounds.

    Execution row b owns the guard cache rows ``guard_start + b * Q + [0, Q)``: physical KV
    outside every request's mapping, which only that row's dummy and past-capacity queries
    write. Token tables are at least ``max_seq_len + Q`` wide: column ``max_seq_len + j``
    is the scratch of an uncommitted output j.
    """

    block_size: int
    max_seq_len: int
    guard_start: int


def prepare_block(
    geometry: BlockGeometry,
    *,
    rows: torch.Tensor,
    frontiers: torch.Tensor,
    done: torch.Tensor,
    token_table: torch.Tensor,
    cache_index_table: torch.Tensor,
    mask_token_id: int,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    rope_positions: torch.Tensor,
    cache_write_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
) -> None:
    """Write a round's ``[anchor, mask, ...]`` inputs and its fixed-query KV metadata.

    ``rows`` are the request rows (int32 ``[B]``); ``frontiers`` (int64) and ``done`` (bool)
    their ``[B]`` state at the round's start. Outputs, each a flat ``B * Q`` view, are the
    input IDs, the logical int64 positions F + i, their rotary positions (past the sequence
    capacity, its last position), and the cache rows the block writes. ``kv_indptr``
    (int32 ``[B + 1]``) and ``kv_indices`` (int32, ``B * (max_seq_len + Q)`` entries) give a
    live row its F + Q keys, a done row its Q guard keys, and padding a guard key.
    """

    q, capacity = geometry.block_size, geometry.max_seq_len
    size = rows.numel()
    request = rows.long()
    device = rows.device
    offsets = torch.arange(q, device=device)
    guard = geometry.guard_start + torch.arange(size, device=device)[:, None] * q
    logical = frontiers[:, None] + offsets
    column = logical.clamp(max=capacity - 1)
    positions.copy_(logical.view(-1))
    rope_positions.copy_(column.view(-1))
    # Live rows write their mapped positions, then guard rows past the capacity.
    writes = torch.where(
        logical < capacity,
        cache_index_table[request[:, None], column],
        guard + (logical - capacity).clamp(0, q - 1),
    )
    writes = torch.where(done[:, None], guard + offsets, writes)
    cache_write_indices.copy_(writes.view(-1))
    ids = input_ids.view(size, q)
    ids.fill_(mask_token_id)
    ids[:, 0] = token_table[request, frontiers.clamp(max=capacity - 1)]

    lengths = torch.where(done, q, frontiers + q)
    ends = lengths.cumsum(0)
    kv_indptr[:1].zero_()
    kv_indptr[1:].copy_(ends)
    flat = torch.arange(kv_indices.numel(), device=device)
    row = torch.searchsorted(ends, flat, right=True).clamp(max=size - 1)
    local = flat - (ends[row] - lengths[row])
    row_guard = guard[row, 0]
    keys = torch.where(
        local < capacity,
        cache_index_table[request[row], local.clamp(0, capacity - 1)],
        row_guard + (local - capacity).clamp(0, q - 1),
    )
    keys = torch.where(done[row], row_guard + local.clamp(0, q - 1), keys)
    kv_indices.copy_(torch.where(flat < ends[-1], keys, geometry.guard_start))


def commit_block(
    geometry: BlockGeometry,
    *,
    rows: torch.Tensor,
    frontiers: torch.Tensor,
    done: torch.Tensor,
    accepted: torch.Tensor,
    tokens: torch.Tensor,
    output_ends: torch.Tensor,
    ignore_eos: torch.Tensor,
    stop_token_ids: torch.Tensor,
    token_table: torch.Tensor,
    frontier_table: torch.Tensor,
    done_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Commit each row's verified block; return its committed length v (int64 ``[B]``) and
    whether it is done (bool ``[B]``).

    ``accepted`` is the raw accepted draft count a (int32), ``tokens`` the ``[B, Q]`` int32
    accepted drafts then correction or bonus, ``output_ends`` (int64) one past the last
    position the row may emit, ``frontiers``/``done`` its state at the round's start. A live
    row keeps the first ``a + 1`` tokens that fit its output budget and the sequence
    capacity, up to and including its first stop token (unless ``ignore_eos``); it writes
    them to ``token_table`` and advances its frontier by v. A done row writes only scratch.
    The row tables (``frontier_table`` int64, ``done_table`` bool) take the new state.
    """

    q, capacity = geometry.block_size, geometry.max_seq_len
    request = rows.long()
    offsets = torch.arange(q, device=rows.device)
    end = output_ends.clamp(max=capacity)
    limit = torch.minimum((end - 1 - frontiers).clamp(0, q), accepted.long() + 1)
    stops = (tokens[..., None] == stop_token_ids).any(-1)
    stops &= ~ignore_eos[:, None] & (offsets < limit[:, None])
    first_stop = torch.where(stops, offsets, q).amin(1)
    committed = torch.where(first_stop < q, first_stop + 1, limit)
    committed = torch.where(done, 0, committed)
    finished = done | (first_stop < q) | (frontiers + committed >= end - 1)
    columns = torch.where(
        offsets < committed[:, None], frontiers[:, None] + 1 + offsets, capacity + offsets
    )
    token_table[request[:, None], columns] = tokens
    frontier_table[request] = frontiers + committed
    done_table[request] = finished
    return committed, finished


def greedy_accepted(proposals: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The raw accepted count a (int32 ``[B]``): the length of the drafts' prefix that
    matches the target argmax tokens before them."""

    matches = proposals == targets[:, :-1]
    return matches.int().cumprod(1).sum(1, dtype=torch.int32)


__all__ = ["BlockGeometry", "commit_block", "greedy_accepted", "prepare_block"]
