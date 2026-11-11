"""The runner's round bookkeeping, ``ctikernels.speculative`` on its CPU reference with the
runner's buffer layout, equals the torch composition it replaced (``dflash_frontier``); a
round's addresses come from the attention backend's ``RoundAddresses``."""

from __future__ import annotations

import pytest
import torch

from cti.attention.fixed_query import FixedQueryShape, RoundAddresses
from ctikernels.speculative import commit_block, prepare_block, prepare_round
from tests.support.dflash_frontier import BlockGeometry, greedy_accepted
from tests.support.dflash_frontier import commit_block as commit_oracle
from tests.support.dflash_frontier import prepare_block as prepare_oracle

_M, _GUARD, _EOS, _MASK = 12, 1000, 9, 49
# Table rows: requests 0..4 plus the graph padding row; widths as the runner aligns them.
_ROWS, _WIDTH = 6, 32


def _tables() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "frontiers": torch.tensor([3, _M - 2, 5, 0, _M - 1, 0]),
        "done": torch.tensor([False, False, True, False, False, False]),
        "tokens": torch.randint(0, 40, (_ROWS, _WIDTH), generator=generator, dtype=torch.int32),
        "cache": (torch.arange(_WIDTH) + 100 * torch.arange(_ROWS)[:, None]).int(),
    }


@pytest.mark.parametrize("greedy", [True, False])
@pytest.mark.parametrize("rows", [[2, 0, 4], [1], [3, 1, 0, 4]])
# Q counts the anchor: one draft, today's v1 16-wide rounds and the most, 16 drafts.
@pytest.mark.parametrize("q", [2, 4, 16, 17])
def test_round_bookkeeping_matches_the_oracle(rows: list[int], greedy: bool, q: int) -> None:
    size = len(rows)
    request = torch.tensor(rows, dtype=torch.int32)
    rope_offsets, prompt_lengths = torch.tensor(rows) * 7 - 3, torch.tensor(rows) + 1
    kernel, oracle = _tables(), _tables()

    def outputs() -> dict[str, torch.Tensor]:
        return {
            "start_frontiers": torch.empty(size, dtype=torch.int64),
            "round_done": torch.empty(size, dtype=torch.bool),
            "k2_slots": torch.empty(size, dtype=torch.int32),
            "input_ids": torch.empty(size * q, dtype=torch.int32),
            "positions": torch.empty(size * q, dtype=torch.int64),
            "rope_positions": torch.empty(size * q, dtype=torch.int64),
            "verify_rope_positions": torch.empty(size * q, dtype=torch.int64),
            "cache_write_indices": torch.empty(size * q, dtype=torch.int32),
            "kv_indptr": torch.empty(size + 1, dtype=torch.int32),
            "kv_indices": torch.full((size * (_M + q),), -7, dtype=torch.int32),
            "cursors": torch.empty(size, dtype=torch.int64),
        }

    got, want, block = outputs(), outputs(), outputs()
    addressed = ("cache_write_indices", "kv_indptr", "kv_indices")
    # The engine's round: ids, positions and rope, then the backend's addresses.
    prepare_round(
        request,
        kernel["frontiers"],
        kernel["done"],
        kernel["tokens"],
        block_size=q,
        max_seq_len=_M,
        mask_token_id=_MASK,
        rope_offsets=rope_offsets,
        prompt_lengths=prompt_lengths,
        **{name: tensor for name, tensor in got.items() if name not in addressed},
    )
    shape = FixedQueryShape(size, q, _M + q, ((True, None),), torch.bfloat16)
    addresses = RoundAddresses(kernel["cache"], shape, max_seq_len=_M, guard_start=_GUARD)
    addresses.prepare(request, got["start_frontiers"], got["round_done"])
    got["cache_write_indices"] = addresses.write
    got["kv_indptr"] = addresses.indptr
    got["kv_indices"] = addresses.indices
    # The one-launch round it replaced writes the same bits.
    prepare_block(
        request,
        _tables()["frontiers"],
        _tables()["done"],
        _tables()["tokens"],
        kernel["cache"],
        block_size=q,
        max_seq_len=_M,
        guard_start=_GUARD,
        mask_token_id=_MASK,
        rope_offsets=rope_offsets,
        prompt_lengths=prompt_lengths,
        **block,
    )
    geometry = BlockGeometry(q, _M, _GUARD)
    index = request.long()
    want["start_frontiers"].copy_(oracle["frontiers"][index])
    want["round_done"].copy_(oracle["done"][index])
    want["k2_slots"].copy_(torch.where(want["round_done"], -1, request))
    prepare_oracle(
        geometry,
        rows=request,
        frontiers=want["start_frontiers"],
        done=want["round_done"],
        token_table=oracle["tokens"],
        cache_index_table=oracle["cache"],
        mask_token_id=_MASK,
        **{
            name: want[name]
            for name in (
                "input_ids",
                "positions",
                "rope_positions",
                "cache_write_indices",
                "kv_indptr",
                "kv_indices",
            )
        },
    )
    want["verify_rope_positions"].copy_(
        (want["rope_positions"].view(size, q) + rope_offsets[:, None]).view(-1)
    )
    want["cursors"].copy_(want["start_frontiers"] - prompt_lengths + 1)
    keys = int(want["kv_indptr"][-1])
    for name in got:
        valid = slice(keys) if name == "kv_indices" else slice(None)
        assert torch.equal(got[name][valid], want[name][valid]), name
        assert torch.equal(got[name][valid], block[name][valid]), name
    # Past the CSR prefix the kernels keep what was there.
    assert (got["kv_indices"][keys:] == 0).all()
    assert (block["kv_indices"][keys:] == -7).all()

    # Commit drafts the target accepts through (b * 2) % Q, with a stop in row 1.
    proposals = torch.arange(size * (q - 1), dtype=torch.int32).view(size, q - 1) + 20
    targets = torch.cat([proposals, torch.full((size, 1), 30, dtype=torch.int32)], 1)
    for b in range(size):
        targets[b, (2 * b) % q :] += 50
    targets[min(1, size - 1), 0] = _EOS
    accepted = greedy_accepted(proposals, targets)
    common = {
        "output_ends": torch.tensor([_M, 9, _M, 6][:size]),
        "ignore_eos": torch.tensor([0, 0, 1, 0][:size], dtype=torch.int32),
        "stop_token_ids": torch.tensor([_EOS, 3], dtype=torch.int32),
    }
    record = {
        "tokens": targets.clone() if not greedy else torch.empty(size, q, dtype=torch.int32),
        "accepted": accepted.clone() if not greedy else torch.empty(size, dtype=torch.int32),
        "committed": torch.empty(size, dtype=torch.int32),
        "record_done": torch.empty(size, dtype=torch.int32),
        "k2_lengths": torch.empty(size, dtype=torch.int32),
    }
    commit_block(
        request,
        got["start_frontiers"],
        got["round_done"],
        token_table=kernel["tokens"],
        frontier_table=kernel["frontiers"],
        done_table=kernel["done"],
        block_size=q,
        max_seq_len=_M,
        proposals=proposals if greedy else None,
        target_ids=targets.long().view(-1) if greedy else None,
        **common,
        **record,
    )
    committed, finished = commit_oracle(
        geometry,
        rows=request,
        frontiers=want["start_frontiers"],
        done=want["round_done"],
        accepted=accepted,
        tokens=targets,
        output_ends=common["output_ends"],
        ignore_eos=common["ignore_eos"] != 0,
        stop_token_ids=common["stop_token_ids"],
        token_table=oracle["tokens"],
        frontier_table=oracle["frontiers"],
        done_table=oracle["done"],
    )
    assert torch.equal(record["tokens"], targets)
    assert torch.equal(record["accepted"], accepted.masked_fill(want["round_done"], -1))
    assert torch.equal(record["committed"], committed.int())
    assert torch.equal(record["record_done"], finished.int())
    assert torch.equal(record["k2_lengths"], committed.int())
    assert torch.equal(kernel["frontiers"], oracle["frontiers"])
    assert torch.equal(kernel["done"], oracle["done"])
    # The oracle also writes uncommitted tokens to scratch columns [M, M + Q).
    assert torch.equal(kernel["tokens"][:, :_M], oracle["tokens"][:, :_M])
