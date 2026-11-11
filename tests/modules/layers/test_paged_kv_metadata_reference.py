"""The CPU reference of ``paged_kv_metadata`` writes what FlashInfer's host index builders
wrote: each request's ``token_slots`` concatenated, at their cumulative counts."""

from __future__ import annotations

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from cti.attention.paged import token_slots
from tests.support import reference_ops

_WIDTH = 6


def _build(pages: torch.Tensor, rows, lengths, *, page_size, granularity="token", capacity=64):
    outputs = {
        "kv_indptr": torch.full((len(rows) + 1,), -1, dtype=torch.int32),
        "kv_indices": torch.full((capacity,), -1, dtype=torch.int32),
        "kv_last_page_len": torch.full((len(rows),), -1, dtype=torch.int32),
    }
    reference_ops.paged_kv_metadata(
        pages,
        torch.tensor(rows, dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
        page_size=page_size,
        granularity=granularity,
        max_kv_len=_WIDTH * page_size,
        **outputs,
    )
    return outputs


@pytest.mark.parametrize("page_size", [1, 16])
def test_token_metadata_matches_the_host_builders(page_size: int) -> None:
    generator = torch.Generator().manual_seed(0)
    pages = torch.randperm(64, generator=generator)[: 4 * _WIDTH].to(torch.int32).view(4, -1)
    table = SimpleNamespace(page_size=page_size, pages=pages)
    # A full row, an empty request, a partial page and one token; row 3 read twice.
    rows, lengths = [3, 0, 3, 1], [_WIDTH * page_size, 0, 5, 1]
    capacity = sum(lengths) + 3

    out = _build(pages, rows, lengths, page_size=page_size, capacity=capacity)

    expected = torch.cat(
        [token_slots(table, row, 0, n) for row, n in zip(rows, lengths, strict=True)]
    )
    assert out["kv_indptr"].tolist() == [0, *accumulate(lengths)]
    assert out["kv_indices"].tolist() == [*expected.tolist(), -1, -1, -1]
    # Every last page holds one token; an empty request has none.
    assert out["kv_last_page_len"].tolist() == [1, 0, 1, 1]

    # Entries past the indices' capacity are dropped.
    short = _build(pages, rows, lengths, page_size=page_size, capacity=4)
    assert short["kv_indices"].tolist() == expected[:4].tolist()
    assert short["kv_indptr"].tolist() == out["kv_indptr"].tolist()


def test_page_metadata_lists_page_ids() -> None:
    pages = torch.arange(2 * _WIDTH, dtype=torch.int32).view(2, -1)

    out = _build(pages, [1, 0], [17, 0], page_size=16, granularity="page")

    assert out["kv_indptr"].tolist() == [0, 2, 2]
    assert out["kv_indices"][:3].tolist() == [6, 7, -1]
    assert out["kv_last_page_len"].tolist() == [1, 0]
