"""The LM head's greedy and top-k views: one tie rule for one rank and for vocabulary
shards."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch.nn.functional as F

from cti.layers import ParallelLMHead
from tests.support.tp_shards import site


class _Gather:
    """Stands in for one rank of a TP group whose ranks run on threads."""

    def __init__(self, parts: list[torch.Tensor | None], barrier: threading.Barrier, rank: int):
        self.rank, self.size = rank, len(parts)
        self._parts = parts
        self._barrier = barrier

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        self._parts[self.rank] = x
        self._barrier.wait()
        result = torch.cat(self._parts)
        self._barrier.wait()
        return result


def _sharded_argmax(weight: torch.Tensor, rows: torch.Tensor, size: int) -> list[list[int]]:
    return [ids.tolist() for ids in _sharded(weight, rows, size, ParallelLMHead.argmax)]


def _sharded(weight: torch.Tensor, rows: torch.Tensor, size: int, method) -> list:
    """``method(head, rows)`` on every rank of ``size`` vocabulary shards of ``weight``."""

    parts: list[torch.Tensor | None] = [None] * size
    barrier = threading.Barrier(size)
    heads = []
    for rank in range(size):
        group = _Gather(parts, barrier, rank)
        head = ParallelLMHead(
            *weight.shape,
            site=site(group=group),
        )
        start, count = head.vocab_range
        # Padding rows past the vocabulary would win if the head read them.
        head.weight = torch.full(
            (head.num_embeddings_tp, weight.shape[1]), 100.0, dtype=weight.dtype
        )
        head.weight[:count] = weight[start : start + count]
        heads.append(head)
    with ThreadPoolExecutor(size) as pool:
        return list(pool.map(lambda head: method(head, rows), heads))


_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def test_one_rank_breaks_ties_to_the_smallest_id() -> None:
    head = ParallelLMHead(4, 2, site=site())
    head.weight = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    rows = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])

    # Logits [0, 2, 2, 1], [3, 0, 0, 1.5] and [1, 1, 1, 1].
    assert head.argmax(rows).tolist() == [1, 0, 0]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab, size", [(7, 2), (7, 3), (4, 3)])
def test_shards_agree_with_the_full_vocabulary(vocab: int, size: int, dtype: torch.dtype) -> None:
    # Small integers keep every logit exact, so ties are real in every dtype.
    generator = torch.Generator().manual_seed(vocab * size)
    weight = torch.randint(-4, 5, (vocab, 3), generator=generator).to(dtype)
    weight[vocab - 1] = weight[0]
    weight[2] = weight[vocab - 2]
    rows = torch.randint(-3, 4, (6, 3), generator=generator).to(dtype)
    # Rows whose maxima tie across shard edges.
    rows = torch.cat([rows, weight[0:1], weight[2:3]])
    expected = F.linear(rows, weight).argmax(-1).tolist()

    assert _sharded_argmax(weight, rows, size) == [expected] * size


@pytest.mark.parametrize("dtype", _DTYPES)
def test_negative_logits_keep_float_order(dtype: torch.dtype) -> None:
    weight = torch.tensor([[-3, -2], [-1, -4], [-2, -6], [-1, -1], [-5, -3]], dtype=dtype)
    rows = torch.eye(2, dtype=dtype)
    assert bool((F.linear(rows, weight) < 0).all())

    # IDs 0-2 on one shard, 3-4 on the other: a -1 tie, then -1 beating -2.
    assert _sharded_argmax(weight, rows, 2) == [[1, 3]] * 2


def test_negative_zero_ties_with_zero() -> None:
    weight = torch.tensor([[-1.0], [0.0], [1.0], [-2.0]])
    rows = torch.tensor([[0.0], [-0.0]])
    # FP32 linear keeps the signs that FP16/BF16 accumulation loses.
    signs = torch.signbit(F.linear(rows, weight))
    assert bool(signs.any()) and not bool(signs.all())

    # Every logit of both rows is a signed zero; ID 0 wins on its shard and overall.
    assert _sharded_argmax(weight, rows, 2) == [[0, 0]] * 2


def _topk(head: ParallelLMHead, rows: torch.Tensor, k: int = 3) -> tuple:
    values, ids = head.gather_topk(head.local_logits(rows), k)
    return values.tolist(), ids.tolist()


def _expected_topk(logits: torch.Tensor, k: int = 3) -> tuple:
    """Each row's k largest logits, the smaller ID first among equal ones, by ascending ID."""

    values, ids = [], []
    for row in logits.float().tolist():
        best = sorted(sorted(range(len(row)), key=lambda i: (-row[i], i))[:k])
        values.append([row[i] for i in best])
        ids.append(best)
    return values, ids


def test_one_rank_takes_the_top_k_by_logit_then_smaller_id() -> None:
    head = ParallelLMHead(5, 2, site=site())
    head.weight = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
    rows = torch.tensor([[2.0, 0.0], [0.0, -3.0]])

    # Logits [0, 2, 2, 1, 2] keep IDs 1, 2, 4; [-3, 0, 0, -1.5, 0] keep 1, 2, 4 too.
    assert _topk(head, rows) == ([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]], [[1, 2, 4], [1, 2, 4]])
    # Logits [2, 1, 1, 1.5, 1]: the tie at 1 keeps ID 1, and IDs ascend.
    assert _topk(head, torch.tensor([[1.0, 2.0]])) == ([[2.0, 1.0, 1.5]], [[0, 1, 3]])


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab, size", [(7, 2), (7, 3), (4, 3), (16, 2)])
def test_sharded_top_k_agrees_with_the_full_vocabulary(
    vocab: int, size: int, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(vocab * size + 1)
    weight = torch.randint(-4, 5, (vocab, 3), generator=generator).to(dtype)
    weight[vocab - 1] = weight[0]
    rows = torch.cat([torch.randint(-3, 4, (6, 3), generator=generator).to(dtype), weight[:1]])
    expected = _expected_topk(F.linear(rows, weight))

    assert _sharded(weight, rows, size, _topk) == [expected] * size
