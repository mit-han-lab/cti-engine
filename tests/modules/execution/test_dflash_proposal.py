"""A DFlash2 round's proposal step: the chain walk over its candidate lattice, against
the independent reference in ``tests.support.dflash2_reference``."""

from __future__ import annotations

from unittest.mock import Mock

import torch

from cti.inference._execution._speculative.proposal import (
    ArgmaxProposal,
    ChainProposal,
    ChainRows,
)
from cti.layers import ParallelLMHead
from cti.models import TokenHead
from cti.models.dflash.config import DFlash2Extension
from cti.models.dflash.dflash2 import CandidateSelector
from ctikernels.sampling import RequestLocalChainSampling
from tests.support import dflash2_reference
from tests.support.tp_shards import site

_CPU = torch.device("cpu")


def _chain_inputs(rows: int = 2):
    """A selector (width-8 codes) and target head over 300 tokens with random weights, and
    hidden rows whose candidate logits and lattice scores tie nowhere."""

    generator = torch.Generator().manual_seed(3)
    selector = CandidateSelector(64, 300, DFlash2Extension(2, 16, 8, 16), site=site())
    selector.load_state_dict(
        {
            name: torch.randn(tensor.shape, generator=generator)
            for name, tensor in selector.state_dict().items()
        }
    )
    lm_head = ParallelLMHead(300, 64, site=site())
    lm_head.weight = torch.randn(300, 64, generator=generator)
    head = TokenHead(lm_head, lm_head)
    hidden = torch.randn(rows, 7, 64, generator=generator)
    anchors = torch.tensor([5, 290][:rows], dtype=torch.int32)
    return selector, head, hidden, anchors


def _reference_scores(selector, head, hidden: torch.Tensor, anchor: int, steer=None) -> tuple:
    logits = head.local_logits(hidden).double()
    if steer is not None:
        logits = steer(logits[None]).double()[0]
    ids, unary = dflash2_reference.top_candidates(logits, 16)
    weights = selector.state_dict()
    return ids, dflash2_reference.lattice(weights, hidden, ids, unary, anchor)


def _rows(temperatures: tuple[float, ...]) -> ChainRows:
    """K1's inputs for rows at ``temperatures``, with fixed seeds and cursors."""

    return ChainRows(
        torch.tensor(temperatures),
        torch.arange(len(temperatures), dtype=torch.int32),
        torch.tensor([[11 + row, 0] for row in range(len(temperatures))], dtype=torch.uint32),
        torch.arange(3, 3 + len(temperatures)),
        torch.empty(0, dtype=torch.uint8),
    )


def test_a_greedy_chain_walks_the_lattice_argmax_from_the_anchor() -> None:
    selector, head, hidden, anchors = _chain_inputs()
    sampling = RequestLocalChainSampling("cpu", 8, 300)
    proposal = ChainProposal(selector, head, sampling, max_rows=4, positions=7, device=_CPU)
    out = torch.zeros(2, 7, dtype=torch.int32)

    proposal.propose(hidden, anchors, out, _rows((0.0, 0.0)))

    for row in range(2):
        ids, scores = _reference_scores(selector, head, hidden[row], int(anchors[row]))
        chain = dflash2_reference.greedy_chain(scores)
        assert out[row].tolist() == [int(ids[e, c]) for e, c in enumerate(chain)]
        assert proposal._ids[row].tolist() == ids.tolist()


def test_a_sampled_chain_draws_each_position_under_its_conditional_q() -> None:
    selector, head, hidden, anchors = _chain_inputs()
    sampling = RequestLocalChainSampling("cpu", 8, 300)
    proposal = ChainProposal(selector, head, sampling, max_rows=4, positions=7, device=_CPU)
    rows = _rows((0.7, 0.0))
    out = torch.zeros(2, 7, dtype=torch.int32)

    proposal.propose(hidden, anchors, out, rows)

    # The sampled row: each draw is a candidate, q is the temperature softmax after the
    # token drawn before it, in ascending ID order.
    ids, scores = _reference_scores(selector, head, hidden[0], int(anchors[0]))
    chain = [ids[e].tolist().index(token) for e, token in enumerate(out[0].tolist())]
    q = dflash2_reference.chain_q(scores, chain, 0.7)
    torch.testing.assert_close(proposal._q[0].double(), q, rtol=1e-5, atol=1e-6)
    # The temperature-0 row walks the argmax.
    ids, scores = _reference_scores(selector, head, hidden[1], int(anchors[1]))
    greedy = dflash2_reference.greedy_chain(scores)
    assert out[1].tolist() == [int(ids[e, c]) for e, c in enumerate(greedy)]
    # The same seeds and cursors draw the same chain again: every rank walks it alike.
    again = torch.zeros_like(out)
    proposal.propose(hidden, anchors, again, rows)
    assert torch.equal(again, out)


def test_sampled_verification_compares_the_drafts_under_the_chains_q() -> None:
    selector, head, hidden, anchors = _chain_inputs()
    sampling = RequestLocalChainSampling("cpu", 8, 300)
    sampling.verify_sparse = Mock()
    proposal = ChainProposal(selector, head, sampling, max_rows=4, positions=7, device=_CPU)
    drafts = torch.zeros(2, 7, dtype=torch.int32)
    proposal.propose(hidden, anchors, drafts, _rows((0.7, 0.7)))
    rows = Mock()
    tokens, accepted = torch.zeros(2, 8, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)

    proposal.verify(torch.ones(2, 8, 300), drafts, rows, out_tokens=tokens, accepted=accepted)

    (call,) = sampling.verify_sparse.call_args_list
    _, verified, ids, q, _ = call.args
    assert verified is drafts
    assert torch.equal(ids, proposal._ids[:2]) and torch.equal(q, proposal._q[:2])
    assert call.kwargs["out_tokens"] is tokens and call.kwargs["accepted_counts"] is accepted


def test_a_v1_draft_takes_the_heads_argmax_under_any_law() -> None:
    _, head, hidden, anchors = _chain_inputs()
    proposal = ArgmaxProposal(head, Mock())
    out = torch.zeros(2, 7, dtype=torch.int32)

    proposal.propose(hidden, anchors, out, _rows((0.7, 0.0)))

    assert torch.equal(out, head.argmax(hidden.flatten(0, 1)).view(2, 7).int())
