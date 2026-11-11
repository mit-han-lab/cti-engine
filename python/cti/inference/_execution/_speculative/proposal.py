"""How a DFlash round proposes its drafts from the draft's hidden rows, and the q its sampled
verification compares them under (docs/dflash.md, K1 sampling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from cti.models import TokenHead
    from ctikernels.sampling import RequestLocalChainSampling


@dataclass(frozen=True, slots=True)
class ChainRows:
    """K1's inputs for a round's rows: temperatures, the rows whose seeds they read, the
    seed words, the RNG cursors and the borrowed scratch."""

    temperatures: torch.Tensor
    request_indices: torch.Tensor
    seeds: torch.Tensor
    cursors: torch.Tensor
    workspace: torch.Tensor


class CandidateHead(Protocol):
    """What a chain walk reads of the target's LM head: rows' shard logits and their merged
    global top-k (FP32 logits and int64 IDs, by ascending ID)."""

    def local_logits(self, x: torch.Tensor) -> torch.Tensor: ...

    def gather_topk(self, local: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]: ...


class CandidateScores(Protocol):
    """What a chain walk reads of the draft's candidate selector: its ``top_k`` and a block's
    FP32 ``[B, G, K, K]`` lattice, each position's candidate scores after each predecessor
    index, of hidden rows ``[B, G, H]``, candidates (ascending IDs and their logits,
    ``[B, G, K]``) and anchors ``[B]``."""

    top_k: int

    def lattice(
        self, hidden: torch.Tensor, ids: torch.Tensor, unary: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor: ...


class Proposal(Protocol):
    def propose(
        self,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        out: torch.Tensor,
        rows: ChainRows,
    ) -> None:
        """Write the int32 ``[B, G]`` proposals ``out`` from the draft's final hidden rows
        ``[B, G, H]`` after each row's ``anchors``, drawn under ``rows``' laws (a
        temperature-0 row greedily). Proposals never follow a round's logits processing,
        as in vLLM and SGLang: verification alone applies it."""

    def verify(
        self,
        probabilities: torch.Tensor,
        drafts: torch.Tensor,
        rows: ChainRows,
        *,
        out_tokens: torch.Tensor,
        accepted: torch.Tensor,
    ) -> None:
        """Draw a sampled round's verdict from target ``probabilities`` ``[B, G + 1, V]``."""


class ArgmaxProposal:
    """DFlash v1: each position's target-head argmax, which sampled verification reads as a
    point mass."""

    def __init__(self, head: TokenHead, sampling: RequestLocalChainSampling) -> None:
        self._head = head
        self._sampling = sampling

    def propose(
        self,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        out: torch.Tensor,
        rows: ChainRows,
    ) -> None:
        out.copy_(self._head.argmax(hidden.flatten(0, 1)).view(out.shape))

    def verify(
        self,
        probabilities: torch.Tensor,
        drafts: torch.Tensor,
        rows: ChainRows,
        *,
        out_tokens: torch.Tensor,
        accepted: torch.Tensor,
    ) -> None:
        self._sampling.verify_point_mass(
            probabilities,
            drafts,
            rows.temperatures,
            request_indices=rows.request_indices,
            seeds=rows.seeds,
            cursors=rows.cursors,
            workspace=rows.workspace,
            out_tokens=out_tokens,
            accepted_counts=accepted,
        )


class ChainProposal:
    """DFlash2: each position's top-K target-head candidates, scored by the draft's selector
    given the token chosen before it, walked from the anchor. A temperature-0 row takes the
    best candidate (the smaller ID on ties); a sampled row draws from its candidates'
    temperature softmax, whose FP32 q sampled verification compares the draft under.

    Every replica walks the same chain without exchanging it: the candidates are global, the
    selector is whole in each and the draws replay the request's counter-based RNG.
    """

    def __init__(
        self,
        selector: CandidateScores,
        head: CandidateHead,
        sampling: RequestLocalChainSampling,
        *,
        max_rows: int,
        positions: int,
        device: torch.device,
    ) -> None:
        self._selector, self._head, self._sampling = selector, head, sampling
        k = selector.top_k
        # The last round's candidates and their q, at fixed addresses for its verify.
        self._ids = torch.zeros((max_rows, positions, k), dtype=torch.int32, device=device)
        self._q = torch.zeros((max_rows, positions, k), dtype=torch.float32, device=device)

    def propose(
        self,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        out: torch.Tensor,
        rows: ChainRows,
    ) -> None:
        size, positions = out.shape
        k, head = self._selector.top_k, self._head
        unary, ids = head.gather_topk(head.local_logits(hidden.flatten(0, 1)), k)
        ids = self._ids[:size].copy_(ids.view(size, positions, k))
        lattice = self._selector.lattice(hidden, ids, unary.view(size, positions, k), anchors)
        # Each row's chosen candidate index at the previous position (any at position 0,
        # whose predecessor is the anchor).
        index = torch.zeros((size, 1, 1), dtype=torch.int64, device=out.device)
        for position in range(positions):
            scores = lattice[:, position].gather(1, index.expand(-1, 1, k)).view(size, k)
            candidates = ids[:, position].contiguous()
            chosen = torch.empty(size, dtype=torch.int32, device=out.device)
            q = torch.empty((size, k), dtype=torch.float32, device=out.device)
            self._sampling.sample_candidates(
                scores,
                candidates,
                rows.temperatures,
                position,
                request_indices=rows.request_indices,
                seeds=rows.seeds,
                cursors=rows.cursors,
                workspace=rows.workspace,
                out_ids=chosen,
                out_q=q,
            )
            self._q[:size, position].copy_(q)
            # The drawn ID's place among the ascending candidates.
            index = torch.searchsorted(candidates, chosen[:, None]).view(size, 1, 1)
            out[:, position].copy_(chosen)

    def verify(
        self,
        probabilities: torch.Tensor,
        drafts: torch.Tensor,
        rows: ChainRows,
        *,
        out_tokens: torch.Tensor,
        accepted: torch.Tensor,
    ) -> None:
        size = len(drafts)
        self._sampling.verify_sparse(
            probabilities,
            drafts,
            self._ids[:size],
            self._q[:size],
            rows.temperatures,
            request_indices=rows.request_indices,
            seeds=rows.seeds,
            cursors=rows.cursors,
            workspace=rows.workspace,
            out_tokens=out_tokens,
            accepted_counts=accepted,
        )
