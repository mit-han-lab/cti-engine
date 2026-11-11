# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors

"""The device buffers DFlash rounds work in, allocated before capacity planning so it counts
them as the models' memory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from cti.forward import RoundShape

from ._records import RoundArgs, RoundLogprobs, RoundRecord


@dataclass(frozen=True, slots=True)
class RoundScratch:
    """Every round's fixed-address buffers, for up to B rows of Q tokens: the target's taps
    (one row per forward token), the staged per-row inputs (``args``), the device record
    and log-probabilities, the rows K1 reads seeds by, the recurrent rows' K2 slots and
    lengths, the block rows' input IDs, positions and rotary rows, the rows' done flags at a
    round's start, the sampling cursors, the proposals, whether replicas diverged (under
    checks), K1's ``[B, Q, V]`` probabilities and workspace, the target's hidden rows of the
    block (which its graph hands the eager tail), the M-RoPE verify rows and the stop token
    IDs."""

    taps: torch.Tensor
    args: RoundArgs
    results: RoundRecord
    logprobs: RoundLogprobs
    rows: torch.Tensor
    k2_slots: torch.Tensor
    k2_lengths: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    rope_rows: torch.Tensor
    round_done: torch.Tensor
    cursors: torch.Tensor
    proposals: torch.Tensor
    divergent: torch.Tensor
    probabilities: torch.Tensor
    workspace: torch.Tensor
    verify_hidden: torch.Tensor
    verify_rope: torch.Tensor
    stop_token_ids: torch.Tensor

    @classmethod
    def allocate(
        cls,
        rounds: RoundShape,
        taps: tuple[int, int],
        hidden_size: int,
        vocab_size: int,
        stop_token_ids: Sequence[int],
        *,
        workspace_bytes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> RoundScratch:
        """The buffers of ``rounds`` of a target of ``hidden_size`` over a ``vocab_size``
        vocabulary; ``taps`` is the taps' ``(rows, width)``, ``workspace_bytes`` K1's
        workspace."""

        b, q, v = rounds.max_rows, rounds.width, vocab_size

        def int32(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.int32, device=device)

        def int64(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.int64, device=device)

        return cls(
            taps=torch.empty(taps, dtype=dtype, device=device),
            args=RoundArgs.allocate(b, device=device),
            results=RoundRecord.allocate(b, q, device=device),
            logprobs=RoundLogprobs.allocate(b, q, device=device),
            # Sampling reads each row's seed by its row number.
            rows=torch.arange(b, dtype=torch.int32, device=device),
            k2_slots=int32(b),
            k2_lengths=int32(b),
            input_ids=int32(b * q),
            positions=int64(b * q),
            rope_rows=int64(b * q),
            round_done=torch.zeros(b, dtype=torch.bool, device=device),
            cursors=int64(b),
            proposals=int32(b, q - 1),
            divergent=torch.zeros((), dtype=torch.bool, device=device),
            probabilities=torch.empty((b, q, v), dtype=torch.float32, device=device),
            workspace=torch.empty(workspace_bytes, dtype=torch.uint8, device=device),
            verify_hidden=torch.empty((b * q, hidden_size), dtype=dtype, device=device),
            # M-RoPE targets verify at sequence positions plus each request's rotary offset,
            # one row that every axis reads.
            verify_rope=int64(b * q),
            stop_token_ids=torch.tensor(stop_token_ids, dtype=torch.int32, device=device),
        )
