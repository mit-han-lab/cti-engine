from __future__ import annotations

import pytest
import torch

from cti.inference._execution._sampling import (
    BatchSamplingArgs,
    LegacySamplingLaw,
    LogitProcessing,
    LogprobRows,
    Sampler,
)

pytestmark = pytest.mark.checkpoints()


def test_request_local_sampling_is_invariant_to_regrouping() -> None:
    torch.manual_seed(20260722)
    batch_size = 16
    vocab_size = 257
    logits = torch.randn((batch_size, vocab_size), device="cuda", dtype=torch.bfloat16)
    temperatures = torch.full((batch_size,), 0.7, device="cuda", dtype=torch.float32)
    seeds = tuple(range(1000, 1000 + batch_size))
    positions = tuple(index % 5 for index in range(batch_size))
    sampler = Sampler(
        torch.device("cuda"),
        vocab_size,
        law=LegacySamplingLaw(None),
        processing=LogitProcessing(torch.zeros(1, 1, dtype=torch.int32, device="cuda"), vocab_size),
        batch_capacity=len(logits),
        logits_dtype=logits.dtype,
    )

    def sample(
        target_logits: torch.Tensor,
        target_temperatures: torch.Tensor,
        target_seeds: tuple[int, ...],
        target_positions: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Processed: each row's own probability; a raw row's log-softmax reduces over the
        # vocabulary as the batch shape picks.
        rows = torch.ones(len(target_seeds), dtype=torch.bool, device="cuda")
        sampled = sampler.sample(
            target_logits,
            BatchSamplingArgs(
                temperatures=target_temperatures,
                seeds=target_seeds,
                positions=target_positions,
                logprobs=LogprobRows(rows, rows),
                redraw=tuple(range(len(target_seeds))),
            ),
        )
        return sampled.token_ids, sampled.logprobs

    first_tokens, first_logprobs = sample(logits, temperatures, seeds, positions)
    repeated_tokens, repeated_logprobs = sample(logits, temperatures, seeds, positions)
    permutation = torch.tensor(
        [7, 2, 13, 0, 15, 4, 9, 1, 11, 6, 14, 3, 10, 5, 12, 8], device="cuda"
    )
    inverse = torch.argsort(permutation)
    order = permutation.cpu().tolist()
    permuted_tokens, permuted_logprobs = sample(
        logits[permutation],
        temperatures[permutation],
        tuple(seeds[index] for index in order),
        tuple(positions[index] for index in order),
    )
    split = tuple(
        sample(logits[start:end], temperatures[start:end], seeds[start:end], positions[start:end])
        for start, end in ((0, 5), (5, 11), (11, batch_size))
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(first_tokens, repeated_tokens, rtol=0, atol=0)
    torch.testing.assert_close(first_logprobs, repeated_logprobs, rtol=0, atol=0)
    torch.testing.assert_close(first_tokens, permuted_tokens[inverse], rtol=0, atol=0)
    torch.testing.assert_close(first_logprobs, permuted_logprobs[inverse], rtol=0, atol=0)
    torch.testing.assert_close(first_tokens, torch.cat([item[0] for item in split]), rtol=0, atol=0)
    torch.testing.assert_close(
        first_logprobs, torch.cat([item[1] for item in split]), rtol=0, atol=0
    )
    oracle = (
        torch.log_softmax(logits.float() / temperatures[:, None], dim=-1)
        .gather(
            1,
            first_tokens.long()[:, None],
        )
        .squeeze(1)
    )
    torch.testing.assert_close(first_logprobs, oracle, rtol=2e-5, atol=2e-5)


def test_a_mixed_batch_draws_its_seeded_rows_as_an_all_seeded_batch_does() -> None:
    torch.manual_seed(20261016)
    batch_size = 4
    vocab_size = 4096
    logits = torch.randn((batch_size, vocab_size), device="cuda", dtype=torch.bfloat16)
    temperatures = torch.full((batch_size,), 0.7, device="cuda", dtype=torch.float32)
    seeds = (101, 202, 303, 404)
    positions = (0, 3, 5, 9)
    sampler = Sampler(
        torch.device("cuda"),
        vocab_size,
        law=LegacySamplingLaw(None),
        processing=LogitProcessing(torch.zeros(1, 1, dtype=torch.int32, device="cuda"), vocab_size),
        batch_capacity=len(logits),
        logits_dtype=logits.dtype,
    )
    seeded_rows = [0, 2]

    def sample(row_seeds: tuple[int | None, ...], filters: dict) -> torch.Tensor:
        args = BatchSamplingArgs(temperatures, seeds=row_seeds, positions=positions, **filters)
        return sampler.sample(logits, args).token_ids

    for top_k, top_p in ((None, None), (20, 0.8)):
        filters = {
            "top_k": None if top_k is None else torch.full((batch_size,), top_k, device="cuda"),
            "top_p": None if top_p is None else torch.full((batch_size,), top_p, device="cuda"),
        }

        all_seeded = sample(seeds, filters)
        # The unseeded rows draw from the shared generator; the seeded ones from their streams.
        mixed = sample(
            tuple(seeds[row] if row in seeded_rows else None for row in range(batch_size)), filters
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(mixed[seeded_rows], all_seeded[seeded_rows], rtol=0, atol=0)
