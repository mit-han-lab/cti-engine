"""The primary rank seeds every unseeded request where requests enter."""

from __future__ import annotations

import torch

from cti.inference._control._seeds import RequestSeeds
from cti.protocol import AbortBackendMsg, SamplingParams, UserMsg


def _message(uid: int, seed: int | None = None) -> UserMsg:
    return UserMsg(
        uid,
        torch.tensor([4, 5], dtype=torch.int32),
        SamplingParams(max_new_tokens=1024, temperature=0, seed=seed).snapshot(),
    )


def test_unseeded_requests_get_distinct_seeds_and_seeded_ones_keep_theirs() -> None:
    seeds = RequestSeeds()
    messages = [_message(0), _message(1, seed=7), AbortBackendMsg(1)]

    first, explicit, abort = seeds.stamp(messages)
    (again,) = seeds.stamp([_message(0)])

    assert explicit.sampling.seed == 7
    assert isinstance(abort, AbortBackendMsg)
    # A later request with a reused ID still gets a new seed, in SamplingParams' range.
    first_seed, again_seed = first.sampling.seed, again.sampling.seed
    assert 0 <= first_seed < 2**63 and 0 <= again_seed < 2**63
    assert first_seed != again_seed
    assert messages[0].sampling.seed is None
