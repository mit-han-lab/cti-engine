"""Settle a scheduler's steps from a test's sampled tokens or round records."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from cti.inference import PageMap, RoundCompletion, Sampled
from cti.inference._control._scheduling.request import DecodeStep
from cti.inference._control._scheduling.strategy import TokenSteps

if TYPE_CHECKING:
    from cti.inference._control._cache import CacheUse
    from cti.inference._control._scheduler import InferenceScheduler, ScheduledStep
    from cti.inference._control._scheduling.prefill import PrefillQueue
    from cti.inference._control._scheduling.request import Request, RequestBatch


class ScriptedDecode(TokenSteps):
    """Ordinary decoding whose next step is the forward over ``batch`` a test scripts, or
    none."""

    def __init__(self) -> None:
        self.batch: RequestBatch | None = None

    def plan(
        self,
        running: Sequence[Request],
        queue: PrefillQueue,
        *,
        max_batch_tokens: int,
        max_prompt_tokens: int,
        reserved_size: int,
    ) -> DecodeStep | None:
        return None if self.batch is None else DecodeStep(self.batch, None)


def page_triples(pages: PageMap) -> torch.Tensor:
    """A step's page table changes, one int32 (slot, page index, page base) row each."""

    return torch.frombuffer(bytearray(pages.entries), dtype=torch.int32).view(-1, 3)


def sampled(tokens: Sequence[int], logprobs: Sequence[float] | None = None) -> Sampled:
    """A forward's output: ``tokens`` sampled, one per row."""

    return Sampled(
        array("i", tokens).tobytes(), None if logprobs is None else array("f", logprobs).tobytes()
    )


def round_record(
    tokens: Sequence[Sequence[int]],
    accepted: Sequence[int],
    committed: Sequence[int],
    done: Sequence[bool],
    frontiers: Sequence[int],
    extended: Sampled | None = None,
    logprobs: Sequence[Sequence[float]] | None = None,
) -> RoundCompletion:
    """A round's device record, one row per request, with each target token's
    ``logprobs`` for a full round."""

    flat = [token for row in tokens for token in row]
    return RoundCompletion(
        len(tokens[0]),
        array("i", flat).tobytes(),
        array("i", accepted).tobytes(),
        array("i", committed).tobytes(),
        bytes(map(bool, done)),
        array("q", frontiers).tobytes(),
        extended,
        None if logprobs is None else sampled(flat, [value for row in logprobs for value in row]),
    )


def settle_forward(
    scheduler: InferenceScheduler,
    step: ScheduledStep,
    sampled_token_ids: torch.Tensor,
    selected_logprobs: torch.Tensor | None = None,
):
    """Settle a forward-only step whose forward sampled ``sampled_token_ids``."""

    logprobs = None if selected_logprobs is None else selected_logprobs.tolist()
    return scheduler.settle_step(step, sampled(sampled_token_ids.tolist(), logprobs))


def request_ids(step: ScheduledStep) -> tuple[int, ...]:
    """The uids of a planned step's rows, in order."""

    return tuple(row.request.uid for row in step.rows)


def reservations(step: ScheduledStep) -> tuple[CacheUse, ...]:
    """The cache uses of a planned step's rows, in order."""

    return tuple(row.use for row in step.rows)
