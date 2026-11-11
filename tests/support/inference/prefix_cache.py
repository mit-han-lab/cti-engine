"""Small real-radix CPU harness, without an unrelated comparison AOT build."""

import pytest
import torch

import ctikernels
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.scheduling import settle_forward


def compare_keys(left: torch.Tensor, right: torch.Tensor) -> int:
    limit = min(len(left), len(right))
    mismatches = (left[:limit] != right[:limit]).nonzero()
    return limit if len(mismatches) == 0 else int(mismatches[0].item())


@pytest.fixture(autouse=True)
def reference_radix_comparison(monkeypatch):
    monkeypatch.setattr(ctikernels, "fast_compare_key", compare_keys)


def make_cache(rows=3, pages=2048):
    return new_cache_manager(
        pages,
        1,
        torch.full((8, 512), -1, dtype=torch.int32),
        reuse=True,
        snapshot_rows=range(8, 8 + rows),
        checks=True,
    )


def _prompt(group: int, length: int) -> list[int]:
    return [group * 1000 + position for position in range(length)]


def sibling_phases() -> list[list[tuple[list[int], int]]]:
    """Three siblings of four 193-token prompts, all at once, then four more prompts, whose
    exports reuse the snapshot rows: (prompt, output tokens)."""
    return [
        [(_prompt(group, 193), 2) for _ in range(3) for group in range(4)],
        [(_prompt(group, 193), 2) for group in range(4, 8)],
    ]


def planning_trace(pages, *, phases=None, rows=2):
    """Serve each phase's requests to quiescence without device work, every sampled token
    0: each plan's rows and the KV tokens evicted."""
    from collections import deque
    from unittest.mock import Mock

    import cti.inference._control._scheduler as scheduler_module
    from cti.inference._control._scheduler import InferenceScheduler
    from cti.inference._control._scheduling.encoder_cache import EncoderCache
    from cti.inference._control._scheduling.strategy import SchedulingLimits, TokenSteps
    from cti.protocol import SamplingParams, UserMsg

    patch = pytest.MonkeyPatch()
    patch.setattr(scheduler_module, "logger", Mock())
    patch.setattr(ctikernels, "fast_compare_key", compare_keys)
    cache = make_cache(rows, pages)
    evicted = []

    def evict(size, evict=cache.prefix_cache.evict):
        evicted.append(size)
        return evict(size)

    patch.setattr(cache.prefix_cache, "evict", evict)
    scheduler = InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(8, 512, 512, 512),
        cache_manager=cache,
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
    )
    trace, uid = [], 0
    for phase in sibling_phases() if phases is None else phases:
        for prompt, max_tokens in phase:
            scheduler.handle_message(
                UserMsg(
                    uid=uid,
                    input_ids=torch.tensor(prompt, dtype=torch.int32),
                    sampling=SamplingParams(
                        temperature=0, max_new_tokens=max_tokens, ignore_eos=True
                    ).snapshot(),
                )
            )
            uid += 1
        pending = deque()
        while scheduler.has_runnable_work or pending:
            plan = scheduler.plan_next()
            if plan is not None:
                trace.append(
                    [
                        (
                            e.request_index,
                            e.kv_prefix_len,
                            e.sequence_len,
                            e.restore_row,
                            [(c.position, c.row) for c in e.captures],
                        )
                        for e in plan.op.rows
                    ]
                )
                scheduler.mark_submitted(plan)
                pending.append(plan)
            if len(pending) >= 2 or plan is None:
                claim = pending.popleft()
                settle_forward(scheduler, claim, torch.zeros(len(claim.rows), dtype=torch.int32))
    scheduler.require_quiescent()
    patch.undo()
    return trace, sum(evicted)


if __name__ == "__main__":
    import json

    # Roomy, then tight enough to evict KV prefixes.
    print(json.dumps([planning_trace(4096), planning_trace(640)]))
