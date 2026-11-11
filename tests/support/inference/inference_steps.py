"""A real scheduler or InferenceEngine over a CPU cache, each step settled from the tokens
a test samples."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from cti.inference import (
    Forward,
    GraphStats,
    RequestTable,
    Round,
    Sampled,
    ServingLimits,
)
from cti.inference._control._cache import CacheManager
from cti.inference._control._engine import InferenceEngine
from cti.inference._control._scheduler import InferenceScheduler, ScheduledStep
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.strategy import (
    DecodeStrategy,
    SchedulingLimits,
    TokenSteps,
)
from cti.kvcache import StateCapacity
from cti.protocol import DetokenizeMsg, SamplingParams, UserMsg
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.scheduling import sampled

TOKEN = 7  # what a step samples unless a test says otherwise


def user_msg(
    uid: int, prompt: Sequence[int], *, max_tokens: int = 4, ignore_eos: bool = True
) -> UserMsg:
    return UserMsg(
        uid=uid,
        input_ids=torch.tensor(prompt, dtype=torch.int32),
        sampling=SamplingParams(
            temperature=0, max_new_tokens=max_tokens, ignore_eos=ignore_eos
        ).snapshot(),
    )


def cache_manager(
    *, max_running: int = 4, max_seq_len: int = 64, num_pages: int = 64, reuse: bool = False
) -> CacheManager:
    """A page-size-1 CPU cache for ``scheduler``'s request slots, which a test may observe."""

    return new_cache_manager(
        num_pages,
        1,
        torch.full((max_running, max_seq_len), -1, dtype=torch.int32),
        reuse=reuse,
        checks=True,
    )


def scheduler(
    *,
    max_running: int = 4,
    max_batch_tokens: int = 16,
    max_prompt_tokens: int | None = None,
    max_seq_len: int = 64,
    num_pages: int = 64,
    reuse: bool = False,
    stop_token_ids: tuple[int, ...] = (2,),
    decode: DecodeStrategy | None = None,
    cache: CacheManager | None = None,
    model_revision: Callable[[], int] = lambda: 0,
) -> InferenceScheduler:
    """A scheduler over ``cache``, by default a new ``cache_manager``."""

    if cache is None:
        cache = cache_manager(
            max_running=max_running, max_seq_len=max_seq_len, num_pages=num_pages, reuse=reuse
        )
    return InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(
            max_running, max_batch_tokens, max_seq_len, max_prompt_tokens or max_batch_tokens
        ),
        cache_manager=cache,
        stop_token_ids=stop_token_ids,
        vocab_size=1 << 20,
        decode=TokenSteps() if decode is None else decode,
        encoder_cache=EncoderCache(0),
        model_revision=model_revision,
    )


def submit(scheduler: InferenceScheduler) -> ScheduledStep:
    """Plan the next step and mark it submitted, as the engine does on launching it."""

    step = scheduler.plan_next()
    assert step is not None
    scheduler.mark_submitted(step)
    return step


def settle(
    scheduler: InferenceScheduler, step: ScheduledStep, tokens: int | Sequence[int] = TOKEN
) -> list[DetokenizeMsg]:
    """Settle ``step`` from ``tokens`` (one per row, or one for every row); its replies."""

    if isinstance(tokens, int):
        tokens = [tokens] * len(step.rows)
    outcome = scheduler.settle_step(step, sampled(tokens))
    return outcome.replies


class _Event:
    def __init__(self, executor: FakeExecutor, index: int) -> None:
        self._executor, self._index = executor, index

    def synchronize(self) -> None:
        self._executor.events.append(f"sync:{self._index}")
        if self._executor.completion_failure is not None:
            raise self._executor.completion_failure


@dataclass(frozen=True)
class _Result:
    completion_event: _Event
    _output: Sampled

    def output(self) -> Sampled:
        return self._output


@dataclass
class FakeExecutor:
    """Runs each forward at once, sampling ``token`` for every row; ``events`` records
    launches and completion waits in order. ``engine`` sets its ``limits``."""

    token: int = TOKEN
    launch_failure: BaseException | None = None
    completion_failure: BaseException | None = None
    events: list[str] = field(default_factory=list)
    forwards: list[Forward] = field(default_factory=list)
    limits: ServingLimits = field(init=False)

    def submit(self, op: Forward | Round, table: RequestTable) -> _Result:
        assert isinstance(op, Forward), op
        if self.launch_failure is not None:
            raise self.launch_failure
        index = len(self.forwards)
        self.forwards.append(op)
        self.events.append(f"launch:{index}")
        return _Result(_Event(self, index), sampled([self.token] * len(op.rows)))

    def graph_stats(self) -> GraphStats:
        return GraphStats((), 0, 0, len(self.forwards))

    def reset_inference_state(self) -> None:
        pass


@dataclass
class EngineHarness:
    engine: InferenceEngine
    executor: FakeExecutor
    replies: list[object]


def engine(
    *,
    overlap: bool = True,
    max_running: int = 4,
    max_batch_tokens: int = 16,
    max_seq_len: int = 64,
    num_pages: int = 64,
    prefix_cache: str = "naive",
    snapshot_rows: range = range(0),
    stop_token_ids: tuple[int, ...] = (2,),
    executor: FakeExecutor | None = None,
    agree_on_step: Callable[[int, str], None] | None = None,
) -> EngineHarness:
    """An InferenceEngine built by its constructor on a ``FakeExecutor``, its radix cache
    restoring recurrent state from ``snapshot_rows``; ``replies`` collects everything it
    publishes."""

    executor = FakeExecutor() if executor is None else executor
    executor.limits = ServingLimits(
        capacity=StateCapacity(
            num_pages=num_pages,
            page_size=1,
            num_slots=max_running,
            max_seq_len=max_seq_len,
            table_width=max_seq_len,
            snapshot_rows=snapshot_rows,
            sink_slot=num_pages,
            guard_slots=range(num_pages + 1, num_pages + 1),
            snapshot_grid=64,  # a Gated DeltaNet model's, on one-token pages
            export_granularity=64,
        ),
        rounds=None,
        stop_token_ids=stop_token_ids,
        vocab_size=1 << 20,
        image_rows=max_batch_tokens,  # as the runner sizes them for an image model
        prefix_cache=prefix_cache,
        max_batch_tokens=max_batch_tokens,
        max_prompt_tokens=max_batch_tokens,
    )
    replies: list[object] = []
    inference = InferenceEngine(
        executor.limits,
        executor,
        publish=replies.extend,
        agree_on_step=agree_on_step,
        text_decoder=None,
        overlap=overlap,
        checks=True,
    )
    return EngineHarness(inference, executor, replies)
