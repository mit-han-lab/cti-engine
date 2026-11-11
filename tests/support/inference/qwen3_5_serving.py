"""Qwen3.5 served on CPU through the engine's own scheduler and runner preparation.

``CpuServing`` admits ``UserMsg`` requests into a ``InferenceScheduler`` over a naive
``CacheManager``, or a radix one with recurrent snapshot rows, and executes each of its steps with ``InferenceRunner.submit``. The
runner is built around the model without its constructor: CPU tables and pools stand in
for its device state (the pools start as NaN, so a row read before it is written poisons
the logits), the reference backends of ``qwen3_5_engine`` for its kernels, and a greedy
sampler that records every step's logits for its sampler. The CUDA streams, events and
pinned allocations the runner asks for are patched out for the test.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from cti import kvcache
from cti.forward import EagerImages, ForwardRows
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.strategy import SchedulingLimits, TokenSteps
from cti.inference._execution._graphs import forward_rows
from cti.inference._execution._sampling import DeviceSampled
from cti.inference._execution.runner import ExecutorParts, InferenceRunner
from cti.inference._step import QueryKind, Row
from cti.kvcache import RecurrentStatePool
from cti.recurrent import RecurrentBackend
from tests.support.gated_delta_net import fp32_history
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.scheduling import request_ids
from tests.support.paged_attention import (
    ReferencePagedAttentionBackend,
    bind_model_state,
    kv_group,
)

if TYPE_CHECKING:
    import pytest

    from cti.forward import ForwardInputs
    from cti.inference import Forward
    from cti.inference._step import RequestSpec
    from cti.models.base import BaseLLMModel
    from cti.protocol import RequestRejectedMsg, UserMsg

CPU = torch.device("cpu")


@dataclass(frozen=True)
class SampledStep:
    """One executed batch: its entries and the logits the sampler received for them."""

    entries: tuple[Row, ...]
    logits: torch.Tensor


class _OneRank:
    """The runner's lockstep group of one rank: it samples, and has no peer to send to."""

    size = 1

    def is_primary(self) -> bool:
        return True

    def broadcast_device(self, _tensor: torch.Tensor, *, root: int = 0) -> None:
        pass


class _Stream:
    def wait_stream(self, _stream: object) -> None:
        pass


class _CompletedEvent:
    def record(self, _stream: object = None) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _EagerGraphs:
    def __init__(self, model: object) -> None:
        self._model = model
        self.images = EagerImages()

    def rows(self, entries: tuple[Row, ...], specs: tuple[RequestSpec, ...]) -> ForwardRows:
        return forward_rows(entries, specs)

    def logits(self, rows: ForwardRows, batch: ForwardInputs) -> torch.Tensor:
        return self._model.forward(batch)


class _RecordingGreedySampler:
    def __init__(self) -> None:
        self.steps: list[SampledStep] = []
        self._entries: tuple[Row, ...] = ()

    def prepare(self, rows: tuple[Row, ...], _specs: object) -> SimpleNamespace:
        self._entries = rows
        return SimpleNamespace(logprobs=None)

    def sample(self, logits: torch.Tensor, _args: object) -> DeviceSampled:
        self.steps.append(SampledStep(self._entries, logits.clone()))
        return DeviceSampled(logits.argmax(dim=-1))


def _drop_pinning(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("empty", "tensor"):
        allocate = getattr(torch, name)

        def allocate_unpinned(*args: object, _allocate=allocate, **kwargs: object) -> torch.Tensor:
            kwargs.pop("pin_memory", None)
            return _allocate(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate_unpinned)


class CpuServing:
    """The model behind a real scheduler and ``InferenceRunner.submit``, on CPU."""

    def __init__(
        self,
        model: BaseLLMModel,
        monkeypatch: pytest.MonkeyPatch,
        *,
        max_running_req: int,
        max_batch_tokens: int,
        max_seq_len: int,
        snapshot_rows: int = 0,
    ) -> None:
        """With ``snapshot_rows``, a radix cache restores recurrent state from those rows."""

        monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
        monkeypatch.setattr(torch.cuda, "Event", _CompletedEvent)
        _drop_pinning(monkeypatch)
        num_pages = max_running_req * max_seq_len
        recurrent_layers = model.iter_recurrent_layers()
        paged_layers = tuple(model.iter_attention_layers())
        recurrent_state = RecurrentStatePool(
            [fp32_history(layer.state_spec) for layer in recurrent_layers],
            num_slots=max_running_req,
            num_snapshot_rows=snapshot_rows,
            device=CPU,
        )
        group = kv_group(
            len(paged_layers), paged_layers[0].num_kv_heads, paged_layers[0].head_dim, num_pages
        )
        for index in range(len(recurrent_layers)):
            for view in recurrent_state.layer_views(index):
                view.fill_(torch.nan)
        for layer in group.layers:
            layer.fill_(torch.nan)

        page_table = kvcache.PageTable(
            SimpleNamespace(page_size=1, table_width=max_seq_len, sink_slot=0, max_seq_len=0),
            max_running_req + 1,
            device=CPU,
        )
        attention = ReferencePagedAttentionBackend(group, page_table.pages)
        recurrent = RecurrentBackend(recurrent_state, device=CPU)
        parts = ExecutorParts(
            model,
            _Stream(),
            _Stream(),
            page_table,
            torch.zeros_like(page_table.pages),
            _RecordingGreedySampler(),
            (attention, recurrent),
            recurrent,
            _EagerGraphs(model),
            None,
            None,
            SimpleNamespace(rounds=None),
            max_running_req,
        )
        runner = InferenceRunner(parts, ranks=_OneRank())
        bind_model_state(model, attention, recurrent_state)
        if getattr(model, "visual", None) is not None:
            # As the runner sizes them: one step's tokens, NaN until the encoder writes.
            hidden = model.model.embed_tokens.weight.shape[1]
            model.test_services.image_rows = torch.full((max_batch_tokens, hidden), torch.nan)

        self.runner = runner
        # Each executed forward, and the uid and constants of each of its rows.
        self.batches: list[Forward] = []
        self.request_ids: list[tuple[int, ...]] = []
        self.specs: list[tuple[RequestSpec, ...]] = []
        self.scheduler = InferenceScheduler(
            checks=True,
            limits=SchedulingLimits(
                max_running_requests=max_running_req,
                max_batch_tokens=max_batch_tokens,
                max_sequence_length=max_seq_len,
                max_prompt_tokens=max_batch_tokens,
            ),
            cache_manager=new_cache_manager(
                num_pages,
                1,
                torch.full((max_running_req, max_seq_len), -1, dtype=torch.int32),
                reuse=bool(snapshot_rows),
                checks=True,
                snapshot_rows=recurrent_state.snapshot_rows,
            ),
            stop_token_ids=(),
            vocab_size=1 << 20,
            decode=TokenSteps(),
            encoder_cache=EncoderCache(max_batch_tokens, checks=True),
        )

    @property
    def steps(self) -> list[SampledStep]:
        return self.runner.sampler.steps

    def scheduled(self, *, decode: bool) -> list[tuple[int, Row, RequestSpec]]:
        """Each executed decode (or extension) row with its uid and constants, in order."""

        kind = QueryKind.DECODE if decode else QueryKind.EXTEND
        return [
            (uid, row, spec)
            for batch, uids, specs in zip(self.batches, self.request_ids, self.specs, strict=True)
            for row, uid, spec in zip(batch.rows, uids, specs, strict=True)
            if row.query_kind is kind
        ]

    def submit(self, *messages: UserMsg) -> tuple[RequestRejectedMsg, ...]:
        rejections = []
        for message in messages:
            rejections.extend(self.scheduler.handle_message(message))
        return tuple(rejections)

    def run(self) -> dict[int, list[int]]:
        """Plan, execute, submit and settle until no request is left; each request's
        sampled tokens by uid."""

        sampled: dict[int, list[int]] = defaultdict(list)
        while self.scheduler.has_runnable_work:
            plan = self.scheduler.plan_next()
            assert plan is not None
            self.batches.append(plan.op)
            self.request_ids.append(request_ids(plan))
            table = self.scheduler.table
            self.specs.append(tuple(table[row.request_index] for row in plan.op.rows))
            result = self.runner.submit(plan.op, table)
            self.scheduler.mark_submitted(plan)
            outcome = self.scheduler.settle_step(plan, result.output())
            for reply in outcome.replies:
                sampled[reply.uid].extend(reply.token_ids)
        return dict(sampled)


__all__ = ["CpuServing", "SampledStep"]
