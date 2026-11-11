# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/engine/engine.py`, `python/minisgl/utils/torch_utils.py`);
# see THIRD_PARTY_NOTICES.

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from math import lcm
from typing import TYPE_CHECKING

import torch

from cti import kvcache
from cti.attention import create_fixed_query_backends, fixed_query_bytes
from cti.forward import ForwardInputs, ForwardRows, ImageBatch
from cti.inference._execution._breakable_graph import BreakableGraph
from cti.inference._execution._graphs import (
    DecodeGraph,
    GraphedForward,
    GraphParticipant,
    RowBuckets,
    resolve_decode_graph_batch_sizes,
)
from cti.inference._execution._sampling import (
    BatchSamplingArgs,
    DeviceSampled,
    LegacySamplingLaw,
    LogitProcessing,
    Sampler,
)
from cti.inference._execution._speculative import (
    RoundBackends,
    RoundResult,
    SpeculativeRuntime,
    TargetForward,
    prepare_rounds,
)
from cti.inference._execution._vision_graph import VisionBudgets, VisionGraph
from cti.inference._step import Forward, QueryKind, Round, Row, Sampled, ServingLimits
from cti.recurrent import RecurrentBackend, RecurrentGeometry, RecurrentRounds
from cti.utils import align_ceil, init_logger, mem_GB

if TYPE_CHECKING:
    from contextlib import ExitStack

    from cti.attention import BaseAttnBackend
    from cti.distributed import ParallelGroup
    from cti.inference._execution._hosting import DeviceBudget, HostedModels, ResidentBeside
    from cti.inference._step import (
        GraphCapture,
        GraphStats,
        PageMap,
        RequestSpec,
        RequestTable,
    )
    from cti.models import BaseLLMModel
    from cti.plan import Plan

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class _SampledTokenWrites:
    request_indices: torch.Tensor
    sequence_positions: torch.Tensor
    sample_indices: torch.Tensor | None = None

    def apply(self, token_id_table: torch.Tensor, sampled_token_ids: torch.Tensor) -> None:
        if self.sample_indices is not None:
            sampled_token_ids = sampled_token_ids[self.sample_indices]
        token_id_table[self.request_indices, self.sequence_positions] = sampled_token_ids


@dataclass(frozen=True, eq=False)
class _PreparedForward:
    """A forward's step-private inputs, prepared on the preparation stream; they and the
    host ``staging`` they were copied from must live until the forward completes. The step's
    ``entries`` and their request constants ``specs`` stay here for the sampler; ``rows``
    are what the model side reads, ``metadata`` each backend's inputs for them."""

    staging: tuple[object, ...]
    entries: tuple[Row, ...]
    specs: tuple[RequestSpec, ...]
    rows: ForwardRows
    positions: torch.Tensor
    rope_positions: torch.Tensor
    logit_rows: torch.Tensor
    metadata: dict[str, object]
    images: ImageBatch | None
    input_mapping: tuple[torch.Tensor, torch.Tensor]
    sampled_token_writes: _SampledTokenWrites | None
    sampling_args: BatchSamplingArgs
    # The extension tokens' next prompt tokens, when a row scores its prompt (host int32).
    prompt_targets: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _Outputs:
    """A forward's samples and its rows' scored prompts, on their way to the host."""

    sampled: DeviceSampled
    prompts: tuple[tuple[int, DeviceSampled] | None, ...] | None = None

    def host(self) -> Sampled:
        host = self.sampled.host()
        if self.prompts is None:
            return host
        prompts = tuple(
            None if scored is None else (scored[0], scored[1].host()) for scored in self.prompts
        )
        return replace(host, prompts=prompts)


@dataclass(frozen=True)
class ForwardResult:
    """One submitted forward; it retains its inputs until ``completion_event``, after which
    its sampled tokens are readable on the host."""

    completion_event: torch.cuda.Event
    _outputs: _Outputs = field(repr=False, compare=False)
    _inputs: _PreparedForward = field(repr=False, compare=False)

    def output(self) -> Sampled:
        return self._outputs.host()


@dataclass(frozen=True, slots=True)
class ExecutorParts:
    """What ``build_executor`` assembles for the step path, sized and captured: the target
    ``model``, the caller's streams, the page and token tables, the sampler, the mixer
    backends (``participants``, the recurrent one among them), the graphed forward, the
    speculative rounds and their capture, the serving limits and the most rows any forward
    runs."""

    model: BaseLLMModel
    preparation_stream: torch.cuda.Stream
    execution_stream: torch.cuda.Stream
    page_table: kvcache.PageTable
    token_id_table: torch.Tensor
    sampler: Sampler
    participants: tuple[GraphParticipant, ...]
    recurrent: RecurrentBackend | None
    graphs: GraphedForward
    speculative: SpeculativeRuntime | None
    round_graphs: GraphCapture | None
    limits: ServingLimits
    max_rows: int


class InferenceRunner:
    """Runs a rank's steps on the parts ``build_executor`` assembled; holds no plan."""

    def __init__(self, parts: ExecutorParts, *, ranks: ParallelGroup, checks: bool = False) -> None:
        """``ranks``: the lockstep group the primary samples for; ``checks`` runs the
        internal checks."""

        self._checks = checks
        self._ranks = ranks
        self.model = parts.model
        self.device = parts.token_id_table.device
        self.preparation_stream = parts.preparation_stream
        self.execution_stream = parts.execution_stream
        self.page_table = parts.page_table
        self.token_id_table = parts.token_id_table
        self.sampler = parts.sampler
        self._participants = parts.participants
        self._recurrent = parts.recurrent
        self.graphs = parts.graphs
        self._speculative = parts.speculative
        self._round_graphs = parts.round_graphs
        self.limits = parts.limits
        self._max_rows = parts.max_rows
        rounds = parts.limits.rounds
        self._round_width = 0 if rounds is None else rounds.width
        self._rope_position_axes = parts.model.rope_position_axes()
        # The logit rows of every uniform decode: each row its own.
        self._row_indices = torch.arange(parts.max_rows, dtype=torch.int64, device=self.device)

    def graph_stats(self) -> GraphStats:
        return self.graphs.stats(self._round_graphs)

    def reset_inference_state(self) -> None:
        """Reset device state the kernels keep beside the pool, after serving's memory came
        back without its contents; before the next forward."""

        for layer in self.model.iter_recurrent_layers():
            layer.reset_scratch()

    def submit(self, op: Forward | Round, table: RequestTable) -> ForwardResult | RoundResult:
        """Submit ``op``; ``table`` holds the request constants of the slots it names."""

        match op:
            case Forward():
                return self._forward(op, table)
            case Round():
                return self._round(op, table)

    @torch.inference_mode()
    def _forward(self, op: Forward, table: RequestTable) -> ForwardResult:
        """Submit one forward asynchronously across preparation and execution streams.

        Preparation is enqueued on the runner-owned preparation stream. The execution
        stream waits for it, and the returned result retains asynchronous inputs until
        its completion event has synchronized.
        """
        inputs = self._prepare_forward(op, table)
        rows = inputs.rows
        with torch.cuda.stream(self.execution_stream):
            batch = self._begin_forward(inputs)
            hidden = None
            if inputs.prompt_targets is not None:
                # Prompt log-probabilities read every prompt row's hidden state.
                hidden = self.graphs.hidden(batch)
                logits = self.model.compute_logits(hidden, batch)
            else:
                logits = self.graphs.logits(rows, batch)
            if self._recurrent is not None:
                self._recurrent.export(batch)
            if self._speculative is not None:
                forward = TargetForward(batch, rows, inputs.positions, inputs.input_mapping[0])
                self._speculative.after_forward(forward)
            outputs = self._sample_forward(inputs, logits[: rows.size], hidden)
            completion_event = torch.cuda.Event()
            completion_event.record(self.execution_stream)

        # The inputs are prepared on a different stream and must outlive their
        # execution-stream consumers. The worker retains this result until
        # completion_event has synchronized.
        return ForwardResult(completion_event, outputs, inputs)

    @torch.inference_mode()
    def _round(self, op: Round, table: RequestTable) -> RoundResult:
        """Submit one round over ``op``'s slots, with the prompt extensions of ``op.extend``,
        which its target forward runs first."""

        inputs = None if op.extend is None else self._prepare_forward(op.extend, table)
        with torch.cuda.stream(self.execution_stream):
            extension = None
            if inputs is not None:
                extension = TargetForward(
                    self._begin_forward(inputs),
                    inputs.rows,
                    inputs.positions,
                    inputs.input_mapping[0],
                    lambda logits, hidden: self._sample_forward(inputs, logits, hidden),
                )
            # Unsettled rounds read the shared mapping: change it in execution order. The
            # page staging lives until the round completes.
            pages = self._apply_pages(op.pages)
            if self._checks:
                extended = None if inputs is None else inputs.rows
                _require_warmed_rows(self._max_rows, extended, len(op.slots) * self._round_width)
            specs = [table[slot] for slot in op.slots]
            result = self._speculative.submit(op, specs, extension, keep=pages)
            if extension is not None and self._recurrent is not None:
                self._recurrent.export(extension.batch)
        return result

    def _prepare_forward(self, op: Forward, table: RequestTable) -> _PreparedForward:
        """Prepare a forward's step-private inputs on the preparation stream."""

        tokens = None if op.extend_token_ids is None else _host(op.extend_token_ids, torch.int32)
        rope_rows = None
        if op.extend_rope_positions is not None:
            assert tokens is not None
            rope_rows = _host(op.extend_rope_positions, torch.int64).view(-1, len(tokens))
        images = None if op.images is None else ImageBatch.unpack(op.images)
        targets = None
        if op.prompt_targets is not None:
            targets = _host(op.prompt_targets, torch.int32)
        restored = tuple((slot, _host(prefix, torch.int32)) for slot, prefix in op.restored_prompts)
        with torch.cuda.stream(self.preparation_stream):
            pages = self._apply_pages(op.pages)
            specs = tuple(table[row.request_index] for row in op.rows)
            rows = self.graphs.rows(op.rows, specs)
            if self._checks:
                _require_warmed_rows(self._max_rows, rows)
            positions = _make_positions(rows, self.device)
            input_mapping = _make_input_tuple(rows, positions, self.device)
            rope_positions = _make_rope_positions(
                self._rope_position_axes, rope_rows, rows, positions, self.device
            )
            device_images = None if images is None else images.to(self.device, non_blocking=True)
            _apply_extend_token_ids(self.token_id_table, tokens, input_mapping)
            # Logits processing reads a restored prompt prefix, which no forward writes.
            for slot, prefix in restored:
                self.token_id_table[slot, : len(prefix)] = prefix.to(self.device, non_blocking=True)
            return _PreparedForward(
                (pages, tokens, rope_rows, images, restored),
                op.rows,
                specs,
                rows,
                positions,
                rope_positions,
                self._logit_rows(rows),
                {
                    name: value
                    for participant in self._participants
                    for name, value in participant.prepare_inputs(rows).items()
                },
                device_images,
                input_mapping,
                _prepare_sampled_token_writes(op.rows, self.device),
                self.sampler.prepare(op.rows, specs),
                targets,
            )

    def _begin_forward(self, inputs: _PreparedForward) -> ForwardInputs:
        """On the execution stream, behind the preparation: gather the forward's input IDs,
        restore its recurrent snapshot rows and return what the model reads."""

        self.execution_stream.wait_stream(self.preparation_stream)
        rows = inputs.rows
        batch = ForwardInputs(
            self.token_id_table[inputs.input_mapping],
            inputs.rope_positions,
            size=rows.size,
            ragged=rows.uses_ragged_attention,
            images=inputs.images,
            logit_rows=inputs.logit_rows,
            image_encoding=self.graphs.images,
            **inputs.metadata,
        )
        if self._recurrent is not None:
            self._recurrent.restore(batch)
        return batch

    def _sample_forward(
        self,
        inputs: _PreparedForward,
        logits: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> _Outputs:
        """Sample each entry's next token from its last row's ``logits``, write the tokens
        decodes and finished prompts carry to the token table, and queue them for the host,
        with each extension's raw prompt log-probabilities from its tokens' ``hidden``
        states when a row scores its prompt."""

        sampled = self._sample(logits, inputs.sampling_args)
        if inputs.sampled_token_writes is not None:
            inputs.sampled_token_writes.apply(self.token_id_table, sampled.token_ids)
        prompts = None
        if inputs.prompt_targets is not None:
            assert hidden is not None
            scored = self.sampler.score_prompts(
                inputs.entries,
                inputs.specs,
                inputs.prompt_targets,
                hidden,
                self.model.token_head().logits,
            )
            prompts = tuple(None if row is None else (row[0], row[1].to_cpu()) for row in scored)
        return _Outputs(sampled.to_cpu(), prompts)

    def _sample(self, logits: torch.Tensor, sample_args: BatchSamplingArgs) -> DeviceSampled:
        """Sample on the primary rank and broadcast tokens and scores in one collective, so
        peers draw no randomness and rank-local schedulers stay in lockstep."""
        group = self._ranks
        if group.is_primary():
            sampled = self.sampler.sample(logits, sample_args)
            sampled = replace(sampled, token_ids=sampled.token_ids.to(torch.int32))
            if group.size == 1:
                return sampled
            buffer = sampled.packed()
        else:
            logprobs = sample_args.logprobs
            sampled, buffer = DeviceSampled.receiving(
                len(logits),
                logprobs=logprobs is not None,
                top=0 if logprobs is None else logprobs.top,
                device=self.device,
            )
        group.broadcast_device(buffer)
        return sampled

    def _logit_rows(self, rows: ForwardRows) -> torch.Tensor:
        """``rows``' logit rows on the device: every row's own for a uniform decode."""

        if rows.is_uniform_decode:
            return self._row_indices[: rows.size]
        host = torch.tensor(rows.logit_rows, dtype=torch.int64, pin_memory=True)
        return host.to(self.device, non_blocking=True)

    def _apply_pages(self, pages: PageMap | None) -> torch.Tensor | None:
        """Copy an op's page-table changes into the page table on the current stream;
        return the host triples, which must live until the copy completes."""

        if pages is None:
            return None
        triples = _host(pages.entries, torch.int32).view(-1, 3)
        self.page_table.apply(triples)
        return triples


def build_executor(
    host: HostedModels,
    memory: DeviceBudget,
    plan: Plan,
    *,
    ranks: ParallelGroup,
    preparation_stream: torch.cuda.Stream,
    execution_stream: torch.cuda.Stream,
    resources: ExitStack,
    beside: Callable[[], ResidentBeside] | None = None,
    checks: bool = False,
    progress: Callable[[str], None] | None = None,
) -> InferenceRunner:
    """The rank's executor over ``host``'s models (the composition root's one entry into
    the execution half): warm the models up, size the inference state around them in what
    ``memory``'s budget leaves beside them and what ``beside`` keeps (prepared once the
    models are warm), then build the tables, backends and graphs on the caller's streams.
    ``resources`` releases what it captures, ``checks`` runs the internal checks, and
    ``progress`` hears each long phase as it begins."""

    progress = progress or _no_progress
    device, model = host.device, host.model
    # The decode graph sizes for a number of request slots, which planning chooses.
    graph_batch_sizes = partial(
        resolve_decode_graph_batch_sizes,
        plan.graph_batch_sizes,
        plan.graph_max_batch,
        memory.boot_bytes,
    )
    # Before any warm-up and capture: the target's taps and the rounds' scratch come out of
    # the KV budget.
    method = plan.speculative
    round_models = scratch = None
    if method is not None:
        round_models, scratch = prepare_rounds(method.config, plan, host)
    # Before capacity planning: the sampling law's workspace comes out of the KV budget.
    law = (
        LegacySamplingLaw(plan.capability)
        if round_models is None
        else round_models.sampling_law(plan.max_running_requests)
    )
    # Sized before planning, so up to the plan's bound on the request slots.
    max_rows = _max_forward_rows(plan, graph_batch_sizes(plan.max_running_requests))
    _warm_up(host, plan, max_rows=max_rows, progress=progress)
    # What runs beside inference compiles after the warm-up, before the KV is sized.
    beside_bytes = 0 if beside is None else beside().resident_bytes

    # Before capacity planning: segments and vision graphs reference no KV or recurrent state,
    # and the pools and static buffers they hold count with the weights against the KV budget.
    graph_pool = memory.graph_pool()
    resources.callback(graph_pool.close)  # last, once every graph is gone
    buckets = RowBuckets.for_limits(plan.max_batch_tokens, plan.graph_row_buckets)
    breakable = None
    if buckets.sizes:
        progress("capturing breakable graphs")
        breakable = BreakableGraph(
            model=model, buckets=buckets, pool=graph_pool, stream=execution_stream, device=device
        )
        resources.callback(breakable.close)
    # Vision graphs capture into a pool of their own, so their memory shows apart.
    vision = None
    if plan.vision_graph_budgets and (tower := model.vision_tower()) is not None:
        progress("capturing vision graphs")
        vision_pool = memory.graph_pool()
        resources.callback(vision_pool.close)
        vision = VisionGraph(
            tower=tower,
            budgets=VisionBudgets(plan.vision_graph_budgets),
            pool=vision_pool,
            stream=execution_stream,
            device=device,
        )
        resources.callback(vision.close)

    # ======================= Inference state initialization ========================
    # The models' state layers by role (the models and the roles both list the target
    # first): the target's KV group, then the draft's, on one page table.
    layout = kvcache.group_layers(
        [
            (role, layer)
            for role, member in zip(kvcache.GroupRole, host.models, strict=False)
            for layer in member.iter_state_layers()
        ],
        plan.state_slots,
    )
    # Rounds' guard rows, one round of queries, come out of the planned pages, past the
    # graph sink page.
    rounds = plan.rounds
    # Up to the plan's bound; the budget chooses the slots when the option is unset.
    geometry = kvcache.TableGeometry(
        plan.page_size,
        plan.max_running_requests,
        plan.max_seq_len,
        guard_tokens=0 if rounds is None else rounds.max_rows * rounds.width,
    )
    # The attention backends allocate their workspaces after the KV pools, each rank for its
    # own heads: the target's, and with rounds theirs over both pools.
    workspace_bytes = plan.attention.workspace_bytes(
        device=device, max_running_req=plan.max_running_requests
    )
    progress("planning capacity")
    image_encoder_bytes = _image_encoder_peak(model, plan, memory, device)
    image_rows = plan.encoder_rows if image_encoder_bytes else 0
    model_config = host.checkpoint.model_config
    capacity = memory.plan(
        layout,
        geometry=geometry,
        snapshots=plan.snapshots,
        workspace_bytes=workspace_bytes
        + (
            0
            if round_models is None
            else fixed_query_bytes(round_models.fixed_query_geometries(), device)
        ),
        image_encoder_bytes=image_encoder_bytes,
        image_rows_bytes=image_rows and image_rows * model_config.hidden_size * plan.dtype.itemsize,
        beside_bytes=beside_bytes,
        allocations=None if round_models is None else round_models.state_allocations,
        fit_slots=plan.auto_running_requests,
    )
    logger.info(
        "Inference capacity: %d KV pages, %d snapshot rows",
        capacity.num_pages,
        len(capacity.snapshot_rows),
    )
    state, image_rows_buffer = memory.allocate(
        layout,
        capacity,
        (image_rows, model_config.hidden_size) if image_rows else None,
        plan.dtype,
    )
    if image_rows_buffer is not None:
        host.services.image_rows = image_rows_buffer
    num_slots = capacity.num_slots
    decode_batch_sizes = graph_batch_sizes(num_slots)

    # ====================== Execution table initialization ======================
    # Rows past the requests' pad graph batches up to their buckets: decode graphs repeat
    # the first, a round takes distinct ones, all done.
    padding = RowBuckets(tuple(decode_batch_sizes)).padding_rows(
        0 if rounds is None else rounds.max_rows
    )
    padding_rows = range(num_slots, num_slots + padding)
    max_seq_len = capacity.max_seq_len
    # Rounds address the cache by token.
    page_table = kvcache.PageTable(
        capacity, padding_rows.stop, device=device, token_view=rounds is not None
    )
    # The graph padding rows must always yield token ID zero. Rounds write uncommitted
    # outputs to scratch columns, one round wide, past the sequence capacity.
    token_id_table = torch.zeros(
        (
            padding_rows.stop,
            _token_id_table_width(
                max_seq_len + (0 if rounds is None else rounds.width), plan.page_size
            ),
        ),
        dtype=torch.int32,
        device=device,
    )
    processing = LogitProcessing(
        token_id_table, model_config.vocab_size, host.checkpoint.facts.stop_token_ids
    )
    sampler = Sampler(
        device,
        model_config.vocab_size,
        law=law,
        processing=processing,
        batch_capacity=num_slots,
        logits_dtype=plan.dtype,
    )

    # ======================= Attention backend initialization ========================
    # The target's KV groups, which its one backend serves.
    target = tuple(g for g in state.groups if g.role is kvcache.GroupRole.TARGET)
    attn_backend: BaseAttnBackend = plan.attention.build(
        target, page_table, max_running_req=num_slots, capability=plan.capability
    )
    # Each KV group's backend, in group order.
    backends: tuple[BaseAttnBackend, ...] = (attn_backend,) * len(target)
    round_backends = None
    if round_models is not None:
        # A round's verify attends on its backend; the target's other batches keep it.
        round_backends = RoundBackends(
            *create_fixed_query_backends(
                round_models.fixed_query_geometries(), state, (attn_backend, None), page_table
            )
        )
        attn_backend = round_backends.target
        backends = round_backends.kv_backends
    layout.bind(backends, state.recurrent)

    # Per-batch state work beside the forward; none without recurrent layers.
    recurrent = (
        None if state.recurrent is None else RecurrentBackend(state.recurrent, device=device)
    )
    # The mixer backends, each filling its own metadata into a forward's inputs.
    participants: tuple[GraphParticipant, ...] = tuple(
        p for p in (attn_backend, recurrent) if p is not None
    )

    # Every rank loads the sampler before readiness: the primary later samples alone while
    # its peers wait in the token broadcast, which a first-use build could time out.
    progress("warming up the sampler")
    sampler.warm_up(decode_batch_sizes or [num_slots])

    logger.info_rank0(f"Free memory after initialization: {mem_GB(memory.free_bytes())}")

    # ======================= Graph capture initialization ========================
    progress("capturing decode graphs")
    decode = DecodeGraph(
        stream=execution_stream,
        device=device,
        model=model,
        participants=participants,
        batch_sizes=decode_batch_sizes,
        max_seq_len=max_seq_len,
        vocab_size=model_config.vocab_size,
        logits_dtype=plan.dtype,
        dummy_entry=Row(padding_rows.start, QueryKind.DECODE, 0, 1, write_position=1),
        pool=graph_pool,
    )
    resources.callback(decode.close)
    graphs = GraphedForward(model, decode, breakable, vision)

    speculative = round_graphs = None
    if round_models is not None:
        assert round_backends is not None
        # Building the rounds compiles their draft attention kernels.
        progress("building speculative rounds")
        # Every transactional recurrent layer's replay, made once, joins the rounds' K2.
        recurrent_rounds = None
        if (pool := state.recurrent) is not None:
            replay_geometry = RecurrentGeometry(
                pool.num_slots, len(pool.snapshot_rows), pool.device
            )
            replays = [
                layer.create_replay(
                    max_batch=round_models.max_batch_size,
                    rounds=round_models.rounds,
                    geometry=replay_geometry,
                )
                for layer in round_models.target.iter_recurrent_layers()
                if layer.transactional
            ]
            recurrent_rounds = RecurrentRounds(replays, shape=round_models.rounds)
        # The primary decides sampled verdicts, as it samples tokens.
        speculative = SpeculativeRuntime(
            round_models,
            scratch,
            state,
            recurrent_rounds=recurrent_rounds,
            backends=round_backends,
            token_id_table=token_id_table,
            processing=processing,
            logprobs=sampler.logprobs,
            padding_rows=padding_rows,
            forward_hidden=graphs.hidden,
            decides=ranks.is_primary(),
            publish=ranks.broadcast_device,
            checks=checks,
            differs=ranks.differs_device,
        )
        resources.callback(speculative.close)
        progress("capturing speculative rounds")
        round_graphs = speculative.capture(
            decode.batch_sizes, stream=execution_stream, pool=graph_pool
        )
    limits = ServingLimits(
        capacity=capacity,
        rounds=rounds,
        stop_token_ids=host.checkpoint.facts.stop_token_ids,
        vocab_size=model_config.vocab_size,
        image_rows=image_rows,
        prefix_cache=plan.prefix_cache,
        max_batch_tokens=plan.max_batch_tokens,
        max_prompt_tokens=plan.prompt_budget.tokens,
    )
    parts = ExecutorParts(
        model,
        preparation_stream,
        execution_stream,
        page_table,
        token_id_table,
        sampler,
        participants,
        recurrent,
        graphs,
        speculative,
        round_graphs,
        limits,
        max_rows,
    )
    return InferenceRunner(parts, ranks=ranks, checks=checks)


def _warm_up(
    host: HostedModels,
    plan: Plan,
    *,
    max_rows: int,
    progress: Callable[[str], None],
) -> None:
    """Compile every GEMM and op the models run up to the most rows a forward takes,
    before capacity planning and before any graph is captured, so the budget counts their
    workspaces and no op's cached storage grows under a captured graph."""

    progress("compiling inference kernels")
    started = time.monotonic()
    gemm_workspace = host.prepare_inference(
        max_rows,
        max_sequences=plan.max_running_requests,
        progress=lambda detail: progress(f"compiling inference kernels ({detail})"),
    )
    logger.info_rank0(
        "Inference kernels compiled in %.1f s; GEMM workspace %.1f MiB",
        time.monotonic() - started,
        gemm_workspace / (1 << 20),
    )


def _image_encoder_peak(
    model: BaseLLMModel, plan: Plan, memory: DeviceBudget, device: torch.device
) -> int:
    """The activation peak, on any rank, of the largest image batch one step may encode.

    A step's images fill at most the plan's ``encoder_rows``, and each image at most its
    ``image_limits``, encoded whole in one call as vLLM profiles them. The model's warm-up
    also holds what a step keeps beside its encoder, such as the step's token embeddings.
    0 for a model without an image encoder, or a configuration that admits no image.
    """

    max_image_tokens = plan.image_limits.max_image_tokens
    if max_image_tokens <= 0:
        return 0

    @torch.inference_mode()
    def encode() -> bool:
        return model.warm_up_image_encoder(
            max_tokens=plan.encoder_rows, max_image_tokens=max_image_tokens, device=device
        )

    return memory.peak_bytes(encode)


def _max_forward_rows(plan: Plan, decode_batch_sizes: Sequence[int]) -> int:
    """The most rows any forward of ``plan`` runs on a rank: a prefill chunk or mixed step,
    the largest decode graph, the largest breakable graph, the largest speculative round.
    Every rank runs every row."""

    rounds = plan.rounds
    breakable = RowBuckets.for_limits(plan.max_batch_tokens, plan.graph_row_buckets)
    return max(
        plan.max_batch_tokens,
        max(decode_batch_sizes, default=0),
        max(breakable.sizes, default=0),
        0 if rounds is None else rounds.max_rows * rounds.width,
    )


def _require_warmed_rows(max_rows: int, forward: ForwardRows | None, round_rows: int = 0) -> None:
    """A forward's rows (its padded rows' extensions and a round's rows) are within the
    ``max_rows`` every op was warmed at before capture, past which an op's cached storage
    could grow under a captured graph. A check, run under ``checks``."""

    rows = round_rows
    if forward is not None:
        rows += sum(forward.extend_len(index) for index in range(forward.padded_size))
    if rows > max_rows:
        raise RuntimeError(
            f"a forward of {rows} rows exceeds the {max_rows} every op was warmed at"
        )


def _make_rope_positions(
    axes: int,
    extend_rows: torch.Tensor | None,
    rows: ForwardRows,
    positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """The step's int64 rotary positions, padding included, for a model whose rotary reads
    ``axes`` position axes: one row that every axis reads, or ``[axes, tokens]`` M-RoPE rows
    when a prompt holds images.

    Without images or rope offsets the row is the sequence positions themselves; otherwise
    the positions are built on the host and copied once.
    """

    offsets = rows.rope_offsets
    if extend_rows is None and not any(offsets):
        return positions
    if axes == 1 or (extend_rows is not None and extend_rows.shape[0] != axes):
        raise ValueError(f"the model's rotary embedding reads {axes} position axes")
    return _make_offset_rope_positions(rows, offsets, extend_rows, device)


def _apply_extend_token_ids(
    token_id_table: torch.Tensor,
    extend_token_ids: torch.Tensor | None,
    input_mapping: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Write a forward's host extension tokens into the token table at their rows."""

    if extend_token_ids is None:
        return
    num_tokens = len(extend_token_ids)
    token_id_table[input_mapping[0][:num_tokens], input_mapping[1][:num_tokens]] = (
        extend_token_ids.to(token_id_table.device, non_blocking=True)
    )


def _no_progress(phase: str) -> None:
    pass


def _token_id_table_width(num_tokens: int, page_size: int) -> int:
    """Rows of ``num_tokens`` aligned to 128 bytes and to whole pages."""
    return align_ceil(num_tokens, lcm(32, page_size))


def _make_positions(batch: ForwardRows, device: torch.device) -> torch.Tensor:
    lengths = [batch.extend_len(index) for index in range(batch.padded_size)]
    indices_host = torch.empty(sum(lengths), dtype=torch.int64, pin_memory=True)
    offset = 0
    for start, end, length in zip(batch.kv_prefix_lens, batch.sequence_lens, lengths, strict=True):
        torch.arange(start, end, dtype=torch.int64, out=indices_host[offset : offset + length])
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_offset_rope_positions(
    batch: ForwardRows,
    offsets: tuple[int, ...],
    extend_rows: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Every token at its sequence position plus its padded row's rope offset in
    ``offsets``, as one row; with the extend tokens' frozen ``[axes, tokens]`` rows, as rows
    of that shape."""

    if batch.is_uniform_decode:
        positions_host = torch.tensor(
            [prefix + offset for prefix, offset in zip(batch.kv_prefix_lens, offsets, strict=True)],
            dtype=torch.int64,
            pin_memory=True,
        )
        return positions_host.to(device, non_blocking=True)

    lengths = [batch.extend_len(index) for index in range(batch.padded_size)]
    shape = (sum(lengths),) if extend_rows is None else (extend_rows.shape[0], sum(lengths))
    positions_host = torch.empty(shape, dtype=torch.int64, pin_memory=True)
    offset = 0
    for index, (length, rope_offset) in enumerate(zip(lengths, offsets, strict=True)):
        columns = positions_host[..., offset : offset + length]
        if extend_rows is not None and index < batch.num_extend:
            # Extensions come first, so their columns are those of the frozen rows.
            columns.copy_(extend_rows[:, offset : offset + length])
        else:
            first = batch.kv_prefix_lens[index] + rope_offset
            columns.copy_(torch.arange(first, first + length))
        offset += length
    return positions_host.to(device, non_blocking=True)


def _host(data: bytes, dtype: torch.dtype) -> torch.Tensor:
    """A plan's bytes as a writable host tensor of ``dtype``."""

    return torch.frombuffer(bytearray(data), dtype=dtype)


def _make_input_tuple(
    batch: ForwardRows,
    positions: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mapping_host = torch.empty(len(positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for index, slot in enumerate(batch.slots):
        length = batch.extend_len(index)
        mapping_host[offset : offset + length].fill_(slot)
        offset += length
    return mapping_host.to(device, non_blocking=True), positions


def _prepare_sampled_token_writes(
    entries: tuple[Row, ...],
    device: torch.device,
) -> _SampledTokenWrites | None:
    writable_batch_indices = tuple(
        index for index, entry in enumerate(entries) if entry.write_position is not None
    )
    if not writable_batch_indices:
        return None
    writable_entries = tuple(entries[index] for index in writable_batch_indices)
    sample_indices = (
        None
        if len(writable_entries) == len(entries)
        else torch.tensor(
            writable_batch_indices,
            dtype=torch.int64,
            pin_memory=True,
        ).to(device, non_blocking=True)
    )

    request_indices = torch.tensor(
        [entry.request_index for entry in writable_entries],
        dtype=torch.int64,
        pin_memory=True,
    ).to(device, non_blocking=True)
    sequence_positions = torch.tensor(
        [entry.write_position for entry in writable_entries],
        dtype=torch.int64,
        pin_memory=True,
    ).to(device, non_blocking=True)
    return _SampledTokenWrites(
        request_indices=request_indices,
        sequence_positions=sequence_positions,
        sample_indices=sample_indices,
    )
