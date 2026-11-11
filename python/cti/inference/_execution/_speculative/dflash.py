# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Round semantics follow z-lab/dflash 07ebd93db9f472af339b644bb70221ad8428328a
# (`dflash/model.py`, `dflash_generate`).

"""DFlash rounds: draft a block from target taps, verify it on the target, commit on device."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import TYPE_CHECKING

import torch

from cti.attention import (
    FixedQueryGeometry,
    FixedQueryShape,
    MixedQuery,
)
from cti.forward import ForwardInputs, RoundShape
from cti.inference._execution._graphs import RowBuckets, capture_graphs
from cti.inference._execution._sampling import (
    LogitRows,
    LogprobRows,
    RequestLocalSamplingLaw,
    SampledLogprobs,
    Steering,
)
from cti.inference._step import Round
from cti.kvcache import next_snapshot_position
from cti.models import FeatureRequest
from cti.sampling import TokenSampling
from cti.utils import init_logger
from ctikernels.speculative import RoundVariant, commit_block, prepare_round, warm_up

from ._records import (
    ExtensionOutputs,
    RoundArgs,
    RoundInput,
    RoundRecords,
    RoundResult,
    Verification,
)
from ._scratch import RoundScratch
from .proposal import ArgmaxProposal, ChainProposal, ChainRows, Proposal

if TYPE_CHECKING:
    from cti.attention import SupportsFixedQuery
    from cti.forward import ForwardRows
    from cti.inference._execution._hosting import HostedModels
    from cti.inference._execution._sampling import LogitProcessing
    from cti.inference._step import GraphCapture, RequestSpec
    from cti.kvcache import InferenceState
    from cti.memory import GraphPool
    from cti.models import BaseLLMModel, TokenHead
    from cti.models.dflash import DFlashConfig
    from cti.plan import Plan
    from cti.recurrent import RecurrentRounds, RoundStatus
    from ctikernels.sampling import RequestLocalChainSampling

    from .method import DraftModel

logger = init_logger(__name__)

# A law that steers (masks the stop tokens), whose full warm-up rounds compile the steering,
# and the warm-up rounds' kinds: (full, steers).
_WARM_UP_STEERING = TokenSampling(min_tokens=1)
_WARM_UP_KINDS = ((False, False), (True, False), (True, True))


@dataclass(frozen=True, slots=True)
class TargetForward:
    """A target forward the runner prepared: an ordinary one the rounds follow, or a step's
    prompt extensions a round's target forward runs ahead of its rows. It holds the model's
    inputs, their rows and positions, each token's request row, and for an extension how
    its next tokens are sampled from its last rows' full-vocabulary logits and its tokens'
    hidden states (which score a prompt that asks for its logprobs)."""

    batch: ForwardInputs
    rows: ForwardRows
    positions: torch.Tensor
    token_rows: torch.Tensor
    sample: Callable[[torch.Tensor, torch.Tensor], ExtensionOutputs] | None = None


@dataclass(frozen=True, slots=True)
class RoundRun:
    """An enqueued round: its recurrent status, if any, whose settlement runs in submission
    order; its extension's sampled next tokens, if any; and under checks each fault's
    message and whether the round raised it (host bools, readable once the round completes)."""

    status: RoundStatus | None
    sampled: ExtensionOutputs | None
    faults: tuple[tuple[str, torch.Tensor], ...] = ()


@dataclass(frozen=True, slots=True)
class _Verdict:
    """A round's greedy target IDs (None: sampled into the round's buffers) and, for a full
    round, its block rows' raw and processed ``[B, Q, V]`` logits."""

    target_ids: torch.Tensor | None
    logits: torch.Tensor | None = None
    processed: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _RoundPlan:
    """One batch size's views of the round scratch; draft and verify share the rows.

    ``draft`` holds one input per distinct draft attention mask and ``draft_layers`` each
    draft layer's; ``slot_verifies`` holds ``verify`` per status slot of a target with
    recurrent layers, carrying that slot's recurrent rows; ``context_positions`` the block
    rows' positions, whose draft context verification stores at the verify's write slots.
    """

    size: int
    draft: tuple[ForwardInputs, ...]
    draft_layers: tuple[ForwardInputs, ...]
    verify: ForwardInputs
    slot_verifies: tuple[ForwardInputs, ...] | None
    context_positions: torch.Tensor


@dataclass(frozen=True, slots=True)
class RoundModels:
    """What rounds run, built before capacity planning beside the scratch: the draft's
    ``config`` and its distinct attention masks (in layer order), the round shape, the
    target with its token head, the draft, K1 (``sampling``) and the proposal step. It
    answers what planning and the runtime size around them."""

    config: DFlashConfig
    masks: tuple[tuple[bool, int | None], ...]
    rounds: RoundShape
    max_kv_tokens: int
    target: BaseLLMModel
    draft: DraftModel
    head: TokenHead
    sampling: RequestLocalChainSampling
    proposal: Proposal
    vocab_size: int
    dtype: torch.dtype
    device: torch.device

    @property
    def block_size(self) -> int:
        return self.rounds.width

    @property
    def max_batch_size(self) -> int:
        return self.rounds.max_rows

    @property
    def rope_axes(self) -> int:
        """The target's rotary position axes: M-RoPE verifies at offset positions."""
        return self.target.rope_position_axes()

    def sampling_law(self, max_rows: int) -> RequestLocalSamplingLaw:
        """The rounds' law (K1), which ordinary steps of at most ``max_rows`` rows sample
        under."""

        return RequestLocalSamplingLaw(self.device, self.vocab_size, max_rows, self.dtype)

    def state_allocations(self, num_slots: int, num_snapshot_rows: int) -> tuple[int, ...]:
        """Bytes of each device allocation the runtime makes for ``num_slots`` request slots
        and a recurrent pool's ``num_snapshot_rows``: the slot rows, then any recurrent
        blocks."""

        allocations = (_slot_rows_bytes(num_slots, self.max_batch_size),)
        layers = self.target.iter_recurrent_layers()
        if not layers:
            return allocations
        return allocations + (
            type(layers[0]).ROUND_TRANSACTION.device_bytes(
                layers,
                num_slots=num_slots,
                shape=self.rounds,
                num_snapshot_rows=num_snapshot_rows,
            ),
        )

    def fixed_query_geometries(self) -> tuple[FixedQueryGeometry, ...]:
        """The rounds' fixed-query attention per KV pool: the target's verify, then the
        draft's layers."""

        q, b, c = self.block_size, self.max_batch_size, self.max_kv_tokens
        masks = (((True, None),), self.masks)
        layers = (next(model.iter_attention_layers()) for model in (self.target, self.draft))
        return tuple(
            FixedQueryGeometry(
                layer.head_dim,
                layer.num_qo_heads,
                layer.num_kv_heads,
                FixedQueryShape(b, q, c, mask, self.dtype),
            )
            for layer, mask in zip(layers, masks, strict=True)
        )


def prepare_rounds(
    dflash: DFlashConfig, plan: Plan, models: HostedModels
) -> tuple[RoundModels, RoundScratch]:
    """The rounds' models and scratch, allocated before capacity planning so it measures
    them as the models' memory: the target's taps, K1, the proposal and the rounds'
    fixed-address buffers, beside ``models``' target and draft."""

    rounds, target, draft = plan.rounds, models.model, models.draft
    assert rounds is not None and draft is not None
    q, b, device = rounds.width, rounds.max_rows, models.device
    checkpoint = models.checkpoint
    v = checkpoint.model_config.vocab_size
    # K1 is a cti-kernels op; import it only where DFlash runs.
    from ctikernels.sampling import RequestLocalChainSampling

    sampling = RequestLocalChainSampling(device, q, v)
    head = target.token_head()
    scratch = RoundScratch.allocate(
        rounds,
        (plan.max_batch_tokens, len(dflash.target_layer_ids) * dflash.model.hidden_size),
        head.hidden_size,
        v,
        checkpoint.facts.stop_token_ids,
        workspace_bytes=max(
            sampling.workspace_size(b, "probabilities"), sampling.workspace_size(b, "dense")
        ),
        dtype=plan.dtype,
        device=device,
    )
    features = target.feature_export()
    if features is None:
        raise ValueError(f"{type(target).__name__} exports no features for a DFlash draft")
    features.bind_features(FeatureRequest(dflash.target_layer_ids), scratch.taps)
    proposal: Proposal = (
        ArgmaxProposal(head, sampling)
        if dflash.dflash2 is None
        else ChainProposal(
            draft.candidate_selector, head, sampling, max_rows=b, positions=q - 1, device=device
        )
    )
    round_models = RoundModels(
        dflash,
        # Each distinct draft attention mask, in layer order.
        tuple(dict.fromkeys(dflash.layer_masks)),
        rounds,
        # Keys a round's row attends at most: the configured sequence capacity and its block.
        plan.max_seq_len + q,
        target,
        draft,
        head,
        sampling,
        proposal,
        v,
        plan.dtype,
        device,
    )
    return round_models, scratch


@dataclass(frozen=True, slots=True)
class RoundBackends:
    """The target's backend, on which a round's verify attends, which addresses every
    round, and other batches on the paged backend it wraps; and the draft's, which its
    layers attend through. Both read one page table."""

    target: SupportsFixedQuery
    draft: SupportsFixedQuery

    @property
    def kv_backends(self) -> tuple[SupportsFixedQuery, ...]:
        """The KV backends in the inference state's order: the target's, then the draft's."""
        return self.target, self.draft


class SpeculativeRuntime:
    """DFlash rounds over a planned inference state, with the scratch the rounds allocated
    before planning.

    Each request row keeps its frontier (int64) and whether it is done on the device; a
    round's ``prepare_round`` and ``commit_block`` (``ctikernels.speculative``) advance both,
    so the next round may run before the host has settled this one. Draft KV below a
    frontier holds the projected taps of committed tokens; a round's block rows hold
    mask-block keys until verification overwrites them with taps. Target and draft pools
    share the scheduler's page mapping and the guard rows. Rounds run on the caller's stream
    without host synchronization.

    A target with recurrent layers verifies blocks through ``recurrent_rounds``: a done row
    is inactive there, and a row whose state an ordinary forward advanced is handed off
    before its next round.
    """

    def __init__(
        self,
        models: RoundModels,
        scratch: RoundScratch,
        state: InferenceState,
        *,
        recurrent_rounds: RecurrentRounds | None,
        backends: RoundBackends,
        token_id_table: torch.Tensor,
        processing: LogitProcessing,
        logprobs: SampledLogprobs,
        padding_rows: range,
        forward_hidden: Callable[[ForwardInputs], torch.Tensor],
        decides: bool,
        publish: Callable[[torch.Tensor], None],
        checks: bool = False,
        differs: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Rounds of ``models`` in ``scratch``'s buffers over ``state``, on ``backends``, the
        target's addressing every round. ``processing`` is the sampler's; ``padding_rows`` follow
        the request rows in the tables, always done. An extended round's target forward runs
        through ``forward_hidden`` (the runner's graphs); ``logprobs`` reads the sampler's
        log-probabilities. Replicas run every round in
        lockstep: the one that ``decides`` draws a sampled round's verdict, its tokens and
        accepted counts, and ``publish`` gives every replica its words, in stream order and
        inside a capture too. A greedy verdict needs neither. ``checks`` verifies each round's
        cache writes and, through ``differs`` (a device bool: whether any replica's tensor
        differs from this one's, in stream order), that every replica proposed the same
        drafts."""

        self._decides, self._publish, self._checks = decides, publish, checks
        self._recurrent = recurrent_rounds
        self._models, self._scratch = models, scratch
        self.block_size = q = models.block_size
        self.max_batch_size = b = models.max_batch_size
        self.vocab_size, self._device = models.vocab_size, models.device
        # Under checks, the replicas compare their proposals.
        self._differs = differs if checks else None
        self._plans: dict[int, _RoundPlan] = {}
        # Per captured round size, the draft's graph, which a round whose target forward
        # also runs prompt extensions replays too, and the target's, one per status slot of
        # the recurrent rounds (one without them). A round's verdict, commit and
        # log-probabilities run eagerly after them, each only when a row needs it.
        self._draft_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._target_graphs: dict[int, tuple[torch.cuda.CUDAGraph, ...]] = {}
        self._buckets = RowBuckets(())
        self._forward_hidden = forward_hidden

        self._target_backend = target_backend = backends.target
        self._draft_backend = backends.draft
        capacity = state.capacity
        frontiers, done = _slot_rows(capacity.num_slots, b, self._device)
        if padding_rows.start != capacity.num_slots or padding_rows.stop > len(done):
            raise ValueError(
                f"padding rows {padding_rows} do not follow the {capacity.num_slots} request "
                f"slots within the {len(done)} planned slot rows"
            )
        # The tables hold exactly the request and padding rows; the kernels match them.
        self._frontiers, self._done = frontiers[: padding_rows.stop], done[: padding_rows.stop]
        self._padding_rows = padding_rows
        self.reset_rows()
        max_seq_len = capacity.max_seq_len
        if token_id_table.shape[1] < max_seq_len + q:
            raise ValueError("the token table is narrower than DFlash rounds address")
        rounds = self._recurrent
        self._snapshot_grid = None
        # Rounds capture recurrent state only into a pool with snapshot rows.
        if rounds is not None and capacity.snapshot_rows:
            self._snapshot_grid = capacity.snapshot_grid
        self._token_id_table = token_id_table
        self._processing = processing
        self._sampled_logprobs = logprobs
        self._cache_tokens = capacity.num_pages * capacity.page_size
        self._max_seq_len = max_seq_len
        mask_indices = tuple(models.masks.index(mask) for mask in models.config.layer_masks)
        for size in range(1, b + 1):
            tokens = size * q
            draft_rows = (scratch.input_ids[:tokens], scratch.rope_rows[:tokens])
            verify_rows = draft_rows
            if models.rope_axes > 1:
                verify_rows = (draft_rows[0], scratch.verify_rope[:tokens])
            metadata = (scratch.args.request_indices[:size], scratch.positions[:tokens])
            drafts = tuple(
                ForwardInputs(
                    *draft_rows,
                    self._draft_backend.create_fixed_query(*metadata, causal=causal, window=window),
                    size,
                    ragged=False,
                )
                for causal, window in models.masks
            )
            verify = ForwardInputs(
                *verify_rows,
                target_backend.create_fixed_query(*metadata, causal=True),
                size,
                ragged=False,
            )
            self._plans[size] = _RoundPlan(
                size,
                drafts,
                tuple(drafts[index] for index in mask_indices),
                verify,
                None
                if rounds is None
                else tuple(
                    replace(
                        verify,
                        recurrent_metadata=rounds.prepare_verify(
                            slot, scratch.k2_slots[:size], scratch.positions[:tokens].view(size, q)
                        ),
                    )
                    for slot in range(rounds.depth)
                ),
                draft_rows[1],
            )
        logger.info(
            "DFlash attention providers: verify %s, draft %s",
            target_backend.providers,
            self._draft_backend.providers,
        )
        self._records = RoundRecords(b, q, self._device)
        # Compile the round kernels now: rounds may run eagerly, without a capture.
        warm_up(
            q,
            max_seq_len,
            scratch.stop_token_ids.numel(),
            device=self._device,
            # K2 rows exist exactly for a target with recurrent layers.
            variants=self._variants(
                rope=models.rope_axes > 1, k2=bool(models.target.iter_recurrent_layers())
            ),
        )

    @staticmethod
    def _variants(*, rope: bool, k2: bool) -> tuple[RoundVariant, ...]:
        """The round bookkeeping ``_draft`` and ``_commit`` run: every round places its rows
        with the sampling cursors (a greedy row draws none); a greedy round commits its
        argmax, a sampled one K1's verdict. M-RoPE and ``k2`` (the recurrent blocks' rows)
        are fixed per model."""

        return (
            RoundVariant("prepare_round", rope=rope, k2=k2, cursors=True),
            RoundVariant("fixed_query_addresses"),
            RoundVariant("commit_block", k2=k2, mode="greedy"),
            RoundVariant("commit_block", k2=k2, mode="sampled"),
        )

    def after_forward(self, forward: TargetForward) -> None:
        """Follow an ordinary target forward: store its tokens' draft context and move each
        entry's frontier just past them."""

        # The forward's tokens lead its rows, and their taps the taps.
        rows = forward.rows
        n = sum(rows.extend_len(index) for index in range(rows.size))
        self._models.draft.store_context(
            self._scratch.taps[:n], forward.positions[:n], forward.batch
        )
        self._advance(forward)

    def _advance(self, forward: TargetForward) -> None:
        """Move the frontier of each entry of a target forward just past its tokens; its
        row is live, and its recurrent state, which the forward advanced, is stale."""

        last = forward.batch.logit_rows
        rows = forward.token_rows[last]
        self._frontiers[rows] = forward.positions[last] + 1
        # Not ``[rows] = False``: assigning a host scalar copies it and syncs the stream.
        self._done.index_fill_(0, rows, False)
        if self._recurrent is not None:
            self._recurrent.mark_stale(forward.rows.slots[: forward.rows.size])

    def submit(
        self,
        op: Round,
        specs: Sequence[RequestSpec],
        extension: TargetForward | None = None,
        *,
        keep: object = None,
    ) -> RoundResult:
        """Enqueue one round over ``op``'s slots on the current stream, its target forward
        running ``extension`` first; its record reaches the host on the result's event, and
        the result holds ``keep`` (host staging the round reads) until then."""

        staged = RoundInput.stage(
            op,
            specs,
            capacity=self.max_batch_size,
            vocab_size=self.vocab_size,
            padding=self._padding_rows,
        )
        run = self.run_round(staged, extension)
        record, host_ready = self._records.copy_out(
            self._scratch.results, self._scratch.logprobs if staged.full else None, run.status
        )
        staging = (op, staged, extension, keep)
        return RoundResult(
            host_ready,
            len(op.slots),
            record,
            self._records,
            run.status,
            run.sampled,
            staging,
            run.faults,
        )

    def run_round(self, staged: RoundInput, extension: TargetForward | None = None) -> RoundRun:
        """Enqueue one staged round; its results land in ``results``, which the next round
        overwrites."""

        size = staged.size
        plan = self._plans.get(size)
        if plan is None:
            raise ValueError(f"a DFlash round holds 1..{self.max_batch_size} requests, not {size}")
        if extension is not None and len(extension.token_rows) + size * self.block_size > len(
            self._scratch.taps
        ):
            raise ValueError("a round's target forward holds more rows than max_batch_tokens")
        # The smallest captured size that holds a plain round, its rows past ``size``
        # padding: the graphs and the tail run that size's plan.
        bucket = None if extension is not None else self._buckets.select(size)
        if bucket is not None:
            plan = self._plans[bucket]
        full = staged.full
        steering = self._steering(plan.size, staged.sampling) if full else None
        self._scratch.args.words.copy_(staged.args.words, non_blocking=True)
        # Nothing between the recurrent rounds' begin and end raises.
        slot = 0
        if self._recurrent is not None:
            slot = self._recurrent.begin(staged.rows, self._frontiers)
        sampled = None
        if extension is not None:
            sampled = self._extended_round(
                plan, staged.verification, slot, extension, full, steering
            )
        else:
            if bucket is None:
                self._draft(plan)
                self._target(plan, slot)
            else:
                self._draft_graphs[bucket].replay()
                self._target_graphs[bucket][slot].replay()
            local = self._block_logits(plan)
            self._finish(plan, staged.verification, slot, local, full, steering)
        faults: tuple[tuple[str, torch.Tensor], ...] = ()
        if self._checks:
            fault = round_write_fault(
                self._target_backend.round_write_rows(size * self.block_size),
                self._scratch.positions[: size * self.block_size],
                self._scratch.round_done[:size],
                cache_tokens=self._cache_tokens,
                max_seq_len=self._max_seq_len,
            )
            flags = [("a round wrote unmapped or shared cache slots", fault)]
            if self._differs is not None:
                flags.append(("replicas proposed different drafts", self._scratch.divergent))
            faults = tuple((message, flag.to("cpu", non_blocking=True)) for message, flag in flags)
        rounds = self._recurrent
        return RoundRun(None if rounds is None else rounds.end(slot, size), sampled, faults)

    @torch.inference_mode()
    def capture(
        self,
        graph_batch_sizes: Sequence[int],
        *,
        stream: torch.cuda.Stream,
        pool: GraphPool,
    ) -> GraphCapture | None:
        """Run every round size whose warm-up fits the KV pools at each verification, full
        or not, so no round compiles on first use, and capture the sizes in
        ``graph_batch_sizes``: each one's draft (``_draft``), which an extended round of that
        size replays too, and its target forward (``_target``) per status slot. A round
        replays the smallest captured size that holds it, padded with done rows; one past
        the largest runs eagerly.

        Runs before admission: warm-up rounds write rows 0..B-1 at distinct cache indices,
        and the row state they move is reset. Returns the capture's report, or None when no
        round fits.
        """

        q = self.block_size
        # Every round size, largest first, whose warm-up blocks fit the KV pools.
        sizes = list(range(min(self.max_batch_size, self._cache_tokens // q), 0, -1))
        if not sizes:
            return None
        graphed = set(graph_batch_sizes).intersection(sizes)
        self._buckets = RowBuckets(tuple(sorted(graphed)))
        args = self._stage_warm_up(sizes[0])
        rows = torch.arange(sizes[0], dtype=torch.int32, device=self._device)
        depth = 1 if self._recurrent is None else self._recurrent.depth

        def capture() -> None:
            for size in sizes:
                args.request_indices[:size].copy_(rows[:size])
                # Every eager tail at each status slot, so none compiles while serving: a
                # full round processes model-dtype logits, or FP32 ones when a row steers.
                for verification, (full, steers) in product(Verification, _WARM_UP_KINDS):
                    args.temperatures.fill_(0.0 if verification is Verification.GREEDY else 1.0)
                    for _ in range(depth):
                        self._warm_up_round(size, verification, full, steers)
                if size not in graphed:
                    continue
                plan = self._plans[size]
                # The draft reads no status slot.
                self._draft_graphs[size] = graph = torch.cuda.CUDAGraph()
                with pool.capture(graph, stream):
                    self._draft(plan)
                targets = []
                for slot in range(depth):
                    targets.append(graph := torch.cuda.CUDAGraph())
                    with pool.capture(graph, stream):
                        self._target(plan, slot)
                self._target_graphs[size] = tuple(targets)

        captured = capture_graphs(
            "round",
            self._buckets.sizes,
            capture,
            pool=pool,
            device=self._device,
            max_rows=sizes[0] * q,
        )
        if self._checks:
            # Checks test every round's cache writes (``run_round``): load their kernels now,
            # on the warm-up rows at every round size (sorts pick kernels by length), so no
            # round loads one while serving.
            for size in sizes:
                round_write_fault(
                    self._target_backend.round_write_rows(size * q),
                    self._scratch.positions[: size * q],
                    self._scratch.round_done[:size],
                    cache_tokens=self._cache_tokens,
                    max_seq_len=self._max_seq_len,
                )
        # Only recurrent rounds have status slots; their handoff is size-independent.
        if graphed and depth > 1:
            self._recurrent.capture_handoffs(pool, stream)
        return captured

    @torch.inference_mode()
    def warm_up_extended(self, extension: TargetForward) -> None:
        """Run one greedy round of row 0 whose target forward runs ``extension`` first, eagerly
        from reset rows, then reset them: the joint forward (mixed-query attention over the
        extension's and the block's rows), its glue and the extension's sampling compile
        before any request, as ``capture`` compiles plain rounds. ``extension`` (the runner's
        warm-up step) holds rows other than row 0, on mapped pages."""

        args = self._stage_warm_up(1)
        args.request_indices[:1].fill_(0)
        args.temperatures.fill_(0.0)
        self.reset_rows()
        rounds, slot = self._recurrent, 0
        if rounds is not None:
            rounds.mark_stale(range(1))
            slot = rounds.begin(range(1), self._frontiers)
        self._extended_round(self._plans[1], Verification.GREEDY, slot, extension, False, None)
        if rounds is not None:
            status = rounds.end(slot, 1)
            for source, target in status.copies:
                target.copy_(source)
            status.settle()
        self.reset_rows()

    def _stage_warm_up(self, rows: int) -> RoundArgs:
        """The round arguments warm-up rounds read, their first ``rows`` request rows mapped
        to distinct cache rows; each round sets its rows and temperatures."""

        args = self._scratch.args
        self._target_backend.map_warm_up(rows)
        args.words.zero_()
        # Row 0 is a live slot's: warm-up rounds capture nothing.
        args.snapshot_rows.fill_(-1)
        args.top_k.fill_(self.vocab_size)
        args.top_p.fill_(1.0)
        args.prompt_lengths.fill_(1)
        args.output_ends.fill_(self._max_seq_len)
        args.repetition.fill_(1.0)
        return args

    def close(self) -> None:
        self._draft_graphs.clear()
        self._target_graphs.clear()

    def reset_rows(self) -> None:
        """Forget every row's device state, as before the first prompt."""

        self._frontiers.zero_()
        self._done.zero_()
        self._done[self._padding_rows.start :].fill_(True)
        if self._recurrent is not None:
            self._recurrent.clear()

    def _warm_up_round(
        self, size: int, verification: Verification, full: bool, steers: bool
    ) -> int:
        """Run a round of ``size`` rows eagerly from reset rows and settle it, a ``full`` one
        whose rows ``steers``; its status slot."""

        self.reset_rows()
        plan, rounds, slot = self._plans[size], self._recurrent, 0
        steering = self._steering(size, (_WARM_UP_STEERING,) * size) if steers else None
        if rounds is None:
            self._round(plan, verification, slot, full, steering)
        else:
            # Every row hands off, compiling the handoff.
            rounds.mark_stale(range(size))
            slot = rounds.begin(range(size), self._frontiers)
            self._round(plan, verification, slot, full, steering)
            status = rounds.end(slot, size)
            for source, target in status.copies:
                target.copy_(source)
            status.settle()
        self.reset_rows()
        return slot

    def _round(
        self,
        plan: _RoundPlan,
        verification: Verification,
        slot: int,
        full: bool,
        steering: Steering | None,
    ) -> None:
        """One round on the current stream, eagerly, as its graphs and tail run it."""

        self._draft(plan)
        self._target(plan, slot)
        self._finish(plan, verification, slot, self._block_logits(plan), full, steering)

    def _extended_round(
        self,
        plan: _RoundPlan,
        verification: Verification,
        slot: int,
        extension: TargetForward,
        full: bool,
        steering: Steering | None,
    ) -> ExtensionOutputs:
        """One round whose target forward runs ``extension`` first: one LM head over its
        last rows and the block rows, its next tokens sampled, and its draft context stored
        with the blocks'. Returns its sampled tokens, on their way to the host."""

        graph = self._draft_graphs.get(plan.size)
        if graph is None:
            self._draft(plan)
        else:
            graph.replay()
        batch, rows = extension.batch, self._verify_rows(plan, slot)
        tokens = len(extension.positions)
        recurrent, paged, fixed = batch.recurrent_metadata, batch.attn_metadata, rows.attn_metadata
        # Ragged, so a breakable graph replays it when its tokens fit a bucket.
        joint = ForwardInputs(
            torch.cat((batch.input_ids, rows.input_ids)),
            _joint_rope_positions(batch.rope_positions, rows.rope_positions),
            MixedQuery(paged, fixed, tokens),
            batch.size,
            ragged=True,
            recurrent_metadata=(
                recurrent
                if recurrent is None or self._recurrent is None
                else self._recurrent.verify_after(recurrent, slot, plan.size)
            ),
            images=batch.images,
            image_encoding=batch.image_encoding,
            logit_rows=batch.logit_rows,
        )
        hidden = self._forward_hidden(joint)
        last = batch.logit_rows
        head = self._models.head
        local = head.local_logits(torch.cat((hidden[:tokens][last], hidden[tokens:])))
        split = len(local) - (len(hidden) - tokens)
        sampled = extension.sample(head.gather_logits(local[:split]), hidden[:tokens])
        # The extension's rows are all tokens (no padding), so the joint rows' taps and write
        # slots are the extension's, then the blocks'.
        context = torch.cat((extension.positions, plan.context_positions))
        self._models.draft.store_context(self._scratch.taps[: len(context)], context, joint)
        self._finish(plan, verification, slot, local[split:], full, steering)
        self._advance(extension)
        return sampled

    def _target(self, plan: _RoundPlan, slot: int) -> None:
        """The target's forward over the block rows at status slot ``slot``, its hidden
        rows into the fixed-address ``verify_hidden``, and the draft context its taps
        store."""

        hidden = self._models.target.forward_hidden(self._verify_rows(plan, slot))
        self._scratch.verify_hidden[: len(hidden)].copy_(hidden)
        context = plan.context_positions
        self._models.draft.store_context(self._scratch.taps[: len(context)], context, plan.verify)

    def _block_logits(self, plan: _RoundPlan) -> torch.Tensor:
        """The block rows' LM head shard logits, of the hidden rows ``_target`` left."""

        rows = plan.size * self.block_size
        return self._models.head.local_logits(self._scratch.verify_hidden[:rows])

    def _finish(
        self,
        plan: _RoundPlan,
        verification: Verification,
        slot: int,
        local: torch.Tensor,
        full: bool,
        steering: Steering | None,
    ) -> None:
        """A round's tail after its LM head, eagerly: verify the block rows' shard logits
        ``local`` (a ``full`` round processes them first, its ``steering`` when a row
        steers), commit, and record a full round's log-probabilities."""

        processing = self._logit_rows(plan.size, steering) if full else None
        verdict = self._verify(plan, verification, local, processing)
        self._commit(plan, verification, slot, verdict.target_ids)
        self._record_logprobs(verdict)

    def _steering(self, size: int, sampling: Sequence[TokenSampling]) -> Steering | None:
        """``sampling``'s steering over a round's first ``size`` rows (those past it steer
        nothing), staged on the host and copied to the device; None when no row steers."""

        if not any(row.steers for row in sampling):
            return None
        staged = self._processing.steering(sampling, size, pin_memory=self._device.type == "cuda")
        return staged.to(self._device)

    def _logit_rows(self, size: int, steering: Steering | None) -> LogitRows:
        """A full round's logits processing over its first ``size`` rows: position i's
        context is the row's tokens through its anchor (the frontiers its draft placed),
        then i drafts."""

        args = self._scratch.args
        return LogitRows(
            args.request_indices[:size],
            args.prompt_lengths[:size],
            self._scratch.results.frontiers[:size] + 1,
            args.presence[:size],
            args.frequency[:size],
            args.repetition[:size],
            steering,
        )

    def _verify(
        self,
        plan: _RoundPlan,
        verification: Verification,
        local: torch.Tensor,
        processing: LogitRows | None,
    ) -> _Verdict:
        """The target's picks for the verify rows from their LM head shard logits ``local``:
        greedy token IDs, or none once sampled into the round's buffers; a full round
        (``processing``) processes the logits first. Rounds pick here alone or beside prompt
        extensions."""

        size, head = plan.size, self._models.head
        greedy = verification is Verification.GREEDY
        if processing is None:
            if greedy:
                return _Verdict(head.gather_argmax(local))
            self._sample(size, head.gather_logits(local), verification.value, min_p=False)
            return _Verdict(None)
        logits = head.gather_logits(local).view(size, self.block_size, -1)
        processed = self._processing.apply(logits, processing, self._scratch.proposals[:size])
        if greedy:
            # Ties go to the smallest ID, as the argmax of the raw logits.
            return _Verdict(processed.argmax(-1).view(-1), logits, processed)
        self._sample(size, processed, verification.value, min_p=True)
        return _Verdict(None, logits, processed)

    def _verify_rows(self, plan: _RoundPlan, slot: int) -> ForwardInputs:
        """The block rows the target verifies, with status slot ``slot``'s recurrent
        metadata, if any."""

        if plan.slot_verifies is None:
            return plan.verify
        return plan.slot_verifies[slot]

    def _draft(self, plan: _RoundPlan) -> None:
        """Place each row's block, draft its proposals and prepare the verify rows'
        attention; reads no status slot and no verification: every row places its sampling
        cursor and draws its drafts under its own law (a temperature-0 row greedily)."""

        size, q, m = plan.size, self.block_size, self._max_seq_len
        args, results = self._scratch.args, self._scratch.results
        rows, start, done = (
            args.request_indices[:size],
            results.frontiers[:size],
            self._scratch.round_done[:size],
        )
        # Every draft input shares the block rows.
        draft, verify = plan.draft[0], plan.verify
        mrope = self._models.rope_axes > 1
        # Each row's state at the round's start; a done row runs as dummies on guard rows.
        prepare_round(
            rows,
            self._frontiers,
            self._done,
            self._token_id_table,
            block_size=q,
            max_seq_len=m,
            mask_token_id=self._models.config.mask_token_id,
            start_frontiers=start,
            round_done=done,
            input_ids=draft.input_ids,
            positions=self._scratch.positions[: size * q],
            rope_positions=draft.rope_positions,
            rope_offsets=args.rope_offsets[:size] if mrope else None,
            verify_rope_positions=verify.rope_positions if mrope else None,
            prompt_lengths=args.prompt_lengths[:size],
            cursors=self._scratch.cursors[:size],
            k2_slots=None if plan.slot_verifies is None else self._scratch.k2_slots[:size],
        )
        self._target_backend.prepare_round(rows, start, done)
        # Draft: [anchor, mask, ...] against context and block keys, under each layer's mask.
        for block in plan.draft:
            block.attn_metadata.prepare()
        hidden = self._models.draft.forward(
            plan.draft_layers, self._models.head.embed(draft.input_ids)
        )
        proposals, inputs = self._scratch.proposals[:size], draft.input_ids.view(size, q)
        self._models.proposal.propose(
            hidden.view(size, q, -1)[:, 1:], inputs[:, 0], proposals, self._chain_rows(size)
        )
        if self._differs is not None:
            self._scratch.divergent.copy_(self._differs(proposals))
        inputs[:, 1:].copy_(proposals)
        # Verify: causal target forward over [anchor, proposals]; its taps become context.
        verify.attn_metadata.prepare()

    def _commit(
        self,
        plan: _RoundPlan,
        verification: Verification,
        slot: int,
        target_ids: torch.Tensor | None,
    ) -> None:
        """Commit each row's accepted prefix on the device; record the round."""

        size, q, m = plan.size, self.block_size, self._max_seq_len
        args, results = self._scratch.args, self._scratch.results
        rows, start, done = (
            args.request_indices[:size],
            results.frontiers[:size],
            self._scratch.round_done[:size],
        )
        greedy = verification is Verification.GREEDY
        k2 = plan.slot_verifies is not None
        # Settlement rejects a raw accepted count outside [-1, Q - 1], K1's range.
        commit_block(
            rows,
            start,
            done,
            tokens=results.tokens[:size],
            accepted=results.accepted[:size],
            output_ends=args.output_ends[:size],
            ignore_eos=args.ignore_eos[:size],
            stop_token_ids=self._scratch.stop_token_ids,
            token_table=self._token_id_table,
            frontier_table=self._frontiers,
            done_table=self._done,
            committed=results.committed[:size],
            record_done=results.done[:size],
            block_size=q,
            max_seq_len=m,
            proposals=self._scratch.proposals[:size] if greedy else None,
            target_ids=target_ids,
            k2_lengths=self._scratch.k2_lengths[:size] if k2 else None,
        )
        if k2:
            lengths = self._scratch.k2_lengths[:size]
            grid = self._snapshot_grid
            if grid is None:
                self._recurrent.commit(slot, lengths)
            else:
                # The first grid point past each frontier, as a position into the block.
                offsets = next_snapshot_position(start, grid) - start
                self._recurrent.commit(slot, lengths, args.snapshot_rows[:size], offsets)

    def _record_logprobs(self, verdict: _Verdict) -> None:
        """A full round's logprobs: the replica that decides reads them, from the p only it
        builds, and publishes them, so every replica reports the round alike."""

        if verdict.logits is None:
            return
        if self._decides:
            self._logprobs(verdict.logits, verdict.processed, sampled=verdict.target_ids is None)
        self._publish(self._scratch.logprobs.words)

    def _logprobs(self, logits: torch.Tensor, processed: torch.Tensor, *, sampled: bool) -> None:
        """Each target token's log-probability and the top alternatives; a processed
        stochastic row reads the p its verdict drew from. Tokens past a row's verdict are
        stale, kept in range."""

        size, args, out = logits.shape[0], self._scratch.args, self._scratch.logprobs
        rows = LogprobRows(
            args.processed_logprobs[:size, None].bool(),
            (args.temperatures[:size] > 0)[:, None],
            TokenSampling.MAX_TOP,
        )
        tokens = self._scratch.results.tokens[:size].clamp(0, self.vocab_size - 1)
        selected, top_ids, top_logprobs = self._sampled_logprobs.read(
            logits,
            tokens,
            rows,
            processed_logits=processed,
            drawn=self._scratch.probabilities[:size] if sampled else None,
            sampling=(
                args.temperatures[:size, None],
                args.top_k[:size, None],
                args.top_p[:size, None],
            ),
        )
        out.logprobs[:size] = selected
        out.top_ids[:size] = top_ids
        out.top_logprobs[:size] = top_logprobs

    def _sample(self, size: int, logits: torch.Tensor, filters: str, *, min_p: bool) -> None:
        """Verify the drafts against target p from the block rows' full-vocabulary
        ``logits`` under the proposal's q (K1): accept draft d with probability
        min(1, p(d)/q(d)); at the first rejection draw from [p - q]+, renormalized; after all
        accept, draw the bonus from the last row. Temperature-0 rows verify greedily; min-p
        applies after the law's filters.

        Every replica computes the logits; the replica that decides draws the verdict, which
        every replica then commits."""

        args, results = self._scratch.args, self._scratch.results
        if self._decides:
            rows, probabilities = self._chain_rows(size), self._scratch.probabilities[:size]
            self._models.sampling.probabilities(
                logits.view(size, self.block_size, -1),
                rows.temperatures,
                args.top_k[:size],
                args.top_p[:size],
                min_p=args.min_p[:size] if min_p else None,
                request_indices=rows.request_indices,
                workspace=rows.workspace,
                out_probs=probabilities,
                filters=filters,
            )
            # A round's draft i draws at response position cursor + i, where prepare_round
            # placed the cursor one past the anchor's response position.
            self._models.proposal.verify(
                probabilities,
                self._scratch.proposals[:size],
                rows,
                out_tokens=results.tokens[:size],
                accepted=results.accepted[:size],
            )
        self._publish(results.verdict)

    def _chain_rows(self, size: int) -> ChainRows:
        """K1's inputs for a round's first ``size`` rows, which read seeds by row number."""

        args = self._scratch.args
        return ChainRows(
            args.temperatures[:size],
            self._scratch.rows[:size],
            args.seeds[:size],
            self._scratch.cursors[:size],
            self._scratch.workspace,
        )


def round_write_fault(
    writes: torch.Tensor,
    positions: torch.Tensor,
    done: torch.Tensor,
    *,
    cache_tokens: int,
    max_seq_len: int,
) -> torch.Tensor:
    """Whether a live row's position below ``max_seq_len`` writes an unmapped cache slot, or
    one another such row writes, as a device bool computed without host synchronization;
    the scheduler reserves and maps a round's pages before it runs."""

    live = done.logical_not().repeat_interleave(len(writes) // len(done))
    live &= positions < max_seq_len
    unmapped = live & ((writes < 0) | (writes >= cache_tokens))
    # Rows that write nothing checked take distinct negative values, which never repeat.
    others = -1 - torch.arange(len(writes), device=writes.device, dtype=writes.dtype)
    slots = torch.where(live, writes, others).sort().values
    return unmapped.any() | (slots[1:] == slots[:-1]).any()


def _slot_rows_bytes(num_slots: int, max_rows: int) -> int:
    """Bytes of the rounds' per-slot rows: an int64 frontier and a done flag for each request
    slot and padding row."""

    return 9 * (num_slots + RowBuckets.padding_rows_bound(max_rows))


def _slot_rows(
    num_slots: int, max_rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The zeroed frontiers and done flags, in one allocation of ``_slot_rows_bytes``."""

    rows = num_slots + RowBuckets.padding_rows_bound(max_rows)
    backing = torch.zeros(_slot_rows_bytes(num_slots, max_rows), dtype=torch.uint8, device=device)
    return backing[: 8 * rows].view(torch.int64), backing[8 * rows :].view(torch.bool)


def _joint_rope_positions(extension: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """The extension tokens' rotary positions, then the rows': ``[axes, tokens]`` when the
    extension carries M-RoPE rows, every axis reading a row's one position."""

    if extension.ndim == 1:
        return torch.cat((extension, rows))
    return torch.cat((extension, rows.expand(extension.shape[0], -1)), dim=1)
