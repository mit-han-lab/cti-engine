# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/engine/sample.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, ClassVar, List, Protocol

import torch

from cti.inference._step import Sampled
from cti.sampling import TokenSampling
from ctikernels.sampling import (
    CategoricalSampling,
    LogitPenalties,
    RequestLocalCategorical,
    RowArgmax,
    TopLogprobs,
    steer_logits_,
)

if TYPE_CHECKING:
    from cti.inference._step import RequestSpec, Row
    from cti.utils import DeviceCapability


@dataclass(frozen=True, slots=True)
class DeviceSampled:
    """Each row's token and, when asked for, its log-probability and the top alternatives
    (rows may carry leading axes, as a DFlash round's ``[B, Q]``), on the device."""

    token_ids: torch.Tensor
    logprobs: torch.Tensor | None = None
    top_ids: torch.Tensor | None = None
    top_logprobs: torch.Tensor | None = None

    @classmethod
    def receiving(
        cls, rows: int, *, logprobs: bool, top: int, device: torch.device
    ) -> tuple[DeviceSampled, torch.Tensor]:
        """Uninitialized results of ``rows`` rows, int32 tokens and alternatives with FP32
        scores, as views of one byte buffer, which ``packed`` of a sender's fills."""

        shapes = [((rows,), torch.int32)]
        if logprobs:
            shapes.append(((rows,), torch.float32))
        if top:
            shapes += [((rows, top), torch.int32), ((rows, top), torch.float32)]
        # Every column holds 4-byte elements, so each view starts aligned.
        buffer = torch.empty(
            4 * sum(math.prod(shape) for shape, _ in shapes), dtype=torch.uint8, device=device
        )
        views, start = [], 0
        for shape, dtype in shapes:
            size = 4 * math.prod(shape)
            views.append(buffer[start : start + size].view(dtype).view(shape))
            start += size
        token_ids, *rest = views
        scores = rest.pop(0) if logprobs else None
        return cls(token_ids, scores, *rest), buffer

    def packed(self) -> torch.Tensor:
        """Its tensors' bytes in ``receiving``'s layout: int32 tokens first."""

        return torch.cat(
            [
                tensor.reshape(-1).view(torch.uint8)
                for tensor in (self.token_ids, self.logprobs, self.top_ids, self.top_logprobs)
                if tensor is not None
            ]
        )

    def to_cpu(self) -> DeviceSampled:
        """Its tensors on their way to the host, behind the current stream."""

        return DeviceSampled(
            *(
                None if tensor is None else tensor.to("cpu", non_blocking=True)
                for tensor in (self.token_ids, self.logprobs, self.top_ids, self.top_logprobs)
            )
        )

    def host(self) -> Sampled:
        """Its host tensors as bytes, once they have arrived."""

        top = 0 if self.top_ids is None else self.top_ids.shape[-1]
        return Sampled(
            *(
                None if tensor is None else tensor.numpy().tobytes()
                for tensor in (self.token_ids.int(), self.logprobs)
            ),
            top,
            *(
                None if tensor is None else tensor.numpy().tobytes()
                for tensor in (self.top_ids, self.top_logprobs)
            ),
        )


@dataclass(frozen=True, slots=True)
class Steering:
    """Logit biases and min_tokens stop masks for up to ``capacity`` requests, as cti-kernels'
    ``steer_logits_`` reads them: each request's min_tokens, then CSR pointers into one id
    list for the stop tokens its min_tokens masks and for its biased tokens, whose biases sit
    at the same places of ``values``. ``words`` is int32 ``[min_tokens C | stop pointers
    C + 1 | bias pointers C + 1 | ids]``; ``LogitProcessing.steering`` stages both on the
    host, each moved by one copy."""

    words: torch.Tensor
    values: torch.Tensor
    capacity: int

    def to(self, device: torch.device) -> Steering:
        return Steering(
            self.words.to(device, non_blocking=True),
            self.values.to(device, non_blocking=True),
            self.capacity,
        )

    def apply_(self, logits: torch.Tensor, outputs: torch.Tensor) -> None:
        """Steer contiguous FP32 ``[N, Q, V]`` ``logits`` of the first N requests in place,
        position i of request n following ``outputs[n] + i`` outputs."""

        n, q, v = logits.shape
        c, device = self.capacity, logits.device
        ids = self.words[3 * c + 2 :]
        steer_logits_(
            logits.view(n * q, v),
            # A copy: at N = 1 a reshape of the expansion stays a stride-0 view.
            torch.arange(n, dtype=torch.int32, device=device)[:, None]
            .expand(n, q)
            .contiguous()
            .view(-1),
            (outputs[:, None] + torch.arange(q, device=device)).view(-1),
            min_tokens=self.words[:c],
            stop_indptr=self.words[c : 2 * c + 1],
            stop_ids=ids,
            bias_indptr=self.words[2 * c + 1 : 3 * c + 2],
            bias_ids=ids,
            bias_values=self.values,
        )


@dataclass(frozen=True, slots=True)
class LogitRows:
    """Each row's logits processing and where its context lies: the token-table row, the
    prompt's length and the context's (prompt plus output so far, the last input token
    included), its penalties, and its requests' ``Steering`` when a row steers (None:
    none)."""

    rows: torch.Tensor
    prompt_lengths: torch.Tensor
    lengths: torch.Tensor
    presence: torch.Tensor
    frequency: torch.Tensor
    repetition: torch.Tensor
    steering: Steering | None = None


@dataclass(frozen=True, slots=True)
class LogprobRows:
    """Which rows read processed log-probabilities (None: none), which of those are
    stochastic, and how many top alternatives the batch returns (0: none)."""

    processed: torch.Tensor | None
    stochastic: torch.Tensor | None
    top: int = 0


@dataclass
class BatchSamplingArgs:
    """``seeds`` holds each row's request seed, None for a row the shared generator draws;
    it is None when no row has one. ``positions`` are the rows' response positions.
    ``min_p``, ``processing`` and ``logprobs`` are None when no row asks for them."""

    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    seeds: tuple[int | None, ...] | None = None
    positions: tuple[int, ...] | None = None
    min_p: torch.Tensor | None = None
    processing: LogitRows | None = None
    logprobs: LogprobRows | None = None
    # Rows drawn from the whole filtered distribution: min-p, or processed logprobs.
    redraw: tuple[int, ...] = ()


class LogitProcessing:
    """vLLM's logits processing on ``[N, V]`` or ``[N, Q, V]`` logits, in its order: until a
    row has min_tokens outputs, its stop tokens (the model's and its own) are masked; its
    logit biases add; then cti-kernels' ``LogitPenalties`` apply the penalties from its
    context in a token table, with vLLM V2's arithmetic: repetition divides a positive logit
    (multiplies a negative one) of every token in the prompt or the output, frequency
    subtracts its weight times the token's output count and presence its weight once per
    output token seen. Position i of a multi-token row (a DFlash round's) follows i more
    outputs: the i lookahead tokens before it."""

    def __init__(
        self, token_table: torch.Tensor, vocab_size: int, stop_token_ids: Sequence[int] = ()
    ) -> None:
        self._penalties = LogitPenalties(token_table, vocab_size)
        self._token_table = token_table
        self._vocab_size = vocab_size
        self._stop_token_ids = tuple(stop_token_ids)

    def warm_up(self, rows: int, dtype: torch.dtype) -> None:
        """Process ``rows`` ``[N, V]`` rows of ``dtype`` logits, unsteered and steered (FP32),
        under every penalty, so no request compiles them; rows read token-table row 0."""

        device = self._token_table.device

        def full(value: float, dtype: torch.dtype) -> torch.Tensor:
            return torch.full((rows,), value, dtype=dtype, device=device)

        steers = TokenSampling(min_tokens=1, logit_bias=((0, 0.0),))
        steering = self.steering((steers,) * rows, rows, pin_memory=device.type == "cuda")
        for steered in (None, steering.to(device)):
            logits = torch.zeros(rows, self._vocab_size, dtype=dtype, device=device)
            self.apply(
                logits,
                LogitRows(
                    full(0, torch.int64),
                    full(1, torch.int64),
                    full(2, torch.int64),
                    full(0.5, torch.float32),
                    full(0.5, torch.float32),
                    full(1.1, torch.float32),
                    steered,
                ),
            )

    def steering(
        self, rows: Sequence[TokenSampling], capacity: int, *, pin_memory: bool
    ) -> Steering:
        """``rows``' steering on the host, the first of ``capacity`` requests (those past
        them steer nothing): each row's logit biases and the tokens its min_tokens masks,
        the model's stop tokens, ``ignore_eos`` notwithstanding (vLLM's), and its own.
        Admission keeps biases inside the vocabulary; a stop token outside it cannot be
        drawn, so its mask drops."""

        stops = [
            sorted(
                {t for t in (*self._stop_token_ids, *row.stop_token_ids) if t < self._vocab_size}
            )
            if row.min_tokens
            else []
            for row in rows
        ]
        pad = capacity - len(rows)
        stop_ends = list(accumulate(map(len, stops), initial=0))
        bias_ends = list(accumulate((len(row.logit_bias) for row in rows), initial=stop_ends[-1]))
        words = [
            *(row.min_tokens for row in rows),
            *[0] * pad,
            *stop_ends,
            *[stop_ends[-1]] * pad,
            *bias_ends,
            *[bias_ends[-1]] * pad,
            *(token for row_stops in stops for token in row_stops),
            *(token for row in rows for token, _ in row.logit_bias),
        ]
        values = [0.0] * stop_ends[-1] + [bias for row in rows for _, bias in row.logit_bias]
        return Steering(
            torch.tensor(words, dtype=torch.int32, pin_memory=pin_memory),
            torch.tensor(values, dtype=torch.float32, pin_memory=pin_memory),
            capacity,
        )

    @staticmethod
    def steer(logits: torch.Tensor, rows: LogitRows) -> torch.Tensor:
        """FP32 ``[N, Q, V]`` ``logits`` in new storage, biased, then masked, position i
        following i more outputs; ``rows`` steer."""

        assert rows.steering is not None
        steered = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
        steered.copy_(logits)
        rows.steering.apply_(steered, rows.lengths - rows.prompt_lengths)
        return steered

    def apply(
        self, logits: torch.Tensor, rows: LogitRows, lookahead: torch.Tensor | None = None
    ) -> torch.Tensor:
        """FP32 processed ``logits`` (``[N, V]``, or ``[N, Q, V]`` with ``lookahead
        [N, Q - 1]``, the outputs before positions 1..Q - 1), in new storage."""

        flat = logits.dim() == 2
        if flat:
            logits = logits[:, None]
        if rows.steering is not None:
            logits = self.steer(logits, rows)
        out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
        self._penalties.apply(logits, rows, lookahead, out=out)
        return out[:, 0] if flat else out


class SampledLogprobs:
    """Log-probabilities, one definition for every path that samples, through cti-kernels'
    ``TopLogprobs``. A raw row reads the log-softmax of its logits; a processed row reads
    what it is drawn from, after ``LogitProcessing``, temperature, top-k, top-p and min-p (a
    greedy row the log-softmax after ``LogitProcessing`` only). Rows may carry leading axes,
    as a DFlash round's ``[B, Q]``. A token that cannot be drawn reads ``FLOOR``, as OpenAI
    reports one."""

    FLOOR = -9999.0

    def __init__(self, device: torch.device, vocab_size: int) -> None:
        # A vocabulary smaller than the API's limit lists all of itself.
        self._max_top = min(TokenSampling.MAX_TOP, vocab_size)
        self._op = TopLogprobs(device, vocab_size, self._max_top)
        self._vocab_size = vocab_size

    def warm_up(self, logits_dtype: torch.dtype) -> None:
        """Launch every plan the op runs on ``logits_dtype`` and FP32 (processed) logits, raw
        and processed, so its first-use builds and their stack growth happen before the
        inference state is sized; its plans do not depend on the row count."""

        dtypes = tuple(dict.fromkeys((logits_dtype, torch.float32)))
        self._op.warm_up([1], dtypes=dtypes, values=("logprobs",))

    def read(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        rows: LogprobRows,
        *,
        processed_logits: torch.Tensor | None = None,
        drawn: torch.Tensor | None = None,
        sampling: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Each token's log-probability, and the ``rows.top`` likeliest IDs (int32) with
        theirs, best first (None without). ``rows`` broadcast over ``tokens``' axes.
        ``processed_logits`` are after ``LogitProcessing`` (None: none); ``drawn`` the
        stochastic rows' sampling distribution (None: no stochastic row), which they read,
        with ``sampling``, their temperatures, top-k and top-p, ordering its ties."""

        shape, vocab = tokens.shape, self._vocab_size
        logits, tokens = logits.reshape(-1, vocab), tokens.reshape(-1).int()
        n = tokens.numel()

        def flat(flags: torch.Tensor | None) -> torch.Tensor | None:
            return None if flags is None else flags.expand(shape).reshape(-1)

        processed, stochastic = flat(rows.processed), flat(rows.stochastic)
        groups: list[tuple[torch.Tensor | None, dict]] = []
        if processed is None:
            groups.append((None, {"logits": logits}))
        else:
            source = logits if processed_logits is None else processed_logits.reshape(-1, vocab)
            reads_drawn = drawn is not None and stochastic is not None
            derived = processed & ~stochastic if reads_drawn else processed
            groups += [(~processed, {"logits": logits}), (derived, {"logits": source})]
            if reads_drawn:
                assert sampling is not None and stochastic is not None
                temperatures, top_k, top_p = (
                    value.expand(shape).reshape(-1).contiguous() for value in sampling
                )
                groups.append(
                    (
                        processed & stochastic,
                        {
                            "logits": source,
                            "mode": "processed",
                            "probabilities": drawn.reshape(-1, vocab),
                            "temperatures": temperatures.float(),
                            "filter_top_k": top_k.int(),
                            "top_p": top_p.float(),
                        },
                    )
                )
        device = tokens.device
        indices = torch.arange(n, dtype=torch.int32, device=device)
        count = torch.full((1,), n, dtype=torch.int32, device=device)
        width = min(rows.top, self._max_top)
        top = torch.full((n,), width, dtype=torch.int32, device=device)
        selected = top_ids = top_logprobs = None
        for group, arguments in groups:
            out = (
                torch.empty(n, dtype=torch.float32, device=device),
                torch.empty(n, dtype=torch.int32, device=device),
                torch.empty(n, self._max_top, dtype=torch.int32, device=device),
                torch.empty(n, self._max_top, dtype=torch.float32, device=device),
            )
            self._op.run(
                row_indices=indices,
                num_rows=count,
                selected_ids=tokens if group is None else torch.where(group, tokens, -1),
                top_k=top,
                floor=self.FLOOR,
                out_selected_logprobs=out[0],
                out_ranks=out[1],
                out_top_ids=out[2],
                out_top_logprobs=out[3],
                **arguments,
            )
            if selected is None or group is None:
                selected, top_ids, top_logprobs = out[0], out[2], out[3]
            else:
                selected = torch.where(group, out[0], selected)
                top_ids = torch.where(group[:, None], out[2], top_ids)
                top_logprobs = torch.where(group[:, None], out[3], top_logprobs)
        assert selected is not None and top_ids is not None and top_logprobs is not None
        if not width:
            return selected.view(shape), None, None
        return (
            selected.view(shape),
            top_ids[:, :width].reshape(*shape, width),
            top_logprobs[:, :width].reshape(*shape, width),
        )

    @staticmethod
    def filter(
        probabilities: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        min_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[..., V]`` probabilities restricted to each row's top-k, then to the smallest
        top-p mass of what remains, then to tokens at least min-p times the likeliest,
        renormalized; ties keep the smaller ID. ``None`` filters are identity."""

        if top_k is not None or top_p is not None:
            ordered, order = probabilities.sort(dim=-1, descending=True, stable=True)
            if top_k is not None:
                ranks = torch.arange(ordered.shape[-1], device=ordered.device)
                ordered = ordered.masked_fill(ranks >= top_k[..., None], 0.0)
            ordered = ordered / ordered.sum(-1, keepdim=True)
            if top_p is not None:
                before = ordered.cumsum(-1) - ordered
                ordered = ordered.masked_fill(before >= top_p[..., None], 0.0)
            probabilities = torch.zeros_like(probabilities).scatter_(-1, order, ordered)
        if min_p is not None:
            floor = min_p[..., None] * probabilities.amax(-1, keepdim=True)
            probabilities = probabilities.masked_fill(probabilities < floor, 0.0)
        return probabilities / probabilities.sum(-1, keepdim=True)


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    pinned = device.type == "cuda"
    return torch.tensor(data, dtype=dtype, pin_memory=pinned).to(device, non_blocking=True)


class SamplingLaw(Protocol):
    """How a ``Sampler`` turns stochastic rows' logits into probabilities and draws them.
    ``SEEDS_EVERY_ROW``: every row it draws comes from its request's own stream; otherwise
    a batch's rows may also share one generator."""

    SEEDS_EVERY_ROW: ClassVar[bool]

    def row_parameters(self, rows: Sequence[TokenSampling]) -> tuple[list[float], list[float]]:
        """Each row's temperature (0 for a greedy one) and top-p, as this law reads them."""
        ...

    def probabilities(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """``[B, V]`` probabilities of ``[B, V]`` stochastic logits."""
        ...

    def sample(self, probabilities: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """One token per row of ``probabilities``."""
        ...

    def draw_filters(
        self, top_k: torch.Tensor, top_p: torch.Tensor
    ) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], ...]:
        """The top-k and top-p filters its draws take apart, from ``top_k`` and ``top_p``."""
        ...

    def drawn(self, probabilities: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """The distribution ``sample`` draws each row of ``probabilities`` from."""
        ...

    def draw(self, drawn: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """One token per row of a ``drawn`` distribution, which holds every filter."""
        ...

    def warm_up(self) -> None:
        """Build what this law runs apart from the draws a ``Sampler`` warms up."""
        ...


class CategoricalDraw:
    """One token per probability row: seeded rows from their request-local streams, the
    rest from the shared generator."""

    def __init__(self, shared: CategoricalSampling) -> None:
        self._shared = shared
        self._request_local = RequestLocalCategorical()

    def __call__(
        self,
        probabilities: torch.Tensor,
        args: BatchSamplingArgs,
        *,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if args.seeds is None:
            return self._shared.sample(probabilities, top_k=top_k, top_p=top_p)
        if args.positions is None:
            raise ValueError("seeded sampling requires request-local positions")
        seeded = [row for row, seed in enumerate(args.seeds) if seed is not None]
        if not seeded:  # a subset of a batch, such as min-p's redraw rows, may hold none
            return self._shared.sample(probabilities, top_k=top_k, top_p=top_p)
        if len(seeded) == len(args.seeds):
            return self._request_local.sample(
                probabilities, args.seeds, args.positions, top_k=top_k, top_p=top_p
            )
        # A seeded row draws from its own stream, whichever rows share its batch. The
        # shared generator draws the whole batch and the seeded rows' draws overwrite
        # theirs, which takes one row upload, gather and scatter; drawing only the unseeded
        # rows would take a second of each.
        rows = make_device_tensor(seeded, torch.int64, probabilities.device)
        local = self._request_local.sample(
            probabilities[rows],
            tuple(args.seeds[row] for row in seeded),
            tuple(args.positions[row] for row in seeded),
            top_k=None if top_k is None else top_k[rows],
            top_p=None if top_p is None else top_p[rows],
        )
        sampled = self._shared.sample(probabilities, top_k=top_k, top_p=top_p)
        sampled[rows] = local.to(sampled.dtype)
        return sampled


class LegacySamplingLaw:
    """Clamps temperatures and top-p, takes an FP32 temperature softmax and filters top-k
    and top-p in the draw."""

    SEEDS_EVERY_ROW: ClassVar[bool] = False
    MIN_TEMPERATURE: ClassVar[float] = 1e-6
    MIN_TOP_P: ClassVar[float] = 1e-6

    def __init__(self, capability: DeviceCapability | None) -> None:
        self.enable_pdl = capability is not None and capability.is_at_least(9, 0)
        self._sampling = CategoricalSampling()
        self._draw = CategoricalDraw(self._sampling)

    def row_parameters(self, rows: Sequence[TokenSampling]) -> tuple[list[float], list[float]]:
        temperatures = [
            max(0.0 if row.is_greedy else row.temperature, self.MIN_TEMPERATURE) for row in rows
        ]
        top_ps = [min(max(row.top_p, self.MIN_TOP_P), 1.0) for row in rows]
        return temperatures, top_ps

    def probabilities(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        assert args.temperatures is not None
        return self._sampling.softmax(logits.float(), args.temperatures, enable_pdl=self.enable_pdl)

    def sample(self, probabilities: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        return self._draw(probabilities, args, top_k=args.top_k, top_p=args.top_p)

    def draw(self, drawn: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        return self._draw(drawn, args)

    @staticmethod
    def draw_filters(
        top_k: torch.Tensor, top_p: torch.Tensor
    ) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], ...]:
        return (None, None), (top_k, None), (None, top_p), (top_k, top_p)

    @staticmethod
    def drawn(probabilities: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """The draw applies top-k and top-p, so they apply here too."""
        return SampledLogprobs.filter(probabilities, args.top_k, args.top_p, args.min_p)

    def warm_up(self) -> None:
        """Nothing beyond the draws a ``Sampler`` warms up."""


class RequestLocalSamplingLaw:
    """Builds p as speculative rounds verify (K1): the argmax at temperature 0, else FP32
    temperature, top-k, top-p, smaller IDs winning ties; temperatures and top-p unclamped,
    the draw unfiltered. DFlash, which builds it, seeds every request where it enters."""

    SEEDS_EVERY_ROW: ClassVar[bool] = True

    def __init__(
        self, device: torch.device, vocab_size: int, capacity: int, dtype: torch.dtype
    ) -> None:
        """At most ``capacity`` rows of ``dtype`` logits over ``vocab_size`` IDs."""
        # K1 is a cti-kernels op; import it only where its law runs. Its blocks hold 2 to 32
        # positions; a row here uses one.
        from ctikernels.sampling import RequestLocalChainSampling

        self._sampling = sampling = RequestLocalChainSampling(device, 2, vocab_size)
        self._draw = CategoricalDraw(CategoricalSampling())
        self._dtype = dtype
        self._capacity = capacity
        self._rows = torch.arange(capacity, dtype=torch.int32, device=device)
        self._workspace = torch.empty(
            sampling.workspace_size(capacity, "probabilities", length=1),
            dtype=torch.uint8,
            device=device,
        )

    @staticmethod
    def row_parameters(rows: Sequence[TokenSampling]) -> tuple[list[float], list[float]]:
        temperatures = [0.0 if row.is_greedy else row.temperature for row in rows]
        return temperatures, [row.top_p for row in rows]

    def probabilities(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """``None`` filters are identity."""

        assert args.temperatures is not None
        size, vocab = logits.shape
        options = {"device": logits.device}
        k = torch.full((size,), vocab, dtype=torch.int32, **options)
        k = k if args.top_k is None else args.top_k
        p = torch.ones(size, **options) if args.top_p is None else args.top_p
        out = torch.empty((size, 1, vocab), dtype=torch.float32, **options)
        self._sampling.probabilities(
            logits.unsqueeze(1),
            args.temperatures,
            k,
            p,
            request_indices=self._rows[:size],
            workspace=self._workspace,
            out_probs=out,
        )
        return out[:, 0]

    def sample(self, probabilities: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        # The probabilities already hold top-k and top-p.
        return self._draw(probabilities, args)

    def draw(self, drawn: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        return self._draw(drawn, args)

    @staticmethod
    def draw_filters(
        _top_k: torch.Tensor, _top_p: torch.Tensor
    ) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], ...]:
        """The probabilities apply any filters in the same kernels and leave none to the
        draw, so the unfiltered draw is the only one."""
        return ((None, None),)

    @staticmethod
    def drawn(probabilities: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.min_p is None:
            return probabilities
        return SampledLogprobs.filter(probabilities, None, None, args.min_p)

    def warm_up(self) -> None:
        # One cti-kernels build per logits dtype covers every row count up to the capacity:
        # the model's, and FP32 for logits ``LogitProcessing`` processed.
        self._sampling.warm_up(
            batch_capacity=self._capacity,
            dtypes=tuple(dict.fromkeys((self._dtype, torch.float32))),
            filters=("auto",),
            min_p=(False,),
            modes=(),
            strided_logits=False,
        )


class Sampler:
    def __init__(
        self,
        device: torch.device,
        vocab_size: int,
        *,
        law: SamplingLaw,
        processing: LogitProcessing,
        batch_capacity: int,
        logits_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Samples up to ``batch_capacity`` rows of ``logits_dtype`` logits under ``law``,
        processed by ``processing``."""
        self.device = device
        self.vocab_size = vocab_size
        self._law = law
        self._processing = processing
        self._logits_dtype = logits_dtype
        self.logprobs = SampledLogprobs(device, vocab_size)
        self._argmax = RowArgmax(device)
        # Zeroed once; every call leaves it zero.
        self._argmax_workspace = torch.zeros(
            RowArgmax.workspace_size(batch_capacity), dtype=torch.uint8, device=device
        )

    def warm_up(self, batch_sizes: Sequence[int]) -> None:
        """Draw through every path this sampler can take, at each batch size, and process
        logits as a request may (penalties, steering), so no request pays a provider's
        first-use build."""

        largest = max(batch_sizes)
        # One build per logits dtype, raw or processed, covers every row count up to it.
        for dtype in dict.fromkeys((self._logits_dtype, torch.float32)):
            self._argmax.warm_up(
                torch.zeros(largest, self.vocab_size, dtype=dtype, device=self.device),
                torch.empty(largest, dtype=torch.int64, device=self.device),
                self._argmax_workspace,
            )
        # Largest first: each smaller size reuses the blocks the larger freed, so warm-up
        # holds one size's working set, not the sum over sizes.
        for size in sorted(batch_sizes, reverse=True):
            logits = torch.zeros(
                size, self.vocab_size, dtype=self._logits_dtype, device=self.device
            )
            for args in self._stochastic_paths(size):
                self.sample(logits, args)
        self._processing.warm_up(largest, self._logits_dtype)
        self._law.warm_up()
        self.logprobs.warm_up(self._logits_dtype)

    def prepare(self, entries: Sequence[Row], specs: Sequence[RequestSpec]) -> BatchSamplingArgs:
        """Sampling arguments for ``entries``, each under its request's ``specs`` entry."""

        rows = [spec.sampling for spec in specs]
        device = self.device

        def tensor(values: list, dtype: torch.dtype) -> torch.Tensor:
            return make_device_tensor(values, dtype, device)

        processing = None
        if any(row.processes_logits for row in rows):
            steering = None
            if any(row.steers for row in rows):
                steering = self._processing.steering(
                    rows, len(rows), pin_memory=device.type == "cuda"
                ).to(device)
            processing = LogitRows(
                tensor([entry.request_index for entry in entries], torch.int64),
                tensor([spec.prompt_len for spec in specs], torch.int64),
                tensor([entry.sequence_len for entry in entries], torch.int64),
                tensor([row.presence_penalty for row in rows], torch.float32),
                tensor([row.frequency_penalty for row in rows], torch.float32),
                tensor([row.repetition_penalty for row in rows], torch.float32),
                steering,
            )
        logprobs = None
        if any(row.logprobs is not None for row in rows):
            # Rows that ask for none ignore theirs.
            processed = [row.reads_processed for row in rows]
            stochastic = [not row.is_greedy for row in rows]
            logprobs = LogprobRows(
                tensor(processed, torch.bool) if any(processed) else None,
                tensor(stochastic, torch.bool) if any(stochastic) else None,
                max(row.top_logprobs for row in rows),
            )
        if all(row.is_greedy for row in rows):
            return BatchSamplingArgs(None, processing=processing, logprobs=logprobs)

        ts, top_ps = self._law.row_parameters(rows)
        top_ks = [row.top_k_within(self.vocab_size) for row in rows]
        top_k, top_p, min_p = None, None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = tensor(top_ks, torch.int32)
        if any(p < 1.0 for p in top_ps):
            top_p = tensor(top_ps, torch.float32)
        if any(row.has_min_p for row in rows):
            min_p = tensor([row.min_p for row in rows], torch.float32)
        seeds = None
        positions = None
        if any(spec.seed is not None for spec in specs):
            seeds = tuple(spec.seed for spec in specs)
            positions = tuple(entry.sampling_position for entry in entries)
        return BatchSamplingArgs(
            tensor(ts, torch.float32),
            top_k=top_k,
            top_p=top_p,
            seeds=seeds,
            positions=positions,
            min_p=min_p,
            processing=processing,
            logprobs=logprobs,
            redraw=tuple(
                index
                for index, row in enumerate(rows)
                if not row.is_greedy and (row.has_min_p or row.reads_processed)
            ),
        )

    # LM-head rows a prompt is scored in at a time, so no prompt holds all of its logits.
    PROMPT_ROWS: ClassVar[int] = 256

    def score_prompts(
        self,
        entries: Sequence[Row],
        specs: Sequence[RequestSpec],
        targets: torch.Tensor,
        hidden: torch.Tensor,
        vocab_logits: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[tuple[int, DeviceSampled] | None, ...]:
        """Each extension's prompt log-probabilities, raw, with its alternatives, and the
        prompt position they start at: extension token j scores ``targets[j]``, the next
        prompt token (host int32, -1: none, as the prompt's last token). ``vocab_logits``
        projects rows of ``hidden``."""

        results: list[tuple[int, DeviceSampled] | None] = []
        offset = 0
        for entry, spec in zip(entries, specs, strict=True):
            count = spec.sampling.prompt_logprobs
            rows = targets[offset : offset + entry.extend_len]
            rows = rows[: int((rows >= 0).sum())]
            if count is None or len(rows) == 0:
                results.append(None)
            else:
                rows = rows.to(self.device, non_blocking=True)
                raw = LogprobRows(None, None, count)
                parts = []
                for start in range(0, len(rows), self.PROMPT_ROWS):
                    stop = min(start + self.PROMPT_ROWS, len(rows))
                    logits = vocab_logits(hidden[offset + start : offset + stop])
                    parts.append(self.logprobs.read(logits, rows[start:stop], raw))
                columns = (
                    None if column[0] is None else torch.cat(column)
                    for column in zip(*parts, strict=True)
                )
                results.append((entry.kv_prefix_len + 1, DeviceSampled(rows, *columns)))
            offset += entry.extend_len
        return tuple(results)

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> DeviceSampled:
        """Sample each row of ``logits``, with the log-probabilities ``args.logprobs`` asks
        for (``SampledLogprobs``)."""

        with torch.cuda.nvtx.range("Sampler"):
            processed_logits = None
            if args.processing is not None:
                processed_logits = self._processing.apply(logits, args.processing)
            processed = logits if processed_logits is None else processed_logits
            drawn = None
            if args.temperatures is None:  # greedy sampling
                token_ids = torch.empty(len(processed), dtype=torch.int64, device=self.device)
                self._argmax(processed, token_ids, self._argmax_workspace)
            else:
                probabilities = self._law.probabilities(processed, args)
                token_ids = self._law.sample(probabilities, args)
                if args.redraw:
                    # Those rows' tokens and logprobs read one distribution; the other rows
                    # draw as they would in any batch.
                    drawn = self._law.drawn(probabilities, args)
                    # A pinned, asynchronous upload: no wait on the execution stream.
                    rows = make_device_tensor(list(args.redraw), torch.int64, token_ids.device)
                    seeds = args.seeds and tuple(args.seeds[row] for row in args.redraw)
                    positions = args.positions and tuple(args.positions[row] for row in args.redraw)
                    redrawn = self._law.draw(
                        drawn[rows], BatchSamplingArgs(None, seeds=seeds, positions=positions)
                    )
                    token_ids[rows] = redrawn.to(token_ids.dtype)
            rows = args.logprobs
            if rows is None:
                return DeviceSampled(token_ids)
            sampling = None
            if drawn is not None:
                assert args.temperatures is not None
                every = torch.ones_like(args.temperatures)
                sampling = (
                    args.temperatures,
                    every * self.vocab_size if args.top_k is None else args.top_k,
                    every if args.top_p is None else args.top_p,
                )
            return DeviceSampled(
                token_ids,
                *self.logprobs.read(
                    logits,
                    token_ids,
                    rows,
                    processed_logits=processed_logits,
                    drawn=drawn,
                    sampling=sampling,
                ),
            )

    def _stochastic_paths(self, size: int) -> Iterator[BatchSamplingArgs]:
        """Arguments for every stochastic path a batch of ``size`` rows can take: each row
        seeding this sampler serves, through each filter path its law's draw takes."""

        temperatures = torch.ones(size, device=self.device)
        filters = self._law.draw_filters(
            torch.full((size,), 2, dtype=torch.int32, device=self.device),
            torch.full((size,), 0.5, device=self.device),
        )
        seeded = (tuple(range(size)), (0,) * size)
        seedings = (seeded,) if self._law.SEEDS_EVERY_ROW else (None, seeded)
        for seeding in seedings:
            seeds, positions = seeding or (None, None)
            for top_k, top_p in filters:
                yield BatchSamplingArgs(
                    temperatures, top_k, top_p, seeds=seeds, positions=positions
                )
            # A batch with a min-p row draws its filtered distribution unfiltered.
            yield BatchSamplingArgs(
                temperatures,
                seeds=seeds,
                positions=positions,
                min_p=torch.full((size,), 0.1, device=self.device),
            )
