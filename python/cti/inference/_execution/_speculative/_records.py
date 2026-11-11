# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors

"""A DFlash round's host-facing values: its staged inputs, how it verifies, its record on
the device and on the host, and the result that settles it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import torch

from cti.inference._execution._sampling import DeviceSampled
from cti.inference._step import Round, RoundCompletion, Sampled
from cti.sampling import TokenSampling

if TYPE_CHECKING:
    from cti.inference._step import RequestSpec
    from cti.recurrent import RoundStatus


class ExtensionOutputs(Protocol):
    """An extension's samples on their way to the host, readable once the round's event
    completed."""

    def host(self) -> Sampled: ...


class Verification(Enum):
    """How a round verifies its drafts; a sampled value is K1's host-static filter hint."""

    GREEDY = None
    SAMPLED = "none"
    FILTERED = "auto"


@dataclass(frozen=True, slots=True)
class RoundArgs:
    """A round's per-row inputs as views of one int32 word buffer, filled by one copy.

    Rows hold request indices, temperatures, canonical top-k/top-p, the seed's low and
    high words, prompt lengths, which place each row's RNG response cursor, rotary minus
    sequence positions, output ends (one past the last position the row may emit at),
    whether the row ignores EOS, the snapshot row its recurrent state is captured into
    (-1: none), whether its log-probabilities are processed, its min-p, and its presence,
    frequency and repetition penalties (``Steering`` holds its min_tokens).
    """

    words: torch.Tensor
    request_indices: torch.Tensor
    temperatures: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    seeds: torch.Tensor
    prompt_lengths: torch.Tensor
    rope_offsets: torch.Tensor
    output_ends: torch.Tensor
    ignore_eos: torch.Tensor
    snapshot_rows: torch.Tensor
    processed_logprobs: torch.Tensor
    min_p: torch.Tensor
    presence: torch.Tensor
    frequency: torch.Tensor
    repetition: torch.Tensor

    @classmethod
    def allocate(cls, capacity: int, **options: object) -> RoundArgs:
        words = torch.zeros(19 * capacity, dtype=torch.int32, **options)
        fields = words.split(capacity)
        return cls(
            words,
            fields[0],
            fields[1].view(torch.float32),
            fields[2],
            fields[3].view(torch.float32),
            words[4 * capacity : 6 * capacity].view(torch.uint32).view(capacity, 2),
            words[6 * capacity : 8 * capacity].view(torch.int64),
            words[8 * capacity : 10 * capacity].view(torch.int64),
            words[10 * capacity : 12 * capacity].view(torch.int64),
            fields[12],
            fields[13],
            fields[14],
            *(field.view(torch.float32) for field in fields[15:]),
        )


@dataclass(frozen=True, slots=True)
class RoundInput:
    """A round staged in host words, which must live until the round completes, and its
    rows' laws. A ``full`` round processes its logits (``LogitProcessing``, min-p) and
    returns log-probabilities."""

    args: RoundArgs
    size: int
    verification: Verification
    rows: tuple[int, ...]
    full: bool
    sampling: tuple[TokenSampling, ...]

    @classmethod
    def stage(
        cls,
        op: Round,
        specs: Sequence[RequestSpec],
        *,
        capacity: int,
        vocab_size: int,
        padding: Sequence[int],
        pin_memory: bool = True,
    ) -> RoundInput:
        """Pack a round's rows and pick how it verifies. A row verifies greedily at
        temperature 0 and ignores its filters then; filters become canonical: top-k in
        [1, V] and top-p in (0, 1], where V and 1 are identity. The rows past them are the
        done ``padding`` rows, greedy and unfiltered, that pad the round to its graph."""

        size = len(op.slots)
        args = RoundArgs.allocate(capacity, pin_memory=pin_memory)
        padding = padding[: capacity - size]
        args.request_indices[size : size + len(padding)] = torch.tensor(padding, dtype=torch.int32)
        args.top_k[size:], args.top_p[size:], args.snapshot_rows[size:] = vocab_size, 1.0, -1
        rows = tuple(spec.sampling for spec in specs)
        top_k = [row.top_k_within(vocab_size) for row in rows]
        top_p = [row.top_p for row in rows]
        args.request_indices[:size] = torch.tensor(op.slots, dtype=torch.int32)
        # A greedy row (``TokenSampling.is_greedy``) verifies at temperature 0.
        args.temperatures[:size] = torch.tensor(
            [0.0 if row.is_greedy else row.temperature for row in rows], dtype=torch.float32
        )
        args.top_k[:size] = torch.tensor(top_k, dtype=torch.int32)
        args.top_p[:size] = torch.tensor(top_p, dtype=torch.float32)
        args.seeds[:size] = torch.tensor(
            [(spec.seed & 0xFFFFFFFF, spec.seed >> 32) for spec in specs], dtype=torch.uint32
        )
        args.prompt_lengths[:size] = torch.tensor([spec.prompt_len for spec in specs])
        args.rope_offsets[:size] = torch.tensor(
            [spec.rope_offset for spec in specs], dtype=torch.int64
        )
        args.output_ends[:size] = torch.tensor(
            [spec.output_end for spec in specs], dtype=torch.int64
        )
        args.ignore_eos[:size] = torch.tensor([row.ignore_eos for row in rows], dtype=torch.int32)
        captures = (-1,) * size if op.capture_rows is None else op.capture_rows
        args.snapshot_rows[:size] = torch.tensor(captures, dtype=torch.int32)
        full = any(row.needs_processing for row in rows)
        if full:
            args.processed_logprobs[:size] = torch.tensor(
                [row.reads_processed for row in rows], dtype=torch.int32
            )
            args.min_p[:size] = torch.tensor([row.min_p for row in rows])
            args.presence[:size] = torch.tensor([row.presence_penalty for row in rows])
            args.frequency[:size] = torch.tensor([row.frequency_penalty for row in rows])
            args.repetition[:size] = torch.tensor([row.repetition_penalty for row in rows])
        sampled = [(k, row.top_p) for k, row in zip(top_k, rows, strict=True) if not row.is_greedy]
        if not sampled:
            verification = Verification.GREEDY
        elif all(k == vocab_size and p == 1 for k, p in sampled):
            verification = Verification.SAMPLED
        else:
            verification = Verification.FILTERED
        return cls(args, size, verification, op.slots, full, rows)


@dataclass(frozen=True, slots=True)
class RoundRecord:
    """A round's per-row results in one int32 word buffer, so one copy moves them: the row's
    frontier at the round's start (int64), its ``[Q]`` target tokens, the raw accepted
    draft count (-1 for a row done before the round), the committed length and whether the
    row is done. Rows past the round's size are stale. ``verdict`` spans the tokens and
    accepted counts, which a sampled round draws before it commits them."""

    words: torch.Tensor
    frontiers: torch.Tensor
    tokens: torch.Tensor
    accepted: torch.Tensor
    committed: torch.Tensor
    done: torch.Tensor
    verdict: torch.Tensor

    @classmethod
    def allocate(cls, rows: int, block_size: int, **options: object) -> RoundRecord:
        words = torch.zeros((5 + block_size) * rows, dtype=torch.int32, **options)
        tokens_end = (2 + block_size) * rows
        return cls(
            words,
            words[: 2 * rows].view(torch.int64),
            words[2 * rows : tokens_end].view(rows, block_size),
            *words[tokens_end:].split(rows),
            words[2 * rows : tokens_end + rows],
        )

    def completion(
        self, size: int, sampled: Sampled | None, logprobs: RoundLogprobs | None
    ) -> RoundCompletion:
        """The first ``size`` rows of a host record, and of a full round's ``logprobs``,
        copied out of them, beside the sampled tokens of the extensions the round ran."""

        tokens = self.tokens[:size]
        return RoundCompletion(
            tokens.shape[1],
            tokens.numpy().tobytes(),
            self.accepted[:size].numpy().tobytes(),
            self.committed[:size].numpy().tobytes(),
            self.done[:size].bool().numpy().tobytes(),
            self.frontiers[:size].numpy().tobytes(),
            sampled,
            None if logprobs is None else logprobs.host(tokens),
        )


@dataclass(frozen=True, slots=True)
class RoundLogprobs:
    """A full round's per-row ``[Q]`` token log-probabilities and ``[Q, K]`` top
    alternatives in one int32 word buffer; rows past the round's size are stale."""

    words: torch.Tensor
    logprobs: torch.Tensor
    top_ids: torch.Tensor
    top_logprobs: torch.Tensor

    @classmethod
    def allocate(cls, rows: int, block_size: int, **options: object) -> RoundLogprobs:
        q, k = block_size, TokenSampling.MAX_TOP
        words = torch.zeros((1 + 2 * k) * q * rows, dtype=torch.int32, **options)
        logprobs, top_ids, top_logprobs = words.split((q * rows, q * k * rows, q * k * rows))
        return cls(
            words,
            logprobs.view(torch.float32).view(rows, q),
            top_ids.view(rows, q, k),
            top_logprobs.view(torch.float32).view(rows, q, k),
        )

    def host(self, tokens: torch.Tensor) -> Sampled:
        """The scores of the round's ``[size, Q]`` host ``tokens``, as bytes."""

        size = len(tokens)
        return DeviceSampled(
            tokens, self.logprobs[:size], self.top_ids[:size], self.top_logprobs[:size]
        ).host()


class RoundRecords:
    """Two record slots per runner: a round's results leave the shared round scratch on
    the execution stream, then reach pinned host memory on a copy stream, so the next round
    never waits for the host. A slot is reused only after its round settled."""

    def __init__(self, rows: int, block_size: int, device: torch.device) -> None:
        def records(**options: object) -> list[tuple[RoundRecord, RoundLogprobs]]:
            return [
                (
                    RoundRecord.allocate(rows, block_size, **options),
                    RoundLogprobs.allocate(rows, block_size, **options),
                )
                for _ in range(2)
            ]

        self._device, self._host = records(device=device), records(pin_memory=True)
        self._stream = torch.cuda.Stream(device=device)
        self._next = 0
        # Each unsettled slot, and whether its round's logprobs were copied.
        self._unsettled: dict[int, bool] = {}

    def copy_out(
        self,
        results: RoundRecord,
        logprobs: RoundLogprobs | None,
        status: RoundStatus | None,
    ) -> tuple[int, torch.cuda.Event]:
        """Enqueue the copies of a round that ran on the current stream, with a full round's
        ``logprobs``; return its record slot and the event after which the host record (and
        status) is readable."""

        slot = self._next % 2
        if slot in self._unsettled:
            raise RuntimeError("a DFlash record slot is reused before its round settled")
        self._next += 1
        self._unsettled[slot] = logprobs is not None
        (device, device_logprobs), (host, host_logprobs) = self._device[slot], self._host[slot]
        device.words.copy_(results.words)
        copies = [(device, host)]
        if logprobs is not None:
            device_logprobs.words.copy_(logprobs.words)
            copies.append((device_logprobs, host_logprobs))
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            for source, target in copies:
                target.words.copy_(source.words, non_blocking=True)
            for source, target in () if status is None else status.copies:
                target.copy_(source, non_blocking=True)
            host_ready = torch.cuda.Event()
            host_ready.record()
        return slot, host_ready

    def settle(self, slot: int, size: int, sampled: Sampled | None) -> RoundCompletion:
        """Read a completed slot's record and free the slot."""

        record, logprobs = self._host[slot]
        full = self._unsettled.pop(slot)
        return record.completion(size, sampled, logprobs if full else None)


@dataclass(frozen=True)
class RoundResult:
    """One submitted round, with the next tokens of the prompt extensions its target forward
    ran, if any; it holds its host staging until ``completion_event``. ``output`` settles
    it, so it is called once, in submission order."""

    completion_event: torch.cuda.Event
    _size: int
    _record: int
    _records: RoundRecords = field(repr=False, compare=False)
    _status: RoundStatus | None = field(repr=False, compare=False)
    _sampled: ExtensionOutputs | None = field(repr=False, compare=False)
    _staging: tuple[object, ...] = field(repr=False, compare=False)
    # Under checks, each fault's message and whether the round raised it (host bools).
    _faults: tuple[tuple[str, torch.Tensor], ...] = field(default=(), repr=False, compare=False)

    def output(self) -> RoundCompletion:
        for message, fault in self._faults:
            if bool(fault):
                raise RuntimeError(message)
        if self._status is not None:
            self._status.settle()
        sampled = None if self._sampled is None else self._sampled.host()
        return self._records.settle(self._record, self._size, sampled)
