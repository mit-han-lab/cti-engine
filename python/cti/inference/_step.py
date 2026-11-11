from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, Protocol

from cti.sampling import TokenLogprobs, TokenSampling

if TYPE_CHECKING:
    from cti.forward import RoundShape
    from cti.kvcache import StateCapacity


class QueryKind(Enum):
    """Per-row inference work represented by one model query."""

    EXTEND = auto()
    DECODE = auto()


@dataclass(frozen=True, slots=True)
class StateCapture:
    """Export the recurrent state after a sequence's first ``position`` tokens to ``row``."""

    position: int
    row: int


@dataclass(frozen=True, slots=True)
class Row:
    """One forward row, exact when planned: the slot it runs, how much of its sequence is
    cached and how far this step extends it. ``write_position`` is where its sampled token
    goes (None: it samples nothing), ``sampling_position`` its output index for seeded
    sampling; an extension may restore and export recurrent state."""

    request_index: int
    query_kind: QueryKind
    kv_prefix_len: int
    sequence_len: int
    write_position: int | None = None
    sampling_position: int = 0
    restore_row: int | None = None
    captures: tuple[StateCapture, ...] = ()

    @property
    def extend_len(self) -> int:
        return self.sequence_len - self.kv_prefix_len


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """A request's constants, given to the runner once when its slot is admitted: where its
    output ends, the law its tokens are sampled under and its seed (None: the shared
    generator draws it)."""

    slot: int
    prompt_len: int
    output_end: int
    sampling: TokenSampling = TokenSampling()
    seed: int | None = None
    rope_offset: int = 0


@dataclass(frozen=True, slots=True)
class PageMap:
    """The page table changes a step makes before it runs: int32 (slot, page index, page
    ID) triples copied out of the cache manager."""

    entries: bytes


@dataclass(frozen=True, slots=True)
class Forward:
    """One step of prompt extensions and decodes: int32 token ids, int64 ``[axes, tokens]``
    rotary positions and the packed images of the extensions, as bytes."""

    rows: tuple[Row, ...]
    extend_token_ids: bytes | None = None
    extend_rope_positions: bytes | None = None
    images: bytes | None = None
    pages: PageMap | None = None
    # Each extension token's next prompt token, which a row that scores its prompt reads
    # (int32, aligned with ``extend_token_ids``; -1: none).
    prompt_targets: bytes | None = None
    # The cached prompt prefix a prefix hit restored, per slot, which no forward writes to
    # the token table and logits processing reads (int32 tokens).
    restored_prompts: tuple[tuple[int, bytes], ...] = ()

    def require_vocabulary_ids(self) -> None:
        """Image key ids never reach the token table the model and the DFlash draft read."""

        ids = self.extend_token_ids
        if ids is not None and min(memoryview(ids).cast("i")) < 0:
            raise RuntimeError("a forward's input ids hold an image key id")


@dataclass(frozen=True, slots=True)
class Round:
    """One speculative round over live slots, with the prompt extensions ``extend`` beside
    it in the round's target forward; lengths stay on the device."""

    slots: tuple[int, ...]
    capture_rows: tuple[int, ...] | None = None
    extend: Forward | None = None
    pages: PageMap | None = None

    def require_vocabulary_ids(self) -> None:
        if self.extend is not None:
            self.extend.require_vocabulary_ids()


@dataclass(frozen=True, slots=True)
class Sampled:
    """Int32 tokens, one per row, and when asked for their float32 log-probabilities and
    ``top`` alternatives per row (int32 IDs and float32 log-probabilities, best first).
    A forward that scores prompts adds, per row, the prompt position its scored prompt
    tokens start at and their scores (None: it scores none)."""

    tokens: bytes
    logprobs: bytes | None = None
    top: int = 0
    top_ids: bytes | None = None
    top_logprobs: bytes | None = None
    prompts: tuple[tuple[int, Sampled] | None, ...] | None = None

    def token_ids(self) -> memoryview:
        return memoryview(self.tokens).cast("i")

    def at(self, index: int, top: int) -> TokenLogprobs:
        """Row ``index``'s token, its log-probability and its first ``top`` alternatives."""

        assert self.logprobs is not None and top <= self.top
        alternatives: tuple[tuple[int, float], ...] = ()
        if top:
            assert self.top_ids is not None and self.top_logprobs is not None
            start = index * self.top
            ids = memoryview(self.top_ids).cast("i")[start : start + top]
            values = memoryview(self.top_logprobs).cast("f")[start : start + top]
            alternatives = tuple(zip(ids.tolist(), values.tolist(), strict=True))
        return TokenLogprobs(
            self.token_ids()[index], memoryview(self.logprobs).cast("f")[index], alternatives
        )


@dataclass(frozen=True, slots=True)
class RoundCompletion:
    """One speculative round per row: int32 ``[B, width]`` target tokens, int32 accepted and
    committed counts, done flags and the int64 frontier at the round's start; ``sampled``
    holds the extension rows' tokens when the round ran beside them."""

    width: int
    tokens: bytes
    accepted: bytes
    committed: bytes
    done: bytes
    frontiers: bytes
    sampled: Sampled | None = None
    # A full round's ``[B, width]`` target tokens with their log-probabilities and
    # alternatives, row ``b``'s position ``i`` at index ``b * width + i``.
    logprobs: Sampled | None = None


class RequestTable:
    """Each live slot's request constants, as the EngineCore admitted them; the per-step
    rows never carry them."""

    def __init__(self) -> None:
        self._specs: dict[int, RequestSpec] = {}

    def admit(self, specs: Sequence[RequestSpec]) -> None:
        for spec in specs:
            self._specs[spec.slot] = spec

    def __getitem__(self, slot: int) -> RequestSpec:
        return self._specs[slot]


@dataclass(frozen=True, slots=True)
class ServingLimits:
    """What the executor serves, as the control half plans for it: its capacity, its
    rounds (None without), the stop tokens, the vocabulary it samples from, the
    encoder-output rows it holds, the prefix cache it keeps (radix reuses prefixes), the
    most tokens one step takes and the most of them prompts may take."""

    capacity: StateCapacity
    rounds: RoundShape | None
    stop_token_ids: tuple[int, ...]
    vocab_size: int
    image_rows: int
    prefix_cache: Literal["naive", "radix"]
    max_batch_tokens: int
    max_prompt_tokens: int


class Completion(Protocol):
    """A submitted step's device work, such as a CUDA event."""

    def query(self) -> bool:
        """Whether it has completed."""
        ...

    def synchronize(self) -> None:
        """Wait until it completes."""
        ...


class StepResult(Protocol):
    """One submitted step: its output is readable once ``completion_event`` completes."""

    @property
    def completion_event(self) -> Completion: ...

    def output(self) -> Sampled | RoundCompletion: ...


@dataclass(frozen=True, slots=True)
class GraphCapture:
    """One group of CUDA graphs the executor captured before serving: its kind, its sizes,
    the seconds its warm-up and capture took and the device bytes they added."""

    kind: Literal["decode", "breakable", "round", "vision"]
    sizes: tuple[int, ...]
    seconds: float
    nbytes: int

    def __str__(self) -> str:
        return f"{self.kind} {self.nbytes / (1 << 30):.2f} GiB in {self.seconds:.1f} s"


@dataclass(frozen=True, slots=True)
class GraphStats:
    """The executor's graphs as a served metric: what it captured, and how many
    forwards replayed a decode or breakable graph or ran eagerly since it started."""

    captures: tuple[GraphCapture, ...]
    decode: int
    breakable: int
    eager: int

    def __str__(self) -> str:
        captured = ", ".join(map(str, self.captures)) or "none"
        return (
            f"forwards {self.decode} decode graph, {self.breakable} breakable graph, "
            f"{self.eager} eager; graphs captured: {captured}"
        )


class Executor(Protocol):
    """Runs the steps the control half plans, within ``limits``; ``table`` holds the
    request constants of the slots a step names. ``graph_stats`` costs a forward one integer
    add: no device work, no sync."""

    limits: ServingLimits

    def submit(self, op: Forward | Round, table: RequestTable) -> StepResult: ...

    def graph_stats(self) -> GraphStats: ...

    def reset_inference_state(self) -> None:
        """Reset the device state the kernels keep beside the pool, once serving holds the
        device again: before the next step."""
        ...
