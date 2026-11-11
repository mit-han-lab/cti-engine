from __future__ import annotations

import argparse
import hashlib
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

MetricDistribution: TypeAlias = dict[str, float | int]
MetricSummary: TypeAlias = dict[str, MetricDistribution | None]
_METRIC_NAMES = (
    "requests_per_second",
    "input_tokens_per_second",
    "output_tokens_per_second",
    "total_tokens_per_second",
    "ttft_ms",
    "tpot_ms",
    "e2e_ms",
    "inter_token_ms",
)


class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


@dataclass(frozen=True, order=True)
class BenchmarkCase:
    input_tokens: int
    concurrency: int
    output_tokens: int

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.concurrency, self.output_tokens) < 1:
            raise ValueError("case dimensions must be positive")
        if self.output_tokens < 2:
            raise ValueError("the latency profile requires at least two output tokens")
        if self.concurrency > 32:
            raise ValueError("the frozen profile supports concurrency at most 32")
        if self.input_tokens + self.output_tokens > 32768:
            raise ValueError("case exceeds the frozen 32768-token sequence limit")

    @property
    def name(self) -> str:
        return f"i{self.input_tokens}-c{self.concurrency}-o{self.output_tokens}"

    @classmethod
    def parse(cls, raw: str) -> BenchmarkCase:
        try:
            input_tokens, concurrency, output_tokens = (int(part) for part in raw.split("x"))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid case {raw!r}; expected INPUTTxCONCURRENCYxOUTPUT"
            ) from error
        return cls(input_tokens, concurrency, output_tokens)


@dataclass(frozen=True)
class RequestTiming:
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    inter_token_ms: tuple[float, ...]
    prompt_tokens: int
    completion_tokens: int
    token_events: int

    @classmethod
    def from_observation(
        cls,
        *,
        started: float,
        token_times: Sequence[float],
        prompt_tokens: int,
        completion_tokens: int,
        expected_output_tokens: int,
    ) -> RequestTiming:
        if completion_tokens != expected_output_tokens:
            raise RuntimeError(
                f"expected {expected_output_tokens} completion tokens, got {completion_tokens}"
            )
        if len(token_times) != completion_tokens:
            raise RuntimeError(
                f"stream emitted {len(token_times)} token events for {completion_tokens} tokens"
            )
        if not token_times:
            raise RuntimeError("stream completed without a token event")

        intervals = tuple(
            (right - left) * 1000 for left, right in zip(token_times, token_times[1:], strict=False)
        )
        return cls(
            ttft_ms=(token_times[0] - started) * 1000,
            tpot_ms=statistics.mean(intervals) if intervals else 0.0,
            e2e_ms=(token_times[-1] - started) * 1000,
            inter_token_ms=intervals,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_events=len(token_times),
        )


@dataclass(frozen=True)
class BatchSample:
    wall_ms: float
    requests: tuple[RequestTiming, ...]

    @property
    def requests_per_second(self) -> float:
        return len(self.requests) / (self.wall_ms / 1000)

    @property
    def input_tokens_per_second(self) -> float:
        return sum(item.prompt_tokens for item in self.requests) / (self.wall_ms / 1000)

    @property
    def output_tokens_per_second(self) -> float:
        return sum(item.completion_tokens for item in self.requests) / (self.wall_ms / 1000)

    @property
    def total_tokens_per_second(self) -> float:
        return self.input_tokens_per_second + self.output_tokens_per_second


@dataclass(frozen=True)
class CaseResult:
    case: BenchmarkCase
    samples: tuple[BatchSample, ...]
    summary: MetricSummary
    peak_device_memory_bytes: int | None

    def as_json(self) -> dict[str, object]:
        return {
            "case": asdict(self.case),
            "name": self.case.name,
            "samples": [sample_payload(sample) for sample in self.samples],
            "summary": self.summary,
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
        }


@dataclass(frozen=True)
class ServerEvidence:
    command: tuple[str, ...]
    returncode: int
    clean_shutdown: bool
    log_path: Path
    log_sha256: str

    def as_json(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "clean_shutdown": self.clean_shutdown,
            "log": str(self.log_path),
            "log_sha256": self.log_sha256,
        }


@dataclass(frozen=True)
class RunResult:
    index: int
    case_order: tuple[BenchmarkCase, ...]
    startup_seconds: float
    ready_device_memory_bytes: int | None
    peak_device_memory_bytes: int | None
    device_memory_telemetry: dict[str, int | str | None]
    resolved_kv_cache_pages: int
    resolved_kv_cache_tokens: int
    cases: tuple[CaseResult, ...]
    server: ServerEvidence

    def as_json(self) -> dict[str, object]:
        return {
            "index": self.index,
            "case_order": [case.name for case in self.case_order],
            "startup_seconds": self.startup_seconds,
            "ready_device_memory_bytes": self.ready_device_memory_bytes,
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
            "device_memory_telemetry": self.device_memory_telemetry,
            "resolved_kv_cache": {
                "pages": self.resolved_kv_cache_pages,
                "tokens": self.resolved_kv_cache_tokens,
            },
            "cases": [case.as_json() for case in self.cases],
            "server": self.server.as_json(),
        }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float]) -> MetricDistribution:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.mean(values)
    stddev = statistics.pstdev(values)
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    return {
        "count": len(values),
        "mean": mean,
        "stddev": stddev,
        "cv": stddev / mean if mean else 0.0,
        "min": min(values),
        "p25": q1,
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "iqr": q3 - q1,
    }


def summarize_samples(samples: Sequence[BatchSample]) -> MetricSummary:
    requests = [request for sample in samples for request in sample.requests]
    return {
        "requests_per_second": distribution([sample.requests_per_second for sample in samples]),
        "input_tokens_per_second": distribution(
            [sample.input_tokens_per_second for sample in samples]
        ),
        "output_tokens_per_second": distribution(
            [sample.output_tokens_per_second for sample in samples]
        ),
        "total_tokens_per_second": distribution(
            [sample.total_tokens_per_second for sample in samples]
        ),
        "ttft_ms": distribution([request.ttft_ms for request in requests]),
        "tpot_ms": distribution([request.tpot_ms for request in requests]),
        "e2e_ms": distribution([request.e2e_ms for request in requests]),
        "inter_token_ms": distribution(
            [interval for request in requests for interval in request.inter_token_ms]
        ),
    }


def sample_payload(sample: BatchSample) -> dict[str, object]:
    return {
        "wall_ms": sample.wall_ms,
        "requests_per_second": sample.requests_per_second,
        "input_tokens_per_second": sample.input_tokens_per_second,
        "output_tokens_per_second": sample.output_tokens_per_second,
        "total_tokens_per_second": sample.total_tokens_per_second,
        "requests": [asdict(request) for request in sample.requests],
    }


def summarize_runs(
    runs: Sequence[RunResult], cases: Sequence[BenchmarkCase]
) -> dict[str, MetricSummary]:
    summary = {}
    for case in cases:
        run_cases = [next(item for item in run.cases if item.case == case) for run in runs]
        case_summary = {
            metric: distribution([float(item.summary[metric]["p50"]) for item in run_cases])
            for metric in _METRIC_NAMES
        }
        memory_peaks = [
            float(item.peak_device_memory_bytes)
            for item in run_cases
            if item.peak_device_memory_bytes is not None
        ]
        case_summary["peak_device_memory_bytes"] = (
            distribution(memory_peaks) if memory_peaks else None
        )
        summary[case.name] = case_summary
    return summary


def exact_prompt(tokenizer: Tokenizer, length: int, rng: random.Random) -> str:
    vocab_size = tokenizer.vocab_size // 2
    token_ids = [rng.randrange(vocab_size) for _ in range(length)]
    for _ in range(128):
        prompt = tokenizer.decode(token_ids)
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        if len(encoded) == length:
            return prompt
        if len(encoded) < length:
            token_ids = encoded + [rng.randrange(vocab_size) for _ in range(length - len(encoded))]
        else:
            token_ids = encoded[:length]
    raise RuntimeError(f"could not construct an exact {length}-token prompt")


def rendered_chat_tokens(tokenizer: Tokenizer, content: str) -> int:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def exact_chat_content(tokenizer: Tokenizer, target_length: int, rng: random.Random) -> str:
    template_tokens = rendered_chat_tokens(tokenizer, "")
    content_length = target_length - template_tokens
    if content_length < 1:
        raise ValueError(
            f"target input length {target_length} does not exceed the "
            f"{template_tokens}-token empty chat template"
        )
    for _ in range(16):
        content = exact_prompt(tokenizer, content_length, rng)
        actual_length = rendered_chat_tokens(tokenizer, content)
        if actual_length == target_length:
            return content
        content_length += target_length - actual_length
        if content_length < 1:
            break
    raise RuntimeError(f"could not construct an exact {target_length}-token rendered chat prompt")


def build_prompt_batches(
    tokenizer: Tokenizer,
    cases: Sequence[BenchmarkCase],
    batch_count: int,
    seed: int,
) -> tuple[dict[BenchmarkCase, tuple[tuple[str, ...], ...]], str]:
    digest = hashlib.sha256()
    result: dict[BenchmarkCase, tuple[tuple[str, ...], ...]] = {}
    for case in cases:
        batches = []
        for batch_index in range(batch_count):
            prompts = []
            for request_index in range(case.concurrency):
                identity = f"{seed}:{case.name}:{batch_index}:{request_index}".encode()
                prompt_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8])
                prompt = exact_chat_content(
                    tokenizer,
                    case.input_tokens,
                    random.Random(prompt_seed),
                )
                digest.update(len(identity).to_bytes(4, "big"))
                digest.update(identity)
                encoded_prompt = prompt.encode()
                digest.update(len(encoded_prompt).to_bytes(8, "big"))
                digest.update(encoded_prompt)
                prompts.append(prompt)
            batches.append(tuple(prompts))
        result[case] = tuple(batches)
    return result, digest.hexdigest()


def validate_prompt_lengths(
    tokenizer: Tokenizer,
    prompts: dict[BenchmarkCase, tuple[tuple[str, ...], ...]],
) -> dict[str, dict[str, int]]:
    result = {}
    for case, batches in prompts.items():
        lengths = []
        for batch in batches:
            for prompt in batch:
                prompt_tokens = rendered_chat_tokens(tokenizer, prompt)
                if prompt_tokens != case.input_tokens:
                    raise ValueError(
                        f"{case.name} expected {case.input_tokens} rendered prompt tokens, "
                        f"got {prompt_tokens}"
                    )
                lengths.append(prompt_tokens)
        result[case.name] = {"min": min(lengths), "max": max(lengths)}
    return result
