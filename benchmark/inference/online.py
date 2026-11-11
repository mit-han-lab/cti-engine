from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .._profile import (
    BatchSample,
    BenchmarkCase,
    build_prompt_batches,
    summarize_samples,
    validate_prompt_lengths,
)
from .._runtime import (
    NVML_MEMORY_SAMPLE_INTERVAL_SECONDS,
    REPO_ROOT,
    ApiServerProcess,
    DeviceMemoryMonitor,
    NvmlDevice,
    git_source_state,
    prepare_output_dir,
    run_environment_verifier,
    server_environment,
    sha256_file,
    validate_post_run,
    wait_ready,
    write_json,
)
from ._client import measure_stream_request
from ._server import ServerLaunch

PROFILE_NAME = "cti-online-inference-api-v1"
PROMPT_SEED = 20260701
_DEFAULT_CASES = ("64x64x32x8", "512x64x32x4")


@dataclass(frozen=True, order=True)
class OnlineCase:
    input_tokens: int
    output_tokens: int
    requests: int
    request_rate: float

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.output_tokens, self.requests) < 1:
            raise ValueError("case dimensions must be positive")
        if self.output_tokens < 2:
            raise ValueError("online timing requires at least two output tokens")
        if self.request_rate <= 0:
            raise ValueError("request rate must be positive")
        if self.input_tokens + self.output_tokens > 32768:
            raise ValueError("case exceeds the 32768-token sequence limit")

    @property
    def name(self) -> str:
        rate = str(self.request_rate).replace(".", "p")
        return f"i{self.input_tokens}-o{self.output_tokens}-n{self.requests}-r{rate}"

    @classmethod
    def parse(cls, raw: str) -> OnlineCase:
        try:
            input_tokens, output_tokens, requests, request_rate = raw.split("x")
            return cls(int(input_tokens), int(output_tokens), int(requests), float(request_rate))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid case {raw!r}; expected INPUT_TOKENSxOUTPUT_TOKENSxREQUESTSxRATE"
            ) from error


@dataclass(frozen=True)
class OnlineConfig:
    output_dir: Path
    cases: tuple[OnlineCase, ...]
    runs: int
    warmup_requests: int
    port: int
    seed: int = PROMPT_SEED
    allow_dirty: bool = False

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("at least one online case is required")
        if self.runs < 1 or self.warmup_requests < 1:
            raise ValueError("runs and warmup requests must be positive")
        if len(set(self.cases)) != len(self.cases):
            raise ValueError("online cases must be unique")
        if not 1 <= self.port <= 65534:
            raise ValueError("port must leave room for the server's port + 1 endpoint")
        if self.output_dir.resolve().is_relative_to(REPO_ROOT):
            raise ValueError("benchmark artifacts must be written outside the repository")


def poisson_arrivals(requests: int, request_rate: float, *, seed: int) -> tuple[float, ...]:
    """Return deterministic Poisson arrival offsets in seconds."""
    if requests < 1:
        raise ValueError("requests must be positive")
    if request_rate <= 0:
        raise ValueError("request rate must be positive")
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(1, requests):
        offsets.append(offsets[-1] + rng.expovariate(request_rate))
    return tuple(offsets)


async def _run_arrivals(
    client,
    model: str,
    prompts: Sequence[str],
    output_tokens: int,
    arrivals: Sequence[float],
) -> BatchSample:
    if len(prompts) != len(arrivals):
        raise ValueError("each online request needs one arrival offset")
    origin = time.perf_counter()

    async def submit(prompt: str, offset: float):
        await asyncio.sleep(max(origin + offset - time.perf_counter(), 0.0))
        return await measure_stream_request(client, model, prompt, output_tokens)

    tasks = [
        asyncio.create_task(submit(prompt, offset), name=f"online-request-{index}")
        for index, (prompt, offset) in enumerate(zip(prompts, arrivals, strict=True))
    ]
    try:
        requests = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return BatchSample(
        wall_ms=(time.perf_counter() - origin) * 1000,
        requests=tuple(requests),
    )


async def _measure(
    *,
    port: int,
    model: str,
    cases: Sequence[OnlineCase],
    prompts: dict[OnlineCase, tuple[str, ...]],
    runs: int,
    warmup_requests: int,
    seed: int,
    memory_monitor: DeviceMemoryMonitor,
) -> tuple[dict[str, object], ...]:
    from openai import DefaultAsyncHttpxClient

    client = DefaultAsyncHttpxClient(
        base_url=f"http://127.0.0.1:{port}/v1",
        timeout=300,
        headers={"Authorization": "Bearer cti-benchmark"},
    )
    results = []
    try:
        for case_index, case in enumerate(cases):
            memory_monitor.reset_region()
            samples = []
            schedules = []
            case_prompts = prompts[case]
            warmup_arrivals = poisson_arrivals(
                warmup_requests,
                case.request_rate,
                seed=seed + case_index * (runs + 1),
            )
            await _run_arrivals(
                client,
                model,
                case_prompts[:warmup_requests],
                case.output_tokens,
                warmup_arrivals,
            )
            for run_index in range(runs):
                arrivals = poisson_arrivals(
                    case.requests,
                    case.request_rate,
                    seed=seed + case_index * runs + run_index,
                )
                schedules.append(arrivals)
                prompt_start = warmup_requests + run_index * case.requests
                samples.append(
                    await _run_arrivals(
                        client,
                        model,
                        case_prompts[prompt_start : prompt_start + case.requests],
                        case.output_tokens,
                        arrivals,
                    )
                )
            summary = summarize_samples(tuple(samples))
            results.append(
                {
                    "case": asdict(case),
                    "name": case.name,
                    "arrival_offsets_seconds": schedules,
                    "summary": summary,
                    "peak_device_memory_bytes": memory_monitor.region_peak_bytes(),
                }
            )
            print(
                json.dumps(
                    {
                        "case": case.name,
                        "requests_per_second": summary["requests_per_second"]["p50"],
                        "ttft_ms_p95": summary["ttft_ms"]["p95"],
                        "tpot_ms_p95": summary["tpot_ms"]["p95"],
                        "e2e_ms_p95": summary["e2e_ms"]["p95"],
                    }
                ),
                flush=True,
            )
    finally:
        await client.aclose()
    return tuple(results)


def _prompt_workload(
    tokenizer,
    cases: Sequence[OnlineCase],
    seed: int,
    warmup_requests: int,
    runs: int,
):
    prompts = {}
    lengths = {}
    digests = {}
    for index, case in enumerate(cases):
        prompt_case = BenchmarkCase(case.input_tokens, 1, case.output_tokens)
        batches, digest = build_prompt_batches(
            tokenizer,
            (prompt_case,),
            case.requests * runs + warmup_requests,
            seed + index,
        )
        prompts[case] = tuple(batch[0] for batch in batches[prompt_case])
        lengths[case.name] = validate_prompt_lengths(tokenizer, batches)[prompt_case.name]
        digests[case.name] = digest
    return prompts, lengths, digests


def run_profile(config: OnlineConfig) -> Path:
    prepare_output_dir(config.output_dir)
    environment = run_environment_verifier()
    environment_path = config.output_dir / "environment.json"
    write_json(environment_path, environment)
    environment_receipt = environment["receipt"]
    if environment_receipt["source_dirty"] and not config.allow_dirty:
        raise RuntimeError("formal performance baselines require a clean working tree")

    launch = ServerLaunch()
    model_path = launch.resolve_model()
    from cti.tokenizer.text import load_tokenizer

    tokenizer = load_tokenizer(str(model_path))
    prompts, prompt_lengths, prompt_digests = _prompt_workload(
        tokenizer,
        config.cases,
        config.seed,
        config.warmup_requests,
        config.runs,
    )
    command = launch.command(model_path, config.port)
    source_paths = tuple(
        path.resolve()
        for path in (
            Path(__file__),
            Path(__file__).with_name("_client.py"),
            Path(__file__).with_name("_server.py"),
            Path(__file__).parents[1] / "_profile.py",
            Path(__file__).parents[1] / "_runtime.py",
        )
    )
    source_files = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_paths}
    log_path = config.output_dir / "server.log"

    with NvmlDevice(str(environment_receipt["device_uuid"])) as device:
        device.wait_until_idle()
        server = ApiServerProcess(command, server_environment(), log_path)
        try:
            with server:
                served_model = wait_ready(server.process, config.port)
                with DeviceMemoryMonitor(
                    device,
                    interval=NVML_MEMORY_SAMPLE_INTERVAL_SECONDS,
                ) as memory_monitor:
                    results = asyncio.run(
                        _measure(
                            port=config.port,
                            model=served_model,
                            cases=config.cases,
                            prompts=prompts,
                            runs=config.runs,
                            warmup_requests=config.warmup_requests,
                            seed=config.seed,
                            memory_monitor=memory_monitor,
                        )
                    )
        except BaseException as error:
            try:
                validate_post_run(server, log_path, device)
            except RuntimeError as validation_error:
                error.add_note(f"post-run validation also failed: {validation_error}")
            raise
        validate_post_run(server, log_path, device)

    final_revision, final_dirty = git_source_state()
    final_sources = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_paths}
    if final_revision != environment_receipt["source_revision"]:
        raise RuntimeError("repository HEAD changed while the benchmark was running")
    if final_dirty != environment_receipt["source_dirty"]:
        raise RuntimeError("repository dirty state changed while the benchmark was running")
    if final_sources != source_files:
        raise RuntimeError("benchmark source changed while the benchmark was running")

    receipt_path = config.output_dir / "receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "benchmark": {
                "name": PROFILE_NAME,
                "driver": str(Path(__file__).resolve()),
                "invocation": [sys.executable, *sys.argv],
                "source_files": source_files,
            },
            "source": {
                "revision": environment_receipt["source_revision"],
                "dirty": environment_receipt["source_dirty"],
            },
            "environment": {**environment, "artifact": str(environment_path)},
            "model": {
                "repo_id": launch.model,
                "revision": launch.revision,
                "snapshot": str(model_path),
            },
            "profile": {
                "cases": [asdict(case) for case in config.cases],
                "runs": config.runs,
                "warmup_requests": config.warmup_requests,
                "seed": config.seed,
                "prompt_lengths": prompt_lengths,
                "prompt_digests": prompt_digests,
                "arrival_process": "poisson",
                "inference_memory_ratio": launch.INFERENCE_MEMORY_RATIO,
                "peak_memory_ratio": launch.PEAK_MEMORY_RATIO,
                "page_size": launch.PAGE_SIZE,
            },
            "server": {
                "command": list(command),
                "returncode": server.process.returncode,
                "clean_shutdown": server.clean_shutdown,
                "log": str(log_path),
                "log_sha256": sha256_file(log_path),
            },
            "results": results,
        },
    )
    print(f"receipt: {receipt_path}", flush=True)
    return receipt_path


def _parse_args(argv: Sequence[str] | None = None) -> OnlineConfig:
    parser = argparse.ArgumentParser(description="Run the cti online inference profile.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", dest="cases", action="append", type=OnlineCase.parse)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--port", type=int, default=19400)
    parser.add_argument("--seed", type=int, default=PROMPT_SEED)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    cases = tuple(args.cases or (OnlineCase.parse(raw) for raw in _DEFAULT_CASES))
    return OnlineConfig(
        args.output_dir,
        cases,
        args.runs,
        args.warmup_requests,
        args.port,
        args.seed,
        args.allow_dirty,
    )


def main() -> None:
    run_profile(_parse_args())


if __name__ == "__main__":
    main()
