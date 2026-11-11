from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .._profile import (
    BatchSample,
    BenchmarkCase,
    CaseResult,
    RunResult,
    ServerEvidence,
    build_prompt_batches,
    summarize_runs,
    summarize_samples,
    validate_prompt_lengths,
)
from .._runtime import (
    NVML_MEMORY_SAMPLE_INTERVAL_SECONDS,
    REPO_ROOT,
    ApiServerProcess,
    DeviceMemoryMonitor,
    KVCacheCapacity,
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
from ._server import CacheType, ServerLaunch

PROMPT_SEED = 20260701
PROFILE_NAME = "cti-offline-inference-api-v1"

_DEFAULT_CASES = (
    "64x1x64",
    "64x8x64",
    "64x32x64",
    "512x1x64",
    "512x8x64",
    "512x32x64",
    "4096x1x64",
    "16384x1x64",
)


@dataclass(frozen=True)
class ProfileConfig:
    output_dir: Path
    cases: tuple[BenchmarkCase, ...]
    runs: int
    warmup_batches: int
    measurement_batches: int
    port: int
    launch: ServerLaunch = ServerLaunch()
    seed: int = PROMPT_SEED
    allow_dirty: bool = False

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("at least one benchmark case is required")
        if self.runs < 1 or self.warmup_batches < 1 or self.measurement_batches < 1:
            raise ValueError("runs, warmups, and measurements must be positive")
        if len(set(self.cases)) != len(self.cases):
            raise ValueError("benchmark cases must be unique")
        if not 1 <= self.port <= 65534:
            raise ValueError("port must leave room for the server's port + 1 endpoint")
        if self.output_dir.resolve().is_relative_to(REPO_ROOT):
            raise ValueError("benchmark artifacts must be written outside the repository")


async def _run_batch(client, model: str, prompts: Sequence[str], output_tokens: int) -> BatchSample:
    started = time.perf_counter()
    tasks = [
        asyncio.create_task(
            measure_stream_request(client, model, prompt, output_tokens),
            name=f"inference-profile-request-{index}",
        )
        for index, prompt in enumerate(prompts)
    ]
    try:
        requests = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return BatchSample(
        wall_ms=(time.perf_counter() - started) * 1000,
        requests=tuple(requests),
    )


async def _measure_cases(
    *,
    port: int,
    model: str,
    ordered_cases: Sequence[BenchmarkCase],
    prompts: dict[BenchmarkCase, tuple[tuple[str, ...], ...]],
    warmup_batches: int,
    measurement_batches: int,
    memory_monitor: DeviceMemoryMonitor,
) -> tuple[CaseResult, ...]:
    from openai import DefaultAsyncHttpxClient

    client = DefaultAsyncHttpxClient(
        base_url=f"http://127.0.0.1:{port}/v1",
        timeout=300,
        headers={"Authorization": "Bearer cti-benchmark"},
    )
    results = []
    try:
        for case in ordered_cases:
            memory_monitor.reset_region()
            case_prompts = prompts[case]
            for batch in case_prompts[:warmup_batches]:
                await _run_batch(client, model, batch, case.output_tokens)
            measured_prompts = case_prompts[warmup_batches : warmup_batches + measurement_batches]
            samples = tuple(
                [
                    await _run_batch(client, model, batch, case.output_tokens)
                    for batch in measured_prompts
                ]
            )
            for sample in samples:
                for request in sample.requests:
                    if request.prompt_tokens != case.input_tokens:
                        raise RuntimeError(
                            f"{case.name} usage reported {request.prompt_tokens} prompt tokens"
                        )
            summary = summarize_samples(samples)
            results.append(
                CaseResult(
                    case=case,
                    samples=samples,
                    summary=summary,
                    peak_device_memory_bytes=memory_monitor.region_peak_bytes(),
                )
            )
            print(
                json.dumps(
                    {
                        "case": case.name,
                        "output_tokens_per_second": summary["output_tokens_per_second"]["p50"],
                        "ttft_ms": summary["ttft_ms"]["p50"],
                        "tpot_ms": summary["tpot_ms"]["p50"],
                        "e2e_ms": summary["e2e_ms"]["p50"],
                    }
                ),
                flush=True,
            )
    finally:
        await client.aclose()
    return tuple(results)


def _run_once(
    *,
    config: ProfileConfig,
    index: int,
    command: tuple[str, ...],
    prompts: dict[BenchmarkCase, tuple[tuple[str, ...], ...]],
    device: NvmlDevice,
) -> RunResult:
    log_path = config.output_dir / f"run-{index}-server.log"
    offset = (index - 1) % len(config.cases)
    ordered_cases = config.cases[offset:] + config.cases[:offset]
    server = ApiServerProcess(command, server_environment(), log_path)
    try:
        startup_started = time.perf_counter()
        with server:
            served_model = wait_ready(server.process, config.port)
            startup_seconds = time.perf_counter() - startup_started
            ready_memory = device.used_memory()
            with DeviceMemoryMonitor(
                device,
                interval=NVML_MEMORY_SAMPLE_INTERVAL_SECONDS,
            ) as memory_monitor:
                cases = asyncio.run(
                    _measure_cases(
                        port=config.port,
                        model=served_model,
                        ordered_cases=ordered_cases,
                        prompts=prompts,
                        warmup_batches=config.warmup_batches,
                        measurement_batches=config.measurement_batches,
                        memory_monitor=memory_monitor,
                    )
                )
                peak_memory = memory_monitor.overall_peak_bytes()
    except BaseException as error:
        try:
            validate_post_run(server, log_path, device)
        except RuntimeError as validation_error:
            note = f"post-run validation also failed: {validation_error}; log={log_path}"
            if hasattr(error, "add_note"):
                error.add_note(note)
            print(note, file=sys.stderr)
        raise

    validate_post_run(server, log_path, device)
    kv_cache_capacity = KVCacheCapacity.from_server_log(
        log_path.read_text(encoding="utf-8"), page_size=ServerLaunch.PAGE_SIZE
    )
    return RunResult(
        index=index,
        case_order=ordered_cases,
        startup_seconds=startup_seconds,
        ready_device_memory_bytes=(
            ready_memory["value"] if type(ready_memory["value"]) is int else None
        ),
        peak_device_memory_bytes=peak_memory,
        device_memory_telemetry=ready_memory,
        resolved_kv_cache_pages=kv_cache_capacity.pages,
        resolved_kv_cache_tokens=kv_cache_capacity.tokens,
        cases=cases,
        server=ServerEvidence(
            command=command,
            returncode=server.process.returncode,
            clean_shutdown=server.clean_shutdown,
            log_path=log_path,
            log_sha256=sha256_file(log_path),
        ),
    )


def _profile_receipt(
    config: ProfileConfig,
    prompt_digest: str,
    rendered_prompt_tokens: dict[str, dict[str, int]],
) -> dict[str, object]:
    return {
        "version": 4,
        "cases": [asdict(case) for case in config.cases],
        "runs": config.runs,
        "warmup_batches": config.warmup_batches,
        "measurement_batches": config.measurement_batches,
        "seed": config.seed,
        "prompt_digest": prompt_digest,
        "rendered_prompt_tokens": rendered_prompt_tokens,
        "input_token_definition": "rendered chat prompt consumed by the model",
        "attention_backend": "fi",
        "cache_type": config.launch.cache_type,
        "expert_parallel": config.launch.expert_parallel,
        "dtype": "bfloat16",
        "cuda_graph_max_batch_size": 32,
        "overlap_scheduling": True,
        "max_running_requests": 32,
        "max_sequence_length": 32768,
        "max_batch_tokens": 16384,
        "page_size": ServerLaunch.PAGE_SIZE,
        "inference_memory_ratio": ServerLaunch.INFERENCE_MEMORY_RATIO,
        "peak_memory_ratio": ServerLaunch.PEAK_MEMORY_RATIO,
        "kv_capacity_policy": "inference_memory_ratio",
        "tensor_parallel_size": config.launch.tensor_parallel_size,
        "use_pynccl": True,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "memory_sampling_interval_ms": NVML_MEMORY_SAMPLE_INTERVAL_SECONDS * 1000,
    }


def run_profile(config: ProfileConfig) -> Path:
    prepare_output_dir(config.output_dir)
    environment = run_environment_verifier()
    environment_path = config.output_dir / "environment.json"
    write_json(environment_path, environment)
    environment_receipt = environment["receipt"]
    if environment_receipt["source_dirty"] and not config.allow_dirty:
        raise RuntimeError("formal performance baselines require a clean working tree")

    model_path = config.launch.resolve_model()
    from cti.tokenizer.text import load_tokenizer

    tokenizer = load_tokenizer(str(model_path))
    prompt_batches, prompt_digest = build_prompt_batches(
        tokenizer,
        config.cases,
        config.warmup_batches + config.measurement_batches,
        config.seed,
    )
    rendered_prompt_tokens = validate_prompt_lengths(tokenizer, prompt_batches)
    command = config.launch.command(model_path, config.port)
    benchmark_sources = tuple(
        path.resolve()
        for path in (
            Path(__file__),
            Path(__file__).parents[1] / "_profile.py",
            Path(__file__).parents[1] / "_runtime.py",
        )
    )
    source_files = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in benchmark_sources
    }

    with NvmlDevice(str(environment_receipt["device_uuid"])) as device:
        device.wait_until_idle()
        gpu_before = device.state()
        gpu_metadata = device.metadata()
        runs = tuple(
            _run_once(
                config=config,
                index=index,
                command=command,
                prompts=prompt_batches,
                device=device,
            )
            for index in range(1, config.runs + 1)
        )
        gpu_after = device.state()

    final_revision, final_dirty = git_source_state()
    final_source_files = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in benchmark_sources
    }
    if final_revision != environment_receipt["source_revision"]:
        raise RuntimeError("repository HEAD changed while the benchmark was running")
    if final_dirty != environment_receipt["source_dirty"]:
        raise RuntimeError("repository dirty state changed while the benchmark was running")
    if final_source_files != source_files:
        raise RuntimeError("benchmark source changed while the benchmark was running")

    receipt = {
        "schema_version": 1,
        "benchmark": {
            "name": PROFILE_NAME,
            "driver": str(Path(__file__).resolve()),
            "driver_sha256": sha256_file(Path(__file__)),
            "invocation": [sys.executable, *sys.argv],
            "source_files": source_files,
        },
        "source": {
            "revision": environment_receipt["source_revision"],
            "dirty": environment_receipt["source_dirty"],
        },
        "environment": {
            **environment,
            "artifact": str(environment_path),
            "artifact_sha256": sha256_file(environment_path),
        },
        "model": {
            "repo_id": config.launch.model,
            "revision": config.launch.revision,
            "snapshot": str(model_path),
        },
        "profile": _profile_receipt(config, prompt_digest, rendered_prompt_tokens),
        "gpu": {"metadata": gpu_metadata, "before": gpu_before, "after": gpu_after},
        "runs": [run.as_json() for run in runs],
        "summary": summarize_runs(runs, config.cases),
    }
    receipt_path = config.output_dir / "receipt.json"
    write_json(receipt_path, receipt)
    print(f"receipt: {receipt_path}", flush=True)
    return receipt_path


def _parse_args(argv: Sequence[str] | None = None) -> ProfileConfig:
    parser = argparse.ArgumentParser(
        description="Run the cti offline inference API performance profile."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=BenchmarkCase.parse,
        help="INPUTTxCONCURRENCYxOUTPUT; repeat to override the frozen matrix",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measurement-batches", type=int, default=5)
    parser.add_argument("--port", type=int, default=19400)
    parser.add_argument("--model", default=ServerLaunch.MODEL_ID, help="Hugging Face repository id")
    parser.add_argument(
        "--revision", default=ServerLaunch.MODEL_REVISION, help="full commit hash of --model"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="ranks on the first visible GPUs; memory telemetry reads the first GPU only",
    )
    # Hybrid models reuse prefixes on one rank only.
    parser.add_argument(
        "--cache-type", type=CacheType, choices=tuple(CacheType), default=CacheType.RADIX
    )
    parser.add_argument(
        "--expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="split a MoE model's whole experts over the ranks; --no-expert-parallel splits "
        "every expert by its width",
    )
    parser.add_argument("--seed", type=int, default=PROMPT_SEED)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow a tooling smoke from an uncommitted checkout; never use for a formal baseline",
    )
    args = parser.parse_args(argv)
    cases = tuple(args.cases or (BenchmarkCase.parse(raw) for raw in _DEFAULT_CASES))
    return ProfileConfig(
        output_dir=args.output_dir,
        cases=cases,
        runs=args.runs,
        warmup_batches=args.warmup_batches,
        measurement_batches=args.measurement_batches,
        port=args.port,
        launch=ServerLaunch(
            args.model,
            args.revision,
            args.tensor_parallel_size,
            args.cache_type,
            args.expert_parallel,
        ),
        seed=args.seed,
        allow_dirty=args.allow_dirty,
    )


def main() -> None:
    run_profile(_parse_args())


if __name__ == "__main__":
    main()
