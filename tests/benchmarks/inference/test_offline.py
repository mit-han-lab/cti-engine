from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmark._profile import (
    BatchSample,
    BenchmarkCase,
    RequestTiming,
    build_prompt_batches,
    distribution,
    exact_chat_content,
    exact_prompt,
    rendered_chat_tokens,
    summarize_samples,
    validate_prompt_lengths,
)
from benchmark._runtime import (
    NVML_MEMORY_SAMPLE_INTERVAL_SECONDS,
    REPO_ROOT,
    DeviceMemoryMonitor,
    DeviceMemorySampler,
    KVCacheCapacity,
    NvmlDevice,
    current_cuda_uuid,
    prepare_output_file,
    validate_post_run,
    validate_server_log,
)
from benchmark.inference._client import measure_stream_request
from benchmark.inference._server import ServerLaunch
from benchmark.inference._server import main as server_main
from benchmark.inference.offline import (
    PROFILE_NAME,
    ProfileConfig,
    _parse_args,
    _profile_receipt,
)


class ReversibleTokenizer:
    vocab_size = 512

    @staticmethod
    def decode(token_ids) -> str:
        return "".join(chr(token_id + 256) for token_id in token_ids)

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = True) -> list[int]:
        return [ord(character) - 256 for character in text]

    @staticmethod
    def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert not tokenize
        assert add_generation_prompt
        return "12345678" + messages[0]["content"]


class CharacterChatTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert not tokenize
        assert add_generation_prompt
        return "12345678" + messages[0]["content"]

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens, "rendered chat already owns its special tokens"
        return list(range(len(text)))


class FakeStream:
    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self._lines = iter([f"data: {json.dumps(chunk)}" for chunk in chunks] + ["data: [DONE]"])
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    def aiter_lines(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._lines)
        except StopIteration as error:
            raise StopAsyncIteration from error

    def raise_for_status(self) -> None:
        pass


class FakeClient:
    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self._chunks = chunks
        self.kwargs = None
        self.response = None

    def stream(self, method: str, path: str, **kwargs):
        self.kwargs = {"method": method, "path": path, **kwargs}
        self.response = FakeStream(self._chunks)
        return self.response


def _fake_client(chunks: list[dict[str, object]]) -> FakeClient:
    return FakeClient(chunks)


def _token_chunk(*, finished: bool = False) -> dict[str, object]:
    return {
        "usage": None,
        "choices": [{"finish_reason": "stop" if finished else None, "delta": {}}],
    }


def _usage_chunk(*, prompt_tokens: int = 7, completion_tokens: int = 2) -> dict[str, object]:
    return {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "choices": [],
    }


def _timing(*, prompt_tokens: int = 7, completion_tokens: int = 3) -> RequestTiming:
    return RequestTiming.from_observation(
        started=10,
        token_times=(10.01, 10.03, 10.05),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        expected_output_tokens=3,
    )


def test_case_parser_names_dimensions_and_rejects_invalid_shapes() -> None:
    case = BenchmarkCase.parse("512x8x64")
    assert case == BenchmarkCase(input_tokens=512, concurrency=8, output_tokens=64)
    assert case.name == "i512-c8-o64"

    with pytest.raises(ValueError, match="positive"):
        BenchmarkCase(0, 1, 1)
    with pytest.raises(ValueError, match="at least two output tokens"):
        BenchmarkCase(1, 1, 1)
    with pytest.raises(ValueError, match="concurrency"):
        BenchmarkCase(1, 33, 2)
    with pytest.raises(ValueError, match="sequence limit"):
        BenchmarkCase(32768, 1, 2)
    with pytest.raises(argparse.ArgumentTypeError, match="INPUTTxCONCURRENCYxOUTPUT"):
        BenchmarkCase.parse("bad")


def test_profile_config_requires_unique_cases_and_external_artifacts(tmp_path: Path) -> None:
    case = BenchmarkCase(64, 1, 64)
    config = ProfileConfig(tmp_path, (case,), 1, 1, 1, 19400)
    assert config.cases == (case,)

    with pytest.raises(ValueError, match="unique"):
        ProfileConfig(tmp_path, (case, case), 1, 1, 1, 19400)
    with pytest.raises(ValueError, match="outside"):
        ProfileConfig(REPO_ROOT / "artifacts", (case,), 1, 1, 1, 19400)
    with pytest.raises(ValueError, match="tensor-parallel size"):
        ServerLaunch(tensor_parallel_size=0)


def test_receipt_output_is_external_and_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "receipt.json"
    prepare_output_file(output)
    assert output.parent.is_dir()

    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_output_file(output)
    with pytest.raises(ValueError, match="outside"):
        prepare_output_file(REPO_ROOT / "receipt.json")


def test_prompt_generation_is_exact_and_workload_digest_is_stable() -> None:
    tokenizer = ReversibleTokenizer()
    prompt = exact_prompt(tokenizer, 32, random.Random(7))
    assert len(tokenizer.encode(prompt, add_special_tokens=False)) == 32

    cases = (BenchmarkCase(16, 2, 4), BenchmarkCase(24, 1, 4))
    first, first_digest = build_prompt_batches(tokenizer, cases, batch_count=3, seed=11)
    second, second_digest = build_prompt_batches(tokenizer, cases, batch_count=3, seed=11)
    changed, changed_digest = build_prompt_batches(tokenizer, cases, batch_count=3, seed=12)
    assert first == second
    assert first_digest == second_digest
    assert first != changed
    assert first_digest != changed_digest
    assert len({prompt for batches in first.values() for batch in batches for prompt in batch}) == 9
    assert all(
        rendered_chat_tokens(tokenizer, prompt) == case.input_tokens
        for case, batches in first.items()
        for batch in batches
        for prompt in batch
    )

    chat_content = exact_chat_content(tokenizer, 32, random.Random(13))
    assert rendered_chat_tokens(tokenizer, chat_content) == 32


def test_rendered_prompt_lengths_are_validated_before_server_start() -> None:
    tokenizer = CharacterChatTokenizer()
    case = BenchmarkCase(64, 1, 64)
    result = validate_prompt_lengths(tokenizer, {case: (("x" * 56,),)})
    assert result == {case.name: {"min": 64, "max": 64}}

    mismatched = BenchmarkCase(32700, 1, 64)
    with pytest.raises(ValueError, match="expected 32700 rendered prompt tokens, got 32708"):
        validate_prompt_lengths(tokenizer, {mismatched: (("x" * 32700,),)})


def test_request_timing_uses_terminal_usage_as_token_oracle() -> None:
    timing = _timing()
    assert timing.ttft_ms == pytest.approx(10)
    assert timing.tpot_ms == pytest.approx(20)
    assert timing.e2e_ms == pytest.approx(50)
    assert timing.inter_token_ms == pytest.approx((20, 20))
    assert timing.token_events == timing.completion_tokens == 3

    with pytest.raises(RuntimeError, match="expected 3 completion tokens"):
        RequestTiming.from_observation(
            started=0,
            token_times=(0.1, 0.2),
            prompt_tokens=4,
            completion_tokens=2,
            expected_output_tokens=3,
        )
    with pytest.raises(RuntimeError, match="2 token events for 3 tokens"):
        RequestTiming.from_observation(
            started=0,
            token_times=(0.1, 0.2),
            prompt_tokens=4,
            completion_tokens=3,
            expected_output_tokens=3,
        )


def test_cti_stream_contract_counts_empty_nonterminal_chunks_as_tokens() -> None:
    client = _fake_client(
        [_token_chunk(), _token_chunk(), _token_chunk(finished=True), _usage_chunk()]
    )
    timing = asyncio.run(measure_stream_request(client, "model", "prompt", 2))

    assert timing.token_events == timing.completion_tokens == 2
    assert client.response.closed
    assert client.kwargs["method"] == "POST"
    assert client.kwargs["path"] == "/chat/completions"
    assert client.kwargs["json"]["stream_options"] == {"include_usage": True}
    assert client.kwargs["json"]["ignore_eos"] is True
    assert client.kwargs["json"]["top_k"] == 1


@pytest.mark.parametrize(
    ("chunks", "message"),
    (
        (
            [_token_chunk(), _token_chunk(), _token_chunk(finished=True)],
            "without terminal token usage",
        ),
        (
            [
                _token_chunk(),
                _token_chunk(),
                _token_chunk(finished=True),
                _usage_chunk(),
                _usage_chunk(),
            ],
            "usage more than once",
        ),
    ),
)
def test_cti_stream_contract_rejects_missing_or_duplicate_usage(
    chunks: list[dict[str, object]],
    message: str,
) -> None:
    client = _fake_client(chunks)
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(measure_stream_request(client, "model", "prompt", 2))


def test_batch_throughput_and_summaries_use_exact_usage_counts() -> None:
    sample = BatchSample(wall_ms=100, requests=(_timing(), _timing()))
    assert sample.requests_per_second == pytest.approx(20)
    assert sample.input_tokens_per_second == pytest.approx(140)
    assert sample.output_tokens_per_second == pytest.approx(60)
    assert sample.total_tokens_per_second == pytest.approx(200)

    summary = summarize_samples((sample, sample))
    assert summary["output_tokens_per_second"]["p50"] == pytest.approx(60)
    assert summary["ttft_ms"]["count"] == 4
    assert summary["inter_token_ms"]["count"] == 8


def test_distribution_reports_interpolated_percentiles_and_variance() -> None:
    summary = distribution((1.0, 2.0, 3.0, 4.0))
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["p25"] == pytest.approx(1.75)
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["iqr"] == pytest.approx(1.5)
    assert summary["stddev"] > 0


def test_kv_cache_capacity_is_resolved_from_server_evidence() -> None:
    line = "[rank {}] Inference capacity: 15000 KV pages, 33 snapshot rows\n"
    capacity = KVCacheCapacity.from_server_log(line.format(0) + line.format(1), page_size=16)

    assert capacity.pages == 15000
    assert capacity.tokens == 240000


@pytest.mark.parametrize(
    ("output", "message"),
    (
        ("no capacity record", r"found \[\]"),
        (
            "Inference capacity: 8 KV pages, 0 snapshot rows\n"
            "Inference capacity: 9 KV pages, 0 snapshot rows",
            r"found \[8, 9\]",
        ),
    ),
)
def test_kv_cache_capacity_rejects_ambiguous_server_evidence(output: str, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        KVCacheCapacity.from_server_log(output, page_size=16)


def test_server_command_freezes_the_production_profile(tmp_path: Path) -> None:
    from cti.server.args import parse_args

    command = ServerLaunch().command(tmp_path / "model", 19400)
    # The engine's own parser takes every option the server is launched with, as the
    # process receives them: strings.
    parse_args([str(arg) for arg in command[2:]])
    joined = " ".join(command)
    assert command[1].endswith("benchmark/inference/_server.py")
    assert "--cache.prefix radix" in joined
    assert "--scheduling.max-running-requests 32" in joined
    assert "--memory.inference-ratio 0.9" in joined
    assert "--memory.peak-ratio 0.9" in joined
    assert "--parallel.num-ranks 1" in joined
    assert "--parallel.enable-expert-parallel true" in joined
    assert command[-1] == "19400"


def test_the_server_pins_its_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    from cti.config import EngineTuning

    launch = Mock()
    monkeypatch.setattr("cti.server.launch_server", launch)

    server_main()

    launch.assert_called_once_with(
        tuning=EngineTuning(
            attention_backend="fi", cuda_graph_max_bs=32, page_size=ServerLaunch.PAGE_SIZE
        )
    )


def test_cli_defaults_freeze_the_reviewed_offline_profile(tmp_path: Path) -> None:
    config = _parse_args(["--output-dir", str(tmp_path)])
    assert PROFILE_NAME == "cti-offline-inference-api-v1"
    assert config.launch == ServerLaunch()
    assert config.runs == 3
    assert config.warmup_batches == 1
    assert config.measurement_batches == 5
    assert [(case.input_tokens, case.concurrency) for case in config.cases] == [
        (64, 1),
        (64, 8),
        (64, 32),
        (512, 1),
        (512, 8),
        (512, 32),
        (4096, 1),
        (16384, 1),
    ]


def test_receipt_records_the_memory_sampling_interval(tmp_path: Path) -> None:
    profile = _profile_receipt(_parse_args(["--output-dir", str(tmp_path)]), "digest", {})
    assert profile["memory_sampling_interval_ms"] == NVML_MEMORY_SAMPLE_INTERVAL_SECONDS * 1000


def test_cli_selects_the_model_revision_rank_count_and_cache(tmp_path: Path) -> None:
    revision = "c202236" + "0" * 33
    config = _parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--model",
            "Qwen/Qwen3.5-9B",
            "--revision",
            revision,
            "--tensor-parallel-size",
            "2",
            "--cache-type",
            "naive",
            "--no-expert-parallel",
        ]
    )
    assert config.launch == ServerLaunch("Qwen/Qwen3.5-9B", revision, 2, "naive", False)
    receipt = _profile_receipt(config, "digest", {})
    assert (receipt["tensor_parallel_size"], receipt["cache_type"]) == (2, "naive")
    assert receipt["expert_parallel"] is False
    command = config.launch.command(tmp_path, config.port)
    assert command[command.index("--parallel.num-ranks") + 1] == "2"
    assert command[command.index("--cache.prefix") + 1] == "naive"
    assert command[command.index("--parallel.enable-expert-parallel") + 1] == "false"


def test_memory_sampler_keeps_run_and_case_peaks_separate() -> None:
    current = [100]
    sampler = DeviceMemorySampler(lambda: current[0], interval=60)
    with sampler:
        current[0] = 150
        assert sampler.region_peak_bytes() == 150
        current[0] = 100
        sampler.reset_region()
        current[0] = 125
        assert sampler.region_peak_bytes() == 125
        assert sampler.overall_peak_bytes() == 150


def test_memory_sampler_requires_an_explicit_interval() -> None:
    with pytest.raises(TypeError, match="interval"):
        DeviceMemorySampler(lambda: 0)


def test_memory_sampler_selects_samples_by_monotonic_window() -> None:
    current = [100]
    sampler = DeviceMemorySampler(lambda: current[0], interval=60)
    with sampler:
        start_ns = time.monotonic_ns()
        current[0] = 175
        sampler.region_peak_bytes()
        end_ns = time.monotonic_ns()

    assert sampler.samples_between(start_ns, end_ns) == (175,)


def _install_fake_nvml(
    monkeypatch: pytest.MonkeyPatch,
    *,
    actual_uuid: str,
) -> list[object]:
    calls: list[object] = []
    module = ModuleType("pynvml")
    handle = object()
    module.nvmlInit = lambda: calls.append("init")
    module.nvmlShutdown = lambda: calls.append("shutdown")

    def get_handle(uuid: str):
        calls.append(("handle-by-uuid", uuid))
        return handle

    module.nvmlDeviceGetHandleByUUID = get_handle
    module.nvmlDeviceGetUUID = lambda selected: (
        calls.append(("get-uuid", selected)),
        actual_uuid,
    )[1]
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return calls


def test_nvml_device_uses_visible_cuda_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    visible_uuid = "GPU-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    calls = _install_fake_nvml(monkeypatch, actual_uuid=visible_uuid.upper())

    with NvmlDevice(visible_uuid):
        pass

    assert calls[0:2] == ["init", ("handle-by-uuid", visible_uuid)]
    assert calls[2][0] == "get-uuid"
    assert calls[3] == "shutdown"


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("GPU-visible", "GPU-visible"), ("visible", "GPU-visible")),
)
def test_current_cuda_uuid_normalizes_torch_device_properties(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: str,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(uuid=raw),
    )

    assert current_cuda_uuid() == expected


def test_nvml_device_rejects_uuid_mismatch_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_nvml(monkeypatch, actual_uuid="GPU-other")

    with pytest.raises(RuntimeError, match="resolved visible CUDA device"):
        NvmlDevice("GPU-visible").__enter__()

    assert calls[-1] == "shutdown"


def test_nvml_device_declares_optional_telemetry_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    class NVMLError_NotSupported(NVMLError):
        pass

    module.NVMLError = NVMLError
    module.NVML_TEMPERATURE_GPU = 0
    module.NVML_CLOCK_GRAPHICS = 1
    module.NVML_CLOCK_MEM = 2
    module.nvmlInit = lambda: None
    module.nvmlShutdown = lambda: None
    module.nvmlDeviceGetHandleByUUID = lambda _uuid: "handle"
    module.nvmlDeviceGetUUID = lambda _handle: "GPU-visible"
    module.nvmlDeviceGetCudaComputeCapability = lambda _handle: (12, 1)
    module.nvmlDeviceGetName = lambda _handle: "portable GPU"
    module.nvmlSystemGetDriverVersion = lambda: "driver"
    module.nvmlDeviceGetMemoryInfo = lambda _handle: SimpleNamespace(total=1024, used=128)
    module.nvmlDeviceGetPowerManagementLimit = lambda _handle: (_ for _ in ()).throw(
        NVMLError_NotSupported()
    )
    module.nvmlDeviceGetTemperature = lambda _handle, _sensor: 40
    module.nvmlDeviceGetClockInfo = lambda _handle, clock: 1000 + clock
    monkeypatch.setitem(sys.modules, "pynvml", module)

    with NvmlDevice("GPU-visible") as device:
        metadata = device.metadata()
        state = device.state()

    assert metadata["compute_capability"] == [12, 1]
    assert metadata["telemetry"] == {
        "total_memory_bytes": {
            "status": "available",
            "value": 1024,
            "reason": None,
        },
        "power_limit_mw": {
            "status": "unavailable",
            "value": None,
            "reason": "NVMLError_NotSupported",
        },
    }
    assert state["temperature_c"] == {
        "status": "available",
        "value": 40,
        "reason": None,
    }
    assert state["power_usage_mw"] == {
        "status": "unavailable",
        "value": None,
        "reason": "function-unavailable",
    }


def test_nvml_device_declares_memory_telemetry_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    class NVMLError_NotSupported(NVMLError):
        pass

    module.NVMLError = NVMLError
    module.NVML_TEMPERATURE_GPU = 0
    module.NVML_CLOCK_GRAPHICS = 1
    module.NVML_CLOCK_MEM = 2
    module.nvmlInit = lambda: None
    module.nvmlShutdown = lambda: None
    module.nvmlDeviceGetHandleByUUID = lambda _uuid: "handle"
    module.nvmlDeviceGetUUID = lambda _handle: "GPU-visible"
    module.nvmlDeviceGetCudaComputeCapability = lambda _handle: (12, 1)
    module.nvmlDeviceGetName = lambda _handle: "NVIDIA GB10"
    module.nvmlSystemGetDriverVersion = lambda: "580.126.09"
    module.nvmlDeviceGetMemoryInfo = lambda _handle: (_ for _ in ()).throw(NVMLError_NotSupported())
    module.nvmlDeviceGetPowerManagementLimit = lambda _handle: 1000
    module.nvmlDeviceGetTemperature = lambda _handle, _sensor: 40
    module.nvmlDeviceGetClockInfo = lambda _handle, clock: 1000 + clock
    module.nvmlDeviceGetPowerUsage = lambda _handle: 500
    monkeypatch.setitem(sys.modules, "pynvml", module)

    with NvmlDevice("GPU-visible") as device:
        metadata = device.metadata()
        state = device.state()

    unavailable = {
        "status": "unavailable",
        "value": None,
        "reason": "NVMLError_NotSupported",
    }
    assert metadata["total_memory_bytes"] is None
    assert metadata["telemetry"]["total_memory_bytes"] == unavailable
    assert state["used_memory_bytes"] == unavailable


def test_memory_monitor_skips_sampling_when_nvml_memory_is_unavailable() -> None:
    unavailable = {
        "status": "unavailable",
        "value": None,
        "reason": "NVMLError_NotSupported",
    }
    device = SimpleNamespace(used_memory=lambda: unavailable)

    with DeviceMemoryMonitor(device, interval=0.002) as monitor:
        monitor.reset_region()
        assert not monitor.available
        assert monitor.region_peak_bytes() is None
        assert monitor.overall_peak_bytes() is None
        assert monitor.samples_between(1, 2) == ()

    assert monitor.telemetry == unavailable


def test_memory_monitor_retains_finalized_observations_after_exit() -> None:
    current = [100]
    available = {
        "status": "available",
        "value": current[0],
        "reason": None,
    }
    device = SimpleNamespace(
        used_memory=lambda: available,
        used_memory_bytes=lambda: current[0],
    )

    with DeviceMemoryMonitor(device, interval=60) as monitor:
        start_ns = time.monotonic_ns()
        current[0] = 175
        assert monitor.region_peak_bytes() == 175
        end_ns = time.monotonic_ns()

    assert monitor.overall_peak_bytes() == 175
    assert monitor.samples_between(start_ns, end_ns) == (175,)


def test_post_run_validation_checks_gpu_idle_even_when_log_is_invalid(tmp_path: Path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text("incomplete startup", encoding="utf-8")
    server = SimpleNamespace(clean_shutdown=False)
    calls = []
    device = SimpleNamespace(wait_until_idle=lambda: calls.append("idle"))

    with pytest.raises(RuntimeError, match="missing required lifecycle evidence"):
        validate_post_run(server, log_path, device)
    assert calls == ["idle"]


def test_server_log_validation_requires_backend_graph_and_clean_shutdown() -> None:
    valid = "\n".join(
        (
            "Selected attention backend fi",
            "Captured decode graphs",
            "Inference engine stopped",
            "Application shutdown complete",
        )
    )
    validate_server_log(valid)

    with pytest.raises(RuntimeError, match="forbidden"):
        validate_server_log(valid + "\nCUDA ERROR")
    with pytest.raises(RuntimeError, match="missing required"):
        validate_server_log("Selected attention backend fi")
