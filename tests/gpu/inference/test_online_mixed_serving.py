from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    ApiServerProcess,
    find_forbidden_log_patterns,
    isolated_runtime_env,
)
from tests.support.paths import REPO_ROOT, source_pythonpath

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

MIXED_ARRIVAL_ROUNDS = 4
DISCONNECT_RECOVERY_ROUNDS = 8


@dataclass(frozen=True, slots=True)
class _StreamResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


def _assert_same_usage(actual: _StreamResult, expected: _StreamResult) -> None:
    # V0 does not promise batch-invariant token selection. Different batching can
    # cross a greedy argmax boundary without changing request routing or usage.
    assert actual.prompt_tokens == expected.prompt_tokens
    assert actual.completion_tokens == expected.completion_tokens


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _model_snapshot() -> Path:
    return QWEN3_0_6B.snapshot()


def _read_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def _wait_ready(process, port: int, log_path: Path) -> str:
    deadline = time.monotonic() + 180
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(log_path.read_text(encoding="utf-8"))
        try:
            return str(_read_json(url)["data"][0]["id"])
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.25)
    pytest.fail(f"API server did not become ready\n{log_path.read_text(encoding='utf-8')}")


async def _stream_completion(
    client,
    *,
    model: str,
    content: str,
    max_tokens: int,
    priority: str = "normal",
    first_token: asyncio.Event | None = None,
) -> _StreamResult:
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": True, "top_k": 1, "priority": priority},
    )
    text: list[str] = []
    observed_tokens = 0
    usage = None
    try:
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            for choice in chunk.choices:
                if choice.finish_reason is None:
                    observed_tokens += 1
                    if choice.delta.content:
                        text.append(choice.delta.content)
                    if first_token is not None:
                        first_token.set()
    finally:
        await stream.close()

    if usage is None:
        raise RuntimeError("stream completed without terminal usage")
    if observed_tokens != max_tokens or usage.completion_tokens != max_tokens:
        raise RuntimeError(
            f"expected {max_tokens} completion tokens, observed "
            f"chunks={observed_tokens}, usage={usage.completion_tokens}"
        )
    return _StreamResult(
        text="".join(text),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )


async def _cancel_after_first_token(client, *, model: str, round_index: int) -> None:
    first_token = asyncio.Event()
    request = asyncio.create_task(
        _stream_completion(
            client,
            model=model,
            content=f"Continue generating until cancellation round {round_index}.",
            max_tokens=512,
            first_token=first_token,
        ),
        name=f"online-cancelled-request-{round_index}",
    )
    await asyncio.wait_for(first_token.wait(), timeout=30)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request


async def _exercise_online_serving(port: int, model: str) -> dict[str, object]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key="cti-test",
        base_url=f"http://127.0.0.1:{port}/v1",
        max_retries=0,
        timeout=300,
    )
    short_prompt = "State the first prime number."
    long_prompt = "Repeat the reasoning silently and answer OK. " * 500
    try:
        references = await asyncio.gather(
            _stream_completion(
                client,
                model=model,
                content=short_prompt,
                max_tokens=8,
            ),
            _stream_completion(
                client,
                model=model,
                content=long_prompt,
                max_tokens=8,
            ),
        )

        for round_index in range(MIXED_ARRIVAL_ROUNDS):
            resident_started = asyncio.Event()
            resident = asyncio.create_task(
                _stream_completion(
                    client,
                    model=model,
                    content=f"Count upward without stopping, round {round_index}.",
                    max_tokens=128,
                    first_token=resident_started,
                ),
                name=f"online-resident-decode-{round_index}",
            )
            await asyncio.wait_for(resident_started.wait(), timeout=30)
            mixed_results = await asyncio.gather(
                _stream_completion(
                    client,
                    model=model,
                    content=short_prompt,
                    max_tokens=8,
                    priority="high",
                ),
                _stream_completion(
                    client,
                    model=model,
                    content=long_prompt,
                    max_tokens=8,
                    priority="low",
                ),
            )
            await resident
            for actual, expected in zip(mixed_results, references, strict=True):
                _assert_same_usage(actual, expected)

        burst = await asyncio.gather(
            *(
                _stream_completion(
                    client,
                    model=model,
                    content=f"Return a short response for request {index}.",
                    max_tokens=4,
                )
                for index in range(16)
            )
        )
        assert all(result.completion_tokens == 4 for result in burst)

        for round_index in range(DISCONNECT_RECOVERY_ROUNDS):
            await _cancel_after_first_token(
                client,
                model=model,
                round_index=round_index,
            )
            recovery = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"Confirm recovery {round_index}."}],
                max_tokens=4,
                temperature=0.0,
                stream=False,
                extra_body={"ignore_eos": True, "top_k": 1},
            )
            assert recovery.usage is not None
            assert recovery.usage.completion_tokens == 4
            assert len(recovery.choices) == 1
        return {
            "usage_parity": True,
            "mixed_arrival_rounds": MIXED_ARRIVAL_ROUNDS,
            "burst_requests": len(burst),
            "disconnect_recovery_rounds": DISCONNECT_RECOVERY_ROUNDS,
            "recovery_tokens": recovery.usage.completion_tokens,
        }
    finally:
        await client.close()


def test_qwen_online_mixed_serving_and_disconnect_recovery(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "online-mixed")
    log_path = log_root / "online-mixed.log"
    env["PYTHONPATH"] = source_pythonpath()
    port = _free_port()
    command = (
        sys.executable,
        str(REPO_ROOT / "tests/support/serve.py"),
        "{attention_backend: fi, cuda_graph_max_bs: 16}",
        "--model.path",
        str(_model_snapshot()),
        "--model.max-seq-len",
        "8192",
        "--cache.prefix",
        "naive",
        "--scheduling.max-running-requests",
        "16",
        "--scheduling.max-batch-tokens",
        "512",
        "--memory.inference-ratio",
        "0.8",
        "--memory.peak-ratio",
        "0.8",
        "--server.host",
        "127.0.0.1",
        "--server.port",
        str(port),
    )

    with ApiServerProcess(command, env, log_path) as server:
        model = _wait_ready(server.process, port, log_path)
        receipt = asyncio.run(_exercise_online_serving(port, model))

    log = log_path.read_text(encoding="utf-8")
    assert receipt == {
        "usage_parity": True,
        "mixed_arrival_rounds": MIXED_ARRIVAL_ROUNDS,
        "burst_requests": 16,
        "disconnect_recovery_rounds": DISCONNECT_RECOVERY_ROUNDS,
        "recovery_tokens": 4,
    }
    assert server.clean_shutdown, log
    assert server.process.returncode == 0, log
    assert log.count("Aborting request for user") >= DISCONNECT_RECOVERY_ROUNDS
    assert "Inference engine stopped" in log
    assert "Application shutdown complete" in log
    assert not find_forbidden_log_patterns(log)
