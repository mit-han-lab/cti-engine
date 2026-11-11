"""A DFlash stream through the OpenAI server on Qwen3-8B sends one chunk per verify round,
as vLLM and SGLang do, whether or not the request asks for logprobs and token ids: the same
greedy request streams the same chunks either way, a round that accepts several tokens
carries them (and their logprobs) in one chunk, and the stream's text and tokens are the
non-streamed reply's."""

from __future__ import annotations

import json
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from tests.support.checkpoints import QWEN3_8B, QWEN3_8B_DFLASH_B16
from tests.support.dflash import PINNED_CHECKPOINTS
from tests.support.gpu import ApiServerProcess, find_forbidden_log_patterns, isolated_runtime_env
from tests.support.paths import REPO_ROOT, source_pythonpath

pytestmark = pytest.mark.checkpoints(QWEN3_8B, QWEN3_8B_DFLASH_B16)

PAIR = PINNED_CHECKPOINTS[0]
# A predictable continuation, so rounds accept several draft tokens.
REQUEST = {"prompt": "Count from one to forty: 1, 2, 3, 4, 5,", "max_tokens": 64, "temperature": 0}
OPTIONS = {"return_token_ids": True, "logprobs": 1}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post(url: str, body: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=300)


@pytest.fixture(scope="module")
def completions(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    tmp_path = tmp_path_factory.mktemp("dflash-stream-chunks")
    env, log_root = isolated_runtime_env(tmp_path, "dflash-stream-chunks")
    env["PYTHONPATH"] = source_pythonpath()
    log_path = log_root / "dflash-stream-chunks.log"
    port = _free_port()
    command = (
        sys.executable,
        str(REPO_ROOT / "tests/support/serve.py"),
        "{}",
        "--model.path",
        str(PAIR.target.snapshot()),
        "--model.max-seq-len",
        "1024",
        "--speculative.model-path",
        str(PAIR.draft.snapshot()),
        "--cache.prefix",
        "naive",
        "--memory.inference-ratio",
        "0.8",
        "--server.host",
        "127.0.0.1",
        "--server.port",
        str(port),
    )
    with ApiServerProcess(command, env, log_path) as process:
        deadline = time.monotonic() + 900
        while True:
            if process.process.poll() is not None:
                pytest.fail(log_path.read_text(encoding="utf-8"))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models") as response:
                    json.load(response)
                break
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                if time.monotonic() > deadline:
                    pytest.fail(f"server not ready\n{log_path.read_text(encoding='utf-8')}")
                time.sleep(0.5)
        yield f"http://127.0.0.1:{port}/v1/completions"
    log = log_path.read_text(encoding="utf-8")
    assert process.clean_shutdown, log
    assert not find_forbidden_log_patterns(log)


def _chunks(url: str, **fields: Any) -> list[dict[str, Any]]:
    """The stream's choice chunks that carry output, before its end chunk."""

    body = {"model": str(PAIR.target.snapshot()), **REQUEST, **fields, "stream": True}
    chunks = []
    with _post(url, body) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line:
                continue
            if line == "data: [DONE]":
                return chunks
            choice = json.loads(line.removeprefix("data: "))["choices"][0]
            if choice["text"] or choice.get("token_ids"):
                chunks.append(choice)
    pytest.fail("stream ended without [DONE]")


def test_a_dflash_stream_sends_one_chunk_per_round_with_or_without_token_options(
    completions: str,
) -> None:
    plain = _chunks(completions)
    detailed = _chunks(completions, **OPTIONS)
    body = {"model": str(PAIR.target.snapshot()), **REQUEST, **OPTIONS}
    with _post(completions, body) as response:
        whole = json.load(response)["choices"][0]

    # The options change what a chunk carries, not how the stream is cut.
    assert [c["text"] for c in detailed] == [c["text"] for c in plain]
    token_ids = [t for c in detailed for t in c["token_ids"]]
    assert token_ids == whole["token_ids"]
    assert "".join(c["text"] for c in detailed) == whole["text"]
    for chunk in detailed:
        assert len(chunk["logprobs"]["tokens"]) == len(chunk["token_ids"])
    # Rounds that accept draft tokens send them together.
    assert len(detailed) < len(token_ids), [c["token_ids"] for c in detailed]
