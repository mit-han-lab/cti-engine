"""A seeded completion with several choices repeats through the OpenAI server, with and without
DFlash, on Qwen3-8B: the server sends one call's requests as one message, so the engine admits
its choices together and a repeated call is batched alike (previously the choices reached
the engine one by one and their batching followed the timing). No prefix reuse: the
first call would otherwise prefill cold and the repeats through a hit."""

from __future__ import annotations

import gc
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from tests.support.checkpoints import QWEN3_8B, QWEN3_8B_DFLASH_B16
from tests.support.dflash import PINNED_CHECKPOINTS
from tests.support.gpu import ApiServerProcess, find_forbidden_log_patterns, isolated_runtime_env
from tests.support.paths import REPO_ROOT, source_pythonpath

pytestmark = pytest.mark.checkpoints(QWEN3_8B, QWEN3_8B_DFLASH_B16)

PAIR = PINNED_CHECKPOINTS[0]
REQUEST = {"prompt": "The three primary colors are", "max_tokens": 24, "temperature": 0.9}
REPEATS = 4


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def whole_card() -> None:
    """Return the blocks this process's CUDA allocator caches from earlier in-process tests:
    the 8B target and its draft need the whole RTX 5090."""

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "dflash"])
def completions(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, whole_card: None
) -> Iterator[str]:
    name = f"seeded-repeat-{'dflash' if request.param else 'plain'}"
    tmp_path = tmp_path_factory.mktemp(name)
    env, log_root = isolated_runtime_env(tmp_path, name)
    env["PYTHONPATH"] = source_pythonpath()
    log_path = log_root / f"{name}.log"
    port = _free_port()
    draft = ("--speculative.model-path", str(PAIR.draft.snapshot())) if request.param else ()
    command = (
        sys.executable,
        str(REPO_ROOT / "tests/support/serve.py"),
        "{}",
        "--model.path",
        str(PAIR.target.snapshot()),
        "--model.max-seq-len",
        "1024",
        *draft,
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


def _choices(url: str, **fields: object) -> list[list[int]]:
    body = {"model": str(PAIR.target.snapshot()), **REQUEST, "return_token_ids": True, **fields}
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        choices = json.load(response)["choices"]
    return [choice["token_ids"] for choice in sorted(choices, key=lambda c: c["index"])]


def test_a_seeded_call_with_three_choices_repeats_from_its_first_run(completions: str) -> None:
    runs = [_choices(completions, n=3, seed=7) for _ in range(REPEATS)]
    assert all(run == runs[0] for run in runs), runs
    # Choice j samples with seed + j: three draws, not one.
    assert len({tuple(choice) for choice in runs[0]}) == 3
