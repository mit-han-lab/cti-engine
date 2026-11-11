"""The OpenAI server's sampling features on Qwen3-8B: for logprobs with alternatives, a stop
string, logit bias, penalties and seeded sampling, a streamed reply equals the non-streamed one
(text, token ids, finish reason, logprobs within tolerance) and its text is the decode of its
token ids; the stop string cuts where it first appears, a banned token never appears and a
forced one always does, and penalties change a repeating greedy continuation."""

from __future__ import annotations

import json
import math
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

TARGET = PINNED_CHECKPOINTS[0].target
COUNTING = "Count from one to twenty: 1, 2, 3, 4, 5,"
REPEATING = "Repeat after me: red green blue red green blue red green blue red green blue"
GREEDY = {"temperature": 0.0, "max_tokens": 24}
PENALTIES = {
    "presence": {"presence_penalty": 1.5},
    "frequency": {"frequency_penalty": 1.5},
    "repetition": {"repetition_penalty": 1.5},
}
STOP = ", 9"
LOGPROB_TOLERANCE = 1e-3


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post(url: str, body: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=300)


class _Server:
    def __init__(self, port: int, model: str, tokenizer) -> None:
        self._url = f"http://127.0.0.1:{port}/v1/completions"
        self._model = model
        self.tokenizer = tokenizer

    def complete(self, prompt: str, **fields: Any) -> dict[str, Any]:
        """The streamed and non-streamed replies to one request, each as its text, token
        ids, finish reason and legacy logprobs; the text must be the decode of the ids up to
        a stop string."""

        body = {"model": self._model, "prompt": prompt, "return_token_ids": True, **fields}
        with _post(self._url, {**body, "stream": False}) as response:
            choice = json.load(response)["choices"][0]
        whole = {
            "text": choice["text"],
            "token_ids": choice["token_ids"],
            "finish_reason": choice["finish_reason"],
            "logprobs": choice["logprobs"],
        }
        streamed: dict[str, Any] = {"text": "", "token_ids": [], "finish_reason": None}
        logprobs: dict[str, list] | None = None
        for chunk in self._stream({**body, "stream": True}):
            choice = chunk["choices"][0]
            streamed["text"] += choice["text"]
            streamed["token_ids"] += choice.get("token_ids", [])
            streamed["finish_reason"] = choice["finish_reason"] or streamed["finish_reason"]
            if choice["logprobs"] is not None:
                logprobs = logprobs or {"tokens": [], "token_logprobs": [], "top_logprobs": []}
                for key in logprobs:
                    logprobs[key] += choice["logprobs"][key]
        streamed["logprobs"] = logprobs

        _assert_same_reply(streamed, whole)
        decoded = self.tokenizer.decode(whole["token_ids"], skip_special_tokens=True)
        if fields.get("stop") is None:
            assert decoded == whole["text"]
        else:
            # Tokens past a cut stop string are still reported, as vLLM does.
            assert decoded.startswith(whole["text"] + fields["stop"]), (decoded, whole)
        return whole

    def _stream(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with _post(self._url, body) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line:
                    continue
                assert line.startswith("data: "), line
                if line == "data: [DONE]":
                    return
                yield json.loads(line.removeprefix("data: "))
        pytest.fail("stream ended without [DONE]")


def _assert_same_reply(streamed: dict[str, Any], whole: dict[str, Any]) -> None:
    for key in ("text", "token_ids", "finish_reason"):
        assert streamed[key] == whole[key], (key, streamed, whole)
    if whole["logprobs"] is None:
        assert streamed["logprobs"] is None
        return
    stream_lp, whole_lp = streamed["logprobs"], whole["logprobs"]
    assert stream_lp["tokens"] == whole_lp["tokens"]
    assert len(whole_lp["tokens"]) == len(whole["token_ids"])
    for a, b in zip(stream_lp["token_logprobs"], whole_lp["token_logprobs"], strict=True):
        assert math.isclose(a, b, abs_tol=LOGPROB_TOLERANCE), (a, b)
    for a, b in zip(stream_lp["top_logprobs"], whole_lp["top_logprobs"], strict=True):
        assert a.keys() == b.keys()
        assert all(math.isclose(a[t], b[t], abs_tol=LOGPROB_TOLERANCE) for t in a), (a, b)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Server]:
    from cti.tokenizer.text import load_tokenizer

    tmp_path = tmp_path_factory.mktemp("online-sampling")
    model_path = str(TARGET.snapshot())
    env, log_root = isolated_runtime_env(tmp_path, "online-sampling")
    env["PYTHONPATH"] = source_pythonpath()
    log_path = log_root / "online-sampling.log"
    port = _free_port()
    command = (
        sys.executable,
        str(REPO_ROOT / "tests/support/serve.py"),
        "{cuda_graph_bs: [1, 2, 4]}",
        "--model.path",
        model_path,
        "--model.max-seq-len",
        "1024",
        # No prefix reuse: the second of a stream / non-stream pair prefills as the first did,
        # so their logprobs agree to rounding, not to the BF16 steps a shorter prefill can move.
        "--cache.prefix",
        "naive",
        "--scheduling.max-running-requests",
        "4",
        "--memory.inference-ratio",
        "0.8",
        "--server.host",
        "127.0.0.1",
        "--server.port",
        str(port),
    )
    with ApiServerProcess(command, env, log_path) as process:
        deadline = time.monotonic() + 600
        while True:
            if process.process.poll() is not None:
                pytest.fail(log_path.read_text(encoding="utf-8"))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models") as response:
                    model = json.load(response)["data"][0]["id"]
                break
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                if time.monotonic() > deadline:
                    pytest.fail(f"server not ready\n{log_path.read_text(encoding='utf-8')}")
                time.sleep(0.5)
        yield _Server(port, model, load_tokenizer(model_path))
    log = log_path.read_text(encoding="utf-8")
    assert process.clean_shutdown, log
    assert not find_forbidden_log_patterns(log)


def test_logprobs_with_alternatives(server: _Server) -> None:
    reply = server.complete(COUNTING, logprobs=5, **GREEDY)
    logprobs = reply["logprobs"]
    for token, logprob, top in zip(
        logprobs["tokens"], logprobs["token_logprobs"], logprobs["top_logprobs"], strict=True
    ):
        assert len(top) == 5 and token in top
        # Greedy: the drawn token is the likeliest one.
        assert logprob <= 0.0 and math.isclose(logprob, max(top.values()), abs_tol=1e-6)


def test_stop_string_cuts_at_its_first_appearance(server: _Server) -> None:
    free = server.complete(COUNTING, **GREEDY)
    stopped = server.complete(COUNTING, stop=STOP, **GREEDY)
    assert stopped["finish_reason"] == "stop"
    assert STOP not in stopped["text"]
    assert free["text"].find(STOP) == len(stopped["text"]) > 0


def test_logit_bias_bans_and_forces_tokens(server: _Server) -> None:
    free = server.complete(COUNTING, **GREEDY)
    banned = free["token_ids"][0]
    without = server.complete(COUNTING, logit_bias={str(banned): -100}, **GREEDY)
    assert banned not in without["token_ids"]
    forced = server.tokenizer.encode(" banana", add_special_tokens=False)[0]
    pinned = server.complete(COUNTING, logit_bias={str(forced): 100}, **GREEDY)
    assert pinned["token_ids"] == [forced] * GREEDY["max_tokens"]


@pytest.mark.parametrize("penalty", PENALTIES, ids=list(PENALTIES))
def test_penalties_change_a_repeating_continuation(server: _Server, penalty: str) -> None:
    free = server.complete(REPEATING, ignore_eos=True, **GREEDY)
    penalized = server.complete(REPEATING, ignore_eos=True, **PENALTIES[penalty], **GREEDY)
    assert penalized["token_ids"] != free["token_ids"]


def test_seeded_sampling_repeats_across_stream_and_non_stream(server: _Server) -> None:
    server.complete(
        REPEATING, temperature=0.9, top_p=0.95, seed=20261018, max_tokens=24, logprobs=3
    )
