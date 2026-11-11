"""The API server's image fetches against a local HTTP server: schemes, limits, retries."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cti.config import ServerOptions
from cti.protocol import ChatPrompt
from cti.server.image_fetch import ImageFetcher, ImageFetchError
from cti.server.openai.requests import OpenAIError

_IMAGE = b"\x89PNG not decoded here"
_MAX_BYTES = 64


class _Upstream(BaseHTTPRequestHandler):
    """Answers by path: ``/<status>`` with that status, ``flaky`` with 503 twice before it
    answers, ``elsewhere`` with a redirect to another host."""

    hits: Counter[str]
    # A connection that sends nothing frees its thread instead of blocking it for good.
    timeout = 10

    def do_GET(self) -> None:
        path = self.path
        self.hits[path] += 1
        if path == "/image" or (path == "/flaky" and self.hits[path] > 2):
            self._answer(200, _IMAGE)
        elif path in ("/redirect", "/elsewhere"):
            self.send_response(302)
            port = self.server.server_port
            target = "/image" if path == "/redirect" else f"http://127.0.0.1:{port}/image"
            self.send_header("Location", target)
            self.end_headers()
        elif path == "/flaky":
            self._answer(503, b"")
        elif path == "/declared-large":
            self.send_response(200)
            self.send_header("Content-Length", str(_MAX_BYTES + 1))
            self.end_headers()
        elif path == "/streamed-large":
            self.send_response(200)  # no Content-Length: the body runs until the close
            self.end_headers()
            self.wfile.write(b"x" * (_MAX_BYTES + 1))
        elif path == "/slow":
            time.sleep(1)
            self._answer(200, _IMAGE)
        else:
            self._answer(int(path[1:]), b"")

    def _answer(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fetcher honours proxy variables, as vLLM's does; the upstream here is local."""

    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "*")


@pytest.fixture
def upstream() -> Iterator[tuple[str, Counter[str]]]:
    hits: Counter[str] = Counter()
    handler = type("Handler", (_Upstream,), {"hits": hits})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", hits
    finally:
        server.shutdown()
        server.server_close()


def _chat(*urls: str) -> ChatPrompt:
    parts = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
    return ChatPrompt(({"role": "user", "content": [*parts, {"type": "text", "text": "Hi"}]},))


def _resolve(
    prompt: ChatPrompt, options: ServerOptions | None = None, *, timeout: float = 10.0
) -> ChatPrompt:
    """Resolve ``prompt``'s images with no backoff; each first attempt may take ``timeout``
    seconds, far longer than the local upstream needs unless a test times it out."""

    async def run() -> ChatPrompt:
        fetcher = ImageFetcher(
            options or ServerOptions(), 4, timeout=timeout, backoff=0, max_bytes=_MAX_BYTES
        )
        try:
            return await fetcher.resolve(prompt)
        finally:
            await fetcher.aclose()

    return asyncio.run(run())


def _images(prompt: ChatPrompt) -> list[object]:
    return [part["image_url"]["url"] for part in prompt.messages[0]["content"][:-1]]


def _assert_failed(error: ImageFetchError, status: int, host: str) -> None:
    assert (error.status_code, error.param) == (status, "image_url")
    assert error.code and error.message.endswith(f": {host}")


def test_http_urls_become_their_bytes_after_transient_failures_and_data_urls_pass(upstream) -> None:
    base, hits = upstream
    data_url = "data:image/png;base64,iVBORw0KGgo="

    # /flaky fails twice, one fewer than the three attempts.
    options = ServerOptions(media_fetch_max_retries=3)
    prompt = _resolve(
        _chat(f"{base}/image", data_url, f"{base}/redirect", f"{base}/flaky"), options
    )

    assert _images(prompt) == [_IMAGE, data_url, _IMAGE, _IMAGE]
    assert prompt.messages[0]["content"][-1] == {"type": "text", "text": "Hi"}
    assert hits == {"/image": 2, "/redirect": 1, "/flaky": 3}  # the flaky one retried


def test_more_images_than_a_request_may_carry_are_refused_before_any_fetch(upstream) -> None:
    base, hits = upstream

    with pytest.raises(OpenAIError, match="at most 4 images") as raised:
        _resolve(_chat(*[f"{base}/image"] * 5))

    assert raised.value.status_code == 400
    assert not hits


@pytest.mark.parametrize(
    ("path", "status", "hits"),
    (
        ("/404", 422, 1),  # vLLM retries only 5xx, connection failures and timeouts
        ("/408", 504, 1),
        ("/429", 502, 1),
        ("/declared-large", 422, 1),
        ("/streamed-large", 422, 1),
        ("/503", 502, 2),
        ("/slow", 504, 2),  # timeouts of 0.2 s, then 0.8 s
    ),
)
def test_failed_fetches_answer_by_cause(upstream, path: str, status: int, hits: int) -> None:
    base, seen = upstream

    with pytest.raises(ImageFetchError) as raised:
        # Only /slow is timed out; every other answer arrives long before its deadline.
        timeout = 0.2 if path == "/slow" else 10.0
        _resolve(_chat(f"{base}{path}"), ServerOptions(media_fetch_max_retries=2), timeout=timeout)

    _assert_failed(raised.value, status, "127.0.0.1")
    assert seen[path] == hits


def test_a_closed_port_is_unreachable() -> None:
    with pytest.raises(ImageFetchError) as raised:
        _resolve(_chat("http://127.0.0.1:1/image"), ServerOptions(media_fetch_max_retries=1))

    _assert_failed(raised.value, 502, "127.0.0.1")


@pytest.mark.parametrize(
    ("domain", "host", "path"),
    (
        ("example.com", "127.0.0.1", "/image"),
        ("localhost", "localhost", "/elsewhere"),  # redirects to 127.0.0.1
    ),
)
def test_hosts_outside_the_allowed_domains_are_invalid(
    upstream, domain: str, host: str, path: str
) -> None:
    base, hits = upstream
    url = base.replace("127.0.0.1", host) + path

    with pytest.raises(ImageFetchError) as raised:
        _resolve(_chat(url), ServerOptions(allowed_media_domains=(domain,)))

    _assert_failed(raised.value, 400, host)
    assert hits["/image"] == 0


@pytest.mark.parametrize(
    "url", ("ftp://example.com/cat.png", "iVBORw0KGgo=", "http://[::1/cat.png", "http:///cat.png")
)
def test_malformed_urls_and_other_schemes_are_invalid(url: str) -> None:
    with pytest.raises(ImageFetchError) as raised:
        _resolve(_chat(url))

    assert raised.value.status_code == 400


@pytest.mark.parametrize("allowed", (False, True))
def test_file_urls_read_only_under_the_allowed_path(tmp_path: Path, allowed: bool) -> None:
    root = tmp_path / "media"
    root.mkdir()
    (root / "cat.png").write_bytes(_IMAGE)
    (tmp_path / "secret.png").write_bytes(_IMAGE)
    options = ServerOptions(allowed_local_media_path=str(root) if allowed else "")

    inside = _chat(f"file://{root}/cat.png")
    outside = _chat(f"file://{root}/../secret.png")

    if allowed:
        assert _images(_resolve(inside, options)) == [_IMAGE]
    for prompt in (outside,) if allowed else (inside, outside):
        with pytest.raises(ImageFetchError) as raised:
            _resolve(prompt, options)
        _assert_failed(raised.value, 400, "localhost")


def _await_hit(hits: Counter[str], path: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while hits[path] == 0 and time.monotonic() < deadline:
        time.sleep(0.05)


def test_a_client_that_leaves_cancels_the_retries(upstream) -> None:
    base, hits = upstream

    async def run() -> float:
        fetcher = ImageFetcher(ServerOptions(), 2, timeout=1, backoff=30, max_bytes=_MAX_BYTES)
        task = asyncio.ensure_future(fetcher.resolve(_chat(f"{base}/503")))
        # The first attempt fails at once and the backoff is 30 s; a loaded host may take a
        # while to serve it, but never the backoff.
        await asyncio.to_thread(_await_hit, hits, "/503", 10)
        assert hits["/503"] == 1
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await fetcher.aclose()
        return time.monotonic() - started

    # Bounded: a cancellation that does not reach the fetch fails here, not at a job timeout.
    assert asyncio.run(asyncio.wait_for(run(), 20)) < 1
    assert hits["/503"] == 1
