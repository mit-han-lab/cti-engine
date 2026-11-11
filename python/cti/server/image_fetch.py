"""Fetching the images online chat requests name by http(s) or file:// URL."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

import httpx

from cti.protocol import ChatPrompt, Prompt
from cti.utils import init_logger

from .openai.requests import OpenAIError

if TYPE_CHECKING:
    from cti.config import ServerOptions

logger = init_logger(__name__, "FrontendAPI")


class ImageFetchError(OpenAIError):
    """A failed image fetch; its message names only the URL's host, as the rest may carry a
    credential. ``retried`` marks a transient failure."""

    CODES = MappingProxyType(
        {
            400: "invalid_image_url",  # malformed, or a scheme, host or path not admitted
            422: "image_url_refused",  # an upstream 4xx, or too large
            502: "image_url_unreachable",  # an upstream 5xx or 429, a connection failure
            504: "image_url_timeout",  # a timeout or an upstream 408
        }
    )

    def __init__(self, status: int, what: str, *, retried: bool = False) -> None:
        super().__init__(what, param="image_url", code=self.CODES[status], status_code=status)
        self.retried = retried


class ImageFetcher:
    """Replaces the http(s) and ``file://`` image URLs of a chat with their bytes, which the
    tokenizer workers decode; ``data:`` URLs pass as they are. A request's images are
    fetched at once on the server's event loop, so a client that leaves cancels them.
    Tests shorten ``timeout``, ``backoff`` and ``max_bytes``; servers keep vLLM's."""

    def __init__(
        self,
        options: ServerOptions,
        max_images: int,
        *,
        timeout: float = 5.0,
        backoff: float = 1.0,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self._domains = frozenset(options.allowed_media_domains)
        local = options.allowed_local_media_path
        self._local_root = Path(local).resolve() if local else None
        self._attempts = max(options.media_fetch_max_retries, 1)
        self._max_images = max_images
        self._timeout = timeout
        self._backoff = backoff
        self._max_bytes = max_bytes
        # Each attempt's own deadline bounds it; every hop's host is checked.
        self._client = httpx.AsyncClient(
            follow_redirects=True, timeout=None, event_hooks={"request": [self._admit_host]}
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def resolve(self, prompt: Prompt) -> Prompt:
        """``prompt`` with the bytes of its chat's fetched images; any other prompt as is."""

        if not isinstance(prompt, ChatPrompt):
            return prompt
        messages = [
            turn if isinstance(turn["content"], str) else {**turn, "content": list(turn["content"])}
            for turn in prompt.messages
        ]
        images = [
            (turn["content"], index)
            for turn in messages
            if not isinstance(turn["content"], str)
            for index, part in enumerate(turn["content"])
            if part["type"] == "image_url"
        ]
        # Before any fetch; the encoder checks it again after decoding.
        if len(images) > self._max_images:
            raise OpenAIError(f"a request may carry at most {self._max_images} images")
        tasks = [
            asyncio.ensure_future(self._fetch(url, parts, index))
            for parts, index in images
            if not (url := parts[index]["image_url"]["url"]).startswith("data:")
        ]
        try:
            await asyncio.gather(*tasks)
        finally:  # the first failure, or the client leaving, cancels the rest
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return replace(prompt, messages=tuple(messages))

    async def _fetch(self, url: str, parts: list, index: int) -> None:
        """Replace ``parts[index]``, the image part naming ``url``, with its bytes."""

        try:
            split = urlsplit(url)
            host = split.hostname or "localhost"
            if split.scheme in ("http", "https"):
                data = await self._download(url, host)
            elif split.scheme == "file":
                data = await asyncio.to_thread(self._read_file, split.netloc, split.path)
            else:
                raise ImageFetchError(400, "image URL must be http(s), data or file")
        except ValueError:  # an unclosed IPv6 bracket, a NUL in a path
            raise ImageFetchError(400, "malformed image URL") from None
        except ImageFetchError as error:
            error.message += f": {host}"
            raise
        parts[index] = {"type": "image_url", "image_url": {"url": data}}

    async def _download(self, url: str, host: str) -> bytes:
        """vLLM's media fetch retries (vllm/connections.py:154-197 ``_is_retryable``,
        :278-320): attempt n times out after timeout * 4**n and, failing transiently,
        sleeps backoff * 4**n before the next."""

        for attempt in range(self._attempts):
            try:
                return await asyncio.wait_for(self._get(url), self._timeout * 4**attempt)
            except (TimeoutError, httpx.TimeoutException):
                error = ImageFetchError(504, "image fetch timed out", retried=True)
            except ImageFetchError as failure:
                if not failure.retried:
                    raise
                error = failure
            if attempt + 1 == self._attempts:
                error.message += f" after {self._attempts} attempts"
                raise error
            logger.warning("Image fetch from %s failed (%s); retrying", host, error.message)
            await asyncio.sleep(self._backoff * 4**attempt)
        raise AssertionError("every attempt returns or raises")

    async def _get(self, url: str) -> bytes:
        try:
            async with self._client.stream("GET", url) as response:
                status = response.status_code
                if status >= 400:  # only 5xx is retried; 408 and 429 answer at once
                    mapped = {408: 504, 429: 502}.get(status, 502 if status >= 500 else 422)
                    what = f"image host returned {status}"
                    raise ImageFetchError(mapped, what, retried=status >= 500)
                declared = response.headers.get("Content-Length", "")
                if declared.isdigit() and int(declared) > self._max_bytes:
                    raise self._oversize()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) > self._max_bytes:
                        raise self._oversize()
                return bytes(body)
        except (httpx.UnsupportedProtocol, httpx.InvalidURL, httpx.TooManyRedirects):
            raise ImageFetchError(400, "malformed image URL") from None
        except httpx.TimeoutException:
            raise
        except httpx.HTTPError:
            raise ImageFetchError(502, "image host unreachable", retried=True) from None

    async def _admit_host(self, request: httpx.Request) -> None:
        if self._domains and request.url.host not in self._domains:
            raise ImageFetchError(400, "image domain not allowed")

    def _read_file(self, host: str, path: str) -> bytes:
        if self._local_root is None:
            raise ImageFetchError(400, "image file URLs are not enabled")
        resolved = Path(unquote(path)).resolve()
        if host not in ("", "localhost") or not resolved.is_relative_to(self._local_root):
            raise ImageFetchError(400, "image file not allowed")
        try:
            with resolved.open("rb") as file:
                data = file.read(self._max_bytes + 1)
        except OSError:
            raise ImageFetchError(422, "image file unreadable") from None
        if len(data) > self._max_bytes:
            raise self._oversize()
        return data

    def _oversize(self) -> ImageFetchError:
        return ImageFetchError(422, f"image larger than {self._max_bytes} bytes")
