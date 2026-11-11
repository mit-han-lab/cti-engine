"""A frontend's tokenizer link for server tests: what the frontend sends each tokenizer, and the
replies the tests deliver, which the frontend's own listener routes to their calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from types import SimpleNamespace

from cti.config import EngineOptions, ModelOptions, ServerOptions
from cti.protocol import AbortMsg, BaseTokenizerMsg, ServingEndpoints, TokenizeMsg
from cti.runtime import EngineProcessError, FailurePhase
from cti.server.api_server import ApiServer
from cti.server.frontend import FrontendManager

# What a link does with one call's requests: deliver their replies, or nothing.
Responder = Callable[["TokenizerLink", Sequence[TokenizeMsg]], Awaitable[None] | None]


class _Send:
    """One tokenizer's link from the frontend."""

    def __init__(self, link: TokenizerLink, index: int) -> None:
        self._link, self._index = link, index

    async def put(self, msgs: Sequence[BaseTokenizerMsg]) -> None:
        await self._link.received(self._index, list(msgs))

    def stop(self) -> None:
        pass


class _Recv:
    """The link the frontend's listener reads its replies from."""

    def __init__(self) -> None:
        self.replies: asyncio.Queue[list[object]] = asyncio.Queue()
        self.stopped = False

    async def get(self) -> list[object]:
        return await self.replies.get()

    def stop(self) -> None:
        self.stopped = True


class TokenizerLink:
    """The tokenizer side of a frontend's links: ``respond`` answers each call's requests,
    and ``deliver`` sends replies at any time."""

    def __init__(self, respond: Responder | None = None, *, tokenizers: int = 1) -> None:
        self._respond = respond
        self._recv = _Recv()
        # Each message a tokenizer received, with the tokenizer's index.
        self.messages: list[tuple[int, list[BaseTokenizerMsg]]] = []
        self.manager = FrontendManager(
            send_tokenizers=tuple(_Send(self, index) for index in range(tokenizers)),
            recv_tokenizer=self._recv,
        )

    @property
    def requests(self) -> list[TokenizeMsg]:
        return [m for _, msgs in self.messages for m in msgs if isinstance(m, TokenizeMsg)]

    @property
    def aborted(self) -> list[int]:
        return [m.uid for _, msgs in self.messages for m in msgs if isinstance(m, AbortMsg)]

    @property
    def stopped(self) -> bool:
        return self._recv.stopped

    def deliver(self, *replies: object) -> None:
        """Send ``replies`` to the frontend as one message."""

        self._recv.replies.put_nowait(list(replies))

    async def received(self, index: int, msgs: list[BaseTokenizerMsg]) -> None:
        self.messages.append((index, msgs))
        requests = [m for m in msgs if isinstance(m, TokenizeMsg)]
        if requests and self._respond is not None:
            responding = self._respond(self, requests)
            if responding is not None:
                await responding

    async def settled(self, timeout: float = 5.0) -> None:
        """Wait until the frontend holds no request: each ended, or released and aborted;
        one it still holds after ``timeout`` leaked."""

        deadline = asyncio.get_running_loop().time() + timeout
        while pending := self.manager.pending:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(f"the frontend still holds requests {sorted(pending)}")
            await asyncio.sleep(0.001)


class FakeEngine:
    """An API server's engine client: it owns the engine's first failure and reports it, as
    its reader thread does when a child exits. Its model takes two images and no tools."""

    def __init__(self, *_args: object, on_failure: Callable[[], None], **_kwargs: object) -> None:
        self.endpoints = ServingEndpoints.create()
        self.info = SimpleNamespace(chat_template=None, image_limits=SimpleNamespace(max_images=2))
        self.failure: EngineProcessError | None = None
        self.on_failure = on_failure

    def fail(self, reason: str) -> None:
        if self.failure is None:
            self.failure = EngineProcessError(reason, FailurePhase.RUNNING)
            self.on_failure()

    def check(self) -> None:
        if self.failure is not None:
            raise self.failure

    def close(self) -> None:
        pass

    def close_at_exit(self, _owner: object) -> SimpleNamespace:
        return SimpleNamespace(cancel=lambda: None)


def serving_api(
    link: TokenizerLink,
    server: ServerOptions | None = None,
    *,
    engine: Callable[..., object] = FakeEngine,
) -> ApiServer:
    """An API server of model ``model``, its engine started, whose frontend is ``link``'s;
    it serves once its app's lifespan runs."""

    api = ApiServer(
        EngineOptions(model=ModelOptions(path="model")),
        ServerOptions() if server is None else server,
        engine=engine,
        link=lambda _endpoints: link.manager,
    )
    api.start_engine()
    return api


__all__ = ["FakeEngine", "Responder", "TokenizerLink", "serving_api"]
