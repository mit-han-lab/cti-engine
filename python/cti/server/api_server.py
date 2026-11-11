# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/server/api_server.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from cti.protocol import AdmissionPriority, SamplingParams, SamplingResult
from cti.runtime import EngineClient
from cti.utils import init_logger

from .frontend import (
    FrontendManager,
    GenerationCall,
    InferenceUnavailableError,
    SamplingCall,
    sampling_prompt,
)
from .image_fetch import ImageFetcher
from .openai.requests import ChatCompletionRequest, CompletionRequest, OpenAIError
from .openai.responses import ChatCompletionFormat, CompletionFormat
from .tool_calls import ToolCallFormat

if TYPE_CHECKING:
    from multiprocessing.util import Finalize

    from cti.config import EngineOptions, EngineTuning, ServerOptions
    from cti.protocol import Prompt, SamplingPrompt, SamplingSpec, ServingEndpoints


logger = init_logger(__name__, "FrontendAPI")


async def _openai_error(_request: Request, error: OpenAIError) -> JSONResponse:
    return error.response()


async def _invalid_request(_request: Request, error: RequestValidationError) -> JSONResponse:
    return OpenAIError.from_validation(error.errors()).response()


class ApiServer:
    """The OpenAI-compatible HTTP frontend (FastAPI + uvicorn) of one online server. It owns
    its app, its HTTP server, the engine it starts as it runs and, while it serves, the
    frontend linked to the engine's tokenizer workers over ZMQ."""

    def __init__(
        self,
        options: EngineOptions,
        server: ServerOptions,
        *,
        tuning: EngineTuning | None = None,
        engine: Callable[..., EngineClient] = EngineClient,
        link: Callable[[ServingEndpoints], FrontendManager] = FrontendManager.linked,
    ) -> None:
        """``options`` and ``tuning`` (internal; None keeps its defaults) are the engine's,
        ``server`` where it listens and how many tokenizer workers it runs; ``/v1/models``
        lists the model's path as configured. ``engine`` starts the engine client and
        ``link`` links the frontend to its tokenizer workers."""

        self._options = options
        self._tuning = tuning
        self._server = server
        self._new_engine = engine
        self._link = link
        self._model_id = options.model.path
        self._engine: EngineClient | None = None
        # While the engine starts, its own startup watchdog bounds the wait: it renews on
        # every phase's progress and raises on a stall.
        self.starting_engine = False
        self._finalizer: Finalize | None = None
        self._closing = False
        self._loop: asyncio.AbstractEventLoop | None = None  # while it serves
        self._state: FrontendManager | None = None
        self._images: ImageFetcher | None = None  # opened and closed with the frontend
        self._tool_calls: ToolCallFormat | None = None  # read from the engine's model
        self.app = self._create_app()
        self.http_server = uvicorn.Server(
            uvicorn.Config(
                self.app, host=server.host, port=server.port, timeout_graceful_shutdown=10
            )
        )

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="cti API Server", version="0.0.1", lifespan=self._lifespan)
        app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])(self.v1_root)
        app.add_exception_handler(OpenAIError, _openai_error)
        app.add_exception_handler(RequestValidationError, _invalid_request)
        app.post("/v1/chat/completions")(self.v1_chat_completions)
        app.post("/v1/completions")(self.v1_completions)
        app.get("/v1/models")(self.available_models)
        return app

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI):
        """Link the frontend to the tokenizer processes, open the image fetcher, and stop
        the server once the engine fails, the frontend listener failing it too."""

        engine = self.engine
        self._state = frontend = self._link(engine.endpoints)
        self._images = images = ImageFetcher(self._server, engine.info.image_limits.max_images)
        self._tool_calls = ToolCallFormat.of(
            engine.info.chat_template, self._server.tool_call_parser
        )
        logger.info("Tool calls: %s", self._tool_calls or "the chat template takes no tools")
        self._loop = asyncio.get_running_loop()
        listener = asyncio.create_task(self._listen(frontend), name="frontend-listener")
        if engine.failure is not None:  # it failed before this loop could hear of it
            self._stop()
        try:
            yield
        finally:
            self._loop = None
            listener.cancel()
            await frontend.close()
            await images.aclose()
            await asyncio.gather(listener, return_exceptions=True)
            self._state = self._images = None

    async def _listen(self, frontend: FrontendManager) -> None:
        """Fail the engine once the frontend listener fails."""

        try:
            await frontend.listen()
        except Exception as error:  # the listener's failure is terminal
            logger.exception("frontend listener failed")
            await asyncio.to_thread(self.engine.fail, f"frontend listener failed: {error}")

    def _engine_failed(self) -> None:
        """Stop serving once the engine failed; its client calls this from the thread that
        saw the failure."""

        loop = self._loop
        if loop is not None:
            with suppress(RuntimeError):  # the loop closed as the server stopped
                loop.call_soon_threadsafe(self._stop)

    def _stop(self) -> None:
        if self._state is None:  # the server already stopped
            return
        reason = str(self.engine.failure)
        logger.error("Stopping API server: %s", reason)
        self.http_server.should_exit = True
        self._state.end(reason)

    def _frontend(self) -> FrontendManager:
        if self._state is None:
            raise RuntimeError("API server is not running")
        return self._state

    async def v1_root(self):
        self._require_engine_available()
        return {"status": "ok"}

    async def v1_chat_completions(self, body: ChatCompletionRequest, request: Request):
        call = GenerationCall(
            self._frontend(), self._images, body, ChatCompletionFormat, self._tool_calls
        )
        return await call.serve(request, prompt_param="messages")

    async def v1_completions(self, body: CompletionRequest, request: Request):
        call = GenerationCall(self._frontend(), self._images, body, CompletionFormat, None)
        return await call.serve(request, prompt_param="prompt")

    def sample(
        self,
        prompts: Sequence[SamplingPrompt],
        sampling_params: SamplingParams | Sequence[SamplingParams],
        priority: AdmissionPriority,
    ) -> SamplingResult:
        """Sample ``prompts`` as one command among the HTTP calls; called from any thread but
        the server's, which serves it. A model that changes while it samples raises."""

        policies = SamplingParams.specs(prompts, sampling_params)
        requests = [sampling_prompt(prompt) for prompt in prompts]
        priority = AdmissionPriority.parse(priority)
        loop = self._loop
        if loop is None:
            raise RuntimeError("API server is not running")
        sampling = asyncio.run_coroutine_threadsafe(
            self._sample(requests, policies, priority), loop
        )
        try:
            return sampling.result()
        except InferenceUnavailableError:
            self.engine.check()  # the engine's own failure, as offline sampling raises it
            raise
        except BaseException:
            sampling.cancel()  # a caller interrupted while waiting aborts the requests
            raise

    async def _sample(
        self,
        prompts: Sequence[Prompt],
        policies: Sequence[SamplingSpec],
        priority: AdmissionPriority,
    ) -> SamplingResult:
        return await SamplingCall(self._frontend()).samples(prompts, policies, priority)

    async def available_models(self):
        self._require_engine_available()
        card = {"id": self._model_id, "object": "model", "created": int(time.time())}
        return {
            "object": "list",
            "data": [{**card, "owned_by": "cti", "root": self._model_id}],
        }

    def _require_engine_available(self) -> None:
        self._frontend()
        if (failure := self.engine.failure) is not None:
            raise HTTPException(status_code=503, detail=str(failure))

    def run(self) -> None:
        """Bind the server's address, start the engine and serve until the HTTP server
        stops; an engine failure stops it and raises. ``close`` shuts the engine down."""

        with self._bind() as listener:
            self.start_engine()
            # The bound port: the configured one, or the OS's pick for port 0.
            logger.info(
                "API server is ready to serve on %s:%d",
                self._server.host,
                listener.getsockname()[1],
            )
            self.http_server.run(sockets=[listener])
        self._engine.check()

    @property
    def engine(self) -> EngineClient:
        if self._engine is None:
            raise RuntimeError("API server is not running")
        return self._engine

    @property
    def is_closing(self) -> bool:
        return self._closing

    def start_engine(self) -> None:
        """Start the online engine, reclaimed when this server is collected or the
        interpreter exits; ``run`` starts it once the server's address is bound."""

        self.starting_engine = True
        try:
            engine = self._new_engine(
                self._options,
                self._tuning,
                num_tokenizers=self._server.num_tokenizers,
                on_failure=self._engine_failed,
            )
        finally:
            self.starting_engine = False
        self._finalizer = engine.close_at_exit(self)
        self._engine = engine
        logger.info("Inference engine is ready")

    def kill_workers(self) -> None:
        """SIGKILL the engine's tokenizer workers, waiting for nothing: a signal handler's
        forced shutdown."""

        if self._engine is not None:
            self._engine.kill_workers()

    def close(self) -> None:
        """Shut the engine down; every process is gone when it returns, even when the close
        itself fails. Closing again, or while closing, does nothing."""

        engine, finalizer = self._engine, self._finalizer
        if engine is None or self._closing:
            return
        self._closing = True
        try:
            engine.close()
        except BaseException as error:
            # The failed close (a busy training session) left the engine running, and
            # nothing closes it once ownership is dropped: reap it now.
            try:
                finalizer()
            except BaseException as reap_error:  # noqa: BLE001 - keep the close error
                error.add_note(f"Engine reclamation also failed: {reap_error}")
            raise
        else:
            finalizer.cancel()
            logger.info("Inference engine stopped")
        finally:
            self._engine = self._finalizer = None
            self._closing = False

    def _bind(self) -> socket.socket:
        """The server's socket, listening before the engine starts so a busy address, or
        another server starting on it, fails at once; connections queue until uvicorn
        serves on it."""

        host, port = self._server.host, self._server.port
        # IPPROTO_TCP, as uvicorn's own sockets: asyncio sets TCP_NODELAY only on such
        # connections, and without it each stream's first chunk waits out a delayed ACK.
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, port))
            listener.listen()
        except OSError as error:
            listener.close()
            raise SystemExit(f"cannot serve on {host}:{port}: {error.strerror}") from None
        return listener
