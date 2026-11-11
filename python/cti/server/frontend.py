# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/server/api_server.py`); see THIRD_PARTY_NOTICES.

"""The API server's frontend: its links to the tokenizer workers, and the calls whose engine
requests they carry, each owning its requests until they end."""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Collection,
    Iterable,
    Sequence,
)
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, NoReturn, TypeAlias, TypeVar

from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from cti.protocol import (
    AbortMsg,
    AdmissionPriority,
    BaseTokenizerMsg,
    PromptReply,
    RejectionKind,
    RequestRejectedError,
    RequestRejectedMsg,
    SampledSequence,
    SamplingResult,
    TextPrompt,
    TokenizeMsg,
    TokenPrompt,
    TokenUsage,
    UserReply,
)
from cti.protocol.link import ZmqAsyncPullQueue, ZmqAsyncPushQueue
from cti.utils import init_logger, log_stage

from .image_fetch import ImageFetcher
from .openai.rendering import CallRenderer, ChoiceDelta, ChoiceOutput
from .openai.requests import ChatCompletionRequest, CompletionRequest, OpenAIError
from .openai.responses import ResponseFormat
from .tool_calls import ToolCallFormat

if TYPE_CHECKING:
    from cti.protocol import Prompt, SamplingPrompt, SamplingSpec, ServingEndpoints

logger = init_logger(__name__, "FrontendAPI")

# A request's output: its prompt, when it repeats it, and its replies.
_RequestOutput: TypeAlias = PromptReply | UserReply
_FrontendReply: TypeAlias = _RequestOutput | RequestRejectedMsg
_T = TypeVar("_T")


class InferenceUnavailableError(RuntimeError):
    """Online inference has failed and cannot settle pending requests."""


# A generation call's replies, each with its request's index in the call.
_Inbox: TypeAlias = asyncio.Queue[tuple[int, _FrontendReply | InferenceUnavailableError]]


async def _disconnected(request: Request) -> None:
    """Return once the client disconnects; the request body has already been read."""

    while (await request.receive())["type"] != "http.disconnect":
        pass


class FrontendManager:
    """The links to the tokenizer processes: it sends requests and routes each reply to the
    inbox of the generation call that owns its request. A call's requests go to one
    tokenizer, the calls taking turns, and so do their aborts: a link keeps its messages'
    order, so an abort reaches the engine after its request."""

    def __init__(
        self,
        send_tokenizers: Sequence[ZmqAsyncPushQueue[BaseTokenizerMsg]],
        recv_tokenizer: ZmqAsyncPullQueue[_FrontendReply],
    ) -> None:
        self._sends = tuple(send_tokenizers)
        self._recv = recv_tokenizer
        self._routes: Dict[int, tuple[int, _Inbox]] = {}
        # Each sent request's tokenizer, until the request is released.
        self._tokenizers: Dict[int, int] = {}
        self._turns = itertools.count()
        self._answered: set[int] = set()  # requests whose first output has arrived
        self._uids = itertools.count()
        self._aborts: dict[asyncio.Task[None], int] = {}  # each abort in flight, by request
        self._ended: str | None = None

    @classmethod
    def linked(cls, endpoints: ServingEndpoints) -> FrontendManager:
        """The frontend over ZMQ links to each of the engine's tokenizer workers."""

        return cls(
            send_tokenizers=tuple(
                ZmqAsyncPushQueue(endpoints, "tokenizer", index=index)
                for index in range(endpoints.num_tokenizers)
            ),
            recv_tokenizer=ZmqAsyncPullQueue(endpoints, "frontend"),
        )

    @property
    def pending(self) -> frozenset[int]:
        """The requests it still holds: routed to a call, or with an abort in flight."""

        return frozenset(self._routes) | frozenset(self._aborts.values())

    def new_user(self, inbox: _Inbox, index: int) -> int:
        """A new request, whose replies reach ``inbox`` as ``(index, reply)``."""

        if self._ended is not None:
            raise InferenceUnavailableError(self._ended)
        uid = next(self._uids)
        self._routes[uid] = index, inbox
        return uid

    async def send(self, msgs: Sequence[TokenizeMsg]) -> None:
        """Send one call's requests as one message: one tokenizer takes them together."""

        tokenizer = next(self._turns) % len(self._sends)
        for msg in msgs:
            self._tokenizers[msg.uid] = tokenizer
        await self._sends[tokenizer].put(msgs)

    async def listen(self) -> NoReturn:
        """Route replies to their requests' inboxes."""

        while True:
            for msg in await self._recv.get():
                if not isinstance(msg, (UserReply, PromptReply, RequestRejectedMsg)):
                    raise TypeError(f"unsupported frontend message type: {type(msg)}")
                self._route(msg)

    def _route(self, reply: _FrontendReply) -> None:
        # A dropped request ignores its replies.
        if (route := self._routes.get(reply.uid)) is not None:
            index, inbox = route
            if isinstance(reply, UserReply) and reply.uid not in self._answered:
                self._answered.add(reply.uid)
                log_stage(logger, "first_reply", (reply.uid,))
            inbox.put_nowait((index, reply))

    def release(self, uids: Iterable[int], running: Collection[int]) -> None:
        """Drop the requests' routes now and abort the ``running`` ones in the engine in the
        background; a request that has ended ignores the abort."""

        for uid in uids:
            self._routes.pop(uid, None)
            self._answered.discard(uid)
            if uid in running:
                self._schedule_abort(uid)
            else:
                self._tokenizers.pop(uid, None)

    def _schedule_abort(self, uid: int) -> None:
        if self._ended is not None:
            self._tokenizers.pop(uid, None)
            return
        task = asyncio.create_task(self._abort(uid), name=f"abort-user-{uid}")
        self._aborts[task] = uid
        task.add_done_callback(self._abort_done)

    async def _abort(self, uid: int) -> None:
        # The request's tokenizer takes its abort after the request; one never sent has none.
        if (tokenizer := self._tokenizers.pop(uid, None)) is None:
            return
        logger.warning("Aborting request for user %s", uid)
        await self._sends[tokenizer].put([AbortMsg(uid=uid)])

    def _abort_done(self, task: asyncio.Task[None]) -> None:
        self._aborts.pop(task, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Frontend task %s failed", task.get_name(), exc_info=error)

    def end(self, reason: str) -> None:
        """Fail every routed request and every new one with ``reason``."""

        if self._ended is not None:
            return
        self._ended = reason
        for index, inbox in self._routes.values():
            inbox.put_nowait((index, InferenceUnavailableError(reason)))

    async def close(self) -> None:
        self.end("frontend manager is closed")
        aborts = tuple(self._aborts)
        for task in aborts:
            task.cancel()
        # Pending async ZMQ operations are not guaranteed to wake promptly from
        # task cancellation alone. Close the owned transports before awaiting
        # task completion.
        self._recv.stop()
        for send in self._sends:
            send.stop()
        await asyncio.gather(*aborts, return_exceptions=True)


class _Call:
    """A call's engine requests, whose replies reach its inbox: the call owns each request
    until it has ended."""

    def __init__(self, state: FrontendManager) -> None:
        self._state = state
        self._uids: List[int] = []
        self._running: set[int] = set()
        self._inbox: _Inbox = asyncio.Queue()

    async def _send(
        self,
        requests: Iterable[tuple[Prompt, SamplingSpec]],
        priority: AdmissionPriority,
        return_prompt: bool,
    ) -> None:
        """Send one request per prompt and policy in ``requests``, as one message: one
        tokenizer takes the call's requests and the engine admits them together, so a
        repeated call is batched alike."""

        msgs = []
        for prompt, policy in requests:
            uid = self._state.new_user(self._inbox, len(self._uids))
            self._uids.append(uid)
            self._running.add(uid)
            msgs.append(TokenizeMsg(uid, prompt, policy, priority, return_prompt))
        await self._state.send(msgs)

    def abort(self) -> None:
        """Drop the call's requests and abort those that have not ended."""

        self._state.release(self._uids, self._running)
        self._running.clear()

    async def _receive(self) -> tuple[int, _RequestOutput]:
        """The next reply to the call; a rejection or an unavailable engine raises."""

        index, reply = await self._inbox.get()
        if isinstance(reply, InferenceUnavailableError):
            raise reply
        if isinstance(reply, RequestRejectedMsg):
            raise reply.error(index)
        return index, reply


class GenerationCall(_Call):
    """One generation call: ``n`` engine requests per prompt, one choice each, in choice
    order. A call that fails or whose client leaves aborts its requests, and a stream owns
    them from its start."""

    def __init__(
        self,
        state: FrontendManager,
        images: ImageFetcher,
        body: ChatCompletionRequest | CompletionRequest,
        response_format: type[ResponseFormat],
        tool_calls: ToolCallFormat | None,
    ) -> None:
        super().__init__(state)
        self._images = images
        self._body = body
        self._response_format = response_format
        self._tool_calls = tool_calls
        self._renderer: CallRenderer | None = None
        self._answered: List[tuple[int, _RequestOutput]] = []

    async def serve(self, request: Request, *, prompt_param: str):
        policies = self._body.sampling()  # a refused field is a 400
        streaming = False
        try:
            answer = await self._unless_disconnected(self._answer(policies), request)
            if answer is None:
                return Response(status_code=499)  # the client left; nginx's code for it
            streaming = isinstance(answer, _OwnedStream)
            return answer
        except RequestRejectedError as rejected:
            kind = rejected.kind
            if kind is RejectionKind.UNAVAILABLE:
                raise OpenAIError(str(rejected), status_code=503) from rejected
            code = (
                "context_length_exceeded" if kind is RejectionKind.CONTEXT_LENGTH_EXCEEDED else None
            )
            raise OpenAIError(rejected.reason, param=prompt_param, code=code) from rejected
        except InferenceUnavailableError as error:
            raise OpenAIError(str(error), status_code=503) from error
        finally:
            if not streaming:
                self.abort()

    @staticmethod
    async def _unless_disconnected(answer: Awaitable[_T], request: Request) -> _T | None:
        """``answer``'s result, or None if the client disconnects first, which cancels it."""

        answering = asyncio.ensure_future(answer)
        disconnected = asyncio.ensure_future(_disconnected(request))
        try:
            await asyncio.wait({answering, disconnected}, return_when=asyncio.FIRST_COMPLETED)
            return answering.result() if answering.done() else None
        finally:
            for future in (answering, disconnected):
                future.cancel()
            await asyncio.gather(answering, disconnected, return_exceptions=True)

    async def _answer(self, policies: Sequence[SamplingSpec]) -> Dict[str, object] | _OwnedStream:
        """Fetch the images the call's prompts name, dispatch its requests, each prompt's
        choices with ``policies``, then answer with the whole response, or with a stream
        once the engine has answered every request."""

        body = self._body
        received = time.monotonic()
        prompts = body.prompts()
        # Before any work: a request for tools the chat template does not take is a 400.
        self._renderer = CallRenderer(body, len(prompts), self._tool_calls)
        prompts = [await self._images.resolve(prompt) for prompt in prompts]
        fetched = time.monotonic()
        requests = ((prompt, policy) for prompt in prompts for policy in policies)
        await self._send(requests, body.priority, body.return_prompt)
        log_stage(logger, "received", self._uids, received)
        log_stage(logger, "images_fetched", self._uids, fetched)
        log_stage(logger, "sent", self._uids)
        response = self._response_format(body, self._uids[0])
        if body.stream:
            await self._wait_for_answers()
            return _OwnedStream(self._stream(response), self)
        return response.response(await self._collect())

    async def _wait_for_answers(self) -> None:
        """Wait until the engine has answered every request, keeping the replies for
        ``_replies``; the tokenizer's prompt reply is not an answer. The first rejection
        raises at once, so a streaming response can still answer with its status."""

        unanswered = set(range(len(self._uids)))
        while unanswered:
            index, reply = await self._receive()
            self._answered.append((index, reply))
            if not isinstance(reply, PromptReply):
                unanswered.discard(index)

    async def _replies(self) -> AsyncIterator[tuple[int, _RequestOutput]]:
        """The requests' output as it arrives, each with its request's index, until each has
        ended and, when the call repeats prompts, sent its prompt, which may come last."""

        ending = set(range(len(self._uids)))
        prompting = set(ending) if self._body.return_prompt else set()
        answered = iter(self._answered)
        while ending or prompting:
            index, reply = next(answered, None) or await self._receive()
            yield index, reply
            if isinstance(reply, PromptReply):
                prompting.discard(index)
            elif reply.finished:
                ending.discard(index)

    def _deltas(self, index: int, reply: _RequestOutput) -> list[tuple[int, ChoiceDelta]]:
        assert self._renderer is not None
        return self._renderer.add(index, reply)

    async def _collect(self) -> List[ChoiceOutput]:
        """Every request's whole output."""

        choices = [ChoiceOutput() for _ in self._uids]
        async for index, reply in self._replies():
            for choice, delta in self._deltas(index, reply):
                choices[choice].add(delta)
                if delta.finish_reason is not None:
                    self._running.discard(self._uids[choice])
        return choices

    async def _stream(self, response: ResponseFormat) -> AsyncGenerator[bytes, None]:
        usages: Dict[int, TokenUsage] = {}
        opened: set[int] = set()
        try:
            async for index, reply in self._replies():
                for choice, delta in self._deltas(index, reply):
                    first = choice not in opened
                    opened.add(choice)
                    # One chunk per released token, as streaming clients time tokens by chunks.
                    if first or delta.text or delta.token_ids or delta.tool_calls:
                        yield _event(response.delta_chunk(choice, delta, first))
                    if delta.finish_reason is not None:
                        self._running.discard(self._uids[choice])
                        usages[choice] = delta.usage
                        yield _event(response.end_chunk(choice, delta.finish_reason))
        except InferenceUnavailableError as error:
            yield _event({"error": {"message": str(error), "type": "server_error"}})
            yield b"data: [DONE]\n\n"
            return
        if self._body.include_usage:
            yield _event(response.usage_chunk([usages[index] for index in sorted(usages)]))
        yield b"data: [DONE]\n\n"


class _OwnedStream(StreamingResponse):
    """An event stream that owns its call's requests: however its response ends, it closes
    its chunks and aborts the requests that have not ended."""

    def __init__(self, chunks: AsyncGenerator[bytes, None], call: GenerationCall) -> None:
        super().__init__(chunks, media_type="text/event-stream")
        self._chunks = chunks
        self._call = call

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._chunks.aclose()
            self._call.abort()


@dataclass
class _Sample:
    """One request of a sampling call, as its replies arrive."""

    sampling: SamplingSpec
    prompt: PromptReply | None = None
    replies: List[UserReply] = field(default_factory=list)

    @property
    def ended(self) -> bool:
        return self.prompt is not None and bool(self.replies) and self.replies[-1].finished

    def add(self, reply: _RequestOutput) -> None:
        if isinstance(reply, PromptReply):
            self.prompt = reply
        else:
            self.replies.append(reply)

    def sample(self, request_id: int) -> tuple[SampledSequence, str]:
        """The request's exact sample, as the offline sampler gathers it, and its text cut
        where the stop string that ended it begins."""

        assert self.prompt is not None
        sample = SampledSequence.assemble(
            request_id,
            self.prompt.token_ids,
            self.sampling,
            token_ids=[t for reply in self.replies for t in reply.token_ids],
            logprobs=[entry for reply in self.replies for entry in reply.logprobs],
            prompt_logprobs=self.replies[0].prompt_logprobs,
            ending=self.replies[-1],
        )
        return sample, sample.visible_text("".join(r.incremental_output for r in self.replies))


class SamplingCall(_Call):
    """One bounded sampling command of the server's own process, admitted with the HTTP
    calls: one request per prompt, gathered into exact samples. Its first rejection aborts
    the rest and raises; however it ends, it aborts the requests that have not ended."""

    async def samples(
        self,
        prompts: Sequence[Prompt],
        policies: Sequence[SamplingSpec],
        priority: AdmissionPriority,
    ) -> SamplingResult:
        """The samples, in prompt order, their texts, and the revision they ran at."""

        requests = [_Sample(policy) for policy in policies]
        try:
            await self._send(zip(prompts, policies, strict=True), priority, return_prompt=True)
            while not all(request.ended for request in requests):
                index, reply = await self._receive()
                requests[index].add(reply)
                if isinstance(reply, UserReply) and reply.finished:
                    self._running.discard(self._uids[index])
        except RequestRejectedError as rejected:
            self._running.discard(self._uids[rejected.request_id])
            raise
        finally:
            self.abort()
        samples, texts = zip(
            *(request.sample(i) for i, request in enumerate(requests)), strict=True
        )
        revisions = (request.replies[-1].model_revision for request in requests)
        return SamplingResult.at_revision(samples, texts, revisions)


def sampling_prompt(prompt: SamplingPrompt) -> Prompt:
    """A sampling prompt as the tokenizer takes it: text or token ids, as online sampling
    takes no chats or images yet."""

    if isinstance(prompt, str):
        return TextPrompt(prompt)
    if isinstance(prompt, list) and all(type(token_id) is int for token_id in prompt):
        return TokenPrompt(tuple(prompt))
    raise TypeError("online sampling takes text or token-id prompts, not chats or images")


def _event(payload: Dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


__all__ = [
    "FrontendManager",
    "GenerationCall",
    "InferenceUnavailableError",
    "SamplingCall",
    "sampling_prompt",
]
