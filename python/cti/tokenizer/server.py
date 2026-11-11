# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/tokenizer/server.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, NoReturn, TypeVar

import torch

from cti.protocol import (
    AbortBackendMsg,
    AbortMsg,
    BaseTokenizerMsg,
    DetokenizeMsg,
    PromptReply,
    RejectionKind,
    RequestAbortedMsg,
    RequestRejectedMsg,
    ServingEndpoints,
    TextPrompt,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from cti.protocol.link import ZmqPullQueue, ZmqPushQueue
from cti.utils import init_logger, log_stage

from .detokenize import IncrementalDetokenizer
from .prompt import PromptEncoder
from .text import load_text_decoder

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from cti.multimodal import ImageLimits


def tokenize(encoder: PromptEncoder, msg: TokenizeMsg) -> tuple[UserMsg, PromptReply | None]:
    """One prompt's engine request, and its reply when asked; ValueError rejects it."""

    request = encoder.request(msg.uid, msg.prompt, msg.sampling, msg.priority)
    if not msg.return_prompt:
        return request, None
    token_ids = tuple(request.input_ids.tolist())
    if isinstance(msg.prompt, TextPrompt):
        text = msg.prompt.text
    else:  # the prompt repeats as encoded, its special tokens too
        text = encoder.decode(token_ids, skip_special_tokens=False)
    reply = PromptReply(msg.uid, token_ids, text)
    if msg.sampling.law.prompt_logprobs is not None and token_ids:
        # The first prompt token has no logprob; the others' texts come with the output.
        reply.token_texts[token_ids[0]] = encoder.decode(token_ids[:1], skip_special_tokens=False)
    return request, reply


@torch.inference_mode()
def _step_replies(
    detokenizer: IncrementalDetokenizer, msgs: list[DetokenizeMsg]
) -> list[UserReply]:
    """One reply per request of the engine steps a batch holds, with every token they
    committed for it (a speculative round commits several), so a stream sends a step's
    tokens together."""

    texts = detokenizer.detokenize(msgs)
    token_texts = detokenizer.token_texts(msgs)
    steps: dict[int, list[int]] = {}
    for index, msg in enumerate(msgs):
        steps.setdefault(msg.uid, []).append(index)
    replies = []
    for indices in steps.values():
        first, last = msgs[indices[0]], msgs[indices[-1]]
        named: dict[int, str] = {}
        for index in indices:
            named.update(token_texts[index])
        end = last.end
        replies.append(
            UserReply(
                uid=last.uid,
                incremental_output="".join(texts[index] for index in indices),
                finished=end is not None,
                usage=None if end is None else end.usage,
                finish_reason=None if end is None else end.finish_reason,
                token_ids=tuple(t for index in indices for t in msgs[index].token_ids),
                stop_str=None if end is None else end.stop_str,
                logprobs=tuple(entry for index in indices for entry in msgs[index].logprobs),
                prompt_logprobs=first.prompt_logprobs,
                token_texts=named,
                max_new_tokens=None if end is None else end.max_new_tokens,
                model_revision=None if end is None else end.model_revision,
            )
        )
    return replies


_Queue = TypeVar("_Queue", ZmqPullQueue, ZmqPushQueue)


def _stopped_with(resources: ExitStack, queue: _Queue) -> _Queue:
    """``queue``, stopped as the process's ``resources`` close."""

    resources.callback(queue.stop)
    return queue


class DetokenizerWorker:
    """The detokenizer process: turns the engine's step replies into each request's text
    for the frontend, and passes the engine's rejections on."""

    def __init__(
        self, decoder: IncrementalDetokenizer, endpoints: ServingEndpoints, resources: ExitStack
    ) -> None:
        self._detokenizer = decoder
        self._replies = _stopped_with(resources, ZmqPullQueue(endpoints, "detokenizer"))
        self._frontend = _stopped_with(resources, ZmqPushQueue(endpoints, "frontend"))
        self._logger = init_logger(__name__, "detokenizer")
        self._started: set[int] = set()  # requests whose first output token has arrived

    def run(self) -> NoReturn:
        while True:
            self._serve(self._replies.get())

    def _serve(self, msgs: Sequence[BaseTokenizerMsg]) -> None:
        steps = [msg for msg in msgs if isinstance(msg, DetokenizeMsg)]
        first = [msg.uid for msg in steps if msg.uid not in self._started]
        self._started.update(first)
        log_stage(self._logger, "first_token", first)
        if steps:
            self._frontend.put(_step_replies(self._detokenizer, steps))
            log_stage(self._logger, "first_reply_sent", first)
            self._started.difference_update(msg.uid for msg in steps if msg.finished)
        # The engine publishes an abort after the request's last token, so release only
        # after decoding the tokens coalesced with it.
        rejected = []
        for msg in msgs:
            match msg:
                case DetokenizeMsg():
                    pass
                case RequestAbortedMsg(uid=uid):
                    self._detokenizer.abort(uid)
                    self._started.discard(uid)
                case RequestRejectedMsg():
                    rejected.append(msg)
                case _:
                    raise TypeError(f"the detokenizer received {type(msg).__name__}")
        self._frontend.put(rejected)


@dataclass(slots=True)
class _Call:
    """One call's prompts as they are preprocessed, and the aborts that must follow its
    requests to the engine."""

    prompts: list[TokenizeMsg]
    results: list[Future[tuple[UserMsg, PromptReply | None] | RequestRejectedMsg]]
    aborts: list[AbortBackendMsg] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return all(result.done() for result in self.results)


class TokenizerWorker:
    """A tokenizer process: preprocesses the prompts of the calls the frontend sends it on a
    thread pool (image decoding releases the GIL), several calls at once, and sends each
    call's requests to the engine as one message, in its order, once all of them are ready:
    the engine admits a call's requests together. An abort follows its call's requests.
    Every socket stays on the process's own thread."""

    POOL_WORKERS = 4
    POLL_MS = 1  # while calls are being preprocessed, how often new ones are taken in

    def __init__(
        self, encoder: PromptEncoder, endpoints: ServingEndpoints, index: int, resources: ExitStack
    ) -> None:
        self._encoder = encoder
        self._calls = _stopped_with(resources, ZmqPullQueue(endpoints, "tokenizer", index=index))
        self._engine = _stopped_with(resources, ZmqPushQueue(endpoints, "engine"))
        self._frontend = _stopped_with(resources, ZmqPushQueue(endpoints, "frontend"))
        self._pool = resources.enter_context(
            ThreadPoolExecutor(self.POOL_WORKERS, thread_name_prefix=f"tokenizer-{index}")
        )
        self._logger = init_logger(__name__, f"tokenizer_{index}")
        self._pending: list[_Call] = []
        self._owners: dict[int, _Call] = {}  # each pending request's call

    def run(self) -> NoReturn:
        while True:
            if self._calls.socket.poll(self.POLL_MS if self._pending else None):
                self._take(self._calls.get())
            self._send_ready()

    def _take(self, msgs: Sequence[BaseTokenizerMsg]) -> None:
        prompts: list[TokenizeMsg] = []
        aborts: list[AbortBackendMsg] = []
        for msg in msgs:
            match msg:
                case TokenizeMsg():
                    prompts.append(msg)
                case AbortMsg(uid=uid):
                    aborts.append(AbortBackendMsg(uid=uid))
                case _:
                    raise TypeError(f"a tokenizer received {type(msg).__name__}")
        if prompts:
            log_stage(self._logger, "tokenizer_received", (msg.uid for msg in prompts))
            call = _Call(prompts, [self._pool.submit(self._tokenize, msg) for msg in prompts])
            self._pending.append(call)
            self._owners.update((msg.uid, call) for msg in prompts)
        # An abort of a call still preprocessing waits for the call's requests.
        now = []
        for abort in aborts:
            if (call := self._owners.get(abort.uid)) is not None:
                call.aborts.append(abort)
            else:
                now.append(abort)
        self._engine.put(now)

    def _send_ready(self) -> None:
        """Send every call whose prompts are all preprocessed."""

        ready = [call for call in self._pending if call.done]
        for call in ready:
            self._pending.remove(call)
            for msg in call.prompts:
                del self._owners[msg.uid]
            self._send(call)

    def _send(self, call: _Call) -> None:
        requests = []
        for msg, future in zip(call.prompts, call.results, strict=True):
            result = future.result()
            if isinstance(result, RequestRejectedMsg):
                self._frontend.put([result])
                continue
            request, reply = result
            log_stage(self._logger, "tokenized", (msg.uid,))
            # The prompt reply leaves before the request reaches the engine.
            if reply is not None:
                self._frontend.put([reply])
            requests.append(request)
        self._engine.put(requests)
        log_stage(self._logger, "to_engine", (request.uid for request in requests))
        self._engine.put(call.aborts)

    @torch.inference_mode()
    def _tokenize(
        self, msg: TokenizeMsg
    ) -> tuple[UserMsg, PromptReply | None] | RequestRejectedMsg:
        try:
            return tokenize(self._encoder, msg)
        except Exception as error:  # noqa: BLE001 - reject only this request
            return RequestRejectedMsg(
                uid=msg.uid,
                reason=f"Tokenization failed: {error}",
                kind=RejectionKind.INVALID_REQUEST,
            )


def tokenize_worker(
    *,
    model_path: str,
    hf_config: PretrainedConfig,
    image_limits: ImageLimits,
    endpoints: ServingEndpoints,
    tokenizer_id: int | None = None,
    ready_connection: Connection | None = None,
) -> None:
    """Serve one worker process of an online server: tokenizer ``tokenizer_id``, or the
    detokenizer when it is None."""

    name = "detokenizer" if tokenizer_id is None else f"tokenizer {tokenizer_id}"
    try:
        with ExitStack() as resources:
            worker: DetokenizerWorker | TokenizerWorker
            if tokenizer_id is None:
                decoder = IncrementalDetokenizer(load_text_decoder(model_path))
                worker = DetokenizerWorker(decoder, endpoints, resources)
            else:
                encoder = PromptEncoder.from_checkpoint(
                    model_path, hf_config=hf_config, limits=image_limits
                )
                worker = TokenizerWorker(encoder, endpoints, tokenizer_id, resources)
            if ready_connection is not None:
                ready_connection.send(f"The {name} is ready")
                ready_connection.close()
                ready_connection = None
            worker.run()
    except KeyboardInterrupt:
        pass
    finally:
        if ready_connection is not None:
            ready_connection.close()
