"""Offline sampling: the requests an online tokenizer sends, sent over the same request link,
their replies gathered into exact samples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

from cti.protocol import (
    AbortBackendMsg,
    BaseBackendMsg,
    BaseTokenizerMsg,
    DetokenizeMsg,
    RequestAbortedMsg,
    RequestRejectedError,
    RequestRejectedMsg,
    SampledSequence,
    SamplingResult,
    UserMsg,
)
from cti.protocol.link import ZmqPullQueue, ZmqPushQueue

if TYPE_CHECKING:
    from cti.protocol import (
        RequestEnd,
        SamplingPrompt,
        SamplingSpec,
        ServingEndpoints,
    )
    from cti.runtime._client import EngineClient
    from cti.sampling import TokenLogprobs
    from cti.tokenizer import PromptEncoder


@dataclass(slots=True)
class _Record:
    """One request of a command, as its replies arrive."""

    prompt_token_ids: tuple[int, ...]
    sampling: SamplingSpec
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[TokenLogprobs] = field(default_factory=list)
    prompt_logprobs: tuple[TokenLogprobs | None, ...] | None = None
    end: RequestEnd | None = None

    def sample(self, request_id: int) -> SampledSequence:
        assert self.end is not None
        return SampledSequence.assemble(
            request_id,
            self.prompt_token_ids,
            self.sampling,
            token_ids=self.token_ids,
            logprobs=self.logprobs,
            prompt_logprobs=self.prompt_logprobs,
            ending=self.end,
        )


class _Command:
    """One sampling command's requests, those not yet ended, and its first rejection, which
    aborts the rest and fails the command once they are released."""

    def __init__(self, first_uid: int, requests: Sequence[UserMsg]) -> None:
        self._first = first_uid
        self.records = {
            msg.uid: _Record(tuple(msg.input_ids.tolist()), msg.sampling) for msg in requests
        }
        self.open = set(self.records)
        self.aborts: list[BaseBackendMsg] = []
        self.rejection: RequestRejectedError | None = None

    def receive(self, reply: BaseTokenizerMsg) -> None:
        """Take one reply to the command: a request's tokens, its release or its rejection."""

        match reply:
            case DetokenizeMsg():
                self._token(reply)
            case RequestAbortedMsg():
                self.open.discard(reply.uid)
            case RequestRejectedMsg():
                self._rejected(reply)
            case _:
                raise TypeError(f"{type(reply).__name__} is not a reply to a request")

    def _token(self, msg: DetokenizeMsg) -> None:
        record = self.records[msg.uid]
        record.token_ids.extend(msg.token_ids)
        record.logprobs.extend(msg.logprobs)
        if msg.prompt_logprobs is not None:
            record.prompt_logprobs = msg.prompt_logprobs
        if msg.end is not None:
            record.end = msg.end
            self.open.discard(msg.uid)

    def _rejected(self, msg: RequestRejectedMsg) -> None:
        self.open.discard(msg.uid)
        request_id = msg.uid - self._first
        if self.rejection is None:
            self.aborts = [AbortBackendMsg(uid=uid) for uid in sorted(self.open)]
        if self.rejection is None or request_id < self.rejection.request_id:
            self.rejection = msg.error(request_id)

    def result(self, encoder: PromptEncoder) -> SamplingResult:
        records = self.records.values()
        samples = tuple(record.sample(uid - self._first) for uid, record in self.records.items())
        return SamplingResult.at_revision(
            samples,
            tuple(_text(encoder, sample) for sample in samples),
            (record.end.model_revision for record in records),
        )


def _text(encoder: PromptEncoder, sample: SampledSequence) -> str:
    """A sample's output text as the detokenizer streams it, cut where the stop string that
    ended it begins."""

    from cti.tokenizer import IncrementalDetokenizer

    return sample.visible_text(
        IncrementalDetokenizer.render_text(
            encoder.decoder,
            sample.output_token_ids,
            skip_special_tokens=sample.sampling.skip_special_tokens,
        )
    )


class OfflineSampler:
    """Samples through ``client``'s engine on its private ``endpoints``, one command at a
    time, its requests numbered like a frontend's."""

    POLL_MS = 100  # how often a wait for replies checks that the engine is alive

    def __init__(self, endpoints: ServingEndpoints, client: EngineClient) -> None:
        self._client = client
        self._requests: ZmqPushQueue[BaseBackendMsg] = ZmqPushQueue(endpoints, "engine")
        self._replies: ZmqPullQueue[BaseTokenizerMsg] = ZmqPullQueue(endpoints, "detokenizer")
        self._next_uid = 0
        self._lock = Lock()

    def close(self) -> None:
        self._requests.stop()
        self._replies.stop()

    def sample(
        self, prompts: Sequence[SamplingPrompt], policies: Sequence[SamplingSpec]
    ) -> SamplingResult:
        with self._lock:
            encoder = self._client.prompt_encoder()
            first, self._next_uid = self._next_uid, self._next_uid + len(prompts)
            requests = [
                encoder.request(first + index, prompt, policy)
                for index, (prompt, policy) in enumerate(zip(prompts, policies, strict=True))
            ]
            command = _Command(first, requests)
            # One message: the engine takes the command's requests in together, so a
            # repeated command is admitted, and batched, alike.
            self._requests.put(requests)
            while command.open:
                if not self._replies.socket.poll(self.POLL_MS):
                    self._client.check()
                    continue
                for reply in self._replies.get():
                    command.receive(reply)
                self._requests.put(command.aborts)
                command.aborts = []
            if command.rejection is not None:
                raise command.rejection
            return command.result(encoder)
