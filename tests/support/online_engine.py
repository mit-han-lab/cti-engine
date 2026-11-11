"""An online engine for GPU tests: an ``EngineClient``, its backend queues, and greedy
requests sent and collected through them. A training session on it can pause for
serving."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cti.config import EngineOptions, EngineTuning
    from cti.runtime import TrainingSession

_REPLY_TIMEOUT_MS = 120_000


class RequestRejected(Exception):
    """The engine refused an online request."""


class OnlineEngine:
    def __init__(self, options: EngineOptions, tuning: EngineTuning) -> None:
        from cti.protocol.link import ZmqPullQueue, ZmqPushQueue
        from cti.runtime import EngineClient

        self._engine = EngineClient(options, tuning)
        endpoints = self._engine.endpoints
        self._send = ZmqPushQueue(endpoints, "engine")
        self._receive = ZmqPullQueue(endpoints, "detokenizer")
        self._next_uid = 0

    def training_session(self) -> TrainingSession:
        from cti.runtime import TrainingSession

        return TrainingSession(self, self._engine)

    @property
    def model_revision(self) -> int:
        return self._engine.model_revision()

    def generate(self, prompt_ids: list[int], max_tokens: int) -> list[int]:
        """Greedy tokens of one request; raises ``RequestRejected`` when it is refused."""

        import torch

        from cti.protocol import (
            DetokenizeMsg,
            RequestRejectedMsg,
            SamplingParams,
            UserMsg,
        )

        uid, self._next_uid = self._next_uid, self._next_uid + 1
        self._send.put(
            [
                UserMsg(
                    uid=uid,
                    input_ids=torch.tensor(prompt_ids, dtype=torch.int32),
                    sampling=SamplingParams(
                        temperature=0, ignore_eos=True, max_new_tokens=max_tokens
                    ).snapshot(),
                )
            ]
        )
        tokens: list[int] = []
        while True:
            if self._receive.socket.poll(timeout=_REPLY_TIMEOUT_MS) == 0:
                raise TimeoutError(f"online request {uid} did not complete")
            for message in self._receive.get():
                if isinstance(message, RequestRejectedMsg) and message.uid == uid:
                    raise RequestRejected(message.reason)
                if isinstance(message, DetokenizeMsg) and message.uid == uid:
                    tokens.extend(message.token_ids)
                    if message.finished:
                        return tokens

    def close(self) -> None:
        self._send.stop()
        self._receive.stop()
        self._engine.close()

    def __enter__(self) -> OnlineEngine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
