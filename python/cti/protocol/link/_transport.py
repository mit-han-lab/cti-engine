# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/scheduler/io.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, List

from cti.protocol import BaseBackendMsg, BaseTokenizerMsg, InferenceReply
from cti.utils import init_logger

from ._queue import ZmqPullQueue, ZmqPushQueue

if TYPE_CHECKING:
    from cti.protocol import ServingEndpoints

logger = init_logger(__name__)


class ZmqRequestTransport:
    """Control-plane ZMQ ingress and egress for online inference, polled by its loop."""

    def __init__(self, endpoints: ServingEndpoints) -> None:
        self._requests: ZmqPullQueue[BaseBackendMsg] = ZmqPullQueue(endpoints, "engine")
        # A rank that fails here exits, and the OS closes the socket above.
        self._replies: ZmqPushQueue[BaseTokenizerMsg] = ZmqPushQueue(endpoints, "detokenizer")

    def receive(self) -> List[BaseBackendMsg]:
        """The requests and aborts already arrived, in arrival order."""
        messages: List[BaseBackendMsg] = []
        while not self._requests.empty():
            messages.extend(self._requests.get())
        return messages

    def ready(self) -> bool:
        """Whether a request or abort has arrived."""
        return not self._requests.empty()

    def wait(self, fd: int) -> None:
        """Block until a request or ``fd`` is readable."""
        self._requests.wait(fd)

    def publish(self, replies: Sequence[InferenceReply]) -> None:
        logger.debug("Replying to tokenizer: %d messages", len(replies))
        self._replies.put(replies)
