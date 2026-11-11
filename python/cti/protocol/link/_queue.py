# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/utils/mp.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import msgpack
import zmq
import zmq.asyncio

import cti.protocol as protocol

if TYPE_CHECKING:
    from cti.protocol import ServingEndpoints
    from cti.protocol._codec import Family

T = TypeVar("T")
Target = Literal["engine", "frontend", "detokenizer", "tokenizer"]
# The message family each link carries; its base class makes the family's codec.
_FAMILIES = MappingProxyType(
    {
        "engine": "BaseBackendMsg",
        "frontend": "BaseFrontendMsg",
        "detokenizer": "BaseTokenizerMsg",
        "tokenizer": "BaseTokenizerMsg",
    }
)


class _ZmqQueue:
    """One socket of ``socket_type`` on its own ``context_type``, on ``target``'s side of its
    link (tokenizer process ``index``'s for ``tokenizer``), coding ``target``'s message
    family. A link message is a batch: a put sends its messages together as one, an empty
    put sends nothing, and a get returns them as sent."""

    def __init__(
        self,
        endpoints: ServingEndpoints,
        target: Target,
        context_type: type[zmq.Context],
        socket_type: int,
        *,
        receiving: bool,
        index: int = 0,
    ):
        address, create = endpoints.link(target, receiving=receiving, index=index)
        self.family: Family = getattr(protocol, _FAMILIES[target]).family()
        self.context = context_type()
        self.socket = self.context.socket(socket_type)
        self.socket.bind(address) if create else self.socket.connect(address)

    def _encode(self, objs: Sequence[Any]) -> bytes:
        return msgpack.packb([self.family.encode(obj) for obj in objs], use_bin_type=True)

    def _decode(self, event: bytes) -> list[Any]:
        batch = msgpack.unpackb(event, raw=False, strict_map_key=False)
        return [self.family.decode(obj) for obj in batch]

    def stop(self):
        self.socket.close(linger=0)
        self.context.term()


class ZmqPushQueue(_ZmqQueue, Generic[T]):
    def __init__(self, endpoints: ServingEndpoints, target: Target, *, index: int = 0):
        super().__init__(endpoints, target, zmq.Context, zmq.PUSH, receiving=False, index=index)

    def put(self, objs: Sequence[T]) -> None:
        if objs:
            self.socket.send(self._encode(objs), copy=False)


class ZmqAsyncPushQueue(_ZmqQueue, Generic[T]):
    def __init__(self, endpoints: ServingEndpoints, target: Target, *, index: int = 0):
        super().__init__(
            endpoints, target, zmq.asyncio.Context, zmq.PUSH, receiving=False, index=index
        )

    async def put(self, objs: Sequence[T]) -> None:
        if objs:
            await self.socket.send(self._encode(objs), copy=False)


class ZmqPullQueue(_ZmqQueue, Generic[T]):
    def __init__(self, endpoints: ServingEndpoints, target: Target, *, index: int = 0):
        super().__init__(endpoints, target, zmq.Context, zmq.PULL, receiving=True, index=index)

    def get(self) -> list[T]:
        return self._decode(self.socket.recv())

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def wait(self, fd: int) -> None:
        """Block until a message or ``fd`` is readable."""
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        poller.register(fd, zmq.POLLIN)
        poller.poll()


class ZmqAsyncPullQueue(_ZmqQueue, Generic[T]):
    def __init__(self, endpoints: ServingEndpoints, target: Target, *, index: int = 0):
        super().__init__(
            endpoints, target, zmq.asyncio.Context, zmq.PULL, receiving=True, index=index
        )

    async def get(self) -> list[T]:
        return self._decode(await self.socket.recv())
