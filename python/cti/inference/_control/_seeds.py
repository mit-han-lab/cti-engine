"""Seeds assigned on the primary rank's ingress, before any rank schedules a request."""

from __future__ import annotations

import secrets
from dataclasses import replace

from cti.protocol import BaseBackendMsg, UserMsg


class RequestSeeds:
    """Give every unseeded request an immutable seed; a request's own seed is kept.

    Speculative rounds sample a request's whole output, its first token included, from this seed.
    Peer ranks replay the stamped requests, so every rank sees the same seed.
    """

    def __init__(self) -> None:
        self._nonce = secrets.randbits(63)
        self._count = 0

    def stamp(self, messages: list[BaseBackendMsg]) -> list[BaseBackendMsg]:
        return [self._stamp(message) for message in messages]

    def _stamp(self, message: BaseBackendMsg) -> BaseBackendMsg:
        if not isinstance(message, UserMsg) or message.sampling.seed is not None:
            return message
        self._count += 1
        # Seeds are request-visible int63 values; counting keeps them distinct.
        seed = (self._nonce + self._count) % 2**63
        return replace(message, sampling=replace(message.sampling, seed=seed))


__all__ = ["RequestSeeds"]
