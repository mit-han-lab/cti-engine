"""The rank's request transport: ingress drained in arrival order, egress one link message per
publish."""

import pytest
import torch

import cti.protocol.link._transport as transport_module
from cti.protocol import (
    BaseBackendMsg,
    DetokenizeMsg,
    SamplingParams,
    ServingEndpoints,
    UserMsg,
)
from cti.protocol.link import ZmqRequestTransport


def _request(uid: int, tokens: list[int]) -> UserMsg:
    return UserMsg(
        uid=uid,
        input_ids=torch.tensor(tokens, dtype=torch.int32),
        sampling=SamplingParams(temperature=0, max_new_tokens=1).snapshot(),
    )


def test_online_transport_drains_ingress_and_batches_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = [_request(0, [1]), _request(1, [2])]

    class PullQueue:
        def __init__(self) -> None:
            self.pending = [requests[:1], requests[1:]]

        def get(self) -> list[BaseBackendMsg]:
            return self.pending.pop(0)

        def empty(self) -> bool:
            return not self.pending

    class PushQueue:
        def __init__(self) -> None:
            self.batches: list[list[object]] = []

        def put(self, batch: list[object]) -> None:
            self.batches.append(list(batch))

    push = PushQueue()
    monkeypatch.setattr(transport_module, "ZmqPullQueue", lambda *_args, **_kwargs: PullQueue())
    monkeypatch.setattr(transport_module, "ZmqPushQueue", lambda *_args, **_kwargs: push)
    transport = ZmqRequestTransport(ServingEndpoints.create())

    assert transport.receive() == requests
    assert transport.receive() == []

    first = DetokenizeMsg(uid=0, token_ids=(7,))
    second = DetokenizeMsg(uid=1, token_ids=(8,))
    transport.publish([])
    transport.publish([first])
    transport.publish([first, second])

    assert push.batches == [[], [first], [first, second]]
