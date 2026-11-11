"""Each link's queues code that link's message family: every message kind crosses a bound queue
pair unchanged, alone or batched with the others, as the bytes its family's encoder makes; a
link message is the list of its messages' encodings."""

import asyncio

import msgpack
import pytest
import torch
import zmq

from cti.protocol import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    DetokenizeMsg,
    PromptReply,
    RequestAbortedMsg,
    RequestEnd,
    RequestRejectedMsg,
    SamplingParams,
    ServingEndpoints,
    TextPrompt,
    TokenizeMsg,
    TokenUsage,
    UserMsg,
    UserReply,
)
from cti.protocol.link import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPullQueue,
    ZmqPushQueue,
)


def _user(uid: int) -> UserMsg:
    return UserMsg(
        uid=uid,
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        sampling=SamplingParams(temperature=0, max_new_tokens=4).snapshot(),
    )


_DETOKENIZE = DetokenizeMsg(uid=1, token_ids=(7,), end=RequestEnd(TokenUsage(3, 1), 4))
_REPLY = UserReply(uid=1, incremental_output="a", finished=False)
_MESSAGES = {
    "engine": (
        BaseBackendMsg,
        (_user(1), AbortBackendMsg(uid=1), _user(2)),
    ),
    "detokenizer": (
        BaseTokenizerMsg,
        (
            _DETOKENIZE,
            RequestRejectedMsg(uid=1, reason="full"),
            RequestAbortedMsg(uid=1),
        ),
    ),
    "tokenizer": (
        BaseTokenizerMsg,
        (
            TokenizeMsg(
                uid=1,
                prompt=TextPrompt("hi"),
                sampling=SamplingParams(temperature=0, max_new_tokens=1024).snapshot(),
            ),
            AbortMsg(uid=1),
        ),
    ),
    "frontend": (
        BaseFrontendMsg,
        (
            _REPLY,
            PromptReply(uid=1, token_ids=(5, 6), text="ab"),
            RequestRejectedMsg(uid=1, reason="full"),
        ),
    ),
}


def _wire(messages: object, family: type) -> bytes:
    return msgpack.packb(
        [family.family().encode(message) for message in messages], use_bin_type=True
    )


def _batches(messages: tuple) -> list[list]:
    """Each message alone, then all of them as one batch."""
    return [[message] for message in messages] + [list(messages)]


@pytest.mark.parametrize("target", list(_MESSAGES))
def test_every_message_crosses_its_links_bound_queues(target: str) -> None:
    family, messages = _MESSAGES[target]
    endpoints = ServingEndpoints.create(num_tokenizers=1)
    receiver, sender = ZmqPullQueue(endpoints, target), ZmqPushQueue(endpoints, target)
    try:
        for batch in _batches(messages):
            sender.put(batch)
            # A UserMsg holds a tensor, which has no plain equality: compare encodings.
            assert _wire(receiver.get(), family) == _wire(batch, family)
    finally:
        sender.stop()
        receiver.stop()


@pytest.mark.parametrize("target", list(_MESSAGES))
def test_every_message_crosses_its_links_async_queues(target: str) -> None:
    family, messages = _MESSAGES[target]
    endpoints = ServingEndpoints.create(num_tokenizers=1)

    async def cross() -> None:
        receiver = ZmqAsyncPullQueue(endpoints, target)
        sender = ZmqAsyncPushQueue(endpoints, target)
        try:
            for batch in _batches(messages):
                await sender.put(batch)
                assert _wire(await receiver.get(), family) == _wire(batch, family)
        finally:
            sender.stop()
            receiver.stop()

    asyncio.run(cross())


@pytest.mark.parametrize("target", list(_MESSAGES))
def test_a_bound_queue_sends_its_families_encoding(target: str) -> None:
    family, messages = _MESSAGES[target]
    endpoints = ServingEndpoints.create(num_tokenizers=1)
    context = zmq.Context()
    raw = context.socket(zmq.PULL)
    address, bind = endpoints.link(target, receiving=True)
    raw.bind(address) if bind else raw.connect(address)
    sender = ZmqPushQueue(endpoints, target)
    try:
        sender.put([])  # an empty put sends nothing
        for batch in _batches(messages):
            sender.put(batch)
            assert raw.recv() == _wire(batch, family)
        assert raw.poll(timeout=0) == 0
    finally:
        sender.stop()
        raw.close(linger=0)
        context.term()
