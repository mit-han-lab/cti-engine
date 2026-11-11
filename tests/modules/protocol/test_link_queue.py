import logging
from unittest.mock import Mock, call

import pytest
import zmq
import zmq.asyncio

from cti.protocol import ServingEndpoints
from cti.protocol.link import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPullQueue,
    ZmqPushQueue,
)

_ADDR = "ipc://queue"
_QUEUE_TYPES = (ZmqPushQueue, ZmqAsyncPushQueue, ZmqPullQueue, ZmqAsyncPullQueue)
_QUEUES = (
    pytest.param(ZmqPushQueue, False, zmq.PUSH, id="push"),
    pytest.param(ZmqAsyncPushQueue, True, zmq.PUSH, id="async-push"),
    pytest.param(ZmqPullQueue, False, zmq.PULL, id="pull"),
    pytest.param(ZmqAsyncPullQueue, True, zmq.PULL, id="async-pull"),
)
_STOP_CALLS = [call.socket.close(linger=0), call.context.term()]
# Every address is _ADDR, and every link's receiver binds it.
_ENDPOINTS = ServingEndpoints(_ADDR, _ADDR, _ADDR, _ADDR, num_tokenizers=1)


def _binds(queue_type) -> bool:
    """Whether ``queue_type`` binds its link's address: a receiver does, a sender connects."""
    return queue_type not in (ZmqPushQueue, ZmqAsyncPushQueue)


def _stoppable(queue_type) -> tuple:
    queue = object.__new__(queue_type)
    zmq_calls = Mock()
    queue.socket, queue.context = zmq_calls.socket, zmq_calls.context
    return queue, zmq_calls


def _patch_contexts(monkeypatch: pytest.MonkeyPatch) -> Mock:
    zmq_calls = Mock()
    monkeypatch.setattr(zmq, "Context", zmq_calls.sync)
    monkeypatch.setattr(zmq.asyncio, "Context", zmq_calls.asyncio)
    return zmq_calls


def _setup_calls(is_async: bool, socket_type: int, create: bool) -> list:
    context = call.asyncio() if is_async else call.sync()
    opened = context.socket().bind if create else context.socket().connect
    return [context, context.socket(socket_type), opened(_ADDR)]


def _assert_raised_as_is(raised: BaseException, error: BaseException) -> None:
    assert raised is error
    assert raised.__cause__ is None and raised.__context__ is None
    assert not hasattr(raised, "__notes__")


@pytest.mark.parametrize("queue_type", _QUEUE_TYPES)
def test_queue_stop_disables_zero_mq_linger(queue_type) -> None:
    queue, zmq_calls = _stoppable(queue_type)

    queue.stop()

    assert zmq_calls.mock_calls == _STOP_CALLS


@pytest.mark.parametrize("failing", (0, 1), ids=("close", "term"))
@pytest.mark.parametrize("queue_type", _QUEUE_TYPES)
def test_queue_stop_raises_the_cleanup_error_as_is(
    caplog: pytest.LogCaptureFixture, queue_type, failing: int
) -> None:
    queue, zmq_calls = _stoppable(queue_type)
    error = zmq.ZMQError(zmq.ENOTSOCK, "cleanup failed")
    (zmq_calls.socket.close, zmq_calls.context.term)[failing].side_effect = error

    caplog.set_level(logging.DEBUG)
    with pytest.raises(zmq.ZMQError, match="^cleanup failed$") as raised:
        queue.stop()

    _assert_raised_as_is(raised.value, error)
    assert zmq_calls.mock_calls == _STOP_CALLS[: failing + 1]
    assert not caplog.records


@pytest.mark.parametrize(("queue_type", "is_async", "socket_type"), _QUEUES)
def test_queue_opens_its_socket_type_on_its_own_context(
    monkeypatch: pytest.MonkeyPatch,
    queue_type,
    is_async: bool,
    socket_type: int,
) -> None:
    zmq_calls = _patch_contexts(monkeypatch)
    create = _binds(queue_type)

    queue = queue_type(_ENDPOINTS, "engine")

    assert zmq_calls.mock_calls == _setup_calls(is_async, socket_type, create)
    context = (zmq_calls.asyncio if is_async else zmq_calls.sync).return_value
    assert queue.context is context
    assert queue.socket is context.socket.return_value


@pytest.mark.parametrize("failing", (0, 1, 2), ids=("context", "socket", "bind-or-connect"))
@pytest.mark.parametrize(("queue_type", "is_async", "socket_type"), _QUEUES)
def test_queue_setup_error_propagates_without_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    queue_type,
    is_async: bool,
    socket_type: int,
    failing: int,
) -> None:
    zmq_calls = _patch_contexts(monkeypatch)
    create = _binds(queue_type)
    context_type = zmq_calls.asyncio if is_async else zmq_calls.sync
    socket = context_type.return_value.socket
    steps = (
        context_type,
        socket,
        socket.return_value.bind if create else socket.return_value.connect,
    )
    error = zmq.ZMQError(zmq.EADDRINUSE, "setup failed")
    steps[failing].side_effect = error

    caplog.set_level(logging.DEBUG)
    with pytest.raises(zmq.ZMQError, match="^setup failed$") as raised:
        queue_type(_ENDPOINTS, "engine")

    _assert_raised_as_is(raised.value, error)
    assert zmq_calls.mock_calls == _setup_calls(is_async, socket_type, create)[: failing + 1]
    assert not caplog.records
