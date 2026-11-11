import asyncio
import contextlib
import logging
import signal
import socket
import threading
from collections.abc import Collection
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from fastapi import HTTPException

from cti.config import BatchLayout, EngineOptions, ModelOptions, ServerOptions
from cti.multimodal import ProcessedPrompt
from cti.protocol import (
    AbortMsg,
    AdmissionPriority,
    PromptReply,
    RejectionKind,
    RequestRejectedError,
    RequestRejectedMsg,
    SampledSequence,
    SamplingParams,
    TextPrompt,
    TokenizeMsg,
    TokenPrompt,
    TokenUsage,
    UserReply,
)
from cti.runtime import EngineProcessError
from cti.sampling import TokenLogprobs
from cti.server.api_server import ApiServer
from cti.server.frontend import FrontendManager, InferenceUnavailableError
from cti.training import TrainingBatch
from tests.support.frontend_link import FakeEngine, TokenizerLink, serving_api


def _api(
    model_id: str = "model",
    port: int | None = None,
    *,
    engine: object = FakeEngine,
    link: object = FrontendManager.linked,
) -> ApiServer:
    options = ServerOptions(port=0 if port is None else port)
    return ApiServer(
        EngineOptions(model=ModelOptions(path=model_id)), options, engine=engine, link=link
    )


def _serving(recv=None) -> tuple[ApiServer, FrontendManager]:
    """An API server, its engine started, and its frontend, which receives from ``recv``
    and sends to a Mock; it serves once its app's lifespan runs."""

    frontend = FrontendManager(send_tokenizers=(Mock(),), recv_tokenizer=recv or BlockingQueue())
    api = _api(link=lambda _endpoints: frontend)
    api.start_engine()
    return api, frontend


async def _settled(frontend: FrontendManager, kept: Collection[int] = ()) -> None:
    """Wait until the frontend's background aborts have run, holding only ``kept``."""

    for _ in range(1000):
        if frontend.pending <= set(kept):
            return
        await asyncio.sleep(0)
    raise AssertionError(f"the frontend still holds {sorted(frontend.pending)}")


def _route(frontend: FrontendManager) -> tuple[asyncio.Queue, int]:
    """A new request of a one-request call: the call's inbox and the request's uid."""

    inbox: asyncio.Queue = asyncio.Queue()
    return inbox, frontend.new_user(inbox, 0)


def _drain(inbox: asyncio.Queue) -> list:
    return [inbox.get_nowait() for _ in range(inbox.qsize())]


class BlockingQueue:
    async def get(self):
        await asyncio.Event().wait()

    def stop(self):
        pass


class _Feed(asyncio.Queue):
    def stop(self):
        pass


class FailingQueue(BlockingQueue):
    async def get(self):
        raise RuntimeError("frontend transport failed")


class _StopRequiredBlockingQueue:
    """Model an async transport whose pending receive wakes only when closed."""

    def __init__(self) -> None:
        self._stopped = asyncio.Event()

    async def get(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self._stopped.wait()
            raise

    def stop(self) -> None:
        self._stopped.set()


def test_server_exposes_only_the_documented_generation_routes(monkeypatch) -> None:
    paths = {route.path for route in _api().app.routes}

    assert "/generate" not in paths
    assert {"/v1/models", "/v1/chat/completions", "/v1/completions"} <= paths


def test_lifespan_closes_transport_before_waiting_for_listener(monkeypatch) -> None:
    async def run_test() -> None:
        recv_queue = _StopRequiredBlockingQueue()
        api, _ = _serving(recv_queue)

        async with api.app.router.lifespan_context(api.app):
            await asyncio.sleep(0)

        assert recv_queue._stopped.is_set()

    asyncio.run(asyncio.wait_for(run_test(), timeout=1))


def test_lifespan_stops_its_listener_and_wakes_pending_requests(monkeypatch) -> None:
    async def run_test() -> None:
        api, frontend = _serving()

        async with api.app.router.lifespan_context(api.app):
            inbox, _ = _route(frontend)
            await asyncio.sleep(0)

        assert asyncio.all_tasks() == {asyncio.current_task()}
        [(index, error)] = _drain(inbox)
        assert index == 0 and "closed" in str(error)
        assert api.engine.failure is None
        with pytest.raises(InferenceUnavailableError, match="closed"):
            frontend.new_user(inbox, 0)

    asyncio.run(run_test())


def test_replies_reach_their_call_in_arrival_order_with_their_index() -> None:
    async def run_test() -> None:
        feed = _Feed()
        manager = FrontendManager(send_tokenizers=(Mock(),), recv_tokenizer=feed)
        call: asyncio.Queue = asyncio.Queue()
        uids = [manager.new_user(call, index) for index in range(2)]
        other, other_uid = _route(manager)
        dropped, dropped_uid = _route(manager)
        manager.release([dropped_uid], running=())
        arrivals = [
            UserReply(uids[1], "b", False, token_ids=(1,)),
            RequestRejectedMsg(other_uid, "ingress hold is full"),
            UserReply(dropped_uid, "x", False, token_ids=(1,)),
            UserReply(uids[0], "a", False, token_ids=(1,)),
        ]
        feed.put_nowait(arrivals)

        listening = asyncio.create_task(manager.listen())
        await asyncio.sleep(0)
        listening.cancel()

        assert _drain(call) == [(1, arrivals[0]), (0, arrivals[3])]
        assert _drain(other) == [(0, arrivals[1])]
        assert _drain(dropped) == []
        await manager.close()

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("recv", "exit_reason", "reason"),
    [
        (FailingQueue(), None, "frontend listener failed: frontend transport failed"),
        (None, "engine exited unexpectedly", "engine exited unexpectedly"),
    ],
    ids=["listener", "engine"],
)
def test_a_failure_wakes_pending_requests_rejects_new_ones_and_stops_the_server(
    monkeypatch, recv, exit_reason: str | None, reason: str
) -> None:
    async def run_test() -> None:
        api, frontend = _serving(recv)

        async with api.app.router.lifespan_context(api.app):
            inbox, _ = _route(frontend)
            await asyncio.sleep(0)
            if exit_reason is not None:
                api.engine.fail(exit_reason)
            index, error = await asyncio.wait_for(inbox.get(), timeout=0.1)

            assert index == 0 and str(error) == reason
            assert str(api.engine.failure) == reason and api.http_server.should_exit
            with pytest.raises(InferenceUnavailableError, match=reason):
                frontend.new_user(inbox, 0)

    asyncio.run(run_test())


def _requests(*uids: int) -> list[TokenizeMsg]:
    """One call's requests, one per uid."""

    spec = SamplingParams(max_new_tokens=1).snapshot()
    return [TokenizeMsg(uid, TextPrompt("x"), spec) for uid in uids]


def test_release_aborts_running_requests_in_the_background(caplog) -> None:
    async def run_test() -> None:
        def put(msgs: list) -> None:
            if msgs == [AbortMsg(uid=failing)]:
                raise RuntimeError("link broke")

        send_queue = Mock(put=AsyncMock(side_effect=put))
        manager = FrontendManager(send_tokenizers=(send_queue,), recv_tokenizer=BlockingQueue())
        _, ended = _route(manager)
        _, running = _route(manager)
        _, failing = _route(manager)
        for uid in (ended, running, failing):
            await manager.send(_requests(uid))

        manager.release([ended, running], running={running})
        manager.release([failing], running={failing})
        with caplog.at_level(logging.ERROR):
            await _settled(manager)
            await asyncio.sleep(0)

        assert send_queue.put.call_args_list[3].args == ([AbortMsg(uid=running)],)
        assert f"Frontend task abort-user-{failing} failed" in caplog.text
        assert not manager.pending
        await manager.close()

    asyncio.run(run_test())


def test_a_calls_requests_and_their_aborts_go_to_one_tokenizer_in_order() -> None:
    """With several tokenizers, an abort follows its request through the same tokenizer, so
    it can never reach the engine first and leave the request running."""

    async def run_test() -> None:
        sent: list[tuple[int, list]] = []
        queues = tuple(
            Mock(put=AsyncMock(side_effect=lambda msgs, i=i: sent.append((i, msgs))))
            for i in range(2)
        )
        manager = FrontendManager(send_tokenizers=queues, recv_tokenizer=BlockingQueue())
        first = [_route(manager)[1] for _ in range(2)]
        second = [_route(manager)[1]]
        await manager.send(_requests(*first))
        await manager.send(_requests(*second))

        manager.release([first[1], *second], running={first[1], *second})
        await _settled(manager, kept={first[0]})

        assert [(i, [m.uid for m in msgs]) for i, msgs in sent] == [
            (0, first),
            (1, second),
            (0, [first[1]]),
            (1, second),
        ]
        assert [type(m) for _, msgs in sent[2:] for m in msgs] == [AbortMsg, AbortMsg]
        await manager.close()

    asyncio.run(run_test())


def test_release_drops_routes_even_when_the_backend_has_failed() -> None:
    async def run_test() -> None:
        manager = FrontendManager(send_tokenizers=(Mock(),), recv_tokenizer=BlockingQueue())
        uids = [_route(manager)[1] for _ in range(3)]
        manager.end("engine died")

        manager.release(uids, running=set(uids))

        assert not manager.pending  # no abort: nothing can reach a failed backend
        await manager.close()

    asyncio.run(run_test())


def test_api_server_raises_when_its_engine_fails_to_start():
    api = _api(engine=Mock(side_effect=RuntimeError("no engine")))
    with pytest.raises(RuntimeError, match="no engine"):
        api.run()
    api.close()  # nothing to close

    with pytest.raises(RuntimeError, match="not running"):
        asyncio.run(api.v1_root())


@pytest.mark.parametrize("endpoint", ["v1_root", "available_models"])
def test_health_endpoints_reflect_engine_failure(endpoint):
    api, _ = _serving()
    endpoint = getattr(api, endpoint)

    async def run_test() -> HTTPException:
        async with api.app.router.lifespan_context(api.app):
            await endpoint()
            api.engine.fail("engine died")
            with pytest.raises(HTTPException) as raised:
                await endpoint()
            return raised.value

    raised = asyncio.run(run_test())
    assert raised.status_code == 503
    assert "engine died" in raised.detail


@pytest.mark.parametrize("failed", [False, True])
def test_http_server_exits_and_reports_engine_failure(failed):
    api, _ = _serving()

    class Server:
        should_exit = False

        def run(self, sockets):
            async def serve():
                async with api.app.router.lifespan_context(api.app):
                    await asyncio.sleep(0)
                    if failed:
                        api.engine.fail("engine exited unexpectedly")
                        await asyncio.sleep(0)
                        assert self.should_exit

            asyncio.run(serve())

    api.http_server = server = Server()
    if failed:
        with pytest.raises(RuntimeError, match="engine exited unexpectedly"):
            api.run()
        assert server.should_exit
    else:
        api.run()
        assert not server.should_exit
    with pytest.raises(RuntimeError, match="not running"):
        asyncio.run(api.v1_root())


def test_a_second_server_on_a_listening_port_fails_to_bind_at_once(monkeypatch):
    """The first server listens as it binds, so a second one starting on its port fails
    before loading a model, its address reuse notwithstanding."""

    with _api()._bind() as listener:
        port = listener.getsockname()[1]
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
        with pytest.raises(SystemExit, match=f"cannot serve on 127.0.0.1:{port}: Address"):
            _api(port=port)._bind()


def test_connections_on_the_bound_socket_send_without_nagle_delay(monkeypatch):
    """asyncio sets TCP_NODELAY only on IPPROTO_TCP sockets; without it a stream's first
    chunk waits out the client's delayed ACK (40 ms on Linux)."""

    async def serve(listener: socket.socket) -> int:
        accepted: asyncio.Future[int] = asyncio.get_running_loop().create_future()

        def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            sock = writer.get_extra_info("socket")
            accepted.set_result(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))
            writer.close()

        server = await asyncio.start_server(on_connect, sock=listener)
        async with server:
            _, writer = await asyncio.open_connection(*listener.getsockname()[:2])
            nodelay = await asyncio.wait_for(accepted, 5)
            writer.close()
            return nodelay

    assert asyncio.run(serve(_api()._bind()))


def test_models_lists_the_model_path_as_configured():
    frontend = FrontendManager(send_tokenizers=(Mock(),), recv_tokenizer=BlockingQueue())
    api = _api(model_id="~/models/qwen", link=lambda _endpoints: frontend)
    api.start_engine()

    async def models() -> dict:
        async with api.app.router.lifespan_context(api.app):
            return await api.available_models()

    models = asyncio.run(models())

    assert [(card["id"], card["root"]) for card in models["data"]] == [("~/models/qwen",) * 2]


# Processed log-probabilities, a seed and no output limit: a GRPO rollout's policy.
_ROLLOUT = SamplingParams(
    max_new_tokens=None,
    temperature=0.7,
    seed=5,
    logprobs=True,
    logprobs_processed=True,
    top_logprobs=1,
    stop=["\n"],
)
_USAGE = TokenUsage(prompt_tokens=3, completion_tokens=2, cached_prompt_tokens=1)


def _logprobs(token_id: int, logprob: float) -> TokenLogprobs:
    return TokenLogprobs(token_id, logprob, ((token_id, logprob),))


# A request's replies: its prompt, then two tokens, the last completing a stop string at
# model revision 4.
_REPLIES = (
    PromptReply(0, (1, 2, 3), "Hi"),
    UserReply(0, "He", False, token_ids=(10,), logprobs=(_logprobs(10, -0.5),)),
    UserReply(
        0,
        "llo\nmore",
        True,
        _USAGE,
        "stop",
        token_ids=(11,),
        stop_str="\n",
        logprobs=(_logprobs(11, -0.25),),
        max_new_tokens=9,
        model_revision=4,
    ),
)


@contextlib.contextmanager
def _sampling(script=lambda _request: _REPLIES):
    """An API server serving on its own loop thread, whose engine requests get the replies
    ``script`` gives each; it yields the server, the requests sent and the uids aborted,
    which hold once the server has stopped."""

    def respond(link: TokenizerLink, requests) -> None:
        for request in requests:
            link.deliver(*(replace(reply, uid=request.uid) for reply in script(request)))

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    link = TokenizerLink(respond)
    api = serving_api(link)
    serving = api.app.router.lifespan_context(api.app)
    sent: list = []
    aborted: list[int] = []
    asyncio.run_coroutine_threadsafe(serving.__aenter__(), loop).result()
    try:
        yield api, sent, aborted
    finally:
        asyncio.run_coroutine_threadsafe(link.settled(), loop).result()
        asyncio.run_coroutine_threadsafe(serving.__aexit__(None, None, None), loop).result()
        sent.extend(link.requests)
        aborted.extend(link.aborted)
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


def test_sampling_gathers_the_samples_the_offline_sampler_does() -> None:
    def script(request):
        # The second request's prompt reply may come after its tokens.
        return _REPLIES if request.uid % 2 == 0 else (*_REPLIES[1:], _REPLIES[0])

    with _sampling(script) as (api, sent, aborted):
        result = api.sample(["Hi", [1, 2, 3]], [_ROLLOUT] * 2, AdmissionPriority.LOW)

    # One message per call, each request at the call's priority, its prompt returned.
    assert [(request.prompt, request.priority, request.return_prompt) for request in sent] == [
        (TextPrompt("Hi"), AdmissionPriority.LOW, True),
        (TokenPrompt((1, 2, 3)), AdmissionPriority.LOW, True),
    ]
    assert aborted == []
    # The sample the offline sampler gathers from the same engine replies, field by field.
    expected = SampledSequence(
        request_id=0,
        prompt_token_ids=(1, 2, 3),
        sampled_token_ids=(10, 11),
        sampling_logprobs=(-0.5, -0.25),
        cached_prompt_tokens=1,
        # The output limit resolved at admission.
        sampling=replace(_ROLLOUT.snapshot(), max_new_tokens=9),
        finish_reason="stop",
        stop_str="\n",
        sampling_top_logprobs=(((10, -0.5),), ((11, -0.25),)),
    )
    assert result.samples == (expected, replace(expected, request_id=1))
    assert result.texts == ("Hello", "Hello") and result.model_revision == 4
    # A grouped rollout trains.
    batch = TrainingBatch.grpo(result, [0.0, 1.0], BatchLayout.PACKED, group_size=2)
    assert batch.batch_size == 2


def test_requests_ended_at_different_revisions_raise() -> None:
    def script(request):
        # An optimizer step ran between the two requests.
        return (
            _REPLIES
            if request.uid % 2 == 0
            else (*_REPLIES[:2], replace(_REPLIES[2], model_revision=5))
        )

    with _sampling(script) as (api, _, _):
        with pytest.raises(RuntimeError, match="the model changed while sampling; resample"):
            api.sample(["Hi", "Hi"], [_ROLLOUT] * 2, AdmissionPriority.NORMAL)
    # The revision is the engine's stamp: the fake engine has no model_revision to read.


def test_a_rejection_aborts_the_other_requests_and_raises() -> None:
    def script(request):
        if request.prompt == TextPrompt("bad"):
            return (RequestRejectedMsg(0, "ingress hold is full"),)
        return _REPLIES[:2]  # still running

    with _sampling(script) as (api, sent, aborted):
        with pytest.raises(RequestRejectedError, match="ingress hold is full") as rejected:
            api.sample(["Hi", "bad", "Hi"], [_ROLLOUT] * 3, AdmissionPriority.NORMAL)

    assert rejected.value.request_id == 1
    assert rejected.value.kind is RejectionKind.UNAVAILABLE
    assert sorted(aborted) == [sent[0].uid, sent[2].uid]


def test_an_interrupted_caller_aborts_its_requests() -> None:
    """A caller interrupted while it waits (here by KeyboardInterrupt) cancels the call on
    the server's loop, which aborts every request still running."""

    with _sampling(lambda _request: _REPLIES[:2]) as (api, sent, aborted):  # never ends
        main = threading.main_thread().ident
        threading.Timer(0.1, signal.pthread_kill, (main, signal.SIGINT)).start()
        with pytest.raises(KeyboardInterrupt):
            api.sample(["Hi", "Hi"], [_ROLLOUT] * 2, AdmissionPriority.NORMAL)

    assert sorted(aborted) == sorted(request.uid for request in sent)
    assert len(sent) == 2


def test_an_engine_failure_wakes_the_sampling_caller() -> None:
    with _sampling(lambda _request: ()) as (api, _, _):
        # The engine client's reader thread reports the failure while the caller waits;
        # the caller gets the engine's own error, as offline sampling raises it.
        threading.Timer(0.05, api.engine.fail, ("engine exited unexpectedly",)).start()
        with pytest.raises(EngineProcessError, match="engine exited unexpectedly"):
            api.sample(["Hi"], [_ROLLOUT], AdmissionPriority.NORMAL)


@pytest.mark.parametrize(
    ("prompt", "error"),
    [
        ([{"role": "user", "content": "Hi"}], TypeError),
        (ProcessedPrompt(torch.tensor([1, 2], dtype=torch.int32), None, None), TypeError),
        ([True], TypeError),
    ],
    ids=["chat", "image", "bool"],
)
def test_sampling_takes_text_and_token_prompts_only(prompt, error) -> None:
    with _sampling() as (api, sent, _):
        with pytest.raises(error, match="text or token-id prompts"):
            api.sample([prompt], [_ROLLOUT], AdmissionPriority.NORMAL)

    assert sent == []


def test_sampling_checks_its_params_as_offline_sampling_does() -> None:
    with _sampling() as (api, sent, _):
        with pytest.raises(ValueError, match="same length"):
            api.sample(["Hi", "Hi"], [_ROLLOUT], AdmissionPriority.NORMAL)
        with pytest.raises(ValueError, match="priority"):
            api.sample(["Hi"], [_ROLLOUT], "urgent")

    assert sent == []
