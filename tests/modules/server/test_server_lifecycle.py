"""Lifecycle contracts for the online server: its engine, its signals, its tokenizer workers."""

from __future__ import annotations

import signal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import cti.server.launch as launch_module
import cti.tokenizer.detokenize as detokenize_module
import cti.tokenizer.server as tokenizer_server_module
import cti.tokenizer.text as text_module
from cti.config import EngineOptions, ModelOptions, ServerOptions
from cti.multimodal import ImageLimits
from cti.protocol import (
    DetokenizeMsg,
    PromptReply,
    RejectionKind,
    RequestAbortedMsg,
    RequestEnd,
    RequestRejectedMsg,
    ServingEndpoints,
    TokenUsage,
    UserReply,
)
from cti.sampling import TokenLogprobs
from cti.server.api_server import ApiServer
from cti.server.args import parse_args
from cti.tokenizer.text import TextStream
from tests.support.cpu_engine import CpuConfig

# Shared tokenization: the detokenizer listens on its one address.
_ENDPOINTS = ServingEndpoints("backend", "frontend", "listener", "tokenizer", num_tokenizers=1)
_OPTIONS = EngineOptions(model=ModelOptions(path="model"))


class _ReadyConnection:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def send(self, message: str) -> None:
        _ = message
        self._events.append("backend.ready")

    def close(self) -> None:
        self._events.append("ready.close")


def _serve_tokenizer(**kwargs: object) -> None:
    """Run a tokenizer worker of a text-only checkpoint until its queue interrupts it."""

    tokenizer_server_module.tokenize_worker(
        model_path="model",
        hf_config=CpuConfig(),
        image_limits=ImageLimits(1, 64),
        endpoints=_ENDPOINTS,
        **kwargs,
    )


def test_server_cli_does_not_expose_kv_page_count(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        parse_args(["--model.path", "unused", "--num-pages", "8"])

    assert caught.value.code == 2
    assert "unrecognized arguments: --num-pages 8" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ("--shell", "--shell-mode"))
def test_server_cli_does_not_expose_an_embedded_shell(
    flag: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        parse_args(["--model.path", "unused", flag])

    assert caught.value.code == 2
    assert f"unrecognized arguments: {flag}" in capsys.readouterr().err


def _api(engine: Mock) -> tuple[ApiServer, Mock]:
    """An API server whose run starts ``engine`` and serves until stopped at once."""

    client = Mock(return_value=engine)
    api = ApiServer(_OPTIONS, ServerOptions(port=0, num_tokenizers=2), engine=client)
    api.http_server = Mock()
    return api, client


def test_api_server_owns_one_engine_and_closes_it_once() -> None:
    engine = Mock(failure=None)
    api, client = _api(engine)

    api.run()
    assert api.engine is engine
    api.close()
    api.close()

    (args, kwargs) = client.call_args
    assert args == (_OPTIONS, None) and kwargs["num_tokenizers"] == 2
    assert callable(kwargs["on_failure"])  # the engine's failure stops the server
    client.assert_called_once()
    engine.check.assert_called_once_with()
    engine.close.assert_called_once_with()
    engine.close_at_exit.return_value.cancel.assert_called_once_with()
    with pytest.raises(RuntimeError, match="not running"):
        _ = api.engine


def test_api_server_reclaims_an_engine_whose_close_failed() -> None:
    engine = Mock()
    engine.close.side_effect = RuntimeError("training session is busy")
    api, _ = _api(engine)
    api.run()

    with pytest.raises(RuntimeError, match="training session is busy"):
        api.close()

    # The finalizer reaps the engine the failed close left running, once.
    engine.close_at_exit.return_value.assert_called_once_with()
    api.close()
    engine.close.assert_called_once_with()


def test_server_termination_is_graceful_once_then_forceful() -> None:
    server = Mock(is_closing=False)
    termination = launch_module._ServerTermination(server)

    with pytest.raises(SystemExit) as caught:
        termination._handle(signal.SIGTERM, None)
    server.kill_workers.assert_not_called()
    termination._handle(signal.SIGTERM, None)

    assert caught.value.code is None
    server.kill_workers.assert_called_once_with()


def test_first_termination_during_engine_cleanup_forces_then_exits() -> None:
    server = Mock(is_closing=True)
    termination = launch_module._ServerTermination(server)

    with pytest.raises(SystemExit) as caught:
        with termination:
            termination._handle(signal.SIGTERM, None)

    assert caught.value.code is None
    server.kill_workers.assert_called_once_with()


@pytest.mark.parametrize(
    ("tokenizer_id", "stops"),
    [
        (None, ["stop:frontend", "stop:listener"]),
        (0, ["stop:frontend", "stop:backend", "stop:tokenizer-0"]),
    ],
    ids=["detokenizer", "tokenizer"],
)
def test_tokenizer_worker_closes_readiness_and_zmq_resources(
    monkeypatch: pytest.MonkeyPatch, tokenizer_id: int | None, stops: list[str]
) -> None:
    events: list[str] = []

    class _PushQueue:
        def __init__(self, endpoints: ServingEndpoints, target: str, index: int = 0) -> None:
            self._addr = endpoints.link(target, receiving=False, index=index)[0]

        def stop(self) -> None:
            events.append(f"stop:{self._addr}")

    class _PullQueue(_PushQueue):
        socket = SimpleNamespace(poll=lambda timeout=None: True)

        def get(self) -> None:
            raise KeyboardInterrupt

    ready = _ReadyConnection(events)
    monkeypatch.setattr(tokenizer_server_module, "ZmqPushQueue", _PushQueue)
    monkeypatch.setattr(tokenizer_server_module, "ZmqPullQueue", _PullQueue)
    monkeypatch.setattr(text_module, "load_tokenizer", lambda _path: Mock(added_tokens_decoder={}))

    _serve_tokenizer(ready_connection=ready, tokenizer_id=tokenizer_id)

    assert events == ["backend.ready", "ready.close", *stops]


def test_tokenizer_worker_forwards_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend_messages: list[object] = []

    class _PushQueue:
        def __init__(self, endpoints: ServingEndpoints, target: str, index: int = 0) -> None:
            self._addr = endpoints.link(target, receiving=False, index=index)[0]

        def put(self, messages: list[object]) -> None:
            if self._addr == "frontend" and messages:
                frontend_messages.append(list(messages))

        def stop(self) -> None:
            pass

    class _PullQueue(_PushQueue):
        def __init__(self, endpoints: ServingEndpoints, target: str, index: int = 0) -> None:
            super().__init__(endpoints, target, index)
            self._pending = [
                [
                    RequestRejectedMsg(
                        uid=7, reason="context too long", kind=RejectionKind.CONTEXT_LENGTH_EXCEEDED
                    )
                ]
            ]

        def get(self) -> object:
            if self._pending:
                return self._pending.pop()
            raise KeyboardInterrupt

    monkeypatch.setattr(tokenizer_server_module, "ZmqPushQueue", _PushQueue)
    monkeypatch.setattr(tokenizer_server_module, "ZmqPullQueue", _PullQueue)
    monkeypatch.setattr(text_module, "load_tokenizer", lambda _path: Mock(added_tokens_decoder={}))

    _serve_tokenizer()

    assert frontend_messages == [
        [
            RequestRejectedMsg(
                uid=7, reason="context too long", kind=RejectionKind.CONTEXT_LENGTH_EXCEEDED
            )
        ],
    ]


def test_tokenizer_worker_forwards_terminal_finish_reason_and_output_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend_messages: list[object] = []

    class _Tokenizer:
        added_tokens_decoder: dict[int, object] = {}

        def batch_decode(self, token_ids: list[list[int]], **_options: bool) -> list[str]:
            return ["" for _ in token_ids]

    class _PushQueue:
        def __init__(self, endpoints: ServingEndpoints, target: str, index: int = 0) -> None:
            self._addr = endpoints.link(target, receiving=False, index=index)[0]

        def put(self, messages: list[object]) -> None:
            if self._addr == "frontend" and messages:
                frontend_messages.append(list(messages))

        def stop(self) -> None:
            pass

    class _PullQueue(_PushQueue):
        def __init__(self, endpoints: ServingEndpoints, target: str, index: int = 0) -> None:
            super().__init__(endpoints, target, index)
            usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
            self._pending = [
                [
                    # One step (a speculative round) commits two tokens of request 5.
                    DetokenizeMsg(
                        uid=5,
                        token_ids=(3, 4),
                        logprobs=(TokenLogprobs(3, -0.5), TokenLogprobs(4, -0.25)),
                    ),
                    DetokenizeMsg(uid=6, token_ids=(3,), end=RequestEnd(usage, 9)),
                    DetokenizeMsg(
                        uid=7, token_ids=(2,), end=RequestEnd(usage, 9, is_stop_token=True)
                    ),
                    DetokenizeMsg(uid=8, token_ids=(4,), end=RequestEnd(usage, 9, stop_str="\n")),
                ]
            ]

        def get(self) -> object:
            if self._pending:
                return self._pending.pop()
            raise KeyboardInterrupt

    monkeypatch.setattr(tokenizer_server_module, "ZmqPushQueue", _PushQueue)
    monkeypatch.setattr(tokenizer_server_module, "ZmqPullQueue", _PullQueue)
    monkeypatch.setattr(text_module, "load_tokenizer", lambda _path: _Tokenizer())

    _serve_tokenizer()

    [batch] = frontend_messages
    # One reply per request of the step, naming every token it committed, a stop token
    # too; a stop string and the output limit admission applied reach the frontend.
    assert [
        (
            reply.uid,
            reply.finished,
            reply.finish_reason,
            reply.token_ids,
            reply.stop_str,
            reply.max_new_tokens,
        )
        for reply in batch
    ] == [
        (5, False, None, (3, 4), None, None),
        (6, True, "length", (3,), None, 9),
        (7, True, "stop", (2,), None, 9),
        (8, True, "stop", (4,), "\n", 9),
    ]
    assert batch[0].logprobs == (TokenLogprobs(3, -0.5), TokenLogprobs(4, -0.25))


def test_tokenizer_worker_releases_decode_state_of_aborted_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managers: list[detokenize_module.IncrementalDetokenizer] = []
    frontend_messages: list[object] = []

    class _Tokenizer:
        added_tokens_decoder: dict[int, object] = {}

        def batch_decode(self, token_ids: list[list[int]], **_options: bool) -> list[str]:
            return ["" for _ in token_ids]

    class _Detokenizer(detokenize_module.IncrementalDetokenizer):
        def __init__(self, decoder: object) -> None:
            super().__init__(decoder)
            managers.append(self)

    class _PushQueue:
        def __init__(self, endpoints: ServingEndpoints, target: str, index: int = 0) -> None:
            self._addr = endpoints.link(target, receiving=False, index=index)[0]

        def put(self, messages: list[object]) -> None:
            if self._addr == "frontend" and messages:
                frontend_messages.append(list(messages))

        def stop(self) -> None:
            pass

    class _PullQueue(_PushQueue):
        def __init__(self, endpoints: ServingEndpoints, target: str, index: int = 0) -> None:
            super().__init__(endpoints, target, index)
            self._pending = [
                [DetokenizeMsg(uid=5, token_ids=(3,))],
                # A release coalesced after the request's last token.
                [
                    DetokenizeMsg(uid=6, token_ids=(3,)),
                    RequestAbortedMsg(uid=6),
                    RequestAbortedMsg(uid=5),
                ],
            ]

        def get(self) -> object:
            if self._pending:
                return self._pending.pop(0)
            raise KeyboardInterrupt

    monkeypatch.setattr(tokenizer_server_module, "ZmqPushQueue", _PushQueue)
    monkeypatch.setattr(tokenizer_server_module, "ZmqPullQueue", _PullQueue)
    monkeypatch.setattr(text_module, "load_tokenizer", lambda _path: _Tokenizer())
    monkeypatch.setattr(tokenizer_server_module, "IncrementalDetokenizer", _Detokenizer)

    _serve_tokenizer()

    [manager] = managers
    assert manager.streams == {}
    assert frontend_messages == [
        [UserReply(uid=5, incremental_output="", finished=False, token_ids=(3,))],
        [UserReply(uid=6, incremental_output="", finished=False, token_ids=(3,))],
    ]


class _PieceTokenizer:
    """Decodes 101 to "é", 102 to "中", ids >= 200 to a word with no trailing space and others
    to "w<id> ".

    100 is the first byte of "é": it decodes to nothing before 101 and to "\ufffd" at the end.
    """

    prefix_stable = True

    def stream(self, *, skip_special_tokens: bool) -> TextStream:
        return TextStream(self, skip_special_tokens=skip_special_tokens)

    def renders(self, token_id: int, *, skip_special_tokens: bool) -> bool:
        return True

    def commits(self, text: str, pending_ids: int) -> bool:
        return bool(text) and not text.endswith("\ufffd")

    def batch_decode(self, token_ids: list[list[int]], **_options: bool) -> list[str]:
        texts = []
        for ids in token_ids:
            text = "".join(_piece(token) for token in ids if token != 100)
            if ids and ids[-1] == 100:
                text += "\ufffd"
            texts.append(text)
        return texts


def _piece(token: int) -> str:
    if token == 101:
        return "é"
    if token == 102:
        return "中"
    return f"x{token}" if token >= 200 else f"w{token} "


def _stream(manager, batches):
    texts: dict[int, str] = {}
    for batch in batches:
        for msg, text in zip(batch, manager.detokenize(batch), strict=True):
            texts[msg.uid] = texts.get(msg.uid, "") + text
    return texts


def test_detokenizer_streams_several_tokens_of_one_request_in_one_batch() -> None:
    def msg(uid: int, token: int, finished: bool = False) -> DetokenizeMsg:
        end = RequestEnd(TokenUsage(prompt_tokens=1, completion_tokens=1), 1) if finished else None
        return DetokenizeMsg(uid=uid, token_ids=(token,), end=end)

    one_by_one = [
        [msg(1, 10)],
        [msg(2, 20)],
        [msg(1, 100)],
        [msg(1, 101)],
        [msg(2, 21)],
        [msg(1, 11)],
        [msg(1, 12, finished=True)],
        [msg(2, 22, finished=True)],
    ]
    # The same tokens arriving as speculative rounds: several per request per batch.
    rounds = [
        [msg(1, 10), msg(2, 20), msg(1, 100)],
        [msg(1, 101), msg(2, 21), msg(1, 11), msg(1, 12, finished=True)],
        [msg(2, 22, finished=True)],
    ]

    reference = detokenize_module.IncrementalDetokenizer(_PieceTokenizer())
    batched = detokenize_module.IncrementalDetokenizer(_PieceTokenizer())
    assert _stream(batched, rounds) == _stream(reference, one_by_one)
    assert _stream(detokenize_module.IncrementalDetokenizer(_PieceTokenizer()), rounds) == {
        1: "w10 éw11 w12 ",
        2: "w20 w21 w22 ",
    }
    assert batched.streams == {}
    # Each message carries the text its own tokens add.
    assert detokenize_module.IncrementalDetokenizer(_PieceTokenizer()).detokenize(
        [msg(3, 30), msg(3, 31)]
    ) == ["w30 ", "w31 "]

    # Text held back for an incomplete character or word is flushed when the request
    # finishes, so the stream always concatenates to the decode of its tokens.
    for batches in (
        [[msg(4, 200), msg(4, 100, finished=True)]],
        [[msg(5, 201), msg(5, 100)], [msg(5, 101), msg(5, 202, finished=True)]],
        [[msg(6, 10), msg(6, 203), msg(6, 100)], [msg(6, 204, finished=True)]],
        [[msg(7, 102), msg(7, 100)], [msg(7, 101), msg(7, 100)], [msg(7, 101, finished=True)]],
    ):
        uid = batches[0][0].uid
        ids = [t for batch in batches for m in batch for t in m.token_ids]
        (expected,) = _PieceTokenizer().batch_decode([ids])
        manager = detokenize_module.IncrementalDetokenizer(_PieceTokenizer())
        assert _stream(manager, batches) == {uid: expected}
        assert manager.streams == {}


@pytest.mark.parametrize("return_prompt", [False, True])
def test_tokenizer_error_rejects_one_request_and_continues(monkeypatch, return_prompt):
    from cti.protocol import (
        ChatPrompt,
        SamplingParams,
        TextPrompt,
        TokenizeMsg,
        UserMsg,
    )

    outputs = {"backend": [], "frontend": []}
    sent = []
    stopped = []

    class Push:
        def __init__(self, endpoints, target, index=0):
            self.addr = endpoints.link(target, receiving=False, index=index)[0]

        def put(self, msgs):
            if msgs:
                outputs[self.addr].extend(msgs)
                sent.append(self.addr)

        def stop(self):
            stopped.append(self.addr)

    class Pull(Push):
        def __init__(self, endpoints, target, index=0):
            super().__init__(endpoints, target, index)
            self.messages = [
                [
                    TokenizeMsg(
                        uid=1,
                        prompt=ChatPrompt(({"role": "user", "content": "bad"},)),
                        sampling=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
                        return_prompt=return_prompt,
                    ),
                    TokenizeMsg(
                        uid=2,
                        prompt=TextPrompt("good"),
                        sampling=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
                        return_prompt=return_prompt,
                    ),
                ]
            ]

        @property
        def socket(self):
            # Nothing waits while calls are preprocessed; once all are sent, the stop.
            return SimpleNamespace(poll=lambda timeout=None: bool(self.messages) or timeout is None)

        def get(self):
            if self.messages:
                return self.messages.pop(0)
            raise KeyboardInterrupt

    tokenizer = Mock(added_tokens_decoder={})
    tokenizer.apply_chat_template.side_effect = ValueError("no chat_template")
    tokenizer.encode.return_value = [1, 2]
    monkeypatch.setattr(tokenizer_server_module, "ZmqPushQueue", Push)
    monkeypatch.setattr(tokenizer_server_module, "ZmqPullQueue", Pull)
    monkeypatch.setattr(text_module, "load_tokenizer", lambda path: tokenizer)
    _serve_tokenizer(tokenizer_id=0)
    rejected, *prompt = outputs["frontend"]
    assert isinstance(rejected, RequestRejectedMsg)
    assert rejected.uid == 1 and rejected.kind is RejectionKind.INVALID_REQUEST
    assert "chat_template" in rejected.reason
    # The prompt reply, when asked for, leaves before the request reaches the engine.
    assert prompt == ([PromptReply(2, (1, 2), "good")] if return_prompt else [])
    assert sent == ["frontend", *(["frontend"] if return_prompt else []), "backend"]
    assert len(outputs["backend"]) == 1
    assert isinstance(outputs["backend"][0], UserMsg)
    assert outputs["backend"][0].uid == 2
    assert outputs["backend"][0].input_ids.tolist() == [1, 2]
    assert set(stopped) == {"backend", "frontend", "tokenizer-0"}


def test_requests_taken_together_reach_the_engine_as_one_message(monkeypatch):
    """A call's requests, tokenized together, reach the engine as one message, in order: the
    engine admits them together, so a repeated call is batched alike."""

    from cti.protocol import SamplingParams, TextPrompt, TokenizeMsg

    outputs = {"backend": [], "frontend": []}

    class Push:
        def __init__(self, endpoints, target, index=0):
            self.addr = endpoints.link(target, receiving=False, index=index)[0]

        def put(self, msgs):
            if msgs:
                outputs[self.addr].append(list(msgs))

        def stop(self):
            pass

    class Pull(Push):
        def __init__(self, endpoints, target, index=0):
            super().__init__(endpoints, target, index)
            requests = [
                TokenizeMsg(
                    uid,
                    TextPrompt("same"),
                    SamplingParams(max_new_tokens=8, seed=7 + uid).snapshot(),
                )
                for uid in range(3)
            ]
            self.messages = [requests]

        @property
        def socket(self):
            # Nothing waits while calls are preprocessed; once all are sent, the stop.
            return SimpleNamespace(poll=lambda timeout=None: bool(self.messages) or timeout is None)

        def get(self):
            if self.messages:
                return self.messages.pop(0)
            raise KeyboardInterrupt

    tokenizer = Mock(added_tokens_decoder={})
    tokenizer.encode.return_value = [1, 2]
    monkeypatch.setattr(tokenizer_server_module, "ZmqPushQueue", Push)
    monkeypatch.setattr(tokenizer_server_module, "ZmqPullQueue", Pull)
    monkeypatch.setattr(text_module, "load_tokenizer", lambda path: tokenizer)
    _serve_tokenizer(tokenizer_id=0)
    (message,) = outputs["backend"]
    assert [request.uid for request in message] == [0, 1, 2]
    assert [request.sampling.seed for request in message] == [7, 8, 9]


def test_a_tokenizer_preprocesses_a_calls_prompts_together_and_sends_them_in_order(monkeypatch):
    """A tokenizer worker preprocesses a call's prompts on its pool, so one slow prompt (an
    image to decode) does not hold the others; the call still reaches the engine as one
    message in its order."""

    import threading

    from cti.protocol import SamplingParams, TextPrompt, TokenizeMsg

    together = threading.Barrier(3, timeout=5)  # passes only if three prompts run at once
    outputs = {"backend": []}

    class _Encoder:
        def request(self, uid, prompt, sampling, priority):
            together.wait()
            return SimpleNamespace(uid=uid)

    class Push:
        def __init__(self, endpoints, target, index=0):
            self.addr = endpoints.link(target, receiving=False, index=index)[0]

        def put(self, msgs):
            if msgs and self.addr == "backend":
                outputs["backend"].append([msg.uid for msg in msgs])

        def stop(self):
            pass

    class Pull(Push):
        def __init__(self, endpoints, target, index=0):
            super().__init__(endpoints, target, index)
            spec = SamplingParams(max_new_tokens=8).snapshot()
            self.messages = [[TokenizeMsg(uid, TextPrompt("x"), spec) for uid in (5, 3, 4)]]

        @property
        def socket(self):
            # Nothing waits while calls are preprocessed; once all are sent, the stop.
            return SimpleNamespace(poll=lambda timeout=None: bool(self.messages) or timeout is None)

        def get(self):
            if self.messages:
                return self.messages.pop(0)
            raise KeyboardInterrupt

    monkeypatch.setattr(tokenizer_server_module, "ZmqPushQueue", Push)
    monkeypatch.setattr(tokenizer_server_module, "ZmqPullQueue", Pull)
    monkeypatch.setattr(
        tokenizer_server_module.PromptEncoder, "from_checkpoint", lambda *_a, **_k: _Encoder()
    )
    _serve_tokenizer(tokenizer_id=0)

    assert outputs["backend"] == [[5, 3, 4]]


def _tokenizer_outputs(monkeypatch, messages: list[list], encoder) -> list[list[object]]:
    """What a tokenizer worker over ``encoder`` sends the engine, message by message, for
    the frontend ``messages``; it serves until they are all taken and sent."""

    engine: list[list[object]] = []

    class Push:
        def __init__(self, endpoints, target, index=0):
            self.addr = endpoints.link(target, receiving=False, index=index)[0]

        def put(self, msgs):
            if msgs and self.addr == "backend":
                engine.append(list(msgs))

        def stop(self):
            pass

    class Pull(Push):
        @property
        def socket(self):
            return SimpleNamespace(poll=lambda timeout=None: bool(messages) or timeout is None)

        def get(self):
            if messages:
                return messages.pop(0)
            raise KeyboardInterrupt

    monkeypatch.setattr(tokenizer_server_module, "ZmqPushQueue", Push)
    monkeypatch.setattr(tokenizer_server_module, "ZmqPullQueue", Pull)
    monkeypatch.setattr(
        tokenizer_server_module.PromptEncoder, "from_checkpoint", lambda *_a, **_k: encoder
    )
    _serve_tokenizer(tokenizer_id=0)
    return engine


def _call(*uids: int) -> list:
    from cti.protocol import SamplingParams, TextPrompt, TokenizeMsg

    spec = SamplingParams(max_new_tokens=8).snapshot()
    return [TokenizeMsg(uid, TextPrompt(str(uid)), spec) for uid in uids]


def test_a_tokenizer_preprocesses_several_calls_at_once(monkeypatch):
    """Calls of one prompt each (one image each) are preprocessed together, not one after
    another; each still reaches the engine as its own message."""

    import threading

    together = threading.Barrier(2, timeout=5)  # passes only if both calls run at once

    class _Encoder:
        def request(self, uid, prompt, sampling, priority):
            together.wait()
            return SimpleNamespace(uid=uid)

    engine = _tokenizer_outputs(monkeypatch, [_call(1), _call(2)], _Encoder())

    assert sorted([msg.uid for msg in message] for message in engine) == [[1], [2]]


def test_an_abort_of_a_call_still_preprocessing_follows_its_requests(monkeypatch):
    """The abort of a request whose call is still preprocessing reaches the engine after
    the call's requests, never before them; an abort of a request no call holds goes at
    once."""

    import threading

    from cti.protocol import AbortBackendMsg, AbortMsg

    release = threading.Event()

    class _Encoder:
        def request(self, uid, prompt, sampling, priority):
            release.wait(5)
            return SimpleNamespace(uid=uid)

    class _Release(list):
        def pop(self, index=-1):
            message = super().pop(index)
            if not self:  # the aborts are taken in: let the call finish
                release.set()
            return message

    messages = _Release([_call(1, 2), [AbortMsg(2), AbortMsg(9)]])
    engine = _tokenizer_outputs(monkeypatch, messages, _Encoder())

    assert engine[0] == [AbortBackendMsg(9)]
    assert [msg.uid for msg in engine[1]] == [1, 2]
    assert engine[2] == [AbortBackendMsg(2)]
