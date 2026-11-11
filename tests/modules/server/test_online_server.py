from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

import cti.server.online_server as online_server_module
from cti.config import (
    AdamWConfig,
    BatchLayout,
    EngineOptions,
    ModelOptions,
    ServerOptions,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.protocol import AdmissionPriority, SamplingParams
from cti.runtime import EngineClient
from cti.server import OnlineServer

_OPTIONS = EngineOptions(model=ModelOptions(path="model"))


def test_online_server_owns_the_engine_it_exposes_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    # A missing EngineClient method fails here, as it would in the server.
    engine = Mock(spec=EngineClient, **{"model_revision.return_value": 3})
    training_config = TrainingConfig(
        trainable_plan=TrainablePlan(),
        optimizer=AdamWConfig(),
        batching=TrainingBatching(
            BatchLayout.PADDED, StaticBatching(microbatch_size=1, max_tokens_per_sample=8)
        ),
    )

    servers: list[_ApiServer] = []

    class _ApiServer:
        def __init__(self, options, server, *, tuning) -> None:
            self.options, self.server, self.tuning = options, server, tuning
            servers.append(self)
            self.http_server = _HTTPServer()
            self.started = False
            self.starting_engine = False

        @property
        def engine(self):
            if not self.started:
                raise RuntimeError("API server is not running")
            return engine

        def run(self) -> None:
            events.append("api.run")
            self.started = True
            self.http_server.started = True
            while not self.http_server.should_exit:
                threading.Event().wait(0.001)
            events.append("api.stopped")

        def close(self) -> None:
            events.append("api.close")

    monkeypatch.setattr(online_server_module, "ApiServer", _ApiServer)

    server = OnlineServer(
        EngineOptions(model=ModelOptions(path="model"), training=training_config),
        ServerOptions(host="127.0.0.1", port=8000, num_tokenizers=2),
    )
    with pytest.raises(RuntimeError, match="not running"):
        server.training_session()

    with server as running:
        assert running is server
        assert server.model_revision == 3
        assert server.training_session() is engine.training_session.return_value
        engine.training_session.assert_called_once_with(server)
        assert (server.host, server.port) == ("127.0.0.1", 8000)
        (api,) = servers
        assert api.options.training is training_config and api.tuning is None
        assert api.server == ServerOptions(host="127.0.0.1", port=8000, num_tokenizers=2)

    server.close()
    assert events == ["api.run", "api.stopped", "api.close"]


class _SlowStart:
    """An API server whose engine takes ``engine_seconds`` to start, then whose HTTP server
    starts or, with ``serves`` false, never does."""

    def __init__(self, engine_seconds: float, *, serves: bool) -> None:
        self.http_server = _HTTPServer()
        self.starting_engine = False
        self._engine_seconds, self._serves = engine_seconds, serves

    def run(self) -> None:
        self.starting_engine = True
        threading.Event().wait(self._engine_seconds)
        self.starting_engine = False
        self.http_server.started = self._serves
        while not self.http_server.should_exit:
            threading.Event().wait(0.001)

    def close(self) -> None:
        pass


def test_the_readiness_deadline_leaves_a_slow_engine_start_to_its_own_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OnlineServer, "_STARTUP_TIMEOUT", 0.05)
    api = _SlowStart(0.3, serves=True)
    monkeypatch.setattr(online_server_module, "ApiServer", lambda *_a, **_k: api)

    with OnlineServer(_OPTIONS, ServerOptions()):
        assert api.http_server.started


def test_the_readiness_deadline_runs_once_the_engine_has_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OnlineServer, "_STARTUP_TIMEOUT", 0.05)
    api = _SlowStart(0.0, serves=False)
    monkeypatch.setattr(online_server_module, "ApiServer", lambda *_a, **_k: api)

    with pytest.raises(TimeoutError, match="seconds of its engine starting"):
        OnlineServer(_OPTIONS, ServerOptions()).start()


def test_online_server_closes_its_engine_after_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = Mock(http_server=_HTTPServer())
    api.run.side_effect = ValueError("engine startup failed")
    monkeypatch.setattr(online_server_module, "ApiServer", lambda *_a, **_k: api)

    server = OnlineServer(_OPTIONS, ServerOptions())
    with pytest.raises(RuntimeError, match="stopped before reporting readiness") as caught:
        server.start()
    server.close()

    assert isinstance(caught.value.__cause__, ValueError)
    api.close.assert_called_once_with()


def test_online_server_samples_through_its_api_server(monkeypatch: pytest.MonkeyPatch) -> None:
    api = Mock(http_server=_HTTPServer())
    monkeypatch.setattr(online_server_module, "ApiServer", lambda *_a, **_k: api)
    server = OnlineServer(_OPTIONS, ServerOptions())
    params = SamplingParams(max_new_tokens=4)

    # The params go as given: SamplingParams.specs applies one to every prompt.
    assert server.sample(["a", "b"], params) is api.sample.return_value
    server.sample(["a"], [params], priority=AdmissionPriority.LOW)

    assert [call.args for call in api.sample.call_args_list] == [
        (["a", "b"], params, AdmissionPriority.NORMAL),
        (["a"], [params], AdmissionPriority.LOW),
    ]


class _HTTPServer:
    def __init__(self) -> None:
        self.started = False
        self.should_exit = False
        self.force_exit = False
