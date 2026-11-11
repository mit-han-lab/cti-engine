from __future__ import annotations

import os
import threading
import time
from types import TracebackType
from typing import TYPE_CHECKING, Self

from cti.config import EngineOptions, LaunchFile, ServerOptions
from cti.protocol import AdmissionPriority, SamplingParams

from .api_server import ApiServer

if TYPE_CHECKING:
    from cti.config import EngineTuning
    from cti.protocol import SamplingPrompt, SamplingResult
    from cti.runtime import TrainingSession


class OnlineServer:
    """Own one online inference frontend, served on its own thread with bounded readiness
    and shutdown, its local bounded sampling and its local typed training facade."""

    _STARTUP_TIMEOUT = 600.0
    _POLL_INTERVAL = 0.01
    _JOIN_TIMEOUT = 60.0
    _FORCE_JOIN_TIMEOUT = 10.0

    def __init__(
        self,
        options: EngineOptions,
        server: ServerOptions,
        *,
        tuning: EngineTuning | None = None,
    ) -> None:
        """``tuning`` is internal, for tests and benchmarks; None keeps its defaults."""

        self._server = server
        self._api = ApiServer(options, server, tuning=tuning)
        self._thread = threading.Thread(target=self._run, name="cti-api-server", daemon=False)
        self._failure: BaseException | None = None
        self._done = threading.Event()
        self._started = False
        self._closed = False

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> Self:
        """Construct serving and optional local training from one launch file."""

        launch = LaunchFile.load(path)
        return cls(launch.engine, launch.server)

    @property
    def host(self) -> str:
        return self._server.host

    @property
    def port(self) -> int:
        return self._server.port

    @property
    def model_revision(self) -> int:
        return self._api.engine.model_revision()

    def sample(
        self,
        prompts: list[SamplingPrompt],
        sampling_params: list[SamplingParams] | SamplingParams,
        *,
        priority: AdmissionPriority = AdmissionPriority.NORMAL,
    ) -> SamplingResult:
        """Return exact bounded samples, as ``LLM.sample`` does, from the served model.

        The requests travel the server's own path and are admitted with its HTTP requests,
        at ``priority``. Admission is strict priority without aging: ``LOW`` gets no minimum
        share and can wait indefinitely while ``NORMAL`` traffic saturates the engine; use
        ``NORMAL`` when the rollout must not wait behind HTTP traffic. A seeded prompt
        may sample differently across calls when one hits the prefix cache (see the
        prefix-hit note under Radix Cache in docs/features.md). A prompt is text or its
        token ids; chats and images are not sampled online yet. The model changing while
        the command samples, as an optimizer step does, raises: resample. A rejection aborts
        the rest and raises ``RequestRejectedError`` naming the first rejected request to
        arrive, not the lowest-numbered one ``LLM.sample`` names.
        """

        return self._api.sample(prompts, sampling_params, priority)

    def training_session(self) -> TrainingSession:
        """A local training session on the served model, beside online requests."""

        return self._api.engine.training_session(self)

    def start(self) -> Self:
        if self._closed:
            raise RuntimeError("OnlineServer is closed")
        if self._started:
            raise RuntimeError("OnlineServer has already started")
        self._started = True
        self._thread.start()
        try:
            self._wait_until_ready()
        except BaseException as error:
            try:
                self._stop(report_failure=False)
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve startup failure
                error.add_note(f"API server cleanup also failed: {cleanup_error}")
            self._closed = True
            raise
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            self._stop(report_failure=True)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del error_type, traceback
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note(f"OnlineServer cleanup also failed: {cleanup_error}")

    def _wait_until_ready(self) -> None:
        """Until the HTTP server serves. The engine's start is bounded by its own watchdog,
        which renews on progress and raises its cause; ``_STARTUP_TIMEOUT`` bounds the rest."""

        deadline: float | None = None
        while not self._api.http_server.started:
            if self._done.is_set():
                self._raise_failure("API server stopped before reporting readiness")
            if self._api.starting_engine:
                deadline = None
            elif deadline is None:
                deadline = time.monotonic() + self._STARTUP_TIMEOUT
            elif time.monotonic() >= deadline:
                raise TimeoutError(
                    f"API server did not become ready within {self._STARTUP_TIMEOUT:.1f} "
                    "seconds of its engine starting"
                )
            self._done.wait(self._POLL_INTERVAL)

    def _stop(self, *, report_failure: bool) -> None:
        http_server = self._api.http_server
        http_server.should_exit = True
        self._thread.join(self._JOIN_TIMEOUT)
        if self._thread.is_alive():
            http_server.force_exit = True
            self._thread.join(self._FORCE_JOIN_TIMEOUT)
        if self._thread.is_alive():
            raise RuntimeError("API server thread did not stop")
        if report_failure and self._failure is not None:
            self._raise_failure("API server failed")

    def _run(self) -> None:
        primary: BaseException | None = None
        try:
            self._api.run()
        except BaseException as error:  # noqa: BLE001 - cross-thread propagation
            primary = error
        finally:
            try:
                # The thread that starts multiprocessing children also performs
                # their protocol shutdown. Public callers only signal and join
                # this complete application owner.
                self._api.close()
            except BaseException as error:  # noqa: BLE001 - finish owned cleanup
                if primary is None:
                    primary = error
                else:
                    primary.add_note(f"Engine cleanup also failed: {error}")
            self._failure = primary
            self._done.set()

    def _raise_failure(self, message: str) -> None:
        raise RuntimeError(message) from self._failure


__all__ = ["OnlineServer"]
