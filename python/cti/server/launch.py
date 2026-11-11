# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/server/launch.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import signal
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cti.config import EngineTuning

    from .api_server import ApiServer


class _ServerTermination:
    """Own first-graceful, repeated-force termination for the server CLI."""

    def __init__(self, server: ApiServer) -> None:
        self._server = server
        self._shutdown_requested = False
        self._previous_handlers: dict[signal.Signals, object] = {}

    def __enter__(self) -> _ServerTermination:
        termination_signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            termination_signals.append(signal.SIGHUP)
        self._previous_handlers = {
            termination_signal: signal.getsignal(termination_signal)
            for termination_signal in termination_signals
        }
        for termination_signal in termination_signals:
            signal.signal(termination_signal, self._handle)
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: object,
    ) -> None:
        for termination_signal, previous_handler in self._previous_handlers.items():
            signal.signal(termination_signal, previous_handler)
        self._previous_handlers.clear()
        if error_type is None and self._shutdown_requested:
            raise SystemExit

    def _handle(self, _signum: int, _frame: object) -> None:
        force = self._shutdown_requested or self._server.is_closing
        self._shutdown_requested = True
        if force:
            self._server.kill_workers()
        else:
            raise SystemExit


def launch_server(argv: Sequence[str] | None = None, *, tuning: EngineTuning | None = None) -> None:
    """Serve what the command line ``argv`` (None: this process's) spells; ``tuning`` is
    internal, for benchmarks."""

    from .api_server import ApiServer
    from .args import parse_args

    options, server = parse_args(sys.argv[1:] if argv is None else argv)
    if options.parallel.node_rank != 0:
        # Another node of an engine that spans hosts: node 0 serves, this one runs its ranks.
        from cti.runtime import NodeAgent

        raise SystemExit(NodeAgent(options).run())
    api = ApiServer(options, server, tuning=tuning)
    with _ServerTermination(api):
        try:
            api.run()
        finally:
            api.close()


if __name__ == "__main__":
    launch_server()
