# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/utils/logger.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING

_LOG_LEVEL: int | None = None
# Every logger init_logger makes writes through this package logger's one handler, where a
# process's distributed owner installs the filter that adds its rank.
_ROOT = "cti"


class _ColorFormatter(logging.Formatter):
    """``[YYYY-MM-DD|HH:MM:SS<suffix>] LEVEL message``, SGLang style, the level colored. A
    record's ``suffix`` names its source; a ``rank`` a filter set adds ``|core|rank=<rank>``."""

    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }

    def format(self, record: logging.LogRecord) -> str:
        suffix = getattr(record, "suffix", "")
        rank = getattr(record, "rank", None)
        if rank is not None:
            suffix = f"{suffix}|core|rank={rank}"
        timestamp = f"{self.formatTime(record, '[%Y-%m-%d|%H:%M:%S')}{suffix}]"
        level = f"{self._COLORS.get(record.levelname, '')}{record.levelname:<8}\033[0m"
        line = f"\033[1m{timestamp}\033[0m {level} {record.getMessage()}"
        # The exception and stack follow, as logging.Formatter.format appends them.
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line = f"{line}\n{record.exc_text}"
        if record.stack_info:
            line = f"{line}\n{self.formatStack(record.stack_info)}"
        return line


class _Source(logging.Filter):
    """Marks a logger's records with the ``suffix`` that names their source."""

    def __init__(self, suffix: str) -> None:
        super().__init__()
        self._suffix = suffix

    def filter(self, record: logging.LogRecord) -> bool:
        record.suffix = self._suffix
        return True


def _package_logger() -> logging.Logger:
    package = logging.getLogger(_ROOT)
    if not package.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ColorFormatter())
        package.addHandler(handler)
        package.propagate = False
    return package


def init_logger(name: str, suffix: str = ""):
    """The colored stdout logger of module ``name``, with ``*_rank0`` methods: their records
    carry ``rank0``, which a rank filter drops off rank 0."""

    global _LOG_LEVEL
    if _LOG_LEVEL is None:
        # A known level name maps to its number; anything else falls back to INFO.
        level = logging.getLevelName(os.getenv("LOG_LEVEL", "").upper())
        _LOG_LEVEL = level if isinstance(level, int) else logging.INFO

    suffix = os.path.basename(suffix)
    if suffix:
        suffix = f"|{suffix}"
    if os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes"):
        suffix = f"|pid={os.getpid()}{suffix}"

    package = _package_logger()
    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)
    logger.handlers.clear()
    logger.filters.clear()
    logger.addFilter(_Source(suffix))
    if name != _ROOT and not name.startswith(f"{_ROOT}."):
        # A module run as __main__: outside the package logger, it shares its handler.
        logger.addHandler(package.handlers[0])
        logger.propagate = False
    else:
        logger.propagate = True

    def _call_rank0(msg, *args, _which, **kwargs):
        getattr(logger, _which)(msg, *args, extra={"rank0": True}, **kwargs)

    if TYPE_CHECKING:

        class WrapperLogger(logging.Logger):
            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)
    else:
        logger.info_rank0 = partial(_call_rank0, _which="info")
        logger.debug_rank0 = partial(_call_rank0, _which="debug")
        logger.warning_rank0 = partial(_call_rank0, _which="warning")
        return logger


def log_stage(logger: logging.Logger, stage: str, uids: Iterable[int], at: float | None = None):
    """At debug level, that the items ``uids`` reached ``stage`` at ``at`` (now by default),
    on ``time.monotonic``, which a host's processes share, so one item's stages line up
    across the logs of several processes."""

    if logger.isEnabledFor(logging.DEBUG):
        at = time.monotonic() if at is None else at
        for uid in uids:
            logger.debug("stage %s uid=%d t=%.6f", stage, uid, at)
