from __future__ import annotations

import logging
import os
import sys

import pytest

import cti.utils.logger as logger_module
from cti.utils.logger import init_logger, log_stage


def _format(logger: logging.Logger, **attributes: object) -> str | None:
    """What the package handler writes for one INFO record of ``logger``, None if a filter
    drops it."""

    record = logging.LogRecord(logger.name, logging.INFO, __file__, 1, "message", None, None)
    record.__dict__.update(attributes)
    for source in logger.filters:
        source.filter(record)
    # The package handler comes first; pytest adds its capture handlers after it.
    handler = logging.getLogger("cti").handlers[0]
    return handler.format(record) if handler.filter(record) else None


def test_a_record_with_a_rank_carries_its_prefix() -> None:
    logger = init_logger("cti.tests.logger.prefix")

    assert "|core|rank=1]" in _format(logger, rank=1)
    assert "rank=" not in _format(logger)


def test_rank0_methods_mark_their_records(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = init_logger("cti.tests.logger.rank0")
    calls: list[dict] = []
    monkeypatch.setattr(logger, "info", lambda message, *args, **kwargs: calls.append(kwargs))

    logger.info_rank0("visible")

    assert calls == [{"extra": {"rank0": True}}]


def test_env_sets_log_level_and_pid_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logger_module, "_LOG_LEVEL", None)
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_PID", "1")

    logger = init_logger("cti.tests.logger.env", "/any/dir/source.py")

    assert logger.level == logging.DEBUG
    assert f"|pid={os.getpid()}|source.py]" in _format(logger)


def _fail_inside() -> None:
    raise ValueError("rank state lost")


def test_an_exception_record_carries_its_traceback_and_stack() -> None:
    logger = init_logger("cti.tests.logger.exception")
    try:
        _fail_inside()
    except ValueError:
        line = _format(logger, exc_info=sys.exc_info(), stack_info="Stack (most recent call last):")

    lines = line.splitlines()
    assert lines[0].endswith(" message")
    assert lines[1] == "Traceback (most recent call last):"
    assert any("in _fail_inside" in text for text in lines)
    assert "ValueError: rank state lost" in lines
    assert lines[-1] == "Stack (most recent call last):"


def test_a_stage_is_logged_per_request_at_debug_level_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("cti.test_stage")
    logger.setLevel(logging.INFO)
    log_stage(logger, "sent", (1, 2), at=3.25)
    assert not caplog.records

    logger.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG, logger="cti.test_stage"):
        log_stage(logger, "sent", (1, 2), at=3.25)
    assert [r.getMessage() for r in caplog.records] == [
        "stage sent uid=1 t=3.250000",
        "stage sent uid=2 t=3.250000",
    ]
