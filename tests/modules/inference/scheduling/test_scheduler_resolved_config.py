from __future__ import annotations

import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

import cti.inference._control._engine as inference_engine_module
import ctikernels.radix
from cti.config import SchedulingOptions
from cti.inference import ServingLimits
from cti.inference._control._engine import InferenceEngine
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.strategy import SchedulingLimits, TokenSteps
from cti.kvcache import KVLayout, RecurrentRows, StateCapacity, TableGeometry

# An attention-only state: snapshots, if any, lie on whole pages.
_ATTENTION = KVLayout((), RecurrentRows(()))


def test_inference_engine_import_does_not_load_device_execution_modules() -> None:
    code = """
import sys
import cti.inference._control._engine

forbidden = (
    "cti.inference._execution.runner",
    "cti.inference._execution._graphs",
    "cti.distributed.runtime",
)
raise SystemExit(any(name in sys.modules for name in forbidden))
"""

    subprocess.run((sys.executable, "-c", code), check=True)


def _engine(capacity: StateCapacity, *, prefix_cache: str = "naive") -> InferenceEngine:
    limits = ServingLimits(
        capacity=capacity,
        rounds=None,
        stop_token_ids=(0, 7),
        vocab_size=1 << 20,
        image_rows=0,
        prefix_cache=prefix_cache,
        max_batch_tokens=SchedulingOptions().max_batch_tokens,
        max_prompt_tokens=8192,
    )
    return InferenceEngine(
        limits,
        Mock(),
        publish=lambda _replies: None,
        agree_on_step=None,
        text_decoder=None,
        overlap=True,
        checks=True,
    )


@pytest.mark.parametrize("snapshot_rows", [0, 2])
def test_inference_engine_builds_page_aligned_cpu_state_from_resolved_capacity(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_rows: int,
) -> None:
    def reject_cuda_stream(*_args: object, **_kwargs: object) -> None:
        pytest.fail("InferenceEngine must not construct or install a CUDA stream")

    monkeypatch.setattr(torch.cuda, "Stream", reject_cuda_stream)
    monkeypatch.setattr(torch.cuda, "set_stream", reject_cuda_stream)
    geometry = TableGeometry(16, 8, 33)

    cache = _engine(geometry.capacity(_ATTENTION, 128, snapshot_rows))._cache

    # One page ID per page column: 33 tokens take 3 pages, rows aligned to 128 bytes.
    assert cache.page_table.shape == (8, 32)
    assert cache.page_table.dtype == torch.int32
    assert cache.page_table.device.type == "cpu"
    # Without reuse no rows are indexed, whatever the capacity holds.
    assert cache.prefix_cache.rows is None
    assert (cache.num_pages, cache.page_size) == (128, 16)


@pytest.mark.parametrize("cache_type", ["naive", "radix"])
def test_inference_engine_logs_the_prefix_cache_the_worker_resolved(
    monkeypatch: pytest.MonkeyPatch,
    cache_type: str,
) -> None:
    # docs/features.md: cache_type "auto" becomes radix or naive, and startup logs which.
    logger = Mock()
    monkeypatch.setattr(inference_engine_module, "logger", logger)
    radix_warm_up = Mock()
    monkeypatch.setattr(ctikernels.radix, "warm_up", radix_warm_up)

    _engine(TableGeometry(1, 256, 8).capacity(_ATTENTION, 8, 0), prefix_cache=cache_type)

    logger.info.assert_called_once_with("Using the %s prefix cache", cache_type)
    # A radix cache loads its key comparison before the first request.
    assert radix_warm_up.call_count == (cache_type == "radix")


@pytest.mark.parametrize(
    ("stop_token_ids", "error_type"),
    (
        (None, TypeError),
        (2, TypeError),
        ([2], TypeError),
        ((1.5,), TypeError),
        ((True,), TypeError),
        ((2, -1), ValueError),
    ),
)
def test_request_scheduler_rejects_invalid_stop_token_ids(
    stop_token_ids: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="stop_token_ids"):
        InferenceScheduler(
            checks=True,
            limits=SchedulingLimits(
                max_running_requests=1,
                max_batch_tokens=8,
                max_sequence_length=8,
                max_prompt_tokens=8,
            ),
            cache_manager=Mock(page_size=1),
            stop_token_ids=stop_token_ids,
            vocab_size=1 << 20,
            decode=TokenSteps(),
            encoder_cache=EncoderCache(0),
        )


@pytest.mark.parametrize("value", [0, True, 1.0])
@pytest.mark.parametrize(
    "field",
    ("max_running_requests", "max_batch_tokens", "max_sequence_length"),
)
def test_scheduling_limits_require_positive_native_integers(
    field: str,
    value: object,
) -> None:
    values = {
        "max_running_requests": 1,
        "max_batch_tokens": 8,
        "max_sequence_length": 8,
        "max_prompt_tokens": 8,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        SchedulingLimits(**values)
