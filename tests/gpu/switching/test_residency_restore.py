"""Release gate: a resume the device cannot back raises and leaves the process running.

A spawned child, with torch_memory_saver preloaded as in serving, owns two KV layer regions
and a persistent training region. With everything suspended, a filler allocation takes
the device's memory so the KV no longer fits: resuming it must raise
``ResidencyRestoreError`` without mapping anything (the provider would exit the process),
while the smaller training region still resumes. Once the filler is gone the KV resumes
at its old addresses, and a training region that no longer fits raises the same way.
The child then parks; the parent proves it was still alive to be terminated.

Not run by the CPU suites; skipped without a CUDA device.
"""

from __future__ import annotations

import json
import signal
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT
from tests.support.residency import (
    PRELOAD_BASENAME,
    one_rank,
    park_child,
    receive_from_parked_child,
    require_cuda13_preload_mapping,
    send_failure,
    start_child,
    terminate_parked_child,
)

pytestmark = pytest.mark.checkpoints()

if TYPE_CHECKING:
    import torch

RECEIPT_PREFIX = "CTI_RESIDENCY_RESTORE="
_GIB = 1024**3
_KV_LAYER_BYTES = _GIB
_KV_LAYERS = 2
_PERSISTENT_BYTES = _GIB // 2
# Free memory the filler leaves: below the KV's 2 GiB, above the training region's.
_LEFT_FOR_TRAINING = _GIB


def _holds(tensor: torch.Tensor, value: int) -> bool:
    """Whether every byte of ``tensor`` is ``value``, without a tensor-sized temporary:
    the device may have no room for one."""

    import torch

    low, high = torch.aminmax(tensor)
    return int(low) == int(high) == value


def _restore_failure(resume: object) -> dict[str, object] | None:
    from cti.memory import ResidencyRestoreError

    try:
        resume(one_rank)
    except ResidencyRestoreError as error:
        return {
            "type": type(error).__name__,
            "region": error.region,
            "required_bytes": error.required_bytes,
            "free_bytes": error.free_bytes,
        }
    return None


def _run_restore_child(connection: Connection) -> NoReturn:
    provider_bound = False
    try:
        import torch

        from cti.memory import MemoryManager
        from tests.support.gpu import configure_worker_process

        configure_worker_process()
        from cti.memory import unified_memory

        memory_manager = MemoryManager.from_stock_provider(
            unified=unified_memory(torch.device("cuda"))
        )
        provider_bound = True
        preload_mapping = require_cuda13_preload_mapping()
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        def free_bytes() -> int:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            return torch.cuda.mem_get_info(device)[0]

        def fill(leave: int) -> torch.Tensor:
            return torch.empty(free_bytes() - leave, dtype=torch.uint8, device=device)

        with memory_manager.kv_allocation():
            layers = [
                torch.empty(_KV_LAYER_BYTES, dtype=torch.uint8, device=device)
                for _ in range(_KV_LAYERS)
            ]
        with memory_manager.persistent_training_allocation():
            persistent = torch.empty(_PERSISTENT_BYTES, dtype=torch.uint8, device=device)
        addresses = [layer.data_ptr() for layer in layers]

        # Training releases its memory for inference, but another allocation took the KV's.
        memory_manager.suspend_kv()
        memory_manager.suspend_training()
        filler = fill(_LEFT_FOR_TRAINING)
        kv_failure = _restore_failure(memory_manager.resume_kv)
        repeated_kv_failure = _restore_failure(memory_manager.resume_kv)
        # The rejected switch keeps training resident.
        memory_manager.resume_training(one_rank)
        persistent.fill_(3)
        persistent_written = _holds(persistent, 3)
        del filler

        memory_manager.suspend_training()
        memory_manager.resume_kv(one_rank)
        for value, layer in enumerate(layers, start=1):
            layer.fill_(value)
        kv_restored = {
            "addresses_kept": [layer.data_ptr() for layer in layers] == addresses,
            "written": all(_holds(layer, value) for value, layer in enumerate(layers, 1)),
        }

        # Inference holds the device and the training region no longer fits.
        filler = fill(_PERSISTENT_BYTES // 2)
        training_failure = _restore_failure(memory_manager.resume_training)
        del filler
        memory_manager.suspend_kv()
        memory_manager.resume_training(one_rank)
        persistent.fill_(4)
        torch.cuda.synchronize(device)
        connection.send(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "preload_mapping": preload_mapping,
                "kv_failure": kv_failure,
                "repeated_kv_failure": repeated_kv_failure,
                "persistent_written": persistent_written,
                "kv_restored": kv_restored,
                "training_failure": training_failure,
                "persistent_rewritten": _holds(persistent, 4),
            }
        )
    except BaseException:  # noqa: BLE001 - isolated child reports complete diagnostics
        send_failure(connection)
    finally:
        if provider_bound:
            park_child()
        connection.close()
    raise RuntimeError("unreachable")


def _run_orchestrator() -> None:
    import torch

    assert not torch.cuda.is_initialized()
    process, connection = start_child(_run_restore_child)
    try:
        restore = receive_from_parked_child(process, connection)
    finally:
        exitcode = terminate_parked_child(process, connection)
    print(
        RECEIPT_PREFIX
        + json.dumps({"restore_child_exitcode": exitcode, "restore": restore}, sort_keys=True)
    )


def test_resume_that_does_not_fit_raises_and_keeps_the_process(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env, log_root = isolated_runtime_env(tmp_path, "residency-restore")
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.switching.test_residency_restore"),
        log_root / "residency-restore.log",
        env=env,
        cwd=REPO_ROOT,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    restore = receipt["restore"]
    # The child parked after the failures, so only the parent's SIGTERM ended it.
    assert receipt["restore_child_exitcode"] == -signal.SIGTERM
    assert_cuda_device_metadata(restore)
    assert restore["preload_mapping"] == PRELOAD_BASENAME
    for failure in (restore["kv_failure"], restore["repeated_kv_failure"]):
        assert failure is not None and failure["region"] == "KV", restore
        assert failure["required_bytes"] == _KV_LAYERS * _KV_LAYER_BYTES
        assert failure["free_bytes"] < failure["required_bytes"]
    training_failure = restore["training_failure"]
    assert training_failure is not None and training_failure["region"] == "training", restore
    assert training_failure["required_bytes"] == _PERSISTENT_BYTES
    assert restore["persistent_written"] and restore["persistent_rewritten"]
    assert restore["kv_restored"] == {"addresses_kept": True, "written": True}


if __name__ == "__main__":
    _run_orchestrator()
