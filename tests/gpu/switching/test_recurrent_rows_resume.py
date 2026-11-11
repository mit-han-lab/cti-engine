"""Release gate: recurrent row copies work after the KV memory comes back.

A spawned child, with torch_memory_saver preloaded as in serving, allocates recurrent rows
through ``DeviceMemory`` in the KV residency and copies a snapshot row onto a request slot
before and after a KV suspend/resume. The resumed region's contents are undefined, so
everything a copy needs besides the rows (its row table) must live outside it: each copy
must leave every layer tensor's slot row bitwise equal to the snapshot row.

Not run by the CPU suites; skipped without a CUDA device.
"""

from __future__ import annotations

import json
import signal
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from typing import NoReturn

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

RECEIPT_PREFIX = "CTI_RECURRENT_ROWS_RESUME="
_LAYERS = 3
_SLOTS = 2
_SNAPSHOTS = 2


def _copies_rows(pool, value: int) -> bool:
    """Write ``value`` into the first snapshot row and 0xFF into slot 0, copy the row onto
    the slot, and whether every layer tensor's slot row now equals the snapshot row."""

    import torch

    source, target = pool.snapshot_rows[0], 0
    views = [view for index in range(pool.num_layers) for view in pool.layer_views(index)]
    for offset, view in enumerate(views):
        view[source].view(torch.uint8).fill_(value + offset)
        view[target].view(torch.uint8).fill_(0xFF)
    pool.copy_rows(torch.tensor([[source], [target]], dtype=torch.int64, device=pool.device))
    torch.cuda.synchronize(pool.device)
    return all(
        torch.equal(view[target].view(torch.uint8), view[source].view(torch.uint8))
        for view in views
    )


def _run_child(connection: Connection) -> NoReturn:
    provider_bound = False
    try:
        from types import SimpleNamespace

        import torch

        from cti.kvcache.recurrent_pool import SlotStateSpec
        from cti.kvcache.state import KVLayout, RecurrentRows, StateCapacity
        from cti.memory import MemoryManager
        from cti.worker._device_memory import DeviceMemory
        from tests.support.gpu import configure_worker_process

        configure_worker_process()
        from cti.memory import unified_memory

        manager = MemoryManager.from_stock_provider(unified=unified_memory(torch.device("cuda")))
        provider_bound = True
        preload_mapping = require_cuda13_preload_mapping()
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        # A Gated DeltaNet row's shape: FP32 state and BF16 convolution history.
        spec = SlotStateSpec(
            (("state", (4, 128, 128), torch.float32), ("conv_state", (3, 1536), torch.bfloat16)),
            export_granularity=64,
        )
        layout = KVLayout((), RecurrentRows([spec] * _LAYERS))
        capacity = StateCapacity(
            page_size=64,
            num_slots=_SLOTS,
            max_seq_len=1024,
            table_width=16,
            num_pages=0,
            snapshot_rows=range(_SLOTS, _SLOTS + _SNAPSHOTS),
            sink_slot=0,
            guard_slots=range(1, 1),
            snapshot_grid=64,
            export_granularity=64,
        )
        plan = SimpleNamespace(inference_ratio=0.5, peak_ratio=None, recurrent_to_kv_ratio=1.0)
        memory = DeviceMemory(
            plan,  # type: ignore[arg-type]
            device=device,
            agree=lambda value: (value, value),
            manager=manager,
        )
        state, _ = memory.allocate(layout, capacity, None, torch.bfloat16)
        pool = state.recurrent
        before = _copies_rows(pool, 1)
        manager.suspend_kv()
        manager.resume_kv(one_rank)
        after = _copies_rows(pool, 2)
        connection.send(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "preload_mapping": preload_mapping,
                "copies_before_suspension": before,
                "copies_after_resume": after,
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
    process, connection = start_child(_run_child)
    try:
        rows = receive_from_parked_child(process, connection)
    finally:
        exitcode = terminate_parked_child(process, connection)
    print(RECEIPT_PREFIX + json.dumps({"exitcode": exitcode, "rows": rows}, sort_keys=True))


def test_recurrent_row_copies_survive_a_kv_suspension(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env, log_root = isolated_runtime_env(tmp_path, "recurrent-rows-resume")
    env["PYTHONNOUSERSITE"] = "1"
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.switching.test_recurrent_rows_resume"),
        log_root / "recurrent-rows-resume.log",
        env=env,
        cwd=REPO_ROOT,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    rows = receipt["rows"]
    assert receipt["exitcode"] == -signal.SIGTERM
    assert_cuda_device_metadata(rows)
    assert rows["preload_mapping"] == PRELOAD_BASENAME
    assert rows["copies_before_suspension"], rows
    assert rows["copies_after_resume"], rows


if __name__ == "__main__":
    _run_orchestrator()
