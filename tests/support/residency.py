"""Child-process and physical-memory helpers of the GPU residency release gates.

Each gate runs its model in a spawned child that publishes one receipt and then parks,
so the CUDA context and its memory-saver mappings stay alive until the parent
terminates it. Physical memory is observed through ``cudaMemGetInfo``, and the
allocator's reserved bytes per memory pool through its segment snapshot.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import traceback
from collections.abc import Callable
from enum import Enum
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    import torch

    from cti.kvcache import PageGroup
    from cti.memory import MemoryManager

CHILD_TIMEOUT_SECONDS = 300.0
MEMORY_TOLERANCE_BYTES = 8 * 1024 * 1024
# A cold kernel cache loads each kernel module lazily on its first launch, about 2 MiB a
# module, outside any allocator and not a stack growth; the first switch may load some.
FIRST_SWITCH_LOAD_BYTES = 32 * 1024 * 1024
# In the formal PyTorch/CUDA stack, independently live large allocations receive
# 2 MiB-granular backing segments. Model this physical cost explicitly so the memory
# observation tolerance cannot hide a missing layer release.
CUDA_LARGE_ALLOCATION_THRESHOLD_BYTES = 10 * 1024 * 1024
CUDA_LARGE_SEGMENT_BYTES = 2 * 1024 * 1024
PRELOAD_BASENAME = "torch_memory_saver_hook_mode_preload_cu13.abi3.so"
DEFAULT_POOL = (0, 0)
TRAINING_PERSISTENT_TAG = "training-persistent"
TRAINING_TRANSIENT_TAG = "training-transient"


def one_rank(spare_bytes: int) -> int:
    """A single rank's vote on a resume: the spare bytes its own device keeps."""

    return spare_bytes


class PhysicalMemoryContract(Enum):
    EXACT = "exact"
    LAYER_RESOLUTION = "layer_resolution"

    @classmethod
    def for_device(cls, device: torch.device) -> PhysicalMemoryContract:
        import torch

        properties = torch.cuda.get_device_properties(device)
        return cls.LAYER_RESOLUTION if properties.is_integrated else cls.EXACT

    def assert_transition(
        self,
        name: str,
        observed_bytes: int,
        expected_bytes: int,
        resolution_bytes: int,
    ) -> None:
        if self is PhysicalMemoryContract.EXACT:
            assert abs(observed_bytes - expected_bytes) <= MEMORY_TOLERANCE_BYTES, (
                f"{name} mismatch: expected={expected_bytes}, observed={observed_bytes}, "
                f"tolerance={MEMORY_TOLERANCE_BYTES}"
            )
            return

        # cudaMemGetInfo on an integrated GPU observes a unified system pool, so
        # unrelated allocator and OS activity can move either endpoint. Pointer
        # identity plus full-cache writes prove that every layer mapping works;
        # this measurement independently rejects a missing complete KV layer.
        delta_bytes = abs(observed_bytes - expected_bytes)
        assert delta_bytes < resolution_bytes, (
            f"{name} exceeded integrated-memory measurement resolution: "
            f"expected={expected_bytes}, observed={observed_bytes}, "
            f"delta={delta_bytes}, resolution={resolution_bytes}"
        )

    def assert_drift(
        self,
        name: str,
        delta_bytes: int,
        smallest_layer_bytes: int,
    ) -> None:
        if self is PhysicalMemoryContract.EXACT:
            assert delta_bytes <= MEMORY_TOLERANCE_BYTES, (
                f"{name} drifted by {delta_bytes} bytes; tolerance={MEMORY_TOLERANCE_BYTES}"
            )
            return

        # Integrated-memory accounting is approximate. Keep a meaningful
        # boundary: drift must remain smaller than one complete KV layer.
        assert delta_bytes < smallest_layer_bytes, (
            f"{name} drifted by {delta_bytes} bytes; "
            f"one_layer={smallest_layer_bytes}, contract={self.value}"
        )

    def assert_stable_levels(
        self,
        name: str,
        levels: list[int],
        smallest_layer_bytes: int,
    ) -> None:
        delta_bytes = max(levels) - min(levels)
        self.assert_drift(f"{name} (levels {levels})", delta_bytes, smallest_layer_bytes)


def assert_warmed_training_windows(
    contract: PhysicalMemoryContract,
    warmup: dict[str, Any],
    windows: list[dict[str, Any]],
    smallest_layer_bytes: int,
) -> None:
    """Exclude cold training capacity only; its first release must still be stable.

    A live window pool retains first-use tuning scratch until suspension. The measured
    windows hold the steady state; the cold window's inference level may differ from
    theirs only by the kernel modules its first switch loaded, so a warm-up cannot
    conceal memory that should have been released on exit.
    """

    contract.assert_stable_levels(
        "inference-phase free memory",
        [window["free_inference_bytes"] for window in windows],
        smallest_layer_bytes,
    )
    first_bytes = abs(warmup["free_inference_bytes"] - windows[0]["free_inference_bytes"])
    assert first_bytes <= FIRST_SWITCH_LOAD_BYTES, (
        f"inference-phase free memory moved by {first_bytes} bytes across the first switch; "
        f"lazily loaded kernel modules account for at most {FIRST_SWITCH_LOAD_BYTES}"
    )
    contract.assert_stable_levels(
        "training-phase free memory",
        [window["free_training_bytes"] for window in windows],
        smallest_layer_bytes,
    )


def park_child() -> NoReturn:
    while True:
        signal.pause()


def require_cuda13_preload_mapping() -> str:
    mappings = Path("/proc/self/maps").read_text(encoding="utf-8")
    if PRELOAD_BASENAME not in mappings:
        raise RuntimeError(f"{PRELOAD_BASENAME} is not mapped in the GPU child")
    return PRELOAD_BASENAME


def tensor_metadata(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "object_id": id(tensor),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_offset": tensor.storage_offset(),
        "nbytes": tensor.nbytes,
        "data_ptr": tensor.data_ptr(),
    }


def physical_storage_nbytes(storage_nbytes: int) -> int:
    return (
        (storage_nbytes + CUDA_LARGE_SEGMENT_BYTES - 1)
        // CUDA_LARGE_SEGMENT_BYTES
        * CUDA_LARGE_SEGMENT_BYTES
    )


def poison_kv_cache(group: PageGroup, value: float = float("nan")) -> None:
    for layer in group.layers:
        layer.fill_(value)


def snapshot_tensors(
    named_tensors: tuple[tuple[str, torch.Tensor], ...],
) -> tuple[tuple[str, int, int, torch.Tensor], ...]:
    return tuple(
        (name, id(tensor), tensor.data_ptr(), tensor.detach().cpu().clone())
        for name, tensor in named_tensors
    )


def assert_tensor_snapshot(
    named_tensors: tuple[tuple[str, torch.Tensor], ...],
    snapshot: tuple[tuple[str, int, int, torch.Tensor], ...],
) -> None:
    import torch

    assert len(named_tensors) == len(snapshot)
    for (name, tensor), (expected_name, object_id, data_ptr, value) in zip(
        named_tensors,
        snapshot,
        strict=True,
    ):
        assert name == expected_name
        assert id(tensor) == object_id
        assert tensor.data_ptr() == data_ptr
        assert torch.equal(tensor.detach().cpu(), value)


def reserved_by_pool() -> dict[tuple[int, int], int]:
    """The caching allocator's reserved bytes by memory pool id."""

    return _bytes_by_pool("total_size")


def allocated_by_pool() -> dict[tuple[int, int], int]:
    """The caching allocator's allocated bytes by memory pool id."""

    return _bytes_by_pool("allocated_size")


def window_pool_id(memory_manager: MemoryManager) -> tuple[int, int]:
    """Read the training window pool's id; a reference kept past suspension would keep it
    mapped."""

    pool = memory_manager._training._transient_pool
    assert pool is not None
    return tuple(pool.id)


def provider_pool_id(tag: str, device: torch.device) -> tuple[int, int]:
    """The id of the pool torch_memory_saver keeps for one no-backup region tag."""

    from torch_memory_saver import torch_memory_saver

    pool = torch_memory_saver._impl._mem_pools.get((tag, False, device.index))
    assert pool is not None, f"torch_memory_saver has no pool for {tag}"
    return tuple(pool.id)


def send_failure(connection: Connection) -> None:
    connection.send({"error": traceback.format_exc()})


def start_child(
    target: Callable[..., NoReturn],
    *args: object,
) -> tuple[mp.Process, Connection]:
    """Spawn ``target(connection, *args)`` under the memory saver's preload."""

    from cti.memory import configure_memory_saver_subprocess

    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(target=target, args=(child_connection, *args), daemon=False)
    try:
        with configure_memory_saver_subprocess():
            process.start()
    except BaseException:
        parent_connection.close()
        child_connection.close()
        raise
    child_connection.close()
    return process, parent_connection


def receive_from_parked_child(
    process: mp.Process,
    connection: Connection,
    *,
    timeout_seconds: float = CHILD_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if not connection.poll(timeout_seconds):
        raise TimeoutError(f"GPU child {process.pid} did not publish a receipt")
    message = connection.recv()
    if "error" in message:
        raise RuntimeError(f"GPU child failed:\n{message['error']}")
    if not process.is_alive():
        raise RuntimeError(f"GPU child {process.pid} exited instead of parking")
    return message


def terminate_parked_child(process: mp.Process, connection: Connection) -> int:
    connection.close()
    process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=10)
    if process.is_alive():
        raise RuntimeError(f"GPU child {process.pid} resisted termination")
    exitcode = process.exitcode
    if exitcode is None:
        raise RuntimeError(f"GPU child {process.pid} has no terminal exit code")
    process.close()
    return exitcode


def _bytes_by_pool(field: str) -> dict[tuple[int, int], int]:
    import torch

    by_pool: dict[tuple[int, int], int] = {}
    for segment in torch.cuda.memory_snapshot(include_traces=False):
        pool = tuple(segment["segment_pool_id"])
        by_pool[pool] = by_pool.get(pool, 0) + segment[field]
    return by_pool
