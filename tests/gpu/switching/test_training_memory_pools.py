"""Release gate: a training window allocates in one pool and suspension frees it.

A spawned child, with torch_memory_saver preloaded as in serving, trains a synthetic
LoRA-shaped stack through ``TrainingRunner`` under ``MemoryManager``. It reads physical
memory (``cudaMemGetInfo``) and the allocator's reserved bytes per pool
(``memory_snapshot`` by pool id) right after backward. Once the later layers' saved
activations are freed, backward needs a scratch as large as them, as a recurrent
layer's backward window does. A first window pays the one-time costs, such as kernel
loading, the cuBLAS workspaces and the persistent region, so the windows compared
after it are warm. The parent proves on the child's receipt:

- in a control window that restores autograd's device threads (the placement before
  single-threaded backward), backward allocates in the default pool, so the step maps
  a forward and a backward high-water side by side;
- with backward on the calling thread, the step's growth lands in the window's pool:
  the default pool and the provider's own transient pool stay flat, the FP32 master
  gradients live in the provider's persistent pool, and the step maps at least half
  the scratch less;
- after backward, physical memory equals the reserved bytes the pools gained;
- suspension returns the step's memory and leaves nothing in the window's pool, the
  first window's included, since the training stream's cuBLAS workspace is bound
  outside it; a later, smaller window reserves only what its own step needs, and free
  memory stays level across windows.

Not run by the CPU suites; skipped without a CUDA device.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
from contextlib import nullcontext
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from unittest import mock

import pytest

from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT
from tests.support.residency import (
    DEFAULT_POOL,
    MEMORY_TOLERANCE_BYTES,
    PRELOAD_BASENAME,
    TRAINING_PERSISTENT_TAG,
    TRAINING_TRANSIENT_TAG,
    one_rank,
    park_child,
    provider_pool_id,
    receive_from_parked_child,
    require_cuda13_preload_mapping,
    reserved_by_pool,
    send_failure,
    start_child,
    terminate_parked_child,
    window_pool_id,
)
from tests.support.training import forward_backward, host_runner

pytestmark = pytest.mark.checkpoints()

if TYPE_CHECKING:
    import torch

    from cti.memory import MemoryManager
    from cti.training import TrainingBatch
    from cti.training.runner import TrainingRunner

RECEIPT_PREFIX = "CTI_TRAINING_MEMORY_POOLS="
_HIDDEN = 2048
_VOCAB = 1024
_LAYERS = 8
_RANK = 8
_ROW_TOKENS = 1024
_LARGE_ROWS = 4
# Forward saves two [tokens, hidden] activations per layer: the LoRA A input and the
# SiLU input. The backward scratch holds as many tensors of that shape.
_SCRATCH_ACTIVATIONS = 2 * _LAYERS
# The windows in order: a warm-up with autograd's device threads pays both placements'
# one-time costs; then single-pool windows with a large, a small and again a large batch
# around a control with the device threads.
_WINDOWS = (
    ("multithreaded", _LARGE_ROWS),
    ("single_pool", _LARGE_ROWS),
    ("multithreaded", _LARGE_ROWS),
    ("single_pool", 1),
    ("single_pool", _LARGE_ROWS),
)


def _activation_bytes(rows: int) -> int:
    return rows * (_ROW_TOKENS - 1) * _HIDDEN * 2


def _synthetic_model(device: torch.device, backward_threads: list[int]) -> object:
    """A frozen BF16 residual stack with BF16 low-rank adapters and a backward scratch."""

    import torch
    import torch.nn.functional as F

    from tests.support.training import TrainingModelStub

    class BackwardScratch(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            hidden: torch.Tensor,
        ) -> torch.Tensor:
            return hidden.view_as(hidden)

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            gradient: torch.Tensor,
        ) -> torch.Tensor:
            backward_threads.append(threading.get_ident())
            scratch = [torch.empty_like(gradient) for _ in range(_SCRATCH_ACTIVATIONS)]
            for tensor in scratch:
                tensor.copy_(gradient)
            return scratch[-1]

    class SyntheticLoRAStack(TrainingModelStub):
        def __init__(self) -> None:
            generator = torch.Generator().manual_seed(20261005)

            def tensor(*shape: int, scale: float) -> torch.nn.Parameter:
                value = torch.randn(*shape, generator=generator) * scale
                return torch.nn.Parameter(
                    value.to(device=device, dtype=torch.bfloat16),
                    requires_grad=False,
                )

            self.embedding = tensor(_VOCAB, _HIDDEN, scale=1.0)
            self.head = tensor(_VOCAB, _HIDDEN, scale=_HIDDEN**-0.5)
            self.layers = tuple(
                (
                    tensor(_HIDDEN, _HIDDEN, scale=_HIDDEN**-0.5),
                    tensor(_RANK, _HIDDEN, scale=_HIDDEN**-0.5),
                    tensor(_HIDDEN, _RANK, scale=_RANK**-0.5),
                )
                for _ in range(_LAYERS)
            )

        def state_dict(self) -> dict[str, torch.Tensor]:
            tensors = {"embedding": self.embedding, "head": self.head}
            for layer, (weight, _a, _b) in enumerate(self.layers):
                tensors[f"layers.{layer}.weight"] = weight
            tensors.update(self.named_trainable_parameters())
            return tensors

        def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
            return tuple(
                pair
                for layer, (_weight, a, b) in enumerate(self.layers)
                for pair in ((f"layers.{layer}.lora_A", a), (f"layers.{layer}.lora_B", b))
            )

        def training_hidden_states(self, input_ids: torch.Tensor, packing: object) -> torch.Tensor:
            hidden = F.embedding(input_ids, self.embedding)
            for layer, (weight, a, b) in enumerate(self.layers):
                projected = F.linear(hidden, weight) + F.linear(
                    F.linear(hidden, self.trained(a)), self.trained(b)
                )
                hidden = hidden + F.silu(projected)
                if layer == 0:
                    # Its backward runs last, after the later layers freed their activations.
                    hidden = BackwardScratch.apply(hidden)
            return hidden.flatten(0, 1)

        def training_loss(self, hidden_states: torch.Tensor, target_ids: torch.Tensor):
            logits = F.linear(hidden_states, self.head)
            return F.cross_entropy(logits.float(), target_ids, reduction="sum")

    return SyntheticLoRAStack()


def _batch(rows: int) -> TrainingBatch:
    import torch

    from cti.training import SFT, PaddedTokens, TrainingBatch

    generator = torch.Generator().manual_seed(rows)
    return TrainingBatch(
        PaddedTokens(
            torch.randint(1, _VOCAB, (rows, _ROW_TOKENS), generator=generator),
            torch.full((rows,), _ROW_TOKENS, dtype=torch.int64),
            0,
        ),
        torch.ones((rows, _ROW_TOKENS - 1), dtype=torch.bool),
        SFT(),
    )


def _gains(
    before: dict[tuple[int, int], int],
    after: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    return {
        pool: after.get(pool, 0) - before.get(pool, 0)
        for pool in before.keys() | after.keys()
        if after.get(pool, 0) != before.get(pool, 0)
    }


def _run_window(
    runner: TrainingRunner,
    memory_manager: MemoryManager,
    training_stream: torch.cuda.Stream,
    batch: TrainingBatch,
    *,
    multithreaded_backward: bool,
    backward_threads: list[int],
) -> dict[str, object]:
    import torch

    device = training_stream.device
    memory_manager.resume_training(one_rank)
    torch.cuda.synchronize(device)
    free_start = torch.cuda.mem_get_info(device)[0]
    reserved_start = reserved_by_pool()

    # The control restores the placement before backward ran on the calling thread.
    placement = (
        mock.patch.object(
            torch.autograd,
            "set_multithreading_enabled",
            lambda _mode: nullcontext(),
        )
        if multithreaded_backward
        else nullcontext()
    )
    backward_threads.clear()
    with placement, torch.cuda.stream(training_stream):
        with memory_manager.transient_training_allocation(0):
            forward_backward(runner, batch)
    training_stream.synchronize()
    free_after_backward = torch.cuda.mem_get_info(device)[0]
    gains = _gains(reserved_start, reserved_by_pool())
    pool_ids = {
        "window": window_pool_id(memory_manager),
        "persistent": provider_pool_id(TRAINING_PERSISTENT_TAG, device),
        "provider_transient": provider_pool_id(TRAINING_TRANSIENT_TAG, device),
    }
    master_gradient_pools = set(runner.memory_usage().gradients.pools)

    with torch.cuda.stream(training_stream):
        runner.discard()
    training_stream.synchronize()
    memory_manager.suspend_training()
    torch.cuda.synchronize(device)
    free_suspended = torch.cuda.mem_get_info(device)[0]
    residue_bytes = reserved_by_pool().get(pool_ids["window"], 0)

    assert len(backward_threads) == 1
    return {
        "multithreaded_backward": multithreaded_backward,
        "rows": batch.batch_size,
        "backward_on_calling_thread": backward_threads[0] == threading.get_ident(),
        "pool_ids": {name: list(pool) for name, pool in pool_ids.items()},
        "master_gradient_pools": sorted(list(pool) for pool in master_gradient_pools),
        "free_start": free_start,
        "free_after_backward": free_after_backward,
        "free_suspended": free_suspended,
        "physical_step_bytes": free_start - free_after_backward,
        "reserved_gain_bytes": sum(gains.values()),
        "window_gain_bytes": gains.get(pool_ids["window"], 0),
        "provider_transient_gain_bytes": gains.get(pool_ids["provider_transient"], 0),
        "default_gain_bytes": gains.get(DEFAULT_POOL, 0),
        "released_bytes": free_suspended - free_after_backward,
        "residue_bytes": residue_bytes,
    }


def _check_windows(windows: list[dict[str, object]]) -> None:
    """Assert the contract on the receipt the child measured."""

    _warm_up, large, control, small, large_again = windows
    scratch_bytes = _SCRATCH_ACTIVATIONS * _activation_bytes(_LARGE_ROWS)

    assert len({tuple(window["pool_ids"]["window"]) for window in windows}) == len(windows)
    for window in windows:
        # Nothing outlives its window in the window's pool, the warm-up's included,
        # where the training stream's cuBLAS workspace is first used.
        assert window["residue_bytes"] <= MEMORY_TOLERANCE_BYTES, window

    assert not control["backward_on_calling_thread"], control
    assert control["default_gain_bytes"] >= scratch_bytes, control
    assert large["window_gain_bytes"] >= scratch_bytes, large
    assert large["physical_step_bytes"] <= control["physical_step_bytes"] - scratch_bytes // 2, (
        control,
        large,
    )

    for window in (large, small, large_again):
        pool_ids = window["pool_ids"]
        assert window["backward_on_calling_thread"], window
        # The step's growth lands in the window's pool; the master gradients reuse the
        # persistent region.
        assert (
            window["window_gain_bytes"] >= window["physical_step_bytes"] - MEMORY_TOLERANCE_BYTES
        ), window
        assert abs(window["default_gain_bytes"]) <= MEMORY_TOLERANCE_BYTES, window
        assert window["provider_transient_gain_bytes"] == 0, window
        assert window["master_gradient_pools"] == [pool_ids["persistent"]], window
        assert (
            abs(window["physical_step_bytes"] - window["reserved_gain_bytes"])
            <= MEMORY_TOLERANCE_BYTES
        ), window
        # Suspension returns the step's memory.
        assert (
            window["released_bytes"]
            >= window["physical_step_bytes"] - window["residue_bytes"] - MEMORY_TOLERANCE_BYTES
        ), window

    assert small["window_gain_bytes"] <= large["window_gain_bytes"] // 2, (large, small)
    assert (
        abs(large_again["window_gain_bytes"] - large["window_gain_bytes"]) <= MEMORY_TOLERANCE_BYTES
    ), (large, large_again)
    warm_starts = [window["free_start"] for window in windows[1:]]
    assert max(warm_starts) - min(warm_starts) <= MEMORY_TOLERANCE_BYTES, warm_starts


def _run_pools_child(connection: Connection) -> NoReturn:
    provider_bound = False
    try:
        import torch

        from cti.config import (
            AdamWConfig,
            BatchLayout,
            StaticBatching,
            TrainablePlan,
            TrainingBatching,
            TrainingConfig,
        )
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
        backward_threads: list[int] = []
        model = _synthetic_model(device, backward_threads)
        training_stream = torch.cuda.Stream(device)
        runner = host_runner(
            model,
            TrainingConfig(
                TrainablePlan.all_declared(),
                AdamWConfig(),
                TrainingBatching(
                    BatchLayout.PADDED,
                    StaticBatching(microbatch_size=_LARGE_ROWS, max_tokens_per_sample=_ROW_TOKENS),
                ),
            ),
            trainables=model,
            persistent_allocation=memory_manager.persistent_training_allocation,
            stream=training_stream,
            checks=True,
        )
        assert runner.memory_usage().master_weights.payload > 0
        windows = [
            _run_window(
                runner,
                memory_manager,
                training_stream,
                _batch(rows),
                multithreaded_backward=placement == "multithreaded",
                backward_threads=backward_threads,
            )
            for placement, rows in _WINDOWS
        ]
        connection.send(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "preload_mapping": preload_mapping,
                "windows": windows,
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
    process, connection = start_child(_run_pools_child)
    try:
        pools = receive_from_parked_child(process, connection)
    finally:
        exitcode = terminate_parked_child(process, connection)
    assert not torch.cuda.is_initialized()
    print(
        RECEIPT_PREFIX
        + json.dumps({"pools_child_exitcode": exitcode, "pools": pools}, sort_keys=True)
    )


def test_training_window_allocates_in_one_pool_and_suspension_frees_it(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env, log_root = isolated_runtime_env(tmp_path, "training-memory-pools")
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.switching.test_training_memory_pools"),
        log_root / "training-memory-pools.log",
        env=env,
        cwd=REPO_ROOT,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    pools = receipt["pools"]
    assert receipt["pools_child_exitcode"] == -signal.SIGTERM
    assert_cuda_device_metadata(pools)
    assert pools["preload_mapping"] == PRELOAD_BASENAME
    windows = pools["windows"]
    assert [
        ("multithreaded" if window["multithreaded_backward"] else "single_pool", window["rows"])
        for window in windows
    ] == list(_WINDOWS)
    print(f"Training memory pools: {json.dumps(windows, sort_keys=True)}")
    _check_windows(windows)


if __name__ == "__main__":
    _run_orchestrator()
