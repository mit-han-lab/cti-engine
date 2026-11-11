"""GPU contract: checkpointed Qwen3.5 losses and gradients match eager bit for bit.

Cover selected-layer sets, delayed copies and compute,
failure recovery, and training-window residency. Requires CUDA and the serving preload.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import signal
import sys
import weakref
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn
from unittest import mock

import pytest

from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    engine_checks_off,
    fixed_kv_page_capacity,
    offline_child_env,
    pause_device,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import save_random_checkpoint
from tests.support.recompute_audit import (
    SEGMENTS,
    BlockRun,
    assert_recomputed,
    record_block_runs,
)
from tests.support.residency import (
    DEFAULT_POOL,
    MEMORY_TOLERANCE_BYTES,
    PRELOAD_BASENAME,
    TRAINING_PERSISTENT_TAG,
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
from tests.support.training import (
    TrainingDriver,
    clone_main_gradients,
    packed_sft_batch,
    synchronize_optimizer_parameters,
)

pytestmark = pytest.mark.checkpoints()

if TYPE_CHECKING:
    import torch

    from cti.config import ActivationCheckpointMode, TrainablePlan
    from cti.memory import MemoryManager
    from cti.training._boundary_offload import HostBoundaryOffload

RECEIPT_PREFIX = "CTI_QWEN3_5_ACTIVATION_CHECKPOINTING="
CHILD_TIMEOUT_SECONDS = 1_800.0
LAYER_TYPES = ("linear_attention",) * 3 + ("full_attention", "linear_attention", "full_attention")
VOCAB_SIZE = 259
# At most 32 tokens each: one FA2 backward key block, whose gradients are deterministic.
SEQUENCE_LENGTHS = (32, 17, 9)
MAX_TOKENS = 64
ADAPTER_SEED = 20261005
LORA_RANK = 4
LORA_ALPHA = 8.0
# The adapters each set trains, by name, and the first decoder layer that holds one.
TRAINABLE_SETS: dict[str, tuple[Callable[[str], bool], int]] = {
    "all": (lambda _name: True, 0),
    "attention": (lambda name: ".self_attn." in name, 3),
    "last-layer": (lambda name: name.startswith(f"model.layers.{len(LAYER_TYPES) - 1}."), 5),
}
# Two eager runs, the determinism control, then the checkpointed runs.
VARIANTS = (
    ("none", "none"),
    ("none-repeat", "none"),
    ("full", "full"),
    ("full-offload", "full_offload"),
)
HIDDEN_SIZE = 256
# The batching bound: 3 samples of at most 32 tokens.
MICROBATCH_TOKENS = len(SEQUENCE_LENGTHS) * max(SEQUENCE_LENGTHS)
# The hazard worker trains every adapter: eleven offloaded inputs. Each delay spins one
# stream for about 10 ms, longer than a layer of this model takes.
HAZARD_SET = "all"
HAZARD_SLEEP_CYCLES = 20_000_000
# Recomputing the last layer fails after reading its input fetched the one below it.
HAZARD_FAILING_LAYER = len(LAYER_TYPES) - 1
# Layer 4 writes layer 3's input, saved and perhaps still being copied out.
HAZARD_WRITING_LAYER = 4
# The offload's methods that issue a copy, out to the arena and back in.
COPY_METHODS = ("_copy_out", "_copy_in")
# The managed worker's set: a frozen prefix, then checkpointed layers.
MANAGED_SET = "attention"
MANAGED_WINDOWS = 2
# What the persistent training region may reserve beyond the FP32 master gradients.
PERSISTENT_SLACK_BYTES = 64 * 1024 * 1024


def _create_worker(
    model_path: Path,
    plan: TrainablePlan,
    mode: ActivationCheckpointMode,
    memory_manager: MemoryManager | None = None,
) -> Any:
    """A worker with the same seeded adapter as every other run."""

    import torch

    from cti.config import (
        AdamWConfig,
        BatchLayout,
        CacheOptions,
        EngineOptions,
        EngineTuning,
        LoRAConfig,
        ModelOptions,
        SchedulingOptions,
        StaticBatching,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
    from tests.support.gpu import start_rank

    with fixed_kv_page_capacity(8):
        worker = start_rank(
            EngineOptions(
                model=ModelOptions(path=str(model_path), max_seq_len=MAX_TOKENS),
                scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=MAX_TOKENS),
                cache=CacheOptions(prefix="naive"),
                lora=LoRAConfig(
                    targets=Qwen3_5ForConditionalGeneration.LORA_TARGETS,
                    rank=LORA_RANK,
                    alpha=LORA_ALPHA,
                ),
                training=TrainingConfig(
                    trainable_plan=plan,
                    optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
                    batching=TrainingBatching(
                        BatchLayout.PACKED,
                        StaticBatching(
                            microbatch_size=len(SEQUENCE_LENGTHS),
                            max_tokens_per_sample=max(SEQUENCE_LENGTHS),
                        ),
                    ),
                    activation_checkpointing=mode,
                ),
            ),
            EngineTuning(attention_backend="fi", cuda_graph_bs=()),
            memory=memory_manager,
            checks=True,
        )
    runner = worker.training
    adapters = worker.worker.host.adapter
    assert adapters is not None
    torch.manual_seed(ADAPTER_SEED)
    with torch.no_grad():
        for tensor in adapters.state_dict().values():
            tensor.normal_(mean=0.0, std=0.02)
    synchronize_optimizer_parameters(runner, runner.trainable_parameters.named_parameters())
    return worker


@functools.cache
def _driver(worker: Any) -> TrainingDriver:
    """The one scheduler each worker's training runs under in these tests, driven through
    its commands."""

    return TrainingDriver(worker.training)


def _submit_microbatch(worker: Any) -> int:
    """Run one packed SFT forward/backward of the fixed samples; return its id."""

    import torch

    generator = torch.Generator().manual_seed(ADAPTER_SEED)
    samples = tuple(
        tuple(torch.randint(4, VOCAB_SIZE, (length,), generator=generator).tolist())
        for length in SEQUENCE_LENGTHS
    )
    return _driver(worker).forward_backward(packed_sft_batch(samples))


def _run_microbatch(worker: Any) -> tuple[int, list[BlockRun]]:
    """Run one packed SFT forward/backward; its id and its decoder-layer block runs."""

    from cti.models.qwen3_5.model import Qwen3_5DecoderLayer

    layers = worker.training.model.model.layers.op_list
    with record_block_runs(Qwen3_5DecoderLayer, layers) as runs:
        submission_id = _submit_microbatch(worker)
    return submission_id, runs


def _observe(worker: Any, submission_id: int, runs: list[BlockRun]) -> dict[str, object]:
    """The microbatch's loss, the selected adapters' FP32 gradients, the policy's first
    layer and the audited block runs."""

    runner = worker.training
    (observation,) = _driver(worker).observe(submission_id)
    completion = observation.completion
    selected = runner.trainable_parameters.named_parameters()
    gradients = clone_main_gradients(runner, selected)
    policy = runner.model.model._activation_checkpoint
    offload = runner._boundary_offload
    return {
        "arena_bytes": None if offload is None else offload.nbytes,
        "block_runs": [tuple(run) for run in runs],
        "first": None if policy is None else policy.first,
        "gradients": {
            name: gradient.cpu() for (name, _), gradient in zip(selected, gradients, strict=True)
        },
        "loss_sum": completion.loss_sum,
    }


def _train_step(
    model_path: Path,
    plan: TrainablePlan,
    mode: ActivationCheckpointMode,
) -> dict[str, object]:
    """One step on a fresh unmanaged worker, closed afterwards."""

    import torch

    worker = _create_worker(model_path, plan, mode)
    try:
        return _observe(worker, *_run_microbatch(worker))
    finally:
        worker.close()
        torch.cuda.empty_cache()


@contextmanager
def _before_copies(
    before: Callable[[HostBoundaryOffload, str, torch.cuda.Stream], None],
) -> Iterator[None]:
    """Call ``before(offload, method, stream)`` as the offload issues each copy, out and
    in; ``stream`` is the training stream the copy is ordered after."""

    from cti.training._boundary_offload import HostBoundaryOffload

    def wrapped(name: str, copy: Callable[..., None]) -> Callable[..., None]:
        def issue(offload: HostBoundaryOffload, saved: object, stream: torch.cuda.Stream) -> None:
            before(offload, name, stream)
            copy(offload, saved, stream)

        return issue

    with ExitStack() as stack:
        for name in COPY_METHODS:
            original = getattr(HostBoundaryOffload, name)
            stack.enter_context(
                mock.patch.object(HostBoundaryOffload, name, wrapped(name, original))
            )
        yield


@contextmanager
def _wrapped_blocks(before: Callable[[object, object], None]) -> Iterator[None]:
    """Call ``before(layer, hidden_states)`` ahead of every decoder layer's segments."""

    from cti.models.qwen3_5.model import Qwen3_5DecoderLayer

    def wrapped(segment: Callable[..., object]) -> Callable[..., object]:
        def run(layer: object, hidden_states: object, *args: object, **kwargs: object) -> object:
            before(layer, hidden_states)
            return segment(layer, hidden_states, *args, **kwargs)

        return run

    with ExitStack() as stack:
        for name in SEGMENTS:
            segment = getattr(Qwen3_5DecoderLayer, name)
            stack.enter_context(mock.patch.object(Qwen3_5DecoderLayer, name, wrapped(segment)))
        yield


def _train_hazards(model_path: Path, plan: TrainablePlan) -> dict[str, object]:
    """Train one ``full_offload`` worker through the offload's hazards, then close it.

    Its microbatches run unaudited: the audit copies each block's output to the host,
    which drains the training stream after every block, so a delayed layer could never
    leave the training stream behind the copies.
    """

    import torch

    from cti.config import ActivationCheckpointMode

    worker = _create_worker(model_path, plan, ActivationCheckpointMode.FULL_OFFLOAD)
    runner = worker.training
    offload = runner._boundary_offload
    assert offload is not None
    layers = runner.model.model.layers.op_list
    indices = {id(layer): index for index, layer in enumerate(layers)}
    results: dict[str, dict[str, object]] = {}
    failures: dict[str, str] = {}
    allocated: list[int] = []
    default_reserved: list[int] = []
    # By copy method, whether the training stream was still busy as each copy was issued.
    lagging: dict[str, list[bool]] = {name: [] for name in COPY_METHODS}
    # The device's allocated bytes as each layer starts its checkpointed forward.
    forward_allocated: list[int] = []

    def arena() -> dict[str, object]:
        return {
            "nbytes": offload.nbytes,
            "address": offload._slots[0].data_ptr(),
            "pinned": offload._slots[0].is_pinned(),
        }

    def settle() -> None:
        gc.collect()
        torch.cuda.synchronize(runner.device)
        allocated.append(torch.cuda.memory_allocated(runner.device))
        default_reserved.append(reserved_by_pool().get(DEFAULT_POOL, 0))

    def train(name: str) -> None:
        submission_id = _submit_microbatch(worker)
        results[name] = _observe(worker, submission_id, [])
        _driver(worker).discard()
        settle()

    def fail(name: str) -> None:
        try:
            _driver(worker).observe(_submit_microbatch(worker))
        except RuntimeError as error:
            failures[name] = str(error)
        else:
            raise AssertionError(f"the {name} microbatch did not fail")
        assert offload._saved == [], name
        settle()

    def delay_copy(offload: HostBoundaryOffload, _name: str, _stream: object) -> None:
        with torch.cuda.stream(offload._copy_stream):
            pause_device(HAZARD_SLEEP_CYCLES)

    def sleep_first(_layer: object, _hidden_states: object) -> None:
        pause_device(HAZARD_SLEEP_CYCLES)

    def record_lag(_offload: object, name: str, stream: torch.cuda.Stream) -> None:
        lagging[name].append(not stream.query())

    def fail_recompute(layer: object, _hidden_states: object) -> None:
        if layer is layers[HAZARD_FAILING_LAYER] and torch.is_grad_enabled():
            raise RuntimeError("injected recompute failure")

    # Weak: the test must not keep an input alive past its microbatch.
    inputs: list[weakref.ref[torch.Tensor]] = []

    def write_previous_input(layer: object, hidden_states: torch.Tensor) -> None:
        if torch.is_grad_enabled():
            return
        # A row-wise segment runs on row tiles, views of its saved input.
        base = hidden_states if hidden_states._base is None else hidden_states._base
        inputs.append(weakref.ref(base))
        # At the writing layer's first segment, write the segment input saved before it.
        if indices[id(layer)] == HAZARD_WRITING_LAYER and len(inputs) % len(SEGMENTS) == 1:
            previous = inputs[-2]()
            assert previous is not None
            previous.add_(1)

    def record_allocated(_layer: object, _hidden_states: object) -> None:
        if not torch.is_grad_enabled():
            forward_allocated.append(torch.cuda.memory_allocated(runner.device))

    try:
        arena_before = arena()
        train("clean")
        with _before_copies(delay_copy):
            train("copy-delay")
        # Checks off: under them cti-kernels validates cu_seqlens with a host read.
        with engine_checks_off(), _wrapped_blocks(sleep_first), _before_copies(record_lag):
            train("training-delay")
        with _before_copies(delay_copy), _wrapped_blocks(fail_recompute):
            fail("recompute-failure")
        with _wrapped_blocks(write_previous_input):
            fail("in-place-write")
        # Measured last: whatever a first microbatch allocates lazily exists by now.
        with _wrapped_blocks(record_allocated):
            train("clean-again")
        return {
            "results": results,
            "failures": failures,
            "allocated_bytes": allocated,
            "default_reserved_bytes": default_reserved,
            "lagging": lagging,
            "forward_allocated_bytes": forward_allocated,
            "arena": [arena_before, arena()],
        }
    finally:
        worker.close()
        torch.cuda.empty_cache()


def _train_windows(
    worker: Any,
    memory_manager: MemoryManager,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """One step in each training window of a worker under ``memory_manager``, and each
    window's pool measurements."""

    import torch

    from cti.memory import Mode

    runner = worker.training
    master_bindings = runner._optimizer._master_bindings
    assert master_bindings
    results = []
    windows = []
    for _ in range(MANAGED_WINDOWS):
        worker.worker.quiesce()
        worker.worker.release(Mode.SERVING)
        worker.worker.acquire(Mode.TRAINING)
        runner.resume()  # as the scheduler does once training holds the device
        reserved_start = reserved_by_pool()
        submission_id, runs = _run_microbatch(worker)
        reserved_after = reserved_by_pool()
        pool_ids = {
            "window": window_pool_id(memory_manager),
            "persistent": provider_pool_id(TRAINING_PERSISTENT_TAG, runner.device),
        }
        master_gradient_bytes = sum(binding.master.grad.nbytes for binding in master_bindings)
        results.append(_observe(worker, submission_id, runs))
        _driver(worker).discard()
        runner.suspend()  # as the scheduler does before the worker moves its memory
        worker.worker.release(Mode.TRAINING)
        torch.cuda.synchronize(runner.device)
        windows.append(
            {
                "pool_ids": {name: list(pool) for name, pool in pool_ids.items()},
                "default_gain_bytes": (
                    reserved_after.get(DEFAULT_POOL, 0) - reserved_start.get(DEFAULT_POOL, 0)
                ),
                "window_reserved_bytes": reserved_after.get(pool_ids["window"], 0),
                "persistent_reserved_bytes": reserved_after.get(pool_ids["persistent"], 0),
                "master_gradient_bytes": master_gradient_bytes,
                "residue_bytes": reserved_by_pool().get(pool_ids["window"], 0),
            }
        )
        worker.worker.acquire(Mode.SERVING)
    return results, windows


def _run_child(connection: Connection, workdir: str) -> NoReturn:
    provider_bound = False
    try:
        import torch

        import cti
        from cti.config import ActivationCheckpointMode, TrainablePlan
        from cti.memory import MemoryManager
        from tests.support.gpu import configure_worker_process

        configure_worker_process()
        from cti.memory import unified_memory

        memory_manager = MemoryManager.from_stock_provider(
            unified=unified_memory(torch.device("cuda"))
        )
        provider_bound = True
        preload_mapping = require_cuda13_preload_mapping()
        model_path = Path(workdir) / "qwen3_5"
        save_random_checkpoint(
            model_path,
            hidden_size=HIDDEN_SIZE,
            layer_types=LAYER_TYPES,
            vocab_size=VOCAB_SIZE,
        )
        plans: dict[str, TrainablePlan] = {}
        # By trainable set and variant.
        results: dict[str, dict[str, dict[str, object]]] = {}
        for trainable_set, (trains, _first) in TRAINABLE_SETS.items():
            # One plan per set: the names the "all" run declared.
            if trainable_set == "all":
                plan = TrainablePlan.all_declared()
            else:
                declared = results["all"]["none"]["gradients"]
                assert isinstance(declared, dict)
                plan = TrainablePlan(tuple(name for name in declared if trains(name)))
            plans[trainable_set] = plan
            results[trainable_set] = {
                variant: _train_step(model_path, plan, ActivationCheckpointMode(mode))
                for variant, mode in VARIANTS
            }
        hazards = _train_hazards(model_path, plans[HAZARD_SET])
        # Provider-bound, this worker stays open until the parent terminates the child.
        managed_worker = _create_worker(
            model_path,
            plans[MANAGED_SET],
            ActivationCheckpointMode.FULL_OFFLOAD,
            memory_manager,
        )
        managed, windows = _train_windows(managed_worker, memory_manager)

        torch.save(
            {"sets": results, "managed": managed, "hazards": hazards["results"]},
            Path(workdir) / "results.pt",
        )
        connection.send(
            {
                "capability": list(torch.cuda.get_device_capability()),
                "device": torch.cuda.get_device_name(),
                "first": {
                    trainable_set: runs["full"]["first"] for trainable_set, runs in results.items()
                },
                "hazards": {name: value for name, value in hazards.items() if name != "results"},
                "managed_windows": windows,
                "preload_mapping": preload_mapping,
                "source_root": str(Path(cti.__file__).resolve().parents[2]),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            }
        )
    except BaseException:  # noqa: BLE001 - isolated child reports complete diagnostics
        send_failure(connection)
    finally:
        if provider_bound:
            park_child()
        connection.close()
    raise RuntimeError("unreachable")


def _run_orchestrator(workdir: Path) -> None:
    import torch

    assert not torch.cuda.is_initialized()
    process, connection = start_child(_run_child, str(workdir))
    try:
        child = receive_from_parked_child(
            process,
            connection,
            timeout_seconds=CHILD_TIMEOUT_SECONDS,
        )
    finally:
        exitcode = terminate_parked_child(process, connection)
    assert not torch.cuda.is_initialized()
    print(RECEIPT_PREFIX + json.dumps({"child_exitcode": exitcode, "child": child}, sort_keys=True))


def _assert_trains_like(
    run: dict[str, object],
    eager: dict[str, object],
    trainable_set: str,
) -> None:
    """``run`` has ``eager``'s loss and selected gradients, bit for bit."""

    import torch

    assert run["loss_sum"] == eager["loss_sum"], trainable_set
    assert tuple(run["gradients"]) == tuple(eager["gradients"]), trainable_set
    for name, gradient in eager["gradients"].items():
        torch.testing.assert_close(
            run["gradients"][name],
            gradient,
            rtol=0,
            atol=0,
            msg=f"{trainable_set}: {name}",
        )


def _assert_managed_windows(windows: list[dict[str, object]]) -> None:
    """Assert the pool contract on the managed worker's window measurements."""

    assert len(windows) == MANAGED_WINDOWS
    assert len({tuple(window["pool_ids"]["window"]) for window in windows}) == len(windows)
    for window in windows:
        # Forward, recomputation and backward allocate in the window's pool, the FP32
        # master gradients in the persistent region.
        assert window["window_reserved_bytes"] > 0, window
        master_gradient_bytes = window["master_gradient_bytes"]
        assert master_gradient_bytes > 0, window
        assert (
            master_gradient_bytes
            <= window["persistent_reserved_bytes"]
            <= master_gradient_bytes + PERSISTENT_SLACK_BYTES
        ), window
        # Suspension frees the window's pool.
        assert window["residue_bytes"] <= MEMORY_TOLERANCE_BYTES, window
    # The first window may bind the training stream's cuBLAS workspace in the default
    # pool; warm, a microbatch allocates nothing there.
    assert abs(windows[-1]["default_gain_bytes"]) <= MEMORY_TOLERANCE_BYTES, windows


def _assert_hazards(hazards: dict[str, object], results: dict[str, dict[str, object]]) -> None:
    """Assert the hazard worker's contract; ``results`` are its completed microbatches."""

    assert tuple(results) == ("clean", "copy-delay", "training-delay", "clean-again")
    failures = hazards["failures"]
    assert failures.keys() == {"recompute-failure", "in-place-write"}, failures
    assert "injected recompute failure" in failures["recompute-failure"], failures
    assert "modified in place after it was saved" in failures["in-place-write"], failures
    # The delayed layers keep the training stream busy as each input is copied out and
    # as each fetch is issued after a delayed recomputation, so only the copies' waits
    # order them after it. The first fetch follows the loss's backward, which no layer
    # delays.
    checkpoints = len(SEGMENTS) * len(LAYER_TYPES)
    offloaded = checkpoints - 1
    lagging = hazards["lagging"]
    assert lagging["_copy_out"] == [True] * offloaded, lagging
    assert len(lagging["_copy_in"]) == offloaded and all(lagging["_copy_in"][1:]), lagging
    # Six settles: after every microbatch, failed or not, the same bytes are allocated,
    # and once the first has warmed the default pool, it reserves the same segments.
    allocated, reserved = hazards["allocated_bytes"], hazards["default_reserved_bytes"]
    assert len(allocated) == len(reserved) == 6, hazards
    assert len(set(allocated)) == 1 and len(set(reserved[1:])) == 1, hazards
    # Each segment's input is freed one segment after it was saved: as a segment starts,
    # at most its predecessor's input and the decoder input are allocated beside its own.
    input_bytes = sum(SEQUENCE_LENGTHS) * HIDDEN_SIZE * 2
    forward = hazards["forward_allocated_bytes"]
    assert len(forward) == checkpoints, forward
    assert max(forward) - forward[0] < 3 * input_bytes, forward
    before, after = hazards["arena"]
    assert before == after and before["pinned"], hazards["arena"]
    assert before["nbytes"] == offloaded * MICROBATCH_TOKENS * HIDDEN_SIZE * 2


def test_checkpointing_trains_qwen3_5_bitwise_like_eager(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env, log_root = offline_child_env(tmp_path, "qwen3-5-activation-checkpointing")
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.training.test_qwen3_5_activation_checkpointing",
            "--workdir",
            str(tmp_path),
        ),
        log_root / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=CHILD_TIMEOUT_SECONDS + 60,
    )
    receipt = child_receipt(output, RECEIPT_PREFIX)
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))
    child = receipt["child"]
    assert receipt["child_exitcode"] == -signal.SIGTERM
    assert_cuda_device_metadata(child)
    assert child["preload_mapping"] == PRELOAD_BASENAME
    assert Path(child["source_root"]) == REPO_ROOT
    results = torch.load(tmp_path / "results.pt", map_location="cpu", weights_only=True)

    sets = results["sets"]
    assert sets.keys() == TRAINABLE_SETS.keys()
    assert len(results["managed"]) == MANAGED_WINDOWS
    itemsize = torch.bfloat16.itemsize
    for trainable_set, (trains, first) in TRAINABLE_SETS.items():
        label = trainable_set
        eager, repeat, checkpointed, offloaded = (
            sets[trainable_set][variant] for variant, _ in VARIANTS
        )
        managed = results["managed"] if trainable_set == MANAGED_SET else []
        hazards = results["hazards"].values() if trainable_set == HAZARD_SET else ()
        assert eager["first"] is None and repeat["first"] is None
        names = tuple(eager["gradients"])
        assert names and all(trains(name) for name in names), label
        _assert_trains_like(repeat, eager, label)
        for run in (checkpointed, offloaded, *managed, *hazards):
            assert run["first"] == first, label
            _assert_trains_like(run, eager, label)
        # The hazard microbatches ran unaudited.
        # Plain checkpoints recompute whole layers, offloaded ones single segments.
        for run in (checkpointed, offloaded, *managed):
            assert_recomputed(
                [BlockRun(*block_run) for block_run in eager["block_runs"]],
                [BlockRun(*block_run) for block_run in run["block_runs"]],
                first=first,
                by_segment=run is not checkpointed,
            )
        assert all(run["arena_bytes"] is None for run in (eager, repeat, checkpointed))
        offloaded_inputs = len(SEGMENTS) * (len(LAYER_TYPES) - first) - 1
        arena_bytes = offloaded_inputs * MICROBATCH_TOKENS * HIDDEN_SIZE
        for run in (offloaded, *managed, *hazards):
            assert run["arena_bytes"] == arena_bytes * itemsize, label
    _assert_hazards(child["hazards"], results["hazards"])
    _assert_managed_windows(child["managed_windows"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    _run_orchestrator(parser.parse_args().workdir)
