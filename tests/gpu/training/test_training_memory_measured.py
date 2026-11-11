"""The startup training-memory plan against one measured window.

Each case builds a ``Rank`` the normal way (its ``InferenceRunner`` plans training
memory from the built model), then runs two windows of one microbatch that fills the
configured ``max_tokens_per_microbatch``, forward/backward and the optimizer step, and
measures them against the plan ``plan_model_training_memory`` makes of the built model:

- master weights, AdamW moments, the FP32 gradients and the SFT staging buffers: the
  bytes of the allocator segments only that component's live blocks occupy, plus its own
  blocks in segments it shares with other allocations (the basis of
  ``tests/modules/training/test_training_memory_plan.py``'s ``_MEASURED`` table), and their
  payload bytes;
- activations: the requested (payload) peak over the microbatch above its start; the
  allocator's block rounding and unsplit remainders belong to the segments the plan adds;
- the window's scope: the reserved peak above the start after ``empty_cache``, without the
  segments the gradients map, against ``scope_bytes``; likewise for the optimizer step.

The worker runs without a memory manager. The runner's persistent allocations (FP32
gradients and folded evidence) go to their own ``MemPool``, as production keeps them in
the persistent region, apart from the microbatch's temporaries.

The plan bounds every component from above (``TrainingMemoryPlan``, ``scope_bytes``, and
the accuracy table's contract); a case fails only after its receipt is printed, when the
plan undercounts. ``python -m tests.gpu.training.test_training_memory_measured --summarize
LOG`` turns the receipts in a pytest ``-s`` log into a table and ``_MEASURED`` rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.support.checkpoints import QWEN3_0_6B, QWEN3_5_9B, QWEN3_8B, Checkpoint
from tests.support.gpu import (
    child_receipt,
    fixed_kv_page_capacity,
    offline_child_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT, source_pythonpath

if TYPE_CHECKING:
    import torch

    from cti.training import TrainingBatch
    from cti.training.runner import TrainingRunner

RECEIPT_PREFIX = "CTI_TRAINING_MEMORY_MEASURED="
CHILD_TIMEOUT_SECONDS = 900.0
SAMPLE_SEED = 20261018
# Below this many tokens the accumulation's FP32 gradients, which the plan counts apart,
# can set the microbatch's allocated peak (the accuracy table starts here).
ACTIVATION_ONLY_BOUND_TOKENS = 1024


@dataclass(frozen=True)
class _Case:
    """One training configuration: ``sequences`` equal rows fill ``tokens``."""

    checkpoint: Checkpoint
    rank: int
    checkpointing: str
    tokens: int
    layout: str = "padded"
    dynamic: bool = False
    sequences: int = 2
    staged: bool = False
    # Device memory the case needs; smaller devices skip it.
    min_device_gib: int = 0

    @property
    def label(self) -> str:
        return self.checkpoint.repo_id.split("/")[1]

    @property
    def id(self) -> str:
        batching = "dynamic" if self.dynamic else "static"
        staging = "-staged" if self.staged else ""
        return (
            f"{self.label}-r{self.rank}-{self.checkpointing}-{self.tokens}-"
            f"{self.layout}-{batching}{self.sequences}{staging}"
        )

    @property
    def table_key(self) -> bool:
        """Whether the case is a row of the accuracy table: padded, two static rows, no
        staging."""

        return (
            self.layout == "padded" and not self.dynamic and self.sequences == 2 and not self.staged
        )


CASES = (
    # The accuracy table's rows (tests/modules/training/test_training_memory_plan.py).
    _Case(QWEN3_8B, 16, "none", 1024),
    _Case(QWEN3_8B, 16, "none", 8192, min_device_gib=64),
    _Case(QWEN3_8B, 16, "full_offload", 1024),
    _Case(QWEN3_8B, 16, "full_offload", 8192),
    _Case(QWEN3_5_9B, 16, "none", 1024),
    _Case(QWEN3_5_9B, 16, "none", 8192, min_device_gib=64),
    _Case(QWEN3_5_9B, 16, "full_offload", 1024),
    _Case(QWEN3_5_9B, 16, "full_offload", 8192),
    _Case(QWEN3_5_9B, 4, "none", 1024),
    # Packed rows under a dynamic budget, recomputed.
    _Case(QWEN3_8B, 16, "full", 2048, layout="packed", dynamic=True, sequences=4),
    _Case(QWEN3_5_9B, 16, "full", 4096, layout="packed", dynamic=True, sequences=4),
    # The tied LM head, staged SFT inputs, a short microbatch and many packed rows.
    _Case(QWEN3_0_6B, 16, "none", 1024, staged=True),
    _Case(QWEN3_0_6B, 16, "none", 2048, layout="packed", staged=True),
    _Case(QWEN3_0_6B, 16, "full", 512),
    _Case(QWEN3_0_6B, 4, "full_offload", 8192, layout="packed", dynamic=True, sequences=8),
)


def _case(case_id: str) -> _Case:
    return next(case for case in CASES if case.id == case_id)


# ================================== Measurement ==================================


def _held_bytes(snapshot: list[dict[str, Any]], tensors: list[torch.Tensor]) -> int:
    """Bytes ``tensors``' storages hold: the large segments whose live blocks are all theirs,
    and their own blocks elsewhere: in small segments, and in segments they share, where
    earlier allocations, such as inference's small blocks, left room they fill."""

    owned = {tensor.untyped_storage().data_ptr() for tensor in tensors}
    total = 0
    for segment in snapshot:
        live = [block for block in segment["blocks"] if block["state"] == "active_allocated"]
        mine = [block for block in live if block["address"] in owned]
        if not mine:
            continue
        if segment["segment_type"] == "large" and len(mine) == len(live):
            total += segment["total_size"]
        else:
            total += sum(block["size"] for block in mine)
    return total


def _payload_bytes(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.untyped_storage().nbytes() for tensor in tensors)


def _state_tensors(runner: TrainingRunner) -> dict[str, list[torch.Tensor]]:
    """Each planned state component's live device tensors, after a microbatch."""

    optimizer = runner._optimizer
    update_parameters = optimizer._update_parameters
    moments = [
        optimizer._torch_optimizer.state[parameter][moment]
        for parameter in update_parameters
        for moment in ("exp_avg", "exp_avg_sq")
    ]
    staging = runner._staging
    staging_values = (
        ()
        if staging is None
        else (
            staging._fixed,
            *staging._buffers,
            *(() if staging._packing is None else (staging._packing.positions,)),
            *(() if staging._packing is None else (staging._packing.cu_seqlens,)),
        )
    )
    return {
        "master_weights": [binding.master for binding in optimizer._master_bindings],
        "optimizer_state": moments,
        "gradients": [
            parameter.grad for parameter in update_parameters if parameter.grad is not None
        ],
        "input_staging": [value for value in staging_values if value is not None and value.is_cuda],
    }


def _pool_bytes(pool: torch.cuda.MemPool) -> int:
    return sum(segment["total_size"] for segment in pool.snapshot(include_traces=False))


def _measure_window(
    runner: TrainingRunner, batch: TrainingBatch, persistent: torch.cuda.MemPool
) -> tuple[dict[str, int], dict[str, dict[str, int]] | None]:
    """One forward/backward and optimizer step from an emptied cache: each command's
    requested, allocated and reserved peaks above its start, and the state components' segments
    while the gradients live."""

    import torch

    from tests.support.training import forward_backward, optim_step

    def start() -> tuple[int, int]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

    allocated, reserved = start()
    requested = torch.cuda.memory_stats()["requested_bytes.all.current"]
    pool_before = _pool_bytes(persistent)
    forward_backward(runner, batch)
    torch.cuda.synchronize()
    persistent_growth = _pool_bytes(persistent) - pool_before
    window = {
        "fb_requested_peak": torch.cuda.memory_stats()["requested_bytes.all.peak"] - requested,
        "fb_allocated_peak": torch.cuda.max_memory_allocated() - allocated,
        "fb_reserved_peak": torch.cuda.max_memory_reserved() - reserved,
        "fb_persistent_growth": persistent_growth,
        "fb_scope_reserved": torch.cuda.max_memory_reserved() - reserved - persistent_growth,
        "fb_allocated_after": torch.cuda.memory_allocated() - allocated,
    }
    snapshot = torch.cuda.memory_snapshot(include_traces=False)
    state = {
        name: {
            "held": _held_bytes(snapshot, tensors),
            "payload": _payload_bytes(tensors),
        }
        for name, tensors in _state_tensors(runner).items()
    }

    allocated, reserved = start()
    optim_step(runner, batch.num_targets)
    torch.cuda.synchronize()
    window["step_allocated_peak"] = torch.cuda.max_memory_allocated() - allocated
    window["step_reserved_peak"] = torch.cuda.max_memory_reserved() - reserved
    return window, state


def _batch(case: _Case) -> TrainingBatch:
    import torch

    from tests.support.training import PAD_TOKEN_ID, packed_sft_batch, padded_sft_batch

    generator = torch.Generator().manual_seed(SAMPLE_SEED)
    length = case.tokens // case.sequences
    samples = tuple(
        tuple(torch.randint(0, 150_000, (length,), generator=generator).tolist())
        for _ in range(case.sequences)
    )
    if case.layout == "packed":
        return packed_sft_batch(samples)
    return padded_sft_batch(samples, PAD_TOKEN_ID)


def _training_config(case: _Case) -> Any:
    from cti.config import (
        ActivationCheckpointMode,
        AdamWConfig,
        BatchLayout,
        DynamicBatching,
        SFTShapeProfile,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
        TrainingStateResidency,
    )

    length = case.tokens // case.sequences
    policy = (
        DynamicBatching(case.tokens, max_sequences_per_microbatch=case.sequences)
        if case.dynamic
        else StaticBatching(microbatch_size=case.sequences, max_tokens_per_sample=length)
    )
    return TrainingConfig(
        trainable_plan=TrainablePlan.all_declared(),
        optimizer=AdamWConfig(),
        batching=TrainingBatching(BatchLayout(case.layout), policy),
        activation_checkpointing=ActivationCheckpointMode(case.checkpointing),
        sft_profile=SFTShapeProfile((length,) * case.sequences) if case.staged else None,
        # The activation table is the residency's: RESIDENT creates the optimizer state with
        # the runner, outside the measured windows.
        training_state_residency=TrainingStateResidency.RESIDENT,
    )


def _relative_errors(plan: dict[str, int], measured: dict[str, int]) -> dict[str, float | None]:
    """(plan - measured) / measured per compared quantity; positive over-counts."""

    return {
        name: None if measured[name] == 0 else (plan[name] - measured[name]) / measured[name]
        for name in measured
    }


def _run_child(case: _Case) -> None:
    import torch

    import cti
    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        LoRAConfig,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
    )
    from cti.memory import native_backing
    from cti.models import get_model_class
    from cti.training._batching import plan_microbatches
    from cti.training._memory_plan import plan_model_training_memory
    from tests.support.gpu import start_rank

    device = {
        "capability": list(torch.cuda.get_device_capability()),
        "device": torch.cuda.get_device_name(),
        "total_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    receipt: dict[str, Any] = {"case": case.id, **device}
    if device["total_bytes"] < case.min_device_gib << 30:
        receipt["skipped"] = f"needs {case.min_device_gib} GiB of device memory"
        print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True), flush=True)
        return

    path = str(case.checkpoint.snapshot())
    architecture = json.loads((Path(path) / "config.json").read_text())["architectures"][0]
    model_class = get_model_class(architecture)
    config = _training_config(case)
    batch = _batch(case)
    lengths = batch.tokens.sequence_lengths()
    assert plan_microbatches(config.batching.policy, lengths) == (tuple(range(len(lengths))),)

    with fixed_kv_page_capacity(8):
        worker = start_rank(
            EngineOptions(
                model=ModelOptions(path=path, max_seq_len=case.tokens),
                memory=MemoryOptions(inference_ratio=0.95),
                scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=case.tokens),
                cache=CacheOptions(prefix="naive"),
                lora=LoRAConfig(
                    targets=model_class.LORA_TARGETS, rank=case.rank, alpha=2.0 * case.rank
                ),
                training=config,
            ),
            EngineTuning(attention_backend="fi", cuda_graph_bs=()),
            checks=False,
        )
    try:
        runner = worker.training
        assert runner is not None
        host = worker.worker.host
        # The plan the rank checked at startup, from the same built model.
        plan = plan_model_training_memory(
            config, host.model, trainables=host, rank=case.rank, allocator=native_backing
        )
        # Production keeps what outlives a microbatch in the persistent region; here a pool
        # of its own keeps it apart from the microbatch's temporaries.
        persistent = torch.cuda.MemPool()
        runner._persistent_allocation = lambda: torch.cuda.use_mem_pool(persistent)

        cold, state = _measure_window(runner, batch, persistent)
        warm, _ = _measure_window(runner, batch, persistent)
        assert state is not None
    finally:
        worker.close()

    planned = {
        **asdict(plan),
        "resident_bytes": plan.resident_bytes,
        "scope_bytes": plan.scope_bytes,
        "training_bytes": plan.training_bytes,
    }
    windows = {"cold": cold, "warm": warm}
    fb_allocated = max(window["fb_allocated_peak"] for window in windows.values())
    fb_requested = max(window["fb_requested_peak"] for window in windows.values())
    fb_scope = max(window["fb_scope_reserved"] for window in windows.values())
    step_reserved = max(window["step_reserved_peak"] for window in windows.values())
    held = {name: values["held"] for name, values in state.items()}
    measured = {
        **held,
        "activations": fb_requested,
        "optimizer_step": step_reserved,
        "activations_and_segments": fb_scope,
        "training_bytes": (
            held["master_weights"]
            + held["optimizer_state"]
            + held["input_staging"]
            + max(cold["fb_reserved_peak"], held["gradients"] + step_reserved)
        ),
    }
    compared = {
        **{name: planned[name] for name in held},
        "activations": plan.activations,
        "optimizer_step": plan.optimizer_step,
        "activations_and_segments": plan.activations + plan.activation_segments,
        "training_bytes": plan.training_bytes,
    }
    # (what, planned, measured): the bounds the plan documents.
    bounds = [(name, planned[name], held[name]) for name in held]
    bounds.append(("activations + gradients", plan.activations + plan.gradients, fb_allocated))
    if case.tokens >= ACTIVATION_ONLY_BOUND_TOKENS:
        bounds.append(("activations", plan.activations, fb_requested))
    bounds.append(("scope_bytes (forward/backward)", plan.scope_bytes, fb_scope))
    bounds.append(("scope_bytes (optimizer step)", plan.scope_bytes, step_reserved))
    receipt.update(
        checkpoint=case.checkpoint.key,
        rank=case.rank,
        checkpointing=case.checkpointing,
        tokens=case.tokens,
        layout=case.layout,
        batching="dynamic" if case.dynamic else "static",
        sequences=case.sequences,
        staged=case.staged,
        input_tokens=sum(lengths) if case.layout == "packed" else sum(lengths) - len(lengths),
        table_key=(
            [case.label, case.rank, case.checkpointing, case.tokens] if case.table_key else None
        ),
        plan=planned,
        windows=windows,
        state=state,
        compared={"plan": compared, "measured": measured},
        relative_error=_relative_errors(compared, measured),
        violations=[
            f"{name}: plan {plan_bytes} < measured {measured_bytes}"
            for name, plan_bytes, measured_bytes in bounds
            if plan_bytes < measured_bytes
        ],
        source_root=str(Path(cti.__file__).resolve().parents[2]),
        torch=torch.__version__,
    )
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True), flush=True)


# ===================================== Test =====================================


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_training_memory_plan_bounds_one_measured_window(tmp_path: Path, case: _Case) -> None:
    env, log_root = offline_child_env(tmp_path, f"training-memory-{case.id}")
    env["PYTHONPATH"] = source_pythonpath(REPO_ROOT)
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.training.test_training_memory_measured",
            "--case",
            case.id,
        ),
        log_root / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=CHILD_TIMEOUT_SECONDS,
    )
    receipt = child_receipt(output, RECEIPT_PREFIX)
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))
    if "skipped" in receipt:
        pytest.skip(receipt["skipped"])
    assert receipt["case"] == case.id
    assert Path(receipt["source_root"]) == REPO_ROOT
    assert not receipt["violations"], receipt["violations"]


# =================================== Summary ===================================


def _summarize(logs: list[Path]) -> None:
    """Print each receipt's relative errors, then the accuracy table's rows."""

    receipts = {}
    for log in logs:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            _, marker, payload = line.partition(RECEIPT_PREFIX)
            if marker and payload.startswith("{"):
                receipt = json.loads(payload)
                receipts[receipt["case"]] = receipt
    names = (
        "master_weights",
        "optimizer_state",
        "gradients",
        "input_staging",
        "activations",
        "activations_and_segments",
        "optimizer_step",
        "training_bytes",
    )
    print("case," + ",".join(names) + ",violations")
    for case_id, receipt in receipts.items():
        if "skipped" in receipt:
            print(f"{case_id},skipped: {receipt['skipped']}")
            continue
        errors = receipt["relative_error"]
        cells = ("" if errors[name] is None else f"{errors[name]:+.4f}" for name in names)
        print(f"{case_id},{','.join(cells)},{len(receipt['violations'])}")
    modes = {"none": "_NONE", "full": "_FULL", "full_offload": "_OFFLOAD"}
    print("\n_MEASURED = {")
    for receipt in receipts.values():
        key = receipt.get("table_key")
        if not key:
            continue
        model, rank, mode, tokens = key
        measured = receipt["compared"]["measured"]
        values = ", ".join(
            str(measured[name])
            for name in ("master_weights", "optimizer_state", "gradients", "activations")
        )
        print(f'    ("{model}", {rank}, {modes[mode]}, {tokens}): ({values}),')
    print("}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--case", choices=[case.id for case in CASES])
    action.add_argument("--summarize", type=Path, nargs="+")
    args = parser.parse_args()
    if args.summarize:
        _summarize(args.summarize)
    else:
        _run_child(_case(args.case))
