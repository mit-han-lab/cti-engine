"""Breakable graph replays of Qwen3, Qwen3.5 and the MoE models through ``runner.graphs``:
bitwise equal to the padded eager forward (``PaddedBreaks``), with the same KV and recurrent
state writes and none from padding rows. The MoE blocks run inside the segments over whole
buckets, so a MoE prompt step replays with no MoE break."""

from __future__ import annotations

import gc
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

from tests.support.checkpoints import QWEN3_0_6B, QWEN3_5_9B, QWEN3_5_35B_A3B
from tests.support.gpu import (
    assert_cuda_device_metadata,
    fixed_kv_page_capacity,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import PYTHON_ROOT, REPO_ROOT, source_pythonpath
from tests.support.qwen3_5_gpu import PINNED_CHECKPOINTS, selected_checkpoint

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B, QWEN3_5_9B)

if TYPE_CHECKING:
    import torch

    from cti.forward import ForwardInputs, ForwardRows, GraphBreaks
    from cti.inference._execution.runner import InferenceRunner

RECEIPT_PREFIX = "CTI_BREAKABLE_GRAPH="
BUCKETS = (64, 128)
DECODE_SIZES = (1, 2)
NUM_KV_PAGES = 8192
LIVE_PREFIX = 16
# (bucket, rows, sequences): the rows around each bucket edge, over 1 and 3 sequences, in
# the bucket a replay selects.
PARITY_CASES = tuple(
    (min(bucket for bucket in BUCKETS if bucket >= rows), rows, sequences)
    for sequences in (1, 3)
    for rows in (1, 63, 64, 65, 127, 128)
    if rows >= sequences
)
LIVE_PREFIX_CASE = (128, 127, 3)
INTERLEAVED_CASES = ((64, 50, 1), (128, 100, 3), (64, 64, 3), (128, 128, 1))
INTERLEAVED_STEPS = 20


class _Storage(NamedTuple):
    rows: ForwardRows
    input_ids: torch.Tensor
    positions: torch.Tensor
    prefix: torch.Tensor  # KV slots of the live prefixes
    sentinels: torch.Tensor  # KV slots around every sequence, never written
    requests: torch.Tensor  # request slots, the recurrent state rows the batch writes


class _Slots:
    """Hands out disjoint requests and KV slots, so every batch writes its own."""

    def __init__(self, runner: InferenceRunner) -> None:
        self._runner = runner
        self._request = 0
        self._slot = 1

    def batch(self, lengths: list[int], *, prefix: int = 0, decode: bool = False) -> _Storage:
        import torch

        from cti.inference._step import QueryKind, Row

        runner, device = self._runner, self._runner.device
        entries, positions, prefixes, sentinels = [], [], [], []
        first_request = self._request
        for length in lengths:
            sequence_len = prefix + length
            entries.append(
                Row(
                    request_index=self._request,
                    query_kind=QueryKind.DECODE if decode else QueryKind.EXTEND,
                    kv_prefix_len=prefix,
                    sequence_len=sequence_len,
                    write_position=sequence_len if decode else None,
                )
            )
            slots = torch.arange(
                self._slot + 1, self._slot + 1 + sequence_len, dtype=torch.int32, device=device
            )
            # The backend's metadata takes each row's write slot from this table.
            runner.page_table.pages[self._request, :sequence_len].copy_(slots)
            positions.append(torch.arange(prefix, sequence_len, device=device))
            prefixes.append(slots[:prefix])
            sentinels += [self._slot, self._slot + sequence_len + 1]
            self._request += 1
            self._slot += sequence_len + 2
        assert self._slot < runner.limits.capacity.num_pages
        rows = runner.graphs.rows(tuple(entries), ())
        positions = torch.cat(positions)
        input_ids = _input_ids(runner, positions.numel(), salt=self._slot)
        sentinels = torch.tensor(sentinels, dtype=torch.int32, device=device)
        requests = torch.arange(first_request, self._request, device=device)
        return _Storage(rows, input_ids, positions, torch.cat(prefixes), sentinels, requests)


def _input_ids(runner: InferenceRunner, rows: int, *, salt: int) -> torch.Tensor:
    import torch

    vocab = runner.sampler.vocab_size
    ids = (torch.arange(rows, device=runner.device) * 17 + salt) % (vocab - 1)
    return ids.to(torch.int32)


def _lengths(rows: int, sequences: int) -> list[int]:
    """``rows`` split raggedly over ``sequences``: the first takes the remainder."""

    base = rows // sequences
    return [rows - base * (sequences - 1)] + [base] * (sequences - 1)


@contextmanager
def _on_runner(runner: InferenceRunner) -> Iterator[None]:
    import torch

    with torch.inference_mode(), torch.cuda.stream(runner.execution_stream):
        yield
    runner.execution_stream.synchronize()


def _kv_slots(runner: InferenceRunner) -> Iterator[tuple[int, torch.Tensor]]:
    """Every layer's K and V caches, one row per slot, each with a distinct fill value."""

    # FlashInfer attends over the target's one KV group.
    for layer, (k_cache, v_cache) in enumerate(runner._participants[0].kv):
        yield layer + 1, k_cache.flatten(0, 1)
        yield -layer - 1, v_cache.flatten(0, 1)


def _recurrent_rows(runner: InferenceRunner) -> Iterator[torch.Tensor]:
    """Every recurrent layer's state and convolution state, one row per request slot."""

    pool = None if runner._recurrent is None else runner._recurrent._pool
    for layer in range(0 if pool is None else pool.num_layers):
        yield from pool.layer_views(layer)


def _run(
    runner: InferenceRunner,
    storage: _Storage,
    forward: Callable[[ForwardRows, ForwardInputs], torch.Tensor],
    *,
    input_ids: torch.Tensor | None = None,
    graph_breaks: GraphBreaks | None = None,
) -> torch.Tensor:
    """``forward(rows, batch)`` as the runner steps it: the batch's metadata built right
    before it runs (FlashInfer keeps one live plan), its recurrent rows restored before the
    forward and exported after it."""

    import torch

    from cti.forward import ForwardInputs, InlineBreaks

    rows, recurrent = storage.rows, runner._recurrent
    attn_metadata = runner._participants[0].prepare_metadata(rows)
    recurrent_metadata = None
    if recurrent is not None:
        recurrent_metadata = recurrent.prepare_metadata(rows)
    batch = ForwardInputs(
        storage.input_ids if input_ids is None else input_ids,
        storage.positions,
        attn_metadata,
        rows.size,
        rows.uses_ragged_attention,
        recurrent_metadata,
        logit_rows=torch.tensor(rows.logit_rows, dtype=torch.int64, device=runner.device),
        graph_breaks=InlineBreaks() if graph_breaks is None else graph_breaks,
    )
    if recurrent is not None:
        recurrent.restore(batch)
    output = forward(rows, batch).clone()
    if recurrent is not None:
        recurrent.export(batch)
    return output


def _hidden(runner: InferenceRunner, storage: _Storage) -> torch.Tensor:
    return _run(runner, storage, lambda _rows, batch: runner.graphs.hidden(batch))


def _logits(runner: InferenceRunner, storage: _Storage) -> torch.Tensor:
    return _run(runner, storage, runner.graphs.logits)


def _reference_hidden(runner: InferenceRunner, storage: _Storage, bucket: int) -> torch.Tensor:
    """The eager forward of ``storage`` padded to ``bucket`` rows, its breaks on the real
    rows."""

    from tests.support.graph_breaks import PaddedBreaks

    real = storage.input_ids
    rows = real.numel()
    padded = real.new_full((bucket,), 123)
    padded[:rows].copy_(real)
    return _run(
        runner,
        storage,
        lambda _rows, batch: runner.model.forward_hidden(batch),
        input_ids=padded,
        graph_breaks=PaddedBreaks(rows),
    )[:rows]


def _exercise_parity(
    runner: InferenceRunner,
    cases: tuple[tuple[int, int, int], ...],
    live_prefix_case: tuple[int, int, int] | None = None,
) -> list[dict[str, int]]:
    """Case 1: replay == padded reference bitwise on twin request and KV slots from equal
    state; same KV and recurrent rows; padding writes no KV."""

    import torch

    dummy = runner.page_table.pages[runner.limits.capacity.num_slots, :1]
    receipts = []
    for bucket, rows, sequences in cases:
        prefix = LIVE_PREFIX if (bucket, rows, sequences) == live_prefix_case else 0
        slots = _Slots(runner)
        eager, graphed = (slots.batch(_lengths(rows, sequences), prefix=prefix) for _ in "ab")
        graphed = graphed._replace(input_ids=eager.input_ids)
        sentinels = torch.cat((eager.sentinels, graphed.sentinels, dummy))
        for fill, cache in _kv_slots(runner):
            cache[eager.prefix] = cache[graphed.prefix] = fill / 128
            cache[sentinels] = fill
        for state in _recurrent_rows(runner):
            state[eager.requests] = state[graphed.requests] = 0
        before = runner.graph_stats().breakable
        with _on_runner(runner):
            reference = _reference_hidden(runner, eager, bucket)
            hidden = _hidden(runner, graphed)
        assert runner.graph_stats().breakable == before + 1
        assert torch.equal(hidden, reference)
        # The KV slots each batch wrote, as its backend metadata places them.
        written = tuple(
            runner._participants[0].prepare_metadata(storage.rows).cache_write_indices
            for storage in (eager, graphed)
        )
        for fill, cache in _kv_slots(runner):
            assert torch.equal(cache[written[0]], cache[written[1]])
            assert bool((cache[sentinels] == fill).all())
        for state in _recurrent_rows(runner):
            assert torch.equal(state[eager.requests], state[graphed.requests])
            assert bool(state[graphed.requests].any())
        receipts.append({"bucket": bucket, "rows": rows, "sequences": sequences, "prefix": prefix})
    return receipts


def _exercise_interleaving(runner: InferenceRunner) -> dict[str, int]:
    """Case 2: replays across buckets, between decode graph replays, stay bitwise."""

    import torch

    slots = _Slots(runner)
    cases = []
    with _on_runner(runner):
        for bucket, rows, sequences in INTERLEAVED_CASES:
            eager, graphed = (slots.batch(_lengths(rows, sequences)) for _ in "ab")
            graphed = graphed._replace(input_ids=eager.input_ids)
            cases.append((graphed, _reference_hidden(runner, eager, bucket)))
        decode = slots.batch([1] * DECODE_SIZES[-1], prefix=32, decode=True)
        for fill, cache in _kv_slots(runner):
            cache[decode.prefix] = fill / 128
        decode_logits = _logits(runner, decode)
    before = runner.graph_stats()
    for step in range(INTERLEAVED_STEPS):
        storage, reference = cases[step % len(cases)]
        with _on_runner(runner):
            hidden = _hidden(runner, storage)
            logits = _logits(runner, decode)
        assert torch.equal(hidden, reference)
        assert torch.equal(logits, decode_logits)
    after = runner.graph_stats()
    return {"breakable": after.breakable - before.breakable, "decode": after.decode - before.decode}


def _exercise_stats(runner: InferenceRunner) -> dict[str, int]:
    """Case 3: a ragged batch that fits a bucket replays; a larger one runs eagerly."""

    slots = _Slots(runner)
    before = runner.graph_stats()
    for rows in (100, BUCKETS[-1] + 1):
        storage = slots.batch(_lengths(rows, 3), prefix=5)
        with _on_runner(runner):
            _hidden(runner, storage)
    after = runner.graph_stats()
    return {
        "breakable": after.breakable - before.breakable,
        "eager": after.eager - before.eager,
        "decode": after.decode - before.decode,
    }


def _exercise_unpadded_eager(runner: InferenceRunner) -> float:
    """Case 4: logits of the unpadded eager forward vs the replay. cuBLAS picks algorithms by
    row count, so this is a diagnostic: only finiteness is asserted."""

    import torch

    storage = _Slots(runner).batch(_lengths(100, 3))
    with _on_runner(runner):
        eager = _run(runner, storage, lambda _rows, batch: runner.model.forward(batch)).float()
        graphed = _logits(runner, storage).float()
    assert bool(torch.isfinite(graphed).all())
    return float((graphed - eager).abs().max())


def _exercise_qwen3(runner: InferenceRunner) -> dict[str, object]:
    return {
        "parity": _exercise_parity(runner, PARITY_CASES, LIVE_PREFIX_CASE),
        "interleaved": _exercise_interleaving(runner),
        "stats": _exercise_stats(runner),
        "unpadded_eager_max_abs": _exercise_unpadded_eager(runner),
    }


def _exercise_qwen3_5(runner: InferenceRunner) -> dict[str, object]:
    return {"parity": _exercise_parity(runner, PARITY_CASES)}


def _exercise_moe(runner: InferenceRunner) -> dict[str, object]:
    return {"parity": _exercise_parity(runner, PARITY_CASES), "stats": _exercise_stats(runner)}


def _run_child(model: str, snapshot: str | None = None) -> None:
    import torch

    import cti
    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
    )
    from tests.support.gpu import start_rank

    exercise, requests, memory = _exercise_qwen3_5, 8, MemoryOptions(inference_ratio=0.85)
    if model == "moe":
        exercise = _exercise_moe
    if model == "qwen3":
        snapshot = str(QWEN3_0_6B.snapshot())
        exercise, requests = _exercise_qwen3, 32
        memory = MemoryOptions(inference_ratio=0.5, peak_ratio=0.5)
    with fixed_kv_page_capacity(NUM_KV_PAGES):
        worker = start_rank(
            EngineOptions(
                model=ModelOptions(path=snapshot, max_seq_len=1024),
                memory=memory,
                scheduling=SchedulingOptions(max_running_requests=requests, max_batch_tokens=256),
                cache=CacheOptions(prefix="naive"),
            ),
            EngineTuning(
                attention_backend="fi", cuda_graph_bs=DECODE_SIZES, graph_row_buckets=BUCKETS
            ),
            checks=True,
        )
    try:
        runner = worker.executor
        captures = {capture.kind: capture for capture in runner.graph_stats().captures}
        assert {kind: capture.sizes for kind, capture in captures.items()} == {
            "breakable": BUCKETS,
            "decode": DECODE_SIZES,
        }
        assert captures["breakable"].nbytes > 0, captures
        receipt = {
            "graph_bytes": {kind: capture.nbytes for kind, capture in captures.items()},
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "source_root": str(Path(cti.__file__).resolve().parent),
            "recurrent_layers": len(list(_recurrent_rows(runner))) // 2,
            **exercise(runner),
        }
    finally:
        worker.close()
    gc.collect()
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def _child_receipt(tmp_path: Path, *argv: str, timeout: float) -> dict:
    env, log_root = isolated_runtime_env(tmp_path, "breakable-graph")
    env.update(
        PYTHONPATH=source_pythonpath(REPO_ROOT),
        PYTHONNOUSERSITE="1",
        PYTHONHASHSEED="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_breakable_graph", *argv),
        log_root / f"breakable-graph-{argv[0]}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=timeout,
    )
    (receipt,) = (
        json.loads(line.removeprefix(RECEIPT_PREFIX))
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    )
    assert_cuda_device_metadata(receipt)
    assert Path(receipt["source_root"]).is_relative_to(PYTHON_ROOT)
    return receipt


def test_qwen_breakable_graph_replays_match_the_padded_forward(tmp_path: Path) -> None:
    receipt = _child_receipt(tmp_path, "qwen3", timeout=600)
    assert receipt["recurrent_layers"] == 0
    assert len(receipt["parity"]) == len(PARITY_CASES)
    assert sum(case["prefix"] > 0 for case in receipt["parity"]) == 1
    steps = INTERLEAVED_STEPS
    assert receipt["interleaved"] == {"breakable": steps, "decode": steps}
    assert receipt["stats"] == {"breakable": 1, "eager": 1, "decode": 0}
    assert receipt["unpadded_eager_max_abs"] >= 0


def test_qwen3_5_breakable_graph_replays_match_the_padded_forward(tmp_path: Path) -> None:
    # A missing pinned checkpoint raises QualificationBlocked: it fails the run, never skips.
    snapshot = str(selected_checkpoint().snapshot())
    receipt = _child_receipt(tmp_path, "qwen3_5", snapshot, timeout=900)
    assert receipt["recurrent_layers"] > 0
    parity = [(case["bucket"], case["rows"], case["sequences"]) for case in receipt["parity"]]
    assert parity == list(PARITY_CASES)


@pytest.mark.checkpoints(QWEN3_5_35B_A3B)
def test_moe_prompt_steps_replay_their_sparse_blocks_inside_the_breakable_graph(
    tmp_path: Path,
) -> None:
    checkpoint = next(checkpoint for checkpoint in PINNED_CHECKPOINTS if checkpoint.moe)
    receipt = _child_receipt(tmp_path, "moe", str(checkpoint.snapshot()), timeout=1200)
    parity = [(case["bucket"], case["rows"], case["sequences"]) for case in receipt["parity"]]
    assert parity == list(PARITY_CASES)
    # The bucketed ragged batch replays, MoE blocks and all; the one past the buckets runs
    # eagerly after the capture.
    assert receipt["stats"] == {"breakable": 1, "eager": 1, "decode": 0}


if __name__ == "__main__":
    _run_child(*sys.argv[1:])
