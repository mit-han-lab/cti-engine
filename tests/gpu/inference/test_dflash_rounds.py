"""DFlash rounds against exact expectations on a designed target.

The target's layers add nothing to the residual stream and its LM head maps each token's
embedding to the next token id, so greedy decoding continues a prompt's ``t -> t + 1``
chain with a wide margin. A scripted draft proposes the chain for ``(anchor * 7) % Q``
positions of each block and a wrong token after them, so every acceptance length occurs.
Requests cover shrinking captured batches, chunked prefill, EOS and budget clipping, rounds
up to the sequence capacity, two rounds in flight per request and slot reuse. At every
settlement the target and draft KV of each committed position must match its token and
position.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.support.dflash_designed import (
    MAX_SEQ_LEN,
    CommittedKV,
    Q,
    admit,
    chain,
    engine_over,
    expected_output,
    scripted_draft,
    write_checkpoints,
)
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    pause_device,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.checkpoints()

RECEIPT_PREFIX = "CTI_DFLASH_ROUNDS="
# Its first token is EOS, which the device never checks: the round it joined before the
# prompt settled runs on and is discarded.
_FIRST_STOP = (192, 8, 30, False)
# Its own stop token, its 11th output, ends it inside a round the device runs on.
_OWN_STOP = (30, 6, 40, True)
_OWN_STOP_TOKEN = 30 + 6 + 10
# (first prompt token, prompt length, max_tokens, ignore_eos), submitted wave by wave.
_WAVES = (
    ((10, 20, 60, True), (40, 5, 33, True), (70, 150, 20, True), (150, 8, 100, False)),
    ((0, MAX_SEQ_LEN - 42, 42, True),),
    ((3, 6, 25, True),),
    ((90, 7, 31, True), (120, 9, 18, True)),
    # First anchors 104..119 cover every residue, hence every acceptance length.
    tuple((100 + index, 4, 18, True) for index in range(Q)),
    (_FIRST_STOP,),
    (_OWN_STOP,),
    # Aborted after its second round settles, while its successor is in flight.
    ((60, 6, 200, True),),
)
_ABORT_AFTER_ROUNDS = 2
# Captured round sizes; a round replays the smallest that holds it, padded with done rows.
_LADDERS = {"captured": [1, 2, 3, 4], "gapped": [1, 2, 4], "largest": [4], "none": []}


def _run_child(root: str, graphs: str) -> None:
    import numpy
    import torch

    from cti.config import (
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        SpeculativeOptions,
    )
    from cti.models.dflash.model import DFlashDraftModel
    from cti.protocol import AbortBackendMsg, DetokenizeMsg, SamplingParams, UserMsg
    from ctikernels.speculative import compile_count
    from tests.support.gpu import start_rank

    DFlashDraftModel.forward = scripted_draft
    target, draft, target_weights, draft_weights = write_checkpoints(Path(root))
    worker = start_rank(
        EngineOptions(
            model=ModelOptions(path=target),
            memory=MemoryOptions(inference_ratio=0.2),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=64),
            speculative=SpeculativeOptions(model_path=draft),
        ),
        EngineTuning(cuda_graph_bs=tuple(_LADDERS[graphs])),
        checks=True,
    )
    compiled_at_bind = compile_count()
    engine, replies = engine_over(worker, overlap=True)
    model_runner = worker.executor
    runner = model_runner._speculative
    scheduler = engine._scheduler
    observed_counts: set[int] = set()

    check_committed_kv = CommittedKV(model_runner, target_weights, draft_weights)

    eager_rounds = 0
    round_sizes: list[int] = []
    count_mismatches: list[tuple[int, int, int]] = []
    one_token_decodes = 0
    discarded: list[int] = []
    overlap: list[bool] = []
    depth: list[int] = []
    settled_rounds: dict[int, int] = {}
    in_flight_at_cancel: list[int] = []
    abort_uid = sum(map(len, _WAVES)) - 1
    first_stop_uid = [spec for wave in _WAVES for spec in wave].index(_FIRST_STOP)
    own_stop_uid = [spec for wave in _WAVES for spec in wave].index(_OWN_STOP)
    eager_target, settle_step = runner._target, scheduler.settle_step
    run_forward, run_round_step = model_runner._forward, model_runner._round
    round_events: list[torch.cuda.Event] = []

    def count_eager(*args) -> None:
        # A plain round runs its target forward eagerly only past its captured sizes.
        nonlocal eager_rounds
        eager_rounds += 1
        eager_target(*args)

    def check_counts(step, output):
        rows = step.round_rows
        if not rows:
            return settle_step(step, output)
        round_sizes.append(len(rows))
        abort = None
        targets = numpy.frombuffer(output.tokens, dtype="int32").reshape(-1, output.width)
        counts = numpy.frombuffer(output.accepted, dtype="int32").tolist()
        commits = numpy.frombuffer(output.committed, dtype="int32").tolist()
        for row, (entry, accepted) in enumerate(zip(rows, counts, strict=True)):
            request = entry.request
            if request.is_finished:
                # A finished request's successor publishes nothing.
                discarded.append((request.uid, commits[row]))
                continue
            anchor = int(request.token_ids[-1])
            observed_counts.add(accepted)
            if accepted != anchor * 7 % Q:
                count_mismatches.append((request.uid, anchor, accepted))
            drafts = targets[row, :accepted].tolist()
            committed = request.cached_len + commits[row]
            check_committed_kv(entry.slot.index, (request.token_ids.tolist() + drafts)[:committed])
            settled_rounds[request.uid] = settled_rounds.get(request.uid, 0) + 1
            if request.uid == abort_uid and settled_rounds[request.uid] == _ABORT_AFTER_ROUNDS:
                abort = request
        outcome = settle_step(step, output)
        if abort is not None:
            in_flight_at_cancel.append(len(scheduler.slots.slot(abort.slot).round_credits))
            published = len(replies)
            engine.handle([AbortBackendMsg(uid=abort.uid)], admitting=True)
            assert len(replies) > published
        return outcome

    def count_decodes(op, table):
        nonlocal one_token_decodes
        one_token_decodes += all(row.query_kind.name == "DECODE" for row in op.rows)
        return run_forward(op, table)

    def count_rounds(op, table):
        # submit(r + 1) before host_ready(r): an unsettled round's record not yet on the
        # host when its successor is submitted.
        round_events[:] = [event for event in round_events if not event.query()]
        overlap.append(bool(round_events))
        depth.append(max((len(s.round_credits) for s in scheduler.slots.running), default=0))
        result = run_round_step(op, table)
        round_events.append(result.completion_event)
        return result

    after_forward = runner.after_forward

    def unsynced_after_forward(*args) -> None:
        # Queued behind unsettled rounds, a prompt's forward must not wait for them.
        torch.cuda.set_sync_debug_mode("error")
        try:
            after_forward(*args)
        finally:
            torch.cuda.set_sync_debug_mode(0)

    run_round = runner.run_round

    def poisoned_round(staged, step=None):
        # A round reads only the CSR prefix it writes: poison every KV index first. The
        # device pause outlasts the host's loop, so successors can be submitted early.
        runner._target_backend._addresses.indices.fill_(torch.iinfo(torch.int32).max)
        pause_device(1 << 21)
        # Rounds, eager ones included, never wait for the device.
        torch.cuda.set_sync_debug_mode("error")
        try:
            return run_round(staged, step)
        finally:
            torch.cuda.set_sync_debug_mode(0)

    runner._target = count_eager
    runner.run_round = poisoned_round
    runner.after_forward = unsynced_after_forward
    scheduler.settle_step = check_counts
    model_runner._forward = count_decodes
    model_runner._round = count_rounds

    uid, expected = 0, {}
    for wave in _WAVES:
        for start, length, max_tokens, ignore_eos in wave:
            own_stop = uid == own_stop_uid
            message = UserMsg(
                uid=uid,
                input_ids=torch.tensor(chain(start, length), dtype=torch.int32),
                # Ingress seeds every DFlash request; greedy rounds draw no randomness.
                sampling=SamplingParams(
                    temperature=0,
                    max_new_tokens=max_tokens,
                    ignore_eos=ignore_eos,
                    seed=uid,
                    stop_token_ids=(_OWN_STOP_TOKEN,) if own_stop else (),
                ).snapshot(),
            )
            admit(engine, replies, message)
            expected[uid] = expected_output(start, length, max_tokens, ignore_eos)
            if own_stop:
                expected[uid] = expected[uid][: expected[uid].index(_OWN_STOP_TOKEN) + 1]
            uid += 1
        while engine.busy:
            engine.step()
        scheduler.require_quiescent()

    # The token replies; the abort's own reply is checked where it is sent.
    replies = [reply for reply in replies if isinstance(reply, DetokenizeMsg)]
    outputs: dict[int, list[int]] = {}
    for reply in replies:
        outputs.setdefault(reply.uid, []).extend(reply.token_ids)
    aborted = outputs.pop(abort_uid)
    full = expected.pop(abort_uid)
    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "outputs_match": outputs == expected,
                "cancelled_output_is_a_prefix": 0 < len(aborted) < len(full)
                and aborted == full[: len(aborted)],
                "in_flight_at_cancel": in_flight_at_cancel,
                "rounds": len(overlap),
                "submitted_before_ready": sum(overlap),
                "max_depth": max(depth),
                "finished": sorted(reply.uid for reply in replies if reply.finished),
                "count_mismatches": count_mismatches,
                "observed_counts": sorted(observed_counts),
                "max_kv_error": max(check_committed_kv.errors),
                "kv_checks": len(check_committed_kv.errors),
                "round_sizes": sorted(set(round_sizes)),
                "captured": sorted(runner._target_graphs),
                "eager_rounds": eager_rounds,
                "round_kernels_compiled_after_bind": compile_count() - compiled_at_bind,
                "one_token_decodes": one_token_decodes,
                # A request that ended at a stop or its budget is done on the device: its
                # successors commit nothing. An aborted one's may, and so may the round a
                # request joined before its first token stopped it.
                "discarded_commits": sorted(
                    {
                        count
                        for uid, count in discarded
                        if uid not in (abort_uid, first_stop_uid, own_stop_uid)
                    }
                ),
                "first_stop_discards": [count for uid, count in discarded if uid == first_stop_uid],
                "own_stop_finish": [
                    (reply.end.is_stop_token, reply.end.finish_reason)
                    for reply in replies
                    if reply.uid == own_stop_uid and reply.finished
                ],
                "requests": len(expected) + 1,
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
            }
        )
    )
    worker.close()


@pytest.mark.parametrize("graphs", list(_LADDERS))
def test_dflash_rounds_follow_the_scripted_acceptance(tmp_path: Path, graphs: str) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "dflash-rounds")
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    env["PYTHONNOUSERSITE"] = "1"
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.inference.test_dflash_rounds",
            str(checkpoint_root),
            graphs,
        ),
        log_root / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert receipt["outputs_match"]
    # Every request but the aborted one finishes once; the aborted one publishes a prefix.
    assert receipt["finished"] == list(range(receipt["requests"] - 1))
    assert receipt["cancelled_output_is_a_prefix"] and receipt["in_flight_at_cancel"] == [1]
    # Two rounds per request in flight, and successors submitted before the host had
    # their predecessors' records. Eager rounds launch slower than this tiny model runs,
    # and each settlement's KV check syncs the device, so only replays can stay ahead.
    assert receipt["max_depth"] == 2
    if graphs != "none":
        assert receipt["submitted_before_ready"] > 0
    assert receipt["count_mismatches"] == []
    assert receipt["observed_counts"] == list(range(Q))
    # BF16 storage of FP32 references; a wrong token or position errs by O(1).
    assert receipt["kv_checks"] > 0 and receipt["max_kv_error"] < 0.05
    assert receipt["round_sizes"] == [1, 2, 3, 4]
    # Padded rounds commit exactly what exact ones do (outputs and KV above), never eagerly.
    assert receipt["captured"] == _LADDERS[graphs]
    assert (receipt["eager_rounds"] > 0) == (graphs == "none")
    # Bind compiled every round-kernel variant the rounds use, eager ones included.
    assert receipt["round_kernels_compiled_after_bind"] == 0
    # Rounds run every request to its end, the one filling the sequence included.
    assert receipt["one_token_decodes"] == 0
    assert set(receipt["discarded_commits"]) <= {0}
    assert len(receipt["first_stop_discards"]) == 1 and receipt["first_stop_discards"][0] > 0
    # Its own stop ends a request on the host: its output is the prefix through the stop.
    assert receipt["own_stop_finish"] == [[True, "stop"]]
    assert_cuda_device_metadata(receipt)


if __name__ == "__main__":
    _run_child(sys.argv[1], sys.argv[2])
