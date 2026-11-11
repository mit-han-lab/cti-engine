"""Mixed steps on the designed DFlash pair of ``tests.support.dflash_designed``: prompts
arriving one step apart fold into the running requests' rounds, which run the prompts'
extensions in the round's target forward.

Each prompt arriving while rows run folds into the next round, whether the engine steps one
submission at a time or plans each step while the one before runs. Every
request's output must follow its chain, and at every settlement the target and draft KV of
each committed position, the folded prompts' included, must match its token and position.
With captured rounds, the mixed steps' rounds replay their size's draft graph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.support.dflash_designed import (
    CommittedKV,
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
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.checkpoints()

RECEIPT_PREFIX = "CTI_DFLASH_FOLD="
# (first prompt token, prompt length, max_tokens), one arriving per step, as many as there
# are request slots; lengths cover one token, a partial block and several whole blocks, and
# every output outlasts the arrivals, so each running row keeps its next round.
_ARRIVALS = ((10, 6, 120), (40, 1, 150), (70, 17, 90), (150, 64, 100))


def _run_child(root: str, graphs: str, loop: str) -> None:
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
    from cti.protocol import SamplingParams, UserMsg
    from tests.support.gpu import start_rank

    DFlashDraftModel.forward = scripted_draft
    target, draft, target_weights, draft_weights = write_checkpoints(Path(root))
    worker = start_rank(
        EngineOptions(
            model=ModelOptions(path=target),
            memory=MemoryOptions(inference_ratio=0.2),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=128),
            speculative=SpeculativeOptions(model_path=draft),
        ),
        EngineTuning(cuda_graph_bs=(1, 2, 3, 4) if graphs == "captured" else ()),
        checks=True,
    )
    model_runner = worker.executor
    engine, replies = engine_over(worker, overlap=loop == "overlapped")
    runner = model_runner._speculative
    check_committed_kv = CommittedKV(model_runner, target_weights, draft_weights)

    settle_step = engine._scheduler.settle_step
    mixed: list[tuple[int, int]] = []
    # Draft replays inside the mixed steps' rounds; plain rounds replay the graphs too.
    draft_replays, extending = 0, False
    replay = {key: graph.replay for key, graph in runner._draft_graphs.items()}
    extended_round = runner._extended_round

    def counted(key):
        def run() -> None:
            nonlocal draft_replays
            if extending:
                draft_replays += 1
            replay[key]()

        return run

    def counting_extended(*args):
        nonlocal extending
        extending = True
        try:
            return extended_round(*args)
        finally:
            extending = False

    for key, graph in runner._draft_graphs.items():
        graph.replay = counted(key)
    runner._extended_round = counting_extended

    def check(step, completion):
        outcome = settle_step(step, completion)
        if step.forward_rows and step.round_rows:
            mixed.append((len(step.forward_rows), len(step.round_rows)))
        for entry in step.rows:
            request = entry.request
            if not request.is_finished:
                tokens = request.token_ids.tolist()[: request.cached_len]
                check_committed_kv(entry.slot.index, tokens)
        return outcome

    engine._scheduler.settle_step = check
    expected = {}
    for uid, (start, length, max_tokens) in enumerate(_ARRIVALS):
        message = UserMsg(
            uid=uid,
            input_ids=torch.tensor(chain(start, length), dtype=torch.int32),
            sampling=SamplingParams(
                temperature=0, max_new_tokens=max_tokens, ignore_eos=True, seed=uid
            ).snapshot(),
        )
        admit(engine, replies, message)
        expected[uid] = expected_output(start, length, max_tokens, True)
        engine.step()
    while engine.busy:
        engine.step()
    engine._scheduler.require_quiescent()

    outputs: dict[int, list[int]] = {}
    for reply in replies:
        outputs.setdefault(reply.uid, []).extend(reply.token_ids)
    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "outputs_match": outputs == expected,
                "finished": sorted(reply.uid for reply in replies if reply.finished),
                "mixed_steps": mixed,
                "draft_replays": draft_replays,
                "max_kv_error": max(check_committed_kv.errors),
                "kv_checks": len(check_committed_kv.errors),
                "totals": str(engine._scheduler.totals),
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
            }
        )
    )
    worker.close()


@pytest.mark.parametrize(
    ("graphs", "loop"), [("captured", "serial"), ("none", "serial"), ("captured", "overlapped")]
)
def test_prompts_fold_into_rounds_and_follow_their_chains(
    tmp_path: Path, graphs: str, loop: str
) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "dflash-fold")
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    env["PYTHONNOUSERSITE"] = "1"
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.inference.test_dflash_fold",
            str(checkpoint_root),
            graphs,
            loop,
        ),
        log_root / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert receipt["outputs_match"]
    assert receipt["finished"] == list(range(len(_ARRIVALS)))
    # Every prompt after the first folds into a round of the requests before it.
    assert len(receipt["mixed_steps"]) == len(_ARRIVALS) - 1
    assert all(prompts == 1 and rows >= 1 for prompts, rows in receipt["mixed_steps"])
    if graphs == "captured":
        assert receipt["draft_replays"] == len(_ARRIVALS) - 1
    else:
        assert receipt["draft_replays"] == 0
    assert receipt["kv_checks"] > 0 and receipt["max_kv_error"] < 0.05
    assert_cuda_device_metadata(receipt)


if __name__ == "__main__":
    _run_child(sys.argv[1], sys.argv[2], sys.argv[3])
