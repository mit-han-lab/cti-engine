"""A round's captured draft graph against the draft run eagerly, on the designed DFlash pair
of ``tests.support.dflash_designed`` with its real draft model.

After one prompt step leaves four rows live, every captured size drafts rows of different
requests at each verification in turn: eagerly, then by replaying the size's one draft graph
over outputs poisoned in between. The proposals, the block inputs and their metadata, the verify
attention's index CSR and the draft pool rows the draft wrote must match bitwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.support.dflash_designed import Q, admit, chain, engine_over, write_checkpoints
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.checkpoints()

RECEIPT_PREFIX = "CTI_DFLASH_DRAFT_GRAPH="
# (first prompt token, prompt length) of the four live rows.
_PROMPTS = ((10, 5), (40, 17), (70, 33), (150, 2))
# Per size, the rows it drafts, in turn: sizes 1-3 see two different row sets, size 4 all.
_ROW_SETS = {1: ((0,), (3,)), 2: ((0, 1), (2, 3)), 3: ((0, 1, 2), (1, 2, 3)), 4: ((0, 1, 2, 3),)}


def _run_child(root: str) -> None:
    import torch

    from cti.config import (
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        SpeculativeOptions,
    )
    from cti.inference import Round
    from cti.inference._execution._speculative.dflash import (
        RoundInput,
        Verification,
        round_write_fault,
    )
    from cti.inference._step import RequestSpec
    from cti.protocol import SamplingParams, UserMsg
    from cti.sampling import TokenSampling
    from tests.support.gpu import start_rank

    target, draft, _, _ = write_checkpoints(Path(root))
    worker = start_rank(
        EngineOptions(
            model=ModelOptions(path=target),
            memory=MemoryOptions(inference_ratio=0.2),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=128),
            speculative=SpeculativeOptions(model_path=draft),
        ),
        EngineTuning(cuda_graph_bs=(1, 2, 3, 4)),
        checks=True,
    )
    engine, replies = engine_over(worker, overlap=False)
    for uid, (start, length) in enumerate(_PROMPTS):
        message = UserMsg(
            uid=uid,
            input_ids=torch.tensor(chain(start, length), dtype=torch.int32),
            sampling=SamplingParams(
                temperature=0, max_new_tokens=40, ignore_eos=True, seed=uid
            ).snapshot(),
        )
        admit(engine, replies, message)
    # One prompt step takes all four; nothing runs after it.
    engine.step()
    running = (slot.request for slot in engine._scheduler.slots.running)
    requests = sorted(running, key=lambda request: request.uid)
    assert len(requests) == len(_PROMPTS)
    # The test drafts rows without running their round: plan one over all four, so the round
    # ledger reserves its pages as in serving, and map them; it is never submitted.
    planned = engine._scheduler.plan_next()
    assert isinstance(planned.op, Round) and len(planned.op.slots) == len(_PROMPTS)
    staging = worker.executor._apply_pages(planned.op.pages)
    torch.cuda.synchronize()
    del staging
    runner = worker.executor._speculative
    # The round scratch the draft writes, and its pool on the draft backend.
    scratch = runner._scratch
    capacity = worker.executor.limits.capacity
    draft_pages = runner._draft_backend.kv
    draft_layers = len(list(runner._models.draft.iter_attention_layers()))

    def outputs(size: int) -> dict[str, torch.Tensor]:
        plan = runner._plans[size]
        tokens = size * Q
        written = runner._target_backend._addresses.write[:tokens].long()
        values = {
            "proposals": scratch.proposals[:size],
            "block_inputs": plan.draft[0].input_ids,
            "positions": scratch.positions[:tokens],
            "rope_rows": scratch.rope_rows[:tokens],
            "cache_write_indices": runner._target_backend._addresses.write[:tokens],
            "start_frontiers": scratch.results.frontiers[:size],
            "round_done": scratch.round_done[:size],
            "kv_indptr": runner._target_backend._addresses.indptr[: size + 1],
            "cursors": scratch.cursors[:size],
        }
        indptr = runner._target_backend._addresses.indptr[: size + 1].long()
        values["kv_indices"] = runner._target_backend._addresses.indices[: int(indptr[-1])]
        for layer in range(draft_layers):
            for name, cache in zip("kv", draft_pages[layer], strict=True):
                values[f"draft_{name}{layer}"] = cache.flatten(0, 1)[written]
        return {name: value.clone() for name, value in values.items()}

    def poison(size: int, verification, written: torch.Tensor) -> None:
        """Overwrite everything the draft writes, so a replay must write it again."""

        buffers = [
            scratch.proposals,
            scratch.positions,
            scratch.rope_rows,
            runner._target_backend._addresses.indices,
        ]
        # Every draft places the cursors, a greedy row's too.
        buffers += [
            runner._target_backend._addresses.write,
            scratch.results.frontiers,
            scratch.cursors,
        ]
        for buffer in buffers:
            buffer.fill_(-7)
        runner._plans[size].draft[0].input_ids.fill_(-7)
        scratch.round_done.fill_(True)
        runner._target_backend._addresses.indptr.fill_(0)
        for layer in range(draft_layers):
            for cache in draft_pages[layer]:
                cache.flatten(0, 1)[written] = -7

    temperatures = {
        Verification.GREEDY: (0.0, None),
        Verification.SAMPLED: (1.0, None),
        Verification.FILTERED: (1.0, 5),
    }
    compared, mismatches = 0, []
    with torch.inference_mode():
        for size, row_sets in _ROW_SETS.items():
            for verification, (temperature, top_k) in temperatures.items():
                for rows in row_sets:
                    chosen = [requests[row] for row in rows]
                    specs = [
                        RequestSpec(
                            request.slot.index,
                            request.num_prompt_tokens,
                            request.max_device_len,
                            TokenSampling(temperature=temperature, top_k=top_k, ignore_eos=True),
                            seed=request.uid,
                        )
                        for request in chosen
                    ]
                    staged = RoundInput.stage(
                        Round(tuple(spec.slot for spec in specs)),
                        specs,
                        capacity=runner.max_batch_size,
                        vocab_size=runner.vocab_size,
                        padding=(),
                    )
                    assert staged.verification is verification
                    scratch.args.words.copy_(staged.args.words)
                    plan = runner._plans[size]
                    runner._draft(plan)
                    eager = outputs(size)
                    assert not round_write_fault(
                        eager["cache_write_indices"],
                        eager["positions"],
                        eager["round_done"],
                        cache_tokens=capacity.num_pages * capacity.page_size,
                        max_seq_len=capacity.max_seq_len,
                    )
                    poison(size, verification, eager["cache_write_indices"].long())
                    runner._draft_graphs[size].replay()
                    replayed = outputs(size)
                    torch.cuda.synchronize()
                    compared += 1
                    mismatches.extend(
                        f"{size}/{verification.name}/{rows}/{name}"
                        for name, value in eager.items()
                        if not torch.equal(value, replayed[name])
                    )
    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "compared": compared,
                "mismatches": mismatches,
                "captured": sorted(runner._draft_graphs),
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
            }
        )
    )
    worker.close()


def test_a_draft_graph_replays_what_the_draft_runs_eagerly(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "dflash-draft-graph")
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    env["PYTHONNOUSERSITE"] = "1"
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_dflash_draft_graph", str(checkpoint_root)),
        log_root / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert receipt["captured"] == [1, 2, 3, 4]
    assert receipt["compared"] == 3 * sum(len(sets) for sets in _ROW_SETS.values())
    assert receipt["mismatches"] == []
    assert_cuda_device_metadata(receipt)


if __name__ == "__main__":
    _run_child(sys.argv[1])
