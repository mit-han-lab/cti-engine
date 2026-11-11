"""Numeric qualification and joint-prefix reuse under eviction pressure, judged on what
the public ``LLM`` reports: greedy tokens, their logprobs and cached prompt tokens."""

import json
import sys
from contextlib import contextmanager

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.gpu import engine_checks_off, offline_child_env, run_isolated_process
from tests.support.hf_truth import PROMPTS_PATH
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import selected_checkpoint

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

CONTINUATION = 128


def _child(tmp_path, workload):
    env, logs = offline_child_env(tmp_path, "prefix-" + workload)
    output = run_isolated_process(
        (sys.executable, "-m", __name__, workload, str(selected_checkpoint().snapshot())),
        logs / (workload + ".log"),
        env=env,
        cwd=REPO_ROOT,
        timeout=3600,
    )
    assert "PREFIX_PASSED" in output


def test_concurrent_prefix_capture_graph_and_eviction(tmp_path):
    _child(tmp_path, "pressure")


def test_prefix_selected_logprobs_against_naive_and_aa(tmp_path):
    _child(tmp_path, "numeric")


@pytest.mark.parametrize("pairs", [1, 4, 17])
def test_cross_layer_copy_is_bitwise_and_bounds_temporary_memory(pairs):
    import torch

    from cti.kvcache import RecurrentStatePool
    from cti.layers.gated_delta_net.layer import gated_delta_net_state

    pool = RecurrentStatePool(
        [gated_delta_net_state(2, 128, 128, 512)] * 3,
        num_slots=pairs,
        num_snapshot_rows=pairs,
        device=torch.device("cuda"),
    )
    for layer in range(pool.num_layers):
        for view in pool.layer_views(layer):
            view.normal_()
    sources = torch.arange(pairs, device="cuda", dtype=torch.int64)
    targets = sources + pairs
    indices = torch.stack((sources, targets))
    # Compile before measuring allocations, then compare all destination bits.
    pool.copy_rows(indices)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with engine_checks_off():
        pool.copy_rows(indices)
    torch.cuda.synchronize()
    # RowCopy takes no scratch.
    assert torch.cuda.max_memory_allocated() == before
    for layer in range(pool.num_layers):
        for view in pool.layer_views(layer):
            assert torch.equal(view[:pairs], view[pairs:])


def test_decode_export_copies_the_replayed_rows_bitwise_and_nothing_else():
    from types import SimpleNamespace

    import torch

    from cti.forward import RecurrentExport
    from cti.kvcache import RecurrentStatePool
    from cti.layers.gated_delta_net.layer import gated_delta_net_state
    from cti.recurrent import RecurrentBackend
    from tests.support.forward_rows import decode_entry, forward_rows

    device = torch.device("cuda")
    pool = RecurrentStatePool(
        [gated_delta_net_state(2, 128, 128, 512)] * 3,
        num_slots=4,
        num_snapshot_rows=4,
        device=device,
    )
    views = [view for layer in range(pool.num_layers) for view in pool.layer_views(layer)]
    for view in views:
        view.normal_()
    # A captured step advances slots 1 and 3 in place; their exports follow its replay.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for view in views:
            view[1:4:2].mul_(1.5).add_(1)
    entries = (
        decode_entry(1, 63, captures=(RecurrentExport(64, 5),)),
        decode_entry(2, 9),
        decode_entry(3, 127, captures=(RecurrentExport(128, 7),)),
    )
    backend = RecurrentBackend(pool, device=device)
    metadata = backend.prepare_metadata(forward_rows(*entries))
    before = [view.clone() for view in views]
    graph.replay()
    backend.export(SimpleNamespace(recurrent_metadata=metadata))
    torch.cuda.synchronize()
    untouched = [0, 2, 4, 6]  # other slots and snapshot-row canaries
    for view, old in zip(views, before, strict=True):
        assert not torch.equal(view[1], old[1])
        assert torch.equal(view[5], view[1]) and torch.equal(view[7], view[3])
        assert torch.equal(view[untouched], old[untouched])


@contextmanager
def _cold(snapshot, budget):
    """Serve singletons on the naive cache, which never restores a prefix."""
    from tests.support.qwen3_5_prefix import greedy, prefix_llm

    llm = prefix_llm(snapshot, cache="naive", budget=budget)

    def serve(prompt, length=CONTINUATION):
        (run,) = greedy(llm, [prompt], length)
        assert run.cached == 0
        return run

    try:
        yield serve
    finally:
        llm.shutdown()


# Where a 600-token prompt exports its one snapshot: its deepest restorable grid point. Decodes
# stay under 40 tokens, short of the next grid point 640, so prompts export every row leased.
PRESSURE_EXPORT = 576


def _pressure(snapshot):
    """Four groups share two snapshot rows. Leader A exports and decodes longest; its
    follower restores the export, then later leaders' exports reclaim A's row while A still
    decodes. Each other follower restores its leader's export if it survived, else
    recomputes."""
    from tests.support.qwen3_5_prefix import greedy, judge_prefix, prefix_llm

    joint = [16 + i % 100 for i in range(600)]
    a, b, c, d = ([token + offset for token in joint] for offset in (200, 400, 600, 800))
    # Admitted first, A decodes longest, so it closes last: whatever reclaims its row does so
    # while it decodes. Followers queue behind other leaders.
    pressure = [(a, 39), (b, 4), (a, 8), (c, 4), (d, 4), (b, 8), (c, 8), (d, 8)]
    leaders = (0, 1, 3, 4)
    workload = [(joint, 16), (a, 39), (b, 8), (c, 8), (d, 8)]

    def references(budget):
        with _cold(snapshot, budget) as serve:
            return {tuple(prompt): serve(prompt, length) for prompt, length in workload}

    truth, aa = references(1024), references(256)
    llm = prefix_llm(snapshot, budget=256, rows=2)
    try:
        shared = greedy(llm, [joint] * 4, 16)
        assert [run.cached for run in shared] == [0] + [PRESSURE_EXPORT] * 3, shared
        runs = greedy(llm, [prompt for prompt, _ in pressure], [n for _, n in pressure])
        cached = [run.cached for run in runs]
        followers = [cached[i] for i in range(len(runs)) if i not in leaders]
        assert all(cached[i] == 0 for i in leaders), cached
        assert set(followers) <= {0, PRESSURE_EXPORT} and PRESSURE_EXPORT in followers, cached
        # A's KV stays cached; only a reclaimed row makes it recompute.
        again = greedy(llm, [a], 8)
        assert again[0].cached == 0, again
        # The pressure reclaimed the joint prefix's row too; serving it once exports it again.
        reused = greedy(llm, [joint], 16) + greedy(llm, [joint], 16)
        assert [run.cached for run in reused] == [0, PRESSURE_EXPORT], reused
    finally:
        llm.shutdown()
    judge_prefix(
        [(truth[key], aa[key]) for key in truth],
        [(truth[tuple(joint)], run) for run in shared + reused]
        + [(truth[tuple(prompt)], run) for (prompt, _), run in zip(pressure, runs, strict=True)]
        + [(truth[tuple(a)], again[0])],
    )


# Hit budgets: 1024 ends a chunk at the branch point; 1023 rounds an unfinished chunk down to
# the 64-token grid (960), and the next forward stops at the branch point to export it.
BUDGETS = (1024, 1023)
# Restored length of each branch prompt after its predecessors, whatever the budget: a prompt
# exports only its deepest restorable point, so branch-1 matches past branch-0's 1280 and
# recomputes, exporting the branch point 1024 that branch-2 restores.
BRANCH_HITS = (0, 0, 1024)
# Decoding 600..727 last exported 704, published when the request closed.
EXTENSION_HIT = 704


def _prompts():
    """Frozen chat/raw seeds, including a long shared system and multi-turn history."""
    seeds = json.loads(PROMPTS_PATH.read_text())["prompts"]
    key = selected_checkpoint().key
    by_name = {seed["name"]: seed["token_ids"][key] for seed in seeds}

    def repeat(tokens, length):
        return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]

    names = (
        "lighthouse-log",
        "multi-turn-arithmetic",
        "fibonacci-raw",
        "gcd-raw",
        "compiler-allocation",
        "python-fibonacci",
        "chinese-blue-sky",
        "translate-french",
    )
    lengths = (300, 600, 1025, 1026, 1290, 2049, 2500, 3000)
    prompts = {
        name: repeat(by_name[name], length) for name, length in zip(names, lengths, strict=True)
    }
    shared = repeat(by_name["system-noble-gases"], 1024)
    # Distinct first suffix tokens make the branch point exactly 1024.
    for index in range(3):
        prompts[f"branch-{index}"] = shared + [16 + index] + repeat(by_name[names[index]], 299)
    return prompts


def _hits(snapshot, budget, prompts, truth):
    """Seed each prompt, then serve four copies that restore its last 64-aligned export; a
    branch restores its predecessors' shared prefix, the extension its parent's decode."""
    from tests.support.qwen3_5_prefix import bitwise, greedy, prefix_llm

    served = {}
    llm = prefix_llm(snapshot, budget=budget, rows=8)
    try:
        for name, prompt in prompts.items():
            if name.startswith("branch-"):
                runs = greedy(llm, [prompt], CONTINUATION)
                expected = BRANCH_HITS[int(name[-1])]
            elif name == "multi-turn-extension":
                # Decode the parent again, so its exports are the newest; it decoded the
                # extension's history only if it agreed with the truth.
                (parent,) = greedy(llm, [prompts["multi-turn-arithmetic"]], CONTINUATION)
                runs = greedy(llm, [prompt], CONTINUATION)
                agreed = parent.ids == truth["multi-turn-arithmetic"].ids
                expected = EXTENSION_HIT if agreed else runs[0].cached
            else:
                assert greedy(llm, [prompt], 2)[0].cached == 0
                runs = greedy(llm, [prompt] * 4, CONTINUATION)
                expected = (len(prompt) - 1) // 64 * 64
            assert [run.cached for run in runs] == [expected] * len(runs), (name, runs)
            served[name] = runs
            if budget == 1024 and expected == 1024 and name in ("fibonacci-raw", "gcd-raw"):
                # A seed forward that ends at the boundary exports exactly the state a cold
                # prefill carries into its next chunk, so the hit is bit-identical.
                (run,) = greedy(llm, [prompt], CONTINUATION)
                assert run.cached == 1024 and bitwise(run, truth[name]), name
    finally:
        llm.shutdown()
    return [(truth[name], run) for name, runs in served.items() for run in runs]


def _numeric(snapshot):
    from tests.support.qwen3_5_prefix import judge_prefix

    prompts = _prompts()
    with _cold(snapshot, 1024) as serve:
        truth = {name: serve(prompt) for name, prompt in prompts.items()}
        prompts["multi-turn-extension"] = (
            prompts["multi-turn-arithmetic"]
            + list(truth["multi-turn-arithmetic"].ids)
            + [42] * CONTINUATION
        )
        truth["multi-turn-extension"] = serve(prompts["multi-turn-extension"])
    with _cold(snapshot, 256) as serve:
        calibration = [(truth[name], serve(prompt)) for name, prompt in prompts.items()]
    for budget in BUDGETS:
        judge_prefix(calibration, _hits(snapshot, budget, prompts, truth))


if __name__ == "__main__":
    from tests.support.qwen3_5_prefix import child_main

    def run():
        {"pressure": _pressure, "numeric": _numeric}[sys.argv[1]](sys.argv[2])
        print("PREFIX_PASSED", flush=True)

    child_main(run)
