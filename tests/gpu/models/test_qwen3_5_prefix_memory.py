"""Snapshot rows, KV pages and row copies at the production budget entry stay within the
budget the engine reports, measured on the device from outside the engine, with and without
prefix reuse."""

import json
import re
import sys

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.gpu import offline_child_env, run_isolated_process
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import selected_checkpoint

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

_GIB = 1 << 30
# The engine logs memory in GiB to two decimals.
_LOGGED = 0.005 * _GIB
_WORKLOAD = _GIB // 4


def _logged_gib(log, pattern):
    (value,) = re.findall(pattern + r" ([0-9.]+) GiB", log)
    return float(value) * _GIB


@pytest.mark.parametrize("sizing", ["rows", "ratio", "naive"])
def test_prefix_cache_real_memory_budget(tmp_path, sizing):
    env, logs = offline_child_env(tmp_path, "prefix-memory-" + sizing)
    output = run_isolated_process(
        (sys.executable, "-m", __name__, str(selected_checkpoint().snapshot()), sizing),
        logs / "memory.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=1800,
    )
    (line,) = [line for line in output.splitlines() if line.startswith("PREFIX_MEMORY=")]
    free = json.loads(line.removeprefix("PREFIX_MEMORY="))
    log = (logs / "memory.log").read_text()
    (rows,) = re.findall(r"Inference capacity: \d+ KV pages, (\d+) snapshot rows", log)
    # Explicit rows hold a full copy group of 16 and its remainder; the naive cache keeps none.
    expected = {"rows": int(rows) == 17, "ratio": int(rows) >= 2, "naive": int(rows) == 0}
    assert expected[sizing], rows
    boot = _logged_gib(log, "Free memory before loading model:")
    budget = _logged_gib(log, "Memory budget: inference")
    graphs = _logged_gib(log, r"Captured decode graphs of sizes \[[^\]]*\]: decode")
    receipt = {"boot": boot, "budget": budget, "graphs": graphs, **free}
    # Weights, workspaces, KV pages and snapshot rows fit the inference budget beside the graphs
    # (RTX 5090, Qwen3.5-9B: 0.09 GiB to spare).
    assert boot - free["idle"] <= budget + graphs + 4 * _LOGGED, receipt
    # Restores and exports copy rows through the planned workspace: the workload adds only its
    # activations (0.09 GiB), far below one temporary group of 16 snapshot rows.
    assert free["idle"] - free["peak"] <= _WORKLOAD, receipt


def _run(snapshot, sizing):
    from threading import Event, Thread

    import torch

    from tests.support.qwen3_5_prefix import greedy, near_tie_divergence, prefix_llm

    # This process's own CUDA context exists before the engine reads its boot memory.
    torch.cuda.mem_get_info()
    llm = prefix_llm(
        snapshot,
        cache="naive" if sizing == "naive" else "radix",
        budget=256,
        rows=17 if sizing == "rows" else None,
        running=17,
        max_seq_len=1024,
        snapshot_ratio=0.1 if sizing == "ratio" else None,
        # The 35B MoE's 63 GiB of weights leave a 0.7 share too little for 17 rows.
        memory_ratio=0.85 if selected_checkpoint().moe else 0.7,
    )
    try:
        idle = torch.cuda.mem_get_info()[0]
        samples = [idle]
        stopped = Event()

        def sample_device():
            while not stopped.wait(0.002):
                samples.append(torch.cuda.mem_get_info()[0])

        monitor = Thread(target=sample_device)
        monitor.start()
        try:
            prompt = [16 + i % 100 for i in range(600)]
            (cold,) = greedy(llm, [prompt], 16)
            # Seventeen slots restore one snapshot, then seventeen new prompts export theirs;
            # the naive cache serves none of the prompt.
            cached = 0 if sizing == "naive" else 576
            for run in greedy(llm, [prompt] * 17, 16):
                assert run.cached == cached and near_tie_divergence(cold, run), run
            others = [[token + offset for token in prompt] for offset in range(100, 1800, 100)]
            greedy(llm, others, 16)
        finally:
            stopped.set()
            monitor.join(timeout=10)
        assert not monitor.is_alive() and len(samples) > 2
    finally:
        llm.shutdown()
    print("PREFIX_MEMORY=" + json.dumps({"idle": idle, "peak": min(samples)}), flush=True)


if __name__ == "__main__":
    from tests.support.qwen3_5_prefix import child_main

    child_main(_run, *sys.argv[1:])
