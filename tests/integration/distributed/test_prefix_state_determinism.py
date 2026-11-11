"""TP-agnostic host decisions under independent interpreter hash seeds."""

import json
import os
import subprocess
import sys


def test_snapshot_admission_and_lru_are_independent_of_hash_seed():
    traces = []
    for seed in ("7", "923"):
        result = subprocess.run(
            [sys.executable, "-m", "tests.support.inference.prefix_cache"],
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        traces.append(json.loads(result.stdout.splitlines()[-1]))
    assert traces[0] == traces[1]
    (roomy, roomy_evicted), (tight, tight_evicted) = traces[0]
    assert roomy_evicted == 0 < tight_evicted
    for trace in (roomy, tight):
        assert any(entry[3] is not None for step in trace for entry in step)
        rows = [row for step in trace for entry in step for _, row in entry[4]]
        assert len(rows) > len(set(rows)), "snapshot rows are reused"
