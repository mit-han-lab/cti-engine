"""Real LoRA commits and no-op sessions through the public train/infer entry."""

import sys

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.gpu import offline_child_env, run_isolated_process
from tests.support.graph_captures import graph_captures
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import selected_checkpoint

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

# Switch cycles, the first _UPDATES of which commit a LoRA update. From the first cycle after
# them, device memory must stay at that cycle's, so a leak per cycle fails in the next one.
_CYCLES = 6
_UPDATES = 3
# Device memory is read from outside the engine, where the CUDA driver and caching allocator
# round in 2 MiB pages; this allows a few pages of that rounding, far below one leaked chunk
# of KV pages or snapshot rows.
_DRIFT_BYTES = 16 << 20
_RELEASED = "PREFIX_SWITCH_RELEASED_EMPTY"


def test_qwen3_5_prefix_cache_training_switch(tmp_path):
    env, logs = offline_child_env(tmp_path, "prefix-switch")
    output = run_isolated_process(
        (sys.executable, "-m", __name__, str(selected_checkpoint().snapshot())),
        logs / "switch.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=2400,
    )
    assert f"PREFIX_SWITCH_PASSED updates={_UPDATES} cycles={_CYCLES}" in output
    # Switches keep the decode graphs captured at startup.
    log = (logs / "switch.log").read_text()
    assert graph_captures(log, "decode") == 1, log
    # Every switch's release left the radix, the snapshot rows and the encoder cache empty.
    assert log.count(_RELEASED) >= _CYCLES, log


def _run(snapshot):
    from pathlib import Path

    import torch

    from cti.config import (
        AdamWConfig,
        BatchLayout,
        LoRAConfig,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.training import SFT, PaddedTokens, TrainingBatch
    from tests.support.qwen3_5_prefix import (
        bitwise,
        device_used_bytes,
        greedy,
        near_tie_divergence,
        prefix_llm,
    )

    device_used_bytes()  # this process's CUDA context exists before the engine boots
    weight_bytes = sum(path.stat().st_size for path in Path(snapshot).glob("*.safetensors"))
    names = ("model.layers.0.mlp.down_proj.full.lora_A", "model.layers.0.mlp.down_proj.full.lora_B")
    llm = prefix_llm(
        snapshot,
        budget=256,
        rows=2,
        max_seq_len=1024,
        lora=LoRAConfig(targets=("mlp",), rank=4, alpha=8),
        training=TrainingConfig(
            trainable_plan=TrainablePlan(names),
            optimizer=AdamWConfig(lr=1e-3, weight_decay=0, max_grad_norm=1),
            batching=TrainingBatching(
                BatchLayout.PADDED, StaticBatching(microbatch_size=1, max_tokens_per_sample=32)
            ),
        ),
    )
    prompt = [16 + i % 100 for i in range(600)]
    batch = TrainingBatch(
        PaddedTokens(torch.tensor([prompt[:32]], dtype=torch.int64), torch.tensor([32]), 0),
        torch.ones((1, 31), dtype=torch.bool),
        SFT(),
    )
    try:
        previous, steady = None, None
        for cycle in range(_CYCLES):
            # Every switch empties the prefix cache: the first serve is cold, the next restores
            # the last 64-aligned export.
            (cold,) = greedy(llm, [prompt], 16)
            (warm,) = greedy(llm, [prompt], 16)
            assert (cold.cached, warm.cached) == (0, 576), (cycle, cold, warm)
            assert near_tie_divergence(cold, warm), (cycle, cold, warm)
            if previous is not None:
                # A committed update changes the outputs; a no-op session leaves them bitwise.
                assert bitwise(cold, previous) == (cycle > _UPDATES), cycle
            previous = cold
            revision = llm.model_revision
            before = device_used_bytes()
            with llm.training_session() as session:
                if cycle < _UPDATES:
                    completion = session.forward_backward(batch)
                    assert session.optim_step().result().step == revision + 1
                    session.publish()
                    assert completion.result().normalization_count == 31
                # Training shares the engine's weights and state rather than copying them.
                assert device_used_bytes() - before < weight_bytes, cycle
            assert llm.model_revision == revision + int(cycle < _UPDATES)
            used = device_used_bytes()
            if cycle >= _UPDATES:  # optimizer state and all inference shapes have warmed up
                steady = used if steady is None else steady
                assert abs(used - steady) <= _DRIFT_BYTES, (cycle, used, steady)
    finally:
        llm.shutdown()
    print(f"PREFIX_SWITCH_PASSED updates={_UPDATES} cycles={_CYCLES}", flush=True)


def _check_releases():
    """In the spawned engine process: after each release, check that nothing is cached."""

    from cti.inference import InferenceEngine

    release = InferenceEngine.release

    def checked(self):
        release(self)
        occupancy = self._cache.occupancy
        assert (occupancy.num_used_pages, occupancy.num_cached_pages) == (0, 0), occupancy
        assert (occupancy.num_used_snapshot_rows, occupancy.num_cached_snapshot_rows) == (0, 0)
        assert not self._encoder_cache._images
        print(_RELEASED, flush=True)

    InferenceEngine.release = checked


if __name__ == "__mp_main__":
    _check_releases()

if __name__ == "__main__":
    from tests.support.qwen3_5_prefix import child_main

    child_main(_run, sys.argv[1])
