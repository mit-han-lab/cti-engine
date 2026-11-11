"""Prefix hits after a training switch restore their recurrent snapshot rows.

A switch brings the recurrent rows back with undefined contents; the engine process fills
them with NaN after every resume, so a request reads only what serving wrote since. After a
no-op training session, a cold prompt exports its snapshot and two prompts sharing its
prefix restore it into their slots, one of them a slot nothing wrote since the resume:
both must sample bitwise what the same commands sampled before the switch.
"""

import sys

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.gpu import offline_child_env, run_isolated_process
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import selected_checkpoint

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

_POISONED = "PREFIX_RESTORE_POISONED_ROWS"
_PASSED = "PREFIX_RESTORE_PASSED"


def test_qwen3_5_prefix_restore_after_switch(tmp_path):
    env, logs = offline_child_env(tmp_path, "prefix-restore")
    output = run_isolated_process(
        (sys.executable, "-m", __name__, str(selected_checkpoint().snapshot())),
        logs / "restore.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=2400,
    )
    assert _PASSED in output
    assert _POISONED in (logs / "restore.log").read_text()


def _run(snapshot):
    from cti.config import (
        AdamWConfig,
        BatchLayout,
        LoRAConfig,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from tests.support.qwen3_5_prefix import bitwise, greedy, prefix_llm

    names = ("model.layers.0.mlp.down_proj.full.lora_A", "model.layers.0.mlp.down_proj.full.lora_B")
    llm = prefix_llm(
        snapshot,
        budget=1024,
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
    # Both restore the cold prompt's export at 576 into their own slots.
    hits = [prompt + [200 + i for i in range(8)], prompt + [300 + i for i in range(8)]]

    def serve():
        (cold,) = greedy(llm, [prompt], 16)
        warm = greedy(llm, hits, 16)
        assert cold.cached == 0 and [run.cached for run in warm] == [576, 576], (cold, warm)
        return warm

    try:
        before = serve()
        with llm.training_session():
            pass
        after = serve()
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            assert bitwise(left, right), (index, left, right)
    finally:
        llm.shutdown()
    print(_PASSED, flush=True)


def _poison_resumed_rows():
    """In the spawned engine process: fill every recurrent row with NaN after each KV
    resume, as the memory may come back."""

    import torch

    from cti.kvcache.recurrent_pool import RecurrentStatePool
    from cti.memory import MemoryManager

    pools = []
    init, resume = RecurrentStatePool.__init__, MemoryManager.resume_kv

    def registered(self, *args, **kwargs):
        init(self, *args, **kwargs)
        pools.append(self)

    def poisoned(self, *args, **kwargs):
        resume(self, *args, **kwargs)
        for pool in pools:
            for index in range(pool.num_layers):
                for view in pool.layer_views(index):
                    view.view(-1).view(torch.uint8).fill_(0xFF)  # NaN in FP32 and BF16
        print(_POISONED, flush=True)

    RecurrentStatePool.__init__ = registered
    MemoryManager.resume_kv = poisoned


if __name__ == "__mp_main__":
    _poison_resumed_rows()

if __name__ == "__main__":
    from tests.support.qwen3_5_prefix import child_main

    child_main(_run, sys.argv[1])
