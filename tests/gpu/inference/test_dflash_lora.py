"""DFlash with a LoRA target on one GPU: with the same adapter, trained one SFT step in the
engine, greedy DFlash follows ordinary decoding token for token until a near tie, with
matching raw selected logprobs, and sampled DFlash rounds return processed logprobs. Each
engine runs in its own child."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from tests.support.dflash import PINNED_CHECKPOINTS
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT

RECEIPT_PREFIX = "CTI_DFLASH_LORA="
PROMPTS = (
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "List three prime numbers:",
    "Water boils at",
)
MAX_TOKENS = 48
SEED = 20261017
LORA_RANK = 16
LORA_ALPHA = 32.0
# One step at this rate moves the adapter far enough to change what the target says.
LEARNING_RATE = 5e-3
TRAINED_ANSWER = " Paris, the largest city of France, on the Seine."
# Greedy paths may part at a near tie (verify rows run at M = B * Q, decode at M = B); before
# they part, raw logprob gaps are compared over at least this fraction of the tokens.
MIN_COMPARED_FRACTION = 0.5
# Verify rows run at M = B * Q and decode rows at M = B: BF16 logits differ in the last bits,
# steps of 0.125 near 16-32 (one measured gap: 0.103), as in test_sampling_features.py.
LOGPROB_TOLERANCE = 0.25
_V1_PAIRS = [pair for pair in PINNED_CHECKPOINTS if pair.architecture == "DFlashDraftModel"]


def _snapshot(checkpoint) -> str:
    return str(checkpoint.snapshot())


def _run_child(target_repo: str, dflash: bool) -> None:
    import torch
    from transformers import AutoTokenizer

    from cti.config import (
        AdamWConfig,
        BatchLayout,
        CacheOptions,
        EngineOptions,
        EngineTuning,
        LoRAConfig,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        SpeculativeOptions,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.llm import LLM
    from cti.models import get_model_class
    from cti.protocol import SamplingParams
    from cti.training import SFTDatum, TrainingBatch

    pair = next(pair for pair in _V1_PAIRS if pair.target.repo_id == target_repo)
    snapshot = _snapshot(pair.target)
    architecture = json.loads((Path(snapshot) / "config.json").read_text())["architectures"][0]
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=snapshot, max_seq_len=256),
            # The 8B and 9B targets beside their drafts and a training step.
            memory=MemoryOptions(inference_ratio=0.75, peak_ratio=0.92),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=256),
            cache=CacheOptions(prefix="naive"),
            speculative=SpeculativeOptions(model_path=_snapshot(pair.draft)) if dflash else None,
            lora=LoRAConfig(
                targets=get_model_class(architecture).LORA_TARGETS,
                rank=LORA_RANK,
                alpha=LORA_ALPHA,
            ),
            training=TrainingConfig(
                trainable_plan=TrainablePlan.all_declared(),
                optimizer=AdamWConfig(lr=LEARNING_RATE, weight_decay=0.0, max_grad_norm=1.0),
                batching=TrainingBatching(
                    layout=BatchLayout("packed"),
                    policy=StaticBatching(microbatch_size=1, max_tokens_per_sample=256),
                ),
            ),
        ),
        tuning=EngineTuning(cuda_graph_bs=(1, 2, 4)),
    )
    try:

        def sample(**policy: object) -> list[dict[str, list]]:
            params = [
                SamplingParams(
                    max_new_tokens=MAX_TOKENS, ignore_eos=True, seed=SEED + index, **policy
                )
                for index in range(len(PROMPTS))
            ]
            result = llm.sample(list(PROMPTS), params)
            samples = sorted(result.samples, key=lambda sample: sample.request_id)
            return [
                {"tokens": list(s.sampled_token_ids), "logprobs": list(s.sampling_logprobs)}
                for s in samples
            ]

        greedy = {"temperature": 0.0, "logprobs": True}
        base = sample(**greedy)
        # One SFT step moves the adapter off the identity, the same in both engines.
        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        prompt, answer = tokenizer.encode(PROMPTS[0]), tokenizer.encode(TRAINED_ANSWER)
        datum = SFTDatum(
            token_ids=tuple(prompt + answer),
            target_mask=(False,) * len(prompt) + (True,) * len(answer),
        )
        with llm.training_session() as session:
            completion = session.forward_backward(TrainingBatch.sft([datum], BatchLayout.PACKED))
            session.optim_step().result()
            assert llm.model_revision == 0
            assert sample(**greedy) == base
            session.publish()
            completion.result()
        receipt = {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "base": base,
            "adapted": sample(**greedy),
            "sampled": sample(temperature=1.0, logprobs=True, logprobs_processed=True),
        }
    finally:
        llm.shutdown()
    print(RECEIPT_PREFIX + json.dumps(receipt))


def _engine(tmp_path: Path, target_repo: str, dflash: bool) -> dict:
    name = f"dflash-lora-{'dflash' if dflash else 'plain'}"
    env, log_root = isolated_runtime_env(tmp_path, name)
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_dflash_lora", target_repo, str(dflash)),
        log_root / f"{name}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )
    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert_cuda_device_metadata(receipt)
    return receipt


@pytest.mark.parametrize(
    "target_repo",
    [
        pytest.param(
            pair.target.repo_id,
            marks=pytest.mark.checkpoints(pair.target, pair.draft),
            id=pair.target.repo_id,
        )
        for pair in _V1_PAIRS
    ],
)
def test_dflash_with_a_lora_target_follows_ordinary_decoding(
    tmp_path: Path, target_repo: str
) -> None:
    plain = _engine(tmp_path, target_repo, dflash=False)
    dflash = _engine(tmp_path, target_repo, dflash=True)

    # The adapter changes what the target says.
    assert dflash["adapted"] != dflash["base"]
    compared = 0
    for expected, actual in zip(plain["adapted"], dflash["adapted"], strict=True):
        parted = next(
            (
                i
                for i, (a, b) in enumerate(zip(expected["tokens"], actual["tokens"], strict=True))
                if a != b
            ),
            MAX_TOKENS,
        )
        compared += parted
        for index in range(parted):
            gap = abs(expected["logprobs"][index] - actual["logprobs"][index])
            assert gap <= LOGPROB_TOLERANCE, (index, expected, actual)
    assert compared >= MIN_COMPARED_FRACTION * MAX_TOKENS * len(PROMPTS)
    for sample in dflash["sampled"]:
        assert len(sample["logprobs"]) == MAX_TOKENS
        assert all(math.isfinite(value) and value <= 0.0 for value in sample["logprobs"])


if __name__ == "__main__":
    _run_child(sys.argv[1], sys.argv[2] == "True")
