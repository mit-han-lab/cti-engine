"""Greedy Qwen3.8-27B DFlash2 follows ordinary greedy decoding, B = 1..4 in flight (TP2:
test_dflash_tensor_parallel.py).

Each parity prompt's DFlash2 output, through its stop token, must equal ordinary greedy
decoding's, or first leave it only where the FP32 truth, teacher-forced on their common prefix,
is a near tie of both tokens with its top-1. The BF16 target needs an RTX PRO 6000.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_8_27B, QWEN3_8_27B_DFLASH2
from tests.support.dflash import PINNED_CHECKPOINTS, preflight
from tests.support.gpu import require_device_memory
from tests.support.hf_truth import PROMPTS_PATH, tokenize_prompt
from tests.support.qwen3_5_gpu import receipts_dir, write_receipt
from tests.support.reference_truth import SequenceTruth, judge_divergence

pytestmark = pytest.mark.checkpoints(QWEN3_8_27B, QWEN3_8_27B_DFLASH2)

_PAIR = next(pair for pair in PINNED_CHECKPOINTS if pair.architecture == "DFlash2DraftModel")
_TOKENS = 128


def _greedy(target: str, draft: str | None, prompts: list[list[int]]) -> list[list[int]]:
    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        SpeculativeOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=target, max_seq_len=4096),
            memory=MemoryOptions(inference_ratio=0.85),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=2048),
            cache=CacheOptions(prefix="radix"),
            speculative=None if draft is None else SpeculativeOptions(model_path=draft),
        ),
        tuning=EngineTuning(cuda_graph_bs=(1, 2, 3, 4)),
    )
    params = SamplingParams(temperature=0.0, max_new_tokens=_TOKENS)
    try:
        result = llm.sample(prompts, [params] * len(prompts))
        return [list(sample.sampled_token_ids) for sample in result.samples]
    finally:
        llm.shutdown()


def test_dflash2_follows_ordinary_decoding_until_a_truth_near_tie(tmp_path: Path) -> None:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    require_device_memory(90)
    # Missing pinned checkpoints raise QualificationBlocked: they fail the run, never skip.
    snapshots = preflight(_PAIR)["snapshots"]
    tokenizer = AutoTokenizer.from_pretrained(snapshots["target"], local_files_only=True)
    frozen = json.loads(PROMPTS_PATH.read_text())
    names = [prompt["name"] for prompt in frozen["prompts"]]
    prompts = [
        tokenize_prompt(tokenizer, prompt, frozen["chat_template_kwargs"])
        for prompt in frozen["prompts"]
    ]
    plain = _greedy(snapshots["target"], None, prompts)
    dflash2 = _greedy(snapshots["target"], snapshots["draft"], prompts)
    divergences = {}
    for name, prompt, ours, theirs in zip(names, prompts, plain, dflash2, strict=True):
        # Both stop at the first stop token, so outputs of different lengths differ before it.
        step = next((i for i, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b), None)
        if step is None:
            assert ours == theirs, name
        else:
            divergences[name] = (prompt, ours, step, (ours[step], theirs[step]))

    rulings = {}
    if divergences:
        checkpoint = _PAIR.target
        sequences = {(tuple(prompt), tuple(ours)) for prompt, ours, _, _ in divergences.values()}
        job = (receipts_dir() or tmp_path) / "dflash2-truth.pt"
        truth, _ = SequenceTruth(checkpoint, Path(snapshots["target"]), tmp_path).require(
            sequences, job
        )
        rulings = {
            name: judge_divergence(truth[tuple(prompt), tuple(ours)], step, tokens)
            for name, (prompt, ours, step, tokens) in divergences.items()
        }
    write_receipt("dflash2-truth", {"identical": len(names) - len(rulings), "rulings": rulings})
    assert all(ruling["allowed"] for ruling in rulings.values()), rulings
