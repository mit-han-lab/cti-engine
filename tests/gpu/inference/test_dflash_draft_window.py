"""Greedy Qwen3.5-9B DFlash past its draft's sliding window follows ordinary decoding.

Each prompt is about 900 tokens longer than a draft window: the draft's own (4096) or its config
rewritten to a narrower or a wider one. DFlash's greedy output must equal ordinary greedy
decoding's, or leave it only where the FP32 truth, teacher-forced on their common prefix, is a
near tie between their two tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_5_9B, QWEN3_5_9B_DFLASH
from tests.support.dflash import PINNED_CHECKPOINTS, preflight
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import receipts_dir, write_receipt
from tests.support.reference_truth import SequenceTruth, judge_divergence

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B, QWEN3_5_9B_DFLASH)

_PAIR = next(pair for pair in PINNED_CHECKPOINTS if pair.target.repo_id == "Qwen/Qwen3.5-9B")
_CONFIG_WINDOW = 4096
_WINDOWS = {"config": None, "narrower": 2560, "wider": 8192}
# Context tokens past the window, so sliding layers drop the earliest keys.
_PAST_WINDOW = 904
_TOKENS = 64


def _draft_with_window(draft: str, window: int, folder: Path) -> str:
    """``draft`` with its config's ``sliding_window`` set to ``window``, its other files
    linked."""

    folder.mkdir()
    for source in Path(draft).iterdir():
        if source.name != "config.json":
            (folder / source.name).symlink_to(source.resolve())
    config = json.loads((Path(draft) / "config.json").read_text())
    (folder / "config.json").write_text(json.dumps({**config, "sliding_window": window}))
    return str(folder)


def _prompts(tokenizer) -> dict[str, list[int]]:
    sources = [REPO_ROOT / "LICENSE", REPO_ROOT / "THIRD_PARTY_NOTICES"]
    text = "\n\n".join(
        path.read_text() for path in (*sources, *sorted(REPO_ROOT.glob("docs/*.md")))
    )
    ids = tokenizer(text)["input_ids"]
    prompts = {}
    for name, window in _WINDOWS.items():
        tokens = (window or _CONFIG_WINDOW) + _PAST_WINDOW
        assert len(ids) >= tokens, len(ids)
        content = tokenizer.decode(ids[:tokens]) + "\n\nSummarize the text above."
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts[name] = tokenizer(chat)["input_ids"]
    return prompts


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
            model=ModelOptions(path=target, max_seq_len=16384),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=8192),
            cache=CacheOptions(prefix="naive"),
            speculative=None if draft is None else SpeculativeOptions(model_path=draft),
        ),
        tuning=EngineTuning(cuda_graph_bs=(1,)),
    )
    params = SamplingParams(temperature=0.0, max_new_tokens=_TOKENS, ignore_eos=True)
    try:
        return [
            list(llm.sample([prompt], params).samples[0].sampled_token_ids) for prompt in prompts
        ]
    finally:
        llm.shutdown()


def test_dflash_past_each_draft_window_follows_ordinary_decoding(tmp_path: Path) -> None:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    # Missing pinned checkpoints raise QualificationBlocked: they fail the run, never skip.
    snapshots = preflight(_PAIR)["snapshots"]
    tokenizer = AutoTokenizer.from_pretrained(snapshots["target"], local_files_only=True)
    prompts = _prompts(tokenizer)
    outputs = _greedy(snapshots["target"], None, list(prompts.values()))
    plain = dict(zip(_WINDOWS, outputs, strict=True))
    divergences = {}
    for name, window in _WINDOWS.items():
        draft = snapshots["draft"]
        if window is not None:
            draft = _draft_with_window(draft, window, tmp_path / f"draft-{name}")
        (dflash,) = _greedy(snapshots["target"], draft, [prompts[name]])
        assert len(dflash) == _TOKENS, name
        pairs = enumerate(zip(plain[name], dflash, strict=True))
        step = next((i for i, (a, b) in pairs if a != b), None)
        if step is not None:
            divergences[name] = (step, (plain[name][step], dflash[step]))

    rulings = {}
    if divergences:
        checkpoint = _PAIR.target
        sequences = {(tuple(prompts[name]), tuple(plain[name])) for name in divergences}
        job = (receipts_dir() or tmp_path) / "dflash-draft-window-truth.pt"
        truth, _ = SequenceTruth(checkpoint, Path(snapshots["target"]), tmp_path).require(
            sequences, job
        )
        rulings = {
            name: judge_divergence(truth[tuple(prompts[name]), tuple(plain[name])], *divergence)
            for name, divergence in divergences.items()
        }
    write_receipt(
        "dflash-draft-window",
        {"prompt_tokens": {name: len(p) for name, p in prompts.items()}, "rulings": rulings},
    )
    assert all(ruling["allowed"] for ruling in rulings.values()), rulings
