"""Prompts folded into rounds on the pinned DFlash pairs follow the FP32 truth: Qwen3-8B and
Qwen3.5-9B with their drafts, greedy, the radix cache, the overlapped engine loop.

The workload runs on two engines behind ``LLM``. The serial arm admits one request at a time
(``max_running_requests=1``), so no prompt ever runs beside rows and nothing folds. The folding
arm runs each arrival group in one bounded sampling call beside an anchor, four slots and a
512-token budget, and folds every prompt that arrives while rows run. Prompt lengths
sit on both sides of the page, draft block, GEMM tier and token budget edges (1 to 2000 tokens,
most after a cached system prompt), and one folded request ends at its first token
(``max_tokens=1``). Which steps fold is the engine's own choice and is not observable here;
forced folds, chunked folds, aborts between chunks and the fold coverage are
``test_dflash_fold``'s and the CPU scheduler tests'.

Every token of every output, in either arm, through its first stop token, must be the FP32
truth's pick on that output's own prefix (``tests.support.reference_truth``), or lie where
the truth's top-two margin is below 1 nat. That margin is a lower bound of how far a pick
other than the truth's top one lies below it, so the check never fails a pick wrongly, but
a pick ranked low where the top two are close passes. Past a stop token, an output that
ignores EOS continues a text the model ended; its picks there are reported. The 1-nat bar
is empirical, a gross-error check for this workload: without folding, the engine's picks
already lie up to 0.50 nats below the truth's here on RTX 5090. So the default tier's
0.25-nat greedy bar is reported for both arms, not judged. Every request must end at its
budget or at a stop token. Where the arms' outputs differ, the first divergence and the
truth's margin there are reported.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.support.dflash import PINNED_CHECKPOINTS, preflight
from tests.support.hf_judge import compare_greedy
from tests.support.qwen3_5_gpu import receipts_dir, write_receipt
from tests.support.reference_truth import SequenceTruth

RECEIPT_PREFIX = "CTI_DFLASH_FOLD_TRUTH="
# The gross-error bar on how far a pick lies below the truth's (see the module docstring).
_GROSS_NATS = 1.0
_SLOTS = 4
_MAX_BATCH_TOKENS = 512
_SYSTEM = (
    "You are the reference desk of a public library. Answer in plain sentences, cite the "
    "shelf a reader should visit, and keep every answer under five sentences. "
) * 6
_BODY = " ".join(f"Item {n}: the number {n} squared is {n * n}." for n in range(1, 1200))
# (role, length, max_tokens, ignore_eos): ``raw`` prompts are ``length`` body tokens with no
# template, every other role the cached system prompt, ``length`` body tokens and the turn's
# end. Each group runs in one sampling call after the anchor, which keeps rows running while
# the members' prompts wait; ``first`` ends at its first token.
_ANCHOR = ("anchor", 40, 160, True)
_GROUPS = (
    (("raw", 1, 48, False), ("raw", 2, 48, False), ("raw", 3, 48, False)),
    (("chat", 48, 300, True), ("chat", 16, 48, False), ("chat", 17, 48, False)),
    (("chat", 101, 160, True), ("long", 2000, 48, False), ("chat", 511, 48, False)),
    (("raw", 15, 48, False), ("raw", 16, 48, False), ("raw", 17, 48, False)),
    (("chat", 100, 48, False), ("chat", 63, 48, False), ("chat", 64, 48, False)),
    (("raw", 63, 48, False), ("first", 65, 1, False), ("raw", 64, 48, False)),
    (("raw", 65, 48, False), ("chat", 127, 48, False), ("chat", 128, 48, False)),
    (("chat", 129, 48, False), ("chat", 255, 48, False), ("chat", 256, 48, False)),
    (("chat", 257, 48, False), ("chat", 335, 48, False), ("chat", 336, 48, False)),
    (("chat", 337, 48, False), ("chat", 380, 48, False), ("chat", 463, 48, False)),
    (("chat", 464, 48, False), ("chat", 465, 48, False), ("chat", 490, 48, False)),
    (("chat", 512, 48, False), ("chat", 513, 48, False), ("chat", 700, 48, False)),
)


def _workload(tokenizer) -> list[list[dict]]:
    """The sampling calls, each the anchor and one group: each request's role, prompt ids and
    sampling."""

    marker = "\x00"
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    head, tail = (
        tokenizer(part, add_special_tokens=False)["input_ids"] for part in text.split(marker)
    )
    body = tokenizer(_BODY, add_special_tokens=False)["input_ids"]

    def request(role: str, length: int, max_tokens: int, ignore_eos: bool, offset: int) -> dict:
        # Each prompt reads its own stretch of the body: prompts share only the system head.
        prompt = body[offset : offset + length]
        if role != "raw":
            prompt = head + prompt + tail
        return {"role": role, "prompt": prompt, "max_tokens": max_tokens, "ignore_eos": ignore_eos}

    anchor = request(*_ANCHOR, offset=0)
    calls, offset = [], _ANCHOR[1] + 7
    for group in _GROUPS:
        calls.append([anchor])
        for member in group:
            calls[-1].append(request(*member, offset=offset))
            offset += member[1] + 7
    if offset > len(body):
        raise ValueError("the body text is too short for the workload")
    return calls


def _arm(snapshots: dict, calls: list[list[dict]], *, slots: int) -> dict:
    """One arm: every call on one engine with ``slots`` running requests."""

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
            model=ModelOptions(path=snapshots["target"], max_seq_len=4096),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(
                max_running_requests=slots, max_batch_tokens=_MAX_BATCH_TOKENS
            ),
            cache=CacheOptions(prefix="radix"),
            speculative=SpeculativeOptions(model_path=snapshots["draft"]),
        ),
        tuning=EngineTuning(cuda_graph_bs=tuple(range(1, slots + 1))),
    )
    requests, uid = [], 0
    try:
        for call in calls:
            params = []
            for request in call:
                params.append(
                    SamplingParams(
                        temperature=0,
                        max_new_tokens=request["max_tokens"],
                        ignore_eos=request["ignore_eos"],
                        seed=uid,
                    )
                )
                uid += 1
            result = llm.sample([request["prompt"] for request in call], params)
            for request, sample in zip(call, result.samples, strict=True):
                requests.append(
                    {
                        **request,
                        "output": list(sample.sampled_token_ids),
                        "finish_reason": sample.finish_reason,
                        "cached_prompt_tokens": sample.cached_prompt_tokens,
                    }
                )
    finally:
        llm.shutdown()
    return {"requests": requests}


def _stop_token_ids(snapshot: str) -> tuple[int, ...]:
    """The ids that end a request, as the engine resolves them (Qwen3.5-9B has no
    generation config)."""

    from cti.resolution import ResolvedModel

    return ResolvedModel.load(snapshot).facts.stop_token_ids


def _label(request: dict) -> str:
    return f"{request['role']}-{len(request['prompt'])}"


@pytest.mark.parametrize("pair", PINNED_CHECKPOINTS[:2], ids=lambda pair: pair.target.repo_id)
def test_folded_prompts_follow_the_fp32_truth(pair, tmp_path: Path, monkeypatch) -> None:
    from transformers import AutoTokenizer

    if "CTI_REFERENCE_CACHE_DIR" not in os.environ:
        # No cache to keep it: the truth this run computes serves this run only.
        monkeypatch.setenv("CTI_REFERENCE_CACHE_DIR", str(tmp_path / "reference-cache"))
    # Missing pinned checkpoints raise QualificationBlocked: they fail the run, never skip.
    inputs = preflight(pair)
    snapshots = inputs["snapshots"]
    calls = _workload(AutoTokenizer.from_pretrained(snapshots["target"], local_files_only=True))
    arms = {
        "never": _arm(snapshots, calls, slots=1),
        "fold": _arm(snapshots, calls, slots=_SLOTS),
    }
    stop_token_ids = _stop_token_ids(snapshots["target"])

    checkpoint = pair.target
    sequences = {
        (tuple(request["prompt"]), tuple(request["output"]))
        for arm in arms.values()
        for request in arm["requests"]
        if request["output"]
    }
    job = (receipts_dir() or tmp_path) / f"dflash-fold-truth-{checkpoint.slug}.pt"
    snapshot = Path(snapshots["target"])
    truth, truth_report = SequenceTruth(checkpoint, snapshot, tmp_path).require(sequences, job)

    import torch

    judged, gaps = {}, {}
    for name, arm in arms.items():
        judged[name], gaps[name] = [], []
        for request in arm["requests"]:
            if not request["output"]:
                judged[name].append(None)
                gaps[name].append(None)
                continue
            rows = truth[tuple(request["prompt"]), tuple(request["output"])]
            output = torch.tensor(request["output"])
            judged[name].append(compare_greedy(rows, {"greedy_ids": output}))
            # At least how far each pick lies below the truth's: 0 for its own pick, else the
            # truth's top-two margin. Judged through the output's first stop token: past it,
            # an output that ignores EOS continues a text the model ended.
            gap = torch.where(output != rows["greedy_ids"], rows["greedy_margins"], 0.0)
            stops = [i for i, token in enumerate(request["output"]) if token in stop_token_ids]
            end = stops[0] + 1 if stops else len(gap)
            gaps[name].append(
                {
                    "nats": float(gap[:end].max()),
                    "step": int(gap[:end].argmax()),
                    "picks": end,
                    "near_tie_or_more": int((gap[:end] >= rows["near_tie_nats"][:end]).sum()),
                    "past_stop_nats": float(gap[end:].max()) if end < len(gap) else None,
                }
            )
    # Per arm, by request index: where its output first leaves the truth at a decided margin.
    flips = {
        name: {
            index: result["divergence"]
            for index, result in enumerate(results)
            if result is not None and not result["agrees_until_truth_near_tie"]
        }
        for name, results in judged.items()
    }
    divergences = []
    for never, folded in zip(arms["never"]["requests"], arms["fold"]["requests"], strict=True):
        ours, theirs = never["output"], folded["output"]
        step = next((i for i, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b), None)
        if step is not None:
            margins = truth[tuple(never["prompt"]), tuple(ours)]["greedy_margins"]
            divergences.append(
                {
                    "role": never["role"],
                    "length": len(never["prompt"]),
                    "step": step,
                    "truth_margin": float(margins[step]),
                }
            )
    receipt = {
        "checkpoint": checkpoint.key,
        "truth": truth_report,
        "cached_prompt_tokens": {
            name: sum(request["cached_prompt_tokens"] for request in arm["requests"])
            for name, arm in arms.items()
        },
        "arm_divergences": divergences,
        "picks_whose_gap_bound_reaches_near_tie": {
            name: [
                sum(gap["near_tie_or_more"] for gap in results if gap is not None),
                sum(gap["picks"] for gap in results if gap is not None),
            ]
            for name, results in gaps.items()
        },
        "worst_gap_bounds": {
            name: max(
                (
                    {"request": _label(arms[name]["requests"][index]), **gap}
                    for index, gap in enumerate(results)
                    if gap is not None
                ),
                key=lambda gap: gap["nats"],
            )
            for name, results in gaps.items()
        },
        "worst_gap_bounds_past_stop": {
            name: max(
                (
                    {
                        "request": _label(arms[name]["requests"][index]),
                        "nats": gap["past_stop_nats"],
                    }
                    for index, gap in enumerate(results)
                    if gap is not None and gap["past_stop_nats"] is not None
                ),
                key=lambda gap: gap["nats"],
                default=None,
            )
            for name, results in gaps.items()
        },
        "decided_flips": {
            name: {_label(arm["requests"][index]): flip for index, flip in flips[name].items()}
            for name, arm in arms.items()
        },
        "truth_divergences": {
            name: [
                {
                    "role": request["role"],
                    "length": len(request["prompt"]),
                    **result["divergence"],
                    "truth_first_near_tie": result["truth_first_near_tie"],
                }
                for request, result in zip(arms[name]["requests"], results, strict=True)
                if result is not None and result["divergence"] is not None
            ]
            for name, results in judged.items()
        },
    }
    write_receipt(f"dflash-fold-truth-{checkpoint.slug}", receipt)
    (tmp_path / "dflash-fold-truth.json").write_text(json.dumps(receipt, indent=1) + "\n")

    failed = {
        f"{name}/{index}/{_label(arms[name]['requests'][index])}": gap
        for name, results in gaps.items()
        for index, gap in enumerate(results)
        if gap is not None and gap["nats"] >= _GROSS_NATS
    }
    assert not failed, failed
    for arm in arms.values():
        for request in arm["requests"]:
            output, role = request["output"], request["role"]
            # A request ends at its budget, or, unless it ignores EOS, at a stop token.
            ended = request["finish_reason"] == "length" and len(output) == request["max_tokens"]
            stopped = (
                request["finish_reason"] == "stop"
                and not request["ignore_eos"]
                and output[-1] in stop_token_ids
            )
            assert len(output) <= request["max_tokens"] and (ended or stopped), role
