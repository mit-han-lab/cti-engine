"""Penalties, logit bias, min_tokens, min-p, top logprobs and prompt logprobs, with DFlash
on and off: greedy with penalties or steering follows the same tokens until a near tie,
steering holds (no stop before min_tokens, banned tokens never drawn, a forced token
always), raw logprobs and their alternatives
agree within tolerance, sampled processed logprobs stay finite, and a prompt's logprobs
over a greedy continuation match the logprobs it was sampled with. Each engine runs in its
own child."""

from __future__ import annotations

import json
import math
import sys
from itertools import pairwise
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

RECEIPT_PREFIX = "CTI_SAMPLING_FEATURES="
PROMPTS = (
    "List the planets of the solar system:",
    "def fibonacci(n):\n    ",
    "Repeat after me: red green blue red green blue",
)
MAX_TOKENS = 40
TOP = 5
SEED = 20261017
PENALTIES = {"presence_penalty": 0.5, "frequency_penalty": 0.3, "repetition_penalty": 1.2}
# Verify rows run at M = B * Q and decode rows at M = B: BF16 logits differ in the last bits,
# steps of 0.125 near 16-32, so greedy paths may part where two picks lie that close.
LOGPROB_TOLERANCE = 0.25
_V1_PAIRS = [pair for pair in PINNED_CHECKPOINTS if pair.architecture == "DFlashDraftModel"]


def _snapshot(checkpoint) -> str:
    return str(checkpoint.snapshot())


def _run_child(target_repo: str, dflash: bool) -> None:
    import torch

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

    pair = next(pair for pair in _V1_PAIRS if pair.target.repo_id == target_repo)
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=_snapshot(pair.target), max_seq_len=512),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=512),
            cache=CacheOptions(prefix="naive"),
            speculative=SpeculativeOptions(model_path=_snapshot(pair.draft)) if dflash else None,
        ),
        tuning=EngineTuning(cuda_graph_bs=(1, 2, 4)),
    )

    def sample_each(prompts, policies) -> list[dict]:
        params = [
            SamplingParams(
                max_new_tokens=MAX_TOKENS, seed=SEED + index, **{"ignore_eos": True, **p}
            )
            for index, p in enumerate(policies)
        ]
        result = llm.sample(list(prompts), params)
        return [
            {
                "prompt": list(s.prompt_token_ids),
                "tokens": list(s.sampled_token_ids),
                "finish": s.finish_reason,
                "logprobs": None if s.sampling_logprobs is None else list(s.sampling_logprobs),
                "top": s.sampling_top_logprobs,
                "prompt_logprobs": s.prompt_logprobs
                and [None if t is None else [t.logprob, t.top] for t in s.prompt_logprobs],
            }
            for s in sorted(result.samples, key=lambda sample: sample.request_id)
        ]

    def sample(prompts, **policy) -> list[dict]:
        return sample_each(prompts, [policy] * len(prompts))

    try:
        greedy = sample(PROMPTS, temperature=0.0, logprobs=True, top_logprobs=TOP)
        # Every first greedy token is banned, and row 0's second ends the output but cannot
        # before MAX_TOKENS; the third is forced.
        banned = sorted({row["tokens"][0] for row in greedy})
        stop = greedy[0]["tokens"][1]
        forced = greedy[0]["tokens"][2]
        steering = {"banned": banned, "stop": stop, "forced": forced}
        receipt = {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "greedy": greedy,
            "penalized": sample(
                PROMPTS,
                temperature=0.0,
                logprobs=True,
                logprobs_processed=True,
                top_logprobs=TOP,
                **PENALTIES,
            ),
            "steering": steering,
            "steered": sample(
                PROMPTS,
                temperature=0.0,
                logprobs=True,
                logprobs_processed=True,
                top_logprobs=TOP,
                ignore_eos=False,
                min_tokens=MAX_TOKENS,
                stop_token_ids=(stop,),
                logit_bias=dict.fromkeys(banned, -100.0),
            ),
            "forced": sample(PROMPTS, temperature=0.0, logit_bias={forced: 100.0}),
            "sampled": sample(
                PROMPTS,
                temperature=0.8,
                min_p=0.05,
                logprobs=True,
                logprobs_processed=True,
                top_logprobs=TOP,
                **PENALTIES,
            ),
            # One batch of three policies: greedy with alternatives, greedy without
            # logprobs, sampled with processed ones.
            "mixed": sample_each(
                PROMPTS,
                [
                    {"temperature": 0.0, "logprobs": True, "top_logprobs": TOP},
                    {"temperature": 0.0},
                    {
                        "temperature": 0.8,
                        "min_p": 0.05,
                        "logprobs": True,
                        "logprobs_processed": True,
                    },
                ],
            ),
            # Each prompt with its greedy continuation, scored as a prompt.
            "rescored": sample(
                [row["prompt"] + row["tokens"] for row in greedy],
                temperature=0.0,
                prompt_logprobs=TOP,
            ),
        }
    finally:
        llm.shutdown()
    print(RECEIPT_PREFIX + json.dumps(receipt))


def _engine(tmp_path: Path, target_repo: str, dflash: bool) -> dict:
    name = f"sampling-features-{'dflash' if dflash else 'plain'}"
    env, log_root = isolated_runtime_env(tmp_path, name)
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.inference.test_sampling_features",
            target_repo,
            str(dflash),
        ),
        log_root / f"{name}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )
    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert_cuda_device_metadata(receipt)
    return receipt


def _follows(expected: list[dict], actual: list[dict]) -> None:
    """Greedy rows agree token for token and logprob for logprob until they part, which they
    may only at a near tie of the distribution the expected row picks from."""

    for want, got in zip(expected, actual, strict=True):
        for index in range(_parts_only_at_a_near_tie(want, got)):
            assert abs(want["logprobs"][index] - got["logprobs"][index]) <= LOGPROB_TOLERANCE


def _parts_only_at_a_near_tie(reference: dict, row: dict) -> int:
    """How many greedy tokens ``row`` shares with ``reference`` before they part, which they
    may only where ``row`` takes another of the reference's alternatives within the logprob
    tolerance of its best. Exact ties list in no set order, so the reference's own pick may
    be any of them (the lowest id, as the greedy rows check)."""

    pairs = zip(reference["tokens"], row["tokens"], strict=True)
    parted = next((i for i, (a, b) in enumerate(pairs) if a != b), None)
    if parted is None:
        return len(reference["tokens"])
    top = reference["top"][parted]
    near = {token for token, value in top if top[0][1] - value <= LOGPROB_TOLERANCE}
    assert reference["tokens"][parted] in near and len(near) > 1, (parted, top)
    assert row["tokens"][parted] in near, (parted, top)
    return parted


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
def test_sampling_features_agree_with_and_without_dflash(tmp_path: Path, target_repo: str) -> None:
    plain = _engine(tmp_path, target_repo, dflash=False)
    dflash = _engine(tmp_path, target_repo, dflash=True)

    for receipt in (plain, dflash):
        # The penalties change what a greedy row says.
        assert receipt["penalized"] != receipt["greedy"]
        for row in receipt["greedy"]:
            # The sampled token is among its best alternatives (ties go to the smallest ID,
            # in any order among the alternatives); they are sorted, best first.
            for token, logprob, top in zip(row["tokens"], row["logprobs"], row["top"], strict=True):
                assert len(top) == TOP and abs(top[0][1] - logprob) <= 1e-5
                assert token == min(t for t, value in top if value == top[0][1])
                assert all(a[1] >= b[1] for a, b in pairwise(top))
        steering = receipt["steering"]
        assert receipt["steered"] != receipt["greedy"]
        for row in receipt["steered"]:
            assert row["finish"] == "length" and len(row["tokens"]) == MAX_TOKENS
            assert not set(row["tokens"]) & {*steering["banned"], steering["stop"]}
        for row in receipt["forced"]:
            assert row["tokens"] == [steering["forced"]] * MAX_TOKENS
        for row in receipt["sampled"]:
            assert all(math.isfinite(value) and value <= 0.0 for value in row["logprobs"])
        # A batch that mixes policies serves each row as the rows' own batches do.
        mixed = receipt["mixed"]
        _follows(receipt["greedy"][:1], mixed[:1])
        assert mixed[1]["logprobs"] is None and mixed[1]["top"] is None
        _parts_only_at_a_near_tie(receipt["greedy"][1], mixed[1])
        assert all(math.isfinite(value) and value <= 0.0 for value in mixed[2]["logprobs"])
        for scored, row in zip(receipt["rescored"], receipt["greedy"], strict=True):
            prompt_logprobs = scored["prompt_logprobs"]
            assert prompt_logprobs[0] is None and len(prompt_logprobs) == len(scored["prompt"])
            tail = prompt_logprobs[len(row["prompt"]) :]
            for (logprob, top), sampled in zip(tail, row["logprobs"], strict=True):
                assert len(top) == TOP
                assert abs(logprob - sampled) <= LOGPROB_TOLERANCE
    _follows(plain["greedy"], dflash["greedy"])
    _follows(plain["penalized"], dflash["penalized"])
    assert plain["steering"] == dflash["steering"]
    _follows(plain["steered"], dflash["steered"])


if __name__ == "__main__":
    _run_child(sys.argv[1], sys.argv[2] == "True")
