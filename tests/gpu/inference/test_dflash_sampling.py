"""Sampled DFlash v1 on the pinned v1 pairs: lossless, and seeded requests repeat.

Lossless: per setting, N requests per arm run 17 tokens with DFlash and without it, alternating
two prompts: a story opening, whose continuations spread, and a chat request to repeat a
sentence, whose first round often accepts every draft. Exact sampling draws every position from
the target distribution in both arms, so per setting and prompt the arms' histograms of the
first three tokens (joint) and of the last token, which only a bonus or a later round reaches,
must agree: a two-sample chi-square homogeneity test (categories below 10 over both arms pooled)
at a p-value of at least 1e-3. The arms' target logits differ only by BF16 rounding (one-row
decode against 16-row verify), far below what N = 1024 resolves. With 12 checks the false-alarm
rate is at most 1.2% per sample; a setting with a failing check runs once more with 4N requests
on seeds the primary's do not reach, and the check fails only if it fails there too.

Deterministic: seeded DFlash requests repeat bitwise across runs, request order and eager
(``cuda_graph_bs=[]``) versus captured rounds.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest
import torch

from tests.support.dflash import PINNED_CHECKPOINTS, preflight
from tests.support.qwen3_5_gpu import write_receipt

pytestmark = pytest.mark.checkpoints()

_STORY = "Once upon a time, in a small village by the sea, there lived a"
_REPEAT = (
    "Repeat this sentence exactly, with nothing else: The quick brown fox jumps over the lazy "
    "dog near the quiet river bank at dawn."
)
_V1_PAIRS = [pair for pair in PINNED_CHECKPOINTS if pair.architecture == "DFlashDraftModel"]
_SAMPLES = 1_024
# Requests one sample call admits at once.
_WAVE = 64
_TOKENS = 17
_MIN_POOLED = 10
_MIN_P_VALUE = 1e-3
_SETTINGS = {
    "T1": {"temperature": 1.0},
    # Seeded: its requests carry consecutive seeds (``_homogeneity_tests``).
    "T0.6/k20/p0.95 seeded": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "seed": 0},
    "T0.7/k20/p0.8": {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
}
# Seeded settings: the arm without DFlash draws on seeds this far from DFlash's; a confirmation
# starts from _CONFIRMATION_SEED, which neither primary arm reaches.
_ARM_SEED_OFFSET = 100_000
_CONFIRMATION_SEED = 1_000_000
_CONFIRMATION_FACTOR = 4
_SEEDED = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
_DETERMINISM_TOKENS = 64
_DETERMINISM_PROMPTS = (
    "The three most important inventions of the last century are",
    "def fibonacci(n):",
    "In 1492, Columbus",
    "A recipe for a quick dinner:",
)


def _chi_square(observed: torch.Tensor, expected: torch.Tensor, dof: int) -> dict:
    statistic = float(((observed - expected) ** 2 / expected).sum())
    half = torch.tensor([dof / 2, statistic / 2], dtype=torch.float64)
    # One category tests nothing: its counts are the totals.
    p_value = float(torch.special.gammaincc(half[0], half[1])) if dof else 1.0
    return {
        "statistic": statistic,
        "dof": dof,
        "p_value": p_value,
        "total_variation": float(
            (observed / observed.sum() - expected / expected.sum()).abs().sum() / 2
        ),
    }


def _pool(table: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """Keep the columns of a ``[rows, V]`` table whose size is at least 10 and pool the rest
    into one, which joins the smallest kept column if its own size is below 10: a chi-square
    needs no cell that small. ``sizes`` are the columns' expected or total counts."""

    keep = sizes >= _MIN_POOLED
    kept, rest = table[:, keep], table[:, ~keep].sum(1, keepdim=True)
    if not keep.any() or sizes[~keep].sum() >= _MIN_POOLED:
        pooled = torch.cat([kept, rest], 1)
        return pooled[:, pooled.sum(0) > 0]
    smallest = int(sizes[keep].argmin())
    kept[:, smallest] += rest[:, 0]
    return kept


def homogeneity(first: torch.Tensor, second: torch.Tensor) -> dict:
    """Two samples' counts over the vocabulary; categories below 10 over both pooled."""

    table = torch.stack([first, second]).double()
    table = _pool(table, table.sum(0))
    expected = table.sum(1, keepdim=True) * table.sum(0, keepdim=True) / table.sum()
    result = _chi_square(table, expected, table.shape[1] - 1)
    rows = table / table.sum(1, keepdim=True)
    return {**result, "total_variation": float((rows[0] - rows[1]).abs().sum() / 2)}


def test_pooled_statistics_keep_every_sample() -> None:
    first = torch.tensor([50.0, 3, 4, 2, 0, 0])
    second = torch.tensor([48.0, 4, 0, 0, 6, 1])
    # Category 0 and the pooled rest (9 against 11) remain.
    result = homogeneity(first, second)
    assert result["dof"] == 1
    assert result["total_variation"] == pytest.approx(abs(50 / 59 - 48 / 59))
    # A rest of fewer than 10 draws over both arms joins the smallest kept cell.
    folded = homogeneity(torch.tensor([50.0, 40, 3, 1]), torch.tensor([45.0, 30, 2, 0]))
    assert folded["dof"] == 1
    single = homogeneity(torch.tensor([2000.0, 0, 0]), torch.tensor([1900.0, 0, 0]))
    assert single["dof"] == 0 and single["p_value"] == 1


def _passes(test: dict | None) -> bool:
    """A check passes at a p-value in [bound, 1]; any other value, NaN included, fails."""

    return test is not None and _MIN_P_VALUE <= test["p_value"] <= 1


def settings_to_confirm(tests: dict[str, dict]) -> list[str]:
    """The settings with a failing check, in ``_SETTINGS`` order."""

    return [
        name
        for name in _SETTINGS
        if any(check.startswith(f"{name}, ") and not _passes(test) for check, test in tests.items())
    ]


def failed_checks(tests: dict[str, dict], confirmations: dict[str, dict]) -> list[str]:
    """The checks that fail on the primary sample and on its confirmation (a missing
    confirmation fails)."""

    return [
        check
        for check, test in tests.items()
        if not _passes(test) and not _passes(confirmations.get(check))
    ]


def test_a_check_fails_only_when_its_confirmation_fails_too() -> None:
    passed, failed = {"p_value": 0.5}, {"p_value": 1e-4}
    tests = {
        "T1, story: first tokens": passed,
        "T0.6/k20/p0.95 seeded, story: first tokens": failed,
        "T0.6/k20/p0.95 seeded, repeat: last token": passed,
        "T0.7/k20/p0.8, repeat: first tokens": failed,
    }
    # Only settings with a check below the bound run again.
    assert settings_to_confirm(tests) == ["T0.6/k20/p0.95 seeded", "T0.7/k20/p0.8"]
    seeded, top_p = (
        "T0.6/k20/p0.95 seeded, story: first tokens",
        "T0.7/k20/p0.8, repeat: first tokens",
    )
    # A fresh sample that passes clears the check; one that fails again confirms it.
    assert failed_checks(tests, {seeded: passed, top_p: failed}) == [top_p]
    assert failed_checks(tests, {seeded: passed, top_p: passed}) == []
    # A check whose confirmation is missing fails; a passing check never needs one.
    assert failed_checks(tests, {}) == [seeded, top_p]
    assert failed_checks({"T1, story: first tokens": passed}, {}) == []
    # A p-value outside [bound, 1] fails, on either sample; the bound itself passes.
    bound, first = {"p_value": _MIN_P_VALUE}, "T1, story: first tokens"
    for invalid in ({"p_value": math.nan}, {"p_value": math.inf}, {"p_value": 1.5}):
        assert settings_to_confirm({first: invalid}) == ["T1"]
        assert failed_checks({first: invalid}, {}) == [first]
        assert failed_checks({first: failed}, {first: invalid}) == [first]
        assert failed_checks({first: invalid}, {first: bound}) == []
    assert failed_checks({first: bound}, {}) == []


def _llm(target: str, draft: str | None, graphs: list[int]):
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

    return LLM(
        EngineOptions(
            model=ModelOptions(path=target, max_seq_len=256),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=_WAVE, max_batch_tokens=1024),
            cache=CacheOptions(prefix="naive"),
            speculative=None if draft is None else SpeculativeOptions(model_path=draft),
        ),
        tuning=EngineTuning(cuda_graph_bs=tuple(graphs)),
    )


def _arm(snapshots: dict, draft: str | None, prompts, runs: dict) -> dict[str, list[list[int]]]:
    """Per setting in ``runs`` (its request count and, if seeded, first seed), the requests'
    tokens, alternating ``prompts``, from one engine with or without the draft."""

    from cti.protocol import SamplingParams

    llm = _llm(snapshots["target"], draft, [1, 2, 3, 4])
    tokens = {}
    try:
        for name, (samples, first_seed) in runs.items():
            setting = {key: value for key, value in _SETTINGS[name].items() if key != "seed"}
            params = [
                SamplingParams(
                    max_new_tokens=_TOKENS,
                    ignore_eos=True,
                    seed=None if first_seed is None else first_seed + index,
                    **setting,
                )
                for index in range(samples)
            ]
            tokens[name] = []
            for start in range(0, samples, _WAVE):
                wave = range(start, min(start + _WAVE, samples))
                result = llm.sample([prompts[i % 2] for i in wave], params[start : wave.stop])
                ordered = sorted(result.samples, key=lambda sample: sample.request_id)
                tokens[name] += [list(sample.sampled_token_ids) for sample in ordered]
    finally:
        llm.shutdown()
    return tokens


def _setting_tests(name: str, dflash: list[list[int]], plain: list[list[int]]) -> dict[str, dict]:
    """Per prompt, homogeneity of the arms' first three tokens and of their last token."""

    tests = {}
    for prompt, label in enumerate(("story", "repeat")):
        for statistic, key in (("first tokens", slice(0, 3)), ("last token", slice(-1, None))):
            counts = [Counter(tuple(row[key]) for row in arm[prompt::2]) for arm in (dflash, plain)]
            categories = sorted(set(counts[0]) | set(counts[1]))
            first, second = (
                torch.tensor([count[c] for c in categories], dtype=torch.float64)
                for count in counts
            )
            tests[f"{name}, {label}: {statistic}"] = homogeneity(first, second)
    return tests


def _homogeneity_tests(snapshots: dict, prompts, names, samples: int, seed: int) -> dict:
    """The checks of the settings ``names`` on ``samples`` requests per arm; seeded settings
    draw DFlash's from ``seed``, the other arm's ``_ARM_SEED_OFFSET`` further."""

    def runs(offset):
        return {
            name: (samples, seed + offset if "seed" in _SETTINGS[name] else None) for name in names
        }

    dflash = _arm(snapshots, snapshots["draft"], prompts, runs(0))
    plain = _arm(snapshots, None, prompts, runs(_ARM_SEED_OFFSET))
    tests = {}
    for name in names:
        tests.update(_setting_tests(name, dflash[name], plain[name]))
    return tests


def _determinism(snapshots: dict, prompts) -> dict[str, bool]:
    """Whether seeded DFlash outputs repeat across runs, request order and eager rounds."""

    from cti.protocol import SamplingParams

    def run(llm, order):
        params = [
            SamplingParams(
                max_new_tokens=_DETERMINISM_TOKENS, ignore_eos=True, seed=100 + i, **_SEEDED
            )
            for i in order
        ]
        result = llm.sample([prompts[i] for i in order], params)
        samples = sorted(result.samples, key=lambda sample: sample.request_id)
        return {i: list(sample.sampled_token_ids) for i, sample in zip(order, samples, strict=True)}

    runs = {}
    for mode, graphs in (("captured", [1, 2, 3, 4]), ("eager", [])):
        llm = _llm(snapshots["target"], snapshots["draft"], graphs)
        try:
            if mode == "captured":
                runs["batched"] = run(llm, [0, 1, 2, 3])
                runs["repeat"] = run(llm, [0, 1, 2, 3])
                runs["reversed"] = run(llm, [3, 2, 1, 0])
            else:
                runs["eager"] = run(llm, [0, 1, 2, 3])
        finally:
            llm.shutdown()
    return {name: tokens == runs["batched"] for name, tokens in runs.items() if name != "batched"}


@pytest.mark.parametrize("pair", _V1_PAIRS, ids=lambda pair: pair.target.repo_id)
def test_sampled_dflash_is_lossless_and_seeded_rounds_repeat(pair) -> None:
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    # Missing pinned checkpoints raise QualificationBlocked: they fail the run, never skip.
    snapshots = preflight(pair)["snapshots"]
    tokenizer = AutoTokenizer.from_pretrained(snapshots["target"], local_files_only=True)
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": _REPEAT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompts = [tokenizer(text)["input_ids"] for text in (_STORY, chat)]
    tests = _homogeneity_tests(snapshots, prompts, list(_SETTINGS), _SAMPLES, 0)
    retry = settings_to_confirm(tests)
    confirmations = (
        _homogeneity_tests(
            snapshots, prompts, retry, _CONFIRMATION_FACTOR * _SAMPLES, _CONFIRMATION_SEED
        )
        if retry
        else {}
    )
    determinism = _determinism(
        snapshots, [tokenizer(text)["input_ids"] for text in _DETERMINISM_PROMPTS]
    )
    write_receipt(
        f"dflash-sampling-{pair.target.repo_id.split('/')[-1]}",
        {"tests": tests, "confirmations": confirmations, "determinism": determinism},
    )

    failed = failed_checks(tests, confirmations)
    assert not failed, {check: (tests[check], confirmations.get(check)) for check in failed}
    assert all(determinism.values()), determinism
