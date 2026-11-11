"""Tensor-parallel inference of real checkpoints on 2-4 local GPUs; opt in with
CTI_TP_QUALIFY=1; the pinned Qwen3-8B and Qwen3.5-9B snapshots come from the local
Hugging Face cache. Prefix reuse at TP1 and TP2 reaches one rank's depths and, teacher-forced
on a cold run's tokens, stays as close to the FP32 truth as the cold run; greedy chat
continuations, text and Qwen3.5 image chats, follow the near-tie rule against one GPU. Each
run is its own process, with CTI_CHECKS comparing every batch's digest across ranks."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_5_9B, QWEN3_8B
from tests.support.hf_judge import DISAGREEMENT_SLACK, P99_ERROR_SLACK
from tests.support.qwen3_5_gpu import receipts_dir
from tests.support.qwen3_5_prompts import image_chats
from tests.support.reference_truth import SequenceTruth, teacher_forced_top1
from tests.support.tp_cases import CHAT_PROMPTS, run_case
from tests.support.tp_parity import Continuation, assert_near_tie_parity

pytestmark = pytest.mark.checkpoints(QWEN3_8B, QWEN3_5_9B)

_QUALIFY = "CTI_TP_QUALIFY"
# The pinned checkpoints the gates run; their FP32 truth judges prefix hits.
_PINNED = {
    "qwen3": QWEN3_8B,
    "qwen3_5": QWEN3_5_9B,
}


def _require_gpus(count: int) -> None:
    if os.environ.get(_QUALIFY) != "1":
        pytest.skip(f"requires an exclusive multi-GPU allocation and {_QUALIFY}=1")
    import torch

    if torch.cuda.device_count() < count:
        pytest.skip(f"requires {count} visible GPUs, found {torch.cuda.device_count()}")


# =================================== Case entry points ===================================


def _chat(tokenizer, messages: list[dict[str, str]], **options) -> list[int] | str:
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_dict=False, **options
    )


def _run_prefix(args: argparse.Namespace) -> None:
    """Real-model prefix reuse at one TP size, one request at a time: four chat prompts
    behind one long system prompt, the same four again, then each conversation's next turn,
    which carries its answer. Greedy, with each token's log-probability and each request's
    cached prompt tokens. Then the sequences of ``--sequences`` (a cold run's receipt; without
    it, this run's own) teacher-forced: every prefix of each continuation as a one-token greedy
    request, recording its top-1 token, log-probability and cached prompt tokens."""

    from transformers import AutoTokenizer

    from cti.config import (
        CacheOptions,
        EngineOptions,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    system = {"role": "system", "content": " ".join(CHAT_PROMPTS * 6)}
    first_turns = [
        _chat(tokenizer, [system, {"role": "user", "content": q}], tokenize=True)
        for q in CHAT_PROMPTS[:4]
    ]
    # The tokens that close an answer and open the next user turn and answer.
    marker = "MARKER"
    template = _chat(
        tokenizer,
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": marker},
            {"role": "user", "content": "Please go on."},
        ],
        tokenize=False,
    )
    next_turn = tokenizer.encode(
        template[template.index(marker) + len(marker) :], add_special_tokens=False
    )
    # 96 tokens cross a snapshot point, so a conversation's next turn can hit past its prompt.
    policy = SamplingParams(temperature=0.0, max_new_tokens=96, ignore_eos=True, logprobs=True)
    top1 = SamplingParams(temperature=0.0, max_new_tokens=1, ignore_eos=True, logprobs=True)
    cache_type = args.mode.split("-")[1]
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=args.model, max_seq_len=4096),
            parallel=ParallelOptions(num_ranks=args.tp),
            scheduling=SchedulingOptions(max_running_requests=4),
            cache=CacheOptions(prefix=cache_type),
        )
    )
    requests = []

    def serve(prompt: list[int]) -> list[int]:
        (sample,) = llm.sample([prompt], [policy]).samples
        requests.append(
            {
                "prompt": list(prompt),
                "token_ids": list(sample.sampled_token_ids),
                "logprobs": list(sample.sampling_logprobs),
                "cached": sample.cached_prompt_tokens,
            }
        )
        return list(sample.sampled_token_ids)

    def force(prompt: list[int], tokens: list[int]) -> dict:
        prefixes = [[*prompt, *tokens[:end]] for end in range(len(tokens))]
        samples = llm.sample(prefixes, [top1] * len(prefixes)).samples
        return {
            "ids": [sample.sampled_token_ids[0] for sample in samples],
            "logprobs": [sample.sampling_logprobs[0] for sample in samples],
            "cached": [sample.cached_prompt_tokens for sample in samples],
        }

    try:
        answers = [serve(prompt) for prompt in first_turns]
        for prompt in first_turns:
            serve(prompt)
        for prompt, answer in zip(first_turns, answers, strict=True):
            serve([*prompt, *answer, *next_turn])
        sequences = (
            requests
            if args.sequences is None
            else json.loads(Path(args.sequences).read_text())["requests"]
        )
        forced = [force(request["prompt"], request["token_ids"]) for request in sequences]
    finally:
        llm.shutdown()
    Path(args.receipt).write_text(json.dumps({"requests": requests, "forced": forced}) + "\n")


def _run_parity(args: argparse.Namespace) -> None:
    """Greedy chat continuations with each token's log-probability at one TP size: of the
    text chats, or with ``--images`` of the image chats, which need the naive cache."""

    from transformers import AutoTokenizer

    from cti.config import (
        CacheOptions,
        EngineOptions,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    policy = SamplingParams(temperature=0.0, max_new_tokens=128, ignore_eos=True, logprobs=True)
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=args.model),
            parallel=ParallelOptions(num_ranks=args.tp),
            scheduling=SchedulingOptions(max_running_requests=16),
            cache=CacheOptions(prefix="naive" if args.images else "auto"),
        )
    )
    try:
        if args.images:
            encoder = llm.prompt_encoder()
            prompts = [
                encoder.chat(messages, template_kwargs={"enable_thinking": False})
                for messages in image_chats().values()
            ]
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            prompts = [
                _chat(tokenizer, [{"role": "user", "content": prompt}], tokenize=True)
                for prompt in CHAT_PROMPTS
            ]
        result = llm.sample(prompts, [policy] * len(prompts))
    finally:
        llm.shutdown()
    continuations = [
        {"token_ids": list(sample.sampled_token_ids), "logprobs": list(sample.sampling_logprobs)}
        for sample in result.samples
    ]
    Path(args.receipt).write_text(json.dumps({"continuations": continuations}) + "\n")


# ======================================== Tests ========================================


def _run_case(tmp_path: Path, component: str, model: Path, tp: int, mode: str, **options):
    return run_case(Path(__file__), tmp_path, component, model, tp, mode, **options)


class _PrefixRuns:
    """A real model's prefix workload, run per TP size on first use: a naive (cold) run, then a
    radix run, which also teacher-forces the cold run's sequences."""

    def __init__(self, gate: str, snapshot: Path, tmp_path: Path) -> None:
        self.gate = gate  # the model's key in _PINNED
        self.snapshot, self._tmp_path = snapshot, tmp_path
        self._runs: dict[int, dict[str, dict]] = {}

    def at(self, tp: int) -> dict[str, dict]:
        """The ``naive`` and ``radix`` receipts at ``tp``: each request's prompt, tokens,
        log-probabilities and cached tokens, and the teacher-forced cold sequences."""

        _require_gpus(tp)
        if tp not in self._runs:
            cold = _run_case(
                self._tmp_path, "tensor-parallel-prefix", self.snapshot, tp, "prefix-naive"
            )
            sequences = self._tmp_path / f"prefix-cold-tp{tp}.json"
            sequences.write_text(json.dumps({"requests": cold["requests"]}))
            hits = _run_case(
                self._tmp_path,
                "tensor-parallel-prefix",
                self.snapshot,
                tp,
                "prefix-radix",
                arguments=("--sequences", str(sequences)),
            )
            self._runs[tp] = {"naive": cold, "radix": hits}
        return self._runs[tp]

    def report(self, run: str, report: dict) -> None:
        line = json.dumps({"gate": self.gate, "run": run, **report})
        print(f"CTI_TP_PREFIX={line}", flush=True)


@pytest.fixture(scope="module", params=sorted(_PINNED))
def prefix_runs(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> _PrefixRuns:
    _require_gpus(1)
    snapshot = _PINNED[request.param].snapshot()
    return _PrefixRuns(request.param, snapshot, tmp_path_factory.mktemp("prefix"))


def test_real_model_tp2_hits_reach_the_depths_of_one_rank(prefix_runs: _PrefixRuns) -> None:
    runs = {
        (cache, tp): receipt["requests"]
        for tp in (1, 2)
        for cache, receipt in prefix_runs.at(tp).items()
    }
    cached = {key: [request["cached"] for request in run] for key, run in runs.items()}
    print(f"CTI_TP_PREFIX_CACHED={json.dumps([prefix_runs.gate, cached['radix', 1]])}")
    assert cached["radix", 1] == cached["radix", 2]
    assert cached["naive", 1] == cached["naive", 2] == [0] * 12
    first, repeated, turns = (cached["radix", 1][start : start + 4] for start in (0, 4, 8))
    # The second prompt's forward exports the branch point with the first, so a model with
    # recurrent layers hits the shared system prompt from the third prompt on.
    assert first[0] == 0 and min(first[2:]) > 0
    assert repeated[0] > 0
    assert all(hit >= shared for hit, shared in zip(repeated, first, strict=True))
    # A conversation's next turn hits past the prompt it answered.
    prompt_lens = [len(request["prompt"]) for request in runs["radix", 1][:4]]
    assert all(hit > length for hit, length in zip(turns, prompt_lens, strict=True))


def _continuations(run: list[dict], indices: Sequence[int]) -> list[Continuation]:
    return [Continuation(tuple(run[i]["token_ids"]), tuple(run[i]["logprobs"])) for i in indices]


@pytest.mark.parametrize("tp", [1, 2])
def test_real_model_hits_follow_the_fp32_truth_against_cold_runs(
    prefix_runs: _PrefixRuns, tp: int, tmp_path: Path
) -> None:
    """Teacher-forced on the cold run's own tokens, so both paths are judged on identical
    tokens, prefix hits stay as close to the FP32 truth as cold prefills: no more decided
    top-1 flips than the cold path plus ``DISAGREEMENT_SLACK``, and a top-1 log-probability
    error whose max and p99 exceed the cold path's by at most the parity gate's
    ``P99_ERROR_SLACK`` (0.1 nats, its engine-vs-BF16-Transformers allowance)."""

    runs = prefix_runs.at(tp)
    cold = runs["naive"]["requests"]
    forced = {path: receipt["forced"] for path, receipt in runs.items()}
    # Every forced prefix of the hit path reused cached tokens; no cold one did.
    hit_tokens = [cached for request in forced["radix"] for cached in request["cached"]]
    assert min(hit_tokens) > 0
    assert not any(cached for request in forced["naive"] for cached in request["cached"])
    sequences = [(tuple(request["prompt"]), tuple(request["token_ids"])) for request in cold]
    checkpoint = _PINNED[prefix_runs.gate]
    job = (receipts_dir() or tmp_path) / f"tp-prefix-truth-{checkpoint.slug}-tp{tp}.pt"
    truth, _ = SequenceTruth(checkpoint, prefix_runs.snapshot, tmp_path).require(
        set(sequences), job
    )
    truths = [truth[sequence] for sequence in sequences]
    paths = {
        path: teacher_forced_top1(
            truths,
            [token for request in requests for token in request["ids"]],
            [logprob for request in requests for logprob in request["logprobs"]],
        )
        for path, requests in forced.items()
    }
    hits, cold_path = paths["radix"], paths["naive"]
    prefix_runs.report(
        f"tp{tp}-hit-vs-cold",
        {"hit": hits, "cold": cold_path, "hit_tokens": [min(hit_tokens), sum(hit_tokens)]},
    )
    assert hits["finite"] and cold_path["finite"]
    assert (
        hits["decided_disagreements"] <= cold_path["decided_disagreements"] + DISAGREEMENT_SLACK
    ), paths
    for statistic in ("max", "p99"):
        assert hits[statistic] <= cold_path[statistic] + P99_ERROR_SLACK, (statistic, paths)


def test_real_model_tp2_hits_follow_the_near_tie_rule_against_one_rank(
    prefix_runs: _PrefixRuns,
) -> None:
    one, two = (prefix_runs.at(tp)["radix"]["requests"] for tp in (1, 2))
    # Next turns carry each size's own answers; compare the prompts both sizes served.
    same = [i for i, (a, b) in enumerate(zip(one, two, strict=True)) if a["prompt"] == b["prompt"]]
    report = assert_near_tie_parity(_continuations(one, same), _continuations(two, same))
    prefix_runs.report("tp2-vs-tp1-hits", asdict(report))


@pytest.mark.parametrize(
    ("model", "size", "images"),
    [("qwen3", 2, False), ("qwen3", 4, False), ("qwen3_5", 2, False), ("qwen3_5", 2, True)],
    ids=["qwen3-tp2", "qwen3-tp4", "qwen3_5-tp2", "qwen3_5-images-tp2"],
)
def test_real_model_tp_follows_the_near_tie_rule(
    tmp_path: Path, model: str, size: int, images: bool
) -> None:
    _require_gpus(size)
    snapshot = _PINNED[model].snapshot()
    runs = {}
    for run, tp in (("tp1", 1), ("tp1-fresh", 1), (f"tp{size}", size)):
        receipt = _run_case(
            tmp_path,
            "tensor-parallel-parity",
            snapshot,
            tp,
            f"parity-{run}",
            arguments=("--images",) if images else (),
        )
        runs[run] = [
            Continuation(tuple(c["token_ids"]), tuple(c["logprobs"]))
            for c in receipt["continuations"]
        ]
    # A fresh single-GPU process repeats itself bit for bit.
    assert runs["tp1-fresh"] == runs["tp1"]
    report = assert_near_tie_parity(runs["tp1"], runs[f"tp{size}"])
    # The GPU receipt quotes this line; tp_parity's bounds cite it as their basis.
    run = f"tp{size}" + ("-images" if images else "")
    line = json.dumps({"gate": model, "run": run, **asdict(report)})
    print(f"CTI_TP_PARITY={line}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "prefix-radix",
            "prefix-naive",
            "parity-tp1",
            "parity-tp1-fresh",
            "parity-tp2",
            "parity-tp4",
        ),
        required=True,
    )
    parser.add_argument("--receipt", required=True)
    # Prefix modes: the cold receipt whose sequences a run teacher-forces.
    parser.add_argument("--sequences")
    # Parity modes: the image chats in place of the text chats.
    parser.add_argument("--images", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode.startswith("parity"):
        _run_parity(arguments)
    else:
        _run_prefix(arguments)
