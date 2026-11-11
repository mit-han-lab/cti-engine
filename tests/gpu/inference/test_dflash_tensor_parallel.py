"""DFlash with tensor parallelism on two local GPUs; opt in with CTI_TP_QUALIFY=1.

Each pinned v1 pair (the Qwen3.5-35B-A3B target expert-parallel) runs one workload at TP1 and
at TP2: four requests and one alone, greedy, then seeded sampled ones. Every TP2 rank settles
the same rounds and ordinary steps, and holds half of the draft's query and KV heads; greedy
acceptance stays TP1's. A dense target's greedy output leaves TP1's only where the FP32 truth,
teacher-forced on their common prefix, is a near tie between their two tokens. A MoE target's
routing amplifies a few ulps (a padded graph bucket's GEMM) into top-8 changes at near ties, so
instead ordinary TP1 and TP2 engines (the target alone) teacher-force TP1's
greedy DFlash tokens, each scored against the FP32 truth, and each DFlash run's own greedy
outputs are scored against the truth of those outputs: TP2 within TP1's scores plus the parity
gate's slacks. Each run is its own process; observers in the engine processes record
every rank's completions. DFlash over the radix cache at TP2 is the DFlash prefix gate's
(test_dflash_prefix_tensor_parallel).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import pytest

from tests.support.dflash import MOE_PAIR, PINNED_CHECKPOINTS, preflight
from tests.support.gpu import isolated_runtime_env, require_device_memory, run_isolated_process
from tests.support.hf_judge import DISAGREEMENT_SLACK, P99_ERROR_SLACK
from tests.support.paths import REPO_ROOT, source_pythonpath
from tests.support.qwen3_5_gpu import receipts_dir, write_receipt
from tests.support.reference_truth import (
    SequenceTruth,
    free_run_flips,
    judge_divergence,
    teacher_forced_top1,
)

_QUALIFY = "CTI_TP_QUALIFY"
_OBSERVE_DIR = "CTI_DFLASH_TP_OBSERVE_DIR"
_PAIRS = [*PINNED_CHECKPOINTS, MOE_PAIR]
_PROMPTS = (
    "Write a detailed explanation of how a hash table works.",
    "Explain the causes of the French Revolution.",
    "Write a Python function that parses a CSV file.",
    "Describe photosynthesis for a high school student.",
)
_TOKENS = 128
# TP2 and TP1 logits differ in the last BF16 bits: steps of 0.125 near 16-32, as in
# test_sampling_features.py.
_LOGPROB_TOLERANCE = 0.25


# ============================== Engine-process observers ==============================


def _install_rank_observers() -> None:
    """Run in every spawned engine process, which re-imports this module as its main."""

    import numpy
    import torch

    from cti.inference._execution.runner import ForwardResult, InferenceRunner, RoundResult
    from cti.worker import GPUWorker

    evidence: dict[str, object] = {"completions": []}
    observe_dir = Path(os.environ[_OBSERVE_DIR])
    initialize = InferenceRunner.__init__

    def observe_initialize(self, *args, **kwargs) -> None:
        initialize(self, *args, **kwargs)
        runner = self._speculative
        draft = None if runner is None else next(runner._models.draft.iter_attention_layers())
        evidence.update(
            rank=kwargs["ranks"].rank,
            device=torch.cuda.get_device_name(self.device),
            draft_heads=None if draft is None else [draft.num_qo_heads, draft.num_kv_heads],
        )

    InferenceRunner.__init__ = observe_initialize

    def observing(output, record):
        def observe(self):
            result = output(self)
            # Wall time places the completion within the workload phase that settled it.
            completions = evidence["completions"]
            completions.append([len(completions), record(result), time.time()])
            return result

        return observe

    def ints(data: bytes, dtype: str) -> list:
        return numpy.frombuffer(data, dtype=dtype).tolist()

    def round_record(result) -> list:
        return [
            numpy.frombuffer(result.tokens, dtype="int32").reshape(-1, result.width).tolist(),
            ints(result.accepted, "int32"),
            ints(result.committed, "int32"),
            ints(result.done, "bool"),
            ints(result.frontiers, "int64"),
        ]

    ForwardResult.output = observing(
        ForwardResult.output, lambda result: [ints(result.tokens, "int32")]
    )
    RoundResult.output = observing(RoundResult.output, round_record)
    quiesce = GPUWorker.quiesce

    def observe_quiesce(self) -> None:
        quiesce(self)
        path = observe_dir / f"rank-{evidence['rank']}.json"
        path.write_text(json.dumps(evidence) + "\n")

    GPUWorker.quiesce = observe_quiesce


# ==================================== Child runs ====================================


def _chat(tokenizer, messages: list[dict]) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _run_workload(args: argparse.Namespace) -> None:
    """One configuration at one TP size: its phases' outputs, and the greedy phase's
    tokens per second on its second pass. With ``--force``, also the greedy sequences of
    ``--sequences`` (a receipt; without it, this run's own) teacher-forced: every prefix of
    each output as a one-token greedy request, its top-1 token and log-probability."""

    from transformers import AutoTokenizer

    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
        SpeculativeOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    dflash = args.run == "dflash"
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=args.target, max_seq_len=2048),
            parallel=ParallelOptions(num_ranks=args.tp),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=512),
            cache=CacheOptions(prefix="naive"),
            speculative=SpeculativeOptions(model_path=args.draft) if dflash else None,
        ),
        tuning=EngineTuning(cuda_graph_bs=(1, 2, 3, 4)),
    )
    phases: dict[str, dict] = {}

    def run(name: str, prompts: list[list[int]], params: list[SamplingParams]) -> None:
        started, wall = time.perf_counter(), time.time()
        result = llm.sample(prompts, params)
        seconds = time.perf_counter() - started
        samples = sorted(result.samples, key=lambda sample: sample.request_id)
        phases[name] = {
            "prompts": prompts,
            "outputs": [list(sample.sampled_token_ids) for sample in samples],
            "logprobs": [
                None if sample.sampling_logprobs is None else list(sample.sampling_logprobs)
                for sample in samples
            ],
            "seconds": seconds,
            "wall": [wall, time.time()],
        }

    try:
        prompts = [_chat(tokenizer, [{"role": "user", "content": text}]) for text in _PROMPTS]
        greedy = SamplingParams(temperature=0.0, max_new_tokens=_TOKENS)
        run("greedy", prompts, [greedy] * len(prompts))
        run("greedy-timed", prompts, [greedy] * len(prompts))
        run("greedy-b1", prompts[:1], [greedy])
        sampled = [
            SamplingParams(temperature=0.7, top_k=20, top_p=0.95, max_new_tokens=96, seed=100 + i)
            for i in range(len(prompts))
        ]
        run("sampled", prompts, sampled)
        if args.force:
            source = phases if args.sequences is None else _phases(args.sequences)
            greedy_phase = source["greedy"]
            top1 = SamplingParams(temperature=0.0, max_new_tokens=1, logprobs=True)
            forced = []
            for prompt, output in zip(
                greedy_phase["prompts"], greedy_phase["outputs"], strict=True
            ):
                prefixes = [[*prompt, *output[:end]] for end in range(len(output))]
                samples = llm.sample(prefixes, [top1] * len(prefixes)).samples
                forced.append(
                    {
                        "ids": [sample.sampled_token_ids[0] for sample in samples],
                        "logprobs": [sample.sampling_logprobs[0] for sample in samples],
                    }
                )
            phases["forced"] = {"sequences": greedy_phase, "requests": forced}
        # Full rounds: the deciding rank's logprobs reach every rank.
        run(
            "greedy-logprobs",
            prompts,
            [SamplingParams(temperature=0.0, max_new_tokens=_TOKENS, logprobs=True, top_logprobs=5)]
            * len(prompts),
        )
        processed = [
            SamplingParams(
                temperature=0.7,
                max_new_tokens=48,
                seed=200 + i,
                logprobs=True,
                logprobs_processed=True,
                presence_penalty=0.5,
                repetition_penalty=1.1,
            )
            for i in range(len(prompts))
        ]
        run("processed", prompts, processed)
    finally:
        llm.shutdown()
    Path(args.receipt).write_text(json.dumps({"phases": phases}) + "\n")


def _phases(receipt: str) -> dict:
    return json.loads(Path(receipt).read_text())["phases"]


# ======================================== Tests ========================================


def _run(tmp_path: Path, inputs: dict, run: str, tp: int, *, force: Path | bool = False) -> dict:
    """One workload run in its own process, with every rank's completions; ``force``
    teacher-forces this run's greedy sequences (True) or a receipt's (a path)."""

    env, logs = isolated_runtime_env(tmp_path, f"{run}-tp{tp}")
    env.update(
        PYTHONPATH=source_pythonpath(REPO_ROOT),
        PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    receipt, observe_dir = logs / "receipt.json", logs / "ranks"
    observe_dir.mkdir()
    env[_OBSERVE_DIR] = str(observe_dir)
    snapshots = inputs["snapshots"]
    run_isolated_process(
        (
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            *("--target", snapshots["target"], "--draft", snapshots["draft"]),
            *("--run", run, "--tp", str(tp), "--receipt", str(receipt)),
            *(() if force is False else ("--force",)),
            *(("--sequences", str(force)) if isinstance(force, Path) else ()),
        ),
        logs / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )
    result = json.loads(receipt.read_text())
    result["receipt"] = str(receipt)
    ranks = [json.loads(path.read_text()) for path in sorted(observe_dir.glob("rank-*.json"))]
    result["ranks"] = ranks
    assert [rank["rank"] for rank in ranks] == list(range(tp)), run
    # Every rank settles the primary's samples and the same rounds, each at its own time.
    settled = [[completion[:2] for completion in rank["completions"]] for rank in ranks]
    for rank in settled[1:]:
        assert rank == settled[0], (run, tp)
    return result


def _divergences(runs: dict[str, dict], reference: str, phases: tuple[str, ...]) -> dict:
    """Per (run, phase, request) whose output leaves ``reference``'s: the prompt, the
    reference's output, the step where they first differ and their tokens there. Outputs end
    only at a stop token or the shared budget, so one cannot be a strict prefix of the other."""

    divergences = {}
    for name, result in runs.items():
        for phase in phases:
            ours, theirs = runs[reference]["phases"][phase], result["phases"][phase]
            assert theirs["prompts"] == ours["prompts"]
            for index, (expected, actual) in enumerate(
                zip(ours["outputs"], theirs["outputs"], strict=True)
            ):
                key = f"{name}/{phase}/{index}"
                pairs = enumerate(zip(expected, actual, strict=False))
                position = next((i for i, (a, b) in pairs if a != b), None)
                if position is None:
                    assert len(expected) == len(actual), (key, len(expected), len(actual))
                    continue
                tokens = (expected[position], actual[position])
                divergences[key] = (ours["prompts"][index], expected, position, tokens)
    return divergences


def _truth_rulings(tmp_path: Path, pair, inputs: dict, divergences: dict) -> dict:
    """Each divergence judged against the FP32 truth of the reference's output."""

    checkpoint = pair.target
    sequences = {(tuple(prompt), tuple(output)) for prompt, output, _, _ in divergences.values()}
    job = (receipts_dir() or tmp_path) / f"dflash-tp-truth-{checkpoint.slug}.pt"
    snapshot = Path(inputs["snapshots"]["target"])
    truth, _ = SequenceTruth(checkpoint, snapshot, tmp_path).require(sequences, job)
    return {
        key: judge_divergence(truth[tuple(prompt), tuple(output)], step, tokens)
        for key, (prompt, output, step, tokens) in divergences.items()
    }


def _teacher_forced(tmp_path: Path, pair, inputs: dict, runs: dict[str, dict]) -> dict:
    """Each run's teacher-forced picks on the same greedy sequences, scored against their
    FP32 truth (``teacher_forced_top1``)."""

    forced_runs = {name: result for name, result in runs.items() if "forced" in result["phases"]}
    sequences = runs["ordinary-tp1"]["phases"]["forced"]["sequences"]
    keys = [
        (tuple(prompt), tuple(output))
        for prompt, output in zip(sequences["prompts"], sequences["outputs"], strict=True)
    ]
    checkpoint = pair.target
    job = (receipts_dir() or tmp_path) / f"dflash-tp-forced-truth-{checkpoint.slug}.pt"
    snapshot = Path(inputs["snapshots"]["target"])
    truth, _ = SequenceTruth(checkpoint, snapshot, tmp_path).require(set(keys), job)
    truths = [truth[key] for key in keys]
    scores = {}
    for name, result in forced_runs.items():
        forced = result["phases"]["forced"]
        assert forced["sequences"] == sequences, name
        requests = forced["requests"]
        scores[name] = teacher_forced_top1(
            truths,
            [token for request in requests for token in request["ids"]],
            [logprob for request in requests for logprob in request["logprobs"]],
        )
    return scores


def _free_run_flips(tmp_path: Path, pair, inputs: dict, runs: dict[str, dict]) -> dict:
    """Each named run's own greedy outputs against the FP32 truth of those same sequences
    (``free_run_flips``)."""

    checkpoint = pair.target
    outputs = {
        name: [
            (tuple(prompt), tuple(output))
            for prompt, output in zip(
                result["phases"]["greedy"]["prompts"],
                result["phases"]["greedy"]["outputs"],
                strict=True,
            )
        ]
        for name, result in runs.items()
    }
    job = (receipts_dir() or tmp_path) / f"dflash-tp-free-truth-{checkpoint.slug}.pt"
    snapshot = Path(inputs["snapshots"]["target"])
    keys = {key for sequences in outputs.values() for key in sequences}
    truth, _ = SequenceTruth(checkpoint, snapshot, tmp_path).require(keys, job)
    return {name: free_run_flips(truth, sequences) for name, sequences in outputs.items()}


def _acceptance(result: dict, phases: tuple[str, ...]) -> float:
    """Mean committed tokens per live row of the primary's rounds in ``phases``."""

    windows = [result["phases"][phase]["wall"] for phase in phases]
    committed = [
        count
        for _submission, record, wall in result["ranks"][0]["completions"]
        if len(record) == 5 and any(start <= wall <= end for start, end in windows)
        for count, accepted in zip(record[2], record[1], strict=True)
        if accepted >= 0
    ]
    return sum(committed) / len(committed)


def _throughput(result: dict, phase: str) -> float:
    """Tokens a phase emitted per second of its wall time."""

    outputs = result["phases"][phase]["outputs"]
    return sum(len(output) for output in outputs) / result["phases"][phase]["seconds"]


def _require_two_gpus() -> None:
    if os.environ.get(_QUALIFY) != "1":
        pytest.skip(f"requires an exclusive two-GPU allocation and {_QUALIFY}=1")
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip(f"requires 2 visible GPUs, found {torch.cuda.device_count()}")


@pytest.mark.parametrize("pair", _PAIRS, ids=lambda pair: pair.target.repo_id)
def test_tensor_parallel_rounds_follow_the_single_gpu(pair, tmp_path) -> None:
    _require_two_gpus()
    moe = pair == MOE_PAIR
    if moe:
        # The BF16 35B-A3B target beside its draft on one rank's GPU: an RTX PRO 6000.
        require_device_memory(90)
    inputs = preflight(pair)
    runs = {f"tp{tp}": _run(tmp_path, inputs, "dflash", tp) for tp in (1, 2)}
    # A MoE target's TP1 and TP2 serving numerics, teacher-forced on TP1's DFlash tokens by
    # ordinary engines: the target's forwards alone, no draft.
    sequences = Path(runs["tp1"]["receipt"]) if moe else False
    if moe:
        runs["ordinary-tp1"] = _run(tmp_path, inputs, "ordinary", 1, force=sequences)
    runs["ordinary-tp2"] = _run(tmp_path, inputs, "ordinary", 2, force=sequences)
    greedy = ("greedy", "greedy-timed", "greedy-b1")
    divergences = _divergences(runs, "tp1", greedy)
    rulings = {} if moe else _truth_rulings(tmp_path, pair, inputs, divergences)
    scores = _teacher_forced(tmp_path, pair, inputs, runs) if moe else {}
    dflash = {name: runs[name] for name in ("tp1", "tp2")}
    free_runs = _free_run_flips(tmp_path, pair, inputs, dflash) if moe else {}
    rates = {name: _acceptance(runs[name], greedy) for name in ("tp1", "tp2")}
    receipt = {
        "inputs": inputs,
        "devices": [rank["device"] for rank in runs["tp2"]["ranks"]],
        "draft_heads": {name: [r["draft_heads"] for r in runs[name]["ranks"]] for name in runs},
        "truth_divergences": rulings,
        "divergence_steps": {
            key: (step, tokens) for key, (_, _, step, tokens) in divergences.items()
        },
        "teacher_forced": scores,
        "free_run_flips": free_runs,
        "acceptance": rates,
        "tokens_per_second": {name: _throughput(runs[name], "greedy-timed") for name in runs},
        "sampled_identical": [
            one == two
            for one, two in zip(
                runs["tp1"]["phases"]["sampled"]["outputs"],
                runs["tp2"]["phases"]["sampled"]["outputs"],
                strict=True,
            )
        ],
    }
    write_receipt(f"dflash-tp-{pair.target.repo_id.split('/')[-1]}", receipt)
    print("CTI_DFLASH_TP=" + json.dumps({k: v for k, v in receipt.items() if k != "inputs"}))

    (draft_tp1,) = receipt["draft_heads"]["tp1"]
    assert receipt["draft_heads"]["tp2"] == [[draft_tp1[0] // 2, draft_tp1[1] // 2]] * 2
    assert receipt["draft_heads"]["ordinary-tp2"] == [None, None]
    if moe:
        assert receipt["draft_heads"]["ordinary-tp1"] == [None]
    for key, ruling in rulings.items():
        assert ruling["allowed"], (key, ruling)
    # MoE: TP2 serves TP1's tokens as close to the FP32 truth as TP1 does, within the parity
    # gate's slacks (tests/support/hf_judge.py).
    for name, score in scores.items():
        reference = scores["ordinary-tp1"]
        assert score["finite"], (name, score)
        bound = reference["decided_disagreements"] + DISAGREEMENT_SLACK
        assert score["decided_disagreements"] <= bound, (name, scores)
        for statistic in ("max", "p99"):
            assert score[statistic] <= reference[statistic] + P99_ERROR_SLACK, (name, statistic)
    # MoE: TP2's own greedy DFlash outputs leave the FP32 truth's top-1 at decided positions no
    # more often than TP1's do, within the same slack.
    if moe:
        bound = free_runs["tp1"]["decided_disagreements"] + DISAGREEMENT_SLACK
        assert free_runs["tp2"]["decided_disagreements"] <= bound, free_runs
    # Greedy TP2 rounds accept what TP1's do, up to the near-tie divergences.
    assert rates["tp2"] >= 0.9 * rates["tp1"], rates
    # Full rounds under TP2 return TP1's raw logprobs until the greedy rows part, and
    # processed ones for sampled rows.
    one, two = (runs[name]["phases"]["greedy-logprobs"] for name in ("tp1", "tp2"))
    for tokens, logprobs, theirs, their_logprobs in zip(
        one["outputs"], one["logprobs"], two["outputs"], two["logprobs"], strict=True
    ):
        pairs = enumerate(zip(tokens, theirs, strict=False))
        parted = next((i for i, (a, b) in pairs if a != b), min(len(tokens), len(theirs)))
        for index in range(parted):
            assert abs(logprobs[index] - their_logprobs[index]) <= _LOGPROB_TOLERANCE
    for logprobs in runs["tp2"]["phases"]["processed"]["logprobs"]:
        assert all(math.isfinite(value) and value <= 0.0 for value in logprobs)


if __name__ == "__mp_main__" and os.environ.get(_OBSERVE_DIR):
    _install_rank_observers()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--run", choices=("dflash", "ordinary"))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sequences")
    _run_workload(parser.parse_args())
