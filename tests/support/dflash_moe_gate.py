"""The DFlash MoE combination gate: DFlash combined with an MoE feature on Qwen3.5-35B-A3B and
its pinned drafter, against its own references, through the public ``LLM``. A case names three
configurations (``CASES``): the plain engine (the flips reference), the reference DFlash
configuration (the acceptance reference) and the combination.

Every configuration runs the DFlash prefix gate's greedy workload (``run_workload``: waves
sharing a system prompt, then conversations on their own output) in its own process. The MoE
rule: the combination's own outputs leave their truth's top-1 at decided positions no more
often than the plain engine's do, plus ``DISAGREEMENT_SLACK``; and its mean acceptance length,
from the engine's round totals, is at least 0.9 times the reference DFlash configuration's. The
truth is the FP32 model's (``reference_truth``); the reference cache holds no adapter, so with
LoRA it is, for now, the BF16 plain engine with the same adapter (trained once and loaded from
its checkpoint by the plain and the DFlash engine), scoring every sequence as a prompt; an FP32
truth with the adapter waits on PEFT export.

Expert TP and the radix cache judge on identical tokens (``TEACHER_FORCED``): expert TP's
sharded reductions add in another order than expert parallelism's, and the radix cache chunks
prefill at its snapshots and its hits, so it accumulates in another order than the naive
cache; either way a free run diverges at the first near-even step and then scores a different
continuation. There every judged configuration scores the plain
engine's sequences as prompts, and the rule counts its top-1s that leave the truth's at decided
positions of those same tokens.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from tests.support.dflash import MOE_PAIR, preflight
from tests.support.dflash_prefix_gate import run_workload, workload
from tests.support.gpu import (
    child_receipt,
    offline_child_env,
    run_isolated_process,
)
from tests.support.hf_judge import DISAGREEMENT_SLACK, NEAR_TIE_NATS, is_near_tie
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import PINNED_CHECKPOINTS, receipts_dir, write_receipt
from tests.support.reference_truth import SequenceTruth, free_run_flips

_MODULE = "tests.support.dflash_moe_gate"
RECEIPT_PREFIX = "CTI_DFLASH_MOE="
# RoundTotals in the engine's drain log, cumulative: the last line covers the whole workload.
_ROUND_TOTALS = re.compile(r"Speculative (\d+) request rounds, (\d+) drafts accepted")
_MIN_ACCEPTANCE_RATIO = 0.9
# One SFT step moves the adapter enough to change some greedy tokens.
_LORA_RANK, _LORA_ALPHA, _LEARNING_RATE = 8, 16.0, 1e-3
_TRAINED = ("What is the capital of France?", "Paris, the largest city of France, on the Seine.")


@dataclass(frozen=True)
class _Arm:
    name: str
    dflash: bool = False
    prefix: str = "naive"
    ranks: int = 1
    expert_parallel: bool = True
    lora: bool = False


# Per case: the plain engine, the reference DFlash configuration and the combination; they run
# in this order's reverse.
CASES = MappingProxyType(
    {
        "radix-prefix": (
            _Arm("plain"),
            _Arm("dflash-naive", dflash=True),
            _Arm("dflash-radix", dflash=True, prefix="radix"),
        ),
        "expert-tp": (
            _Arm("plain-ep2", ranks=2),
            _Arm("dflash-ep2", dflash=True, ranks=2),
            _Arm("dflash-etp2", dflash=True, ranks=2, expert_parallel=False),
        ),
        "expert-lora": (
            _Arm("plain-lora", lora=True),
            _Arm("dflash", dflash=True),
            _Arm("dflash-lora", dflash=True, lora=True),
        ),
    }
)
# The cases judged on the plain engine's tokens rather than on each configuration's own.
TEACHER_FORCED = frozenset({"expert-tp", "radix-prefix"})
# Trains the LoRA cases' adapter and saves it, before any arm runs.
_ADAPTER = _Arm("adapter", lora=True)
_CHECKPOINT = "adapter.ckpt"


def _llm(arm: _Arm, snapshots: dict, checkpoint: Path | None = None):
    from cti.config import (
        ActivationCheckpointMode,
        AdamWConfig,
        BatchLayout,
        CacheOptions,
        EngineOptions,
        EngineTuning,
        LoRAConfig,
        MemoryOptions,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
        SpeculativeOptions,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.llm import LLM

    training = TrainingConfig(
        trainable_plan=TrainablePlan.all_declared(),
        optimizer=AdamWConfig(lr=_LEARNING_RATE, weight_decay=0.0, max_grad_norm=1.0),
        batching=TrainingBatching(
            layout=BatchLayout("packed"),
            policy=StaticBatching(microbatch_size=1, max_tokens_per_sample=128),
        ),
        activation_checkpointing=ActivationCheckpointMode.FULL,
    )
    return LLM(
        EngineOptions(
            model=ModelOptions(path=snapshots["target"], max_seq_len=2048),
            parallel=ParallelOptions(
                num_ranks=arm.ranks, enable_expert_parallel=arm.expert_parallel
            ),
            # Training keeps the adapter's FP32 optimizer state (7 GiB) inside the budget.
            memory=(
                MemoryOptions(inference_ratio=0.9, peak_ratio=0.95)
                if arm.lora
                else MemoryOptions(inference_ratio=0.8)
            ),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=512),
            cache=CacheOptions(prefix=arm.prefix),
            speculative=SpeculativeOptions(model_path=snapshots["draft"]) if arm.dflash else None,
            lora=(
                LoRAConfig(targets=("experts",), rank=_LORA_RANK, alpha=_LORA_ALPHA)
                if arm.lora
                else None
            ),
            training=training if arm.lora else None,
        ),
        tuning=EngineTuning(cuda_graph_bs=(1, 2, 3, 4), snapshot_rows=32),
        resume_from_checkpoint=checkpoint,
    )


def _train_adapter(llm, tokenizer, checkpoint: Path) -> None:
    from cti.config import BatchLayout
    from cti.training import SFTDatum, TrainingBatch

    question, answer = _TRAINED
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt, target = (tokenizer(t, add_special_tokens=False)["input_ids"] for t in (text, answer))
    datum = SFTDatum(
        token_ids=(*prompt, *target), target_mask=(False,) * len(prompt) + (True,) * len(target)
    )
    with llm.training_session() as session:
        completion = session.forward_backward(TrainingBatch.sft([datum], BatchLayout.PACKED))
        session.optim_step().result()
        session.publish()
        completion.result()
        session.save_checkpoint(checkpoint)


def _rescore(llm, sequences: list[tuple[list[int], list[int]]]) -> list[dict]:
    """Each (prompt, output) scored as one prompt: per output position, this engine's
    top-1 and its margin over the second, in nats."""

    from cti.protocol import SamplingParams

    params = SamplingParams(temperature=0, max_new_tokens=1, prompt_logprobs=2)
    samples = llm.sample([prompt + output for prompt, output in sequences], params).samples
    scored = []
    for (prompt, output), sample in zip(sequences, samples, strict=True):
        rows = sample.prompt_logprobs[len(prompt) :]
        assert len(rows) == len(output) and all(len(row.top) == 2 for row in rows)
        scored.append(
            {
                "prompt": prompt,
                "output": output,
                "greedy_ids": [row.top[0][0] for row in rows],
                "greedy_margins": [row.top[0][1] - row.top[1][1] for row in rows],
            }
        )
    return scored


def _run_child(case: str, name: str, root: str) -> None:
    import hashlib

    from transformers import AutoTokenizer

    plain, _, candidate = arms = CASES[case]
    arm = next(arm for arm in (*arms, _ADAPTER) if arm.name == name)
    root_path = Path(root)
    snapshots = json.loads((root_path / "inputs.json").read_text())["snapshots"]
    work = json.loads((root_path / "workload.json").read_text())
    checkpoint = root_path / _CHECKPOINT
    llm = _llm(arm, snapshots, checkpoint if arm.lora and arm != _ADAPTER else None)
    requests, rescored = [], None
    try:
        if arm == _ADAPTER:
            _train_adapter(llm, AutoTokenizer.from_pretrained(snapshots["target"]), checkpoint)
        else:
            requests = run_workload(llm, work)
        if arm.lora and arm == plain:
            # Its own sequences and the LoRA DFlash run's, which ran before it.
            theirs = json.loads((root_path / f"{candidate.name}.json").read_text())["requests"]
            sequences = {(tuple(r["prompt"]), tuple(r["output"])) for r in (*requests, *theirs)}
            rescored = _rescore(llm, [(list(p), list(o)) for p, o in sorted(sequences)])
        elif case in TEACHER_FORCED:
            # The plain engine's sequences, which ran first; its own for the plain engine.
            mine = requests
            if arm != plain:
                mine = json.loads((root_path / f"{plain.name}.json").read_text())["requests"]
            sequences = {(tuple(r["prompt"]), tuple(r["output"])) for r in mine}
            rescored = _rescore(llm, [(list(p), list(o)) for p, o in sorted(sequences)])
        revision = llm.model_revision
    finally:
        llm.shutdown()
    receipt = {"requests": requests, "rescored": rescored, "model_revision": revision}
    if arm.lora:
        receipt["adapter_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (root_path / f"{name}.json").write_text(json.dumps(receipt))
    print(RECEIPT_PREFIX + json.dumps({"name": name}))


def _run_arm(root: Path, case: str, arm: _Arm) -> dict:
    env, log_root = offline_child_env(root, arm.name)
    started = time.monotonic()
    output = run_isolated_process(
        (sys.executable, "-m", _MODULE, case, arm.name, str(root)),
        log_root / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=1200,
    )
    assert child_receipt(output, RECEIPT_PREFIX) == {"name": arm.name}
    result = json.loads((root / f"{arm.name}.json").read_text())
    result["seconds"] = round(time.monotonic() - started, 1)
    if arm.dflash:
        rounds, drafts = map(int, _ROUND_TOTALS.findall(output)[-1])
        result["acceptance"] = 1 + drafts / rounds
    return result


def _sequences(run: dict) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [(tuple(r["prompt"]), tuple(r["output"])) for r in run["requests"]]


def _flip_positions(truth: dict, sequences) -> list[dict]:
    """Each decided position whose output token is not the truth's top-1: request, step, the
    two tokens and the truth's margin."""

    flips = []
    for index, key in enumerate(sequences):
        entry = truth[key]
        for step, token in enumerate(key[1]):
            margin = float(entry["greedy_margins"][step])
            top = int(entry["greedy_ids"][step])
            if not is_near_tie(margin, entry["near_tie_nats"][step]) and token != top:
                flips.append(
                    {"request": index, "step": step, "token": token, "truth": top, "margin": margin}
                )
    return flips


def _teacher_forced_flips(truth: dict, rescored: list[dict]) -> dict:
    """``free_run_flips``' count on identical tokens: over the scored sequences' decided
    positions, those where the configuration's own top-1 leaves the truth's. Each distinct
    position counts once."""

    decided = disagreements = 0
    seen = set()
    for scored in rescored:
        prompt, output = tuple(scored["prompt"]), tuple(scored["output"])
        entry = truth[(prompt, output)]
        for step, top in enumerate(scored["greedy_ids"]):
            position = (prompt, output[:step])
            tie = is_near_tie(entry["greedy_margins"][step], entry["near_tie_nats"][step])
            if position in seen or tie:
                continue
            seen.add(position)
            decided += 1
            disagreements += int(top != int(entry["greedy_ids"][step]))
    return {"decided": decided, "decided_disagreements": disagreements}


def _truth(case: str, root: Path, snapshot: str, runs: dict) -> dict:
    """Per sequence, its truth's top-1 ids and top-two margins: the FP32 model's, or with LoRA
    the plain engine's with the same adapter."""

    import torch

    if case == "expert-lora":
        return {
            (tuple(s["prompt"]), tuple(s["output"])): {
                "greedy_ids": torch.tensor(s["greedy_ids"]),
                "greedy_margins": torch.tensor(s["greedy_margins"]),
                # The engine reports log-probabilities, not the logits whose BF16 grid sets
                # the bound elsewhere: the floor.
                "near_tie_nats": torch.full((len(s["greedy_margins"]),), NEAR_TIE_NATS),
            }
            for s in runs["plain-lora"]["rescored"]
        }
    checkpoint = next(checkpoint for checkpoint in PINNED_CHECKPOINTS if checkpoint.moe)
    keys = {key for run in runs.values() for key in _sequences(run)}
    job = (receipts_dir() or root) / f"dflash-moe-{case}-truth.pt"
    truth, _ = SequenceTruth(checkpoint, Path(snapshot), root).require(keys, job)
    return truth


def run_gate(case: str, tmp_path: Path) -> None:
    """Run ``case`` in ``tmp_path``; fail on any check, after writing its receipt."""

    from transformers import AutoTokenizer

    plain, reference, candidate = CASES[case]
    inputs = preflight(MOE_PAIR)
    snapshot = inputs["snapshots"]["target"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    (tmp_path / "workload.json").write_text(json.dumps(workload(tokenizer)))
    if case in TEACHER_FORCED:
        # The plain engine first: the others score its sequences.
        order = (plain, reference, candidate)
    else:
        order = ((_ADAPTER,) if candidate.lora else ()) + (reference, candidate, plain)
    runs = {arm.name: _run_arm(tmp_path, case, arm) for arm in order}
    truth = _truth(case, tmp_path, snapshot, runs)
    # With LoRA, the truth scores only the adapted runs' sequences.
    judged = (plain, candidate) if candidate.lora else (plain, reference, candidate)
    if case in TEACHER_FORCED:
        flips = {
            arm.name: _teacher_forced_flips(truth, runs[arm.name]["rescored"]) for arm in judged
        }
    else:
        flips = {arm.name: free_run_flips(truth, _sequences(runs[arm.name])) for arm in judged}
    acceptance = {name: run["acceptance"] for name, run in runs.items() if "acceptance" in run}
    cached = {name: [r["cached"] for r in run["requests"]] for name, run in runs.items()}
    receipt = {
        "case": case,
        "flips": flips,
        "flip_positions": {
            arm.name: _flip_positions(truth, _sequences(runs[arm.name])) for arm in judged
        },
        "acceptance": acceptance,
        "cached": cached,
        "seconds": {name: run["seconds"] for name, run in runs.items()},
    }
    write_receipt(f"dflash-moe-{case}", receipt)
    print(RECEIPT_PREFIX + json.dumps(receipt))

    bound = flips[plain.name]["decided_disagreements"] + DISAGREEMENT_SLACK
    assert flips[candidate.name]["decided_disagreements"] <= bound, flips
    ratio = acceptance[candidate.name] / acceptance[reference.name]
    assert ratio >= _MIN_ACCEPTANCE_RATIO, acceptance
    if case == "radix-prefix":
        # Every wave after the first hits the shared system prompt and every later turn its
        # history; the naive cache hits nothing.
        later = [r for r in runs[candidate.name]["requests"] if r["phase"] not in ("cold", "turn0")]
        assert min(r["cached"] for r in later) > 0, cached
        assert not any(cached[plain.name] + cached[reference.name]), cached
    if case == "expert-lora":
        # Both engines serve the one adapter trained, at its revision, and it changes what
        # the target says.
        loaded = {
            arm.name: (runs[arm.name]["adapter_sha256"], runs[arm.name]["model_revision"])
            for arm in (_ADAPTER, plain, candidate)
        }
        assert len(set(loaded.values())) == 1 and loaded[_ADAPTER.name][1] > 0, loaded
        assert _sequences(runs[candidate.name]) != _sequences(runs[reference.name])


if __name__ == "__main__":
    from tests.support.qwen3_5_prefix import child_main

    child_main(_run_child, *sys.argv[1:4])
