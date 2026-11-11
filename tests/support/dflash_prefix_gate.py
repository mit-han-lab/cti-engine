"""The DFlash prefix gate: DFlash with the radix cache, DFlash with the naive cache and ordinary
decoding with the radix cache each follow the FP32 truth, with a pinned v1 pair, greedy, at one
KV page size and tensor-parallel size.

Each configuration runs one workload through the public ``LLM`` in its own process: four cold
requests sharing a system prompt, waves of one to four that hit it, then two four-turn
conversations whose history is the configuration's own output verbatim. Radix runs report every
cached prefix in ``cached_prompt_tokens``, the naive cache none; Qwen3.5 hits stop at recurrent
snapshots, on a 64-token grid, so a later turn restores the previous turn's committed tokens.
Every request's greedy output leaves the FP32 truth's pick (``tests.support.reference_truth``,
computed on this host where the cache lacks it) only at or after a truth near tie.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tests.support.dflash import CheckpointPair, preflight
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.hf_judge import compare_greedy
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import receipts_dir, write_receipt
from tests.support.reference_truth import SequenceTruth

RECEIPT_PREFIX = "CTI_DFLASH_PREFIX="
_CONFIGS = ("dflash-radix", "dflash-naive", "radix")
_TOKENS = 64
_GRAPH_SIZES = [1, 2, 3, 4]
# Qwen3.5 snapshot rows: enough that no row a later turn restores is reused before it.
_SNAPSHOT_ROWS = 32
_GRID = 64
_TOPICS = "billing shipping returns warranty accounts privacy outages rentals".split()
_SYSTEM = "You are the support assistant of a hardware store. " + " ".join(
    f"Policy {index + 1} covers {topic}: answer questions about {topic} in at most three "
    f"sentences, cite policy {index + 1}, and hand off to a human when the customer mentions "
    f"a legal dispute about {topic}."
    for index, topic in enumerate(_TOPICS * 2)
)
_QUESTIONS = (
    "My invoice shows the same charge twice. What should I do?",
    "How long does standard shipping to Canada take?",
    "Can I return a drill I bought 40 days ago?",
    "Does the warranty cover water damage on a pressure washer?",
    "I forgot the email on my account. How do I recover it?",
    "Do you share my purchase history with advertisers?",
)
# Question indices per wave: four cold, then one to four that hit, some whole prompts.
_WAVES = ((0, 1, 2, 3), (4,), (5, 0), (1, 4, 5), (0, 1, 2, 3))
_CONVERSATIONS = (
    (
        "Explain how a hash table handles collisions.",
        "Show a short Python example.",
        "What is its worst-case lookup time?",
        "How does Python's dict keep lookups fast in practice?",
    ),
    (
        "Summarize the causes of the French Revolution.",
        "Which of them mattered most?",
        "How did it affect the rest of Europe?",
        "Name three books for further reading.",
    ),
)


def workload(tokenizer) -> dict:
    """The waves and turns the gate runs, as token ids; ``run_workload`` runs them."""

    def tokens(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def chat(messages):
        return tokens(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )

    def user(text):
        return [{"role": "user", "content": text}]

    system = [{"role": "system", "content": _SYSTEM}]
    # A later turn appends the previous output, without its stop token, and this suffix.
    suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    suffix += "<think>\n\n</think>\n\n"
    turns = [[chat(user(texts[0])) for texts in _CONVERSATIONS]]
    for turn in range(1, 4):
        turns.append([tokens(suffix.format(texts[turn])) for texts in _CONVERSATIONS])
    return {
        "waves": [[chat(system + user(_QUESTIONS[i])) for i in wave] for wave in _WAVES],
        "turns": turns,
        "stop_ids": tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]),
    }


def run_workload(llm, work: dict) -> list[dict]:
    """``work``'s waves, then its conversations turn by turn, each later turn on the
    previous turn's own output: every request's phase, prompt, greedy output and cached
    prompt tokens."""

    from cti.protocol import SamplingParams

    requests, params = [], SamplingParams(temperature=0, max_new_tokens=_TOKENS)

    def run(phase, prompts):
        samples = sorted(llm.sample(prompts, params).samples, key=lambda sample: sample.request_id)
        for prompt, sample in zip(prompts, samples, strict=True):
            requests.append(
                {
                    "phase": phase,
                    "prompt": prompt,
                    "output": list(sample.sampled_token_ids),
                    "cached": sample.cached_prompt_tokens,
                }
            )
        return [list(sample.sampled_token_ids) for sample in samples]

    for index, wave in enumerate(work["waves"]):
        run("shared" if index else "cold", wave)
    histories = [[] for _ in work["turns"][0]]
    for turn, suffixes in enumerate(work["turns"]):
        histories = [h + suffix for h, suffix in zip(histories, suffixes, strict=True)]
        outputs = run(f"turn{turn}", histories)
        histories = [
            history + (output[:-1] if output[-1] in work["stop_ids"] else output)
            for history, output in zip(histories, outputs, strict=True)
        ]
    return requests


def _run_child(name: str, page_size: int, root: Path, tp: int) -> None:
    """One configuration's greedy workload, written to ``<name>.json``."""

    import torch

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

    inputs = json.loads((root / "inputs.json").read_text())
    work = json.loads((root / "workload.json").read_text())
    snapshots = inputs["snapshots"]
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=snapshots["target"], max_seq_len=2048),
            parallel=ParallelOptions(num_ranks=tp),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=512),
            cache=CacheOptions(prefix="naive" if name.endswith("naive") else "radix"),
            speculative=(
                SpeculativeOptions(model_path=snapshots["draft"]) if "dflash" in name else None
            ),
        ),
        tuning=EngineTuning(
            attention_backend="fi",
            page_size=page_size,
            cuda_graph_bs=tuple(_GRAPH_SIZES),
            snapshot_rows=_SNAPSHOT_ROWS,
        ),
    )
    try:
        requests = run_workload(llm, work)
    finally:
        llm.shutdown()
    receipt = {
        "name": name,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "requests": requests,
    }
    (root / f"{name}.json").write_text(json.dumps(receipt))
    print(RECEIPT_PREFIX + json.dumps({"name": name}))


def _assert_prefix_reuse(runs: dict, hybrid: bool, page_size: int) -> None:
    """Radix runs hit the system prompt and every previous turn; the naive cache hits nothing."""

    assert not any(r["cached"] for r in runs["dflash-naive"]["requests"])
    for name in ("dflash-radix", "radix"):
        requests = runs[name]["requests"]
        system_hits = [r["cached"] for r in requests if r["phase"] == "shared"]
        assert min(system_hits) > 0, (name, system_hits)
        for turn in range(1, 4):
            for previous, current in zip(
                [r for r in requests if r["phase"] == f"turn{turn - 1}"],
                [r for r in requests if r["phase"] == f"turn{turn}"],
                strict=True,
            ):
                valid = len(previous["prompt"]) + len(previous["output"]) - 1
                grid = _GRID if hybrid else page_size
                assert current["cached"] >= valid // grid * grid, (name, turn)


def run_gate(pair: CheckpointPair, page_size: int, root: Path, *, tp: int = 1) -> None:
    """Run the gate for ``pair`` in ``root``; fail on any check, after writing its receipt."""

    import torch
    from transformers import AutoTokenizer

    # Missing pinned checkpoints raise QualificationBlocked: they fail the run, never skip.
    inputs = preflight(pair)
    hybrid = pair.target.repo_id == "Qwen/Qwen3.5-9B"
    tokenizer = AutoTokenizer.from_pretrained(inputs["snapshots"]["target"], local_files_only=True)
    (root / "inputs.json").write_text(json.dumps(inputs))
    (root / "workload.json").write_text(json.dumps(workload(tokenizer)))

    for name in _CONFIGS:
        env, log_root = isolated_runtime_env(root, name)
        if tp > 1:
            # Every rank compares each batch's digest with its peers' before running it.
            env["CTI_CHECKS"] = "1"
        command = ("-m", "tests.support.dflash_prefix_gate", name, str(page_size), str(root))
        output = run_isolated_process(
            (sys.executable, *command, str(tp)),
            log_root / "child.log",
            env=env,
            cwd=REPO_ROOT,
            timeout=900,
        )
        assert child_receipt(output, RECEIPT_PREFIX) == {"name": name}
    runs = {name: json.loads((root / f"{name}.json").read_text()) for name in _CONFIGS}
    assert_cuda_device_metadata(runs["radix"])
    _assert_prefix_reuse(runs, hybrid, page_size)

    checkpoint = pair.target
    sequences = {
        (tuple(request["prompt"]), tuple(request["output"]))
        for run in runs.values()
        for request in run["requests"]
    }
    tag = f"{checkpoint.slug}-{page_size}" + (f"-tp{tp}" if tp > 1 else "")
    job = (receipts_dir() or root) / f"dflash-prefix-truth-{tag}.pt"
    snapshot = Path(inputs["snapshots"]["target"])
    truth, truth_report = SequenceTruth(checkpoint, snapshot, root).require(sequences, job)
    judged = {
        name: [
            compare_greedy(
                truth[tuple(request["prompt"]), tuple(request["output"])],
                {"greedy_ids": torch.tensor(request["output"])},
            )
            for request in run["requests"]
        ]
        for name, run in runs.items()
    }
    receipt = {
        "checkpoint": checkpoint.key,
        "page_size": page_size,
        "tensor_parallel_size": tp,
        "device": runs["radix"]["device"],
        "truth": truth_report,
        "cached": {name: [r["cached"] for r in run["requests"]] for name, run in runs.items()},
        "greedy": judged,
    }
    write_receipt(f"dflash-prefix-{tag}", receipt)
    (root / "dflash-prefix.json").write_text(json.dumps(receipt, indent=1) + "\n")
    failed = {
        f"{name}/{index}": result["divergence"]
        for name, results in judged.items()
        for index, result in enumerate(results)
        if not result["agrees_until_truth_near_tie"]
    }
    assert not failed, failed


if __name__ == "__main__":
    _run_child(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
