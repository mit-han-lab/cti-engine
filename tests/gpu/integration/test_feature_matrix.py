"""The feature-combination matrix, through the public ``LLM`` API only; opt in with
CTI_MATRIX=1.

Each cell is one model with a combination of tensor parallelism, DFlash, prefix reuse, a
page size, a serving LoRA adapter and training (a LoRA adapter trained by one SFT step in the same engine,
then sampled from: the RL rollout loop). A cell's engine runs one workload in its own process:

- two waves of the same greedy prompts, which share a long system text, so a prefix-reusing
  cell hits its cache on the second wave;
- one greedy request with selected logprobs;
- one seeded stochastic request, submitted alone twice; with the naive cache both draw the same
  sample.

The cell is judged against its reference: the same model with every other feature off, trained
by the same step for a training cell, without an adapter otherwise. Greedy outputs may first
leave the reference only where the reference's own pick has probability below
``_NEAR_TIE_PROBABILITY``, the most a top-two margin under the default tier's near tie (0.25
nats) allows. Selected logprobs follow the reference's within the default tier's per-token and
mean bounds, over the whole reference continuation: where a cell's output leaves the
reference's, over the cell's logprobs teacher-forced on the reference's tokens. Every failing
check of a cell is reported together, with the rejection message when the engine refuses the
combination.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from tests.support.checkpoints import (
    QWEN3_0_6B,
    QWEN3_5_0_8B,
    QWEN3_5_9B,
    QWEN3_5_9B_DFLASH,
    QWEN3_5_35B_A3B,
    QWEN3_5_35B_A3B_DFLASH,
    QWEN3_8_27B,
    QWEN3_8_27B_DFLASH2,
    QWEN3_8B,
    QWEN3_8B_DFLASH_B16,
    QWEN3_30B_A3B,
    Checkpoint,
)
from tests.support.gpu import (
    child_receipt,
    isolated_runtime_env,
    require_device_memory,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT, source_pythonpath

_OPT_IN = "CTI_MATRIX"
_RECEIPT = "CTI_MATRIX_CELL="
_NEAR_TIE_PROBABILITY = 1.0 / (1.0 + math.exp(-0.25))
_LOGPROB_TOKEN_BOUND = 0.5
_LOGPROB_MEAN_BOUND = 0.1
_MAX_TOKENS = 32
_SYSTEM = (
    "You are a careful assistant. Answer in plain sentences, keep every step of a calculation "
    "visible, name the units of every quantity, and say so when a question has no single "
    "answer. Prefer short paragraphs. When a list helps, number its items. Do not repeat the "
    "question. If a fact depends on a date, give the date you assume. "
) * 3
_QUESTIONS = (
    "What is 17 times 23?",
    "Name three prime numbers between 40 and 60.",
    "Explain why the sky looks blue during the day.",
    "Write a haiku about a lighthouse.",
)


@dataclass(frozen=True)
class Model:
    checkpoint: Checkpoint
    # The BF16 checkpoint's weights, GiB.
    weights_gib: float
    draft: Checkpoint | None = None


MODELS = {
    "qwen3-0.6b": Model(QWEN3_0_6B, 1.5),
    "qwen3-8b": Model(QWEN3_8B, 15.3, QWEN3_8B_DFLASH_B16),
    "qwen3.5-0.8b": Model(QWEN3_5_0_8B, 1.7),
    "qwen3.5-9b": Model(QWEN3_5_9B, 17.5, QWEN3_5_9B_DFLASH),
    "qwen3.8-27b": Model(QWEN3_8_27B, 52.0, QWEN3_8_27B_DFLASH2),
    "qwen3-30b-a3b": Model(QWEN3_30B_A3B, 57.0),
    "qwen3.5-35b-a3b": Model(QWEN3_5_35B_A3B, 67.0, QWEN3_5_35B_A3B_DFLASH),
}


@dataclass(frozen=True)
class Cell:
    model: str
    tp: int = 1
    dflash: bool = False
    prefix: bool = False
    # "none", "serve" (an adapter as initialized) or "train" (trained in the engine first).
    lora: str = "none"
    # None takes the default, the smallest page the attention backends accept.
    page_size: int | None = None

    @property
    def reference(self) -> Cell:
        """A serving adapter starts as the identity, so its reference is the bare model."""

        return Cell(self.model, lora="train" if self.lora == "train" else "none")

    @property
    def id(self) -> str:
        parts = [f"tp{self.tp}"]
        parts += [name for name in ("dflash", "prefix") if getattr(self, name)]
        parts += [] if self.lora == "none" else [f"lora-{self.lora}"]
        parts += [] if self.page_size is None else [f"page{self.page_size}"]
        return f"{self.model}-{'-'.join(parts)}"


def _inference_ratio(cell: Cell) -> float:
    return 0.75 if cell.lora == "train" else 0.85


def _required_gib(cell: Cell) -> int:
    """A rank's share of the BF16 weights within the cell's inference ratio, plus 2 GiB for
    the KV cache, a draft and activations."""

    share = MODELS[cell.model].weights_gib / cell.tp
    return math.ceil(share / _inference_ratio(cell) + 2)


def _cells() -> list[Cell]:
    small = ("qwen3-0.6b", "qwen3.5-0.8b", "qwen3-30b-a3b")
    drafted = ("qwen3-8b", "qwen3.5-9b", "qwen3.8-27b", "qwen3.5-35b-a3b")
    cells = []
    for model in small:
        for tp in (1, 2):
            cells += [Cell(model, tp), Cell(model, tp, prefix=True)]
            cells += [Cell(model, tp, lora="serve"), Cell(model, tp, prefix=True, lora="train")]
        cells += [Cell(model, prefix=True, page_size=size) for size in (16, 64)]
    for model in drafted:
        cells += [Cell(model, prefix=True), Cell(model, tp=2)]
        for tp in (1, 2):
            cells += [Cell(model, tp, dflash=True), Cell(model, tp, dflash=True, prefix=True)]
            cells += [Cell(model, tp, dflash=True, lora="serve")]
            cells += [Cell(model, tp, dflash=True, prefix=True, lora="train")]
        cells += [Cell(model, dflash=True, prefix=True, page_size=size) for size in (16, 64)]
    return cells


def _run_cell(cell: Cell, continuations: list[list[list[int]]] | None) -> dict:
    """``cell``'s receipt; ``continuations`` (per wave and prompt, its reference's greedy
    tokens; None for a reference) are also scored teacher-forced."""

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
        ParallelOptions,
        SchedulingOptions,
        SpeculativeOptions,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams
    from cti.training import SFTDatum, TrainingBatch

    model = MODELS[cell.model]
    path = str(model.checkpoint.snapshot())
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompts = [
        tokenizer.encode(f"{_SYSTEM}\nQuestion: {question}\nAnswer:") for question in _QUESTIONS
    ]
    training = None
    if cell.lora == "train":
        training = TrainingConfig(
            trainable_plan=TrainablePlan.all_declared(),
            optimizer=AdamWConfig(lr=5e-3, weight_decay=0.0, max_grad_norm=1.0),
            batching=TrainingBatching(
                layout=BatchLayout.PACKED,
                policy=StaticBatching(microbatch_size=1, max_tokens_per_sample=512),
            ),
        )
    receipt: dict = {"cell": asdict(cell)}
    try:
        llm = LLM(
            EngineOptions(
                model=ModelOptions(path=path, max_seq_len=1024),
                parallel=ParallelOptions(num_ranks=cell.tp),
                memory=MemoryOptions(
                    inference_ratio=_inference_ratio(cell),
                    peak_ratio=0.92 if training else None,
                ),
                scheduling=SchedulingOptions(max_running_requests=8),
                cache=CacheOptions(prefix="radix" if cell.prefix else "naive"),
                speculative=SpeculativeOptions(model_path=str(model.draft.snapshot()))
                if cell.dflash
                else None,
                lora=None
                if cell.lora == "none"
                else LoRAConfig(targets=("attention",), rank=8, alpha=16.0),
                training=training,
            ),
            tuning=EngineTuning(page_size=cell.page_size),
        )
    except Exception as error:  # noqa: BLE001 - the receipt names the refusal
        return {**receipt, "rejected": f"{type(error).__name__}: {error}"}
    try:
        if training is not None:
            answer = tokenizer.encode(" 17 times 23 is 391.")
            datum = SFTDatum(
                token_ids=tuple(prompts[0] + answer),
                target_mask=(False,) * len(prompts[0]) + (True,) * len(answer),
            )
            with llm.training_session() as session:
                completion = session.forward_backward(
                    TrainingBatch.sft((datum,), BatchLayout.PACKED)
                )
                step = session.optim_step().result()
                session.publish()
                completion.result()
            receipt["training"] = {"step": step.step, "grad_norm": step.grad_norm}
        greedy = SamplingParams(
            temperature=0, max_new_tokens=_MAX_TOKENS, ignore_eos=True, logprobs=True
        )
        waves = []
        for _ in range(2):
            result = llm.sample(prompts, greedy)
            waves.append(
                [
                    {
                        "tokens": list(sample.sampled_token_ids),
                        "logprobs": sample.sampling_logprobs,
                        "cached": sample.cached_prompt_tokens,
                    }
                    for sample in result.samples
                ]
            )
        receipt["waves"] = waves
        if continuations is not None:
            receipt["teacher_forced"] = _attempt(
                lambda: [_teacher_forced(llm, prompts, wave) for wave in continuations]
            )
        receipt["logprobs"] = _attempt(
            lambda: list(
                llm.sample(prompts[:1], replace(greedy, logprobs=True)).samples[0].sampling_logprobs
            )
        )
        stochastic = SamplingParams(
            temperature=0.8, top_p=0.95, max_new_tokens=_MAX_TOKENS, ignore_eos=True, seed=20261017
        )
        receipt["stochastic"] = _attempt(
            lambda: [
                list(llm.sample(prompts[1:2], stochastic).samples[0].sampled_token_ids)
                for _ in range(2)
            ]
        )
    finally:
        llm.shutdown()
    return receipt


def _teacher_forced(llm, prompts: list[list[int]], continuations: list[list[int]]) -> list:
    """Each prompt's continuation's logprobs, the continuation read as prompt tokens."""

    from cti.protocol import SamplingParams

    params = SamplingParams(temperature=0, max_new_tokens=1, prompt_logprobs=0)
    sequences = [prompt + tokens for prompt, tokens in zip(prompts, continuations, strict=True)]
    samples = llm.sample(sequences, params).samples
    return [
        [row.logprob for row in sample.prompt_logprobs[len(prompt) :]]
        for prompt, sample in zip(prompts, samples, strict=True)
    ]


def _attempt(call):
    try:
        return {"value": call()}
    except Exception as error:  # noqa: BLE001 - the receipt names the refusal
        return {"error": f"{type(error).__name__}: {error}"}


def _judge(cell: Cell, receipt: dict, reference: dict) -> list[str]:
    if "rejected" in receipt:
        return [f"rejected: {receipt['rejected']}"]
    failures = []
    if cell.lora == "train" and not receipt["training"]["grad_norm"] > 0.0:
        failures.append(f"training: {receipt['training']}")
    forced = receipt.get("teacher_forced", {"value": None})
    if "error" in forced:
        failures.append(f"teacher-forced: {forced['error']}")
    for wave, (samples, expected) in enumerate(
        zip(receipt["waves"], reference["waves"], strict=True)
    ):
        for request, (sample, truth) in enumerate(zip(samples, expected, strict=True)):
            where = f"wave {wave} request {request}"
            failures += _judge_greedy(where, sample["tokens"], truth)
            if sample["logprobs"] is not None and "error" not in forced:
                rows = None if forced["value"] is None else forced["value"][wave][request]
                failures += _judge_logprobs(where, sample, truth, rows)
            if cell.prefix and wave == 1 and sample["cached"] == 0:
                failures.append(f"prefix: {where} hit no cached tokens")
    if "error" in receipt["logprobs"]:
        failures.append(f"logprobs: {receipt['logprobs']['error']}")
    stochastic = receipt["stochastic"]
    if "error" in stochastic:
        failures.append(f"stochastic: {stochastic['error']}")
    # The second submission hits the first's cache, which the default tier lets round apart.
    elif not cell.prefix and stochastic["value"][0] != stochastic["value"][1]:
        failures.append(f"stochastic: one seed sampled {stochastic['value']}")
    return failures


def _common_prefix(tokens: list[int], expected: list[int]) -> int:
    pairs = zip(tokens, expected, strict=False)
    return next((i for i, (a, b) in enumerate(pairs) if a != b), min(len(tokens), len(expected)))


def _judge_greedy(where: str, tokens: list[int], truth: dict) -> list[str]:
    expected = truth["tokens"]
    step = _common_prefix(tokens, expected)
    if step == len(expected):
        return [] if len(tokens) == len(expected) else [f"greedy: {where} length {len(tokens)}"]
    if step == len(tokens):
        return [f"greedy: {where} length {len(tokens)}"]
    probability = math.exp(truth["logprobs"][step])
    if probability < _NEAR_TIE_PROBABILITY:
        return []
    return [
        f"greedy: {where} left the reference at step {step}, where its pick had p={probability:.3f}"
    ]


def _judge_logprobs(where: str, sample: dict, truth: dict, forced: list[float] | None) -> list[str]:
    """The sample's logprobs against the reference's over the reference's whole continuation:
    its own while it follows the reference, else ``forced``, teacher-forced on the reference's
    tokens (None only for the reference itself, which never leaves its own tokens)."""

    common = _common_prefix(sample["tokens"], truth["tokens"])
    rows, how = sample["logprobs"], ""
    if common < len(truth["tokens"]):
        rows, how, common = forced, " teacher-forced", len(truth["tokens"])
    errors = [abs(a - b) for a, b in zip(rows[:common], truth["logprobs"][:common], strict=True)]
    if not errors:
        return []
    worst, mean = max(errors), sum(errors) / len(errors)
    if worst > _LOGPROB_TOKEN_BOUND or mean > _LOGPROB_MEAN_BOUND:
        return [f"logprobs{how}: {where} max {worst:.3f} mean {mean:.3f}"]
    return []


def _launch(cell: Cell, tmp_path: Path, reference: dict | None = None) -> dict:
    """``cell``'s receipt from its own process; with its ``reference``'s receipt, the cell
    also scores the reference's greedy tokens teacher-forced."""

    env, log_root = isolated_runtime_env(tmp_path, cell.id)
    env.update(
        PYTHONPATH=source_pythonpath(REPO_ROOT),
        PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.integration.test_feature_matrix",
            json.dumps(asdict(cell)),
            json.dumps(
                None
                if reference is None
                else [[sample["tokens"] for sample in wave] for wave in reference["waves"]]
            ),
        ),
        log_root / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=1800,
    )
    return child_receipt(output, _RECEIPT)


@pytest.fixture(scope="session")
def references() -> dict[Cell, dict]:
    """Reference receipts, each run once per session."""

    return {}


def _pins(cell: Cell) -> tuple[Checkpoint, ...]:
    model = MODELS[cell.model]
    return (model.checkpoint, model.draft) if cell.dflash else (model.checkpoint,)


@pytest.mark.parametrize(
    "cell",
    [
        pytest.param(
            cell,
            # Only an opted-in run reads them, so the checkpoint manifest leaves them out.
            marks=pytest.mark.checkpoints(*(_pins(cell) if os.environ.get(_OPT_IN) == "1" else ())),
            id=cell.id,
        )
        for cell in _cells()
    ],
)
def test_feature_combination_follows_its_reference(cell: Cell, references, tmp_path) -> None:
    if os.environ.get(_OPT_IN) != "1":
        pytest.skip(f"the feature matrix runs with {_OPT_IN}=1")
    import torch

    if torch.cuda.device_count() < cell.tp:
        pytest.skip(f"requires {cell.tp} visible GPUs, found {torch.cuda.device_count()}")
    # A model the device cannot hold skips, as the engine would fail it at startup.
    require_device_memory(_required_gib(cell))
    failures = _cell_failures(cell, references, tmp_path)
    report = os.environ.get("CTI_MATRIX_REPORT")
    if report:
        with open(report, "a") as handle:
            handle.write(json.dumps({"cell": cell.id, "failures": failures}) + "\n")
    assert not failures, f"{cell.id}:\n" + "\n".join(failures)


def _cell_failures(cell: Cell, references: dict[Cell, dict], tmp_path: Path) -> list[str]:
    """The cell's failed checks; a child that dies, as a refusal on a rank does, is one."""

    try:
        if cell.reference not in references:
            references[cell.reference] = _launch(cell.reference, tmp_path)
        reference = references[cell.reference]
        if "rejected" in reference:
            return [f"reference {cell.reference.id} rejected: {reference['rejected']}"]
        receipt = reference if cell == cell.reference else _launch(cell, tmp_path, reference)
    except pytest.fail.Exception as error:
        # A refusal on a peer rank also prints its traceback; the receipt still names it.
        lines = str(error).splitlines()
        named = [line.split(_RECEIPT, 1)[1] for line in lines if _RECEIPT in line]
        if named and "rejected" in (receipt := json.loads(named[0])):
            return [f"rejected: {receipt['rejected']}"]
        return [f"child failed: {lines[0]}"]
    return _judge(cell, receipt, reference)


if __name__ == "__main__":
    cell, continuations = Cell(**json.loads(sys.argv[1])), json.loads(sys.argv[2])
    print(_RECEIPT + json.dumps(_run_cell(cell, continuations)), flush=True)
