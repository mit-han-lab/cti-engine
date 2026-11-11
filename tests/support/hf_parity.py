"""The Transformers parity check of the engine on a pinned checkpoint: an FP32 truth child
(``tests/support/hf_truth.py``), a BF16 calibration child and an engine child that samples
through the public ``LLM`` API, judged per prompt by the teacher-forced top-1 gates of
``tests/support/hf_judge.py``. The engine child forces the truth's continuation one request per
prefix and keeps each request's top-1 token and logprob, cold (no prefix reuse); a MoE
checkpoint's child also forces it through prefix hits, and each path may excuse one prompt that
fails only on top-1 logprob error at a near tie of the truth. Every engine runs on one GPU or
over the first ``TP_SIZE`` (a MoE model's experts expert-parallel).

The family tests in ``tests/hf_parity`` select the checkpoint and the settings (``ParityRun``)
and call ``check_parity``; run as a module, this file is the engine child."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import pytest

from tests.support.checkpoints import Checkpoint
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    find_forbidden_log_patterns,
    offline_child_env,
)
from tests.support.hf_judge import (
    BF16_MAX_DISAGREEMENT_SHARE,
    BF16_MAX_MEAN_ERROR,
    BOUNDS,
    CALIBRATION_FACTOR,
    DISAGREEMENT_SLACK,
    MAX_ERROR_FLOOR,
    MEAN_ERROR_SLACK,
    NEAR_TIE_NATS,
    P99_ERROR_SLACK,
    compare_greedy,
    is_near_tie,
    near_tie_nats,
)
from tests.support.hf_truth import (
    IEEE_FP32_PRECISION,
    PROMPTS_PATH,
    check_tokenizer,
    check_truth,
    code_version,
    sha256_file,
)
from tests.support.hf_truth import RECEIPT_PREFIX as TRUTH_RECEIPT_PREFIX
from tests.support.paths import PYTHON_ROOT, REPO_ROOT

if TYPE_CHECKING:
    import torch

RECEIPT_PREFIX = "CTI_HF_PARITY="
CheckpointT = TypeVar("CheckpointT", bound=Checkpoint)
TOP_K = 8
# Below the 743/756-token prompt, so its prefill runs in three chunks.
PREFILL_CHUNK_TOKENS = 256
# The second cold pass's prefill chunk: another scheduling of the same rows, whose top-1 flips
# against the others mark the rows scheduling alone decides (``unstable_rows``).
SMALL_PREFILL_CHUNK_TOKENS = 96
# The most of a prompt's rows that may be scheduling-unstable: twice the BF16 engine's own
# worst share (Qwen3.5-35B-A3B over the parity prompts, cold at 256 and 96 and prefix hits:
# 4.7% at most, 2.2% pooled). Calibrated on that one model only.
UNSTABLE_SHARE_CEILING = 0.10
# Per path, the prompts a MoE checkpoint may excuse.
MOE_EXCEPTIONS = 1
_BF16_TIMEOUT_SECONDS = 2400
_ENGINE_TIMEOUT_SECONDS = 3600
_INFERENCE_MEMORY_RATIO = 0.85
# Lines of each log a failed child's report quotes; the logs themselves are kept whole.
_REPORT_LINES = 200
_LOGPROB_GATES = frozenset(f"top1_logprob_error_{s}" for s in ("mean", "p99", "max"))


@dataclass(frozen=True)
class ParitySettings:
    """A model family's ``<prefix>*`` environment settings, read by its parity check and its
    GPU model tests; ``tests/hf_parity/README.md`` lists them."""

    prefix: str

    def checkpoint(self, pins: Sequence[CheckpointT]) -> CheckpointT:
        """The pin ``REPO`` names, the first by default; ``REVISION`` must be its own, as the
        frozen parity prompts hold token ids for the pinned revisions only."""

        by_repo = {pin.repo_id: pin for pin in pins}
        repo_id = os.environ.get(f"{self.prefix}REPO", pins[0].repo_id)
        if repo_id not in by_repo:
            raise ValueError(f"{self.prefix}REPO={repo_id!r} is not one of {sorted(by_repo)}")
        checkpoint = by_repo[repo_id]
        revision = os.environ.get(f"{self.prefix}REVISION", checkpoint.revision)
        if revision != checkpoint.revision:
            raise ValueError(
                f"{self.prefix}REVISION={revision!r}, but {repo_id} is pinned at "
                f"{checkpoint.revision}"
            )
        return checkpoint

    def positive_int(self, name: str, default: int) -> int:
        variable = f"{self.prefix}{name}"
        value = os.environ.get(variable)
        if value is None:
            return default
        try:
            number = int(value)
        except ValueError:
            raise ValueError(f"{variable} must be a positive integer, got {value!r}") from None
        if number < 1:
            raise ValueError(f"{variable} must be a positive integer, got {value!r}")
        return number

    def flag(self, name: str, default: bool) -> bool:
        variable = f"{self.prefix}{name}"
        value = os.environ.get(variable)
        if value is None:
            return default
        if value not in ("0", "1"):
            raise ValueError(f"{variable} must be 0 or 1, got {value!r}")
        return value == "1"

    def positive_float(self, name: str) -> float | None:
        variable = f"{self.prefix}{name}"
        value = os.environ.get(variable)
        if value is None:
            return None
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"{variable} must be a positive number, got {value!r}") from None
        if not 0 < number < float("inf"):
            raise ValueError(f"{variable} must be a positive number, got {value!r}")
        return number

    def path(self, name: str) -> Path | None:
        value = os.environ.get(f"{self.prefix}{name}")
        return Path(value) if value else None


@dataclass(frozen=True)
class ParityRun:
    """One parity check's checkpoint and settings, as a family test selects them."""

    checkpoint: Checkpoint
    # Named in the check's refusals.
    settings: ParitySettings
    # Judge prefix hits too, and excuse a near-tie prompt per path.
    moe: bool
    # The engine's ranks; a MoE model's experts split over them.
    tensor_parallel_size: int
    # A MoE model's routed experts over the ranks: whole (expert parallel) or each cut by width.
    expert_parallel: bool
    greedy_tokens: int
    max_running_requests: int
    truth_gpus: int
    truth_gpu_gib: float | None
    truth_timeout: int
    reused_truth: Path | None
    receipts: Path | None
    # The checkpoint whose token ids in the parity prompts this one reads: its own, or ones its
    # tokenizer and chat template render too (checked before any child runs).
    prompt_ids_of: Checkpoint

    @classmethod
    def from_settings(
        cls,
        settings: ParitySettings,
        checkpoint: Checkpoint,
        *,
        moe: bool,
        prompt_ids_of: Checkpoint | None = None,
    ) -> ParityRun:
        return cls(
            checkpoint=checkpoint,
            prompt_ids_of=prompt_ids_of or checkpoint,
            settings=settings,
            moe=moe,
            tensor_parallel_size=settings.positive_int("TP_SIZE", 1),
            expert_parallel=settings.flag("EXPERT_PARALLEL", True),
            greedy_tokens=settings.positive_int("GREEDY_TOKENS", 256),
            max_running_requests=settings.positive_int("MAX_RUNNING_REQ", 16),
            truth_gpus=settings.positive_int("TRUTH_GPUS", 1),
            truth_gpu_gib=settings.positive_float("TRUTH_GPU_GIB"),
            truth_timeout=settings.positive_int("TRUTH_TIMEOUT", 7200),
            reused_truth=settings.path("REUSE_TRUTH"),
            receipts=settings.path("RECEIPTS"),
        )

    @property
    def paths(self) -> tuple[str, ...]:
        """The engine's teacher-forced paths: cold, and for a MoE model prefix hits."""

        return ("cold", "prefix_hit") if self.moe else ("cold",)


def _run_engine(arguments: argparse.Namespace) -> None:
    """Per path and prompt: the top-1 token and logprob after every prefix of the prompt and the
    truth's continuation (one request each); the cold path also the free greedy continuation."""

    import importlib.metadata

    import torch
    from safetensors.torch import load_file, save_file

    import cti
    import ctikernels
    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams
    from tests.support.hf_truth import load_prompts

    _, prompts = load_prompts(arguments.prompts, arguments.model_key)
    longest = max(len(ids) for _, ids in prompts) + arguments.greedy_tokens + 1
    max_seq_len = -(-longest // PREFILL_CHUNK_TOKENS) * PREFILL_CHUNK_TOKENS
    greedy = SamplingParams(
        temperature=0, ignore_eos=True, max_new_tokens=arguments.greedy_tokens, logprobs=True
    )
    top1 = SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=1, logprobs=True)
    for path in arguments.paths:
        chunk = SMALL_PREFILL_CHUNK_TOKENS if path == "cold_chunked" else PREFILL_CHUNK_TOKENS
        llm = LLM(
            EngineOptions(
                model=ModelOptions(path=str(arguments.snapshot), max_seq_len=max_seq_len),
                parallel=ParallelOptions(
                    num_ranks=arguments.tensor_parallel_size,
                    enable_expert_parallel=not arguments.expert_tensor_parallel,
                ),
                memory=MemoryOptions(inference_ratio=_INFERENCE_MEMORY_RATIO),
                scheduling=SchedulingOptions(
                    max_running_requests=arguments.max_running_req,
                    max_batch_tokens=max(chunk, arguments.max_running_req),
                ),
                cache=CacheOptions(prefix="radix" if path == "prefix_hit" else "naive"),
            ),
            tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)),
        )
        folder = arguments.output / path
        folder.mkdir(parents=True, exist_ok=True)
        try:
            for name, ids in prompts:
                continuation = load_file(arguments.truth / f"{name}.safetensors")["greedy_ids"]
                sequence = ids + continuation.tolist()
                prefixes = [sequence[:end] for end in range(1, len(sequence) + 1)]
                if path == "prefix_hit":
                    # One request at a time: each prefix hits the one before it.
                    forced = [llm.sample([prefix], top1).samples[0] for prefix in prefixes]
                else:
                    forced = list(llm.sample(prefixes, top1).samples)
                tensors = {
                    "input_ids": torch.tensor(ids, dtype=torch.int64),
                    "teacher_forced_ids": continuation,
                    "top1_ids": torch.tensor([s.sampled_token_ids[0] for s in forced]),
                    "top1_logprobs": torch.tensor([s.sampling_logprobs[0] for s in forced]),
                }
                if path == "cold":
                    (free,) = llm.sample([ids], greedy).samples
                    tensors["greedy_ids"] = torch.tensor(free.sampled_token_ids, dtype=torch.int64)
                save_file(tensors, folder / f"{name}.safetensors")
                print(f"{path}/{name}: {len(ids)} prompt tokens", flush=True)
        finally:
            llm.shutdown()

    receipt = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "engine_root": str(Path(cti.__file__).resolve().parent),
        "kernels": str(Path(ctikernels.__file__).resolve().parent),
        "kernels_version": importlib.metadata.version("cti-kernels"),
        "model_key": arguments.model_key,
        "num_ranks": arguments.tensor_parallel_size,
        "paths": list(arguments.paths),
        "prompts": [name for name, _ in prompts],
    }
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True), flush=True)


def _top1(dump: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """A Transformers dump's top-1 token and logprob at every teacher-forced position."""

    return dump["topk_ids"][:, 0], dump["topk_logits"][:, 0] - dump["logsumexp"]


def teacher_forced(
    truth: Mapping[str, torch.Tensor],
    ids: torch.Tensor,
    logprobs: torch.Tensor,
    *,
    eos_row: int | None = None,
    stable: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Top-1 disagreements where the truth is decided, and the top-1 logprob error where the
    top-1 agrees: the only positions where a sampled logprob is the truth token's. Also the
    truth's top-1 margin at the worst such position and, given ``eos_row`` (the row that
    predicts the truth's first EOS), the error up to that row and past it, reported apart.
    Given ``stable`` (``~unstable_rows``), disagreements and errors count on those rows only,
    and the others are reported."""

    import torch

    truth_ids, truth_logprobs = _top1(truth)
    if ids.shape != truth_ids.shape:
        raise ValueError(f"top-1 of {tuple(ids.shape)} and {tuple(truth_ids.shape)}")
    logits = truth["topk_logits"]
    margins = logits[:, 0] - logits[:, 1]
    bounds = near_tie_nats(logits[:, 0])
    decided = ~is_near_tie(margins, bounds)
    counted = decided if stable is None else decided & stable
    agrees = ids == truth_ids
    kept = agrees if stable is None else agrees & stable
    error = (logprobs.float() - truth_logprobs).abs().where(kept, 0.0)
    worst = int(error.argmax())
    rows = torch.arange(len(ids))
    halves = {}
    if eos_row is not None:
        for half, part in (("before_eos", rows <= eos_row), ("after_eos", rows > eos_row)):
            errors = error[part & kept]
            halves[half] = {
                "rows": int(part.sum()),
                "top1_logprob_error_max": float(errors.max()) if len(errors) else None,
                "top1_logprob_error_p99": float(errors.quantile(0.99)) if len(errors) else None,
                "decided_disagreements": int((part & counted & ~agrees).sum()),
            }
    error = error[kept]
    return {
        "positions": len(ids),
        "decided": int(counted.sum()),
        "decided_disagreements": int((counted & ~agrees).sum()),
        **({} if stable is None else {"unstable": int((~stable).sum())}),
        "top1_logprob_error": {
            "mean": float(error.mean()),
            "p99": float(error.quantile(0.99)),
            "max": float(error.max()),
        },
        "worst_position": worst,
        "worst_truth_margin": float(margins[worst]),
        "worst_truth_near_tie_nats": float(bounds[worst]),
        "finite": bool(logprobs.isfinite().all() and truth_logprobs.isfinite().all()),
        **({"first_eos_row": eos_row, **halves} if eos_row is not None else {}),
    }


def _eos_token_ids(snapshot: Path) -> frozenset[int]:
    """The checkpoint's EOS ids: generation_config.json's, else config.json's (its text config's
    for a multimodal model). Qwen3.5-9B ships no generation_config.json."""

    generation = snapshot / "generation_config.json"
    if generation.exists():
        ids = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id", [])
    else:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        ids = config.get("eos_token_id", config.get("text_config", {}).get("eos_token_id", []))
    return frozenset(ids if isinstance(ids, list) else [ids])


def _first_eos_row(truth: Mapping[str, torch.Tensor], eos_ids: frozenset[int]) -> int | None:
    """The row that predicts the first EOS of the truth's continuation, if it has one."""

    continuation = truth["greedy_ids"].tolist()
    first = next((index for index, token in enumerate(continuation) if token in eos_ids), None)
    return None if first is None else len(truth["input_ids"]) - 1 + first


def unstable_rows(top1_ids: Sequence[torch.Tensor]) -> torch.Tensor:
    """The rows whose top-1 differs between any two of one engine's schedulings of them: rows
    scheduling decides, which no top-1 comparison can judge."""

    import torch

    passes = torch.stack(list(top1_ids))
    return (passes != passes[0]).any(0)


def top1_gates(
    bf16: Mapping[str, Any],
    engine: Mapping[str, Any],
    *,
    bf16_max_disagreement_share: float = BF16_MAX_DISAGREEMENT_SHARE,
) -> dict[str, dict[str, Any]]:
    """The engine within bounds BF16 Transformers' own distance from the truth sets, and BF16
    within the ceilings that keep it a calibration (a case may declare its own share)."""

    errors, calibration = engine["top1_logprob_error"], bf16["top1_logprob_error"]
    bounds = {
        "bf16_decided_disagreements": (
            bf16["decided_disagreements"],
            bf16_max_disagreement_share * bf16["decided"],
        ),
        "bf16_top1_logprob_error_mean": (calibration["mean"], BF16_MAX_MEAN_ERROR),
        "decided_disagreements": (
            engine["decided_disagreements"],
            CALIBRATION_FACTOR * bf16["decided_disagreements"] + DISAGREEMENT_SLACK,
        ),
        "top1_logprob_error_mean": (
            errors["mean"],
            CALIBRATION_FACTOR * calibration["mean"] + MEAN_ERROR_SLACK,
        ),
        "top1_logprob_error_p99": (
            errors["p99"],
            CALIBRATION_FACTOR * calibration["p99"] + P99_ERROR_SLACK,
        ),
        "top1_logprob_error_max": (
            errors["max"],
            max(CALIBRATION_FACTOR * calibration["max"], MAX_ERROR_FLOOR),
        ),
    }
    gates = {
        name: {"value": value, "bound": bound, "passed": bool(value <= bound)}
        for name, (value, bound) in bounds.items()
    }
    if "unstable" in engine:
        share = engine["unstable"] / engine["positions"]
        gates["unstable_share"] = {
            "value": share,
            "bound": UNSTABLE_SHARE_CEILING,
            "passed": bool(share <= UNSTABLE_SHARE_CEILING),
        }
    gates["finite"] = {"passed": bf16["finite"] and engine["finite"]}
    return gates


def path_failures(
    comparisons: Sequence[Mapping[str, Any]], path: str, *, moe: bool
) -> dict[str, list[str]]:
    """``path``'s failed gates by prompt; a MoE checkpoint excuses up to ``MOE_EXCEPTIONS``
    prompts that fail only on top-1 logprob error, at a near tie of the truth."""

    failing = {
        comparison["name"]: [
            gate
            for gate, result in comparison["paths"][path]["gates"].items()
            if not result["passed"]
        ]
        for comparison in comparisons
    }
    failing = {name: gates for name, gates in failing.items() if gates}
    if moe and len(failing) <= MOE_EXCEPTIONS:
        for comparison in comparisons:
            judged = comparison["paths"][path]
            gates = failing.get(comparison["name"], ())
            engine = judged["engine"]
            tie = is_near_tie(engine["worst_truth_margin"], engine["worst_truth_near_tie_nats"])
            if gates and set(gates) <= _LOGPROB_GATES and tie:
                judged["excused"] = True
                del failing[comparison["name"]]
    return failing


def check_borrowed_prompt_ids(snapshot: Path, frozen: Mapping[str, Any], key: str) -> None:
    """Raise unless ``snapshot``'s tokenizer and chat template render every parity prompt as
    ``key``'s frozen ids, which that checkpoint's own tokenizer rendered (its truth child
    re-derives them on every run): the tokens a borrowed truth was computed on."""

    if check_tokenizer(snapshot, frozen, key) != "matched":
        raise ValueError(f"{snapshot} ships no tokenizer to render {key}'s prompt ids with")


def check_parity(run: ParityRun, tmp_path: Path, name: str) -> dict[str, Any]:
    """Run the truth, the calibration and the engine on ``run``'s checkpoint, judge them, keep
    the receipt ``<name>-<checkpoint>`` and assert that every prompt passes every gate on every
    path. Returns the receipt, for the family test's own checks."""

    import torch
    from safetensors.torch import load_file

    checkpoint = run.checkpoint
    snapshot = checkpoint.snapshot()
    prompt_key = run.prompt_ids_of.key
    frozen = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    if run.prompt_ids_of != checkpoint:
        check_borrowed_prompt_ids(snapshot, frozen, prompt_key)
    names = [prompt["name"] for prompt in frozen["prompts"]]
    env, log_root = offline_child_env(tmp_path, name)
    receipt_name = f"{name}-{checkpoint.slug}"
    root = (run.receipts or tmp_path) / receipt_name
    reused = run.reused_truth
    folders = {
        "truth": root / "truth" if reused is None else reused,
        "bf16": root / "bf16",
        "engine": root / "engine",
    }
    for child, folder in folders.items():
        if (child != "truth" or reused is None) and folder.exists() and any(folder.iterdir()):
            pytest.fail(
                f"{folder} holds an earlier run's output: name another "
                f"{run.settings.prefix}RECEIPTS (and, to judge against a kept truth, "
                f"{run.settings.prefix}REUSE_TRUTH)"
            )
    common = ("--model-key", prompt_key, "--prompts", str(PROMPTS_PATH))
    common += ("--greedy-tokens", str(run.greedy_tokens))
    hf_common = ("--checkpoint", str(snapshot), *common)
    receipt: dict[str, Any] = {
        "checkpoint": {"key": checkpoint.key, "snapshot": str(snapshot)},
        "prompt_ids_of": prompt_key,
        "tensor_parallel_size": run.tensor_parallel_size,
        "bounds": {"near_tie_nats": NEAR_TIE_NATS, "moe_exceptions": MOE_EXCEPTIONS, **BOUNDS},
        "outputs": {child: str(folder) for child, folder in folders.items()},
        "truth_reused": reused is not None,
        "children": {},
    }

    def record(status: str) -> None:
        receipt["status"] = status
        _write_receipt(run, receipt_name, receipt)

    record("truth running")
    if reused is None:
        truth_options = (
            "--dtype",
            "float32",
            "--device-map",
            "auto",
            "--gpus",
            str(run.truth_gpus),
        )
        if run.truth_gpu_gib is not None:
            truth_options += ("--max-gpu-gib", str(run.truth_gpu_gib))
        receipt["children"]["truth"] = _run_transformers(
            folders["truth"],
            hf_common,
            *truth_options,
            log_path=log_root / "truth.log",
            env=env,
            timeout=run.truth_timeout,
        )
    else:
        # The kept truth must be this run's: checkpoint, prompts, greedy length, top-k and
        # the code that computed it (the helper's source and the library versions).
        check_truth(
            reused,
            {
                "model_key": prompt_key,
                "config_sha256": sha256_file(snapshot / "config.json"),
                "prompts_sha256": sha256_file(PROMPTS_PATH),
                "greedy_tokens": run.greedy_tokens,
                "top_k": TOP_K,
                "dtype": "float32",
                "mode": "whole",
                "code_version": code_version(),
                "prompt_names": names,
            },
        )
        receipt["children"]["truth"] = json.loads(
            (reused / "manifest.json").read_text(encoding="utf-8")
        )
    record("calibration running")
    receipt["children"]["bf16"] = _run_transformers(
        folders["bf16"],
        hf_common,
        "--dtype",
        "bfloat16",
        "--truth",
        str(folders["truth"]),
        log_path=log_root / "bf16.log",
        env=env,
        timeout=_BF16_TIMEOUT_SECONDS,
    )
    record("engine running")
    engine_output = _run_child(
        (
            sys.executable,
            "-m",
            "tests.support.hf_parity",
            "--snapshot",
            str(snapshot),
            *common,
            "--truth",
            str(folders["truth"]),
            "--output",
            str(folders["engine"]),
            "--max-running-req",
            str(run.max_running_requests),
            "--tensor-parallel-size",
            str(run.tensor_parallel_size),
            *(() if run.expert_parallel else ("--expert-tensor-parallel",)),
            "--paths",
            *run.paths,
        ),
        log_root / "engine.log",
        env=env,
        timeout=_ENGINE_TIMEOUT_SECONDS,
    )
    engine = receipt["children"]["engine"] = child_receipt(engine_output, RECEIPT_PREFIX)
    eos_ids = _eos_token_ids(snapshot)
    comparisons = receipt["comparisons"] = []
    for prompt in names:
        truth = load_file(folders["truth"] / f"{prompt}.safetensors")
        eos_row = _first_eos_row(truth, eos_ids)
        bf16_dump = load_file(folders["bf16"] / f"{prompt}.safetensors")
        bf16 = teacher_forced(truth, *_top1(bf16_dump), eos_row=eos_row)
        # NEAR_TIE_NATS is BF16-vs-BF16 drift; a prompt where BF16 drifts further widens it.
        floor = max(NEAR_TIE_NATS, CALIBRATION_FACTOR * bf16["top1_logprob_error"]["p99"])
        # The truth's teacher-forced rows that predict its own greedy tokens.
        start = len(truth["input_ids"]) - 1
        top = truth["topk_logits"][start : start + len(truth["greedy_ids"]), 0]
        greedy_truth = {**truth, "near_tie_nats": near_tie_nats(top, floor)}
        comparison: dict[str, Any] = {
            "name": prompt,
            "prompt_tokens": len(truth["input_ids"]),
            "bf16": bf16,
            "paths": {},
        }
        for path in run.paths:
            dump = load_file(folders["engine"] / path / f"{prompt}.safetensors")
            assert torch.equal(dump["input_ids"], truth["input_ids"]), (path, prompt)
            assert torch.equal(dump["teacher_forced_ids"], truth["greedy_ids"]), (path, prompt)
            forced = teacher_forced(truth, dump["top1_ids"], dump["top1_logprobs"], eos_row=eos_row)
            comparison["paths"][path] = {"engine": forced, "gates": top1_gates(bf16, forced)}
            if path == "cold":
                comparison["greedy"] = compare_greedy(greedy_truth, dump)
        comparisons.append(comparison)
    failed = {path: path_failures(comparisons, path, moe=run.moe) for path in run.paths}
    # Each excused prompt and the truth's top-1 margin at its worst row.
    receipt["excused"] = {
        path: [
            {
                "prompt": comparison["name"],
                "worst_position": judged["engine"]["worst_position"],
                "worst_truth_margin": judged["engine"]["worst_truth_margin"],
                "worst_truth_near_tie_nats": judged["engine"]["worst_truth_near_tie_nats"],
            }
            for comparison in comparisons
            if (judged := comparison["paths"][path]).get("excused")
        ]
        for path in run.paths
    }
    record("judged")

    _check_manifests(run, receipt["children"], names)
    assert_cuda_device_metadata(engine)
    assert Path(engine["engine_root"]).is_relative_to(PYTHON_ROOT)
    assert (engine["num_ranks"], engine["paths"]) == (run.tensor_parallel_size, list(run.paths))
    assert engine["prompts"] == names
    failed = {path: prompts for path, prompts in failed.items() if prompts}
    assert not failed, f"failed gates (receipt in {root}):\n{json.dumps(failed, indent=1)}"
    return receipt


def _check_manifests(
    run: ParityRun, manifests: Mapping[str, Mapping[str, Any]], names: list[str]
) -> None:
    """The Transformers children ran their torch paths on the frozen prompts: the truth in
    IEEE FP32 (every parameter, offloaded or not, and state) on its own continuation, the
    calibration in BF16, wholly on the first GPU, on the truth's."""

    truth_gpus_used = manifests["truth"]["gpus"]
    truth_devices = {f"cuda:{index}" for index in range(truth_gpus_used)} | {"offloaded"}
    for child, dtype, device_map, teacher_forced, devices in (
        ("truth", "float32", "auto", "own", truth_devices),
        ("bf16", "bfloat16", "device", "truth", {"cuda:0"}),
    ):
        manifest = manifests[child]
        assert manifest["model_key"] == run.prompt_ids_of.key, child
        assert manifest["mode"] == "whole", child
        assert manifest["tokenizer_check"] == "matched", child
        assert (manifest["dtype"], manifest["attention"]) == (dtype, "eager"), child
        # A MoE model's experts run Transformers' per-expert loop; a dense model has none.
        if run.moe:
            assert manifest["experts"] == "eager", child
        assert (manifest["device_map"], manifest["teacher_forced"]) == (device_map, teacher_forced)
        assert set(manifest["parameter_bytes"]) <= devices, (child, manifest["parameter_bytes"])
        placed = manifest["parameter_dtypes"]
        assert all(dtypes == [dtype] for dtypes in placed.values()), (child, placed)
        assert (manifest["greedy_tokens"], manifest["top_k"]) == (run.greedy_tokens, TOP_K)
        assert all(value == "torch" for value in manifest["kernel_bindings"].values()), child
        assert manifest["blocked_packages"] == {"fla": True, "causal_conv1d": True}, child
        assert manifest["loaded_blocked_modules"] == [], child
        assert manifest["hub_kernels_enabled"] is False, child
        assert [prompt["name"] for prompt in manifest["prompts"]] == names, child
    truth = manifests["truth"]
    assert truth["fp32_precision"] == IEEE_FP32_PRECISION, truth["fp32_precision"]
    if run.reused_truth is None:
        assert truth_gpus_used == run.truth_gpus, truth_gpus_used
        gpu_budgets = {key: value for key, value in truth["max_memory"].items() if key != "cpu"}
        assert sorted(gpu_budgets) == [str(index) for index in range(run.truth_gpus)], gpu_budgets
        if run.truth_gpu_gib is not None:
            assert set(gpu_budgets.values()) == {int(run.truth_gpu_gib * (1 << 30))}, gpu_budgets


def _run_child(
    command: tuple[str, ...], log_path: Path, *, env: dict[str, str], timeout: float
) -> str:
    """Run one child in its own session, streaming to ``log_path`` and ``*.stderr.log``; return
    its stdout or fail with both logs' ends. The session is then killed: no stray keeps the GPU."""

    stderr_path = log_path.with_suffix(".stderr.log")
    with log_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            returncode: int | None = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            returncode = None
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    output = log_path.read_text(encoding="utf-8", errors="replace")
    errors = stderr_path.read_text(encoding="utf-8", errors="replace")
    report = "\n".join(
        f"{name} ({path}, last {_REPORT_LINES} lines):\n"
        + "\n".join(text.splitlines()[-_REPORT_LINES:])
        for name, path, text in (("stdout", log_path, output), ("stderr", stderr_path, errors))
    )
    if returncode is None:
        pytest.fail(f"child process timed out after {timeout:.0f}s\n{report}")
    if returncode != 0:
        pytest.fail(f"child process failed with {returncode}\n{report}")
    forbidden = find_forbidden_log_patterns(f"{output}\n{errors}")
    if forbidden:
        pytest.fail(f"child process emitted {forbidden}\n{report}")
    return output


def _run_transformers(
    output: Path,
    common: tuple[str, ...],
    *options: str,
    log_path: Path,
    env: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    stdout = _run_child(
        (
            sys.executable,
            "-m",
            "tests.support.hf_truth",
            *common,
            "--output",
            str(output),
            "--top-k",
            str(TOP_K),
            *options,
        ),
        log_path,
        env=env,
        timeout=timeout,
    )
    return child_receipt(stdout, TRUTH_RECEIPT_PREFIX)


def _write_receipt(run: ParityRun, name: str, receipt: Mapping[str, Any]) -> None:
    """Keep ``receipt`` as ``<name>.json`` in the run's receipt directory, if it names one."""

    if run.receipts is None:
        return
    path = run.receipts / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="The parity check's engine child.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--greedy-tokens", type=int, required=True)
    parser.add_argument("--max-running-req", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--expert-tensor-parallel", action="store_true")
    parser.add_argument(
        "--paths", nargs="+", choices=("cold", "cold_chunked", "prefix_hit"), required=True
    )
    _run_engine(parser.parse_args())


if __name__ == "__main__":
    main()
