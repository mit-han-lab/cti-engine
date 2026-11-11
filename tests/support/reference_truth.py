"""The FP32 Transformers truth of teacher-forced token sequences, through the reference cache.

A GPU test asks for the sequences its engine produced, in ``qwen3_5_reference_cache``'s
``top4`` row format. What ``CTI_REFERENCE_CACHE_DIR`` lacks, a child computes on this
host and stores there (without that variable, in a cache that serves the test only); the test
fails only when this host cannot hold the FP32 model, naming the job another host can compute.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from tests.support import qwen3_5_reference_cache as reference_cache
from tests.support.checkpoints import Checkpoint
from tests.support.gpu import child_receipt, offline_child_env, run_isolated_process
from tests.support.hf_judge import is_near_tie, near_tie_nats
from tests.support.hf_truth import content_digest as digest
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import SETTINGS

_COMMAND = "python -m tests.support.qwen3_5_reference_cache"
_CACHE = "CTI_REFERENCE_CACHE_DIR"
# What keys an image prompt's truth: its original images, HF's FP32 pixels and grid, and the
# engine's M-RoPE positions.
_IMAGE_IDENTITY = ("originals", "pixel_values", "image_grid_thw", "engine_positions")
# The image inputs a job carries to ``score_continuations``: tensors only, which
# ``torch.load(weights_only=True)`` reads back.
_IMAGE_INPUTS = ("input_ids", "pixel_values", "image_grid_thw", "engine_positions")


class SequenceTruth:
    """One checkpoint's truth; its children log under ``tmp_path``."""

    def __init__(self, checkpoint: Checkpoint, snapshot: Path, tmp_path: Path) -> None:
        self.checkpoint, self.snapshot, self.tmp_path = checkpoint, snapshot, tmp_path

    def require(self, sequences, job: Path, images: Mapping | None = None) -> tuple[dict, dict]:
        """Per (prompt, continuation ids), the truth's top-2 scores with its greedy ids and
        margins and near-tie bounds; and how many came from the cache and what computing the rest took. The
        misses' job is written to ``job``. A text prompt is its ids; ``images`` maps an image
        prompt to its ``prepare_image_inputs``, whose ids it takes, so prompts whose ids match
        but whose images differ (same-sized images) are told apart by their keys."""

        with pytest.MonkeyPatch.context() as patch:
            if _CACHE not in os.environ:
                patch.setenv(_CACHE, str(self.tmp_path / "reference-cache"))
            return self._require(sequences, job, images or {})

    def _require(self, sequences, job: Path, images: Mapping) -> tuple[dict, dict]:
        cache = reference_cache.ReferenceCache(rows="top4")
        identity = reference_cache.reference_identity(self.snapshot, self.checkpoint)
        keys = {
            (prompt, continuation): cache.key(
                _prompt_ids(prompt, images.get(prompt)),
                list(continuation),
                **identity,
                images=digest(_image_identity(images.get(prompt))),
            )
            for prompt, continuation in sequences
        }
        misses = {}
        for (prompt, continuation), key in keys.items():
            if cache.load(key) is None:
                inputs = images.get(prompt)
                request = {
                    "prompt_ids": _prompt_ids(prompt, inputs),
                    "continuation_ids": list(continuation),
                }
                if inputs is not None:
                    request["images"] = {name: inputs[name] for name in _IMAGE_INPUTS}
                misses[key] = request
        report = {"cache": str(cache.folder), "cached": len(keys) - len(misses), "computed": 0}
        if misses:
            report.update(self._compute(misses, identity, job))
        found = {sequence: cache.load(key) for sequence, key in keys.items()}
        if any(entry is None for entry in found.values()):
            raise RuntimeError(f"the truth child did not store every request of {job}")
        return {sequence: _truth(entry) for sequence, entry in found.items()}, report

    def _compute(self, misses: dict, identity: dict, job: Path) -> dict:
        job.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "manifest": {"model_key": self.checkpoint.key, "recipe": None},
                "rows": "top4",
                "requests": misses,
                "implementation": identity["implementation_hash"],
            },
            job,
        )
        env, log_root = offline_child_env(self.tmp_path, "reference-truth")
        started = time.monotonic()
        output = run_isolated_process(
            (sys.executable, "-m", "tests.support.qwen3_5_reference_cache", str(job)),
            log_root / "child.log",
            env=env,
            cwd=REPO_ROOT,
            timeout=SETTINGS.positive_int("TRUTH_TIMEOUT", 7200),
        )
        result = child_receipt(output, reference_cache.RECEIPT_PREFIX)
        if "unavailable" in result:
            pytest.fail(
                f"no FP32 truth for {len(misses)} sequences of {self.checkpoint.key} in "
                f"{os.environ[_CACHE]}, and this host cannot compute "
                f"it: {result['unavailable']}. Compute them on a host whose GPU and CPU RAM hold "
                f"the FP32 model, with CTI_REFERENCE_CACHE_DIR naming that cache: "
                f"`{_COMMAND} {job}`; then rerun this test with it.",
                pytrace=False,
            )
        return {**result, "seconds": round(time.monotonic() - started, 3)}


def judge_divergence(truth, step: int, tokens) -> dict:
    """Two greedy continuations of one prompt that first differ at ``step``, with ``tokens``,
    against ``truth``, the truth of either one (it is teacher-forced on their common prefix
    there): they may differ only where both tokens are a near tie with the truth's top-1,
    each among its top four and within the near-tie bound below it (a k-way tie)."""

    ids = [int(token) for token in truth["topk_ids"][step]]
    logits = truth["topk_logits"][step]
    gaps = [float(logits[0] - logits[ids.index(t)]) if t in ids else None for t in map(int, tokens)]
    return {
        "step": step,
        "tokens": [int(token) for token in tokens],
        "truth_top": ids,
        "truth_gaps": gaps,
        "allowed": all(
            gap is not None and bool(is_near_tie(gap, near_tie_nats(logits[0]))) for gap in gaps
        ),
    }


def teacher_forced_top1(truths, ids, logprobs) -> dict:
    """Top-1 picks teacher-forced on the sequences of ``truths`` (``require``'s entries; ``ids``
    and ``logprobs`` are their positions concatenated in order) against the truth: flips where
    the truth is decided, and the top-1 log-probability error where the pick agrees, the only
    positions where a sampled log-probability is the truth token's."""

    truth_ids = torch.cat([truth["greedy_ids"] for truth in truths])
    truth_logprobs = torch.cat(
        [truth["topk_logits"][:, 0] - truth["logsumexp"] for truth in truths]
    )
    decided = ~is_near_tie(
        torch.cat([truth["greedy_margins"] for truth in truths]),
        torch.cat([truth["near_tie_nats"] for truth in truths]),
    )
    ids, logprobs = torch.tensor(ids), torch.tensor(logprobs, dtype=torch.float32)
    if ids.shape != truth_ids.shape or logprobs.shape != truth_ids.shape:
        raise ValueError(f"{tuple(ids.shape)} picks for {tuple(truth_ids.shape)} truth positions")
    agrees = ids == truth_ids
    error = (logprobs - truth_logprobs.float()).abs()[agrees]
    return {
        "positions": len(ids),
        "decided": int(decided.sum()),
        "decided_disagreements": int((decided & ~agrees).sum()),
        "mean": float(error.mean()),
        "p99": float(error.quantile(0.99)),
        "max": float(error.max()),
        "finite": bool(logprobs.isfinite().all()),
    }


def free_run_flips(truth, sequences) -> dict:
    """Free-run outputs against the FP32 truth of those same sequences: ``sequences`` are
    ``(prompt, output)`` keys of ``truth`` (``require``'s entries); the positions where the truth
    is decided, and those of them whose output token is not the truth's top-1. Each distinct
    position (its prompt and the output before it) counts once: a repeated request is not more
    evidence."""

    decided = disagreements = 0
    seen = set()
    for key in sequences:
        entry = truth[key]
        prompt, output = key
        tokens = torch.tensor(output)
        if tokens.shape != entry["greedy_ids"].shape:
            raise ValueError(f"{tuple(tokens.shape)} tokens for {tuple(entry['greedy_ids'].shape)}")
        settled = ~is_near_tie(entry["greedy_margins"], entry["near_tie_nats"])
        for step in range(len(output)):
            position = (tuple(prompt), tuple(output[:step]))
            if position in seen or not bool(settled[step]):
                continue
            seen.add(position)
            decided += 1
            disagreements += int(output[step] != int(entry["greedy_ids"][step]))
    return {"decided": decided, "decided_disagreements": disagreements}


def _prompt_ids(prompt, inputs):
    return list(prompt) if inputs is None else inputs["input_ids"][0].tolist()


def _image_identity(inputs):
    return None if inputs is None else [inputs[name] for name in _IMAGE_IDENTITY]


def _truth(entry):
    top = entry["topk_logits"]
    return {
        **entry,
        "greedy_ids": entry["topk_ids"][:, 0],
        "greedy_margins": top[:, 0] - top[:, 1],
        "near_tie_nats": near_tie_nats(top[:, 0]),
    }
