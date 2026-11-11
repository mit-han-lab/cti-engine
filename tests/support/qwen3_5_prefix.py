"""Prefix-cache checks through the public ``LLM``: greedy runs with their sampled-token
logprobs, judged on the tokens two runs share."""

from typing import NamedTuple

import torch

from tests.support.hf_judge import (
    CALIBRATION_FACTOR,
    MEAN_ERROR_SLACK,
    NEAR_TIE_NATS,
    P99_ERROR_SLACK,
)


class Run(NamedTuple):
    """One greedy sample: its tokens, their logprobs and the prompt tokens the cache served."""

    ids: tuple[int, ...]
    logprobs: torch.Tensor
    cached: int


def prefix_llm(
    snapshot,
    *,
    cache="radix",
    budget=1024,
    rows=None,
    running=4,
    max_seq_len=4096,
    snapshot_ratio=None,
    memory_ratio=0.9,
    lora=None,
    training=None,
):
    """A Qwen3.5 ``LLM`` with prefix cache ``cache`` and ``rows`` snapshot rows."""
    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM

    options = EngineOptions(
        model=ModelOptions(path=str(snapshot), max_seq_len=max_seq_len),
        memory=MemoryOptions(inference_ratio=memory_ratio),
        scheduling=SchedulingOptions(max_running_requests=running, max_batch_tokens=budget),
        cache=CacheOptions(prefix=cache),
        lora=lora,
        training=training,
    )
    tuning = EngineTuning(
        snapshot_rows=rows, snapshot_memory_ratio=snapshot_ratio, cuda_graph_bs=(1, 2, 4)
    )
    return LLM(options, tuning=tuning)


def greedy(llm, prompts, max_tokens):
    """One batched greedy command; ``max_tokens`` is one length or one per prompt."""
    from cti.protocol import SamplingParams

    lengths = [max_tokens] * len(prompts) if isinstance(max_tokens, int) else max_tokens
    params = [
        SamplingParams(temperature=0, max_new_tokens=n, ignore_eos=True, logprobs=True)
        for n in lengths
    ]
    runs = [
        Run(
            sample.sampled_token_ids,
            torch.tensor(sample.sampling_logprobs, dtype=torch.float64),
            sample.cached_prompt_tokens,
        )
        for sample in llm.sample(prompts, params).samples
    ]
    for run, length in zip(runs, lengths, strict=True):
        assert len(run.ids) == length and run.logprobs.isfinite().all(), run
    return runs


def bitwise(left, right):
    return left.ids == right.ids and torch.equal(left.logprobs, right.logprobs)


def _shared(truth, candidate):
    """Logprob errors while both runs read the same tokens, and the step the candidate first
    picks another token (None if it never does)."""
    length = min(len(truth.ids), len(candidate.ids))
    step = next((i for i in range(length) if truth.ids[i] != candidate.ids[i]), None)
    agreed = length if step is None else step
    return (candidate.logprobs[:agreed] - truth.logprobs[:agreed]).abs(), step


def near_tie_divergence(truth, candidate, slack=0.0):
    """Whether the candidate picks the truth's tokens until a near tie. At the first other pick
    both runs read the same context, so the gap between their picks' logprobs estimates the
    truth's top-two margin, up to the candidate's logprob error ``slack``."""
    _, step = _shared(truth, candidate)
    return step is None or truth.logprobs[step] - candidate.logprobs[step] < NEAR_TIE_NATS + slack


def _pooled_errors(pairs):
    """The logprob errors of every (truth, run) pair on the tokens the two share, joined."""
    return torch.cat([_shared(truth, run)[0] for truth, run in pairs])


def judge_prefix(calibration, candidates):
    """Judge (truth, run) pairs pooled over the workload: the candidates' logprob errors on
    the tokens each shares with its truth, joined into one sample, keep their mean and p99
    within ``CALIBRATION_FACTOR`` times the same statistic of the A/A ``calibration`` pairs'
    joined errors plus slack; each candidate first picks another token only at a near tie."""
    aa, errors = _pooled_errors(calibration), _pooled_errors(candidates)
    assert len(aa) and len(errors), (len(aa), len(errors))
    bounds = {
        "mean": CALIBRATION_FACTOR * aa.mean().item() + MEAN_ERROR_SLACK,
        "p99": CALIBRATION_FACTOR * aa.quantile(0.99).item() + P99_ERROR_SLACK,
    }
    observed = {"mean": errors.mean().item(), "p99": errors.quantile(0.99).item()}
    assert all(observed[key] <= bounds[key] for key in bounds), (observed, bounds)
    for index, (truth, run) in enumerate(candidates):
        assert near_tie_divergence(truth, run, bounds["p99"]), (index, truth, run)
    return bounds


def device_used_bytes():
    """The device's memory in use by every process, read from the test process."""
    free, total = torch.cuda.mem_get_info()
    return total - free


def child_main(run, *args):
    """Report a failure before the process exits, whatever the engine left running."""
    import os
    import sys
    import traceback

    try:
        run(*args)
    except BaseException:  # noqa: BLE001 - report failure before provider-bound process exit
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
