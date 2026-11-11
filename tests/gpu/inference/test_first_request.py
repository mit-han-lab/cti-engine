"""The first requests after startup compile nothing (Qwen3.5-9B and Qwen3-8B, Qwen3.5-9B
with its DFlash draft, and Qwen3-8B with a rank-16 LoRA on every target; the default cache and
graphs): the child starts on private, empty JIT
caches, and from the moment the engine is ready no JIT cache (CuTe DSL, TVM-FFI, FlashInfer,
TorchInductor, Triton) gains an entry. This is also the GPU suites' cold start: every other
test may find kernels a runner kept.

Every prompt is a distinct 32-token one (no prefix hit). Without a draft, a child serves four
single-token requests one at a time, then four 8-token ones; the receipt keeps every time,
the startup time and the first-to-steady time-to-first-token ratio (report-only: one-time
host work, such as loading the input processor, is not a compile). With the draft, it
serves a greedy request, then a sampled one (T 0.6, top-k 20, top-p 0.95), then a batch of
each round size (1 to 4 requests) greedy, sampled at T 1, and filtered as above.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_5_9B, QWEN3_5_9B_DFLASH, QWEN3_8B, QWEN3_8B_DFLASH_B16
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    isolated_runtime_env,
    jit_cache_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import write_receipt

RECEIPT_PREFIX = "CTI_FIRST_REQUEST="
# The caches a first-use compile writes: CuTe DSL kernels, TVM-FFI modules (the radix
# comparison), FlashInfer modules, Triton kernels.
_COMPILE_CACHES = (
    "CUTE_DSL_CACHE_DIR",
    "TVM_FFI_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "TRITON_CACHE_DIR",
)
_PROMPT_TOKENS = 32
_REQUESTS = 4
_DFLASH_TOKENS = 40
_DFLASH_ROUND_SIZES = range(1, 5)


def _cache_entries() -> dict[str, set[str]]:
    """Each compile cache's files, relative to its folder."""

    entries = {}
    for variable in _COMPILE_CACHES:
        root = Path(os.environ[variable])
        files = root.rglob("*") if root.is_dir() else ()
        entries[variable] = {str(path.relative_to(root)) for path in files if path.is_file()}
    return entries


def _new_entries(before: dict[str, set[str]], after: dict[str, set[str]]) -> dict[str, list]:
    return {variable: sorted(after[variable] - before[variable]) for variable in after}


def _run_child(target: str, variant: str) -> None:
    import statistics
    import time
    from importlib.metadata import version

    import torch

    from cti.config import (
        EngineOptions,
        LoRAConfig,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        SpeculativeOptions,
    )
    from cti.llm import LLM
    from cti.models import get_model_class
    from cti.protocol import SamplingParams
    from tests.support.dflash import PINNED_CHECKPOINTS, preflight

    # DFlash rounds verify greedily, sampled, or filtered (top-k, top-p).
    greedy = {"temperature": 0.0}
    sampled = {"temperature": 1.0}
    filtered = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
    pair = next(p for p in PINNED_CHECKPOINTS if p.target.repo_id == target)
    snapshots = preflight(pair)["snapshots"]
    dflash, lora = variant == "dflash", None
    if variant == "lora":
        from transformers import AutoConfig

        architecture = AutoConfig.from_pretrained(snapshots["target"]).architectures[0]
        lora = LoRAConfig(targets=get_model_class(architecture).LORA_TARGETS, rank=16, alpha=32.0)
    started = time.perf_counter()
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=snapshots["target"], max_seq_len=2048),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=2048),
            speculative=SpeculativeOptions(model_path=snapshots["draft"]) if dflash else None,
            lora=lora,
        )
    )
    startup_seconds = time.perf_counter() - started
    ready = _cache_entries()
    served = 0
    spans: dict[str, dict] = {}

    def timed(max_tokens: int, *settings: dict) -> float:
        """Serve one request per setting at once, each on its own prompt; the seconds."""

        nonlocal served
        # Prompts differ from their first token, so none reuses another's prefix.
        prompts = [
            [1000 + 97 * (served + n) + i for i in range(_PROMPT_TOKENS)]
            for n in range(len(settings))
        ]
        served += len(settings)
        params = [
            SamplingParams(max_new_tokens=max_tokens, ignore_eos=True, **setting)
            for setting in settings
        ]
        start = time.perf_counter()
        llm.sample(prompts, params)
        return time.perf_counter() - start

    def close_span(name: str) -> None:
        """What every request so far added since the engine was ready."""

        spans[name] = _new_entries(ready, _cache_entries())

    results: dict[str, object] = {}
    try:
        if dflash:
            results["first_greedy_seconds"] = timed(_DFLASH_TOKENS, greedy)
            close_span("first_greedy")
            results["first_sampled_seconds"] = timed(_DFLASH_TOKENS, filtered)
            close_span("first_sampled")
            results["batch_seconds"] = [
                timed(_DFLASH_TOKENS, *[setting] * size)
                for setting in (greedy, sampled, filtered)
                for size in _DFLASH_ROUND_SIZES
            ]
        else:
            first_token = [timed(1, greedy)]
            close_span("first_request")
            first_token += [timed(1, greedy) for _ in range(1, _REQUESTS)]
            steady = statistics.median(first_token[1:])
            results = {
                "first_token_seconds": first_token,
                "steady_first_token_seconds": steady,
                "first_to_steady_ratio": first_token[0] / steady,
                "eight_token_seconds": [timed(8, greedy) for _ in range(_REQUESTS)],
            }
        close_span("all_requests")
    finally:
        llm.shutdown()
    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "target": target,
                "variant": variant,
                "startup_seconds": round(startup_seconds, 1),
                "ready_cache_entries": {variable: len(files) for variable, files in ready.items()},
                "spans": spans,
                **results,
                "cti_kernels": version("cti-kernels"),
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
            }
        ),
        flush=True,
    )


@pytest.mark.parametrize(
    ("target", "variant"),
    [
        pytest.param(target, variant, marks=pytest.mark.checkpoints(*pins), id=name)
        for target, variant, pins, name in (
            ("Qwen/Qwen3.5-9B", "", (QWEN3_5_9B, QWEN3_5_9B_DFLASH), "Qwen3.5-9B"),
            ("Qwen/Qwen3-8B", "", (QWEN3_8B, QWEN3_8B_DFLASH_B16), "Qwen3-8B"),
            ("Qwen/Qwen3.5-9B", "dflash", (QWEN3_5_9B, QWEN3_5_9B_DFLASH), "Qwen3.5-9B-DFlash"),
            ("Qwen/Qwen3-8B", "lora", (QWEN3_8B, QWEN3_8B_DFLASH_B16), "Qwen3-8B-LoRA"),
        )
    ],
)
def test_the_first_requests_compile_nothing(tmp_path: Path, target: str, variant: str) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env, log_root = isolated_runtime_env(tmp_path, "first-request")
    # Private, empty JIT caches: no earlier test's or run's cache can stand in for a missed
    # warm-up, and startup compiles everything cold.
    private = tmp_path / "private-cache"
    env.update(jit_cache_env(private), TRITON_CACHE_DIR=str(private / "triton"))
    command = (sys.executable, "-m", "tests.gpu.inference.test_first_request", target)
    output = run_isolated_process(
        (*command, variant) if variant else command,
        log_root / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )
    receipt = child_receipt(output, RECEIPT_PREFIX)
    write_receipt(
        f"first-request-{target.split('/')[-1]}{f'-{variant}' if variant else ''}", receipt
    )
    # The file caches are live: startup built the radix comparison module into TVM-FFI's.
    assert receipt["ready_cache_entries"]["TVM_FFI_CACHE_DIR"] > 0, receipt
    for name, new_entries in receipt["spans"].items():
        assert not any(new_entries.values()), (name, new_entries)
    assert_cuda_device_metadata(receipt)


if __name__ == "__main__":
    _run_child(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
