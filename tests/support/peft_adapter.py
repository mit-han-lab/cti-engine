"""A PEFT adapter built by PEFT on a pinned checkpoint (deterministic, no Hub adapter to pin):
Qwen3-0.6B, or Qwen3.5-9B with its Gated DeltaNet projections adapted too; Transformers'
teacher-forced rows with it, and the engine's launch-adapter comparison against them, at any
rank count."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

from tests.support.checkpoints import QWEN3_0_6B, QWEN3_5_9B, Checkpoint
from tests.support.gpu import child_receipt, isolated_runtime_env, run_isolated_process
from tests.support.hf_judge import DISAGREEMENT_SLACK, MAX_ERROR_FLOOR, is_near_tie, near_tie_nats
from tests.support.hf_truth import load_reference
from tests.support.paths import REPO_ROOT

PROMPTS = ("The capital of France is", "def fibonacci(n):\n    ")
RECEIPT_PREFIX = "CTI_PEFT_TRANSFORMERS="
FORCED_TOKENS = 24
# Both sides run BF16, each off the FP32 truth: their mean top-1 logprob error, in nats, is
# bounded as BF16 Transformers' is in the parity calibration.
MEAN_ERROR_BOUND = 0.1
_ATTENTION_MLP = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_GATED_DELTA_NET = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
# Each checkpoint, the modules its adapter adapts (every one its LoRA targets hold) and the
# standard deviation of its random B: large enough that the adapter moves the logprobs, small
# enough that the adapted model stays near its checkpoint, where BF16 runs agree. Qwen3.5-9B
# at 0.1 read 0.12 nats from Transformers BF16, but Transformers BF16 itself read 0.085 from
# its FP32 run and the engine 0.053 (mean, worse prompt): shared BF16 noise far from the
# checkpoint, not the engine's; each target group alone read under 0.04.
ADAPTERS = MappingProxyType(
    {
        "Qwen3-0.6B": (QWEN3_0_6B, _ATTENTION_MLP, 0.02),
        "Qwen3.5-9B": (QWEN3_5_9B, _ATTENTION_MLP + _GATED_DELTA_NET, 0.05),
    }
)


def _transformers_rows(
    checkpoint: Checkpoint, target_modules: tuple[str, ...], b_std: float, folder: Path
) -> list[dict[str, list]]:
    """Save a non-zero PEFT adapter to ``folder``; per prompt, its greedy continuation's token
    ids and each next token's top-1 id, logprob and margin, teacher-forced with the adapter.
    The adapter must move them."""

    import peft
    import torch
    from transformers import AutoTokenizer

    snapshot = checkpoint.snapshot()
    model = load_reference(snapshot, torch.device("cuda"), torch.bfloat16, attention="sdpa")
    config = peft.LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=list(target_modules)
    )
    # PEFT draws A from the global generator: seeded, the adapter is the same every run.
    torch.manual_seed(0)
    adapted_model = peft.get_peft_model(model, config)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in adapted_model.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * b_std)
    adapted_model.save_pretrained(folder)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)

    def rows(model) -> list[dict[str, list]]:
        result = []
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            ids = model.generate(
                ids, max_new_tokens=FORCED_TOKENS, min_new_tokens=FORCED_TOKENS, do_sample=False
            )
            logits = model(ids).logits[0, :-1].float()
            top = logits.log_softmax(-1).topk(2, dim=-1)
            result.append(
                {
                    "token_ids": ids[0].tolist(),
                    "top1": top.indices[:, 0].tolist(),
                    "logprobs": top.values[:, 0].tolist(),
                    "margins": (top.values[:, 0] - top.values[:, 1]).tolist(),
                    "near_tie_nats": near_tie_nats(logits.max(-1).values).tolist(),
                }
            )
        return result

    with torch.inference_mode():
        adapted = rows(adapted_model)
        with adapted_model.disable_adapter():
            base = rows(adapted_model)
    moved = [
        abs(a - b)
        for with_adapter, without in zip(adapted, base, strict=True)
        for a, b in zip(with_adapter["logprobs"], without["logprobs"], strict=True)
    ]
    assert max(moved) > 0.5, max(moved)
    return adapted


def assert_launch_adapter_follows_transformers(
    tmp_path: Path, num_ranks: int, model: str = "Qwen3-0.6B"
) -> None:
    """A PEFT adapter on ``model`` launched on ``num_ranks`` GPUs follows Transformers with
    it."""

    import torch

    from cti.config import (
        EngineOptions,
        LoRAConfig,
        MemoryOptions,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    if torch.cuda.device_count() < num_ranks:
        pytest.skip(f"needs {num_ranks} GPUs")
    checkpoint = ADAPTERS[model][0]
    # Transformers runs in its own process, so its memory is gone before the engine starts.
    env, log_root = isolated_runtime_env(tmp_path, "peft-transformers")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.support.peft_adapter", model, str(tmp_path / "adapter")),
        log_root / "peft-transformers.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=1200,
    )
    truth = child_receipt(output, RECEIPT_PREFIX)
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=str(checkpoint.snapshot()), max_seq_len=256),
            parallel=ParallelOptions(num_ranks=num_ranks),
            memory=MemoryOptions(inference_ratio=0.85, peak_ratio=0.9),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=256),
            lora=LoRAConfig(adapter=str(tmp_path / "adapter")),
        )
    )
    try:
        params = SamplingParams(temperature=0, max_new_tokens=1, prompt_logprobs=1)
        samples = llm.sample([row["token_ids"] for row in truth], params).samples
    finally:
        llm.shutdown()

    for expected, sample in zip(truth, samples, strict=True):
        scored = sample.prompt_logprobs[1:]
        top1 = [row.top[0][0] for row in scored]
        flips = sum(
            not is_near_tie(margin, nats) and a != b
            for margin, nats, a, b in zip(
                expected["margins"], expected["near_tie_nats"], expected["top1"], top1, strict=True
            )
        )
        assert flips <= DISAGREEMENT_SLACK, (expected, top1)
        # Over the greedy continuation the forced token is the truth's top-1.
        errors = [
            abs(a - row.logprob)
            for a, row in zip(
                expected["logprobs"][-FORCED_TOKENS:], scored[-FORCED_TOKENS:], strict=True
            )
        ]
        assert sum(errors) / len(errors) <= MEAN_ERROR_BOUND, errors
        assert max(errors) <= MAX_ERROR_FLOOR, errors


if __name__ == "__main__":
    print(
        RECEIPT_PREFIX + json.dumps(_transformers_rows(*ADAPTERS[sys.argv[1]], Path(sys.argv[2])))
    )
