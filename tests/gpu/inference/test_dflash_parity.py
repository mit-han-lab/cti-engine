"""Greedy DFlash follows HF greedy until an HF near tie, at B = 1..4 with captured rounds.

The HF reference runs in its own child process; the engine runs behind the public ``LLM``,
whose engine process frees its device memory at shutdown.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pytest
import torch

from tests.support.dflash import PINNED_CHECKPOINTS, preflight
from tests.support.gpu import child_receipt, isolated_runtime_env, run_isolated_process
from tests.support.hf_judge import (
    NEAR_TIE_BF16_SPACINGS,
    NEAR_TIE_NATS,
    compare_greedy,
    near_tie_nats,
)
from tests.support.hf_truth import ARCHITECTURES, PROMPTS_PATH, tokenize_prompt
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import write_receipt

_Q = 16
_TOKENS = 1 + 2 * _Q
_SIZES = [1, 2, 3, 4]
# Keep the repeated-row cases and add a batch with distinct contexts and frontiers.
_CASES = [(f"B{size}", [index] * size) for index, size in enumerate(_SIZES)] + [
    ("B4/mixed", [2, 0, 3, 1]),
]
_V1_PAIRS = [pair for pair in PINNED_CHECKPOINTS if pair.architecture == "DFlashDraftModel"]
_RECEIPT_PREFIX = "CTI_DFLASH_DUMP="


@torch.inference_mode()
def _reference(snapshot, prompts):
    """HF greedy continuations and their top-two margins, in a child whose Qwen3.5 reference
    runs without optional kernels (``block_optional_kernels`` precedes the transformers
    import)."""

    config = json.loads((Path(snapshot) / "config.json").read_text())
    if config["architectures"][0] in ARCHITECTURES:
        from tests.support.hf_truth import block_optional_kernels, load_reference

        block_optional_kernels()
        model = load_reference(Path(snapshot), torch.device("cuda:0"), torch.bfloat16)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="eager",
        ).eval()
    references = []
    for prompt in prompts:
        output = model(
            input_ids=torch.tensor([prompt], device="cuda"), use_cache=True, logits_to_keep=1
        )
        tokens, margins, tops = [], [], []
        for step in range(_TOKENS):
            top = output.logits[:, -1].float().topk(2, dim=-1)
            tokens.append(int(top.indices[0, 0]))
            margins.append(float(top.values[0, 0] - top.values[0, 1]))
            tops.append(float(top.values[0, 0]))
            if step + 1 < _TOKENS:
                output = model(
                    input_ids=top.indices[:, :1],
                    past_key_values=output.past_key_values,
                    use_cache=True,
                )
        references.append(
            {
                "greedy_ids": torch.tensor(tokens),
                "greedy_margins": torch.tensor(margins),
                "near_tie_nats": near_tie_nats(tops),
            }
        )
    return references


def _engine(
    snapshot: str, draft: str, prompts: list[list[int]], speculative_tokens: int | None = None
) -> dict[str, list[list[int]]]:
    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        SpeculativeOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=snapshot, max_seq_len=256),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=256),
            cache=CacheOptions(prefix="naive"),
            speculative=SpeculativeOptions(
                model_path=draft, num_speculative_tokens=speculative_tokens
            ),
        ),
        tuning=EngineTuning(cuda_graph_bs=tuple(_SIZES)),
    )
    params = SamplingParams(temperature=0, max_new_tokens=_TOKENS, ignore_eos=True)
    try:
        return {
            name: [
                list(sample.sampled_token_ids)
                for sample in sorted(
                    llm.sample([prompts[index] for index in indices], params).samples,
                    key=lambda sample: sample.request_id,
                )
            ]
            for name, indices in _CASES
        }
    finally:
        llm.shutdown()


def _references(pair, tmp_path: Path) -> tuple[dict, list, list[list[int]], list]:
    """``pair``'s pinned inputs, the four frozen prompts and their token IDs, and HF's
    greedy continuations of them, from a child process."""

    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    # Missing pinned checkpoints raise QualificationBlocked: they fail the run, never skip.
    inputs = preflight(pair)
    snapshot = inputs["snapshots"]["target"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    frozen = json.loads(PROMPTS_PATH.read_text())
    selected = frozen["prompts"][:4]
    prompts = [
        tokenize_prompt(tokenizer, prompt, frozen["chat_template_kwargs"]) for prompt in selected
    ]
    assert len({tuple(prompt) for prompt in prompts}) == 4
    assert len({len(prompt) for prompt in prompts}) == 4
    (tmp_path / "prompts.json").write_text(json.dumps(prompts))

    env, _ = isolated_runtime_env(tmp_path, "dflash-parity")
    child = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_dflash_parity", snapshot, str(tmp_path)),
        tmp_path / "reference.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )
    return inputs, selected, prompts, torch.load(child_receipt(child, _RECEIPT_PREFIX)["path"])


@pytest.mark.parametrize("pair", _V1_PAIRS, ids=lambda pair: pair.target.repo_id)
def test_dflash_default_tier_matches_transformers(pair, tmp_path: Path) -> None:
    started = time.monotonic()
    inputs, selected, prompts, reference = _references(pair, tmp_path)
    snapshot = inputs["snapshots"]["target"]
    outputs = _engine(snapshot, inputs["snapshots"]["draft"], prompts)
    comparisons = {
        f"{name}/{selected[index]['name']}/request{uid}": compare_greedy(
            reference[index], {"greedy_ids": torch.tensor(outputs[name][uid])}
        )
        for name, indices in _CASES
        for uid, index in enumerate(indices)
    }
    receipt = {
        "inputs": inputs,
        "device": torch.cuda.get_device_name(),
        "near_tie_nats": NEAR_TIE_NATS,
        "near_tie_bf16_spacings": NEAR_TIE_BF16_SPACINGS,
        "tokens": _TOKENS,
        "prompt_ids": prompts,
        "outputs": outputs,
        "comparisons": comparisons,
        "seconds": time.monotonic() - started,
    }
    (tmp_path / "dflash-parity.json").write_text(json.dumps(receipt, indent=1) + "\n")
    write_receipt(f"dflash-parity-{pair.target.repo_id.split('/')[-1]}", receipt)
    print("CTI_DFLASH_PARITY=" + json.dumps(receipt), flush=True)

    assert len(comparisons) == sum(_SIZES) + 4
    for name, comparison in comparisons.items():
        assert comparison["agrees_until_truth_near_tie"], (name, comparison)


# Drafts a round, whatever block the draft was trained on (16): one, a few, the trained
# block's and the most, 16.
_SPECULATIVE_TOKENS = (1, 3, 15, 16)


@pytest.mark.parametrize(
    "pair",
    [pair for pair in _V1_PAIRS if pair.target.repo_id == "Qwen/Qwen3.5-9B"],
    ids=lambda pair: pair.target.repo_id,
)
def test_greedy_dflash_matches_transformers_at_every_speculative_token_count(
    pair, tmp_path: Path
) -> None:
    """Qwen3.5-9B (M-RoPE, its recurrent layers verifying through round replay) with its v1
    draft: greedy rounds of any width emit HF's greedy tokens."""

    inputs, selected, prompts, reference = _references(pair, tmp_path)
    snapshot, draft = inputs["snapshots"]["target"], inputs["snapshots"]["draft"]
    failures = {}
    for count in _SPECULATIVE_TOKENS:
        outputs = _engine(snapshot, draft, prompts, count)
        for name, indices in _CASES:
            for uid, index in enumerate(indices):
                comparison = compare_greedy(
                    reference[index], {"greedy_ids": torch.tensor(outputs[name][uid])}
                )
                if not comparison["agrees_until_truth_near_tie"]:
                    failures[f"{count}/{name}/{selected[index]['name']}/{uid}"] = comparison
    assert not failures, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    dump = _reference(
        arguments.snapshot, json.loads((arguments.directory / "prompts.json").read_text())
    )
    path = arguments.directory / "reference.pt"
    torch.save(dump, path)
    print(_RECEIPT_PREFIX + json.dumps({"path": str(path)}), flush=True)
