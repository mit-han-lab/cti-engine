from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.inference_correctness import (
    CORRECTNESS_PROFILES,
    EAGER_MODE,
    GOLDEN_CASES,
    HF_RUNNER,
    RECEIPT_PREFIX,
    SEED,
    SERIAL_MODE,
    TARGET_MODE,
    WORKSPACE_PYTHON_ROOT,
    CorrectnessProfile,
    build_prompt,
    edge_settings,
    get_profile,
    hf_settings,
    model_identity,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _workspace_source_file(obj: object) -> str:
    source_file = Path(inspect.getfile(obj)).resolve()
    assert source_file.is_relative_to(WORKSPACE_PYTHON_ROOT), (
        "cti resolved outside the current workspace: "
        f"{source_file} is not under {WORKSPACE_PYTHON_ROOT}"
    )
    return str(source_file)


def _model_snapshot() -> str:
    return str(QWEN3_0_6B.snapshot())


def _run_edge(profile: CorrectnessProfile, mode_name: str) -> None:

    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
    )
    from cti.env import ENV
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    source_file = _workspace_source_file(LLM)
    mode = profile.runtime_mode(mode_name)
    assert ENV.disable_overlap_scheduling is mode.disable_overlap
    snapshot = _model_snapshot()
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=snapshot, max_seq_len=profile.max_sequence_length),
            memory=MemoryOptions(
                peak_ratio=profile.peak_memory_ratio, inference_ratio=profile.inference_memory_ratio
            ),
            scheduling=SchedulingOptions(
                max_running_requests=profile.max_running_requests,
                max_batch_tokens=profile.max_batch_tokens,
            ),
            cache=CacheOptions(prefix=mode.cache_type),
        ),
        tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=tuple(mode.cuda_graph_bs)),
    )
    try:
        golden_prompts = [list(case.prompt_ids) for case in GOLDEN_CASES]
        golden_batch_results = llm.generate(
            golden_prompts,
            SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=8),
        )
        golden_single_results = [
            llm.generate(
                [prompt],
                SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=8),
            )[0]
            for prompt in golden_prompts
        ]

        cases = []
        for batch_index, batch in enumerate(profile.batches):
            prompts = [build_prompt(case) for case in batch]
            sampling_params = [
                SamplingParams(
                    temperature=0,
                    ignore_eos=True,
                    max_new_tokens=case.max_tokens,
                )
                for case in batch
            ]
            results = llm.generate(prompts, sampling_params)
            cases.extend(
                {
                    "batch_index": batch_index,
                    "workload_batch_size": len(batch),
                    "generate_call_batch_size": len(batch),
                    "name": case.name,
                    "prompt_ids": prompt_ids,
                    "max_tokens": case.max_tokens,
                    "result": result,
                }
                for case, prompt_ids, result in zip(batch, prompts, results, strict=True)
            )
    finally:
        llm.shutdown()

    receipt = {
        "profile": profile.name,
        "model": model_identity(snapshot),
        "runner": "cti",
        "source_file": source_file,
        "mode": mode_name,
        "settings": edge_settings(profile, mode),
        "golden": {
            "hf_min_top2_margins": [case.min_hf_top2_margin for case in GOLDEN_CASES],
            "batch_results": golden_batch_results,
            "single_results": golden_single_results,
        },
        "cases": cases,
    }
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def _hf_greedy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: list[int],
    max_tokens: int,
    logits_to_keep: int | None,
) -> tuple[list[int], str, list[float]]:
    import torch

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    past_key_values = None
    output_ids: list[int] = []
    forward_options: dict[str, int] = {}
    if logits_to_keep is not None:
        forward_options["logits_to_keep"] = logits_to_keep
    top2_margins: list[float] = []
    with torch.inference_mode():
        for _ in range(max_tokens):
            output = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                **forward_options,
            )
            logits = output.logits[:, -1, :].float()
            top_values, top_indices = torch.topk(logits, k=2, dim=-1)
            input_ids = top_indices[:, :1]
            output_ids.append(int(input_ids.item()))
            top2_margins.append(float((top_values[0, 0] - top_values[0, 1]).item()))
            past_key_values = output.past_key_values
    return output_ids, tokenizer.decode(output_ids), top2_margins


def _run_hf(profile: CorrectnessProfile) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    snapshot = _model_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        attn_implementation=profile.hf_attention_implementation,
        local_files_only=True,
    ).to("cuda")
    model.eval()
    try:
        cases = []
        for batch_index, batch in enumerate(profile.batches):
            for case in batch:
                prompt_ids = build_prompt(case)
                output_ids, output_text, top2_margins = _hf_greedy(
                    model,
                    tokenizer,
                    prompt_ids,
                    case.max_tokens,
                    profile.hf_logits_to_keep,
                )
                cases.append(
                    {
                        "batch_index": batch_index,
                        "workload_batch_size": len(batch),
                        "generate_call_batch_size": 1,
                        "name": case.name,
                        "prompt_ids": prompt_ids,
                        "max_tokens": case.max_tokens,
                        "result": {"text": output_text, "token_ids": output_ids},
                        "top2_margins": top2_margins,
                        "min_top2_margin": min(top2_margins),
                    }
                )
    finally:
        del model
        torch.cuda.empty_cache()

    settings = hf_settings(profile)
    settings.update({"seed": SEED, "transformers": transformers.__version__})

    receipt = {
        "profile": profile.name,
        "model": model_identity(snapshot),
        "runner": "huggingface",
        "settings": settings,
        "cases": cases,
    }
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=tuple(profile.name for profile in CORRECTNESS_PROFILES),
        default="boundary",
    )
    parser.add_argument(
        "--runner",
        required=True,
        choices=(TARGET_MODE, EAGER_MODE, SERIAL_MODE, HF_RUNNER),
    )
    args = parser.parse_args()
    profile = get_profile(args.profile)
    if args.runner == HF_RUNNER:
        _run_hf(profile)
    else:
        _run_edge(profile, args.runner)


if __name__ == "__main__":
    main()
