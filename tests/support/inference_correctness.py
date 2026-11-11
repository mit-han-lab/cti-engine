from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from .checkpoints import QWEN3_0_6B
from .paths import REPO_ROOT

WORKSPACE_PYTHON_ROOT = (REPO_ROOT / "python").resolve()
SEED = 20260701
RECEIPT_PREFIX = "CTI_INFERENCE_CORRECTNESS="
TARGET_MODE = "graph-overlap-radix"
EAGER_MODE = "eager-overlap-radix"
SERIAL_MODE = "graph-serial-radix"
HF_RUNNER = "hf-differential"


class GoldenCase(NamedTuple):
    prompt_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    output_text: str
    min_hf_top2_margin: float


class DifferentialCase(NamedTuple):
    name: str
    prompt_length: int
    max_tokens: int
    prefix_token: int


class RuntimeMode(NamedTuple):
    cache_type: str
    cuda_graph_bs: tuple[int, ...]
    disable_overlap: bool


class ExecutionInvariant(NamedTuple):
    name: str
    reference: str
    candidate: str
    expected_setting_differences: tuple[str, ...] | None

    def as_receipt(self) -> dict[str, object]:
        return {
            "name": self.name,
            "reference": self.reference,
            "candidate": self.candidate,
            "expected_setting_differences": (
                list(self.expected_setting_differences)
                if self.expected_setting_differences is not None
                else None
            ),
        }


@dataclass(frozen=True)
class CorrectnessProfile:
    name: str
    batches: tuple[tuple[DifferentialCase, ...], ...]
    runtime_modes: tuple[tuple[str, RuntimeMode], ...]
    execution_invariants: tuple[ExecutionInvariant, ...]
    max_running_requests: int
    max_sequence_length: int
    max_batch_tokens: int
    inference_memory_ratio: float
    peak_memory_ratio: float
    hf_attention_implementation: str
    hf_logits_to_keep: int | None
    minimum_hf_top2_margin: float

    def runtime_mode(self, name: str) -> RuntimeMode:
        for mode_name, mode in self.runtime_modes:
            if mode_name == name:
                return mode
        raise KeyError(f"profile {self.name!r} has no runtime mode {name!r}")


# Generated independently with Transformers BF16 greedy decoding from QWEN3_0_6B.
# Every selected token had a top-2 logit margin of at least the recorded minimum.
GOLDEN_CASES = (
    GoldenCase(
        prompt_ids=(785, 14002, 315, 4017, 374),
        output_ids=(9255, 11, 323, 279, 14002, 315, 9255, 374),
        output_text=" cold, and the opposite of cold is",
        min_hf_top2_margin=0.75,
    ),
    GoldenCase(
        prompt_ids=(785, 7772, 11580, 304, 1039, 12941, 1849, 374),
        output_ids=(1112, 30, 362, 13, 43562, 425, 13, 48976),
        output_text="...? A. Mercury B. Venus",
        min_hf_top2_margin=0.625,
    ),
)

# The boundary profile exercises the 127/128/129 prefill budget transition,
# dynamic batch contraction, and graph padding 3 -> 4.
BOUNDARY_BATCHES = (
    (DifferentialCase("extend-127", 127, 4, 7772),),
    (DifferentialCase("extend-128", 128, 4, 11580),),
    (DifferentialCase("extend-129", 129, 4, 304),),
    (DifferentialCase("long", 257, 4, 1039),),
    (
        DifferentialCase("batch-2-short", 16, 4, 1112),
        DifferentialCase("batch-2-long", 24, 8, 1849),
    ),
    (
        DifferentialCase("batch-3-short", 8, 1, 14002),
        DifferentialCase("batch-3-medium", 16, 4, 315),
        DifferentialCase("batch-3-long", 24, 8, 9255),
    ),
    (
        DifferentialCase("graph-4-short", 24, 2, 362),
        DifferentialCase("graph-4-medium", 24, 4, 43562),
        DifferentialCase("graph-4-long", 24, 8, 48976),
    ),
)

# The long profile keeps requests isolated so each receipt identifies one exact
# sequence path. The 16K+1 case crosses the production prefill budget and then
# exercises 1,024 graph-enabled decode steps. Its periodic token fixture is deliberate:
# the fixed HF reference retains a high top-2 margin for the complete continuation.
LONG_BATCHES = (
    (DifferentialCase("prefill-8192", 8192, 8, 11580),),
    (DifferentialCase("prefill-decode-16385", 16385, 1024, 7772),),
    (DifferentialCase("prefill-24577", 24577, 8, 304),),
)

# All generated prompts end in the independent high-margin golden prompt while
# retaining distinct first tokens, so radix execution starts each case cold.
_DIFFERENTIAL_SUFFIX = GOLDEN_CASES[0].prompt_ids
_FILLER_TOKENS = (323, 279, 14002, 315, 9255, 374, 11)


def _runtime_modes(
    graph_batch_sizes: tuple[int, ...],
    *,
    include_serial: bool = False,
) -> tuple[tuple[str, RuntimeMode], ...]:
    modes = (
        (
            TARGET_MODE,
            RuntimeMode(
                cache_type="radix",
                cuda_graph_bs=graph_batch_sizes,
                disable_overlap=False,
            ),
        ),
        (
            EAGER_MODE,
            RuntimeMode(
                cache_type="radix",
                cuda_graph_bs=(),
                disable_overlap=False,
            ),
        ),
    )
    if include_serial:
        modes += (
            (
                SERIAL_MODE,
                RuntimeMode(
                    cache_type="radix",
                    cuda_graph_bs=graph_batch_sizes,
                    disable_overlap=True,
                ),
            ),
        )
    return modes


def _execution_invariants(*, include_serial: bool = False) -> tuple[ExecutionInvariant, ...]:
    invariants = (
        ExecutionInvariant(
            name="hf-vs-target",
            reference=HF_RUNNER,
            candidate=TARGET_MODE,
            expected_setting_differences=None,
        ),
        ExecutionInvariant(
            name="eager-vs-graph",
            reference=EAGER_MODE,
            candidate=TARGET_MODE,
            expected_setting_differences=("cuda_graph_bs",),
        ),
    )
    if include_serial:
        invariants += (
            ExecutionInvariant(
                name="serial-vs-overlap",
                reference=SERIAL_MODE,
                candidate=TARGET_MODE,
                expected_setting_differences=("overlap_scheduling",),
            ),
        )
    return invariants


BOUNDARY_PROFILE = CorrectnessProfile(
    name="boundary",
    batches=BOUNDARY_BATCHES,
    runtime_modes=_runtime_modes((1, 2, 4), include_serial=True),
    execution_invariants=_execution_invariants(include_serial=True),
    max_running_requests=4,
    max_sequence_length=512,
    max_batch_tokens=128,
    inference_memory_ratio=0.5,
    peak_memory_ratio=0.5,
    hf_attention_implementation="eager",
    hf_logits_to_keep=None,
    minimum_hf_top2_margin=0.5,
)
LONG_PROFILE = CorrectnessProfile(
    name="long",
    batches=LONG_BATCHES,
    runtime_modes=_runtime_modes((1,)),
    execution_invariants=_execution_invariants(),
    max_running_requests=1,
    max_sequence_length=32768,
    max_batch_tokens=16384,
    inference_memory_ratio=0.9,
    peak_memory_ratio=0.9,
    hf_attention_implementation="sdpa",
    hf_logits_to_keep=1,
    minimum_hf_top2_margin=0.5,
)
CORRECTNESS_PROFILES = (BOUNDARY_PROFILE, LONG_PROFILE)


def get_profile(name: str) -> CorrectnessProfile:
    for profile in CORRECTNESS_PROFILES:
        if profile.name == name:
            return profile
    raise KeyError(f"unknown correctness profile {name!r}")


def build_prompt(case: DifferentialCase) -> list[int]:
    filler_length = case.prompt_length - len(_DIFFERENTIAL_SUFFIX) - 1
    assert filler_length >= 0
    filler = [
        _FILLER_TOKENS[(index + case.prompt_length) % len(_FILLER_TOKENS)]
        for index in range(filler_length)
    ]
    prompt_ids = [case.prefix_token, *filler, *_DIFFERENTIAL_SUFFIX]
    assert len(prompt_ids) == case.prompt_length
    return prompt_ids


def expected_golden_results() -> list[dict[str, str | list[int]]]:
    return [
        {
            "text": case.output_text,
            "token_ids": list(case.output_ids),
        }
        for case in GOLDEN_CASES
    ]


def model_identity(snapshot: str | None = None) -> dict[str, str]:
    identity = {
        "repo_id": QWEN3_0_6B.repo_id,
        "revision": QWEN3_0_6B.revision,
    }
    if snapshot is not None:
        identity["snapshot"] = str(Path(snapshot).resolve())
    return identity


def edge_settings(profile: CorrectnessProfile, mode: RuntimeMode) -> dict[str, object]:
    return {
        "attention_backend": "fi",
        "cache_type": mode.cache_type,
        "cuda_graph_bs": list(mode.cuda_graph_bs),
        "dtype": "bfloat16",
        "ignore_eos": True,
        "max_batch_tokens": profile.max_batch_tokens,
        "max_running_requests": profile.max_running_requests,
        "max_sequence_length": profile.max_sequence_length,
        "inference_memory_ratio": profile.inference_memory_ratio,
        "peak_memory_ratio": profile.peak_memory_ratio,
        "overlap_scheduling": not mode.disable_overlap,
        "temperature": 0,
    }


def hf_settings(profile: CorrectnessProfile) -> dict[str, object]:
    settings: dict[str, object] = {
        "attention_implementation": profile.hf_attention_implementation,
        "dtype": "bfloat16",
        "greedy": True,
        "ignore_eos": True,
    }
    if profile.hf_logits_to_keep is not None:
        settings["logits_to_keep"] = profile.hf_logits_to_keep
    return settings


def planned_workload(profile: CorrectnessProfile = BOUNDARY_PROFILE) -> dict[str, object]:
    return {
        "profile": profile.name,
        "model": model_identity(),
        "seed": SEED,
        "minimum_hf_top2_margin": profile.minimum_hf_top2_margin,
        "expected_edge_source_root": str(WORKSPACE_PYTHON_ROOT),
        "hf_settings": hf_settings(profile),
        "edge_modes": {name: edge_settings(profile, mode) for name, mode in profile.runtime_modes},
        "execution_invariants": [
            invariant.as_receipt() for invariant in profile.execution_invariants
        ],
        "batches": [
            {
                "batch_index": batch_index,
                "workload_batch_size": len(batch),
                "cases": [
                    {
                        "name": case.name,
                        "prompt_ids": build_prompt(case),
                        "max_tokens": case.max_tokens,
                    }
                    for case in batch
                ],
            }
            for batch_index, batch in enumerate(profile.batches)
        ],
    }
