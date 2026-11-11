"""The Qwen3-0.6B FP32 LoRA backward contract: HF PEFT FP32 and the engine both measured
against HF PEFT FP64 on the same weights, batch and initial adapter.

The engine passes when its loss is within 5e-6 relative of FP64's, its gradient
collection's relative L2 error is at most 1.5x HF FP32's, and every named LoRA gradient's
error RMS is at most 4x HF FP32's (plus a 1e-12 floor). The op-level property that the
fused gated-MLP LoRA backward's FP32 error stays within 1.25x the unfused path's is owned
by cti-kernels' own tests; this test does not repeat it.

Receipts, 2026-10-19, on a cti-kernels prerelease with torch 2.13. Collection relative L2
vs FP64: RTX 5090 HF FP32 2.83e-5, engine 1.35e-5 (ratio 0.48); RTX PRO 6000 HF FP32
1.84e-5, engine 1.73e-5 (ratio 0.94). Worst per-tensor ratio 1.44 (5090), 2.31 (PRO).
Loss relative error 5.9e-7 (5090), 2.9e-7 (PRO). With the fused MLP op off the engine's
collection error was 0.65e-5 (5090) and 1.45e-5 (PRO), its worst per-tensor ratio 2.63.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TypedDict

import torch

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.lora_backward import (
    BackwardComparisonSpec,
    ParameterSchemaEntry,
    ProfileFingerprint,
)

CASE_ID = "qwen3-0.6b-padded-r4-fp32-seed-20260825"
RECEIPT_PREFIX = "CTI_LORA_BACKWARD_HF="

_CASE_SEED = 20260825
_PAD_TOKEN_ID = 151643
_SAMPLES = (
    (785, 14002, 315, 4017, 374, 9255, 11, 323),
    (785, 7772, 11580, 304),
    (785, 14002),
)
_NUM_LAYERS = 28
_LORA_RANK = 4
_LORA_ALPHA = 8.0
_EXPECTED_NUM_TARGET_TOKENS = 11
_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Canonical dense-Qwen3 projection layout. Provider-specific names and fused
# projection mappings belong to runner.py; this module owns only the external
# canonical schema.
_PROJECTIONS = (
    ("self_attn", "q_proj", 1024, 2048),
    ("self_attn", "k_proj", 1024, 1024),
    ("self_attn", "v_proj", 1024, 1024),
    ("self_attn", "o_proj", 2048, 1024),
    ("mlp", "gate_proj", 1024, 3072),
    ("mlp", "up_proj", 1024, 3072),
    ("mlp", "down_proj", 3072, 1024),
)


class _LoRACasePayload(TypedDict):
    rank: int
    alpha: float
    targets: tuple[str, ...]


class _BackwardCasePayload(TypedDict):
    case_id: str
    model: str
    revision: str
    seed: int
    token_ids: torch.Tensor
    lengths: torch.Tensor
    pad_token_id: int
    lora: _LoRACasePayload
    initial_lora: dict[str, torch.Tensor]


def _expected_parameter_schema() -> tuple[ParameterSchemaEntry, ...]:
    entries = []
    for layer in range(_NUM_LAYERS):
        for parent, projection, input_size, output_size in _PROJECTIONS:
            prefix = f"model.layers.{layer}.{parent}.{projection}"
            entries.extend(
                (
                    (f"{prefix}.lora_A", (_LORA_RANK, input_size), "float32"),
                    (f"{prefix}.lora_B", (output_size, _LORA_RANK), "float32"),
                )
            )
    return tuple(entries)


_EXPECTED_PARAMETER_SCHEMA = _expected_parameter_schema()
QUALIFIED_DEVICES = (
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
)
# The RTX 5090 profile; the RTX PRO 6000 one differs only in ``device``.
_RTX_5090_PROFILE = ProfileFingerprint(
    model=QWEN3_0_6B.repo_id,
    revision=QWEN3_0_6B.revision,
    python_version="3.12.13",
    torch_version="2.13.0+cu130",
    cuda_version="13.0",
    transformers_version="5.16.1",
    peft_version="0.19.1",
    device="NVIDIA GeForce RTX 5090",
    capability=(12, 0),
    dtype="float32",
    case_id=CASE_ID,
    training_attention="torch-sdpa",
    cublas_workspace_config=":4096:8",
    deterministic_algorithms=True,
    deterministic_warn_only=False,
    float32_matmul_precision="highest",
    cuda_matmul_allow_tf32=False,
    cudnn_allow_tf32=False,
)


def comparison_spec(device: str) -> BackwardComparisonSpec:
    """The contract on one of ``QUALIFIED_DEVICES``."""

    if device not in QUALIFIED_DEVICES:
        raise ValueError(f"unqualified device: {device}")
    profile = replace(_RTX_5090_PROFILE, device=device)
    return BackwardComparisonSpec(
        expected_profile=profile,
        expected_reference_profile=replace(profile, dtype="float64"),
        expected_parameter_schema=_EXPECTED_PARAMETER_SCHEMA,
        expected_num_target_tokens=_EXPECTED_NUM_TARGET_TOKENS,
        loss_relative_bound=5e-6,
        collection_error_ratio_bound=1.5,
        tensor_error_ratio_bound=4.0,
        tensor_error_floor=1e-12,
    )


def build_case_payload() -> _BackwardCasePayload:
    """Build the shared CPU fixture consumed independently by every runner role."""

    lengths = torch.tensor(tuple(map(len, _SAMPLES)), dtype=torch.int64)
    token_ids = torch.full(
        (len(_SAMPLES), int(lengths.max())),
        _PAD_TOKEN_ID,
        dtype=torch.int64,
    )
    for row, sample in enumerate(_SAMPLES):
        token_ids[row, : len(sample)] = torch.tensor(sample, dtype=torch.int64)

    num_target_tokens = int(lengths.sum()) - len(_SAMPLES)
    if num_target_tokens != _EXPECTED_NUM_TARGET_TOKENS:
        raise RuntimeError("fixed backward case has an invalid target-token count")

    generator = torch.Generator(device="cpu").manual_seed(_CASE_SEED)
    initial_lora = {
        name: torch.randn(shape, dtype=torch.float32, generator=generator).mul_(0.02)
        for name, shape, _ in _EXPECTED_PARAMETER_SCHEMA
    }
    if any(torch.count_nonzero(tensor).item() == 0 for tensor in initial_lora.values()):
        raise RuntimeError("fixed backward case produced an all-zero LoRA tensor")

    return {
        "case_id": CASE_ID,
        "model": QWEN3_0_6B.repo_id,
        "revision": QWEN3_0_6B.revision,
        "seed": _CASE_SEED,
        "token_ids": token_ids,
        "lengths": lengths,
        "pad_token_id": _PAD_TOKEN_ID,
        "lora": {
            "rank": _LORA_RANK,
            "alpha": _LORA_ALPHA,
            "targets": _TARGET_MODULES,
        },
        "initial_lora": initial_lora,
    }


__all__ = [
    "CASE_ID",
    "QUALIFIED_DEVICES",
    "RECEIPT_PREFIX",
    "build_case_payload",
    "comparison_spec",
]
