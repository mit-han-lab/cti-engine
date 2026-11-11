# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tests.support.checkpoints import (
    QWEN3_5_9B,
    QWEN3_5_9B_DFLASH,
    QWEN3_5_35B_A3B,
    QWEN3_5_35B_A3B_DFLASH,
    QWEN3_8_27B,
    QWEN3_8_27B_DFLASH2,
    QWEN3_8B,
    QWEN3_8B_DFLASH_B16,
    Checkpoint,
)


@dataclass(frozen=True)
class CheckpointPair:
    target: Checkpoint
    draft: Checkpoint
    architecture: str = "DFlashDraftModel"


PINNED_CHECKPOINTS = (
    CheckpointPair(QWEN3_8B, QWEN3_8B_DFLASH_B16),
    CheckpointPair(QWEN3_5_9B, QWEN3_5_9B_DFLASH),
    CheckpointPair(QWEN3_8_27B, QWEN3_8_27B_DFLASH2, architecture="DFlash2DraftModel"),
)
# A MoE target: its BF16 weights need an RTX PRO 6000, so only the two-GPU gates take it.
MOE_PAIR = CheckpointPair(QWEN3_5_35B_A3B, QWEN3_5_35B_A3B_DFLASH)


# The geometry of the pinned Qwen3-8B target and its z-lab b16 draft, scaled down.
TINY_TARGET_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "vocab_size": 300,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 36,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "hidden_act": "silu",
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "tie_word_embeddings": False,
}
TINY_DRAFT_CONFIG = {
    **TINY_TARGET_CONFIG,
    "architectures": ["DFlashDraftModel"],
    "num_hidden_layers": 5,
    "block_size": 16,
    "num_target_layers": 36,
    "layer_types": ["full_attention"] * 5,
    "dflash_config": {"mask_token_id": 290, "target_layer_ids": [1, 9, 17, 25, 33]},
}


def qwen3_hf_config(document: Mapping):
    """Transformers' Qwen3 config of a tiny ``document``."""

    from transformers import Qwen3Config

    return Qwen3Config(**{key: value for key, value in document.items() if key != "model_type"})


def target_engine_config(document: Mapping = TINY_TARGET_CONFIG):
    """What the engine reads of the tiny Qwen3 target ``document``."""

    from cti.models import get_model_class

    return get_model_class("Qwen3ForCausalLM").engine_config(qwen3_hf_config(document))


def selected_checkpoints(environ: Mapping[str, str] | None = None) -> CheckpointPair:
    env = os.environ if environ is None else environ
    repo = env.get("CTI_DFLASH_TARGET_REPO", PINNED_CHECKPOINTS[0].target.repo_id)
    pair = next((item for item in PINNED_CHECKPOINTS if item.target.repo_id == repo), None)
    if pair is None:
        raise ValueError(f"unqualified DFlash target: {repo!r}")
    for role in ("target", "draft"):
        checkpoint = getattr(pair, role)
        for suffix, expected in (("REPO", checkpoint.repo_id), ("REVISION", checkpoint.revision)):
            name = f"CTI_DFLASH_{role.upper()}_{suffix}"
            if env.get(name, expected) != expected:
                raise ValueError(f"{name} must be {expected!r} for this pinned pair")
    return pair


def config_oracle(draft: Mapping) -> dict:
    architecture = draft.get("architectures")
    if architecture not in (["DFlashDraftModel"], ["DFlash2DraftModel"]):
        raise ValueError("unsupported draft architecture")
    v2 = architecture == ["DFlash2DraftModel"]
    if (v2 and draft.get("is_causal") is not False) or (not v2 and "is_causal" in draft):
        raise ValueError("is_causal must be absent for v1, explicitly false for DFlash2")
    spec = draft["dflash_config"]
    q = spec.get("block_size", draft.get("block_size"))
    if q != (8 if v2 else 16):
        raise ValueError("pinned block_size must be Q8 for DFlash2 or Q16 for v1")
    for key, expected in (
        ("conv_kernel_size", 2),
        ("conv_group_size", 16),
        ("selector_rank", 256),
        ("selector_top_k", 16),
    ):
        if (v2 and (type(spec.get(key)) is not int or spec[key] != expected)) or (
            not v2 and (key in spec or key in draft)
        ):
            raise ValueError(f"invalid {key} for {architecture[0]}")
    layers, window = draft["layer_types"], draft.get("sliding_window")
    if not layers or any(x not in ("full_attention", "sliding_attention") for x in layers):
        raise ValueError("invalid layer_types")
    if v2 and (
        layers != ["sliding_attention"] * 5
        or type(draft.get("num_hidden_layers")) is not int
        or draft["num_hidden_layers"] != len(layers)
    ):
        raise ValueError("DFlash2 requires five sliding layers and matching num_hidden_layers")
    if "sliding_attention" in layers and window != (2048 if v2 else 4096):
        raise ValueError("invalid sliding_window")
    return dict(
        spec, architecture=architecture[0], block_size=q, layer_types=layers, sliding_window=window
    )


def mask_oracle(config, layer, query, key, *, block_start):
    """Callers supply existing context/block keys at logical positions."""
    if config["layer_types"][layer] == "full_attention":
        return True
    if config["architecture"] == "DFlash2DraftModel":
        return key >= block_start or query - key < config["sliding_window"]
    return query - config["sliding_window"] < key <= query


@dataclass(frozen=True)
class Round:
    accepted_drafts: int
    committed_inputs: tuple[int, ...]
    output_tokens: tuple[int, ...]
    visible_tokens: tuple[int, ...]


def count_oracle(anchor, drafts, target_tokens, *, output_budget, eos_token_ids=()) -> Round:
    """EOS/budget clip visibility; commit anchor + accepted drafts, never the bonus."""
    if not 2 <= len(target_tokens) <= 32 or len(drafts) != len(target_tokens) - 1:
        raise ValueError("Q target predictions require Q-1 drafts, Q in 2..32")
    accepted = 0
    for proposal, prediction in zip(drafts, target_tokens[:-1], strict=True):
        if proposal != prediction:
            break
        accepted += 1
    output = (*drafts[:accepted], target_tokens[accepted])
    visible = []
    for token in output[:output_budget]:
        visible.append(token)
        if token in eos_token_ids:
            break
    return Round(accepted, (anchor, *drafts[:accepted]), output, tuple(visible))


class QualificationBlocked(RuntimeError):
    """Invalid local inputs block qualification, never skip."""


def _read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def preflight(pair: CheckpointPair, *, cache_dir=None) -> dict:
    """Check local pins and readability; this does not qualify weights or execution."""
    configs, snapshots = {}, {}
    try:
        if pair not in (*PINNED_CHECKPOINTS, MOE_PAIR):
            raise ValueError("preflight requires a pinned pair")
        for role in ("target", "draft"):
            checkpoint = getattr(pair, role)
            folder = checkpoint.snapshot(cache_dir)
            repo_folder = "models--" + checkpoint.repo_id.replace("/", "--")
            if folder.parts[-3:] != (repo_folder, "snapshots", checkpoint.revision):
                raise ValueError("snapshot repo/revision does not match pin")
            configs[role] = _read_json(folder / "config.json")
            if role == "target":
                for name in ("tokenizer.json", "tokenizer_config.json"):
                    _read_json(folder / name)
            index = folder / "model.safetensors.index.json"
            if index.exists() or index.is_symlink():
                weights = _read_json(index).get("weight_map")
                if not isinstance(weights, dict) or not weights:
                    raise ValueError("invalid weight_map")
                names = set(weights.values())
            else:
                names = ("model.safetensors",)
            for name in names:
                if (
                    not isinstance(name, str)
                    or Path(name).name != name
                    or not name.endswith(".safetensors")
                ):
                    raise ValueError("invalid weight shard name")
                with (folder / name).open("rb") as stream:
                    start = stream.read(128)
                if not start or start.startswith(b"version https://git-lfs.github.com/spec/"):
                    raise ValueError(f"{name}: empty file or LFS pointer")
            snapshots[role] = str(folder)
        config = config_oracle(configs["draft"])
        if config["architecture"] != pair.architecture:
            raise ValueError("draft architecture does not match pinned pair")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise QualificationBlocked(f"DFlash qualification blocked: {error}") from error
    pins = {
        role: {"repo_id": checkpoint.repo_id, "revision": checkpoint.revision}
        for role, checkpoint in (("target", pair.target), ("draft", pair.draft))
    }
    return {
        "checkpoints": {**pins, "architecture": pair.architecture},
        "snapshots": snapshots,
        "config": config,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(preflight(selected_checkpoints(), cache_dir=args.cache_dir), indent=2))
    except (QualificationBlocked, ValueError) as error:
        parser.exit(1, f"{error}\n")
