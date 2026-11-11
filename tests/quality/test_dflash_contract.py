# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors

import json
from dataclasses import replace

import numpy as np
import pytest

from tests.support.dflash import (
    PINNED_CHECKPOINTS,
    QualificationBlocked,
    config_oracle,
    count_oracle,
    mask_oracle,
    preflight,
)

EXTENSION = {
    "conv_kernel_size": 2,
    "conv_group_size": 16,
    "selector_rank": 256,
    "selector_top_k": 16,
}


def draft_config(index):
    draft = {
        "architectures": [PINNED_CHECKPOINTS[index].architecture],
        "dflash_config": {},
        "num_hidden_layers": 6 if index == 1 else 5,
        "layer_types": ["full_attention"] * 5 if index == 0 else ["sliding_attention"] * 5,
        "sliding_window": (None, 4096, 2048)[index],
    }
    if index == 0:
        draft["block_size"] = 16
    else:
        draft["dflash_config"]["block_size"] = 16 if index == 1 else 8
    if index == 1:
        draft["layer_types"].append("full_attention")
    if index == 2:
        draft["is_causal"] = False
        draft["dflash_config"].update(EXTENSION)
    return draft


@pytest.fixture(params=range(3))
def cached_pair(request, tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    pair = PINNED_CHECKPOINTS[request.param]
    folders = {}
    for role in ("target", "draft"):
        checkpoint = getattr(pair, role)
        repo_dir = "models--" + checkpoint.repo_id.replace("/", "--")
        folder = tmp_path / repo_dir / "snapshots" / checkpoint.revision
        folder.mkdir(parents=True)
        config = draft_config(request.param) if role == "draft" else {}
        (folder / "config.json").write_text(json.dumps(config))
        if role == "target":
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (folder / name).write_text("{}")
            (folder / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "part.safetensors"}})
            )
        for weight in ("model.safetensors", "part.safetensors"):
            (folder / weight).write_bytes(b"fixture")
        folders[role] = folder

    def snapshot_download(*, repo_id, revision, cache_dir, allow_patterns, local_files_only):
        assert local_files_only is True
        assert allow_patterns == ["*.json", "*.safetensors"]
        role = "target" if repo_id == pair.target.repo_id else "draft"
        assert revision == getattr(pair, role).revision
        if not folders[role].exists():
            raise LocalEntryNotFoundError("not cached")
        return folders[role]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return pair, folders, tmp_path


def test_pinned_preflight(cached_pair):
    pair, _, cache = cached_pair
    receipt = preflight(pair, cache_dir=cache)
    assert receipt["config"]["architecture"] == pair.architecture
    assert receipt["config"]["block_size"] == (
        8 if pair.architecture == "DFlash2DraftModel" else 16
    )
    with pytest.raises(QualificationBlocked):
        preflight(replace(pair, draft=replace(pair.draft, revision="main")), cache_dir=cache)


@pytest.mark.parametrize(
    "role,name,content",
    [
        (role, name, None)
        for role, name in (
            ("target", "part.safetensors"),
            ("target", "tokenizer.json"),
            ("target", "tokenizer_config.json"),
            ("target", "config.json"),
            ("draft", "config.json"),
            ("draft", "model.safetensors"),
        )
    ]
    + [
        ("target", "model.safetensors.index.json", content)
        for content in (
            "not JSON",
            "{}",
            '{"weight_map": {"x": "../escape.safetensors"}}',
            '{"weight_map": {"x": null}}',
            '{"weight_map": {"x": []}}',
        )
    ]
    + [
        ("draft", "model.safetensors", "version https://git-lfs.github.com/spec/v1"),
        ("draft", "config.json", '{"architectures": ["WrongDraftModel"]}'),
        ("draft", "revision", "0" * 40),
        ("draft", "snapshot", None),
        ("target", "part.safetensors", "symlink"),
        ("target", "model.safetensors.index.json", "symlink"),
    ],
)
def test_preflight_blocks(cached_pair, role, name, content):
    pair, folders, cache = cached_pair
    path = folders[role] / name
    if name in ("revision", "snapshot"):
        wrong = folders[role].with_name(content or "absent")
        if content:
            folders[role].rename(wrong)
        folders[role] = wrong
    elif content is None or content == "symlink":
        path.unlink()
        if content:
            path.symlink_to(cache / "absent-blob")
    else:
        path.write_text(content)
    with pytest.raises(QualificationBlocked, match="qualification blocked"):
        preflight(pair, cache_dir=cache)


@pytest.mark.parametrize(
    "index,key,value",
    [(i, "is_causal", v) for i in (0, 1) for v in (False, True, None, 0)]
    + [(2, "is_causal", v) for v in (True, None, 0, "false", "missing")]
    + [(2, "num_hidden_layers", v) for v in ("missing", 1, 6)]
    + [(2, key, value) for key in EXTENSION for value in ("missing", True, 0, 3)]
    + [
        (2, "architectures", ["DFlashDraftModel"]),
        (2, "architectures", ["Other"]),
        (2, "layer_types", ["full_attention"]),
        (2, "layer_types", ["sliding_attention"]),
        (1, "sliding_window", 2048),
        (2, "sliding_window", 4096),
        (1, "sliding_window", None),
        (2, "sliding_window", None),
    ],
)
def test_config_contract_rejections(index, key, value):
    draft = draft_config(index)
    fields = draft["dflash_config"] if key in EXTENSION else draft
    if value == "missing":
        del fields[key]
    else:
        fields[key] = value
    with pytest.raises(ValueError):
        config_oracle(draft)


@pytest.mark.parametrize("count", [1, 5])
def test_dflash2_rejects_single_layer(count):
    draft = draft_config(2)
    draft.update(layer_types=["sliding_attention"], num_hidden_layers=count)
    with pytest.raises(ValueError, match="five sliding layers"):
        config_oracle(draft)


@pytest.mark.parametrize("q", [2, 8, 16, 32])
@pytest.mark.parametrize("budget", [0, 1, 7, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("eos_position", [None, 0, -1])
def test_acceptance_and_visible_prefix(q, budget, eos_position):
    drafts = tuple(range(10, 10 + q - 1))
    for a in range(q):
        output = (*drafts[:a], 900)
        eos = () if eos_position is None else (output[eos_position],)
        result = count_oracle(
            5, drafts, (*drafts[:a], *([900] * (q - a))), output_budget=budget, eos_token_ids=eos
        )
        assert result.accepted_drafts == a
        assert result.committed_inputs == (5, *drafts[:a])
        assert result.output_tokens == output
        stop = 1 if eos_position == 0 else a + 1
        assert result.visible_tokens == output[: min(budget, stop)]


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("start", [10000, 2**31 - 4, 2**32 + 10000])
def test_mask_boundaries(index, start):
    config = config_oracle(draft_config(index))
    start, window = np.int64(start), config["sliding_window"] or 4096
    for layer, kind in enumerate(config["layer_types"]):
        full = kind == "full_attention"
        for offset in range(config["block_size"]):
            query = start + offset
            cases = [(0, full), (query, True)]
            cases += [(query - window + delta, full or delta == 1) for delta in (-1, 0, 1)]
            cases += [
                (key, full or index == 2) for key in range(query + 1, start + config["block_size"])
            ]
            for key, visible in cases:
                assert (
                    mask_oracle(config, layer, query, np.int64(key), block_start=start) == visible
                )
