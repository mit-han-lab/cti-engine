from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from pathlib import Path

import pytest
import tomllib

from tests.support.paths import PYTHON_ROOT, REPO_ROOT

MINISGL_REVISION = "9a91cfafe754aa85daee49998176275667eb58f2"
QWEN35_TRANSFORMERS_REVISION = "93c8b7b485963a10800c91f55304db6be211c2bd"

# Copyright text copied exactly from the source files at the pinned 5.16.1 revision.
TRANSFORMERS_SOURCE_CREDITS = {
    "src/transformers/models/qwen3_5/modeling_qwen3_5.py": (
        "Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved."
    ),
    "src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py": (
        "Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved."
    ),
    "src/transformers/modeling_rope_utils.py": (
        "Copyright 2024 The HuggingFace Team. All rights reserved."
    ),
    "src/transformers/vision_utils.py": "Copyright 2026 The HuggingFace Inc. team. All rights reserved.",
}
_QWEN35_MODEL_SOURCE = "src/transformers/models/qwen3_5/modeling_qwen3_5.py"
TRANSFORMERS_ADAPTED_SOURCES = {
    "python/cti/models/qwen3_5/model.py": (_QWEN35_MODEL_SOURCE,),
    "python/cti/models/qwen3_5_moe/model.py": (
        _QWEN35_MODEL_SOURCE,
        "src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
    ),
    "python/cti/models/qwen3_5/vision.py": (
        _QWEN35_MODEL_SOURCE,
        "src/transformers/vision_utils.py",
    ),
    "python/cti/models/qwen3_5_moe/vision.py": (
        _QWEN35_MODEL_SOURCE,
        "src/transformers/vision_utils.py",
    ),
    "python/cti/layers/rotary.py": (
        _QWEN35_MODEL_SOURCE,
        "src/transformers/modeling_rope_utils.py",
    ),
    "python/cti/multimodal/qwen3_5/rope.py": (_QWEN35_MODEL_SOURCE,),
    "tests/modules/models/test_qwen3_5_model.py": (_QWEN35_MODEL_SOURCE,),
    "tests/modules/layers/test_rotary.py": (_QWEN35_MODEL_SOURCE,),
    "tests/gpu/layers/test_layer_operations.py": (_QWEN35_MODEL_SOURCE,),
    "tests/support/gated_delta_net.py": (_QWEN35_MODEL_SOURCE,),
}

ROOT_MIT_LICENSE_SHA256 = "fc42b25fd118a8a4e3a3c38b2e39bf56461b8336c982aebc4cbb51d15f0424e7"
# The verbatim Apache License 2.0 that the Transformers-derived sources carry.
APACHE_LICENSE_SHA256 = "50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a"
MINISGL_LICENSE_SHA256 = "de93fca1c2ab406b177d405d0b05e0598ed22f082ce1b5acb5ee7991007f096c"
MINISGL_DERIVED_SOURCES = frozenset(
    {
        "python/cti/attention/base.py",
        "python/cti/attention/flash_attention.py",
        "python/cti/attention/flashinfer.py",
        "python/cti/attention/mha.py",
        "python/cti/attention/paged.py",
        "python/cti/attention/tensorrt_llm.py",
        "python/cti/distributed/__init__.py",
        "python/cti/distributed/runtime.py",
        "python/cti/env.py",
        "python/cti/inference/_control/_cache.py",
        "python/cti/inference/_control/_engine.py",
        "python/cti/inference/_execution/_graphs.py",
        "python/cti/inference/_execution/_sampling.py",
        "python/cti/inference/_control/_scheduler.py",
        "python/cti/inference/_control/_scheduling/prefill.py",
        "python/cti/inference/_control/_scheduling/request.py",
        "python/cti/inference/_control/_scheduling/strategy.py",
        "python/cti/inference/_control/_scheduling/table.py",
        "python/cti/protocol/__init__.py",
        "python/cti/protocol/_backend.py",
        "python/cti/protocol/_codec.py",
        "python/cti/protocol/_frontend.py",
        "python/cti/protocol/link/_queue.py",
        "python/cti/protocol/link/_transport.py",
        "python/cti/protocol/_tokenizer.py",
        "python/cti/inference/_execution/runner.py",
        "python/cti/kvcache/__init__.py",
        "python/cti/kvcache/base.py",
        "python/cti/kvcache/budget.py",
        "python/cti/kvcache/naive_cache.py",
        "python/cti/kvcache/radix_cache.py",
        "python/cti/layers/__init__.py",
        "python/cti/layers/attention.py",
        "python/cti/layers/base.py",
        "python/cti/layers/embedding.py",
        "python/cti/layers/linear.py",
        "python/cti/layers/moe.py",
        "python/cti/layers/norm.py",
        "python/cti/layers/rotary.py",
        "python/cti/llm/__init__.py",
        "python/cti/llm/llm.py",
        "python/cti/models/__init__.py",
        "python/cti/models/_download.py",
        "python/cti/models/base.py",
        "python/cti/models/checkpoint.py",
        "python/cti/models/config.py",
        "python/cti/models/mapping.py",
        "python/cti/models/qwen3/config.py",
        "python/cti/models/qwen3/model.py",
        "python/cti/models/qwen3_moe/model.py",
        "python/cti/models/register.py",
        "python/cti/models/weight.py",
        "python/cti/moe/__init__.py",
        "python/cti/moe/base.py",
        "python/cti/moe/fused.py",
        "python/cti/plan.py",
        "python/cti/resolution.py",
        "python/cti/server/__init__.py",
        "python/cti/server/api_server.py",
        "python/cti/server/args.py",
        "python/cti/server/frontend.py",
        "python/cti/server/launch.py",
        "python/cti/tokenizer/detokenize.py",
        "python/cti/tokenizer/server.py",
        "python/cti/tokenizer/text.py",
        "python/cti/utils/__init__.py",
        "python/cti/utils/arch.py",
        "python/cti/utils/logger.py",
        "python/cti/utils/misc.py",
        "python/cti/utils/torch_utils.py",
        "python/cti/worker/_device_memory.py",
        "python/cti/worker/_model_host.py",
    }
)
# Original cti sources: no upstream code, so they carry the project's MIT license.
ENGINE_ONLY_SOURCES = frozenset(
    {
        "python/cti/inference/_execution/_hosting.py",
        "python/cti/inference/_execution/_speculative/_records.py",
        "python/cti/inference/_execution/_speculative/_scratch.py",
        "python/cti/inference/_execution/_speculative/dflash.py",
        "python/cti/layers/block_conv.py",
        "python/cti/models/activation_checkpointing.py",
        "python/cti/models/dflash/dflash2.py",
        "python/cti/moe/training.py",
        "python/cti/protocol/_request.py",
    }
)
# Adapted sources keep their upstream license: Mini-SGLang and DFlash are MIT,
# Transformers is Apache-2.0.
EXPECTED_SOURCE_LICENSES = {
    **dict.fromkeys(ENGINE_ONLY_SOURCES, ("MIT",)),
    **dict.fromkeys(MINISGL_DERIVED_SOURCES, ("MIT",)),
    "python/cti/models/dflash/model.py": ("MIT",),
    "python/cti/layers/rotary.py": ("Apache-2.0",),
    "python/cti/models/qwen3_5/model.py": ("Apache-2.0",),
    "python/cti/models/qwen3_5/vision.py": ("Apache-2.0",),
    "python/cti/models/qwen3_5_moe/model.py": ("Apache-2.0",),
    "python/cti/models/qwen3_5_moe/vision.py": ("Apache-2.0",),
    "python/cti/multimodal/qwen3_5/rope.py": ("Apache-2.0",),
}

_RELEASE_SOURCE_SUFFIXES = frozenset({".cu", ".cuh", ".cpp", ".h", ".py"})
_SPDX_PATTERN = re.compile(r"SPDX-License-Identifier:\s*([^\r\n]+)")
_ADAPTATION_PATTERN = re.compile(
    r"\b(?:adapted|ported|rewritten|borrowed|copied)\s+from\b", re.IGNORECASE
)


def _normalized_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()) + "\n"


def _notice_section(notice: str, name: str) -> str:
    begin = f"----- BEGIN {name} (VERBATIM) -----\n"
    end = f"----- END {name} (VERBATIM) -----"
    body = notice.split(begin, maxsplit=1)[1].split(end, maxsplit=1)[0]
    return _normalized_text(body)


def _source_licenses() -> dict[str, tuple[str, ...]]:
    licenses: dict[str, tuple[str, ...]] = {}
    package_root = PYTHON_ROOT / "cti"
    assert package_root.is_dir()
    for path in package_root.rglob("*"):
        if not path.is_file() or path.suffix not in _RELEASE_SOURCE_SUFFIXES:
            continue
        source = path.read_text()
        identifiers = tuple(_SPDX_PATTERN.findall(source))
        if _ADAPTATION_PATTERN.search(source):
            assert identifiers, f"{path.relative_to(REPO_ROOT)} declares adaptation without SPDX"
        if identifiers:
            licenses[path.relative_to(REPO_ROOT).as_posix()] = identifiers
    return licenses


def test_distribution_includes_mit_metadata_and_owned_notices() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "cti"
    assert metadata["project"]["license"] == "MIT"
    assert metadata["project"]["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES"]
    assert metadata["project"]["urls"] == {
        "Repository": "https://github.com/mit-han-lab/cti-engine",
        "Issues": "https://github.com/mit-han-lab/cti-engine/issues",
    }

    license_text = _normalized_text((REPO_ROOT / "LICENSE").read_text())
    assert hashlib.sha256(license_text.encode()).hexdigest() == ROOT_MIT_LICENSE_SHA256

    notice = (REPO_ROOT / "THIRD_PARTY_NOTICES").read_text()
    assert notice.startswith("THIRD-PARTY NOTICES\n")
    assert "cti is licensed under the MIT License" in notice
    assert "cti is licensed under the Apache" not in notice
    assert "reproduced in the root LICENSE" not in notice
    (apache_section,) = re.findall(r"----- BEGIN ([^\n]*APACHE[^\n]*) \(VERBATIM\) -----", notice)
    apache_license = _notice_section(notice, apache_section)
    assert hashlib.sha256(apache_license.encode()).hexdigest() == APACHE_LICENSE_SHA256
    minisgl_license = _notice_section(notice, "MINI-SGLANG MIT LICENSE")
    assert hashlib.sha256(minisgl_license.encode()).hexdigest() == MINISGL_LICENSE_SHA256
    for kernel_only_notice in (
        "Liger Kernel",
        "Efficient Cross Entropy",
        "Flash Linear Attention",
        "NVIDIA NCCL",
        "Unsloth",
        "vLLM",
    ):
        assert kernel_only_notice not in notice


def test_distribution_tracks_every_engine_source_local_spdx_identifier() -> None:
    assert _source_licenses() == EXPECTED_SOURCE_LICENSES


@pytest.mark.parametrize("declaration", ["Adapted", "Ported", "Rewritten", "Borrowed", "Copied"])
def test_distribution_rejects_unregistered_adaptations_without_spdx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declaration: str
) -> None:
    package = tmp_path / "python" / "cti"
    package.mkdir(parents=True)
    (package / "unregistered.py").write_text(f'"""{declaration} from Transformers."""\n')
    monkeypatch.setitem(_source_licenses.__globals__, "PYTHON_ROOT", tmp_path / "python")
    monkeypatch.setitem(_source_licenses.__globals__, "REPO_ROOT", tmp_path)

    with pytest.raises(AssertionError, match="unregistered.py declares adaptation without SPDX"):
        _source_licenses()


def test_distribution_credits_engine_modifications_without_replacing_upstream_credit() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    notice = (REPO_ROOT / "THIRD_PARTY_NOTICES").read_text()
    assert "Original cti code and modifications are copyright 2026" in readme
    assert "Original cti source and cti modifications are copyright 2026" in notice
    assert "does not replace or diminish the upstream attribution" in notice

    for local_path in MINISGL_DERIVED_SOURCES:
        source = (REPO_ROOT / local_path).read_text()
        assert "Copyright (c) 2026 cti contributors" in source


def test_distribution_preserves_engine_minisglang_lineage() -> None:
    notice = (REPO_ROOT / "THIRD_PARTY_NOTICES").read_text()
    assert "Copyright (c) 2026 sgl-project" in _notice_section(notice, "MINI-SGLANG MIT LICENSE")
    assert "https://github.com/sgl-project/mini-sglang" in notice
    assert MINISGL_REVISION in notice
    assert "https://github.com/sgl-project/sglang" in notice
    assert "design reference" in notice
    assert "No production source from the full SGLang repository" in notice
    assert "sglang-kernel package remains an\nexternal dependency" in notice

    for local_path in MINISGL_DERIVED_SOURCES:
        source = (REPO_ROOT / local_path).read_text()
        expected_license = EXPECTED_SOURCE_LICENSES[local_path][0]
        assert source.startswith(f"# SPDX-License-Identifier: {expected_license}\n")
        assert "Copyright (c) 2026 sgl-project" in source
        assert MINISGL_REVISION in source
        assert "`python/minisgl/" in source
        assert "THIRD_PARTY_NOTICES" in source


@pytest.mark.parametrize("local_path, upstream_paths", TRANSFORMERS_ADAPTED_SOURCES.items())
def test_distribution_preserves_direct_transformers_provenance(
    local_path: str, upstream_paths: tuple[str, ...]
) -> None:
    notice = (REPO_ROOT / "THIRD_PARTY_NOTICES").read_text()
    source = (REPO_ROOT / local_path).read_text()
    header = source.split("\n\n", maxsplit=1)[0]
    assert header.startswith("# SPDX-License-Identifier: Apache-2.0\n")
    assert "# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors" in header.splitlines()
    assert f"`{local_path}`" in notice
    attribution = " ".join(line.removeprefix("# ") for line in header.splitlines())
    for upstream_path in upstream_paths:
        credit = TRANSFORMERS_SOURCE_CREDITS[upstream_path]
        assert f"# SPDX-FileCopyrightText: {credit}" in header.splitlines()
        assert (
            f"Adapted from Transformers {QWEN35_TRANSFORMERS_REVISION} "
            f"(`{upstream_path}`); see THIRD_PARTY_NOTICES."
        ) in attribution
        assert f"transformers/blob/{QWEN35_TRANSFORMERS_REVISION}/{upstream_path}" in notice
        assert credit in notice


def test_distribution_registers_every_declared_source_adaptation() -> None:
    registered = EXPECTED_SOURCE_LICENSES.keys() | TRANSFORMERS_ADAPTED_SOURCES.keys()
    for root in (PYTHON_ROOT, REPO_ROOT / "tests", REPO_ROOT / "scripts", REPO_ROOT / "benchmark"):
        for path in root.rglob("*.py"):
            source = path.read_text()
            # Inspect comments and docstrings, not test fixtures or expected header strings.
            declarations = [
                token.string
                for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT
            ]
            declarations.extend(
                ast.get_docstring(node) or ""
                for node in ast.walk(ast.parse(source))
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
            )
            if _ADAPTATION_PATTERN.search("\n".join(declarations)):
                local_path = path.relative_to(REPO_ROOT).as_posix()
                assert local_path in registered, f"Unregistered adaptation: {local_path}"
                # Sources outside the package registry adapt Transformers only.
                expected_license = EXPECTED_SOURCE_LICENSES.get(local_path, ("Apache-2.0",))[0]
                assert source.startswith(f"# SPDX-License-Identifier: {expected_license}\n"), (
                    local_path
                )


def test_distribution_preserves_the_dflash_notice() -> None:
    notice = (REPO_ROOT / "THIRD_PARTY_NOTICES").read_text()
    revision = "07ebd93db9f472af339b644bb70221ad8428328a"
    assert f"https://github.com/z-lab/dflash/blob/{revision}/LICENSE" in notice
    assert "Copyright (c) 2026 Z Lab" in _notice_section(notice, "DFLASH MIT LICENSE")
    header = (PYTHON_ROOT / "cti" / "models" / "dflash" / "model.py").read_text().split("\n\n")[0]
    assert "# SPDX-FileCopyrightText: Copyright (c) 2026 Z Lab" in header.splitlines()
    assert f"Adapted from z-lab/dflash {revision}" in header
