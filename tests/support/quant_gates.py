"""The quantized checkpoints' GPU gates, run by hand by
``tests/hf_parity/test_quant_gates.py``: their catalog, settings, the parity prompts under each
checkpoint, the dequantized FP32 truth folder and the children each gate runs.

Every gate writes its result to ``$QUANT_GATE_OUT/<gate>/<case>.json`` before it asserts. The
other ``QUANT_GATE_*`` settings are listed in ``tests/hf_parity/README.md``."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from tests.support.checkpoints import (
    QWEN3_5_35B_A3B,
    QWEN3_5_35B_A3B_FP8,
    QWEN3_8_27B,
    QWEN3_8_27B_FP8,
    QWEN3_8_27B_NVFP4,
    QWEN3_8B,
    QWEN3_8B_FP8,
    Checkpoint,
)
from tests.support.hf_parity import ParitySettings
from tests.support.hf_truth import PROMPTS_PATH, sha256_file, tokenize_prompt
from tests.support.paths import PYTHON_ROOT
from tests.support.quant_truth import NUMERICS_PATH, load_numerics

SETTINGS = ParitySettings("QUANT_GATE_")
# The cti-kernels attention dispatch issue that makes graph and eager differ at B > 1.
GRAPH_EAGER_ISSUE = "cti-kernels attention dispatch differs between graph and eager at B>1"
# Checkpoint-name suffixes of a quantized linear's companions.
SCALE_SUFFIXES = ("weight_scale_inv", "weight_scale", "weight_scale_2", "input_scale")
_SHARD_BYTES = 4 << 30


@dataclass(frozen=True)
class QuantCheckpoint:
    """A quantized checkpoint and its BF16 sibling, both pinned. ``floor_disagreement_share``:
    G-C's ceiling on the BF16 floor's own decided flips, when measured above ``hf_judge``'s.
    ``trains``: whether its recipe trains (ModelOpt checkpoints serve, with LoRA, only)."""

    name: str
    pin: Checkpoint
    sibling: Checkpoint
    moe: bool = False
    floor_disagreement_share: float | None = None
    trains: bool = True


@dataclass(frozen=True)
class Topology:
    """The engine's ranks: one GPU, or ``ranks`` with a MoE model's experts whole (expert
    parallel) or cut by width (expert tensor parallel)."""

    name: str
    ranks: int = 1
    expert_parallel: bool = True


@dataclass(frozen=True)
class GateCase:
    checkpoint: QuantCheckpoint
    topology: Topology
    # The devices the README runs this case on (documentation; never enforced).
    devices: tuple[str, ...]

    @property
    def id(self) -> str:
        return f"{self.checkpoint.name}-{self.topology.name}"


CHECKPOINTS = MappingProxyType(
    {
        checkpoint.name: checkpoint
        for checkpoint in (
            QuantCheckpoint("qwen3-8b-fp8", QWEN3_8B_FP8, QWEN3_8B),
            QuantCheckpoint("qwen3.8-27b-nvfp4", QWEN3_8_27B_NVFP4, QWEN3_8_27B, trains=False),
            QuantCheckpoint(
                "qwen3.8-27b-fp8",
                QWEN3_8_27B_FP8,
                QWEN3_8_27B,
                # The BF16 floor flips 6/282 (system-noble-gases) and 33/951 (lighthouse-log)
                # decided rows against the FP32 truth, RTX PRO 6000, 2026-10-18; 3.5% plus a
                # 0.5-point margin. The engine stays bounded by the floor as everywhere.
                floor_disagreement_share=0.04,
            ),
            QuantCheckpoint(
                "qwen3.5-35b-a3b-fp8",
                QWEN3_5_35B_A3B_FP8,
                QWEN3_5_35B_A3B,
                moe=True,
                # The BF16 floor flips 26/950 decided rows (lighthouse-log) against the FP32
                # truth, RTX PRO 6000, 2026-10-19 (the same in every topology); 2.74% plus a
                # 0.5-point margin. The engine stays bounded by the floor as everywhere.
                floor_disagreement_share=0.035,
            ),
        )
    }
)
_ONE = Topology("tp1")
CASES = (
    GateCase(CHECKPOINTS["qwen3-8b-fp8"], _ONE, ("5090",)),
    GateCase(CHECKPOINTS["qwen3.8-27b-nvfp4"], _ONE, ("5090", "PRO")),
    GateCase(CHECKPOINTS["qwen3.8-27b-fp8"], _ONE, ("PRO",)),
    GateCase(CHECKPOINTS["qwen3.5-35b-a3b-fp8"], Topology("ep1"), ("PRO",)),
    GateCase(CHECKPOINTS["qwen3.5-35b-a3b-fp8"], Topology("ep2", 2), ("PRO pair",)),
    GateCase(
        CHECKPOINTS["qwen3.5-35b-a3b-fp8"], Topology("tp2", 2, expert_parallel=False), ("PRO pair",)
    ),
)


def out_dir() -> Path:
    """``QUANT_GATE_OUT``: where every gate writes its result and keeps its children's output."""

    out = SETTINGS.path("OUT")
    if out is None:
        pytest.fail("set QUANT_GATE_OUT to the directory the gate results go to")
    return out


def write_result(gate: str, case: str, result: Mapping[str, Any]) -> Path:
    """Keep ``result`` as ``$QUANT_GATE_OUT/<gate>/<case>.json``."""

    path = out_dir() / gate / f"{case}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_folder(gate: str, case: str) -> Path:
    """A new, empty folder for one run's children under ``$QUANT_GATE_OUT/<gate>/<case>``."""

    folder = out_dir() / gate / case
    if folder.exists() and any(folder.iterdir()):
        pytest.fail(f"{folder} holds an earlier run's output: name another QUANT_GATE_OUT")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def visible_gpus() -> int:
    import torch

    return torch.cuda.device_count()


def require_ranks(topology: Topology) -> None:
    if visible_gpus() < topology.ranks:
        pytest.fail(f"{topology.name} needs {topology.ranks} visible GPUs")


# ================================ The parity prompts ================================


def prompt_set(checkpoint: QuantCheckpoint) -> tuple[Path, str]:
    """The prompts file and id-set key every child of ``checkpoint`` reads, the BF16
    sibling's: the frozen parity prompts where they hold its ids (so a sibling truth kept by the
    parity checks matches), else ``derive_prompts``. The quantized checkpoint's tokenizer
    re-derives them either way."""

    from tests.support.hf_truth import check_tokenizer

    frozen = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    key = checkpoint.sibling.key
    if key in frozen["models"]:
        check_tokenizer(checkpoint.pin.snapshot(), frozen, key)
        return PROMPTS_PATH, key
    return derive_prompts(checkpoint), key


def derive_prompts(checkpoint: QuantCheckpoint) -> Path:
    """The frozen parity prompts with token ids under the sibling's key, derived by the
    quantized checkpoint's tokenizer, in ``$QUANT_GATE_OUT/prompts``. The same input writes
    the same bytes, so a kept truth's ``prompts_sha256`` matches across runs."""

    from transformers import AutoTokenizer

    frozen = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.pin.snapshot(), local_files_only=True)
    sibling = checkpoint.sibling
    prompts = []
    for prompt in frozen["prompts"]:
        ids = tokenize_prompt(tokenizer, prompt, frozen["chat_template_kwargs"])
        prompts.append(
            {**prompt, "num_tokens": {sibling.key: len(ids)}, "token_ids": {sibling.key: ids}}
        )
    models = {sibling.key: {"repo": sibling.repo_id, "revision": sibling.revision}}
    derived = {**frozen, "models": models, "prompts": prompts}
    folder = out_dir() / "prompts"
    path = folder / f"parity_prompts-{sibling.slug}.json"
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(derived, indent=1, sort_keys=True) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return path


def prompt_ids(prompts: Path, key: str) -> list[tuple[str, list[int]]]:
    from tests.support.hf_truth import load_prompts

    return load_prompts(prompts, key)[1]


# ============================ The dequantized FP32 folder ============================


def dequantized_folder(checkpoint: QuantCheckpoint) -> Path:
    """The checkpoint's weights dequantized to FP32 by ``layers/quant/numerics.py``, laid out
    as the BF16 sibling's Transformers class reads them, with the activation quantization of
    every quantized module in ``activation_quant.json`` (``tests/support/quant_truth.py``).
    Built once into ``QUANT_GATE_DEQUANT_DIR`` (default ``$QUANT_GATE_OUT/dequantized``) and
    reused while its source checkpoint and the numerics contract are unchanged."""

    root = SETTINGS.path("DEQUANT_DIR") or out_dir() / "dequantized"
    folder = root / checkpoint.pin.slug
    identity = {
        "source": checkpoint.pin.key,
        "numerics_sha256": sha256_file(NUMERICS_PATH),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    stamp = folder / "dequantized.json"
    if stamp.exists() and json.loads(stamp.read_text(encoding="utf-8")) == identity:
        return folder
    if folder.exists():
        shutil.rmtree(folder)
    building = folder.with_name(folder.name + ".partial")
    if building.exists():
        shutil.rmtree(building)
    write_dequantized(checkpoint.pin.snapshot(), building)
    (building / "dequantized.json").write_text(json.dumps(identity, indent=1) + "\n")
    building.rename(folder)
    return folder


def drop_dequantized(checkpoint: QuantCheckpoint) -> None:
    """Delete the dequantized folder (``QUANT_GATE_DROP_DEQUANT``): the truths' checks read
    its config's identity from the checkpoint (``dequantized_config``), not from it."""

    root = SETTINGS.path("DEQUANT_DIR") or out_dir() / "dequantized"
    shutil.rmtree(root / checkpoint.pin.slug, ignore_errors=True)


def dequantized_config(snapshot: Path) -> str:
    """The dequantized folder's ``config.json``: the checkpoint's, without its quantization
    config."""

    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config.pop("quantization_config", None)
    config.get("text_config", {}).pop("quantization_config", None)
    return json.dumps(config, indent=1) + "\n"


def write_dequantized(snapshot: Path, folder: Path) -> None:
    """Write ``folder``: every tensor of ``snapshot`` but the MTP head's, each quantized weight
    replaced by its FP32 dequantization and its scales dropped; the config without its
    quantization config; the tokenizer and generation files as they are."""

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    contract = load_numerics()
    folder.mkdir(parents=True)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    files = sorted(set(index["weight_map"].values()))
    where = {name: snapshot / file for name, file in index["weight_map"].items()}

    def tensor(name: str) -> torch.Tensor:
        with safe_open(where[name], framework="pt") as handle:
            return handle.get_tensor(name)

    formats: dict[str, dict[str, Any]] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_bytes, shards, weight_map = 0, [], {}

    def flush() -> None:
        nonlocal shard, shard_bytes
        if shard:
            name = f"model-{len(shards) + 1:05d}.safetensors"
            save_file(shard, folder / name, metadata={"format": "pt"})
            shards.append(name)
            weight_map.update(dict.fromkeys(shard, name))
            shard, shard_bytes = {}, 0

    for file in files:
        with safe_open(snapshot / file, framework="pt") as handle:
            names = sorted(handle.keys())
        for name in names:
            if name.startswith("mtp.") or name.rpartition(".")[2] in SCALE_SUFFIXES:
                continue
            value = tensor(name)
            module = name.removesuffix(".weight")
            if name.endswith(".weight") and value.dtype in (contract.E4M3, torch.uint8):
                value, formats[module] = _dequantize(contract, module, value, where, tensor)
            shard[name] = value.contiguous()
            shard_bytes += value.nbytes
            if shard_bytes >= _SHARD_BYTES:
                flush()
    flush()
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1) + "\n"
    )
    (folder / "activation_quant.json").write_text(
        json.dumps(activation_plan(formats), indent=1, sort_keys=True) + "\n"
    )
    (folder / "config.json").write_text(dequantized_config(snapshot))
    for path in snapshot.iterdir():
        skipped = path.suffix == ".safetensors" or path.name in {
            "config.json",
            "model.safetensors.index.json",
            "hf_quant_config.json",
        }
        if path.is_file() and not skipped:
            shutil.copyfile(path, folder / path.name)


def _dequantize(
    contract: Any, module: str, weight: Any, where: Mapping[str, Path], tensor: Any
) -> tuple[Any, dict[str, Any]]:
    """``W_hat`` (FP32) of one quantized weight by ``contract`` and its activation format."""
    if f"{module}.weight_scale_inv" in where:
        scale = tensor(f"{module}.weight_scale_inv")
        return contract.dequantize_block_weight(weight, scale), {"format": "fp8_block"}
    input_scale = float(tensor(f"{module}.input_scale"))
    if weight.dtype == contract.E4M3:
        dequantized = contract.dequantize_static(weight, tensor(f"{module}.weight_scale"))
        return dequantized, {"format": "fp8_static", "input_scale": input_scale}
    dequantized = contract.dequantize_nvfp4_weight(
        weight, tensor(f"{module}.weight_scale"), tensor(f"{module}.weight_scale_2")
    )
    return dequantized, {"format": "nvfp4", "input_scale": input_scale}


# Siblings that read one input and that the engine merges into one module: they quantize it
# by one input scale, the largest of theirs (``fp8_static.merged_input_scale``).
_MERGED_SIBLINGS = (
    frozenset({"q_proj", "k_proj", "v_proj"}),
    frozenset({"gate_proj", "up_proj"}),
    frozenset({"in_proj_qkv", "in_proj_z"}),
)


def activation_plan(formats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per Transformers module, the activation quantization the engine applies to its input:
    a linear's own, a merged group's largest static scale, and a MoE block's routed experts
    (``.experts``, Transformers' fused module) once for all of them."""

    modules: dict[str, dict[str, Any]] = {}
    experts: dict[str, set[str]] = {}
    for module, spec in formats.items():
        parent = module.rpartition(".")[0]
        owner, _, index = parent.rpartition(".")
        if owner.endswith(".experts") and index.isdigit():
            if spec["format"] != "fp8_block":
                raise NotImplementedError(f"routed experts in {spec['format']}: {module}")
            experts.setdefault(owner, set()).add(spec["format"])
            continue
        modules[module] = dict(spec)
    merged = []
    for module, spec in modules.items():
        parent, _, leaf = module.rpartition(".")
        group = next((group for group in _MERGED_SIBLINGS if leaf in group), None)
        if group is None or "input_scale" not in spec:
            continue
        siblings = [modules.get(f"{parent}.{name}") for name in sorted(group)]
        scales = [s["input_scale"] for s in siblings if s and s["format"] == spec["format"]]
        if max(scales) != spec["input_scale"]:
            merged.append(module)
        spec["input_scale"] = max(scales)
    return {
        "linears": modules,
        "experts": dict.fromkeys(experts, "fp8_block"),
        "merged_input_scales_changed": sorted(merged),
    }


# ==================================== Children ====================================


def code_identity() -> dict[str, str]:
    """The sources a quantized truth depends on beyond ``hf_truth.code_version``."""

    return {
        "numerics_sha256": sha256_file(NUMERICS_PATH),
        "quant_truth_sha256": sha256_file(Path(__file__).with_name("quant_truth.py")),
    }


def run_child(
    module: str, args: Sequence[str], log: Path, *, env: dict[str, str], timeout: float
) -> str:
    """``python -m module args`` from the repository root in its own session; its stdout, or
    the test fails with the ends of its logs."""

    from tests.support.hf_parity import _run_child

    log.parent.mkdir(parents=True, exist_ok=True)
    return _run_child((sys.executable, "-m", module, *args), log, env=env, timeout=timeout)


def engine_child(
    gate: str, case: GateCase, folder: Path, tmp_path: Path, **env: str
) -> dict[str, Any]:
    """Run ``quant_gate_child``'s ``gate`` on ``case``; its output and receipt."""

    prompts, key = prompt_set(case.checkpoint)
    from tests.support.gpu import assert_cuda_device_metadata, child_receipt, offline_child_env
    from tests.support.quant_gate_child import RECEIPT_PREFIX

    child_env, logs = offline_child_env(tmp_path, f"quant_{gate}")
    topology = case.topology
    output = folder / "child.json"
    stdout = run_child(
        "tests.support.quant_gate_child",
        (
            gate,
            "--snapshot",
            str(case.checkpoint.pin.snapshot()),
            "--model-key",
            key,
            "--prompts",
            str(prompts),
            "--output",
            str(output),
            "--ranks",
            str(topology.ranks),
            *(() if topology.expert_parallel else ("--expert-tensor-parallel",)),
        ),
        logs / f"{gate}.log",
        env={**child_env, **env},
        timeout=_ENGINE_TIMEOUT_SECONDS,
    )
    receipt = child_receipt(stdout, RECEIPT_PREFIX)
    assert_cuda_device_metadata(receipt)
    assert Path(receipt["engine_root"]).is_relative_to(PYTHON_ROOT)
    return {"receipt": receipt, **json.loads(output.read_text(encoding="utf-8"))}


_ENGINE_TIMEOUT_SECONDS = 1700
_FLOOR_TIMEOUT_SECONDS = 1200
_BENCHMARK_TIMEOUT_SECONDS = 1700


# ============================== Gate 1: graph == eager ==============================


def graph_eager(case: GateCase, tmp_path: Path) -> dict[str, Any]:
    """Both engines' greedy outputs at B = 1..4: computed once per case, then reused by the
    test of each B."""

    folder = out_dir() / "graph_eager" / case.id
    output = folder / "child.json"
    if not output.exists():
        require_ranks(case.topology)
        folder.mkdir(parents=True, exist_ok=True)
        engine_child("graph-eager", case, folder, tmp_path)
    return json.loads(output.read_text(encoding="utf-8"))


def compare_outputs(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Where two decodes of the same rows first differ in token id and in logprob."""

    def first(key: str) -> list[int | None]:
        return [
            next((step for step, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)
            for a, b in zip(left[key], right[key], strict=True)
        ]

    ids, logprobs = first("ids"), first("logprobs")
    return {
        "ids_equal": left["ids"] == right["ids"],
        "logprobs_equal": left["logprobs"] == right["logprobs"],
        "first_id_difference": ids,
        "first_logprob_difference": logprobs,
    }


def graph_eager_judgement(graph: Mapping[str, Any], eager: Mapping[str, Any]) -> dict[str, Any]:
    """Graph against eager decoding within the accuracy tolerance: per row the same tokens up
    to a first divergence, if any, at a near tie (the two picks' logprobs within
    ``NEAR_TIE_NATS``), and on the shared steps logprobs within ``GRAPH_LOGPROB_TOLERANCE``."""

    from tests.support.hf_judge import GRAPH_LOGPROB_TOLERANCE, NEAR_TIE_NATS

    rows = []
    for ids_g, ids_e, lp_g, lp_e in zip(
        graph["ids"], eager["ids"], graph["logprobs"], eager["logprobs"], strict=True
    ):
        diverged = next(
            (i for i, (x, y) in enumerate(zip(ids_g, ids_e, strict=True)) if x != y), None
        )
        shared = len(ids_g) if diverged is None else diverged
        error = max(
            (abs(x - y) for x, y in zip(lp_g[:shared], lp_e[:shared], strict=True)), default=0.0
        )
        tie = None if diverged is None else abs(lp_g[diverged] - lp_e[diverged])
        rows.append(
            {
                "diverged_at": diverged,
                "divergence_gap_nats": tie,
                "max_logprob_error": error,
                "passed": error <= GRAPH_LOGPROB_TOLERANCE
                and (tie is None or tie <= NEAR_TIE_NATS),
            }
        )
    return {
        "bounds": {
            "graph_logprob_tolerance": GRAPH_LOGPROB_TOLERANCE,
            "near_tie_nats": NEAR_TIE_NATS,
        },
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }


# ========================= Gate 2: against the FP32 truths =========================


class Truths:
    """The truths of a checkpoint, as gate 2's segments generate and read them.

    ``quantized``: the quantized checkpoint's (``quant_truth`` on its dequantized folder); its
    identity is the dequantized config and ``quant_truth.json`` (source checkpoint, numerics
    contract, hooks). ``floor``: the same in BF16 on one GPU, forced along the quantized truth,
    kept beside it; G-C's calibration. Once it is written, ``QUANT_GATE_DROP_DEQUANT=1`` deletes
    the dequantized folder. ``bf16``: the BF16 sibling's, a parity check's ``truth/`` as well; its
    identity is the sibling's own (model key, config, prompts, code), checked by
    ``hf_truth.check_truth`` as the parity checks' ``REUSE_TRUTH`` does. Each is read from
    ``QUANT_GATE_REUSE_QUANT_TRUTH`` / ``QUANT_GATE_REUSE_BF16_TRUTH`` if set, else from
    ``$QUANT_GATE_OUT/truths/<repo>/``, where its generation segment writes it."""

    def __init__(self, checkpoint: QuantCheckpoint) -> None:
        self.checkpoint = checkpoint
        self.prompts, self.key = prompt_set(checkpoint)
        self.greedy_tokens = SETTINGS.positive_int("GREEDY_TOKENS", 256)

    def folder(self, kind: str) -> Path:
        if kind == "floor":
            return self.folder("quantized").with_name("floor")
        variable = {"quantized": "REUSE_QUANT_TRUTH", "bf16": "REUSE_BF16_TRUTH"}[kind]
        pin = self.checkpoint.pin if kind == "quantized" else self.checkpoint.sibling
        return SETTINGS.path(variable) or out_dir() / "truths" / pin.slug / kind

    def source(self, kind: str) -> Path:
        """The folder the truth's Transformers class loads (built if missing)."""

        if kind in ("quantized", "floor"):
            return dequantized_folder(self.checkpoint)
        return self.checkpoint.sibling.snapshot()

    def config_sha256(self, kind: str) -> str:
        """The ``config.json`` digest of ``source(kind)``, without building it."""

        if kind == "bf16":
            return sha256_file(self.checkpoint.sibling.snapshot() / "config.json")
        text = dequantized_config(self.checkpoint.pin.snapshot())
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def generate(self, kind: str, tmp_path: Path) -> dict[str, Any]:
        """Run the truth child into ``folder(kind)``, unless a finished truth is there."""

        from tests.support.gpu import offline_child_env
        from tests.support.hf_parity import TOP_K

        folder = self.folder(kind)
        if not (folder / "manifest.json").exists():
            if folder.exists() and any(folder.iterdir()):
                pytest.fail(f"{folder} holds an unfinished truth: remove it or name another")
            env, logs = offline_child_env(tmp_path, f"quant_truth_{kind}")
            if kind == "floor":
                # The quantized truth in BF16 on one GPU, forced along the FP32 truth.
                self.check("quantized")
                options = ["--dtype", "bfloat16", "--truth", str(self.folder("quantized"))]
                timeout = _FLOOR_TIMEOUT_SECONDS
            else:
                options = ["--dtype", "float32", "--device-map", "auto"]
                options += ["--gpus", str(SETTINGS.positive_int("TRUTH_GPUS", 1))]
                if (gib := SETTINGS.positive_float("TRUTH_GPU_GIB")) is not None:
                    options += ["--max-gpu-gib", str(gib)]
                timeout = SETTINGS.positive_int("TRUTH_TIMEOUT", 1700)
            run_child(
                "tests.support.hf_truth" if kind == "bf16" else "tests.support.quant_truth",
                (
                    "--checkpoint",
                    str(self.source(kind)),
                    "--model-key",
                    self.key,
                    "--prompts",
                    str(self.prompts),
                    "--greedy-tokens",
                    str(self.greedy_tokens),
                    "--top-k",
                    str(TOP_K),
                    "--output",
                    str(folder),
                    *options,
                ),
                logs / f"{kind}.log",
                env=env,
                timeout=timeout,
            )
        manifest = self.check(kind)
        if kind == "floor" and SETTINGS.flag("DROP_DEQUANT", False):
            drop_dequantized(self.checkpoint)
        return manifest

    def check(self, kind: str) -> dict[str, Any]:
        """``folder(kind)`` holds this checkpoint's finished truth; its manifest."""

        from tests.support.hf_parity import TOP_K
        from tests.support.hf_truth import IEEE_FP32_PRECISION, check_truth, code_version

        folder = self.folder(kind)
        if not (folder / "manifest.json").exists():
            pytest.fail(f"no {kind} truth in {folder}: run its generation segment first")
        dtype = "bfloat16" if kind == "floor" else "float32"
        check_truth(
            folder,
            {
                "model_key": self.key,
                "config_sha256": self.config_sha256(kind),
                "prompts_sha256": sha256_file(self.prompts),
                "greedy_tokens": self.greedy_tokens,
                "top_k": TOP_K,
                "dtype": dtype,
                "mode": "whole",
                "code_version": code_version(),
                "prompt_names": [name for name, _ in prompt_ids(self.prompts, self.key)],
            },
        )
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if kind != "floor":
            assert manifest["fp32_precision"] == IEEE_FP32_PRECISION, folder
        assert manifest["tokenizer_check"] == "matched", folder
        assert all(value == "torch" for value in manifest["kernel_bindings"].values()), folder
        placed = manifest["parameter_dtypes"]
        assert all(dtypes == [dtype] for dtypes in placed.values()), (folder, placed)
        if kind != "bf16":
            record = json.loads((folder / "quant_truth.json").read_text(encoding="utf-8"))
            identity = {"quantized_checkpoint": self.checkpoint.pin.key, **code_identity()}
            if {name: record.get(name) for name in identity} != identity:
                pytest.fail(f"{folder} is another checkpoint's or ran other numerics or hooks")
        return manifest


def dequantize(case: GateCase) -> dict[str, Any]:
    """Gate 2's first segment, on the CPU: the dequantized FP32 folder."""

    folder = dequantized_folder(case.checkpoint)
    plan = json.loads((folder / "activation_quant.json").read_text(encoding="utf-8"))
    return {
        "case": case.id,
        "checkpoint": case.checkpoint.pin.key,
        "folder": str(folder),
        "quantized_linears": len(plan["linears"]),
        "quantized_experts": len(plan["experts"]),
        "merged_input_scales_changed": plan["merged_input_scales_changed"],
    }


def generate_truth(case: GateCase, kind: str, tmp_path: Path) -> dict[str, Any]:
    """Gate 2's truth segment: the ``quantized`` or ``bf16`` FP32 truth, unless kept."""

    truths = Truths(case.checkpoint)
    manifest = truths.generate(kind, tmp_path)
    return {
        "case": case.id,
        "kind": kind,
        "folder": str(truths.folder(kind)),
        "model_key": manifest["model_key"],
        "seconds": manifest["seconds"],
        "gpus": manifest["gpus"],
    }


def _teacher_forced_engine(
    case: GateCase, truths: Truths, kind: str, folder: Path, tmp_path: Path
) -> tuple[Path, tuple[str, ...]]:
    """The engine child of ``hf_parity`` forced along the ``kind`` truth's continuations:
    cold, cold again at a smaller prefill chunk, and for a MoE checkpoint through prefix
    hits; the schedulings that mark each row stable or not (``hf_parity.unstable_rows``)."""

    from tests.support.gpu import offline_child_env

    env, logs = offline_child_env(tmp_path, f"quant_engine_{kind}")
    topology = case.topology
    paths = ("cold", "cold_chunked", *(("prefix_hit",) if case.checkpoint.moe else ()))
    output = folder / "engine"
    run_child(
        "tests.support.hf_parity",
        (
            "--snapshot",
            str(case.checkpoint.pin.snapshot()),
            "--model-key",
            truths.key,
            "--prompts",
            str(truths.prompts),
            "--greedy-tokens",
            str(truths.greedy_tokens),
            "--truth",
            str(truths.folder(kind)),
            "--output",
            str(output),
            "--max-running-req",
            str(SETTINGS.positive_int("MAX_RUNNING_REQ", 16)),
            "--tensor-parallel-size",
            str(topology.ranks),
            *(() if topology.expert_parallel else ("--expert-tensor-parallel",)),
            "--paths",
            *paths,
        ),
        logs / "engine.log",
        env=env,
        timeout=_ENGINE_TIMEOUT_SECONDS,
    )
    return output, paths


def quantized_fidelity(case: GateCase, tmp_path: Path) -> dict[str, Any]:
    """G-C: the engine teacher-forced against the FP32 quantized truth, within the
    bounds of ``hf_judge`` calibrated by the same truth in BF16 (the floor truth), under the dense
    or MoE (cold and prefix hits, one near-tie prompt excused per path) parity rules. Rows
    whose top-1 the engine's own schedulings disagree on are reported and not counted; a prompt
    with more than ``UNSTABLE_SHARE_CEILING`` of them fails."""

    from safetensors.torch import load_file

    from tests.support.hf_judge import BOUNDS, NEAR_TIE_NATS
    from tests.support.hf_parity import (
        MOE_EXCEPTIONS,
        UNSTABLE_SHARE_CEILING,
        _first_eos_row,
        _top1,
        path_failures,
        teacher_forced,
        top1_gates,
        unstable_rows,
    )

    require_ranks(case.topology)
    truths = Truths(case.checkpoint)
    truths.check("quantized")
    truths.check("floor")
    truth, floor = truths.folder("quantized"), truths.folder("floor")
    folder = run_folder("quantized_fidelity", case.id)
    engine, paths = _teacher_forced_engine(case, truths, "quantized", folder, tmp_path)
    eos_ids = _eos_ids(case.checkpoint.pin.snapshot())
    ceiling = case.checkpoint.floor_disagreement_share
    share = {} if ceiling is None else {"bf16_max_disagreement_share": ceiling}
    comparisons = []
    for name, _ in prompt_ids(truths.prompts, truths.key):
        quant = load_file(truth / f"{name}.safetensors")
        eos_row = _first_eos_row(quant, eos_ids)
        floor_dump = load_file(floor / f"{name}.safetensors")
        dumps = {path: load_file(engine / path / f"{name}.safetensors") for path in paths}
        # Rows the engine's own schedulings disagree on are reported, never counted.
        stable = ~unstable_rows([dump["top1_ids"] for dump in dumps.values()])
        bf16 = teacher_forced(quant, *_top1(floor_dump), eos_row=eos_row, stable=stable)
        comparison: dict[str, Any] = {"name": name, "floor": bf16, "paths": {}}
        for path, dump in dumps.items():
            forced = teacher_forced(
                quant, dump["top1_ids"], dump["top1_logprobs"], eos_row=eos_row, stable=stable
            )
            comparison["paths"][path] = {
                "engine": forced,
                "gates": top1_gates(bf16, forced, **share),
            }
        comparisons.append(comparison)
    failed = {path: path_failures(comparisons, path, moe=case.checkpoint.moe) for path in paths}
    bounds = {
        "near_tie_nats": NEAR_TIE_NATS,
        "moe_exceptions": MOE_EXCEPTIONS,
        "unstable_share": UNSTABLE_SHARE_CEILING,
        **BOUNDS,
        **share,
    }
    return {
        "case": case.id,
        "checkpoint": case.checkpoint.pin.key,
        "truth": str(truth),
        "bounds": bounds,
        "comparisons": comparisons,
        "failed": {path: prompts for path, prompts in failed.items() if prompts},
    }


def bf16_fidelity(case: GateCase, tmp_path: Path) -> dict[str, Any]:
    """G-D, report only until its bounds are frozen: the engine teacher-forced
    along the BF16 sibling's FP32 truth; per path, top-1 agreement and the chosen-token
    logprob error where the top-1 agrees (the truth's target token)."""

    from safetensors.torch import load_file

    from tests.support.hf_parity import teacher_forced

    require_ranks(case.topology)
    truths = Truths(case.checkpoint)
    truths.check("bf16")
    truth = truths.folder("bf16")
    folder = run_folder("bf16_fidelity", case.id)
    engine, paths = _teacher_forced_engine(case, truths, "bf16", folder, tmp_path)
    pooled = {path: _Pooled() for path in paths}
    prompts = {}
    for name, _ in prompt_ids(truths.prompts, truths.key):
        reference = load_file(truth / f"{name}.safetensors")
        prompts[name] = {}
        for path in paths:
            dump = load_file(engine / path / f"{name}.safetensors")
            ids, logprobs = dump["top1_ids"], dump["top1_logprobs"]
            prompts[name][path] = teacher_forced(reference, ids, logprobs)
            pooled[path].add(reference, ids, logprobs)
    return {
        "case": case.id,
        "checkpoint": case.checkpoint.pin.key,
        "truth_of": case.checkpoint.sibling.key,
        "truth": str(truth),
        "report_only": "bounds not frozen yet",
        "pooled": {path: summary.summary() for path, summary in pooled.items()},
        "prompts": prompts,
    }


class _Pooled:
    """G-D's report over every prompt's teacher-forced rows: top-1 agreement with the BF16
    truth and the chosen-token logprob error where the top-1 agrees (the truth's target)."""

    def __init__(self) -> None:
        self.rows = self.agree = 0
        self.errors: list[Any] = []

    def add(self, truth: Mapping[str, Any], ids: Any, logprobs: Any) -> None:
        from tests.support.hf_parity import _top1

        truth_ids, truth_logprobs = _top1(truth)
        agrees = ids == truth_ids
        self.rows += len(ids)
        self.agree += int(agrees.sum())
        self.errors.append((logprobs.float() - truth_logprobs)[agrees].abs())

    def summary(self) -> dict[str, float | int]:
        import torch

        errors = torch.cat(self.errors)
        return {
            "rows": self.rows,
            "top1_agreement": self.agree / self.rows,
            "target_logprob_mae": float(errors.mean()),
            "target_logprob_error_p99": float(errors.quantile(0.99)),
            "target_logprob_error_max": float(errors.max()),
        }


def _eos_ids(snapshot: Path) -> frozenset[int]:
    generation = json.loads((snapshot / "generation_config.json").read_text(encoding="utf-8"))
    eos = generation.get("eos_token_id", [])
    return frozenset(eos if isinstance(eos, list) else [eos])


# ============================ Gate 3: TPOT / TTFT vs BF16 ============================

# Prompt and output tokens of each measured batch, B = 1..4.
PERF_INPUT_TOKENS = 512
PERF_OUTPUT_TOKENS = 128
PERF_BATCHES = (1, 2, 3, 4)


def performance(case: GateCase, role: str, tmp_path: Path) -> dict[str, Any]:
    """One segment of gate 3: ``benchmark.inference.offline`` on the quantized checkpoint
    (``role`` "quantized") or its BF16 sibling ("bf16") at B = 1..4, the TTFT and TPOT medians
    per B. Once both roles' results are in ``$QUANT_GATE_OUT/performance``, the second also
    reports the speedup BF16 / quantized. A sibling whose weights do not fit the visible GPUs
    is recorded as such, not run."""

    from tests.support.gpu import offline_child_env

    require_ranks(case.topology)
    pin = case.checkpoint.pin if role == "quantized" else case.checkpoint.sibling
    result: dict[str, Any] = {"case": case.id, "role": role, "model": pin.key}
    if role == "bf16" and not _fits(pin, case.topology.ranks):
        return {**result, "skipped": "the BF16 weights do not fit the visible GPUs"}
    folder = run_folder("performance", f"{case.id}-{role}")
    env, logs = offline_child_env(tmp_path, f"quant_performance_{role}")
    runs = SETTINGS.positive_int("PERF_RUNS", 3)
    cases = [f"{PERF_INPUT_TOKENS}x{b}x{PERF_OUTPUT_TOKENS}" for b in PERF_BATCHES]
    run_child(
        "benchmark.inference.offline",
        (
            "--output-dir",
            str(folder / "benchmark"),
            *(argument for raw in cases for argument in ("--case", raw)),
            "--runs",
            str(runs),
            "--model",
            pin.repo_id,
            "--revision",
            pin.revision,
            "--tensor-parallel-size",
            str(case.topology.ranks),
            "--expert-parallel" if case.topology.expert_parallel else "--no-expert-parallel",
            *(("--allow-dirty",) if SETTINGS.flag("ALLOW_DIRTY", False) else ()),
        ),
        logs / "benchmark.log",
        env=env,
        timeout=_BENCHMARK_TIMEOUT_SECONDS,
    )
    receipt = folder / "benchmark" / "receipt.json"
    summary = json.loads(receipt.read_text(encoding="utf-8"))["summary"]
    medians = {
        str(b): {
            metric: summary[f"i{PERF_INPUT_TOKENS}-c{b}-o{PERF_OUTPUT_TOKENS}"][metric]["p50"]
            for metric in ("ttft_ms", "tpot_ms")
        }
        for b in PERF_BATCHES
    }
    result.update(
        workload={"input_tokens": PERF_INPUT_TOKENS, "output_tokens": PERF_OUTPUT_TOKENS},
        runs=runs,
        receipt=str(receipt),
        medians_ms=medians,
    )
    other = (
        out_dir()
        / "performance"
        / f"{case.id}-{'bf16' if role == 'quantized' else 'quantized'}.json"
    )
    if other.exists() and "medians_ms" in (kept := json.loads(other.read_text(encoding="utf-8"))):
        quantized, bf16 = (
            (medians, kept["medians_ms"]) if role == "quantized" else (kept["medians_ms"], medians)
        )
        result["speedup_bf16_over_quantized"] = {
            b: {metric: bf16[b][metric] / quantized[b][metric] for metric in quantized[b]}
            for b in quantized
        }
    return result


def _fits(pin: Checkpoint, ranks: int) -> bool:
    """Whether ``pin``'s weights take at most 90% of the first ``ranks`` GPUs' memory."""

    import torch

    index = json.loads((pin.snapshot() / "model.safetensors.index.json").read_text())
    capacity = sum(torch.cuda.get_device_properties(i).total_memory for i in range(ranks))
    return index["metadata"]["total_size"] <= 0.9 * capacity


# ===================== Gates 4 and 5: switching and LoRA-on-quant =====================


def switching(case: GateCase, tmp_path: Path) -> dict[str, Any]:
    """Serve, then enter and leave three training sessions that discard their step, serving
    after each; the rank probe's tensors at every switch."""

    require_ranks(case.topology)
    folder = run_folder("switching", case.id)
    probe = folder / "probe"
    probe.mkdir()
    from tests.support.quant_gate_child import PROBE_VARIABLE

    child = engine_child("switching", case, folder, tmp_path, **{PROBE_VARIABLE: str(probe)})
    ranks = {}
    for path in sorted(probe.glob("rank-*.jsonl")):
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        first = events[0]["tensors"] if events else {}
        ranks[path.stem] = {
            "events": [f"{event['event']}:{event['mode']}" for event in events],
            "tensors": len(first),
            "quantized_bytes_checksummed": bool(first),
            "moved_or_changed": sorted(
                {
                    name
                    for event in events[1:]
                    for name, entry in event["tensors"].items()
                    if first.get(name) != entry
                }
                | {name for event in events[1:] for name in set(first) ^ set(event["tensors"])}
            ),
        }
    phases = child["phases"]
    return {
        "case": case.id,
        "checkpoint": case.checkpoint.pin.key,
        "receipt": child["receipt"],
        "ranks": ranks,
        "outputs_after_each_switch": [compare_outputs(phases[0], phase) for phase in phases[1:]],
        "loss_sums": child["loss_sums"],
        "model_revision": child["model_revision"],
    }


def lora(case: GateCase, tmp_path: Path) -> dict[str, Any]:
    require_ranks(case.topology)
    folder = run_folder("lora", case.id)
    child = engine_child("lora", case, folder, tmp_path)
    return {
        "case": case.id,
        "checkpoint": case.checkpoint.pin.key,
        "receipt": child["receipt"],
        "zero_adapter_vs_base": compare_outputs(child["base"], child["zero_adapter"]),
        "trained_vs_zero_adapter": compare_outputs(child["zero_adapter"], child["trained"]),
        "loss_sum": child["loss_sum"],
        "grad_norm": child["grad_norm"],
        "step": child["step"],
        "model_revision": child["model_revision"],
    }
