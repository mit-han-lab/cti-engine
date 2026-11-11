"""The Transformers truth child on CPU, on tiny random checkpoints, beside stand-in ``fla`` and
``causal_conv1d`` packages that record any import and raise if called. The CPU auto map offloads
nothing, so GPU budgets and offloaded forwards are tested in this process. The streamed truth
computes the whole model's bits, one layer resident at a time."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)

from tests.support.hf_truth import (
    AUTO_GPU_HEADROOM_GIB,
    IEEE_FP32_PRECISION,
    RECEIPT_PREFIX,
    check_truth,
    code_version,
    load_reference,
    offload_max_memory,
    parameter_bytes,
    parameter_dtypes,
    pin_offloaded_weights,
    run_prompt,
    tokenize_prompt,
)
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5 import TINY_QWEN3_5_CONFIG, randomize_hf_weights
from tests.support.qwen3_5_moe import TINY_QWEN3_5_MOE_CONFIG
from tests.support.qwen3_moe import save_tiny_checkpoint

HELPER_MODULE = "tests.support.hf_truth"
MODEL_KEY = "tiny/qwen3_5@0"
GREEDY_TOKENS = 12
TOP_K = 4
# Two Gated DeltaNet and two full-attention layers of the tiny image-text geometry.
TINY_CONFIG = {
    **TINY_QWEN3_5_CONFIG,
    "text_config": {
        **TINY_QWEN3_5_CONFIG["text_config"],
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "full_attention"] * 2,
        "max_position_embeddings": 256,
    },
}
GATED_DELTA_NET_LAYERS = (0, 2)
# 5 tokens, and 70: longer than one 64-token chunk of Transformers' chunked delta rule.
PROMPT_IDS = {
    "short": [3, 17, 42, 8, 61],
    "long": [(7 * index + 3) % 64 for index in range(70)],
}
_FAKE_KERNEL = """
import os
from pathlib import Path

with Path(os.environ["FAKE_KERNEL_MARKER"]).open("a") as marker:
    marker.write(__name__ + "\\n")


def _called(*args, **kwargs):
    raise RuntimeError(__name__ + " was called")


{names} = (_called,) * {count}
"""
_WORDS = ("<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant", "system", "count", "to")
_WORDS += ("three", "one", "two", "four", "please", "and", "then", "stop")
_CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content }} <|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)
_CHAT_KWARGS = {"add_generation_prompt": True, "enable_thinking": False}
_TOKENIZED_PROMPTS = (
    {
        "name": "chat",
        "kind": "chat",
        "messages": [{"role": "user", "content": "count to three please"}],
    },
    {"name": "text", "kind": "text", "text": "one two four and then stop"},
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("tiny-qwen3_5")
    (folder / "config.json").write_text(json.dumps(TINY_CONFIG))
    config = AutoConfig.from_pretrained(folder)
    model = Qwen3_5ForConditionalGeneration(config).eval()
    randomize_hf_weights(model)
    model.save_pretrained(folder)
    return folder


@pytest.fixture(scope="module")
def fake_kernels(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Importable stand-ins exporting the names Transformers looks up."""

    root = tmp_path_factory.mktemp("fake-kernels")
    for package, names in (
        ("fla", ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")),
        ("causal_conv1d", ("causal_conv1d_fn", "causal_conv1d_update")),
    ):
        (root / package).mkdir()
        (root / package / "__init__.py").write_text(
            _FAKE_KERNEL.format(names=", ".join(names), count=len(names))
        )
    return root


def _prompts_file(folder: Path, prompts: dict[str, list[int]]) -> Path:
    path = folder / "prompts.json"
    frozen = {
        "schema": 1,
        "chat_template_kwargs": _CHAT_KWARGS,
        "models": {MODEL_KEY: {}},
        "prompts": [
            {"name": name, "kind": "text", "text": "", "token_ids": {MODEL_KEY: ids}}
            for name, ids in prompts.items()
        ],
    }
    path.write_text(json.dumps(frozen))
    return path


@pytest.fixture(scope="module")
def truth(
    checkpoint: Path, fake_kernels: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict, Path]:
    """The FP32 truth of the parity check, on the auto device map: its manifest and folder."""

    folder = tmp_path_factory.mktemp("truth")
    marker = folder / "imported"
    output = folder / "dumps"
    manifest = _receipt(
        _run_helper(
            checkpoint,
            _prompts_file(folder, PROMPT_IDS),
            output,
            _child_env(fake_kernels, marker),
            "--dtype",
            "float32",
            "--device-map",
            "auto",
        )
    )
    _assert_blocked(manifest, marker)
    return manifest, output


def _child_env(fake_kernels: Path, marker: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(fake_kernels),
        FAKE_KERNEL_MARKER=str(marker),
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="4",
        USE_HUB_KERNELS="YES",  # the child must turn it off itself
    )
    return env


def _run_helper(
    checkpoint: Path,
    prompts: Path,
    output: Path,
    env: dict[str, str],
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        HELPER_MODULE,
        "--checkpoint",
        str(checkpoint),
        "--model-key",
        MODEL_KEY,
    ]
    command += ["--prompts", str(prompts), "--output", str(output), "--device", "cpu"]
    command += ["--greedy-tokens", str(GREEDY_TOKENS), "--top-k", str(TOP_K), *extra]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=600,
        check=False,
    )


def _receipt(completed: subprocess.CompletedProcess[str]) -> dict:
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith(RECEIPT_PREFIX)]
    assert len(lines) == 1, completed.stdout
    return json.loads(lines[0].removeprefix(RECEIPT_PREFIX))


def _reference(checkpoint: Path, dtype: torch.dtype) -> PreTrainedModel:
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=dtype, attn_implementation="eager"
    ).eval()


def _prefill_states(model: PreTrainedModel, input_ids: list[int]) -> dict[str, torch.Tensor]:
    """Transformers' cached states after the prompt, in the cache's own dtypes."""

    with torch.no_grad():
        cache = model(input_ids=torch.tensor([input_ids]), use_cache=True).past_key_values
    states = {}
    for index in GATED_DELTA_NET_LAYERS:
        states[f"recurrent_state.{index}"] = cache.layers[index].recurrent_states[0][0]
        states[f"conv_state.{index}"] = cache.layers[index].conv_states[0][0]
    return states


def _assert_blocked(manifest: dict, marker: Path) -> None:
    assert manifest["kernel_bindings"] == {
        "causal_conv1d_update": "torch",
        "causal_conv1d_fn": "torch",
        "torch_chunk_gated_delta_rule": "torch",
        "torch_recurrent_gated_delta_rule": "torch",
    }
    assert manifest["blocked_packages"] == {"fla": True, "causal_conv1d": True}
    assert manifest["loaded_blocked_modules"] == []
    assert manifest["use_hub_kernels"] == "NO"
    assert manifest["hub_kernels_enabled"] is False
    assert not marker.exists(), marker.read_text()


def test_child_dumps_greedy_teacher_forced_and_states_on_the_torch_paths(
    checkpoint: Path, fake_kernels: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "imported"
    prompts = _prompts_file(tmp_path, PROMPT_IDS)
    output = tmp_path / "reference"
    manifest = _receipt(
        _run_helper(
            checkpoint, prompts, output, _child_env(fake_kernels, marker), "--dtype", "float32"
        )
    )

    _assert_blocked(manifest, marker)
    assert manifest["tokenizer_check"] == "absent"
    assert manifest["architecture"] == "Qwen3_5ForConditionalGeneration"
    assert manifest["attention"] == "eager"
    assert manifest["device_name"] == "cpu"
    assert (manifest["device_map"], manifest["max_memory"]) == ("device", None)
    assert list(manifest["parameter_bytes"]) == ["cpu"]
    assert (manifest["teacher_forced"], manifest["truth"]) == ("own", None)
    # IEEE FP32 throughout: torch's precision settings, every parameter and every state.
    assert manifest["fp32_precision"] == IEEE_FP32_PRECISION
    assert manifest["parameter_dtypes"] == {"cpu": ["float32"]}
    assert manifest["pinned_offloaded_bytes"] == 0
    assert manifest["state_dtypes"] == {"recurrent_state": ["float32"], "conv_state": ["float32"]}
    assert manifest["code_version"] == code_version()
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert [entry["name"] for entry in manifest["prompts"]] == list(PROMPT_IDS)

    model = _reference(checkpoint, torch.float32)
    for name, input_ids in PROMPT_IDS.items():
        dump = load_file(output / f"{name}.safetensors")
        num_prompt, num_positions = len(input_ids), len(input_ids) + GREEDY_TOKENS
        assert dump["input_ids"].tolist() == input_ids
        assert dump["greedy_ids"].shape == dump["greedy_margins"].shape == (GREEDY_TOKENS,)
        assert dump["topk_ids"].shape == dump["topk_logits"].shape == (num_positions, TOP_K)
        assert dump["logsumexp"].shape == (num_positions,)
        assert dump["prefill_logits"].shape == (64,)
        assert torch.equal(dump["teacher_forced_ids"], dump["greedy_ids"])
        assert "at_truth_ids" not in dump

        # Teacher-forced dumps equal this process's forward over the same sequence.
        sequence = input_ids + dump["greedy_ids"].tolist()
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([sequence]), use_cache=False).logits[0].float()
        values, indices = logits.topk(TOP_K, dim=-1)
        assert torch.equal(dump["topk_ids"], indices)
        torch.testing.assert_close(dump["topk_logits"], values, rtol=0, atol=1e-5)
        torch.testing.assert_close(dump["logsumexp"], logits.logsumexp(-1), rtol=0, atol=1e-5)

        # The cached greedy continuation agrees with the full forward wherever it is
        # decided, and its margins are the full forward's top-2 gaps.
        decided = dump["greedy_margins"] > 1e-3
        assert decided.sum() >= GREEDY_TOKENS // 2
        continuation = logits[num_prompt - 1 : num_positions - 1]
        assert torch.equal(dump["greedy_ids"][decided], continuation.argmax(-1)[decided])
        gaps = continuation.topk(2).values
        torch.testing.assert_close(
            dump["greedy_margins"], gaps[:, 0] - gaps[:, 1], rtol=0, atol=1e-4
        )
        torch.testing.assert_close(
            dump["prefill_logits"], logits[num_prompt - 1], rtol=0, atol=1e-4
        )

        # States after the prompt, before the decode that updates them in place.
        expected = _prefill_states(model, input_ids)
        assert set(expected) == {key for key in dump if "_state." in key}
        for key, state in expected.items():
            # The child runs on OMP_NUM_THREADS=4, so CPU reductions may split differently.
            torch.testing.assert_close(dump[key], state.float(), rtol=1e-5, atol=1e-5)
        assert dump["recurrent_state.0"].shape == (2, 128, 128)  # [H, K, V]


def test_child_runs_a_moe_checkpoint_on_the_per_expert_loop(
    fake_kernels: Path, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "tiny-qwen3_5_moe"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(TINY_QWEN3_5_MOE_CONFIG, default=dict))
    model = Qwen3_5MoeForConditionalGeneration(AutoConfig.from_pretrained(checkpoint)).eval()
    randomize_hf_weights(model)
    model.save_pretrained(checkpoint)
    marker = tmp_path / "imported"
    output = tmp_path / "reference"
    input_ids = PROMPT_IDS["long"]
    prompts = _prompts_file(tmp_path, {"long": input_ids})

    manifest = _receipt(
        _run_helper(
            checkpoint, prompts, output, _child_env(fake_kernels, marker), "--dtype", "float32"
        )
    )

    _assert_blocked(manifest, marker)
    assert manifest["architecture"] == "Qwen3_5MoeForConditionalGeneration"
    assert (manifest["attention"], manifest["experts"]) == ("eager", "eager")
    assert manifest["state_dtypes"] == {"recurrent_state": ["float32"], "conv_state": ["float32"]}
    dump = load_file(output / "long.safetensors")
    reference = load_reference(checkpoint, torch.device("cpu"), torch.float32)
    sequence = input_ids + dump["greedy_ids"].tolist()
    with torch.no_grad():
        logits = reference(input_ids=torch.tensor([sequence]), use_cache=False).logits[0]
    values, indices = logits.float().topk(TOP_K, dim=-1)
    assert torch.equal(dump["topk_ids"], indices)
    # The child runs on OMP_NUM_THREADS=4, so CPU reductions may split differently.
    torch.testing.assert_close(dump["topk_logits"], values, rtol=1e-5, atol=1e-5)


def test_auto_device_map_places_the_same_model(
    checkpoint: Path, fake_kernels: Path, truth: tuple[dict, Path], tmp_path: Path
) -> None:
    """On the CPU, ``--device-map auto`` (the truth's placement) puts every weight on the CPU
    and computes exactly what ``--device-map device`` does."""

    manifest, folder = truth
    output = tmp_path / "device"
    placed = _receipt(
        _run_helper(
            checkpoint,
            _prompts_file(tmp_path, PROMPT_IDS),
            output,
            _child_env(fake_kernels, tmp_path / "imported"),
            "--dtype",
            "float32",
        )
    )
    assert (manifest["device_map"], placed["device_map"]) == ("auto", "device")
    assert list(manifest["max_memory"]) == ["cpu"] and manifest["max_memory"]["cpu"] > 0
    assert manifest["parameter_bytes"] == placed["parameter_bytes"]
    assert list(manifest["parameter_bytes"]) == ["cpu"]
    for name in PROMPT_IDS:
        expected = load_file(output / f"{name}.safetensors")
        dump = load_file(folder / f"{name}.safetensors")
        assert dump.keys() == expected.keys()
        for key, tensor in expected.items():
            assert torch.equal(dump[key], tensor), (name, key)


def test_offloaded_weights_compute_what_resident_weights_do(checkpoint: Path) -> None:
    """Accelerate's offload hooks, which the truth's auto map attaches to CPU-RAM layers, give
    ``run_prompt`` the resident weights' dump, following a truth or not."""

    from accelerate import cpu_offload

    cpu = torch.device("cpu")
    resident = load_reference(checkpoint, cpu, torch.float32)
    offloaded = load_reference(checkpoint, cpu, torch.float32)
    cpu_offload(offloaded, execution_device=cpu)
    total = parameter_bytes(resident)["cpu"]
    assert parameter_bytes(offloaded) == {"offloaded": total}
    assert parameter_dtypes(resident) == {"cpu": ["float32"]}
    assert parameter_dtypes(offloaded) == {"offloaded": ["float32"]}
    # Pinning replaces each copy the hooks hold, once, here by a plain copy (no GPU).
    copies: list[torch.Tensor] = []

    def pin(tensor: torch.Tensor) -> torch.Tensor:
        copies.append(tensor.clone())
        return copies[-1]

    assert pin_offloaded_weights(offloaded, pin) == total
    assert len(copies) == len(list(offloaded.parameters()))
    with torch.inference_mode():
        for input_ids in PROMPT_IDS.values():
            expected, _ = run_prompt(
                resident, input_ids, device=cpu, greedy_tokens=GREEDY_TOKENS, top_k=TOP_K
            )
            dump, _ = run_prompt(
                offloaded,
                input_ids,
                device=cpu,
                greedy_tokens=GREEDY_TOKENS,
                top_k=TOP_K,
                truth=expected,
            )
            assert dump.keys() == expected.keys() | {"at_truth_ids"}
            for key, tensor in expected.items():
                assert torch.equal(dump[key], tensor), key
            assert torch.equal(dump["at_truth_ids"], expected["topk_logits"])
    # The copies are what the forward streams in: one in another dtype shows up, beside its
    # FP32 placeholder, although the forward would quietly cast it back.
    assert pin_offloaded_weights(offloaded, lambda tensor: tensor.to(torch.bfloat16)) == total
    assert parameter_dtypes(offloaded) == {"offloaded": ["bfloat16", "float32"]}


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_child_follows_the_truth_continuation(
    checkpoint: Path, fake_kernels: Path, truth: tuple[dict, Path], tmp_path: Path, dtype: str
) -> None:
    """With ``--truth`` the child teacher-forces the truth's continuation and gathers its
    logits at the truth's top-k ids; its own greedy continuation is still its own."""

    truth_manifest, truth_folder = truth
    marker = tmp_path / "imported"
    output = tmp_path / "follower"
    manifest = _receipt(
        _run_helper(
            checkpoint,
            _prompts_file(tmp_path, PROMPT_IDS),
            output,
            _child_env(fake_kernels, marker),
            "--dtype",
            dtype,
            "--truth",
            str(truth_folder),
        )
    )
    _assert_blocked(manifest, marker)
    assert (manifest["dtype"], manifest["device_map"]) == (dtype, "device")
    assert (manifest["teacher_forced"], manifest["truth"]) == ("truth", str(truth_folder))
    assert manifest["prompts_sha256"] == truth_manifest["prompts_sha256"]
    bf16_model = _reference(checkpoint, torch.bfloat16) if dtype == "bfloat16" else None
    for name, input_ids in PROMPT_IDS.items():
        expected = load_file(truth_folder / f"{name}.safetensors")
        dump = load_file(output / f"{name}.safetensors")
        num_positions = len(input_ids) + GREEDY_TOKENS
        assert dump["input_ids"].tolist() == input_ids
        assert torch.equal(dump["teacher_forced_ids"], expected["greedy_ids"])
        assert dump["at_truth_ids"].shape == (num_positions, TOP_K)
        # Where an id is in both top-k lists, its gathered logit is the child's own.
        shared = dump["topk_ids"][:, :, None] == expected["topk_ids"][:, None, :]
        position, own, truth_column = shared.nonzero().unbind(1)
        assert shared.any(dim=(1, 2)).all()
        assert torch.equal(
            dump["topk_logits"][position, own], dump["at_truth_ids"][position, truth_column]
        )
        logprobs = dump["at_truth_ids"] - dump["logsumexp"][:, None]
        truth_logprobs = expected["topk_logits"] - expected["logsumexp"][:, None]
        if dtype == "float32":
            # The truth's own computation, on the truth's own sequence.
            assert torch.equal(dump["greedy_ids"], expected["greedy_ids"])
            assert torch.equal(dump["topk_ids"], expected["topk_ids"])
            torch.testing.assert_close(logprobs, truth_logprobs, rtol=0, atol=1e-5)
        else:
            # This process's BF16 forward over the truth's sequence, gathered the same way.
            sequence = torch.tensor([input_ids + expected["greedy_ids"].tolist()])
            with torch.no_grad():
                logits = bf16_model(input_ids=sequence, use_cache=False).logits[0].float()
            torch.testing.assert_close(
                dump["at_truth_ids"], logits.gather(1, expected["topk_ids"]), rtol=0, atol=1e-5
            )
            error = (logprobs[:, 0] - truth_logprobs[:, 0]).abs()
            assert 0 < error.mean() < 0.1, error


@pytest.mark.parametrize(
    ("prompt_names", "options"),
    [
        (tuple(PROMPT_IDS), ("--greedy-tokens", str(GREEDY_TOKENS + 1))),
        (tuple(PROMPT_IDS), ("--top-k", str(TOP_K + 1))),
        (("long",), ()),
    ],
    ids=["greedy-tokens", "top-k", "prompts"],
)
def test_child_refuses_a_truth_that_ran_something_else(
    checkpoint: Path,
    fake_kernels: Path,
    truth: tuple[dict, Path],
    tmp_path: Path,
    prompt_names: tuple[str, ...],
    options: tuple[str, ...],
) -> None:
    _, truth_folder = truth
    prompts = _prompts_file(tmp_path, {name: PROMPT_IDS[name] for name in prompt_names})
    output = tmp_path / "follower"
    completed = _run_helper(
        checkpoint,
        prompts,
        output,
        _child_env(fake_kernels, tmp_path / "imported"),
        "--truth",
        str(truth_folder),
        *options,
    )
    assert completed.returncode != 0
    assert "ran something else" in completed.stderr, completed.stderr
    assert not output.exists()


_GPU_GIB_ERROR = "--max-gpu-gib takes a positive number, with --device-map auto"
_GPUS_ERROR = "--gpus takes a positive count"


@pytest.mark.parametrize(
    ("options", "error"),
    [
        (("--max-gpu-gib", "4"), _GPU_GIB_ERROR),
        (("--device-map", "auto", "--max-gpu-gib", "0"), _GPU_GIB_ERROR),
        (("--gpus", "2"), _GPUS_ERROR),
        (("--device-map", "auto", "--gpus", "0"), _GPUS_ERROR),
    ],
    ids=["gib-without-auto", "gib-not-positive", "gpus-without-auto", "gpus-not-positive"],
)
def test_gpu_options_need_the_auto_device_map(
    tmp_path: Path, options: tuple[str, ...], error: str
) -> None:
    command = [sys.executable, "-m", HELPER_MODULE, "--checkpoint", str(tmp_path)]
    command += ["--model-key", MODEL_KEY, "--prompts", str(tmp_path / "prompts.json")]
    command += ["--output", str(tmp_path / "out"), *options]
    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=REPO_ROOT, timeout=120, check=False
    )
    assert completed.returncode == 2
    assert error in completed.stderr


def test_fp32_refuses_nvidia_tf32_override(
    checkpoint: Path, fake_kernels: Path, tmp_path: Path
) -> None:
    """The CUDA libraries read ``NVIDIA_TF32_OVERRIDE`` behind torch's settings."""

    output = tmp_path / "reference"
    env = _child_env(fake_kernels, tmp_path / "imported") | {"NVIDIA_TF32_OVERRIDE": "1"}
    prompts = _prompts_file(tmp_path, PROMPT_IDS)
    completed = _run_helper(checkpoint, prompts, output, env, "--dtype", "float32")
    assert completed.returncode != 0
    assert "NVIDIA_TF32_OVERRIDE is set" in completed.stderr, completed.stderr
    assert not output.exists()


def test_check_truth_refuses_another_code_version_and_an_unfinished_run(tmp_path: Path) -> None:
    """What the parity check compares before reusing a kept truth."""

    manifest = {
        "model_key": MODEL_KEY,
        "code_version": code_version(),
        "prompts": [{"name": name} for name in PROMPT_IDS],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    expected = {
        "model_key": MODEL_KEY,
        "code_version": code_version(),
        "prompt_names": list(PROMPT_IDS),
    }
    check_truth(tmp_path, expected)
    for changed in (
        {"code_version": {**code_version(), "helper_sha256": "0" * 64}},
        {"model_key": "tiny/qwen3_5@1"},
        {"prompt_names": ["long"]},
        {"dtype": "float32"},  # a key the manifest lacks
    ):
        with pytest.raises(ValueError, match="ran something else"):
            check_truth(tmp_path, expected | changed)
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="no finished run"):
        check_truth(tmp_path, expected)


def test_offload_max_memory_bounds_each_gpu_and_keeps_the_rest_in_cpu_ram() -> None:
    gib, ram = 1 << 30, 100 << 30
    cpu = {"cpu": int(0.9 * ram)}

    assert (
        offload_max_memory(torch.device("cpu"), None, free_gpu_bytes=[], available_cpu_bytes=ram)
        == cpu
    )
    assert offload_max_memory(
        torch.device("cuda", 1), 84.5, free_gpu_bytes=[0], available_cpu_bytes=ram
    ) == {1: int(84.5 * gib), **cpu}
    # Unset: each GPU's free memory less the headroom; "cuda" is the first device, and the
    # GPUs after it follow.
    assert offload_max_memory(
        torch.device("cuda"), None, free_gpu_bytes=[94 * gib, 90 * gib], available_cpu_bytes=ram
    ) == {0: (94 - AUTO_GPU_HEADROOM_GIB) * gib, 1: (90 - AUTO_GPU_HEADROOM_GIB) * gib, **cpu}
    assert offload_max_memory(
        torch.device("cuda", 2), 60.0, free_gpu_bytes=[0, 0], available_cpu_bytes=ram
    ) == {2: 60 * gib, 3: 60 * gib, **cpu}
    for device, max_gpu_gib, free, error in (
        ("cpu", 4.0, [], "need a CUDA --device"),
        ("cpu", None, [94 * gib], "need a CUDA --device"),
        ("cuda:0", None, [], "no GPU from cuda:0 on"),
        (
            "cuda:0",
            None,
            [94 * gib, AUTO_GPU_HEADROOM_GIB * gib],
            "no GPU memory for weights on cuda:1",
        ),
        ("meta", None, [], "runs on a CUDA device or the CPU"),
    ):
        with pytest.raises(ValueError, match=error):
            offload_max_memory(
                torch.device(device), max_gpu_gib, free_gpu_bytes=free, available_cpu_bytes=ram
            )


def test_child_in_bfloat16_records_the_cache_dtypes(
    checkpoint: Path, fake_kernels: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "imported"
    prompts = _prompts_file(tmp_path, {"long": PROMPT_IDS["long"]})
    output = tmp_path / "reference"
    manifest = _receipt(_run_helper(checkpoint, prompts, output, _child_env(fake_kernels, marker)))

    _assert_blocked(manifest, marker)
    assert manifest["dtype"] == "bfloat16"
    assert manifest["parameter_dtypes"] == {"cpu": ["bfloat16"]}
    dump = load_file(output / "long.safetensors")
    assert all(tensor.isfinite().all() for tensor in dump.values() if tensor.is_floating_point())
    expected = _prefill_states(_reference(checkpoint, torch.bfloat16), PROMPT_IDS["long"])
    for key, state in expected.items():
        torch.testing.assert_close(dump[key], state.float(), rtol=2e-2, atol=2e-2)
    # Transformers keeps the delta-rule state in FP32 and the convolution history in the
    # model's dtype, as the engine's pool does (FP32 state, BF16 history).
    assert {key.split(".")[0]: state.dtype for key, state in expected.items()} == {
        "recurrent_state": torch.float32,
        "conv_state": torch.bfloat16,
    }
    assert manifest["state_dtypes"] == {"recurrent_state": ["float32"], "conv_state": ["bfloat16"]}


def test_without_the_block_transformers_binds_the_stand_ins(
    fake_kernels: Path, tmp_path: Path
) -> None:
    """The negative control: the stand-ins are what Transformers binds when not blocked,
    and kernel_bindings reports it."""

    marker = tmp_path / "imported"
    script = (
        "import json\n"
        f"from {HELPER_MODULE} import ARCHITECTURES, kernel_bindings\n"
        "from transformers.models.qwen3_5 import modeling_qwen3_5\n"
        "names = ARCHITECTURES['Qwen3_5ForConditionalGeneration'].kernel_bound\n"
        "print(json.dumps(kernel_bindings(modeling_qwen3_5, names)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_child_env(fake_kernels, marker) | {"USE_HUB_KERNELS": "NO"},
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "causal_conv1d_update": "causal_conv1d._called",
        "causal_conv1d_fn": "causal_conv1d._called",
        "torch_chunk_gated_delta_rule": "fla._called",
        "torch_recurrent_gated_delta_rule": "fla._called",
    }
    assert sorted(marker.read_text().split()) == ["causal_conv1d", "fla"]


@pytest.mark.parametrize(
    ("before", "error"),
    [
        ("", None),
        ("import transformers", "transformers is already imported"),
        ("import fla", "fla is already imported"),
    ],
    ids=["blocks", "after-transformers", "after-fla"],
)
def test_block_raises_on_every_import_and_refuses_to_start_late(
    fake_kernels: Path, tmp_path: Path, before: str, error: str | None
) -> None:
    script = (
        "import importlib\n"
        f"{before}\n"
        f"from {HELPER_MODULE} import block_optional_kernels\n"
        "try:\n"
        "    block_optional_kernels()\n"
        "except RuntimeError as refused:\n"
        "    print('refused:', refused); raise SystemExit(0)\n"
        "for name in ('fla', 'fla.ops.gated_delta_rule', 'causal_conv1d'):\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ModuleNotFoundError:\n"
        "        print('blocked:', name)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_child_env(fake_kernels, tmp_path / "imported"),
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    if error is None:
        assert completed.stdout.split("\n")[:3] == [
            "blocked: fla",
            "blocked: fla.ops.gated_delta_rule",
            "blocked: causal_conv1d",
        ]
        assert not (tmp_path / "imported").exists()
    else:
        assert completed.stdout.startswith(f"refused: {error}")


def test_tokenizer_must_re_derive_the_frozen_ids(
    checkpoint: Path, fake_kernels: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "with-tokenizer"
    folder.mkdir()
    for path in checkpoint.iterdir():
        (folder / path.name).symlink_to(path)
    backend = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(_WORDS)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", additional_special_tokens=list(_WORDS[1:3])
    )
    tokenizer.chat_template = _CHAT_TEMPLATE
    tokenizer.save_pretrained(folder)
    frozen = {
        "schema": 1,
        "chat_template_kwargs": _CHAT_KWARGS,
        "models": {MODEL_KEY: {}},
        "prompts": [
            {**prompt, "token_ids": {MODEL_KEY: tokenize_prompt(tokenizer, prompt, _CHAT_KWARGS)}}
            for prompt in _TOKENIZED_PROMPTS
        ],
    }
    assert frozen["prompts"][0]["token_ids"][MODEL_KEY] == [1, 3, 6, 7, 8, 12, 2, 1, 4]
    env = _child_env(fake_kernels, tmp_path / "imported")

    matched = tmp_path / "matched.json"
    matched.write_text(json.dumps(frozen))
    manifest = _receipt(
        _run_helper(folder, matched, tmp_path / "out-matched", env, "--dtype", "float32")
    )
    assert manifest["tokenizer_check"] == "matched"

    stale = copy.deepcopy(frozen)
    stale["prompts"][1]["token_ids"][MODEL_KEY][0] = 10
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale))
    completed = _run_helper(folder, stale_path, tmp_path / "out-stale", env)
    assert completed.returncode != 0
    assert "prompt 'text'" in completed.stderr and "tokenizes it differently" in completed.stderr
    assert not (tmp_path / "out-stale").exists()


# Longer than one 64-row block of the query-blocked attention test, and a short one.
QWEN3_MOE_PROMPT_IDS = {
    "short": [3, 17, 42, 8, 61],
    "long": [(5 * index + 1) % 64 for index in range(70)],
}


@pytest.fixture(scope="module")
def qwen3_moe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("tiny-qwen3_moe")
    save_tiny_checkpoint(folder)
    return folder


@pytest.fixture(scope="module")
def qwen3_moe_truth(
    qwen3_moe: Path, fake_kernels: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict, Path]:
    """The whole model's FP32 truth of the tiny Qwen3-MoE: its manifest and folder."""

    folder = tmp_path_factory.mktemp("qwen3_moe-truth")
    output = folder / "dumps"
    manifest = _receipt(
        _run_helper(
            qwen3_moe,
            _prompts_file(folder, QWEN3_MOE_PROMPT_IDS),
            output,
            _child_env(fake_kernels, folder / "imported"),
            "--dtype",
            "float32",
        )
    )
    return manifest, output


def _streamed(
    qwen3_moe: Path, truth: Path, fake_kernels: Path, folder: Path, *options: str
) -> tuple[dict, Path]:
    output = folder / "streamed"
    manifest = _receipt(
        _run_helper(
            qwen3_moe,
            _prompts_file(folder, QWEN3_MOE_PROMPT_IDS),
            output,
            _child_env(fake_kernels, folder / "imported"),
            "--dtype",
            "float32",
            "--mode",
            "stream",
            "--continuation",
            str(truth),
            *options,
        )
    )
    return manifest, output


def test_child_runs_qwen3_moe_on_the_per_expert_loop_without_states(
    qwen3_moe: Path, qwen3_moe_truth: tuple[dict, Path]
) -> None:
    manifest, output = qwen3_moe_truth

    assert manifest["architecture"] == "Qwen3MoeForCausalLM"
    assert (manifest["mode"], manifest["attention"], manifest["experts"]) == (
        "whole",
        "eager",
        "eager",
    )
    # No Gated DeltaNet kernel to bind and no recurrent state to cache.
    assert (manifest["kernel_bindings"], manifest["state_dtypes"]) == ({}, {})
    assert manifest["blocked_packages"] == {"fla": True, "causal_conv1d": True}
    assert manifest["loaded_blocked_modules"] == []
    assert manifest["fp32_precision"] == IEEE_FP32_PRECISION
    assert manifest["parameter_dtypes"] == {"cpu": ["float32"]}
    reference = load_reference(qwen3_moe, torch.device("cpu"), torch.float32)
    for name, input_ids in QWEN3_MOE_PROMPT_IDS.items():
        dump = load_file(output / f"{name}.safetensors")
        assert not any("_state." in key for key in dump)
        sequence = input_ids + dump["greedy_ids"].tolist()
        with torch.no_grad():
            logits = reference(input_ids=torch.tensor([sequence]), use_cache=False).logits[0]
        values, indices = logits.topk(TOP_K, dim=-1)
        assert torch.equal(dump["topk_ids"], indices)
        # The child runs on OMP_NUM_THREADS=4, so CPU reductions may split differently.
        torch.testing.assert_close(dump["topk_logits"], values, rtol=1e-5, atol=1e-5)


def test_the_streamed_truth_computes_the_whole_models_bits(
    qwen3_moe: Path, fake_kernels: Path, qwen3_moe_truth: tuple[dict, Path], tmp_path: Path
) -> None:
    """One layer resident at a time, on the whole run's continuations, every teacher-forced
    row comes out the same bits; each weight loads once."""

    _, truth = qwen3_moe_truth
    manifest, output = _streamed(qwen3_moe, truth, fake_kernels, tmp_path)

    assert (manifest["mode"], manifest["teacher_forced"]) == ("stream", "continuation")
    assert (manifest["continuation"], manifest["truth"]) == (str(truth), None)
    assert (manifest["attention"], manifest["attention_chunk"]) == ("eager", None)
    assert manifest["parameter_dtypes"] == {"streamed": ["float32"]}
    reference = load_reference(qwen3_moe, torch.device("cpu"), torch.float32)
    assert manifest["parameter_bytes"] == {"streamed": parameter_bytes(reference)["cpu"]}
    for (name, input_ids), summary in zip(
        QWEN3_MOE_PROMPT_IDS.items(), manifest["prompts"], strict=True
    ):
        whole = load_file(truth / f"{name}.safetensors")
        streamed = load_file(output / f"{name}.safetensors")
        assert streamed.keys() == whole.keys()
        for key in ("input_ids", "greedy_ids", "teacher_forced_ids", "topk_ids"):
            assert torch.equal(streamed[key], whole[key]), key
        for key in ("topk_logits", "logsumexp"):
            assert torch.equal(streamed[key], whole[key]), key
        # The whole run's margins and prefill row come from its cached decode, the streamed
        # run's from the teacher-forced rows.
        for key in ("greedy_margins", "prefill_logits"):
            torch.testing.assert_close(streamed[key], whole[key], rtol=0, atol=1e-4)
        first = len(input_ids) - 1
        rows = whole["topk_ids"][first : first + GREEDY_TOKENS, 0]
        assert summary["continuation_not_top1"] == int((rows != whole["greedy_ids"]).sum())


def test_the_stream_names_the_device_as_its_tensors_do(qwen3_moe: Path) -> None:
    """``cpu:0`` is the CPU, whose tensors report ``cpu``: a valid alias loads and runs."""

    from tests.support.hf_layer_stream import LayerStream

    stream = LayerStream(qwen3_moe, device=torch.device("cpu:0"), dtype=torch.float32)
    with torch.inference_mode():
        (hidden,) = stream.forward([QWEN3_MOE_PROMPT_IDS["short"]])
        (logits,) = stream.logits(hidden)
    stream.drop_head()

    assert hidden.device == torch.device("cpu")
    reference = load_reference(qwen3_moe, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        expected = reference(input_ids=torch.tensor([QWEN3_MOE_PROMPT_IDS["short"]])).logits[0]
    torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-5)


def test_query_blocked_attention_computes_the_same_attention(
    qwen3_moe: Path, fake_kernels: Path, qwen3_moe_truth: tuple[dict, Path], tmp_path: Path
) -> None:
    """Blocks of 7 query rows reorder only the reductions over keys."""

    _, truth = qwen3_moe_truth
    manifest, output = _streamed(qwen3_moe, truth, fake_kernels, tmp_path, "--attention-chunk", "7")

    assert (manifest["attention"], manifest["attention_chunk"]) == ("cti_query_chunked_eager", 7)
    for name in QWEN3_MOE_PROMPT_IDS:
        whole = load_file(truth / f"{name}.safetensors")
        blocked = load_file(output / f"{name}.safetensors")
        for key in ("topk_logits", "logsumexp"):
            torch.testing.assert_close(blocked[key], whole[key], rtol=0, atol=1e-5)
        margins = whole["topk_logits"][:, 0] - whole["topk_logits"][:, 1]
        decided = margins > 1e-3
        assert torch.equal(blocked["topk_ids"][decided, 0], whole["topk_ids"][decided, 0])


def test_streaming_refuses_a_recurrent_model_and_options_without_it(
    checkpoint: Path, fake_kernels: Path, truth: tuple[dict, Path], tmp_path: Path
) -> None:
    _, truth_folder = truth
    output = tmp_path / "streamed"
    completed = _run_helper(
        checkpoint,
        _prompts_file(tmp_path, PROMPT_IDS),
        output,
        _child_env(fake_kernels, tmp_path / "imported"),
        "--mode",
        "stream",
        "--continuation",
        str(truth_folder),
    )
    assert completed.returncode != 0
    assert "layer streaming runs attention-only decoders" in completed.stderr, completed.stderr
    assert not any(output.iterdir())

    for options, error in (
        (("--mode", "stream"), "--mode stream takes --continuation"),
        (("--mode", "stream", "--continuation", "x", "--truth", "y"), "without --truth"),
        (("--attention-chunk", "4"), "need --mode stream"),
        (("--continuation", "x"), "need --mode stream"),
    ):
        command = [sys.executable, "-m", HELPER_MODULE, "--checkpoint", str(tmp_path)]
        command += ["--model-key", MODEL_KEY, "--prompts", str(tmp_path / "prompts.json")]
        command += ["--output", str(tmp_path / "out"), *options]
        completed = subprocess.run(
            command, capture_output=True, text=True, cwd=REPO_ROOT, timeout=120, check=False
        )
        assert completed.returncode == 2, options
        assert error in completed.stderr, completed.stderr
