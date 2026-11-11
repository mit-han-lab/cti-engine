"""The FP32 truth of teacher-forced sequences on a tiny random dense Qwen3 on the CPU: computed
whole or one layer at a time, stored as top-2 rows, then served from the cache (the test's own
without a named one); a host that cannot hold the model fails the test with the job that
computes it elsewhere."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from transformers import Qwen3ForCausalLM

from tests.support import qwen3_5_reference_cache as reference_cache
from tests.support import reference_truth
from tests.support.checkpoints import QWEN3_5_9B, Checkpoint
from tests.support.hf_truth import load_reference, use_ieee_fp32
from tests.support.tiny_configs import tiny_hf_config

REQUESTS = [
    {"prompt_ids": [3, 17, 42, 8, 61], "continuation_ids": [7, 9, 11]},
    {"prompt_ids": [(5 * index + 1) % 64 for index in range(40)], "continuation_ids": [1, 2]},
]
SEQUENCES = {(tuple(r["prompt_ids"]), tuple(r["continuation_ids"])) for r in REQUESTS}


@pytest.fixture(scope="module")
def qwen3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("tiny-qwen3")
    torch.manual_seed(0)
    Qwen3ForCausalLM(tiny_hf_config("Qwen3ForCausalLM", folder)).save_pretrained(folder)
    return folder


@pytest.fixture
def ieee_fp32():
    backends = (
        torch.backends,
        torch.backends.cuda.matmul,
        torch.backends.cudnn,
        torch.backends.cudnn.conv,
        torch.backends.cudnn.rnn,
    )
    before = [backend.fp32_precision for backend in backends]
    use_ieee_fp32()
    yield
    for backend, precision in zip(backends, before, strict=True):
        backend.fp32_precision = precision


def _whole_model_rows(qwen3: Path) -> list[torch.Tensor]:
    model = load_reference(qwen3, torch.device("cpu"), torch.float32)
    rows = []
    for request in REQUESTS:
        sequence = request["prompt_ids"] + request["continuation_ids"]
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([sequence]), use_cache=False).logits[0]
        first = len(request["prompt_ids"]) - 1
        rows.append(logits[first : first + len(request["continuation_ids"])])
    return rows


@pytest.mark.parametrize("placement", ["whole", "stream"])
def test_fp32_truth_is_the_whole_models_on_either_placement(qwen3, ieee_fp32, placement):
    rows, proof = reference_cache.fp32_truth(
        qwen3, REQUESTS, device=torch.device("cpu"), placement=placement
    )
    for actual, expected in zip(rows, _whole_model_rows(qwen3), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert proof["placement"]["mode"] == placement
    assert proof["dtype"] == "torch.float32" and proof["max_nats"] <= 1e-4


@pytest.fixture
def truth(qwen3, tmp_path, monkeypatch):
    """``SequenceTruth`` of the tiny checkpoint, its child run in this process on the CPU."""

    monkeypatch.setenv("CTI_REFERENCE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(Checkpoint, "snapshot", lambda self: qwen3)
    monkeypatch.setattr(reference_cache, "implementation_identity", lambda: "build")
    identity = dict.fromkeys(
        "model_revision config_hash weights_hash dtype body head attention ieee torch_version "
        "transformers_version implementation_hash tokenizer_hash template_hash processor_hash "
        "adapter".split(),
        "pinned",
    )
    identity["implementation_hash"] = "build"
    monkeypatch.setattr(reference_cache, "reference_identity", lambda *_: identity)
    children = []

    def child(command, log_path, **_):
        children.append(command)
        report = reference_cache._compute_misses(Path(command[-1]), device=torch.device("cpu"))
        return reference_cache.RECEIPT_PREFIX + json.dumps(report) + "\n"

    monkeypatch.setattr(reference_truth, "run_isolated_process", child)
    checkpoint = QWEN3_5_9B
    return reference_truth.SequenceTruth(checkpoint, qwen3, tmp_path), children


def test_sequence_truth_computes_what_the_cache_lacks_then_serves_it(
    qwen3, truth, ieee_fp32, tmp_path
):
    sequence_truth, children = truth
    found, report = sequence_truth.require(SEQUENCES, tmp_path / "job.pt")
    assert (report["cached"], report["computed"], len(children)) == (0, 2, 1)
    assert report["placement"]["mode"] == "whole"
    for request, logits in zip(REQUESTS, _whole_model_rows(qwen3), strict=True):
        entry = found[tuple(request["prompt_ids"]), tuple(request["continuation_ids"])]
        values, ids = logits.topk(4, dim=-1)
        assert torch.equal(entry["topk_ids"], ids)
        assert torch.equal(entry["greedy_ids"], ids[:, 0])
        torch.testing.assert_close(entry["greedy_margins"], values[:, 0] - values[:, 1])
    again, report = sequence_truth.require(SEQUENCES, tmp_path / "job.pt")
    assert (report["cached"], report["computed"], len(children)) == (2, 0, 1)
    assert again.keys() == found.keys()


def test_without_a_named_cache_the_truth_serves_the_test_only(
    truth, ieee_fp32, tmp_path, monkeypatch
):
    sequence_truth, children = truth
    monkeypatch.delenv("CTI_REFERENCE_CACHE_DIR")

    _, report = sequence_truth.require(SEQUENCES, tmp_path / "job.pt")

    assert report["cache"] == str((tmp_path / "reference-cache").resolve())
    assert (report["computed"], len(children)) == (2, 1)
    assert "CTI_REFERENCE_CACHE_DIR" not in os.environ


def test_a_host_that_cannot_hold_the_model_fails_with_the_job(truth, tmp_path, monkeypatch):
    sequence_truth, _ = truth

    def unavailable(*_, **__):
        raise reference_cache.TruthUnavailable("33 GiB of FP32 weights exceed 9 GiB")

    monkeypatch.setattr(reference_cache, "fp32_truth", unavailable)
    job = tmp_path / "receipts" / "job.pt"
    with pytest.raises(pytest.fail.Exception, match=f"exceed 9 GiB.*{job}"):
        sequence_truth.require(SEQUENCES, job)
    requests = torch.load(job, weights_only=True)["requests"].values()
    assert sorted(r["continuation_ids"] for r in requests) == [[1, 2], [7, 9, 11]]


def test_image_prompts_key_their_truth_by_their_images_and_carry_them(truth, tmp_path, monkeypatch):
    sequence_truth, _ = truth

    def unavailable(*_, **__):
        raise reference_cache.TruthUnavailable("no GPU")

    monkeypatch.setattr(reference_cache, "fp32_truth", unavailable)
    prompt, continuation = tuple(REQUESTS[0]["prompt_ids"]), tuple(REQUESTS[0]["continuation_ids"])
    # Same-sized images: the same ids, other pixels.
    images = {
        (prompt, shade): {
            "originals": [("RGB", (2, 1), bytes([shade]) * 6)],
            "input_ids": torch.tensor([prompt]),
            "pixel_values": torch.full((4, 6), float(shade)),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "engine_positions": torch.zeros(3, len(prompt), dtype=torch.int64),
        }
        for shade in (0, 1)
    }
    jobs = {"text": (SEQUENCES, None), "image": ({(key, continuation) for key in images}, images)}
    for name, (sequences, inputs) in jobs.items():
        with pytest.raises(pytest.fail.Exception):
            sequence_truth.require(sequences, tmp_path / f"{name}.pt", inputs)
    text, image = (
        torch.load(tmp_path / f"{name}.pt", weights_only=True)["requests"] for name in jobs
    )
    assert len(image) == 2 and not text.keys() & image.keys()
    for request in image.values():
        assert request["prompt_ids"] == list(prompt)
        assert request["images"].keys() == {
            "input_ids",
            "pixel_values",
            "image_grid_thw",
            "engine_positions",
        }
