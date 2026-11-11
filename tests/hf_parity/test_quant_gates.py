"""The quantized checkpoints' GPU gates, one test per
segment, gate and case (checkpoint and topology), each a GPU call of at most 30 minutes selected
by ``-k``. Each writes its result to ``$QUANT_GATE_OUT/<gate>/<case>.json`` before it asserts
(``tests/support/quant_gates.py``). Run by hand only, in the order of ``README.md``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.support.quant_gates import (
    CASES,
    GRAPH_EAGER_ISSUE,
    GateCase,
    bf16_fidelity,
    compare_outputs,
    dequantize,
    generate_truth,
    graph_eager,
    graph_eager_judgement,
    lora,
    performance,
    quantized_fidelity,
    switching,
    write_result,
)

_CASES = [pytest.param(case, id=case.id) for case in CASES]
# The gates that train: a checkpoint whose recipe serves only is refused before them.
_TRAINING_CASES = [pytest.param(case, id=case.id) for case in CASES if case.checkpoint.trains]
# One case per checkpoint: its truths do not depend on the engine's topology.
_CHECKPOINTS = [
    pytest.param(case, id=case.checkpoint.name)
    for case in {case.checkpoint.name: case for case in reversed(CASES)}.values()
]


@pytest.mark.parametrize("batch", [1, 2, 3, 4], ids=lambda batch: f"b{batch}")
@pytest.mark.parametrize("case", _CASES)
def test_graph_equals_eager(case: GateCase, batch: int, tmp_path: Path) -> None:
    """Decode-graph replay against eager decoding. B = 1: the same tokens with bitwise-equal
    logprobs. B = 2..4: within the accuracy tolerance (``graph_eager_judgement``); bitwise
    equality is recorded, pending a cti-kernels fix (attention dispatch differs between graph
    and eager at B>1)."""

    outputs = graph_eager(case, tmp_path)
    graph, eager = outputs["graph"][str(batch)], outputs["eager"][str(batch)]
    result = {"case": case.id, "batch": batch, "bitwise": compare_outputs(graph, eager)}
    if batch > 1:
        result["tolerance"] = graph_eager_judgement(graph, eager)
        result["bitwise_pending"] = GRAPH_EAGER_ISSUE
    write_result("graph_eager", f"{case.id}-b{batch}", result)
    if batch == 1:
        assert result["bitwise"]["ids_equal"] and result["bitwise"]["logprobs_equal"], result
    else:
        assert result["tolerance"]["passed"], result


@pytest.mark.parametrize("case", _CHECKPOINTS)
def test_dequantize_checkpoint(case: GateCase) -> None:
    """Gate 2, segment 1 (CPU): the checkpoint dequantized to FP32 by the numerics contract."""

    result = dequantize(case)
    write_result("dequantize", case.checkpoint.name, result)
    assert result["quantized_linears"] or result["quantized_experts"], result


@pytest.mark.parametrize("kind", ["quantized", "floor", "bf16"])
@pytest.mark.parametrize("case", _CHECKPOINTS)
def test_generate_truth(case: GateCase, kind: str, tmp_path: Path) -> None:
    """Gate 2, segment 2: the FP32 quantized truth, its BF16 floor (after it) or the BF16
    sibling's FP32 truth, unless a kept one is named or already in place."""

    result = generate_truth(case, kind, tmp_path)
    write_result("truths", f"{case.checkpoint.name}-{kind}", result)


@pytest.mark.parametrize("case", _CASES)
def test_matches_quantized_fp32_truth(case: GateCase, tmp_path: Path) -> None:
    """G-C: every prompt within the calibrated bounds of the FP32 quantized truth, every path."""

    result = quantized_fidelity(case, tmp_path)
    path = write_result("quantized_fidelity", case.id, result)
    assert not result["failed"], f"G-C failed (result in {path})"


@pytest.mark.parametrize("case", _CASES)
def test_reports_bf16_fp32_truth(case: GateCase, tmp_path: Path) -> None:
    """G-D against the BF16 sibling's FP32 truth: recorded, not gated until its bounds freeze."""

    result = bf16_fidelity(case, tmp_path)
    write_result("bf16_fidelity", case.id, result)
    assert result["pooled"], result


@pytest.mark.parametrize("role", ["quantized", "bf16"])
@pytest.mark.parametrize("case", _CASES)
def test_latency(case: GateCase, role: str, tmp_path: Path) -> None:
    """TTFT and TPOT at B = 1..4 of the quantized checkpoint or its BF16 sibling; the second
    role run also records the speedup BF16 / quantized. Not gated."""

    result = performance(case, role, tmp_path)
    write_result("performance", f"{case.id}-{role}", result)
    assert "medians_ms" in result or "skipped" in result, result


@pytest.mark.parametrize("case", _TRAINING_CASES)
def test_switching_keeps_quantized_tensors(case: GateCase, tmp_path: Path) -> None:
    """Inference -> training -> inference: every quantized tensor keeps its address and bytes on
    every rank, and serving after each switch repeats the first outputs bit for bit."""

    result = switching(case, tmp_path)
    path = write_result("switching", case.id, result)
    assert result["ranks"], f"the rank probe recorded nothing (result in {path})"
    for rank, record in result["ranks"].items():
        assert record["tensors"] and len(record["events"]) >= 2, (rank, path)
        assert not record["moved_or_changed"], (rank, record["moved_or_changed"][:8], path)
    for after in result["outputs_after_each_switch"]:
        assert after["ids_equal"] and after["logprobs_equal"], (after, path)
    assert all(math.isfinite(loss) for loss in result["loss_sums"]), path
    assert result["model_revision"] == 0, path


@pytest.mark.parametrize("case", _TRAINING_CASES)
def test_lora_on_quantized_base(case: GateCase, tmp_path: Path) -> None:
    """A zero-initialized adapter serves the base's outputs bit for bit; one committed LoRA
    step trains finitely and changes them."""

    result = lora(case, tmp_path)
    path = write_result("lora", case.id, result)
    zero = result["zero_adapter_vs_base"]
    assert zero["ids_equal"] and zero["logprobs_equal"], (zero, path)
    assert math.isfinite(result["loss_sum"]) and result["grad_norm"] > 0, path
    assert (result["step"], result["model_revision"]) == (1, 1), path
    assert not result["trained_vs_zero_adapter"]["logprobs_equal"], path
