from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.support.gpu import isolated_runtime_env, run_isolated_process
from tests.support.paths import REPO_ROOT
from tests.support.training import (
    TrainingModelStub,
    assert_optimizer_state_exact,
    forward_backward,
    host_runner,
    optim_step,
)

pytestmark = pytest.mark.checkpoints()

RECEIPT_PREFIX = "CTI_MIXED_PRECISION_OPTIMIZER="


def _run_smoke() -> dict[str, object]:
    import statistics

    import torch
    import torch.nn.functional as F

    from cti.config import (
        AdamWConfig,
        BatchLayout,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.training import SFT, PaddedTokens, TrainingBatch

    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(20260717)

    class TinyCausalLM(TrainingModelStub):
        def __init__(self) -> None:
            self.weight_fp32 = torch.nn.Parameter(
                torch.randn(
                    7,
                    4,
                    generator=generator,
                    dtype=torch.float32,
                ).to(device),
                requires_grad=False,
            )
            self.weight_bf16 = torch.nn.Parameter(
                torch.randn(
                    7,
                    4,
                    generator=generator,
                    dtype=torch.bfloat16,
                ).to(device),
                requires_grad=False,
            )

        def state_dict(self) -> dict[str, torch.Tensor]:
            return {
                "weight_fp32": self.weight_fp32,
                "weight_bf16": self.weight_bf16,
            }

        def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
            return (
                ("weight_fp32", self.weight_fp32),
                ("weight_bf16", self.weight_bf16),
            )

        def forward_training(self, input_ids: torch.Tensor) -> torch.Tensor:
            features = F.one_hot(input_ids.remainder(4), num_classes=4)
            return (
                F.linear(features.float(), self.trained(self.weight_fp32))
                + F.linear(
                    features.to(torch.bfloat16),
                    self.trained(self.weight_bf16),
                ).float()
            )

        def training_hidden_states(self, input_ids: torch.Tensor, packing: object) -> torch.Tensor:

            return self.forward_training(input_ids).flatten(0, -2)

        def training_loss(
            self, hidden_states: torch.Tensor, target_ids: torch.Tensor
        ) -> torch.Tensor:

            return F.cross_entropy(hidden_states.float(), target_ids, reduction="sum")

    model = TinyCausalLM()
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(("weight_fp32", "weight_bf16")),
            AdamWConfig(lr=0.03, weight_decay=0.0, max_grad_norm=1.0),
            TrainingBatching(
                BatchLayout.PADDED,
                StaticBatching(microbatch_size=1, max_tokens_per_sample=5),
            ),
        ),
        trainables=model,
        checks=True,
    )
    # Both live weights have separate FP32 masters on the device.
    initial = runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer
    assert all(master.dtype == torch.float32 for master in initial.master)
    for master, live in zip(initial.master, initial.live, strict=True):
        torch.testing.assert_close(master, live.float(), rtol=0, atol=0)
    masters = runner.memory_usage().master_weights
    assert (
        masters.payload
        == (model.weight_fp32.numel() + model.weight_bf16.numel()) * torch.float32.itemsize
    )
    assert masters.pools

    live_ids = (id(model.weight_fp32), id(model.weight_bf16))
    live_pointers = (model.weight_fp32.data_ptr(), model.weight_bf16.data_ptr())
    static_input = torch.randn(8, 4, dtype=torch.bfloat16, device=device)

    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(2):
            (
                F.linear(static_input.float(), model.weight_fp32)
                + F.linear(
                    static_input,
                    model.weight_bf16,
                ).float()
            )
    torch.cuda.current_stream(device).wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        graph_output = (
            F.linear(static_input.float(), model.weight_fp32)
            + F.linear(
                static_input,
                model.weight_bf16,
            ).float()
        )
    graph.replay()
    torch.cuda.synchronize(device)
    before = graph_output.clone()

    batch = TrainingBatch(
        PaddedTokens(
            torch.tensor(((1, 2, 3, 4, 5),), dtype=torch.int64),
            torch.tensor((5,), dtype=torch.int64),
            0,
        ),
        torch.ones((1, 4), dtype=torch.bool),
        SFT(),
    )
    warmup = 5
    measurements = 5
    forward_backward_ms = []
    optimizer_ms = []
    grad_norm = None
    for iteration in range(warmup + measurements):
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)
        forward_start.record()
        result = forward_backward(runner, batch)
        forward_end.record()
        assert model.trained.masters["weight_fp32"].grad is not None
        assert model.trained.masters["weight_bf16"].grad is not None
        assert model.weight_fp32.grad is None
        assert model.weight_bf16.grad is None
        if iteration < warmup:  # untimed: the view copies to the host
            assert all(
                gradient.dtype == torch.float32 for gradient in runner.accumulated_gradients()
            )
        grad_norm = optim_step(runner, result.normalization_count)
        optimizer_end.record()
        optimizer_end.synchronize()
        if iteration >= warmup:
            forward_backward_ms.append(forward_start.elapsed_time(forward_end))
            optimizer_ms.append(forward_end.elapsed_time(optimizer_end))
    assert grad_norm is not None

    assert (id(model.weight_fp32), id(model.weight_bf16)) == live_ids
    assert (model.weight_fp32.data_ptr(), model.weight_bf16.data_ptr()) == live_pointers
    assert model.weight_fp32.grad is None
    assert model.weight_bf16.grad is None
    assert not runner.holds_gradients
    branch = runner.checkpoint(num_optimizer_steps=0, model_revision=0)
    for state in branch.optimizer.adamw["state"].values():
        assert state.keys() == {"step", "exp_avg", "exp_avg_sq"}
        assert all(value.dtype == torch.float32 for value in state.values())

    expected_result = forward_backward(runner, batch)
    expected_norm = optim_step(runner, expected_result.normalization_count)
    expected_state = runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer

    # The restore copies in place: the graph captured before it replays below.
    runner.restore(branch)
    assert (id(model.weight_fp32), id(model.weight_bf16)) == live_ids
    assert (model.weight_fp32.data_ptr(), model.weight_bf16.data_ptr()) == live_pointers
    resumed_result = forward_backward(runner, batch)
    resumed_norm = optim_step(runner, resumed_result.normalization_count)
    resumed_state = runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer
    assert resumed_norm == expected_norm
    assert_optimizer_state_exact(resumed_state, expected_state)

    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(graph_output, before, rtol=0, atol=0)
    runner.publish()
    graph.replay()
    torch.cuda.synchronize(device)
    after = graph_output.clone()
    with torch.inference_mode():
        eager_after = (
            F.linear(static_input.float(), model.weight_fp32)
            + F.linear(
                static_input,
                model.weight_bf16,
            ).float()
        )
    torch.testing.assert_close(after, eager_after, rtol=0, atol=0)
    assert not torch.equal(after, before)

    return {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "grad_norm": grad_norm,
        "warmup": warmup,
        "measurements": measurements,
        "forward_backward_median_device_ms": statistics.median(forward_backward_ms),
        "optimizer_median_device_ms": statistics.median(optimizer_ms),
        "master_dtype": str(initial.master[1].dtype),
        "fp32_live_dtype": str(model.weight_fp32.dtype),
        "live_dtype": str(model.weight_bf16.dtype),
        "checkpoint_exact_resume": True,
        "optimizer_state_fp32": True,
        "graph_update_visible": True,
        "graph_matches_eager": True,
        "live_identity_preserved": True,
        "live_storage_preserved": True,
    }


def _run_child() -> None:
    import torch

    receipt = _run_smoke()
    torch.cuda.synchronize()
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def test_mixed_precision_optimizer_cuda_graph_smoke(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "mixed-precision-optimizer")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.training.test_mixed_precision_optimizer"),
        log_root / "mixed-precision-optimizer.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=120,
    )

    receipt_lines = [
        line.removeprefix(RECEIPT_PREFIX)
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    assert len(receipt_lines) == 1
    receipt = json.loads(receipt_lines[0])
    assert receipt["device"]
    assert receipt["capability"][0] >= 8
    assert receipt["master_dtype"] == "torch.float32"
    assert receipt["fp32_live_dtype"] == "torch.float32"
    assert receipt["live_dtype"] == "torch.bfloat16"
    assert receipt["checkpoint_exact_resume"] is True
    assert receipt["warmup"] == 5
    assert receipt["measurements"] == 5
    assert receipt["forward_backward_median_device_ms"] > 0
    assert receipt["optimizer_median_device_ms"] > 0
    assert receipt["optimizer_state_fp32"] is True
    assert receipt["graph_update_visible"] is True
    assert receipt["graph_matches_eager"] is True
    assert receipt["live_identity_preserved"] is True
    assert receipt["live_storage_preserved"] is True


if __name__ == "__main__":
    _run_child()
