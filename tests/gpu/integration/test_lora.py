from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    assert_cuda_device_metadata,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.inference_correctness import GOLDEN_CASES
from tests.support.paths import REPO_ROOT
from tests.support.training import (
    TrainingModelStub,
    forward_backward,
    host_runner,
    optim_step,
)

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

if TYPE_CHECKING:
    import torch

    from cti.training import TrainingCheckpoint

RECEIPT_PREFIX = "CTI_LORA_SMOKE="
# CUDA foreach changes FP32 operation order; this is the near-zero-aware
# ``atol + rtol * abs(reference)`` gate against the independent loop oracle.
FOREACH_ORACLE_RTOL = 1e-6
FOREACH_ORACLE_ATOL = 1e-8


def _expected_qwen_adapter_names(num_layers: int = 28) -> tuple[str, ...]:
    names = []
    for layer_id in range(num_layers):
        prefix = f"model.layers.{layer_id}"
        projections = (
            (f"{prefix}.self_attn.qkv_proj", ("q", "k", "v")),
            (f"{prefix}.self_attn.o_proj", ("full",)),
            (f"{prefix}.mlp.gate_up_proj", ("gate", "up")),
            (f"{prefix}.mlp.down_proj", ("full",)),
        )
        for projection, slices in projections:
            for name in slices:
                names.append(f"{projection}.{name}.lora_A")
                names.append(f"{projection}.{name}.lora_B")
    return tuple(names)


def _reference_adamw_step(
    parameters: tuple[torch.nn.Parameter, ...],
    optimizer: torch.optim.AdamW,
    raw_gradients: tuple[torch.Tensor, ...],
    *,
    num_target_tokens: int,
    max_grad_norm: float | None,
) -> float:
    import torch

    for parameter, gradient in zip(parameters, raw_gradients, strict=True):
        assert parameter.grad is None
        parameter.grad = gradient.clone()

    gradients = tuple(parameter.grad for parameter in parameters)
    assert all(gradient is not None for gradient in gradients)
    raw_norm = torch.nn.utils.get_total_norm(
        gradients,
        norm_type=2.0,
        error_if_nonfinite=True,
        foreach=False,
    )
    normalized_norm = raw_norm / num_target_tokens
    with torch.no_grad():
        for gradient in gradients:
            assert gradient is not None
            gradient.div_(num_target_tokens)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grads_with_norm_(
                parameters,
                max_grad_norm,
                normalized_norm,
                foreach=False,
            )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(normalized_norm)


def _assert_adamw_reference_matches(
    live_parameters: tuple[torch.Tensor, ...],
    checkpoint: TrainingCheckpoint,
    reference_parameters: tuple[torch.nn.Parameter, ...],
    reference_optimizer: torch.optim.AdamW,
    *,
    expected_step: int,
) -> None:
    import torch

    optimizer = checkpoint.optimizer
    for index, (live, master, reference) in enumerate(
        zip(live_parameters, optimizer.master, reference_parameters, strict=True)
    ):
        torch.testing.assert_close(
            master,
            reference.detach().cpu(),
            rtol=FOREACH_ORACLE_RTOL,
            atol=FOREACH_ORACLE_ATOL,
        )
        torch.testing.assert_close(live.cpu(), master.to(live.dtype), rtol=0, atol=0)

        state = optimizer.adamw["state"][index]
        reference_state = {
            name: value.cpu() for name, value in reference_optimizer.state[reference].items()
        }
        assert state.keys() == reference_state.keys() == {"step", "exp_avg", "exp_avg_sq"}
        assert int(state["step"].item()) == expected_step
        torch.testing.assert_close(state["step"], reference_state["step"], rtol=0, atol=0)
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                state[name],
                reference_state[name],
                rtol=FOREACH_ORACLE_RTOL,
                atol=FOREACH_ORACLE_ATOL,
            )


def _captured_projection_update() -> dict[str, object]:
    import torch
    import torch.nn.functional as F

    from cti.config import (
        AdamWConfig,
        BatchLayout,
        LoRAConfig,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.layers import Adapter, BaseOP, LinearReplicated
    from cti.training import SFT, PaddedTokens, TrainingBatch
    from tests.support.tp_shards import site

    device = torch.device("cuda")
    torch.manual_seed(20260704)
    base = LinearReplicated(16, 12, has_bias=False, lora="attention", site=site(device=device))
    base.weight = torch.randn(
        12,
        16,
        dtype=torch.bfloat16,
        device=device,
    )

    adapters = Adapter(
        LoRAConfig(
            targets=("attention",),
            rank=4,
            alpha=8.0,
        )
    )

    class Projection(BaseOP):
        def __init__(self) -> None:
            self.projection = base

    adapters.attach(Projection())
    projection = base
    adapters.materialize(device=device, dtype=torch.bfloat16)

    class ProjectionCausalModel(TrainingModelStub):
        @staticmethod
        def state_dict() -> dict[str, torch.Tensor]:
            return {"projection.weight": base.weight}

        @staticmethod
        def named_trainable_parameters() -> tuple[tuple[str, torch.Tensor], ...]:
            return adapters.named_parameters()

        def bind_training_weights(self, masters: dict[str, torch.Tensor]) -> None:
            super().bind_training_weights(masters)
            adapters.bind_training(masters)

        def release_training_weights(self) -> None:
            super().release_training_weights()
            adapters.release_training()

        @staticmethod
        def forward_training(input_ids: torch.Tensor) -> torch.Tensor:
            features = F.one_hot(input_ids.remainder(16), num_classes=16).to(torch.bfloat16)
            return projection.forward_training(features)

        def training_hidden_states(self, input_ids: torch.Tensor, packing: object) -> torch.Tensor:
            return self.forward_training(input_ids).flatten(0, -2)

        def training_loss(
            self, hidden_states: torch.Tensor, target_ids: torch.Tensor
        ) -> torch.Tensor:
            return F.cross_entropy(hidden_states.float(), target_ids, reduction="sum")

    model = ProjectionCausalModel()
    named_parameters = adapters.named_parameters()
    optimizer_config = AdamWConfig(
        lr=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(tuple(name for name, _ in named_parameters)),
            optimizer_config,
            TrainingBatching(
                BatchLayout.PADDED,
                StaticBatching(microbatch_size=1, max_tokens_per_sample=6),
            ),
        ),
        trainables=model,
        checks=True,
    )
    live_parameters = tuple(t for _, t in adapters.named_parameters())
    initial = runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer
    assert initial.names == tuple(name for name, _ in adapters.named_parameters())
    assert len(initial.master) == len(live_parameters)
    assert all(parameter.dtype == torch.bfloat16 for parameter in live_parameters)
    # Every BF16 adapter updates through an FP32 master.
    assert all(master is not None and master.dtype == torch.float32 for master in initial.master)
    batch = TrainingBatch(
        PaddedTokens(
            torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64),
            torch.tensor([6], dtype=torch.int64),
            0,
        ),
        torch.ones((1, 5), dtype=torch.bool),
        SFT(),
    )
    static_input = torch.randn(8, 16, dtype=torch.bfloat16, device=device)

    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(2):
            projection.forward(static_input)
    torch.cuda.current_stream(device).wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    graph_identity = id(graph)
    with torch.inference_mode(), torch.cuda.graph(graph):
        graph_output = projection.forward(static_input)
    graph.replay()
    torch.cuda.synchronize(device)
    before = graph_output.clone()
    assert not graph_output.requires_grad
    with torch.inference_mode():
        eager_before = projection.forward(static_input)
    torch.testing.assert_close(before, eager_before)
    pointers = tuple(parameter.data_ptr() for parameter in live_parameters)
    identities = tuple(id(parameter) for parameter in live_parameters)
    reference_parameters = tuple(torch.nn.Parameter(master.to(device)) for master in initial.master)
    reference_optimizer = torch.optim.AdamW(
        reference_parameters,
        lr=optimizer_config.lr,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
        foreach=False,
        fused=False,
    )

    previous_graph_output = before
    microbatches_per_commit = 2
    for expected_step in range(1, 3):
        first_main_gradients = None
        num_target_tokens = 0
        for microbatch_index in range(microbatches_per_commit):
            training_result = forward_backward(runner, batch)
            num_target_tokens += training_result.normalization_count
            assert all(parameter.grad is None for parameter in live_parameters)
            main_gradients = [gradient.to(device) for gradient in runner.accumulated_gradients()]
            assert all(gradient.dtype == torch.float32 for gradient in main_gradients)
            if microbatch_index == 0:
                first_main_gradients = tuple(main_gradients)
            else:
                assert first_main_gradients is not None
                for accumulated, first in zip(
                    main_gradients,
                    first_main_gradients,
                    strict=True,
                ):
                    torch.testing.assert_close(accumulated, first * 2, rtol=0, atol=0)

        assert num_target_tokens == microbatches_per_commit * batch.num_targets
        raw_main_gradients = tuple(main_gradients)

        grad_norm = optim_step(runner, num_target_tokens)
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(graph_output, previous_graph_output, rtol=0, atol=0)
        runner.publish()
        reference_grad_norm = _reference_adamw_step(
            reference_parameters,
            reference_optimizer,
            raw_main_gradients,
            num_target_tokens=num_target_tokens,
            max_grad_norm=optimizer_config.max_grad_norm,
        )
        assert grad_norm == reference_grad_norm
        torch.cuda.synchronize(device)

        assert id(graph) == graph_identity
        assert tuple(parameter.data_ptr() for parameter in live_parameters) == pointers
        assert tuple(id(parameter) for parameter in live_parameters) == identities
        assert all(parameter.grad is None for parameter in live_parameters)
        assert not runner.holds_gradients
        _assert_adamw_reference_matches(
            live_parameters,
            runner.checkpoint(num_optimizer_steps=expected_step, model_revision=0),
            reference_parameters,
            reference_optimizer,
            expected_step=expected_step,
        )

        graph.replay()
        torch.cuda.synchronize(device)
        replayed_output = graph_output.clone()
        with torch.inference_mode():
            eager_output = projection.forward(static_input)
        torch.testing.assert_close(replayed_output, eager_output)
        assert not torch.equal(replayed_output, previous_graph_output)
        previous_graph_output = replayed_output

    return {
        "optimizer_update_completed": True,
        "optimizer_commit_count": 2,
        "microbatches_per_commit": microbatches_per_commit,
        "foreach_loop_oracle": {
            "matches": True,
            "rtol": FOREACH_ORACLE_RTOL,
            "atol": FOREACH_ORACLE_ATOL,
        },
        "live_publication_master_cast_exact": True,
        "graph_object_preserved": True,
        "graph_update_visible": True,
        "graph_matches_eager": True,
        "parameter_identity_preserved": True,
        "parameter_storage_preserved": True,
        "parameter_count": len(pointers),
        "live_dtype": str(live_parameters[0].dtype),
        "master_dtype": str(initial.master[0].dtype),
        "optimizer_state_dtype": str(initial.adamw["state"][0]["exp_avg"].dtype),
    }


def _run_child() -> None:
    import torch

    receipt = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "projection": _captured_projection_update(),
    }
    torch.cuda.synchronize()
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def test_lora_projection_update_reaches_captured_graph(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "lora")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.integration.test_lora"),
        log_root / "lora.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
    )

    receipt_lines = [
        line.removeprefix(RECEIPT_PREFIX)
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    assert len(receipt_lines) == 1
    receipt = json.loads(receipt_lines[0])
    assert_cuda_device_metadata(receipt)
    assert receipt["projection"]["optimizer_update_completed"] is True
    assert receipt["projection"]["optimizer_commit_count"] == 2
    assert receipt["projection"]["microbatches_per_commit"] == 2
    assert receipt["projection"]["foreach_loop_oracle"] == {
        "matches": True,
        "rtol": FOREACH_ORACLE_RTOL,
        "atol": FOREACH_ORACLE_ATOL,
    }
    assert receipt["projection"]["live_publication_master_cast_exact"] is True
    assert receipt["projection"]["graph_object_preserved"] is True
    assert receipt["projection"]["graph_update_visible"] is True
    assert receipt["projection"]["graph_matches_eager"] is True
    assert receipt["projection"]["parameter_identity_preserved"] is True
    assert receipt["projection"]["parameter_storage_preserved"] is True
    assert receipt["projection"]["live_dtype"] == "torch.bfloat16"
    assert receipt["projection"]["master_dtype"] == "torch.float32"
    assert receipt["projection"]["optimizer_state_dtype"] == "torch.float32"


def test_qwen_lora_training_updates_served_adapter_in_place() -> None:
    """Zero-initialized adapters serve the base model's golden output; one optimizer step
    reaches the served output at once, with no reload, and training copies no base weights."""

    import torch

    from cti.config import (
        AdamWConfig,
        BatchLayout,
        CacheOptions,
        EngineOptions,
        EngineTuning,
        LoRAConfig,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams
    from cti.training import SFT, PaddedTokens, TrainingBatch

    snapshot = QWEN3_0_6B.snapshot()
    base_weight_bytes = sum(path.stat().st_size for path in snapshot.glob("*.safetensors"))
    case = GOLDEN_CASES[0]
    free_idle = torch.cuda.mem_get_info()[0]
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=str(snapshot), max_seq_len=128),
            memory=MemoryOptions(inference_ratio=0.5, peak_ratio=0.85),
            scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=128),
            cache=CacheOptions(prefix="naive"),
            lora=LoRAConfig(targets=("attention", "mlp"), rank=8, alpha=16.0),
            # Every attention and MLP adapter: the plan fails to resolve if one is missing.
            training=TrainingConfig(
                trainable_plan=TrainablePlan(_expected_qwen_adapter_names()),
                optimizer=AdamWConfig(lr=1e-4, weight_decay=0.0, max_grad_norm=1.0),
                batching=TrainingBatching(
                    BatchLayout.PADDED,
                    StaticBatching(microbatch_size=1, max_tokens_per_sample=64),
                ),
            ),
        ),
        tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)),
    )
    params = SamplingParams(
        temperature=0, ignore_eos=True, max_new_tokens=len(case.output_ids), logprobs=True
    )
    training_tokens = tuple(case.prompt_ids) + tuple(case.output_ids[:8])
    batch = TrainingBatch(
        PaddedTokens(
            torch.tensor([training_tokens], dtype=torch.int64),
            torch.tensor([len(training_tokens)], dtype=torch.int64),
            151643,
        ),
        torch.ones((1, len(training_tokens) - 1), dtype=torch.bool),
        SFT(),
    )
    try:
        first, control = (llm.sample([list(case.prompt_ids)], params) for _ in range(2))
        free_serving = torch.cuda.mem_get_info()[0]
        with llm.training_session() as session:
            completion = session.forward_backward(batch)
            step = session.optim_step().result()
            assert llm.model_revision == 0
            assert llm.sample([list(case.prompt_ids)], params).samples == first.samples
            session.publish()
            free_training = torch.cuda.mem_get_info()[0]
        updated = llm.sample([list(case.prompt_ids)], params)
        revision = llm.model_revision
    finally:
        llm.shutdown()

    assert first.samples[0].sampled_token_ids == tuple(case.output_ids)
    assert first.texts[0] == case.output_text
    assert control.samples == first.samples and first.model_revision == 0
    assert completion.result().normalization_count == len(training_tokens) - 1
    assert completion.result().loss_sum > 0
    assert step.step == 1 and revision == updated.model_revision == 1
    # The served adapters are the trained ones: the step moved the greedy-token logprobs.
    assert updated.samples[0].sampling_logprobs != first.samples[0].sampling_logprobs
    # Training shares the served base weights rather than holding a copy: the switch releases
    # the serving state, and the engine then holds one copy of the weights and less than another
    # copy's worth besides (2.30 GB for 1.50 GB of weights on RTX 5090).
    assert free_training > free_serving
    assert free_idle - free_training < 2 * base_weight_bytes


if __name__ == "__main__":
    _run_child()
