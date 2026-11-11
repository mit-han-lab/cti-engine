from __future__ import annotations

import gc
import math
import weakref
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_pre_hook

import cti.training._optimizer as optimizer_module
from cti.config import (
    AdamWConfig,
    BatchLayout,
    LoRAConfig,
    SFTShapeProfile,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.layers import Adapter, BaseOP, LinearReplicated
from cti.training import (
    SFT,
    InvalidGradientError,
    PackedTokens,
    PaddedTokens,
    Refusal,
    TrainingBatch,
    TrainingInputError,
)
from cti.training._staging import SFTStaging
from cti.training.runner import TrainingRunner
from tests.support.tp_shards import site
from tests.support.training import (
    TrainingModelStub,
    forward_backward,
    host_runner,
    optim_step,
    synchronize_optimizer_parameters,
)


def _padded_batching(*, microbatch_size: int = 8) -> TrainingBatching:
    return TrainingBatching(
        BatchLayout.PADDED,
        StaticBatching(microbatch_size=microbatch_size, max_tokens_per_sample=64),
    )


CPU = torch.device("cpu")


def _packed_batching(*, microbatch_size: int = 8) -> TrainingBatching:
    return TrainingBatching(
        BatchLayout.PACKED,
        StaticBatching(microbatch_size=microbatch_size, max_tokens_per_sample=64),
    )


class _TinyCausalLM(TrainingModelStub, BaseOP):
    def __init__(self) -> None:
        torch.manual_seed(20260730)
        self.projection = LinearReplicated(4, 7, has_bias=False, lora="attention", site=site())
        self.projection.weight.normal_()
        self.lora_adapters = Adapter(LoRAConfig(targets=("attention",), rank=2, alpha=4.0))
        self.lora_adapters.attach(self)
        self.lora_adapters.materialize(device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad():
            for tensor in self.lora_adapters.state_dict().values():
                tensor.normal_(mean=0.0, std=0.1)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"base.weight": self.projection.weight}

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return self.lora_adapters.named_parameters()

    def bind_training_weights(self, masters: dict[str, torch.Tensor]) -> None:
        super().bind_training_weights(masters)
        self.lora_adapters.bind_training(masters)

    def release_training_weights(self) -> None:
        super().release_training_weights()
        self.lora_adapters.release_training()

    def _training_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = F.one_hot(input_ids.remainder(4), num_classes=4).float()
        return self.projection.forward_training(features)

    def forward_training(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._training_logits(input_ids)


class _ValidatingTinyCausalLM(_TinyCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.validated_targets: list[torch.Tensor] = []

    def validate_training_targets(self, target_ids: torch.Tensor) -> None:
        assert target_ids.device.type == "cpu"
        self.validated_targets.append(target_ids.clone())
        if int(target_ids.max()) >= self.projection.weight.shape[0]:
            raise ValueError("Training targets must be in the model vocabulary")


def _padded_batch(token_ids: torch.Tensor, lengths: torch.Tensor, targets, pad: int = 0):
    return TrainingBatch(PaddedTokens(token_ids, lengths, pad), torch.as_tensor(targets), SFT())


def _packed_batch(token_ids: tuple[int, ...], cu_seqlens: tuple[int, ...]) -> TrainingBatch:
    tokens = PackedTokens(
        torch.tensor(token_ids, dtype=torch.int64), torch.tensor(cu_seqlens, dtype=torch.int32)
    )
    return TrainingBatch(tokens, torch.ones(tokens.target_shape, dtype=torch.bool), SFT())


def _batch(rows: tuple[tuple[int, ...], ...], lengths: tuple[int, ...]) -> TrainingBatch:
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    return _padded_batch(
        torch.tensor(rows, dtype=torch.int64),
        lengths_tensor,
        torch.arange(len(rows[0]) - 1).unsqueeze(0) < (lengths_tensor - 1).unsqueeze(1),
    )


def _combined_batch() -> TrainingBatch:
    return _batch(((1, 2, 3, 4), (4, 5, 0, 0)), (4, 2))


def _packed_combined_batch() -> TrainingBatch:
    return _packed_batch((1, 2, 3, 4, 4, 5), (0, 4, 6))


def _partitions() -> tuple[TrainingBatch, TrainingBatch]:
    return (
        _batch(((1, 2, 3, 4),), (4,)),
        _batch(((4, 5),), (2,)),
    )


def _optimizer_batches() -> tuple[TrainingBatch, ...]:
    return (
        _combined_batch(),
        _batch(((2, 6, 1, 5, 3),), (5,)),
        _batch(((6, 3, 2, 0), (5, 1, 4, 2)), (3, 4)),
    )


def _runner(
    *,
    config: AdamWConfig | None = None,
) -> tuple[_TinyCausalLM, Adapter, TrainingRunner]:
    model = _TinyCausalLM()
    adapters = model.lora_adapters
    named_adapters = adapters.named_parameters()
    training_runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(tuple(name for name, _ in named_adapters)),
            config
            or AdamWConfig(
                lr=0.03,
                betas=(0.8, 0.9),
                eps=1e-6,
                weight_decay=0.02,
                max_grad_norm=0.25,
            ),
            _padded_batching(),
        ),
        trainables=model,
        checks=True,
    )
    return model, adapters, training_runner


@pytest.mark.parametrize("staged", (False, True), ids=("unstaged", "staged"))
@pytest.mark.parametrize("packed", (False, True), ids=("padded", "packed"))
def test_dropped_runner_frees_model_and_staging_without_cycle_collection(
    packed: bool, staged: bool
) -> None:
    def stagings() -> int:
        return sum(isinstance(held, SFTStaging) for held in gc.get_objects())

    gc.collect()
    baseline = stagings()
    model = _TinyCausalLM() if packed else _ValidatingTinyCausalLM()
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(tuple(name for name, _ in model.lora_adapters.named_parameters())),
            AdamWConfig(),
            _packed_batching() if packed else _padded_batching(),
            sft_profile=SFTShapeProfile((4, 2)) if staged else None,
        ),
        trainables=model,
        checks=True,
    )
    forward_backward(runner, _packed_combined_batch() if packed else _combined_batch())
    runner.discard()
    assert stagings() == baseline + staged
    dropped = [weakref.ref(model), weakref.ref(runner)]
    del model
    # The model and its staging must not wait for the cycle collector.
    gc.disable()
    try:
        del runner
        assert all(reference() is None for reference in dropped)
        assert stagings() == baseline
    finally:
        gc.enable()


def test_padded_runner_validates_the_targets_on_cpu_before_the_loss() -> None:
    model = _ValidatingTinyCausalLM()
    adapters = model.lora_adapters
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(tuple(name for name, _ in adapters.named_parameters())),
            AdamWConfig(),
            _padded_batching(),
        ),
        trainables=model,
        checks=True,
    )
    # Padding is validated along with the input tokens before loss dispatch.
    batch = _padded_batch(
        torch.tensor(((1, 2, 3, 4), (4, 5, 0, 0)), dtype=torch.int64),
        torch.tensor((4, 2), dtype=torch.int64),
        ((True, True, True), (True, False, False)),
    )
    forward_backward(runner, batch)
    assert len(model.validated_targets) == 1
    torch.testing.assert_close(
        model.validated_targets[0],
        torch.tensor((1, 2, 3, 4, 4, 5, 0, 0, 0), dtype=torch.int64),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("sft_profile", "rows", "lengths", "message"),
    (
        (None, ((1, 2, 7),), (3,), "vocabulary"),
        (SFTShapeProfile((4, 2)), ((1, 2, 3),), (3,), "profile mismatch"),
        (SFTShapeProfile((4, 2)), ((1, 2),) * 9, (2,) * 9, "profile mismatch"),
    ),
)
def test_invalid_batch_is_rejected_on_cpu_before_it_can_discard_accumulation(
    sft_profile: SFTShapeProfile | None,
    rows: tuple[tuple[int, ...], ...],
    lengths: tuple[int, ...],
    message: str,
) -> None:
    model = _ValidatingTinyCausalLM()
    adapters = model.lora_adapters
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(tuple(name for name, _ in adapters.named_parameters())),
            AdamWConfig(),
            _padded_batching(),
            sft_profile=sft_profile,
        ),
        trainables=model,
        checks=True,
    )
    batch = _combined_batch()
    forward_backward(runner, batch)
    gradients = tuple(parameter.grad.clone() for parameter in tuple(model.trained.masters.values()))

    with pytest.raises(TrainingInputError, match=message):
        runner.forward_backward(_batch(rows, lengths))

    for parameter, gradient in zip(tuple(model.trained.masters.values()), gradients, strict=True):
        assert torch.equal(parameter.grad, gradient)
    forward_backward(runner, batch)
    optim_step(runner, 2 * batch.num_targets)


def test_profile_padding_past_the_microbatch_token_bound_is_refused_at_construction() -> None:
    model = _ValidatingTinyCausalLM()
    plan = TrainablePlan(tuple(name for name, _ in model.lora_adapters.named_parameters()))

    with pytest.raises(ValueError, match="padding exceeds the microbatch token bound"):
        host_runner(
            model,
            TrainingConfig(
                plan,
                AdamWConfig(),
                TrainingBatching(
                    BatchLayout.PADDED, StaticBatching(microbatch_size=2, max_tokens_per_sample=4)
                ),
                sft_profile=SFTShapeProfile((4, 2), padded_width=6),
            ),
            trainables=model,
            checks=True,
        )


class _InputRefusingTinyCausalLM(_TinyCausalLM):
    """Refuses token 6 anywhere in a batch, like a model whose inputs need a tower."""

    def __init__(self) -> None:
        super().__init__()
        self.validated: list[torch.Tensor] = []
        self.executed = 0

    def validate_training_inputs(self, token_ids: torch.Tensor) -> None:
        assert token_ids.device.type == "cpu"
        self.validated.append(token_ids)
        if bool((token_ids == 6).any()):
            raise ValueError("token 6 is not trainable")

    def _training_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.executed += 1
        return super()._training_logits(input_ids)


@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
def test_runner_validates_every_input_token_before_any_microbatch(packed: bool) -> None:
    model = _InputRefusingTinyCausalLM()
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(tuple(name for name, _ in model.lora_adapters.named_parameters())),
            AdamWConfig(),
            _packed_batching(microbatch_size=1) if packed else _padded_batching(),
        ),
        trainables=model,
        checks=True,
    )
    refused = _batch(((1, 2, 3, 4), (6, 5, 0, 0)), (4, 2))
    accepted = _combined_batch()
    if packed:
        refused = _packed_batch((1, 2, 3, 4, 6, 5), (0, 4, 6))
        accepted = _packed_batch((1, 2, 3, 4, 4, 5), (0, 4, 6))

    # The refused token is an input only, never a target, and the first microbatch
    # alone would not reach it.
    with pytest.raises(TrainingInputError, match="token 6 is not trainable"):
        forward_backward(runner, refused)
    assert model.executed == 0
    assert model.validated[-1] is refused.tokens.token_ids

    forward_backward(runner, accepted)
    assert model.validated[-1] is accepted.tokens.token_ids
    assert model.executed > 0


def test_packed_batch_is_rejected_before_model_execution_when_not_enabled() -> None:
    model, adapters, runner = _runner()
    batch = _packed_batch((1, 2, 3), (0, 3))

    with pytest.raises(ValueError, match="does not match configured padded training layout"):
        forward_backward(runner, batch)

    assert all(parameter.grad is None for parameter in tuple(model.trained.masters.values()))
    forward_backward(runner, _combined_batch())


def test_padded_runner_passes_compact_targets_to_the_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _adapters, runner = _runner()
    observed: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    loss = model.training_loss

    def observe_loss(hidden_states: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        observed.append((tuple(hidden_states.shape), tuple(target_ids.shape)))
        return loss(hidden_states, target_ids)

    monkeypatch.setattr(model, "training_loss", observe_loss)

    result = forward_backward(runner, _combined_batch())

    assert result.normalization_count == 4
    assert result.loss_sum.shape == ()
    # Only the 4 target positions' hidden states of the 2 x 3 padded inputs reach the loss.
    assert observed == [((4, 7), (4,))]


def _adamw(runner: TrainingRunner) -> dict:
    """The runner's AdamW ``state_dict()``, from a checkpoint of its settled state."""

    return runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer.adamw


def _never_stepped(runner: TrainingRunner) -> bool:
    return all(
        state["step"].item() == 0 and not state["exp_avg"].any() and not state["exp_avg_sq"].any()
        for state in _adamw(runner)["state"].values()
    )


def _clone_optimizer_state(
    optimizer: torch.optim.AdamW,
    parameters: tuple[torch.Tensor, ...],
) -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(
        {name: value.clone() for name, value in optimizer.state[parameter].items()}
        for parameter in parameters
    )


def _runner_state(runner: TrainingRunner) -> tuple[dict[str, torch.Tensor], ...]:
    """The runner's AdamW state per parameter, in parameter order."""

    state = _adamw(runner)["state"]
    return tuple(state[index] for index in range(len(state)))


def _assert_optimizer_state_exact(
    actual: tuple[dict[str, torch.Tensor], ...],
    expected: tuple[dict[str, torch.Tensor], ...],
) -> None:
    for actual_state, expected_state in zip(actual, expected, strict=True):
        assert expected_state.keys() == {"step", "exp_avg", "exp_avg_sq"}
        assert actual_state.keys() == expected_state.keys()
        for name, expected_value in expected_state.items():
            torch.testing.assert_close(actual_state[name], expected_value, rtol=0, atol=0)


def _assert_tensors_exact(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def _gradient_snapshot(parameters: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    gradients = []
    for parameter in parameters:
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert parameter.grad.device.type == "cpu"
        assert bool(torch.all(torch.isfinite(parameter.grad)))
        gradients.append(parameter.grad.detach().clone())
    return tuple(gradients)


def _install_gradient_snapshot(
    parameters: tuple[torch.Tensor, ...],
    gradients: tuple[torch.Tensor, ...],
) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.clone()


def _reference_optim_step(
    parameters: tuple[torch.Tensor, ...],
    optimizer: torch.optim.AdamW,
    *,
    num_target_tokens: int,
    max_grad_norm: float | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
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
    adamw_input = _gradient_snapshot(parameters)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return normalized_norm, adamw_input


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"lr": 0.0}, ValueError, "lr"),
        ({"lr": math.inf}, ValueError, "lr"),
        ({"betas": (-0.1, 0.9)}, ValueError, "beta"),
        ({"betas": (0.9, 1.0)}, ValueError, "beta"),
        ({"eps": 0.0}, ValueError, "eps"),
        ({"weight_decay": -0.1}, ValueError, "weight_decay"),
        ({"max_grad_norm": 0.0}, ValueError, "max_grad_norm"),
        ({"max_grad_norm": math.nan}, ValueError, "max_grad_norm"),
    ],
)
def test_adamw_config_rejects_invalid_values(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        AdamWConfig(**kwargs)


def test_cuda_foreach_parameter_groups_preserve_order_and_bound_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(optimizer_module, "_CUDA_FOREACH_GROUP_TARGET_BYTES", 16)
    parameters = tuple(
        torch.empty(elements, dtype=torch.float32, device="meta") for elements in (2, 2, 1, 5, 1)
    )

    groups = optimizer_module._cuda_foreach_parameter_groups(parameters)

    assert tuple(map(len, groups)) == (2, 1, 1, 1)
    flattened = tuple(parameter for group in groups for parameter in group)
    assert all(actual is expected for actual, expected in zip(flattened, parameters, strict=True))


@pytest.mark.parametrize(
    "max_grad_norm",
    [0.1, None],
    ids=["clipped", "unclipped"],
)
def test_optimizer_commit_matches_same_gradient_adamw_trajectory(
    max_grad_norm: float | None,
) -> None:
    config = AdamWConfig(
        lr=0.03,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.02,
        max_grad_norm=max_grad_norm,
    )
    model, adapters, runner = _runner(config=config)
    named_parameters = adapters.named_parameters()
    assert tuple(name for name, _ in named_parameters) == (
        "projection.full.lora_A",
        "projection.full.lora_B",
    )
    parameters = tuple(model.trained.masters.values())
    reference_parameters = tuple(
        torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters
    )
    _assert_tensors_exact(parameters, reference_parameters)
    reference_optimizer = torch.optim.AdamW(
        reference_parameters,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        foreach=False,
        fused=False,
    )
    live_parameters = tuple(parameter for _, parameter in named_parameters)
    live_ids = tuple(map(id, live_parameters))
    live_pointers = tuple(parameter.data_ptr() for parameter in live_parameters)
    parameter_ids = tuple(map(id, parameters))
    storage_pointers = tuple(parameter.untyped_storage().data_ptr() for parameter in parameters)
    previous_raw_snapshot: tuple[torch.Tensor, ...] | None = None
    production_adamw_inputs: list[tuple[torch.Tensor, ...]] = []

    parameter_groups = _adamw(runner)["param_groups"]
    assert len(parameter_groups) == 1
    parameter_group = parameter_groups[0]
    assert parameter_group["params"] == list(range(len(parameters)))
    assert {
        name: parameter_group[name]
        for name in ("lr", "betas", "eps", "weight_decay", "foreach", "fused")
    } == {
        "lr": config.lr,
        "betas": config.betas,
        "eps": config.eps,
        "weight_decay": config.weight_decay,
        "foreach": False,
        "fused": False,
    }

    def capture_adamw_input(
        optimizer: torch.optim.Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        if optimizer is not reference_optimizer:
            production_adamw_inputs.append(_gradient_snapshot(parameters))

    hook = register_optimizer_step_pre_hook(capture_adamw_input)
    try:
        for expected_step, batch in enumerate(_optimizer_batches(), start=1):
            live_before = tuple(parameter.detach().clone() for _, parameter in named_parameters)
            values_before = tuple(parameter.detach().clone() for parameter in parameters)
            forward_backward(runner, batch)
            assert all(
                torch.equal(parameter, before)
                for parameter, before in zip(parameters, values_before, strict=True)
            )

            raw_snapshot = _gradient_snapshot(parameters)
            if previous_raw_snapshot is not None:
                assert any(
                    not torch.equal(actual, previous)
                    for actual, previous in zip(
                        raw_snapshot,
                        previous_raw_snapshot,
                        strict=True,
                    )
                )
            previous_raw_snapshot = raw_snapshot

            _install_gradient_snapshot(parameters, raw_snapshot)
            _install_gradient_snapshot(reference_parameters, raw_snapshot)
            _assert_tensors_exact(_gradient_snapshot(parameters), raw_snapshot)
            _assert_tensors_exact(_gradient_snapshot(reference_parameters), raw_snapshot)
            for production, reference in zip(parameters, reference_parameters, strict=True):
                assert production.grad is not None
                assert reference.grad is not None
                assert (
                    production.grad.untyped_storage().data_ptr()
                    != reference.grad.untyped_storage().data_ptr()
                )

            reference_norm, reference_adamw_input = _reference_optim_step(
                reference_parameters,
                reference_optimizer,
                num_target_tokens=batch.num_targets,
                max_grad_norm=max_grad_norm,
            )
            grad_norm = optim_step(runner, batch.num_targets)
            _assert_tensors_exact(tuple(p for _, p in named_parameters), live_before)
            runner.publish()
            _assert_tensors_exact(tuple(p for _, p in named_parameters), reference_parameters)

            assert len(production_adamw_inputs) == 1
            _assert_tensors_exact(production_adamw_inputs.pop(), reference_adamw_input)
            assert grad_norm == float(reference_norm)
            if max_grad_norm is not None:
                assert grad_norm > max_grad_norm

            _assert_tensors_exact(parameters, reference_parameters)
            for actual, expected, before in zip(
                parameters,
                reference_parameters,
                values_before,
                strict=True,
            ):
                torch.testing.assert_close(
                    actual - before,
                    expected - before,
                    rtol=0,
                    atol=0,
                )
            expected_state = _clone_optimizer_state(
                reference_optimizer,
                reference_parameters,
            )
            assert all(int(state["step"].item()) == expected_step for state in expected_state)
            _assert_optimizer_state_exact(_runner_state(runner), expected_state)
            assert tuple(map(id, live_parameters)) == live_ids
            assert tuple(parameter.data_ptr() for parameter in live_parameters) == live_pointers
            assert all(parameter.grad is None for parameter in live_parameters)
            assert tuple(map(id, parameters)) == parameter_ids
            assert (
                tuple(parameter.untyped_storage().data_ptr() for parameter in parameters)
                == storage_pointers
            )
            assert all(parameter.grad is None for parameter in parameters)
    finally:
        hook.remove()
    assert all(tensor.grad is None for tensor in model.state_dict().values())


def test_commit_is_invariant_to_gradient_accumulation_partition() -> None:
    _, combined_adapters, combined = _runner()
    _, partitioned_adapters, partitioned = _runner()
    partitioned_adapters.load_state_dict(combined_adapters.state_dict())
    synchronize_optimizer_parameters(partitioned, partitioned_adapters.named_parameters())

    combined_result = forward_backward(combined, _combined_batch())
    combined_norm = optim_step(combined, combined_result.normalization_count)
    partition_results = [forward_backward(partitioned, batch) for batch in _partitions()]
    partitioned_tokens = sum(result.normalization_count for result in partition_results)
    partitioned_norm = optim_step(partitioned, partitioned_tokens)
    combined.publish()
    partitioned.publish()

    assert sum(result.normalization_count for result in partition_results) == (
        combined_result.normalization_count
    )
    assert partitioned_tokens == combined_result.normalization_count
    assert partitioned_norm == pytest.approx(combined_norm, rel=1e-6)
    for actual, expected in zip(
        tuple(t for _, t in partitioned_adapters.named_parameters()),
        tuple(t for _, t in combined_adapters.named_parameters()),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_pending_accumulation_guard_checks_all_model_gradients() -> None:
    model, _adapters, runner = _runner()
    batch = _combined_batch()

    assert not runner.holds_gradients

    model.projection.weight.grad = torch.ones_like(model.projection.weight)
    assert runner.holds_gradients
    runner.discard()
    assert not runner.holds_gradients

    result = forward_backward(runner, batch)
    assert runner.holds_gradients
    optim_step(runner, result.normalization_count)
    assert not runner.holds_gradients


def _remove_gradient(parameter: torch.Tensor) -> None:
    parameter.grad = None


def _write_nan(parameter: torch.Tensor) -> None:
    assert parameter.grad is not None
    parameter.grad.flatten()[0] = math.nan


def _write_inf(parameter: torch.Tensor) -> None:
    assert parameter.grad is not None
    parameter.grad.flatten()[0] = math.inf


@pytest.mark.parametrize("parameter_index", [0, -1])
@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_remove_gradient, "missing"),
        (_write_nan, "finite"),
        (_write_inf, "finite"),
    ],
)
def test_invalid_gradient_aborts_without_parameter_or_optimizer_state_mutation(
    parameter_index: int,
    corrupt: Callable[[torch.Tensor], None],
    message: str,
) -> None:
    model, adapters, runner = _runner()
    batch = _combined_batch()
    parameters = tuple(model.trained.masters.values())
    live_parameters = tuple(parameter for _, parameter in adapters.named_parameters())
    live_before = tuple(parameter.detach().clone() for parameter in live_parameters)
    pointers = tuple(parameter.data_ptr() for parameter in parameters)
    before = tuple(parameter.clone() for parameter in parameters)
    forward_backward(runner, batch)
    corrupt(parameters[parameter_index])

    with pytest.raises(InvalidGradientError, match=message):
        optim_step(runner, batch.num_targets)

    _assert_tensors_exact(live_parameters, live_before)
    assert tuple(parameter.data_ptr() for parameter in parameters) == pointers
    assert all(
        torch.equal(parameter, expected)
        for parameter, expected in zip(parameters, before, strict=True)
    )
    assert _never_stepped(runner)
    assert all(parameter.grad is None for parameter in parameters)
    with pytest.raises(InvalidGradientError, match="missing"):
        optim_step(runner, batch.num_targets)

    forward_backward(runner, batch)
    optim_step(runner, batch.num_targets)


def test_optimizer_exception_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, runner = _runner()
    batch = _combined_batch()
    forward_backward(runner, batch)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(torch.optim.AdamW, "step", fail)
    with pytest.raises(RuntimeError, match="injected optimizer failure") as raised:
        optim_step(runner, batch.num_targets)
    assert not isinstance(raised.value, Refusal)


def test_invalid_gradient_preserves_existing_adamw_moments() -> None:
    model, adapters, runner = _runner()
    parameters = tuple(model.trained.masters.values())
    live_parameters = tuple(parameter for _, parameter in adapters.named_parameters())
    live_before = tuple(parameter.detach().clone() for parameter in live_parameters)
    batch = _combined_batch()
    forward_backward(runner, batch)
    optim_step(runner, batch.num_targets)

    before_state = _runner_state(runner)
    forward_backward(runner, batch)
    before_parameters = tuple(parameter.clone() for parameter in parameters)
    _write_nan(parameters[-1])

    with pytest.raises(InvalidGradientError, match="finite"):
        optim_step(runner, batch.num_targets)

    _assert_tensors_exact(live_parameters, live_before)
    for actual, expected in zip(parameters, before_parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_optimizer_state_exact(_runner_state(runner), before_state)
    forward_backward(runner, batch)
    optim_step(runner, batch.num_targets)


def test_post_step_cleanup_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, adapters, runner = _runner()
    parameters = tuple(model.trained.masters.values())
    live_parameters = tuple(parameter for _, parameter in adapters.named_parameters())
    live_before = tuple(parameter.detach().clone() for parameter in live_parameters)
    before = tuple(parameter.clone() for parameter in parameters)
    batch = _combined_batch()
    forward_backward(runner, batch)

    def fail_zero_grad(_optimizer: object, *, set_to_none: bool) -> None:
        raise RuntimeError(f"injected zero_grad failure: {set_to_none}")

    monkeypatch.setattr(torch.optim.AdamW, "zero_grad", fail_zero_grad)
    with pytest.raises(RuntimeError, match="injected zero_grad failure"):
        optim_step(runner, batch.num_targets)

    _assert_tensors_exact(live_parameters, live_before)
    assert any(
        not torch.equal(parameter, expected)
        for parameter, expected in zip(parameters, before, strict=True)
    )


def test_failed_forward_backward_fails_an_existing_accumulation(monkeypatch):
    model, adapters, runner = _runner()
    batch = _combined_batch()
    forward_backward(runner, batch)
    failure = torch.OutOfMemoryError("injected OOM")

    def fail_forward(*args, **kwargs):
        raise failure

    monkeypatch.setattr(model, "training_loss", fail_forward)
    with pytest.raises(RuntimeError, match="accumulation window") as raised:
        forward_backward(runner, batch)
    assert not isinstance(raised.value, Refusal)
    assert raised.value.__cause__ is failure
    assert all(parameter.grad is None for parameter in tuple(model.trained.masters.values()))


def test_internal_sft_microbatches_match_one_logical_batch() -> None:
    config = AdamWConfig(
        lr=0.03,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.02,
        max_grad_norm=None,
    )

    def create(microbatch_size: int) -> tuple[Adapter, TrainingRunner]:
        model = _TinyCausalLM()
        adapters = model.lora_adapters
        runner = host_runner(
            model,
            TrainingConfig(
                TrainablePlan(tuple(name for name, _ in adapters.named_parameters())),
                config,
                _padded_batching(microbatch_size=microbatch_size),
            ),
            trainables=model,
            checks=True,
        )
        return adapters, runner

    full_adapters, full_runner = create(microbatch_size=2)
    split_adapters, split_runner = create(microbatch_size=1)
    batch = _combined_batch()

    full_result = forward_backward(full_runner, batch)
    split_result = forward_backward(split_runner, batch)
    full_commit = full_runner.commit(batch.num_targets, (full_result,))
    split_commit = split_runner.commit(batch.num_targets, (split_result,))
    full_runner.publish()
    split_runner.publish()

    torch.testing.assert_close(split_result.loss_sum, full_result.loss_sum)
    assert split_commit.grad_norm == pytest.approx(full_commit.grad_norm, rel=1e-6)
    for actual, expected in zip(
        tuple(t for _, t in split_adapters.named_parameters()),
        tuple(t for _, t in full_adapters.named_parameters()),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_internal_packed_microbatching_preserves_sft_update() -> None:
    config = AdamWConfig(
        lr=0.03,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.02,
        max_grad_norm=None,
    )

    def create(microbatch_size: int) -> tuple[Adapter, TrainingRunner]:
        model = _TinyCausalLM()
        adapters = model.lora_adapters
        runner = host_runner(
            model,
            TrainingConfig(
                TrainablePlan(tuple(name for name, _ in adapters.named_parameters())),
                config,
                _packed_batching(microbatch_size=microbatch_size),
            ),
            trainables=model,
            checks=True,
        )
        return adapters, runner

    full_adapters, full_runner = create(microbatch_size=2)
    split_adapters, split_runner = create(microbatch_size=1)
    batch = _packed_combined_batch()

    full_result = forward_backward(full_runner, batch)
    split_result = forward_backward(split_runner, batch)
    full_commit = full_runner.commit(batch.num_targets, (full_result,))
    split_commit = split_runner.commit(batch.num_targets, (split_result,))
    full_runner.publish()
    split_runner.publish()

    torch.testing.assert_close(split_result.loss_sum, full_result.loss_sum)
    assert split_commit.grad_norm == pytest.approx(full_commit.grad_norm, rel=1e-6)
    for actual, expected in zip(
        tuple(t for _, t in split_adapters.named_parameters()),
        tuple(t for _, t in full_adapters.named_parameters()),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_aborted_accumulation_cleanup_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, adapters, runner = _runner()
    parameters = tuple(model.trained.masters.values())
    batch = _combined_batch()
    forward_backward(runner, batch)
    _write_nan(parameters[-1])

    def fail_zero_grad(_optimizer: object, *, set_to_none: bool) -> None:
        parameters[0].grad = None
        raise RuntimeError(f"injected discard failure: {set_to_none}")

    monkeypatch.setattr(torch.optim.AdamW, "zero_grad", fail_zero_grad)
    with pytest.raises(RuntimeError, match="injected discard failure"):
        optim_step(runner, batch.num_targets)


@pytest.mark.parametrize("location", ["first", "prompt", "padding", "unused_pad"])
@pytest.mark.parametrize("token", [-1, 7])
def test_runner_rejects_non_target_token_ids_before_execution(location, token, monkeypatch):
    from types import SimpleNamespace

    from cti.layers.embedding import ParallelLMHead

    model = _ValidatingTinyCausalLM()
    # Use the production CPU range validator with a tiny vocabulary.
    model.validate_training_targets = lambda ids: ParallelLMHead.validate_training_targets(
        SimpleNamespace(num_embeddings=7), ids
    )
    adapters = model.lora_adapters
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(tuple(name for name, _ in adapters.named_parameters())),
            AdamWConfig(),
            _padded_batching(),
        ),
        trainables=model,
        checks=True,
    )
    tokens = torch.tensor([[1, 2, 3, 0]])
    pad = token if location in ("padding", "unused_pad") else 0
    if location == "first":
        tokens[0, 0] = token
    elif location == "prompt":
        tokens[0, 1] = token
    elif location == "padding":
        tokens[0, 3] = token
    length = 4 if location == "unused_pad" else 3

    def unexpected_execution(*args, **kwargs):
        pytest.fail("invalid token reached model execution")

    monkeypatch.setattr(model, "training_hidden_states", unexpected_execution)
    with pytest.raises((TrainingInputError, ValueError), match="(observed range|pad_token_id)"):
        batch = _padded_batch(tokens, torch.tensor([length]), [[False, True, False]], pad)
        forward_backward(runner, batch)
    assert all(parameter.grad is None for parameter in tuple(model.trained.masters.values()))
