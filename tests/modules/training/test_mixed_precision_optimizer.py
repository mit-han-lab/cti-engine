from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

import pytest
import torch
import torch.nn.functional as F

from cti.config import (
    AdamWConfig,
    BatchLayout,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
    TrainingStateResidency,
)
from cti.training import (
    SFT,
    InvalidGradientError,
    PaddedTokens,
    TrainingBatch,
)
from cti.training._checkpoint import OptimizerState
from cti.training._errors import TrainingResidencyError
from cti.training._optimizer import AdamWOptimizer
from cti.training.runner import TrainingRunner
from tests.support.training import (
    TrainingModelStub,
    assert_optimizer_state_exact,
    forward_backward,
    host_runner,
    optim_step,
)

_CPU = torch.device("cpu")


def _padded_batching() -> TrainingBatching:
    return TrainingBatching(
        BatchLayout.PADDED,
        StaticBatching(microbatch_size=8, max_tokens_per_sample=64),
    )


class _MixedPrecisionCausalLM(TrainingModelStub):
    """Small ordinary model with both supported selected-parameter policies."""

    def __init__(self, device: torch.device = _CPU) -> None:
        generator = torch.Generator().manual_seed(20260717)
        self.selected_fp32 = torch.nn.Parameter(
            torch.randn(7, 4, generator=generator, dtype=torch.float32).to(device),
            requires_grad=False,
        )
        self.selected_bf16 = torch.nn.Parameter(
            torch.randn(7, 4, generator=generator, dtype=torch.bfloat16).to(device),
            requires_grad=False,
        )
        self.frozen = torch.nn.Parameter(
            torch.randn(7, 4, generator=generator, dtype=torch.float32).to(device),
            requires_grad=False,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "selected_fp32": self.selected_fp32,
            "selected_bf16": self.selected_bf16,
            "frozen": self.frozen,
        }

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("selected_fp32", self.selected_fp32),
            ("selected_bf16", self.selected_bf16),
        )

    def forward_training(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = F.one_hot(input_ids.remainder(4), num_classes=4)
        fp32_features = features.to(torch.float32)
        bf16_features = features.to(torch.bfloat16)
        return (
            F.linear(fp32_features, self.trained(self.selected_fp32))
            + F.linear(bf16_features, self.trained(self.selected_bf16)).float()
            + F.linear(fp32_features, self.frozen)
        )


def _batch(rows: tuple[tuple[int, ...], ...], lengths: tuple[int, ...]) -> TrainingBatch:
    tokens = PaddedTokens(
        torch.tensor(rows, dtype=torch.int64), torch.tensor(lengths, dtype=torch.int64), 0
    )
    return TrainingBatch(tokens, tokens.valid_targets(), SFT())


def _microbatches() -> tuple[TrainingBatch, ...]:
    return (
        _batch(((1, 2, 3, 4),), (4,)),
        _batch(((4, 5, 0), (2, 6, 1)), (2, 3)),
        _batch(((6, 3, 2, 5, 1),), (5,)),
    )


def _runner(
    *,
    device: torch.device = _CPU,
    persistent_allocation: Callable[[], AbstractContextManager[None]] = nullcontext,
) -> tuple[_MixedPrecisionCausalLM, TrainingRunner, AdamWConfig]:
    model = _MixedPrecisionCausalLM(device)
    config = AdamWConfig(
        lr=0.03,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.02,
        max_grad_norm=0.25,
    )
    return (
        model,
        host_runner(
            model,
            TrainingConfig(
                TrainablePlan(("selected_fp32", "selected_bf16")), config, _padded_batching()
            ),
            trainables=model,
            persistent_allocation=persistent_allocation,
            checks=True,
        ),
        config,
    )


def _state(runner: TrainingRunner) -> OptimizerState:
    """The runner's settled optimizer state, through a checkpoint."""

    return runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer


def _never_stepped(runner: TrainingRunner) -> bool:
    return all(
        state["step"].item() == 0 and not state["exp_avg"].any() and not state["exp_avg_sq"].any()
        for state in _state(runner).adamw["state"].values()
    )


def _adamw_state(runner: TrainingRunner) -> tuple[dict[str, torch.Tensor], ...]:
    state = _state(runner).adamw["state"]
    return tuple(state[index] for index in range(len(state)))


def _reference_step(
    parameters: tuple[torch.nn.Parameter, ...],
    optimizer: torch.optim.AdamW,
    gradients: tuple[torch.Tensor, ...],
    *,
    num_target_tokens: int,
    max_grad_norm: float | None,
) -> float:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.clone()
    raw_norm = torch.nn.utils.get_total_norm(
        gradients,
        norm_type=2.0,
        error_if_nonfinite=True,
        foreach=False,
    )
    normalized_norm = raw_norm / num_target_tokens
    with torch.no_grad():
        for parameter in parameters:
            assert parameter.grad is not None
            parameter.grad.div_(num_target_tokens)
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


def _clone_optimizer_state(
    optimizer: torch.optim.AdamW,
    parameters: tuple[torch.Tensor, ...],
) -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(
        {name: value.clone() for name, value in optimizer.state[parameter].items()}
        for parameter in parameters
    )


def test_mixed_precision_optimizer_matches_fp32_master_oracle() -> None:
    model, runner, config = _runner()
    initial = _state(runner)
    for master, live in zip(initial.master, initial.live, strict=True):
        assert master.dtype == torch.float32
        torch.testing.assert_close(master, live.float(), rtol=0, atol=0)
    for name, live in model.named_trainable_parameters():
        master = model.trained.masters[name]
        assert master.is_leaf and master.data_ptr() != live.data_ptr()

    reference_parameters = tuple(torch.nn.Parameter(master.clone()) for master in initial.master)
    reference_optimizer = torch.optim.AdamW(
        reference_parameters,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        foreach=False,
        fused=False,
    )
    live_ids = (id(model.selected_fp32), id(model.selected_bf16))
    live_pointers = (model.selected_fp32.data_ptr(), model.selected_bf16.data_ptr())
    observed_bf16_gradients: list[torch.Tensor] = []

    def observe_bf16_gradient(gradient: torch.Tensor) -> None:
        assert gradient.dtype == torch.float32
        observed_bf16_gradients.append(gradient.detach().clone())

    hook = model.trained.masters["selected_bf16"].register_hook(observe_bf16_gradient)
    try:
        windows = (_microbatches()[:2], _microbatches()[2:])
        for expected_step, microbatches in enumerate(windows, start=1):
            observed_bf16_gradients.clear()
            live_before = (
                model.selected_fp32.detach().clone(),
                model.selected_bf16.detach().clone(),
            )
            results = [forward_backward(runner, batch) for batch in microbatches]
            num_target_tokens = sum(result.normalization_count for result in results)

            assert len(observed_bf16_gradients) == len(microbatches)
            expected_main_gradient = observed_bf16_gradients[0].float()
            for gradient in observed_bf16_gradients[1:]:
                expected_main_gradient.add_(gradient)
            assert model.selected_bf16.grad is None
            fp32_gradient, master_gradient = runner.accumulated_gradients()
            torch.testing.assert_close(master_gradient, expected_main_gradient, rtol=0, atol=0)
            assert fp32_gradient.dtype == torch.float32
            assert torch.equal(model.selected_fp32, live_before[0])
            assert torch.equal(model.selected_bf16, live_before[1])
            assert runner.holds_gradients

            gradients = (fp32_gradient, master_gradient)
            expected_norm = _reference_step(
                reference_parameters,
                reference_optimizer,
                gradients,
                num_target_tokens=num_target_tokens,
                max_grad_norm=config.max_grad_norm,
            )
            actual_norm = optim_step(runner, num_target_tokens)

            assert actual_norm == expected_norm
            assert torch.equal(model.selected_fp32, live_before[0])
            assert torch.equal(model.selected_bf16, live_before[1])
            runner.publish()
            stepped = _state(runner)
            torch.testing.assert_close(stepped.master[0], reference_parameters[0], rtol=0, atol=0)
            torch.testing.assert_close(
                model.selected_fp32,
                reference_parameters[0],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                stepped.master[1],
                reference_parameters[1].detach().cpu(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                model.selected_bf16,
                reference_parameters[1].to(torch.bfloat16),
                rtol=0,
                atol=0,
            )
            expected = _clone_optimizer_state(reference_optimizer, reference_parameters)
            actual = _adamw_state(runner)
            for actual_state, expected_state in zip(actual, expected, strict=True):
                assert actual_state.keys() == expected_state.keys()
                for name, expected_value in expected_state.items():
                    torch.testing.assert_close(actual_state[name], expected_value, rtol=0, atol=0)
            assert all(int(state["step"].item()) == expected_step for state in actual)
            assert (id(model.selected_fp32), id(model.selected_bf16)) == live_ids
            assert (model.selected_fp32.data_ptr(), model.selected_bf16.data_ptr()) == (
                live_pointers
            )
            assert model.selected_fp32.grad is None
            assert model.selected_bf16.grad is None
            assert not runner.holds_gradients
    finally:
        hook.remove()

    assert not torch.equal(_state(runner).master[1], model.selected_bf16.float().cpu())
    assert model.frozen.grad is None


def test_discard_clears_master_live_and_complement_gradients() -> None:
    model, runner, _ = _runner()
    before = tuple(tensor.detach().clone() for tensor in model.state_dict().values())
    forward_backward(runner, _microbatches()[0])
    model.frozen.grad = torch.ones_like(model.frozen)

    assert model.trained.masters["selected_fp32"].grad is not None
    assert len(runner.accumulated_gradients()) == 2
    assert model.selected_bf16.grad is None
    assert runner.holds_gradients

    runner.discard()

    assert not runner.holds_gradients
    assert all(master.grad is None for master in model.trained.masters.values())
    assert model.selected_fp32.grad is None
    assert model.selected_bf16.grad is None
    assert model.frozen.grad is None
    assert _never_stepped(runner)
    for actual, expected in zip(model.state_dict().values(), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _BackwardProbe(torch.autograd.Function):
    """Identity whose backward reports whether autograd may run it on a device thread."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: torch.Tensor,
        report: Callable[[bool], None],
    ) -> torch.Tensor:
        ctx.report = report
        return logits.view_as(logits)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        ctx.report(torch.autograd.is_multithreading_enabled())
        return gradient, None


def _probe_backward(
    monkeypatch: pytest.MonkeyPatch,
    model: _MixedPrecisionCausalLM,
    report: Callable[[bool], None],
) -> None:
    forward_training = model.forward_training
    monkeypatch.setattr(
        model,
        "forward_training",
        lambda input_ids: _BackwardProbe.apply(forward_training(input_ids), report),
    )


def test_backward_runs_on_the_calling_thread_and_accumulates_in_the_persistent_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    master_gradients: list[torch.Tensor | None] = []

    @contextmanager
    def persistent_allocation() -> Iterator[None]:
        events.append(f"persistent+ multithreading={torch.autograd.is_multithreading_enabled()}")
        master_gradients.append(model.trained.masters["selected_bf16"].grad)
        try:
            yield
        finally:
            master_gradients.append(model.trained.masters["selected_bf16"].grad)
            events.append("persistent-")

    model, runner, _ = _runner(persistent_allocation=persistent_allocation)
    _probe_backward(
        monkeypatch,
        model,
        lambda multithreading: events.append(f"backward multithreading={multithreading}"),
    )

    for batch in _microbatches()[:2]:
        forward_backward(runner, batch)

    # Only backward is single-threaded. Its FP32 master gradient moves into the
    # persistent scope once; subsequent microbatches accumulate there in place.
    microbatch_events = [
        "backward multithreading=False",
        "persistent+ multithreading=True",
        "persistent-",
    ]
    assert events == microbatch_events * 2
    assert [gradient is not None for gradient in master_gradients] == [True, True] * 2
    assert master_gradients[0].data_ptr() != master_gradients[1].data_ptr()
    assert master_gradients[1] is master_gradients[2] is master_gradients[3]
    assert len(runner.accumulated_gradients()) == 2
    assert torch.autograd.is_multithreading_enabled()


def test_backward_failure_restores_multithreaded_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, runner, _ = _runner()
    reports: list[bool] = []

    def fail_backward(multithreading: bool) -> None:
        reports.append(multithreading)
        raise RuntimeError("injected backward failure")

    _probe_backward(monkeypatch, model, fail_backward)
    with pytest.raises(RuntimeError, match="injected backward failure"):
        forward_backward(runner, _microbatches()[0])

    assert reports == [False]
    assert torch.autograd.is_multithreading_enabled()
    assert not runner.holds_gradients


def test_failed_forward_discards_prior_mixed_precision_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, runner, _ = _runner()
    batch = _microbatches()[0]
    forward_backward(runner, batch)

    def fail_forward(_hidden_states: torch.Tensor, _target_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(model, "training_loss", fail_forward)
    with pytest.raises(RuntimeError, match="accumulation window"):
        forward_backward(runner, batch)

    assert not runner.holds_gradients
    assert all(master.grad is None for master in model.trained.masters.values())
    assert model.selected_fp32.grad is None
    assert model.selected_bf16.grad is None
    assert _never_stepped(runner)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda parameter: setattr(parameter, "grad", None), "missing"),
        (
            lambda parameter: parameter.grad.flatten().__setitem__(0, math.nan),
            "finite",
        ),
    ],
    ids=("missing", "nonfinite"),
)
def test_invalid_main_gradient_preserves_live_master_and_optimizer_state(
    corrupt: Callable[[torch.Tensor], None],
    message: str,
) -> None:
    model, runner, _ = _runner()
    first = forward_backward(runner, _microbatches()[0])
    optim_step(runner, first.num_targets)

    settled = _state(runner)
    forward_backward(runner, _microbatches()[1])
    live_before = (model.selected_fp32.clone(), model.selected_bf16.clone())
    # The FP32 live parameter also trains through its master.
    corrupt(model.trained.masters["selected_fp32"])

    with pytest.raises(InvalidGradientError, match=message):
        optim_step(runner, _microbatches()[1].num_targets)

    for actual, expected in zip(
        (model.selected_fp32, model.selected_bf16),
        live_before,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    after = _state(runner)
    for actual, expected in zip(after.master, settled.master, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for index, expected in settled.adamw["state"].items():
        for name in expected:
            torch.testing.assert_close(
                after.adamw["state"][index][name], expected[name], rtol=0, atol=0
            )
    assert all(master.grad is None for master in model.trained.masters.values())
    assert model.selected_fp32.grad is None
    assert model.selected_bf16.grad is None
    assert not runner.holds_gradients


def test_master_publication_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model, runner, _ = _runner()
    result = forward_backward(runner, _microbatches()[0])

    def fail_publication(_optimizer: object) -> None:
        raise RuntimeError("injected publication failure")

    optim_step(runner, result.normalization_count)
    monkeypatch.setattr(AdamWOptimizer, "publish", fail_publication)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        runner.publish()


def test_optimizer_state_restores_exactly_either_way() -> None:
    _, runner, _ = _runner()
    assert _never_stepped(runner)
    untrained = runner.checkpoint(num_optimizer_steps=0, model_revision=0)
    first = forward_backward(runner, _microbatches()[0])
    optim_step(runner, first.num_targets)
    trained = runner.checkpoint(num_optimizer_steps=1, model_revision=0)

    runner.restore(untrained)
    assert _never_stepped(runner)
    runner.restore(trained)
    expected_states = trained.optimizer.adamw["state"]
    for index, actual in enumerate(_adamw_state(runner)):
        for name, expected in expected_states[index].items():
            torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)


def _flat_storage(shapes, device) -> list[torch.Tensor]:
    """The state as views of one flat buffer, as the offloading residencies place it."""

    sizes = [shape.numel() for shape in shapes]
    flat = torch.zeros(sum(sizes), dtype=torch.float32, device=device)
    return [view.view(shape) for view, shape in zip(flat.split(sizes), shapes, strict=True)]


def test_state_in_one_flat_buffer_trains_bit_for_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    _, reference, _ = _runner()
    materialize = AdamWOptimizer.materialize
    monkeypatch.setattr(
        AdamWOptimizer,
        "materialize",
        lambda self, **_: materialize(self, masters=_flat_storage, moments=_flat_storage),
    )
    _, flat, _ = _runner()
    storages = {
        tensor.untyped_storage().data_ptr()
        for tensors in flat._optimizer.device_state().values()
        for tensor in tensors
    }
    # The masters' buffer and the moments' buffer.
    assert len(storages) == 2

    for runner in (reference, flat):
        for batch in _microbatches():
            result = forward_backward(runner, batch)
            optim_step(runner, result.num_targets)
    assert_optimizer_state_exact(_state(flat), _state(reference))


def test_an_fp32_gradient_moves_into_the_persistent_scope_once_per_window() -> None:
    addresses: list[tuple[int, int]] = []

    @contextmanager
    def persistent_allocation() -> Iterator[None]:
        before = master_parameter.grad
        yield
        if before is not None and master_parameter.grad is not None:
            addresses.append((before.data_ptr(), master_parameter.grad.data_ptr()))

    model, runner, _ = _runner(persistent_allocation=persistent_allocation)
    master_parameter = model.trained.masters["selected_fp32"]
    for batch in _microbatches()[:2]:
        forward_backward(runner, batch)
    runner.discard()
    forward_backward(runner, _microbatches()[0])

    # The gradient autograd creates in a window's first microbatch is copied inside the
    # persistent scope; later microbatches add into that copy in place.
    assert [before != after for before, after in addresses] == [True, False, True]
    assert addresses[1][0] == addresses[0][1]


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        (("selected_fp32", "selected_bf16"), "1 of the 2 selected parameters are not BF16"),
        (("selected_bf16",), "needs a CUDA device, not cpu"),
    ],
)
def test_offload_all_takes_bf16_parameters_on_cuda_only(
    selected: tuple[str, ...], message: str
) -> None:
    model = _MixedPrecisionCausalLM()
    config = TrainingConfig(
        TrainablePlan(selected),
        AdamWConfig(lr=0.03),
        _padded_batching(),
        training_state_residency=TrainingStateResidency.OFFLOAD_ALL,
    )
    with pytest.raises(TrainingResidencyError, match=message):
        host_runner(model, config, trainables=model)
