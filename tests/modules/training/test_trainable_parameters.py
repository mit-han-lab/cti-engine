from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

import cti.training as training
from cti.config import (
    ActivationCheckpointMode,
    AdamWConfig,
    BatchLayout,
    LoRAConfig,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.layers import Adapter, BaseOP, LinearReplicated
from cti.models.activation_checkpointing import ReentrantActivationCheckpoint
from cti.training import SFT, PaddedTokens, TrainingBatch, TrainingInputError
from cti.training._parameters import TrainableParameterMetadata, resolve_for_model
from cti.training.runner import TrainingRunner
from tests.support.host_offload import FakeCuda
from tests.support.tp_shards import site
from tests.support.training import (
    DeclaredTensors,
    TrainingModelStub,
    forward_backward,
    host_runner,
    optim_step,
)


def _padded_batching() -> TrainingBatching:
    return TrainingBatching(
        BatchLayout.PADDED,
        StaticBatching(microbatch_size=8, max_tokens_per_sample=64),
    )


def _parameter(
    *shape: int,
    requires_grad: bool = False,
) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.randn(*shape), requires_grad=requires_grad)


def test_resolution_uses_canonical_universe_order_and_freezes_complement() -> None:
    frozen = _parameter(3, requires_grad=True)
    selected_a = _parameter(3)
    selected_b = _parameter(3)
    resolved = resolve_for_model(
        TrainablePlan(("selected_b", "selected_a")),
        DeclaredTensors(
            (
                ("frozen", frozen),
                ("selected_a", selected_a),
                ("selected_b", selected_b),
            )
        ),
    )

    assert resolved.selected_names == ("selected_a", "selected_b")
    assert resolved.complement_names == ("frozen",)
    assert resolved.named_parameters() == (
        ("selected_a", selected_a),
        ("selected_b", selected_b),
    )
    assert resolved.parameters() == (selected_a, selected_b)
    assert resolved.metadata == (
        TrainableParameterMetadata("selected_a", (3,), selected_a.dtype),
        TrainableParameterMetadata("selected_b", (3,), selected_b.dtype),
    )
    assert resolved.named_complement() == (("frozen", frozen),)
    assert resolved.complement() == (frozen,)
    assert selected_a.requires_grad
    assert selected_b.requires_grad
    assert not frozen.requires_grad


@pytest.mark.parametrize(
    ("selected_names", "error", "message"),
    [
        pytest.param((), ValueError, "non-empty", id="empty"),
        pytest.param(("",), ValueError, "non-empty", id="empty-name"),
        pytest.param((" selected",), ValueError, "whitespace", id="leading-space"),
        pytest.param(("selected ",), ValueError, "whitespace", id="trailing-space"),
        pytest.param(("selected", "selected"), ValueError, "unique", id="duplicate"),
    ],
)
def test_plan_rejects_invalid_selected_names(
    selected_names: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        TrainablePlan(selected_names)  # type: ignore[arg-type]


def _non_tensor_universe() -> tuple[tuple[str, object], ...]:
    return (("selected", object()),)


def _non_floating_universe() -> tuple[tuple[str, torch.Tensor], ...]:
    return (("selected", torch.ones(3, dtype=torch.int64)),)


def _non_strided_universe() -> tuple[tuple[str, torch.Tensor], ...]:
    sparse = torch.ones(3).to_sparse()
    return (("selected", sparse),)


def _non_leaf_universe() -> tuple[tuple[str, torch.Tensor], ...]:
    return (("selected", torch.ones(3, requires_grad=True) * 2),)


def _pre_gradient_universe() -> tuple[tuple[str, torch.Tensor], ...]:
    selected = _parameter(3, requires_grad=True)
    selected.grad = torch.ones_like(selected)
    return (("selected", selected),)


def _duplicate_name_universe() -> tuple[tuple[str, torch.Tensor], ...]:
    return (("selected", _parameter(3)), ("selected", _parameter(3)))


def _object_alias_universe() -> tuple[tuple[str, torch.Tensor], ...]:
    shared = _parameter(3)
    return (("selected", shared), ("alias", shared))


def _storage_alias_universe() -> tuple[tuple[str, torch.Tensor], ...]:
    shared = torch.randn(6)
    return (("selected", shared[:3]), ("alias", shared[3:]))


@pytest.mark.parametrize(
    ("universe_factory", "error", "message"),
    [
        pytest.param(_non_tensor_universe, TypeError, "torch.Tensor", id="non-tensor"),
        pytest.param(_non_floating_universe, ValueError, "floating", id="non-floating"),
        pytest.param(_non_strided_universe, ValueError, "strided", id="non-strided"),
        pytest.param(_non_leaf_universe, ValueError, "leaf", id="non-leaf"),
        pytest.param(_pre_gradient_universe, ValueError, "gradient", id="pre-gradient"),
        pytest.param(_duplicate_name_universe, ValueError, "unique", id="duplicate-name"),
        pytest.param(_object_alias_universe, ValueError, "unique", id="object-alias"),
        pytest.param(_storage_alias_universe, ValueError, "share storage", id="storage-alias"),
    ],
)
def test_resolution_rejects_invalid_universe(
    universe_factory: Callable[[], tuple[tuple[str, object], ...]],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        resolve_for_model(TrainablePlan(("selected",)), DeclaredTensors(universe_factory()))


@pytest.mark.parametrize(
    "universe",
    [
        pytest.param((("", torch.ones(1)),), id="empty-name"),
        pytest.param(((" selected", torch.ones(1)),), id="leading-space"),
        pytest.param((("selected ", torch.ones(1)),), id="trailing-space"),
        pytest.param(((1, torch.ones(1)),), id="non-string-name"),
        pytest.param((("selected", torch.ones(1), torch.ones(1)),), id="not-a-pair"),
    ],
)
def test_resolution_rejects_malformed_named_pairs(universe: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        resolve_for_model(TrainablePlan(("selected",)), DeclaredTensors(universe))  # type: ignore[arg-type]


def test_a_frozen_tensor_of_any_dtype_is_accepted_and_a_selected_one_must_float() -> None:
    # A packed quantized base weight is uint8; only what training updates must float.
    packed = torch.zeros(4, 2, dtype=torch.uint8)
    selected = _parameter(3)

    resolved = resolve_for_model(
        TrainablePlan(("selected",)),
        DeclaredTensors((("selected", selected), ("packed", packed))),
    )

    assert selected.requires_grad and not packed.requires_grad
    assert resolved.complement_names == ("packed",)
    with pytest.raises(ValueError, match="selected tensor 'packed' must have a floating dtype"):
        resolve_for_model(TrainablePlan(("packed",)), DeclaredTensors((("packed", packed),)))


def test_resolution_rejects_unknown_selection_without_mutation() -> None:
    existing = _parameter(3, requires_grad=True)
    with pytest.raises(ValueError, match="unknown"):
        resolve_for_model(TrainablePlan(("missing",)), DeclaredTensors((("existing", existing),)))
    assert existing.requires_grad


def test_failed_resolution_preserves_every_requires_grad_flag() -> None:
    would_be_selected = _parameter(3)
    would_be_frozen = _parameter(3, requires_grad=True)
    invalid = _parameter(3, requires_grad=True)
    invalid.grad = torch.ones_like(invalid)
    before = tuple(tensor.requires_grad for tensor in (would_be_selected, would_be_frozen, invalid))

    with pytest.raises(ValueError, match="gradient"):
        resolve_for_model(
            TrainablePlan(("selected",)),
            DeclaredTensors(
                (
                    ("selected", would_be_selected),
                    ("frozen", would_be_frozen),
                    ("invalid", invalid),
                )
            ),
        )

    assert (
        tuple(tensor.requires_grad for tensor in (would_be_selected, would_be_frozen, invalid))
        == before
    )


def test_inference_tensor_rejection_is_mutation_atomic() -> None:
    would_be_selected = _parameter(3)
    with torch.inference_mode():
        inference_tensor = torch.ones(3)

    with pytest.raises(ValueError, match="inference"):
        resolve_for_model(
            TrainablePlan(("selected",)),
            DeclaredTensors((("selected", would_be_selected), ("invalid", inference_tensor))),
        )

    assert not would_be_selected.requires_grad
    assert not inference_tensor.requires_grad


def test_iterable_failure_is_mutation_atomic() -> None:
    would_be_selected = _parameter(3)

    def failing_universe():
        yield "selected", would_be_selected
        raise RuntimeError("injected universe failure")

    with pytest.raises(RuntimeError, match="injected universe failure"):
        resolve_for_model(TrainablePlan(("selected",)), DeclaredTensors(failing_universe()))
    assert not would_be_selected.requires_grad


def test_independent_empty_storages_are_not_aliases() -> None:
    selected = torch.empty(0)
    complement = torch.empty(0)
    resolved = resolve_for_model(
        TrainablePlan(("selected",)),
        DeclaredTensors((("selected", selected), ("complement", complement))),
    )

    assert resolved.parameters()[0] is selected
    assert resolved.complement()[0] is complement
    assert selected.untyped_storage() != complement.untyped_storage()


def test_distinct_storage_wrappers_over_the_same_memory_are_aliases() -> None:
    selected = torch.randn(6)
    complement = torch.from_dlpack(selected)
    assert selected.untyped_storage() != complement.untyped_storage()
    assert selected.data_ptr() == complement.data_ptr()

    with pytest.raises(ValueError, match="share storage"):
        resolve_for_model(
            TrainablePlan(("selected",)),
            DeclaredTensors((("selected", selected), ("complement", complement))),
        )


def test_distinct_storage_wrappers_with_partially_overlapping_intervals_are_aliases() -> None:
    backing = torch.randn(8)
    selected = torch.from_dlpack(backing[:-1])
    complement = torch.from_dlpack(backing[1:])
    assert selected.untyped_storage() != complement.untyped_storage()
    assert selected.data_ptr() != complement.data_ptr()

    with pytest.raises(ValueError, match="share storage"):
        resolve_for_model(
            TrainablePlan(("selected",)),
            DeclaredTensors((("selected", selected), ("complement", complement))),
        )


def test_resolved_parameters_are_the_live_tensors() -> None:
    selected = _parameter(2)
    resolved = resolve_for_model(
        TrainablePlan(("selected",)), DeclaredTensors((("selected", selected),))
    )

    with torch.no_grad():
        selected.add_(1)
    assert torch.equal(resolved.parameters()[0], selected)


class _TrainableHeadModel(TrainingModelStub):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(20260713)
        self.frozen = torch.nn.Parameter(
            torch.randn(4, 4, generator=generator),
            requires_grad=True,
        )
        self.selected = torch.nn.Parameter(
            torch.randn(7, 4, generator=generator),
            requires_grad=False,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "features.weight": self.frozen,
            "projection.weight": self.selected,
        }

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (("projection.weight", self.selected),)

    def forward_training(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = F.one_hot(input_ids.remainder(4), num_classes=4).float()
        hidden = F.linear(features, self.trained(self.frozen))
        return F.linear(hidden, self.trained(self.selected))


class _GradientAliasModel(_TrainableHeadModel):
    def __init__(self) -> None:
        super().__init__()
        self.gradient_alias = _parameter(*self.selected.shape)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            **super().state_dict(),
            "gradient_alias.weight": self.gradient_alias,
        }


class _CheckpointableTrainableHeadModel(_TrainableHeadModel):
    """A head without decoder layers: the policy has no layer to checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.checkpoint_policy: ReentrantActivationCheckpoint | None = None

    def decoder_layer_trainable_parameters(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        return ()

    def residual_stream_layout(self) -> tuple[int, torch.dtype]:
        return self.frozen.shape[0], self.frozen.dtype

    def enable_activation_checkpointing(self, policy: ReentrantActivationCheckpoint) -> None:
        self.checkpoint_policy = policy


def _base_selection_batch() -> TrainingBatch:
    tokens = PaddedTokens(
        torch.tensor(((1, 2, 3, 4), (4, 5, 0, 0)), dtype=torch.int64),
        torch.tensor((4, 2), dtype=torch.int64),
        0,
    )
    return TrainingBatch(tokens, tokens.valid_targets(), SFT())


def test_runner_updates_only_one_selected_ordinary_model_tensor() -> None:
    model = _TrainableHeadModel()
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(("projection.weight",)),
            AdamWConfig(lr=0.03, weight_decay=0.0, max_grad_norm=None),
            _padded_batching(),
        ),
        trainables=model,
        checks=True,
    )
    # The FP32 live parameter also has a separate master.
    state = runner.checkpoint(num_optimizer_steps=0, model_revision=0).optimizer
    assert state.names == ("projection.weight",)
    assert len(state.master) == 1
    torch.testing.assert_close(state.master[0], model.selected, rtol=0, atol=0)
    assert model.trained.masters["projection.weight"].data_ptr() != model.selected.data_ptr()
    selected_id = id(model.selected)
    selected_storage = model.selected.untyped_storage()
    selected_before = model.selected.detach().clone()
    frozen_before = model.frozen.detach().clone()

    result = forward_backward(runner, _base_selection_batch())
    assert result.normalization_count == 4
    assert model.trained.masters["projection.weight"].grad is not None
    assert model.trained.masters["projection.weight"].grad.dtype == torch.float32
    assert model.frozen.grad is None
    assert torch.equal(model.selected, selected_before)
    assert torch.equal(model.frozen, frozen_before)
    optim_step(runner, result.normalization_count)
    assert torch.equal(model.selected, selected_before)
    assert model.selected.grad is None
    runner.publish()

    assert id(model.selected) == selected_id
    assert model.selected.untyped_storage() is selected_storage
    assert not torch.equal(model.selected, selected_before)
    assert torch.equal(model.frozen, frozen_before)
    assert model.trained.masters["projection.weight"].grad is None
    assert model.frozen.grad is None


def _microbatch_runner(model: _TrainableHeadModel) -> TrainingRunner:
    return host_runner(
        model,
        TrainingConfig(
            TrainablePlan(("projection.weight",)),
            AdamWConfig(lr=0.03, weight_decay=0.0, max_grad_norm=None),
            TrainingBatching(
                BatchLayout.PADDED,
                StaticBatching(microbatch_size=1, max_tokens_per_sample=64),
            ),
        ),
        trainables=model,
        checks=True,
    )


def test_padding_past_the_microbatch_token_bound_is_refused_before_execution() -> None:
    runner = _microbatch_runner(_TrainableHeadModel())
    batch = _base_selection_batch()
    # One row per microbatch, padded to 70 columns: 69 input tokens against a bound of 64.
    tokens = batch.tokens
    over_padded = TrainingBatch(
        PaddedTokens(F.pad(tokens.token_ids, (0, 66)), tokens.lengths, tokens.pad_token_id),
        F.pad(batch.targets, (0, 66)),
        SFT(),
    )

    with pytest.raises(TrainingInputError, match="69-token rows exceed the 64-token"):
        forward_backward(runner, over_padded)

    assert not runner.holds_gradients


def test_forward_backward_work_advances_one_microbatch_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TrainableHeadModel()
    runner = _microbatch_runner(model)
    batch = _base_selection_batch()
    executed_batch_sizes: list[int] = []
    hidden_states = model.training_hidden_states

    def observe_microbatch(input_ids: torch.Tensor, packing: object) -> torch.Tensor:
        executed_batch_sizes.append(input_ids.shape[0])
        return hidden_states(input_ids, packing)

    monkeypatch.setattr(model, "training_hidden_states", observe_microbatch)

    runner.forward_backward(batch)
    assert executed_batch_sizes == []

    assert runner.microbatch() is None
    assert executed_batch_sizes == [1]
    with pytest.raises(RuntimeError, match="already accepted"):
        runner.forward_backward(batch)
    with pytest.raises(RuntimeError, match="must complete"):
        optim_step(runner, batch.num_targets)

    result = runner.microbatch()
    assert result is not None
    assert result.normalization_count == batch.num_targets
    assert executed_batch_sizes == [1, 1]
    with pytest.raises(RuntimeError, match="no forward/backward batch is accepted"):
        runner.microbatch()


def test_microbatch_work_matches_the_synchronous_result_and_gradients() -> None:
    stepped_model = _TrainableHeadModel()
    stepped_runner = _microbatch_runner(stepped_model)
    batch = _base_selection_batch()
    stepped_runner.forward_backward(batch)
    assert stepped_runner.microbatch() is None
    stepped_result = stepped_runner.microbatch()
    assert stepped_result is not None
    stepped_gradient = stepped_model.trained.masters["projection.weight"].grad
    assert stepped_gradient is not None

    reference_model = _TrainableHeadModel()
    reference_runner = _microbatch_runner(reference_model)
    reference_result = forward_backward(reference_runner, batch)
    reference_gradient = reference_model.trained.masters["projection.weight"].grad
    assert reference_gradient is not None

    torch.testing.assert_close(
        stepped_result.loss_sum,
        reference_result.loss_sum,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(stepped_gradient, reference_gradient, rtol=0, atol=0)


def test_partial_microbatch_work_survives_exact_training_state_suspension() -> None:
    resumed_model = _TrainableHeadModel()
    resumed_runner = _microbatch_runner(resumed_model)
    batch = _base_selection_batch()
    resumed_runner.forward_backward(batch)
    assert resumed_runner.microbatch() is None
    partial_gradient = resumed_model.trained.masters["projection.weight"].grad
    assert partial_gradient is not None
    expected_partial_gradient = partial_gradient.detach().clone()

    resumed_runner.suspend()
    # RESIDENT keeps the gradient through serving's turn, at its address.
    suspended_gradient = resumed_model.trained.masters["projection.weight"].grad
    assert suspended_gradient is partial_gradient

    resumed_runner.resume()
    restored_gradient = resumed_model.trained.masters["projection.weight"].grad
    assert restored_gradient is partial_gradient
    torch.testing.assert_close(restored_gradient, expected_partial_gradient, rtol=0, atol=0)
    resumed_result = resumed_runner.microbatch()
    assert resumed_result is not None

    reference_model = _TrainableHeadModel()
    reference_runner = _microbatch_runner(reference_model)
    reference_result = forward_backward(reference_runner, batch)
    reference_gradient = reference_model.trained.masters["projection.weight"].grad
    assert reference_gradient is not None
    torch.testing.assert_close(resumed_result.loss_sum, reference_result.loss_sum, rtol=0, atol=0)
    torch.testing.assert_close(restored_gradient, reference_gradient, rtol=0, atol=0)


def test_failed_microbatch_work_discards_the_accumulation_and_releases_the_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TrainableHeadModel()
    runner = _microbatch_runner(model)
    batch = _base_selection_batch()
    training_loss = model.training_loss
    num_calls = 0

    def fail_second_microbatch(hidden_states: torch.Tensor, target_ids: torch.Tensor):
        nonlocal num_calls
        num_calls += 1
        if num_calls == 2:
            raise RuntimeError("injected microbatch failure")
        return training_loss(hidden_states, target_ids)

    monkeypatch.setattr(model, "training_loss", fail_second_microbatch)

    runner.forward_backward(batch)
    assert runner.microbatch() is None
    assert model.trained.masters["projection.weight"].grad is not None
    with pytest.raises(RuntimeError, match="accumulation window"):
        runner.microbatch()

    assert model.trained.masters["projection.weight"].grad is None
    assert not runner.holds_gradients


@pytest.mark.parametrize("mode", list(ActivationCheckpointMode), ids=lambda mode: mode.value)
def test_runner_hands_the_model_a_checkpoint_policy_unless_mode_is_none(
    monkeypatch: pytest.MonkeyPatch,
    mode: ActivationCheckpointMode,
) -> None:
    fake_cuda = FakeCuda().install(monkeypatch)
    model = _CheckpointableTrainableHeadModel()

    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(("projection.weight",)),
            AdamWConfig(),
            _padded_batching(),
            activation_checkpointing=mode,
        ),
        trainables=model,
        checks=True,
    )

    policy = model.checkpoint_policy
    if mode is ActivationCheckpointMode.NONE:
        assert policy is None
    else:
        assert isinstance(policy, ReentrantActivationCheckpoint)
        assert policy.first == 0
        # Only full_offload gives the policy an offload.
        assert (policy._offload is None) == (mode is ActivationCheckpointMode.FULL)
    runner.close()
    fake_cuda.require_released()


def test_runner_refuses_full_offload_without_a_cuda_device() -> None:
    model = _CheckpointableTrainableHeadModel()

    with pytest.raises(ValueError, match="requires a CUDA device"):
        host_runner(
            model,
            TrainingConfig(
                TrainablePlan(("projection.weight",)),
                AdamWConfig(),
                _padded_batching(),
                activation_checkpointing=ActivationCheckpointMode.FULL_OFFLOAD,
            ),
            trainables=model,
            checks=True,
        )
    assert model.checkpoint_policy is None


def test_optimizer_rejects_and_clears_frozen_gradient_pollution() -> None:
    model = _TrainableHeadModel()
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(("projection.weight",)),
            AdamWConfig(lr=0.03, weight_decay=0.0, max_grad_norm=None),
            _padded_batching(),
        ),
        trainables=model,
        checks=True,
    )
    batch = _base_selection_batch()
    selected_before = model.selected.detach().clone()

    forward_backward(runner, batch)
    model.frozen.grad = torch.ones_like(model.frozen)
    with pytest.raises(training.InvalidGradientError, match="Frozen complement gradient"):
        optim_step(runner, batch.num_targets)

    assert torch.equal(model.selected, selected_before)
    assert model.trained.masters["projection.weight"].grad is None
    assert model.frozen.grad is None
    forward_backward(runner, batch)
    optim_step(runner, batch.num_targets)


def test_runner_can_discard_an_uncommitted_accumulation_window() -> None:
    model = _TrainableHeadModel()
    selected_before = model.selected.detach().clone()
    frozen_before = model.frozen.detach().clone()
    runner = host_runner(
        model,
        TrainingConfig(TrainablePlan(("projection.weight",)), AdamWConfig(), _padded_batching()),
        trainables=model,
        checks=True,
    )

    batch = _base_selection_batch()
    forward_backward(runner, batch)
    assert model.trained.masters["projection.weight"].grad is not None
    runner.discard()

    assert model.trained.masters["projection.weight"].grad is None
    assert model.frozen.grad is None
    assert torch.equal(model.selected, selected_before)
    assert torch.equal(model.frozen, frozen_before)
    with pytest.raises(training.InvalidGradientError, match="missing"):
        optim_step(runner, batch.num_targets)


def test_a_discard_waits_for_the_gradients_copy_back_before_freeing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A resumption may still be copying the window's gradients back: both discards, the
    # caller's and a failed microbatch's, order their freeing after that copy.
    model = _TrainableHeadModel()
    runner = _microbatch_runner(model)
    calls: list[str] = []

    def record(owner: object, name: str) -> None:
        method = getattr(owner, name)

        def recorded(*args: object) -> object:
            calls.append(name)
            return method(*args)

        monkeypatch.setattr(owner, name, recorded)

    record(runner._state, "ready_for_accumulation")
    record(runner._optimizer, "discard_accumulation")
    batch = _base_selection_batch()

    forward_backward(runner, batch)
    calls.clear()
    runner.discard()
    assert calls == ["ready_for_accumulation", "discard_accumulation"]

    training_loss = model.training_loss
    losses = 0

    def fail_second_microbatch(hidden_states: torch.Tensor, target_ids: torch.Tensor):
        nonlocal losses
        losses += 1
        if losses == 2:
            raise RuntimeError("injected microbatch failure")
        return training_loss(hidden_states, target_ids)

    monkeypatch.setattr(model, "training_loss", fail_second_microbatch)
    runner.forward_backward(batch)
    assert runner.microbatch() is None
    calls.clear()
    with pytest.raises(RuntimeError, match="accumulation window"):
        runner.microbatch()
    assert calls == ["ready_for_accumulation", "discard_accumulation"]
    assert not runner.holds_gradients


def test_maintained_lora_profile_resolves_historical_name_order() -> None:
    class _Model(BaseOP):
        def __init__(self) -> None:
            self.projection = LinearReplicated(
                4,
                7,
                has_bias=False,
                lora="attention",
                site=site(),
            )

    model = _Model()
    base = model.projection
    adapters = Adapter(LoRAConfig(targets=("attention",), rank=2, alpha=4.0))
    adapters.attach(model)
    adapters.materialize(device=torch.device("cpu"), dtype=torch.float32)
    named_adapters = adapters.named_parameters()

    resolved = resolve_for_model(
        TrainablePlan(tuple(name for name, _ in named_adapters)),
        DeclaredTensors(
            (("base.weight", base.weight), *named_adapters),
        ),
    )

    assert resolved.selected_names == (
        "projection.full.lora_A",
        "projection.full.lora_B",
    )
    assert resolved.named_parameters() == named_adapters
    assert resolved.named_complement() == (("base.weight", base.weight),)
    assert not base.weight.requires_grad


def test_model_resolution_includes_the_complete_base_universe() -> None:
    model = _TrainableHeadModel()
    trainable = resolve_for_model(TrainablePlan(("projection.weight",)), model)
    assert trainable.selected_names == ("projection.weight",)
    assert trainable.complement_names == ("features.weight",)


def test_plans_are_values_of_their_selection() -> None:
    assert TrainablePlan.all_declared() == TrainablePlan() == TrainablePlan(None)
    assert TrainablePlan(("a", "b")) == TrainablePlan(("a", "b")) != TrainablePlan(("b", "a"))
    assert hash(TrainablePlan(("a",))) == hash(TrainablePlan(("a",)))
    assert TrainablePlan(("a",)).selected_names == ("a",)


def test_all_declared_selects_the_complete_non_base_declaration() -> None:
    model = _TrainableHeadModel()

    trainable = resolve_for_model(TrainablePlan.all_declared(), model)

    assert trainable.selected_names == ("projection.weight",)
    assert trainable.complement_names == ("features.weight",)


def test_all_declared_requires_a_model_declaration() -> None:
    class ServingOnlyModel(_TrainableHeadModel):
        def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
            return ()

    with pytest.raises(ValueError, match="declares no non-base trainable parameters"):
        resolve_for_model(TrainablePlan.all_declared(), ServingOnlyModel())


def test_model_resolution_freezes_unselected_declared_parameters() -> None:
    class MultipleHeadModel(_TrainableHeadModel):
        def __init__(self) -> None:
            super().__init__()
            self.auxiliary = _parameter(7, 4, requires_grad=True)

        def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
            return (
                *super().named_trainable_parameters(),
                ("auxiliary.weight", self.auxiliary),
            )

    model = MultipleHeadModel()
    trainable = resolve_for_model(TrainablePlan(("projection.weight",)), model)

    assert trainable.selected_names == ("projection.weight",)
    assert trainable.complement_names == ("features.weight", "auxiliary.weight")
    assert not model.auxiliary.requires_grad


def test_model_resolution_rejects_same_name_with_different_identity() -> None:
    class InvalidDeclarationModel(_TrainableHeadModel):
        def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
            return (("projection.weight", _parameter(*self.selected.shape)),)

    with pytest.raises(ValueError, match="does not match state_dict"):
        resolve_for_model(TrainablePlan(("projection.weight",)), InvalidDeclarationModel())


def test_model_resolution_rejects_base_parameter_selection() -> None:
    with pytest.raises(ValueError, match="model-declared non-base parameters"):
        resolve_for_model(TrainablePlan(("features.weight",)), _TrainableHeadModel())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_runner_rejects_unsupported_selected_parameter_dtype(dtype: torch.dtype) -> None:
    model = _TrainableHeadModel()
    model.selected = torch.nn.Parameter(model.selected.to(dtype), requires_grad=False)
    with pytest.raises(ValueError, match="FP32 and BF16"):
        host_runner(
            model,
            TrainingConfig(
                TrainablePlan(("projection.weight",)), AdamWConfig(), _padded_batching()
            ),
            trainables=model,
            checks=True,
        )
