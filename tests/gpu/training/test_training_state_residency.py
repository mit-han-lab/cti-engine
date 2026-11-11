"""The offloading ``TrainingStateResidency`` tiers on CUDA: what a tier releases (the AdamW
moments; under OFFLOAD_ALL also the FP32 masters and an open window's gradients) is created
when training first holds the device and comes back, at its addresses, from the host after
a switch that loses its device contents, mid-window and after a commit. A runner switched
that way trains, and checkpoints, bit for bit as a RESIDENT one never switched.
"""

from __future__ import annotations

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
from cti.training import SFT, PaddedTokens, TrainingBatch
from cti.training._state_residency import MirroredTrainingState, OffloadedTrainingState
from tests.support.training import TrainingModelStub, forward_backward, host_runner, optim_step

pytestmark = pytest.mark.checkpoints()

_DEVICE = torch.device("cuda")


class _Bf16CausalLM(TrainingModelStub):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(20261021)
        self.selected = torch.nn.Parameter(
            torch.randn(7, 4, generator=generator, dtype=torch.bfloat16).to(_DEVICE),
            requires_grad=False,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"selected": self.selected}

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (("selected", self.selected),)

    def forward_training(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = F.one_hot(input_ids.remainder(4), num_classes=4).to(torch.bfloat16)
        return F.linear(features, self.trained(self.selected)).float()


def _batch(rows: tuple[tuple[int, ...], ...], lengths: tuple[int, ...]) -> TrainingBatch:
    tokens = PaddedTokens(
        torch.tensor(rows, dtype=torch.int64), torch.tensor(lengths, dtype=torch.int64), 0
    )
    return TrainingBatch(tokens, tokens.valid_targets(), SFT())


_BATCHES = (
    _batch(((1, 2, 3, 4),), (4,)),
    _batch(((4, 5, 0), (2, 6, 1)), (2, 3)),
    _batch(((6, 3, 2, 5, 1),), (5,)),
)


def _runner(residency: TrainingStateResidency, *, microbatch_size: int = 8):
    model = _Bf16CausalLM()
    config = TrainingConfig(
        TrainablePlan(("selected",)),
        AdamWConfig(lr=0.03, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.02, max_grad_norm=0.25),
        TrainingBatching(BatchLayout.PADDED, StaticBatching(microbatch_size, 64)),
        training_state_residency=residency,
    )
    return model, host_runner(model, config, trainables=model)


def _released(runner) -> tuple[torch.Tensor, ...]:
    """What the runner's residency releases with serving's turn: the moments, and under
    OFFLOAD_ALL the masters and the open window's gradients."""

    state, owner = runner._optimizer.device_state(), runner._state
    released = state["optimizer_state"] if isinstance(owner, MirroredTrainingState) else ()
    if isinstance(owner, OffloadedTrainingState):
        released = (*state["master_weights"], *released, *state["gradients"])
    return released


def _switch(runner) -> None:
    """A switch to serving and back, whose pause loses what the runner released."""

    runner.suspend()
    torch.cuda.synchronize()
    with torch.no_grad():
        pointers = [tensor.data_ptr() for tensor in _released(runner)]
        for tensor in _released(runner):
            tensor.fill_(float("nan"))
    runner.resume()
    assert [tensor.data_ptr() for tensor in _released(runner)] == pointers


def _train(runner, *, switch: str | None) -> list[tuple[float, float]]:
    trajectory = []
    for step in range(3):
        result = forward_backward(runner, _BATCHES[step])
        if switch == "mid-window":
            _switch(runner)
        grad_norm = optim_step(runner, result.normalization_count)
        if switch == "after-commit":
            _switch(runner)
        trajectory.append((float(result.loss_sum), grad_norm))
    return trajectory


def _assert_same_state(runner, reference) -> None:
    expected = reference.checkpoint(num_optimizer_steps=3, model_revision=0).optimizer
    restored = runner.checkpoint(num_optimizer_steps=3, model_revision=0).optimizer
    for left, right in zip(
        (*expected.live, *expected.master), (*restored.live, *restored.master), strict=True
    ):
        assert torch.equal(left, right)
    for left, right in zip(
        expected.adamw["state"].values(), restored.adamw["state"].values(), strict=True
    ):
        assert all(torch.equal(left[key], right[key]) for key in left)


_OFFLOADING = [TrainingStateResidency.OFFLOAD_OPTIMIZER, TrainingStateResidency.OFFLOAD_ALL]


@pytest.mark.parametrize("residency", _OFFLOADING, ids=lambda residency: residency.value)
@pytest.mark.parametrize("switch", ["mid-window", "after-commit"])
def test_offloaded_state_round_trips_bit_for_bit(
    switch: str, residency: TrainingStateResidency
) -> None:
    reference_model, reference = _runner(TrainingStateResidency.RESIDENT)
    model, runner = _runner(residency)
    # Nothing is on the device before training first holds it.
    assert runner._optimizer.device_state()["optimizer_state"] == ()

    assert _train(runner, switch=switch) == _train(reference, switch=None)
    _switch(runner)
    _assert_same_state(runner, reference)
    assert torch.equal(model.selected, reference_model.selected)


@pytest.mark.parametrize("residency", _OFFLOADING, ids=lambda residency: residency.value)
def test_a_checkpoint_restored_before_the_state_exists_takes_effect_when_it_does(
    residency: TrainingStateResidency,
) -> None:
    _, reference = _runner(TrainingStateResidency.RESIDENT)
    _train(reference, switch=None)
    checkpoint = reference.checkpoint(num_optimizer_steps=3, model_revision=0)

    model, runner = _runner(residency)
    runner.restore(checkpoint)
    # The live weights, which inference reads, are restored at once.
    assert runner._optimizer.device_state()["optimizer_state"] == ()
    assert torch.equal(model.selected.cpu(), checkpoint.optimizer.live[0])

    result = forward_backward(runner, _BATCHES[0])
    expected = forward_backward(reference, _BATCHES[0])
    assert optim_step(runner, result.normalization_count) == optim_step(
        reference, expected.normalization_count
    )


def test_a_window_switched_between_its_microbatches_trains_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OFFLOAD_ALL stashes an open window's gradients at the switch and copies them back on
    its copy stream; the next microbatch's backward, which adds into them, must wait for
    them. The copy back is held behind a sleep on that stream, so a backward that did not
    wait would add into the gradients before they land, and lose its contribution."""

    _, reference = _runner(TrainingStateResidency.RESIDENT, microbatch_size=1)
    _, runner = _runner(TrainingStateResidency.OFFLOAD_ALL, microbatch_size=1)
    optimizer = runner._optimizer
    resume_gradients = optimizer.resume_gradients

    def delayed(stash: torch.Tensor) -> None:
        # On the copy stream: about 0.5 s at the device's clock.
        torch.cuda._sleep(1_000_000_000)
        resume_gradients(stash)

    monkeypatch.setattr(optimizer, "resume_gradients", delayed)

    def train(trained, *, switch: bool) -> list[tuple[float, float]]:
        trajectory = []
        for _ in range(3):
            batch = _BATCHES[1]  # two rows, so two microbatches of one
            trained.forward_backward(batch)
            assert trained.microbatch() is None
            if switch:
                _switch(trained)
            result = trained.microbatch()
            assert result is not None
            trajectory.append(
                (float(result.loss_sum), optim_step(trained, result.normalization_count))
            )
        return trajectory

    assert train(runner, switch=True) == train(reference, switch=False)
    _assert_same_state(runner, reference)


@pytest.mark.parametrize("residency", _OFFLOADING, ids=lambda residency: residency.value)
def test_the_mirror_copies_each_state_class_once_per_direction(
    residency: TrainingStateResidency, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, runner = _runner(residency)
    result = forward_backward(runner, _BATCHES[0])
    optim_step(runner, result.normalization_count)
    copies: list[torch.Size] = []
    copy = torch.Tensor.copy_

    def counted(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        copies.append(tensor.shape)
        return copy(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", counted)
    # The moments, and under OFFLOAD_ALL the masters: one flat copy each.
    classes = 2 if residency is TrainingStateResidency.OFFLOAD_ALL else 1
    runner._state.after_step()
    assert len(copies) == classes
    copies.clear()
    runner.suspend()
    runner.resume()
    assert len(copies) == classes
