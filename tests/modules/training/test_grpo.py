from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from cti.config import (
    AdamWConfig,
    BatchLayout,
    DynamicBatching,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.protocol import SampledSequence, SamplingResult, SamplingSpec
from cti.sampling import LogprobMode, TokenSampling
from cti.training import GRPO, ForwardBackwardCompletion, TrainingBatch
from cti.training._errors import TrainingInputError
from cti.training._objective import (
    Metrics,
    Reduction,
    clipped_surrogate,
    group_relative_advantages,
)
from cti.training.runner import TrainingRunner
from tests.support.training import (
    TrainingModelStub,
    forward_backward,
    host_runner,
    optim_step,
)

_REWARDS = [1.0, 3.0, 2.0, 5.0]


class _DenseSelectedTokenLogprobs:
    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        logits = F.linear(hidden_states, weight, bias).float() / temperature
        return torch.log_softmax(logits, dim=-1).gather(1, target_ids[:, None]).squeeze(1)


class _TinyGRPOModel(TrainingModelStub):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(20260722)
        self.frozen_head = torch.randn((64, 8), generator=generator)
        self.side = torch.nn.Parameter(torch.eye(8), requires_grad=False)
        self._logprob_operation = _DenseSelectedTokenLogprobs()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"frozen_head": self.frozen_head}

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (("side.weight", self.side),)

    def training_hidden_states(self, input_ids: torch.Tensor, packing: object) -> torch.Tensor:
        features = F.one_hot(input_ids.reshape(-1).remainder(8), num_classes=8).float()
        return features @ self.trained(self.side)

    def training_loss(self, hidden_states: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        raise AssertionError("GRPO must not use the supervised objective")

    def training_logprobs(
        self, hidden_states: torch.Tensor, target_ids: torch.Tensor, *, temperature: float
    ) -> torch.Tensor:
        return self._logprob_operation.forward(
            hidden_states, self.frozen_head, target_ids, temperature=temperature
        )


def _sample(
    request_id: int,
    *,
    prompt: tuple[int, ...] = (10, 11),
    response: tuple[int, ...] = (20, 21),
    sampling_logprobs: tuple[float, ...] = (-0.4, -0.7),
    temperature: float = 0.7,
    finish_reason: str = "length",
    seed: int | None = None,
) -> SampledSequence:
    return SampledSequence(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampled_token_ids=response,
        sampling_logprobs=sampling_logprobs,
        cached_prompt_tokens=0,
        sampling=SamplingSpec(
            law=TokenSampling(
                temperature=temperature, top_k=None, top_p=1.0, logprobs=LogprobMode.PROCESSED
            ),
            max_new_tokens=len(response),
            seed=request_id if seed is None else seed,
        ),
        finish_reason=finish_reason,  # type: ignore[arg-type]
    )


def _sampling_result() -> SamplingResult:
    samples = (
        _sample(0),
        _sample(1, response=(22,), sampling_logprobs=(-1.2,), finish_reason="stop"),
        _sample(2, prompt=(30,), response=(31, 32, 33), sampling_logprobs=(-0.2, -0.3, -0.4)),
        _sample(3, prompt=(30,), response=(34,), sampling_logprobs=(-0.6,)),
    )
    return SamplingResult(samples=samples, texts=("", "", "", ""), model_revision=7)


def _batch(layout: BatchLayout) -> TrainingBatch:
    padding = {"pad_token_id": 0, "padded_length": 6} if layout is BatchLayout.PADDED else {}
    return TrainingBatch.grpo(_sampling_result(), _REWARDS, layout, group_size=2, **padding)


def _runner(
    layout: BatchLayout, policy: StaticBatching | DynamicBatching | None = None
) -> tuple[_TinyGRPOModel, TrainingRunner]:
    model = _TinyGRPOModel()
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan(("side.weight",)),
            AdamWConfig(lr=0.03, max_grad_norm=None),
            TrainingBatching(
                layout, policy or StaticBatching(microbatch_size=4, max_tokens_per_sample=8)
            ),
        ),
        trainables=model,
        checks=True,
    )
    return model, runner


_LAYOUTS = pytest.mark.parametrize("layout", list(BatchLayout), ids=lambda layout: layout.value)


@_LAYOUTS
def test_runner_refuses_a_batch_over_the_sample_budget_before_running(layout: BatchLayout):
    _model, runner = _runner(layout, StaticBatching(microbatch_size=4, max_tokens_per_sample=3))

    with pytest.raises(TrainingInputError, match="exceeds max_tokens_per_sample=3"):
        runner.forward_backward(_batch(layout))


def test_runner_refuses_a_value_that_is_not_a_training_batch() -> None:
    _model, runner = _runner(BatchLayout.PADDED)

    with pytest.raises(TrainingInputError, match="TrainingBatch") as raised:
        runner.forward_backward(object())  # type: ignore[arg-type]
    assert raised.value.__cause__ is None


@_LAYOUTS
def test_runner_refuses_the_other_layout(layout: BatchLayout) -> None:
    other = BatchLayout.PACKED if layout is BatchLayout.PADDED else BatchLayout.PADDED
    _model, runner = _runner(layout)

    with pytest.raises(TrainingInputError, match=f"{other.value} batch .* {layout.value}"):
        runner.forward_backward(_batch(other))


def test_malformed_grpo_logprobs_fail_at_the_sampling_comparison() -> None:
    model, runner = _runner(BatchLayout.PADDED)
    logprobs = model.training_logprobs
    model.training_logprobs = lambda *args, **kwargs: logprobs(*args, **kwargs)[:-1]

    with pytest.raises(RuntimeError, match="size of tensor"):
        forward_backward(runner, _batch(BatchLayout.PADDED))


@pytest.mark.parametrize(
    ("sample", "message"),
    (
        (_sample(0, temperature=0.0), "at least"),
        (replace(_sample(0), sampling=replace(_sample(0).sampling, seed=None)), "seed"),
        (
            replace(
                _sample(0),
                sampling=replace(
                    _sample(0).sampling, law=replace(_sample(0).sampling.law, logprobs=None)
                ),
                sampling_logprobs=None,
            ),
            "log-probabilities",
        ),
        (_sample(0, prompt=(99,)), "one exact prompt"),
    ),
)
def test_batch_rejects_incomplete_provenance_and_mixed_prompts(
    sample: SampledSequence, message: str
) -> None:
    samples = (_sample(0), replace(sample, request_id=1))
    with pytest.raises(ValueError, match=message):
        TrainingBatch.grpo(
            SamplingResult(samples=samples, texts=("", ""), model_revision=0),
            [1.0, 2.0],
            BatchLayout.PACKED,
            group_size=2,
        )


@_LAYOUTS
def test_batch_targets_each_response_and_keeps_its_provenance(layout: BatchLayout) -> None:
    batch = _batch(layout)
    objective = batch.objective

    assert isinstance(objective, GRPO)
    assert batch.row_target_counts.tolist() == [2, 1, 3, 1]
    assert objective.normalization_count(batch) == 4
    # Each rollout's prompt, never a response or padding token.
    assert objective.supplied_token_ids(batch).tolist() == [10, 11, 10, 11, 30, 30]
    torch.testing.assert_close(
        objective.sampling_logprobs, torch.tensor([-0.4, -0.7, -1.2, -0.2, -0.3, -0.4, -0.6])
    )
    assert (objective.rollout_revision, objective.temperature) == (7, 0.7)
    rows = batch.microbatch((0, 1, 2, 3), torch.device("cpu"))
    assert rows.target_ids.tolist() == [20, 21, 22, 31, 32, 33, 34]


def test_completions_read_each_result_with_its_reductions() -> None:
    def metrics(loss: float, clip: float, ratio: float, count: int) -> Metrics:
        values = (
            ("loss_sum", Reduction.SUM, torch.tensor(loss)),
            ("clip_fraction", Reduction.MEAN, torch.tensor(clip)),
            ("max_abs_log_ratio", Reduction.MAX, torch.tensor(ratio)),
        )
        return Metrics(values, count, 2 * count)

    first, second = metrics(1.5, 0.25, 0.125, 2), metrics(-0.5, 0.5, 0.375, 4)
    folded = first.fold(second)

    assert Metrics.complete((first, folded)) == (
        ForwardBackwardCompletion(1.5, 2, {"clip_fraction": 0.25, "max_abs_log_ratio": 0.125}),
        ForwardBackwardCompletion(
            1.0, 6, {"clip_fraction": pytest.approx(5 / 12), "max_abs_log_ratio": 0.375}
        ),
    )


def test_group_relative_advantages_normalize_each_group_independently() -> None:
    rewards = torch.tensor([1.0, 3.0, -4.0, -4.0, 7.0, 8.0])
    wide = 1.0 / (math.sqrt(2.0) + 1e-6)
    narrow = 0.5 / (math.sqrt(0.5) + 1e-6)
    torch.testing.assert_close(
        group_relative_advantages(rewards, 2),
        torch.tensor([-wide, wide, 0.0, 0.0, -narrow, narrow]),
        atol=1e-6,
        rtol=1e-6,
    )
    equal = torch.tensor([0.1] * 128 + [0.3] * 128)
    assert torch.equal(group_relative_advantages(equal, 128), torch.zeros_like(equal))


def _objective_oracle(logprobs, old_logprobs, advantages, response_mask, low, high):
    losses = []
    for row in range(logprobs.shape[0]):
        terms = []
        for column in range(logprobs.shape[1]):
            if bool(response_mask[row, column]):
                ratio = torch.exp(logprobs[row, column] - old_logprobs[row, column])
                clipped = torch.clamp(ratio, 1.0 - low, 1.0 + high)
                terms.append(torch.minimum(ratio * advantages[row], clipped * advantages[row]))
        losses.append(-torch.stack(terms).mean())
    return torch.stack(losses).sum()


def test_clipped_surrogate_matches_an_independent_oracle_and_its_gradient() -> None:
    old = torch.tensor([[-1.0, -1.5, 0.0], [-0.8, -0.7, -0.6]], dtype=torch.float64)
    current = (old + torch.tensor([[0.3, -0.4, 9.0], [-0.3, 0.1, 0.4]])).requires_grad_()
    advantages = torch.tensor([2.0, -1.5], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    compact = current.masked_select(mask)

    ((_, _, actual), *_) = clipped_surrogate(
        compact,
        old.masked_select(mask),
        advantages,
        mask.sum(dim=1),
        epsilon_low=0.1,
        epsilon_high=0.2,
    )
    expected = _objective_oracle(current, old, advantages, mask, 0.1, 0.2)

    torch.testing.assert_close(actual, expected.float())
    actual_grad = torch.autograd.grad(actual, current, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, current)[0])
    assert torch.count_nonzero(actual_grad[~mask]) == 0


@_LAYOUTS
def test_runner_uses_same_revision_policy_and_updates_only_selected_parameters(
    layout: BatchLayout,
) -> None:
    model, runner = _runner(layout)
    frozen_before = model.frozen_head.clone()
    selected_before = model.side.detach().clone()

    result = forward_backward(runner, _batch(layout))
    (completion,) = Metrics.complete((result.relocate("cpu"),))

    assert completion.normalization_count == 4
    assert completion.metrics["clip_fraction"] == 0.0
    assert completion.metrics["max_abs_log_ratio"] == 0.0
    assert completion.metrics["max_abs_sampling_logprob_delta"] > 0.0
    assert bool(torch.count_nonzero(model.trained.masters["side.weight"].grad))
    assert model.frozen_head.grad is None
    optim_step(runner, completion.normalization_count)
    assert torch.equal(model.side, selected_before)
    runner.publish()
    torch.testing.assert_close(model.frozen_head, frozen_before)
    assert not torch.equal(model.side, selected_before)


@pytest.mark.parametrize(
    ("layout", "policy"),
    (
        (BatchLayout.PADDED, StaticBatching(microbatch_size=1, max_tokens_per_sample=8)),
        (BatchLayout.PADDED, StaticBatching(microbatch_size=2, max_tokens_per_sample=8)),
        (BatchLayout.PACKED, StaticBatching(microbatch_size=2, max_tokens_per_sample=8)),
        (BatchLayout.PACKED, DynamicBatching(max_tokens_per_microbatch=8)),
    ),
    ids=("padded-1", "padded-2", "packed-static", "packed-dynamic"),
)
def test_internal_microbatching_preserves_grpo_update(
    layout: BatchLayout, policy: StaticBatching | DynamicBatching
) -> None:
    batch = _batch(layout)
    full_model, full_runner = _runner(layout)
    split_model, split_runner = _runner(layout, policy)

    full = forward_backward(full_runner, batch).relocate("cpu")
    split = forward_backward(split_runner, batch).relocate("cpu")
    full_commit = full_runner.commit(4, (full,))
    split_commit = split_runner.commit(4, (split,))
    full_runner.publish()
    split_runner.publish()

    ((full_completion,), (split_completion,)) = full_commit.completions, split_commit.completions
    assert split_completion.loss_sum == pytest.approx(full_completion.loss_sum, rel=1e-6)
    assert split_completion.metrics == pytest.approx(full_completion.metrics, rel=1e-6)
    assert split_commit.grad_norm == pytest.approx(full_commit.grad_norm, rel=1e-6)
    torch.testing.assert_close(split_model.side, full_model.side, rtol=1e-6, atol=1e-7)


def test_padded_and_packed_layouts_train_alike() -> None:
    outcomes = []
    for layout in BatchLayout:
        model, runner = _runner(layout)
        result = forward_backward(runner, _batch(layout)).relocate("cpu")
        commit = runner.commit(4, (result,))
        runner.publish()
        outcomes.append((commit, model.side.detach().clone()))
    (padded, padded_side), (packed, packed_side) = outcomes

    assert packed.completions == padded.completions
    torch.testing.assert_close(packed_side, padded_side, rtol=0, atol=0)
