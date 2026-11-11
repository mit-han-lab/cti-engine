"""Qwen3.5 LoRA training against Transformers FP32 autograd on CPU.

Tiny hybrid models cover unequal key/value head counts and sequences beyond the
64-token delta-rule chunk. Padded and packed SFT/GRPO compare logits, objectives,
adapter gradients and AdamW updates across multiple steps and microbatches.
Base weights stay frozen; kernel operations use CPU references.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.overrides import TorchFunctionMode
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

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
from cti.layers import Adapter
from cti.models import create_model
from cti.models.base import BaseLLMModel
from cti.models.qwen3_5.config import Qwen3_5Config
from cti.models.qwen3_5.model import Qwen3_5DecoderLayer, Qwen3_5ForConditionalGeneration
from cti.models.weight import load_weight
from cti.protocol import SampledSequence, SamplingResult, SamplingSpec
from cti.sampling import LogprobMode, TokenSampling
from cti.training import PaddedTokens, SFTDatum, TrainingBatch, TrainingInputError
from cti.training._batch import DeviceRows
from cti.training._objective import clipped_surrogate
from cti.training.runner import TrainingRunner
from tests.support import reference_ops
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.gated_delta_net import reference_kernel_module
from tests.support.qwen3_5 import (
    TINY_QWEN3_5_CONFIG,
    load_hf_config,
    randomize_hf_weights,
    text_only_config,
)
from tests.support.qwen3_5_lora import (
    TEXT_ONLY_PREFIX,
    Qwen3_5LoRAMapping,
    attach_peft_lora,
    random_peft_state,
)
from tests.support.recompute_audit import assert_recomputed, record_block_runs
from tests.support.tp_shards import default_dtype, site
from tests.support.training import (
    AdaptedTrainables,
    forward_backward,
    host_runner,
    optim_step,
    synchronize_optimizer_parameters,
    training_logits,
)
from tests.support.training_ops import (
    bind_training_references,
)

CPU = torch.device("cpu")
_TARGETS = Qwen3_5ForConditionalGeneration.LORA_TARGETS
_RANK = 4
_ALPHA = 2.0
_PAD_TOKEN_ID = 0
_STEPS = 3
_TEMPERATURE = 0.8
# The tiny image-text checkpoint's image and video placeholders; random text stays below.
_IMAGE_TOKEN_ID = 60
_VIDEO_TOKEN_ID = 61
# The first sequence spans two of Transformers' 64-token delta-rule chunks.
_SFT_LENGTHS = (70, 5, 9)
# Each rollout group shares a prompt; responses differ in length.
_GRPO_ROLLOUTS = ((40, 30), (40, 5), (3, 20), (3, 2))
_MAX_TOKENS_PER_SAMPLE = 128
_OPTIMIZER = AdamWConfig(
    lr=1e-2,
    betas=(0.9, 0.99),
    eps=1e-6,
    weight_decay=0.01,
    max_grad_norm=1.0,
)


class _SDPACalls(TorchFunctionMode):
    """Count calls of ``F.scaled_dot_product_attention`` inside the context."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __torch_function__(
        self,
        func: Callable[..., object],
        _types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func is F.scaled_dot_product_attention:
            self.calls += 1
        return func(*args, **(kwargs or {}))


@pytest.fixture(autouse=True)
def _reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind a CPU reference at every kernel seam the model reaches."""

    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())
    reference_ops.bind_norm_references(monkeypatch)
    reference_ops.bind_vision_references(monkeypatch)
    bind_training_references(monkeypatch)


@dataclass(frozen=True)
class _Checkpoint:
    folder: Path
    num_attention_layers: int
    # A non-zero adapter in PEFT's layout, so every LoRA gradient is non-zero.
    peft_state: Mapping[str, torch.Tensor]


# (Gated DeltaNet key heads, value heads, attention query heads, KV heads): value and
# query heads are two (Qwen3.5-9B) or three (27B) times the key heads. With two key
# heads, value head j reading key head j // 3 (Transformers' repeat_interleave) differs
# from reading key head j % 2.
@pytest.fixture(
    scope="module",
    params=[(1, 2, 2, 1), (2, 6, 6, 2)],
    ids=["heads-1x2", "heads-2x6"],
)
def checkpoint(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> _Checkpoint:
    num_key_heads, num_value_heads, num_query_heads, num_kv_heads = request.param
    published = copy.deepcopy(dict(TINY_QWEN3_5_CONFIG))
    published["text_config"].update(
        linear_num_key_heads=num_key_heads,
        linear_num_value_heads=num_value_heads,
        num_attention_heads=num_query_heads,
        num_key_value_heads=num_kv_heads,
    )
    folder = tmp_path_factory.mktemp("qwen3_5")
    config = load_hf_config(text_only_config(published), folder)
    reference = AutoModelForCausalLM.from_config(config)
    randomize_hf_weights(reference, conv_dtype=torch.bfloat16)
    reference.save_pretrained(folder)
    peft_state = random_peft_state(
        reference.state_dict(),
        prefix=TEXT_ONLY_PREFIX,
        targets=_TARGETS,
        rank=_RANK,
        generator=torch.Generator().manual_seed(3),
    )
    return _Checkpoint(folder, config.layer_types.count("full_attention"), peft_state)


def _build_engine(
    hf_config: PretrainedConfig,
    load_state: Callable[[BaseLLMModel], Mapping[str, torch.Tensor]],
    engine_adapter: Mapping[str, torch.Tensor] | None,
    layout: BatchLayout,
    *,
    activation_checkpointing: ActivationCheckpointMode = ActivationCheckpointMode.NONE,
    trains: Callable[[str], bool] | None = None,
    microbatch_size: int = 4,
) -> tuple[BaseLLMModel, Adapter, TrainingRunner]:
    """``trains`` selects the adapters that train by name; ``None`` trains every one."""

    adapters = Adapter(LoRAConfig(targets=_TARGETS, rank=_RANK, alpha=_ALPHA))
    with torch.device("meta"), default_dtype(torch.float32):
        model = create_model(
            Qwen3_5ForConditionalGeneration,
            checkpoint_facts(hf_config),
            site=site(training=True),
        )
    adapters.attach(model)
    model.load_state_dict(dict(load_state(model)))
    adapters.materialize(device=CPU, dtype=torch.float32)
    if engine_adapter is not None:
        adapters.load_state_dict(engine_adapter)
    plan = (
        TrainablePlan.all_declared()
        if trains is None
        else TrainablePlan(tuple(name for name, _ in adapters.named_parameters() if trains(name)))
    )
    runner = host_runner(
        model,
        TrainingConfig(
            plan,
            _OPTIMIZER,
            TrainingBatching(
                layout,
                StaticBatching(
                    microbatch_size=microbatch_size,
                    max_tokens_per_sample=_MAX_TOKENS_PER_SAMPLE,
                ),
            ),
            activation_checkpointing=activation_checkpointing,
        ),
        trainables=AdaptedTrainables(model, adapters),
        checks=True,
    )
    return model, adapters, runner


def _mapping(checkpoint: _Checkpoint) -> Qwen3_5LoRAMapping:
    text = Qwen3_5Config.from_hf(AutoConfig.from_pretrained(checkpoint.folder)).text
    return Qwen3_5LoRAMapping(text, prefix=TEXT_ONLY_PREFIX)


def _engine(
    checkpoint: _Checkpoint,
    layout: BatchLayout,
    *,
    activation_checkpointing: ActivationCheckpointMode = ActivationCheckpointMode.NONE,
    trains: Callable[[str], bool] | None = None,
    microbatch_size: int = 4,
) -> tuple[BaseLLMModel, Adapter, TrainingRunner]:
    """The engine model as the runner builds it for training, with the PEFT adapter."""

    hf_config = AutoConfig.from_pretrained(checkpoint.folder)

    def load_state(model: BaseLLMModel) -> Mapping[str, torch.Tensor]:
        return dict(
            load_weight(
                str(checkpoint.folder),
                CPU,
                mapping=model.checkpoint_mapping,
                params=model.params(),
            )
        )

    return _build_engine(
        hf_config,
        load_state,
        _mapping(checkpoint).to_engine(checkpoint.peft_state),
        layout,
        activation_checkpointing=activation_checkpointing,
        trains=trains,
        microbatch_size=microbatch_size,
    )


class _Reference:
    """Transformers with the PEFT-shaped adapter, trained with the runner's AdamW."""

    def __init__(self, checkpoint: _Checkpoint) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint.folder,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        assert model.config._attn_implementation == "sdpa"
        self.parameters = attach_peft_lora(
            model,
            checkpoint.peft_state,
            scaling=_ALPHA / _RANK,
        )
        self.model: PreTrainedModel = model.train()
        self._num_attention_layers = checkpoint.num_attention_layers
        self._optimizer = torch.optim.AdamW(
            self.parameters.values(),
            lr=_OPTIMIZER.lr,
            betas=_OPTIMIZER.betas,
            eps=_OPTIMIZER.eps,
            weight_decay=_OPTIMIZER.weight_decay,
            foreach=False,
        )

    def logits(self, sequences: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Each input sequence's logits ``[length, vocab]``, one SDPA forward per sequence."""

        calls = _SDPACalls()
        with calls:
            logits = tuple(self.model(input_ids=tokens[None]).logits[0] for tokens in sequences)
        assert calls.calls == self._num_attention_layers * len(sequences)
        return logits

    def gradients(self) -> dict[str, torch.Tensor]:
        return {name: parameter.grad.clone() for name, parameter in self.parameters.items()}

    def step(self, normalization_count: int) -> None:
        """The runner's commit: divide by the count, clip the global norm, then AdamW."""

        parameters = tuple(self.parameters.values())
        with torch.no_grad():
            gradients = [parameter.grad.div_(normalization_count) for parameter in parameters]
            norm = torch.nn.utils.get_total_norm(gradients)
            torch.nn.utils.clip_grads_with_norm_(parameters, _OPTIMIZER.max_grad_norm, norm)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)


def _engine_gradients(
    runner: TrainingRunner, adapters: Adapter, checkpoint: _Checkpoint
) -> dict[str, torch.Tensor]:
    """The engine's adapter gradients in PEFT's layout: a shared A has one gradient."""

    named = dict(
        zip(
            (name for name, _ in adapters.named_parameters()),
            runner.accumulated_gradients(),
            strict=True,
        )
    )
    assert all(gradient is not None for gradient in named.values())
    return _mapping(checkpoint).to_peft(named)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    # Both sides run in FP32 end to end, with different Gated DeltaNet orders (the
    # engine reference's per-token recurrence, Transformers' 64-token chunks). Over four
    # seed sets the logits and log-probabilities measured within 3e-5 and the summed
    # losses (about 300) within 6.1e-5.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4, msg=label)


def _assert_gradients_match(
    engine: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> None:
    assert engine.keys() == reference.keys()
    for name, expected in reference.items():
        actual = engine[name]
        assert torch.count_nonzero(expected) > 0, name
        # Relative to the tensor's own scale: LoRA A and B gradients differ by orders
        # of magnitude between projections. The worst element measured 1.0e-4 of it.
        scale = float(expected.abs().max())
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-4 * scale, msg=name)


def _commit_and_continue_from_the_reference(
    runner: TrainingRunner,
    reference: _Reference,
    adapters: Adapter,
    checkpoint: _Checkpoint,
    normalization_count: int,
    step: int,
) -> None:
    """Commit one AdamW step on both sides, compare the adapters it leaves, then continue
    the engine from Transformers' adapter.

    AdamW divides each gradient element by its own running RMS, so free-running
    trajectories amplify FP32 rounding in small gradients from step to step; continuing
    from one adapter checks every step from the same weights.
    """

    live_before = {name: value.detach().clone() for name, value in adapters.named_parameters()}
    optim_step(runner, normalization_count)
    for name, live in adapters.named_parameters():
        assert live.grad is None
        torch.testing.assert_close(live, live_before[name], rtol=0, atol=0)
    runner.publish()
    reference.step(normalization_count)
    mapping = _mapping(checkpoint)
    engine = mapping.to_peft(adapters.state_dict())
    expected = {
        name: parameter.detach().clone() for name, parameter in reference.parameters.items()
    }
    for name, value in expected.items():
        # Each element moves by about lr. One whose gradient is near eps moves by up to
        # lr / eps times its gradient's rounding: a few elements measured up to 1.9% of
        # lr over four seed sets, the rest within 0.1%.
        torch.testing.assert_close(
            engine[name], value, rtol=0, atol=0.1 * _OPTIMIZER.lr, msg=f"step {step} {name}"
        )
    adapters.load_state_dict(mapping.to_engine(expected))
    synchronize_optimizer_parameters(runner, adapters.named_parameters())


def _sft_data(step: int) -> tuple[SFTDatum, ...]:
    """Random text whose first two tokens are an unsupervised prompt."""

    generator = torch.Generator().manual_seed(100 + step)
    vocab_size = TINY_QWEN3_5_CONFIG["text_config"]["vocab_size"]
    data = []
    for length in _SFT_LENGTHS:
        tokens = torch.randint(1, _IMAGE_TOKEN_ID, (length,), generator=generator)
        assert int(tokens.max()) < vocab_size
        data.append(
            SFTDatum(
                token_ids=tuple(tokens.tolist()),
                target_mask=(False, False, *([True] * (length - 2))),
            )
        )
    return tuple(data)


def _padding(layout: BatchLayout) -> dict[str, int]:
    return {"pad_token_id": _PAD_TOKEN_ID} if layout is BatchLayout.PADDED else {}


def _sft_batch(layout: BatchLayout, step: int) -> TrainingBatch:
    return TrainingBatch.sft(_sft_data(step), layout, **_padding(layout))


def _all_rows(batch: TrainingBatch) -> DeviceRows:
    return batch.microbatch(tuple(range(batch.batch_size)), torch.device("cpu"))


def _reference_sft_loss(
    logits: tuple[torch.Tensor, ...],
    data: tuple[SFTDatum, ...],
) -> torch.Tensor:
    losses = []
    for sequence_logits, datum in zip(logits, data, strict=True):
        targets = torch.tensor(datum.token_ids[1:])
        mask = torch.tensor(datum.target_mask[1:])
        losses.append(F.cross_entropy(sequence_logits[mask], targets[mask], reduction="sum"))
    return torch.stack(losses).sum()


def _engine_logits(
    model: BaseLLMModel,
    batch: TrainingBatch,
    lengths: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    """The training forward's logits of each sequence's input tokens."""

    rows = _all_rows(batch)
    if rows.packing is None:
        calls = _SDPACalls()
        with calls:
            logits = training_logits(model, rows.input_ids)
        # Padded training attention is one torch SDPA call per full-attention layer.
        num_attention_layers = len(tuple(model.iter_attention_layers()))
        assert calls.calls == num_attention_layers
        return tuple(row[: length - 1] for row, length in zip(logits, lengths, strict=True))
    logits = training_logits(model, rows.input_ids, rows.packing)
    return logits.split([length - 1 for length in lengths])


@pytest.mark.parametrize("layout", [BatchLayout.PADDED, BatchLayout.PACKED], ids=lambda x: x.value)
def test_sft_trajectory_matches_transformers_fp32_autograd(
    checkpoint: _Checkpoint,
    layout: BatchLayout,
) -> None:
    model, adapters, runner = _engine(checkpoint, layout, microbatch_size=2)
    reference = _Reference(checkpoint)

    for step in range(_STEPS):
        data = _sft_data(step)
        batch = _sft_batch(layout, step)
        sequences = tuple(torch.tensor(datum.token_ids[:-1]) for datum in data)
        with torch.no_grad():
            engine_logits = _engine_logits(model, batch, _SFT_LENGTHS)
        reference_logits = reference.logits(sequences)
        for index, (actual, expected) in enumerate(
            zip(engine_logits, reference_logits, strict=True)
        ):
            _assert_close(actual, expected.detach(), f"step {step} sequence {index} logits")

        result = forward_backward(runner, batch)
        reference_loss = _reference_sft_loss(reference_logits, data)
        reference_loss.backward()
        _assert_close(result.loss_sum, reference_loss.detach(), f"step {step} loss")
        assert result.normalization_count == sum(length - 2 for length in _SFT_LENGTHS)
        _assert_gradients_match(
            _engine_gradients(runner, adapters, checkpoint), reference.gradients()
        )

        _commit_and_continue_from_the_reference(
            runner, reference, adapters, checkpoint, result.normalization_count, step
        )

    # The adapters moved, and no gradient ever reached a frozen base weight.
    assert all(
        not tensor.requires_grad and tensor.grad is None for tensor in model.state_dict().values()
    )
    assert not torch.equal(
        _mapping(checkpoint).to_peft(adapters.state_dict())[next(iter(checkpoint.peft_state))],
        checkpoint.peft_state[next(iter(checkpoint.peft_state))],
    )


def test_the_training_entry_points_compute_the_runner_loss(checkpoint: _Checkpoint) -> None:
    model, _, runner = _engine(checkpoint, BatchLayout.PADDED)
    batch = _sft_batch(BatchLayout.PADDED, 0)
    rows = _all_rows(batch)
    assert rows.target_indices is not None

    with torch.no_grad():
        hidden_states = model.training_hidden_states(rows.input_ids, rows.packing)
        loss = model.training_loss(
            hidden_states.index_select(0, rows.target_indices), rows.target_ids
        )
        logits = training_logits(model, rows.input_ids)
    result = forward_backward(runner, batch)

    expected = F.cross_entropy(logits[batch.targets], rows.target_ids, reduction="sum")
    for actual in (loss, result.loss_sum):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)


def _rollouts(reference: _Reference) -> SamplingResult:
    """Grouped rollouts whose sampling log-probabilities are Transformers' at step 0."""

    generator = torch.Generator().manual_seed(7)
    samples = []
    for group in range(len(_GRPO_ROLLOUTS) // 2):
        prompt_length = _GRPO_ROLLOUTS[2 * group][0]
        prompt = torch.randint(1, _IMAGE_TOKEN_ID, (prompt_length,), generator=generator)
        for request_id in (2 * group, 2 * group + 1):
            response_length = _GRPO_ROLLOUTS[request_id][1]
            response = torch.randint(1, _IMAGE_TOKEN_ID, (response_length,), generator=generator)
            tokens = torch.cat((prompt, response))
            with torch.no_grad():
                (logits,) = reference.logits((tokens[:-1],))
            logprobs = _selected_logprobs(logits[prompt_length - 1 :], tokens[prompt_length:])
            samples.append(
                SampledSequence(
                    request_id=request_id,
                    prompt_token_ids=tuple(prompt.tolist()),
                    sampled_token_ids=tuple(response.tolist()),
                    sampling_logprobs=tuple(logprobs.tolist()),
                    cached_prompt_tokens=0,
                    sampling=SamplingSpec(
                        law=TokenSampling(
                            temperature=_TEMPERATURE,
                            top_k=None,
                            top_p=1.0,
                            logprobs=LogprobMode.PROCESSED,
                            ignore_eos=True,
                        ),
                        max_new_tokens=response_length,
                        seed=request_id,
                    ),
                    finish_reason="length",
                )
            )
    return SamplingResult(samples=tuple(samples), texts=("",) * len(samples), model_revision=0)


def _selected_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logprobs = torch.log_softmax(logits.float() / _TEMPERATURE, dim=-1)
    return logprobs.gather(1, targets[:, None]).squeeze(1)


def _grpo_batch(
    layout: BatchLayout,
    rollouts: SamplingResult,
    rewards: list[float],
) -> TrainingBatch:
    return TrainingBatch.grpo(rollouts, rewards, layout, group_size=2, **_padding(layout))


def _engine_grpo_logprobs(
    model: BaseLLMModel,
    batch: TrainingBatch,
) -> torch.Tensor:
    """The response tokens' log-probabilities from the model's GRPO forward."""

    rows = _all_rows(batch)
    hidden_states = model.training_hidden_states(rows.input_ids, rows.packing)
    if rows.target_indices is not None:
        hidden_states = hidden_states.index_select(0, rows.target_indices)
    return model.training_logprobs(hidden_states, rows.target_ids, temperature=_TEMPERATURE)


@pytest.mark.parametrize("layout", [BatchLayout.PADDED, BatchLayout.PACKED], ids=lambda x: x.value)
def test_grpo_trajectory_matches_transformers_fp32_autograd(
    checkpoint: _Checkpoint,
    layout: BatchLayout,
) -> None:
    model, adapters, runner = _engine(checkpoint, layout, microbatch_size=2)
    reference = _Reference(checkpoint)
    rollouts = _rollouts(reference)
    batch = _grpo_batch(layout, rollouts, [1.0, 3.0, 2.0, -1.0])
    prompt_lengths = [len(sample.prompt_token_ids) for sample in rollouts.samples]
    sequences = tuple(
        torch.tensor((*sample.prompt_token_ids, *sample.sampled_token_ids))
        for sample in rollouts.samples
    )
    response_lengths = torch.tensor([len(sample.sampled_token_ids) for sample in rollouts.samples])

    for step in range(_STEPS):
        logits = reference.logits(tuple(tokens[:-1] for tokens in sequences))
        reference_logprobs = torch.cat(
            [
                _selected_logprobs(sequence_logits[prompt_length - 1 :], tokens[prompt_length:])
                for sequence_logits, tokens, prompt_length in zip(
                    logits, sequences, prompt_lengths, strict=True
                )
            ]
        )
        with torch.no_grad():
            engine_logprobs = _engine_grpo_logprobs(model, batch)
        _assert_close(engine_logprobs, reference_logprobs.detach(), f"step {step} logprobs")

        result = forward_backward(runner, batch)
        ((_, _, objective), *_) = clipped_surrogate(
            reference_logprobs,
            reference_logprobs.detach(),
            batch.objective.advantages,
            response_lengths,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )
        objective.backward()
        # On policy the ratio is one, so the objective is zero; its gradient is not.
        _assert_close(result.loss_sum, objective.detach(), f"step {step} objective")
        if step == 0:
            # The rollouts' log-probabilities are Transformers' at these weights.
            metrics = {name: value for name, _, value in result.values}
            assert float(metrics["max_abs_sampling_logprob_delta"]) < 1e-4
        _assert_gradients_match(
            _engine_gradients(runner, adapters, checkpoint), reference.gradients()
        )

        assert result.normalization_count == len(sequences)
        _commit_and_continue_from_the_reference(
            runner, reference, adapters, checkpoint, result.normalization_count, step
        )


class _RecordingTrainingOperation:
    """Count the bound Gated DeltaNet training operation's calls, their boundaries and
    their sequence bounds."""

    def __init__(self, operation: Callable[..., torch.Tensor]) -> None:
        self._operation = operation
        self.cu_seqlens: list[torch.Tensor] = []
        self.max_sequence_lengths: list[int] = []

    def __call__(
        self, projected: torch.Tensor, *args: torch.Tensor, max_sequence_length: int
    ) -> torch.Tensor:
        self.cu_seqlens.append(args[-1].clone())
        self.max_sequence_lengths.append(max_sequence_length)
        return self._operation(projected, *args, max_sequence_length=max_sequence_length)


@pytest.mark.parametrize(
    ("trains", "first"),
    [
        pytest.param(None, 0, id="all-adapters"),
        # Layer 0, the Gated DeltaNet layer, holds none: it runs once, without autograd.
        pytest.param(lambda name: ".self_attn." in name, 1, id="attention-adapters"),
    ],
)
@pytest.mark.parametrize("layout", [BatchLayout.PADDED, BatchLayout.PACKED], ids=lambda x: x.value)
def test_activation_checkpointing_recomputes_from_the_first_trained_layer_to_the_same_gradients(
    checkpoint: _Checkpoint,
    layout: BatchLayout,
    trains: Callable[[str], bool] | None,
    first: int,
) -> None:
    batch = _sft_batch(layout, 0)
    results = {}
    for mode in (ActivationCheckpointMode.NONE, ActivationCheckpointMode.FULL):
        model, _, runner = _engine(checkpoint, layout, activation_checkpointing=mode, trains=trains)
        (layer,) = model.iter_recurrent_layers()
        recording = _RecordingTrainingOperation(layer._training_op)
        layer._training_op = recording
        layers = model.model.layers.op_list
        with record_block_runs(Qwen3_5DecoderLayer, layers) as runs:
            result = forward_backward(runner, batch)
        gradients = dict(
            zip(
                runner.trainable_parameters.selected_names,
                runner.accumulated_gradients(),
                strict=True,
            )
        )
        results[mode] = (result.loss_sum, gradients, recording, runs)

    loss, gradients, calls, runs = results[ActivationCheckpointMode.NONE]
    checkpointed_loss, checkpointed_gradients, recomputed_calls, checkpointed_runs = results[
        ActivationCheckpointMode.FULL
    ]
    assert_recomputed(runs, checkpointed_runs, first=first)
    # A recomputed Gated DeltaNet block repeats its call with the same boundaries.
    boundaries = calls.cu_seqlens
    assert len(boundaries) == 1
    assert len(recomputed_calls.cu_seqlens) == (2 if first == 0 else 1)
    for recorded in recomputed_calls.cu_seqlens:
        torch.testing.assert_close(recorded, boundaries[0], rtol=0, atol=0)
    tokens = batch.tokens
    if isinstance(tokens, PaddedTokens):
        # Right-padded rows are whole sequences: row b spans [b * L, (b + 1) * L).
        width = tokens.width
        expected = torch.arange(0, batch.batch_size * width + 1, width, dtype=torch.int32)
        longest = width
    else:
        expected = tokens.input_cu_seqlens
        longest = max(tokens.sequence_lengths()) - 1
    torch.testing.assert_close(boundaries[0], expected, rtol=0, atol=0)
    for recording in (calls, recomputed_calls):
        assert recording.max_sequence_lengths == [longest] * len(recording.cu_seqlens)
    torch.testing.assert_close(checkpointed_loss, loss, rtol=0, atol=0)
    assert checkpointed_gradients.keys() == gradients.keys()
    for name, gradient in gradients.items():
        assert torch.count_nonzero(gradient) > 0, name
        torch.testing.assert_close(checkpointed_gradients[name], gradient, rtol=0, atol=0)


@pytest.mark.parametrize(
    "freeze_breaker",
    [
        pytest.param(lambda model: model.iter_recurrent_layers()[0].A_log, id="gdn-a-log"),
        pytest.param(lambda model: model.iter_recurrent_layers()[0].conv_weight, id="gdn-conv"),
        pytest.param(lambda model: model.model.norm.weight, id="final-norm"),
        pytest.param(
            lambda model: model.model.layers.op_list[1].self_attn.q_norm.weight,
            id="query-norm",
        ),
    ],
)
def test_a_base_weight_that_requires_grad_fails_the_training_forward(
    checkpoint: _Checkpoint,
    freeze_breaker: Callable[[BaseLLMModel], torch.Tensor],
) -> None:
    model, _, _ = _engine(checkpoint, BatchLayout.PADDED)
    batch = _sft_batch(BatchLayout.PADDED, 0)
    freeze_breaker(model).requires_grad_(True)

    with pytest.raises(ValueError, match="frozen"):
        model.training_hidden_states(batch.tokens.token_ids[:, :-1], None)


def _image_text_engine(
    tmp_path: Path,
    layout: BatchLayout,
) -> tuple[BaseLLMModel, TrainingRunner]:
    """An image-text checkpoint's engine model with random weights."""

    published = copy.deepcopy(dict(TINY_QWEN3_5_CONFIG))
    published.update(image_token_id=_IMAGE_TOKEN_ID, video_token_id=_VIDEO_TOKEN_ID)
    hf_config = load_hf_config(published, tmp_path)
    generator = torch.Generator().manual_seed(5)

    def load_state(model: BaseLLMModel) -> dict[str, torch.Tensor]:
        return {
            name: torch.randn(tensor.shape, generator=generator).to(tensor.dtype) * 0.1
            for name, tensor in model.state_dict().items()
        }

    model, _, runner = _build_engine(hf_config, load_state, None, layout)
    return model, runner


def _one_group_rollouts(prompt: tuple[int, ...], response: tuple[int, ...]) -> SamplingResult:
    """Two rollouts of one prompt, each sampling ``response``."""

    return SamplingResult(
        samples=tuple(
            SampledSequence(
                request_id=request_id,
                prompt_token_ids=prompt,
                sampled_token_ids=response,
                sampling_logprobs=(-1.0,) * len(response),
                cached_prompt_tokens=0,
                sampling=SamplingSpec(
                    law=TokenSampling(
                        temperature=_TEMPERATURE,
                        top_k=None,
                        top_p=1.0,
                        logprobs=LogprobMode.PROCESSED,
                        ignore_eos=True,
                    ),
                    max_new_tokens=len(response),
                    seed=request_id,
                ),
                finish_reason="length",
            )
            for request_id in range(2)
        ),
        texts=("", ""),
        model_revision=0,
    )


@pytest.mark.parametrize("layout", [BatchLayout.PADDED, BatchLayout.PACKED], ids=lambda x: x.value)
def test_training_rejects_the_vision_placeholders_a_caller_supplies(
    tmp_path: Path,
    layout: BatchLayout,
) -> None:
    model, runner = _image_text_engine(tmp_path, layout)
    text = _sft_data(0)
    options = _padding(layout)

    # Images and videos reach the model only through the vision tower, which training
    # does not run yet; a placeholder embedded as text would give silently wrong logprobs.
    for placeholder in (_IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID):
        with_placeholder = (
            SFTDatum(token_ids=(3, placeholder, 5, 7), target_mask=(False, False, True, True)),
            *text[1:],
        )
        with pytest.raises(TrainingInputError, match=rf"placeholder tokens \[{placeholder}\]"):
            forward_backward(runner, TrainingBatch.sft(with_placeholder, layout, **options))
    with pytest.raises(TrainingInputError, match=rf"placeholder tokens \[{_IMAGE_TOKEN_ID}\]"):
        forward_backward(
            runner,
            _grpo_batch(layout, _one_group_rollouts((2, _IMAGE_TOKEN_ID, 4), (6,)), [1.0, 0.0]),
        )
    # A sampled placeholder is the policy's own token: sampling embedded it as text too.
    result = forward_backward(
        runner,
        _grpo_batch(layout, _one_group_rollouts((2, 3, 4), (_VIDEO_TOKEN_ID, 6)), [1.0, 0.0]),
    )
    assert torch.isfinite(result.loss_sum)
    runner.discard()

    result = forward_backward(runner, TrainingBatch.sft(text, layout, **options))
    assert torch.isfinite(result.loss_sum)
    with pytest.raises(
        ValueError, match=rf"placeholder tokens \[{_IMAGE_TOKEN_ID}, {_VIDEO_TOKEN_ID}\]"
    ):
        model.validate_training_inputs(torch.tensor([[1, _VIDEO_TOKEN_ID], [_IMAGE_TOKEN_ID, 3]]))


def test_a_text_only_export_trains_on_every_token(checkpoint: _Checkpoint) -> None:
    # A text-only export has no placeholders: Transformers embeds every id as text.
    model, _, runner = _engine(checkpoint, BatchLayout.PADDED)
    data = (
        SFTDatum(
            token_ids=(3, _IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID, 5),
            target_mask=(False, True, True, True),
        ),
    )

    model.validate_training_inputs(torch.tensor([_IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID]))
    result = forward_backward(
        runner, TrainingBatch.sft(data, BatchLayout.PADDED, pad_token_id=_PAD_TOKEN_ID)
    )

    assert torch.isfinite(result.loss_sum)


def test_peft_shaped_reference_adapts_every_targeted_module(checkpoint: _Checkpoint) -> None:
    reference = _Reference(checkpoint)
    adapted = [
        name for name, module in reference.model.named_modules() if hasattr(module, "lora_A")
    ]

    assert len(reference.parameters) == len(checkpoint.peft_state)
    assert len(adapted) == len(checkpoint.peft_state) // 2
    assert all(
        not parameter.requires_grad
        for name, parameter in reference.model.named_parameters()
        if ".lora_" not in name
    )
    assert isinstance(reference.model.get_submodule(adapted[0]).base_layer, nn.Linear)
