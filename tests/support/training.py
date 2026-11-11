"""Shared builders and numerical contracts for GPU training tests."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

from tests.support.checkpoints import QWEN3_0_6B

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from contextlib import AbstractContextManager

    import torch

    from cti.config import TrainingPauseBoundary
    from cti.layers import Adapter
    from cti.models.qwen3.model import Qwen3ForCausalLM
    from cti.models.training_contract import Packing
    from cti.models.training_output import TrainingOutputMixin
    from cti.training import (
        ForwardBackwardObservation,
        OptimizerStepResult,
        Refusal,
        TrainingBatch,
        TrainingVerb,
    )
    from cti.training._checkpoint import OptimizerState
    from cti.training._objective import Metrics
    from cti.training.runner import TrainingRunner

PAD_TOKEN_ID = 151643
RAW_SAMPLES = (
    (785, 14002),
    (785, 7772, 11580, 304),
    (785, 14002, 315, 4017, 374, 9255, 11, 323),
)


@dataclass(frozen=True)
class SFTNumericalTolerance:
    """Fixed-profile BF16 acceptance limits for an SFT layout."""

    logit_max_absolute: float
    logit_global_relative_l2: float
    loss_max_absolute: float
    loss_global_relative_l2: float
    gradient_global_relative_l2: float
    gradient_worst_relative_l2: float


# Exact 1.25x limits from separately calibrated padded and packed profiles. Keep
# separate values even while equal so one profile cannot silently redefine the other.
PADDED_BF16_TOLERANCE = SFTNumericalTolerance(
    logit_max_absolute=1.484375,
    logit_global_relative_l2=0.04993358626961708,
    loss_max_absolute=1.431570053100586,
    loss_global_relative_l2=0.021480454597622156,
    gradient_global_relative_l2=0.10572055354714394,
    gradient_worst_relative_l2=0.4077637940645218,
)
PACKED_BF16_TOLERANCE = SFTNumericalTolerance(
    logit_max_absolute=1.484375,
    logit_global_relative_l2=0.04993358626961708,
    loss_max_absolute=1.431570053100586,
    loss_global_relative_l2=0.021480454597622156,
    gradient_global_relative_l2=0.10572055354714394,
    gradient_worst_relative_l2=0.4077637940645218,
)


class TensorError(NamedTuple):
    max_absolute: float
    global_relative_l2: float
    relative_linf: float


class GradientError(NamedTuple):
    max_absolute: float
    global_relative_l2: float
    worst_relative_l2: float
    worst_name: str
    above_five_percent: int
    above_ten_percent: int
    relative_l2_by_name: dict[str, float]


def expected_qwen_adapter_names(num_layers: int = 28) -> tuple[str, ...]:
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


def qwen3_model_snapshot() -> str:
    return str(QWEN3_0_6B.snapshot())


def padded_sft_batch(
    samples: tuple[tuple[int, ...], ...] = RAW_SAMPLES,
    pad_token_id: int = PAD_TOKEN_ID,
) -> TrainingBatch:
    """Every causal target of ``samples``, right-padded."""

    from cti.config import BatchLayout

    return _sft_batch(samples, BatchLayout.PADDED, pad_token_id=pad_token_id)


def packed_sft_batch(samples: tuple[tuple[int, ...], ...] = RAW_SAMPLES) -> TrainingBatch:
    """Every causal target of ``samples``, packed."""

    from cti.config import BatchLayout

    return _sft_batch(samples, BatchLayout.PACKED)


def _sft_batch(samples, layout, **padding) -> TrainingBatch:
    from cti.training import SFTDatum, TrainingBatch

    data = [SFTDatum(sample, (False,) + (True,) * (len(sample) - 1)) for sample in samples]
    return TrainingBatch.sft(data, layout, **padding)


class TrainedTensors:
    """A test model's training reads of its declared tensors: one bound to an FP32 master
    (``bind``) reads as the master cast to its dtype, as the engine's layers read a factor's
    master; any other as itself."""

    def __init__(self) -> None:
        self._masters: dict[int, torch.Tensor] = {}
        self._named_masters: dict[str, torch.Tensor] = {}

    def bind(
        self, named: Iterable[tuple[str, torch.Tensor]], masters: Mapping[str, torch.Tensor]
    ) -> None:
        self._named_masters = dict(masters)
        tensors = dict(named)
        self._masters = {id(tensors[name]): master for name, master in masters.items()}

    @property
    def masters(self) -> Mapping[str, torch.Tensor]:
        """The bound FP32 leaves, by declared parameter name."""

        return MappingProxyType(self._named_masters)

    def release(self) -> None:
        self._masters = {}
        self._named_masters = {}

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        master = self._masters.get(id(tensor))
        return tensor if master is None else master.to(tensor.dtype)


class AdaptedTrainables:
    """A model and its adapter as the model host owns them for training: the model's state,
    the adapter's factors declared trainable, grouped by the model's decoder layers."""

    def __init__(self, model: Any, adapter: Adapter | None) -> None:
        self.model, self.adapter = model, adapter

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.model.state_dict()

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        adapter = self.adapter
        return () if adapter is None else adapter.named_parameters()

    def decoder_layer_trainable_parameters(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        adapter, model = self.adapter, self.model
        ops = dict(model.named_ops())
        return tuple(
            () if adapter is None else adapter.parameters_in(ops[path])
            for path in model.decoder_layer_paths()
        )

    def check_training_selection(self, names: tuple[str, ...]) -> None:
        if self.adapter is not None:
            self.adapter.check_training_selection(names)

    def bind_training_weights(self, masters: Mapping[str, torch.Tensor]) -> None:
        if self.adapter is not None:
            self.adapter.bind_training(masters)

    def release_training_weights(self) -> None:
        if self.adapter is not None:
            self.adapter.release_training()


class DeclaredTensors:
    """A parameter owner whose every tensor is declared trainable, in order."""

    def __init__(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
        self._named_tensors = named_tensors
        self.trained = TrainedTensors()

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        return {}

    def named_trainable_parameters(self) -> Iterable[tuple[str, torch.Tensor]]:
        return self._named_tensors

    def decoder_layer_trainable_parameters(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        return ()

    def check_training_selection(self, names: tuple[str, ...]) -> None:
        pass

    def bind_training_weights(self, masters: Mapping[str, torch.Tensor]) -> None:
        self.trained.bind(self._named_tensors, masters)

    def release_training_weights(self) -> None:
        self.trained.release()


class TrainingModelStub:
    """The training hooks of a test model whose ``forward_training`` computes logits: it
    accepts every token and takes cross-entropy of the logits. Each decoder layer is one
    training segment."""

    TRAINING_LAYER_SEGMENTS = 1

    @property
    def trained(self) -> TrainedTensors:
        """How this model's training forward reads its declared tensors."""

        return self.__dict__.setdefault("_trained_tensors", TrainedTensors())

    def check_training_selection(self, names: tuple[str, ...]) -> None:
        pass

    def bind_training_weights(self, masters: Mapping[str, torch.Tensor]) -> None:
        self.trained.bind(self.named_trainable_parameters(), masters)

    def release_training_weights(self) -> None:
        self.trained.release()

    def validate_training_targets(self, target_ids: torch.Tensor) -> None:
        pass

    def validate_training_inputs(self, token_ids: torch.Tensor) -> None:
        pass

    def training_hidden_states(self, input_ids: torch.Tensor, packing: object) -> torch.Tensor:
        """Its ``forward_training`` logits stand in for hidden states."""

        return self.forward_training(input_ids).flatten(0, -2)

    def training_loss(self, hidden_states: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return F.cross_entropy(hidden_states.float(), target_ids, reduction="sum")


class HostStream:
    """A training stream for a runner on the host: nothing to order or wait for."""

    def wait_stream(self, stream: object) -> None:
        pass

    def synchronize(self) -> None:
        pass


class HostAllocation:
    """Training memory on the host: ``persistent`` scopes what outlives a microbatch."""

    def __init__(self, persistent: Callable[[], AbstractContextManager[None]]) -> None:
        self.persistent_training_allocation = persistent

    def transient_training_allocation(self, scope_bytes: int) -> AbstractContextManager[None]:
        from contextlib import nullcontext

        return nullcontext()

    def optimizer_training_allocation(self) -> AbstractContextManager[None]:
        from contextlib import nullcontext

        return nullcontext()


def host_runner(
    model: object,
    config: object,
    *,
    trainables: object,
    checks: bool = True,
    persistent_allocation: Callable[[], AbstractContextManager[None]] | None = None,
    stream: object | None = None,
    ranks: object = None,
    placed: object = None,
    adapter_writer: Callable[[str, object], None] | None = None,
) -> TrainingRunner:
    """A TrainingRunner on the default allocator and ``stream`` (by default the current one),
    as the composition root builds one with the worker's training stream and memory; one rank unless
    ``placed`` places its parameters on tensor-parallel ``ranks``."""

    from contextlib import nullcontext

    import torch

    from cti.training.runner import TrainingRunner

    if stream is None:
        # On a device the runner's work stays on the current stream, as a test drives it.
        stream = torch.cuda.current_stream() if torch.cuda.is_available() else HostStream()
    return TrainingRunner.create(
        model,  # type: ignore[arg-type]
        config,  # type: ignore[arg-type]
        trainables=trainables,  # type: ignore[arg-type]
        stream=stream,  # type: ignore[arg-type]
        after=stream,  # type: ignore[arg-type]
        memory=HostAllocation(persistent_allocation or nullcontext),
        scope_bytes=0,
        model_signature=lambda: "host-model",
        adapter_writer=adapter_writer or _no_adapter,
        ranks=ranks,  # type: ignore[arg-type]
        placed=placed,  # type: ignore[arg-type]
        # The device's stack is the driver's, which the tests driving a runner do not size.
        stack_limit=lambda: 0,
        checks=checks,
    )


def _no_adapter(path: str, factors: object) -> None:
    raise AssertionError("this runner's model has no adapter to save")


def assert_optimizer_state_exact(actual: OptimizerState, expected: OptimizerState) -> None:
    """Two optimizer states hold the same names, values, masters and AdamW state, bit for
    bit."""

    import torch

    assert (actual.config, actual.names) == (expected.config, expected.names)
    for value, other in zip(
        (*actual.live, *actual.master), (*expected.live, *expected.master), strict=True
    ):
        torch.testing.assert_close(value, other, rtol=0, atol=0)
    assert actual.adamw["state"].keys() == expected.adamw["state"].keys()
    for index, state in expected.adamw["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(actual.adamw["state"][index][key], value, rtol=0, atol=0)


def forward_backward(runner: TrainingRunner, batch: TrainingBatch) -> Metrics:
    """Run every microbatch of ``batch``; its metrics come back on the host."""

    runner.forward_backward(batch)
    while (result := runner.microbatch()) is None:
        pass
    return result


def optim_step(runner: TrainingRunner, normalization_count: int) -> float:
    """Commit the accumulated gradients through the runner's one commit gate."""

    return runner.commit(normalization_count, ()).grad_norm


@dataclass(frozen=True)
class Reply:
    """One reply to the command ``id`` as the parent's call sees it: a progress report that
    ``renews`` the wait, an answer's ``value`` or a refusal's ``error``."""

    id: int
    value: object = None
    error: Refusal | None = None
    renews: bool = False

    def outcome(self) -> object:
        if self.error is not None:
            raise self.error
        return self.value


class RecordedReplies:
    """``TrainingReplies`` that hand each reply to ``sink`` as a ``Reply``."""

    def __init__(self, sink: Callable[[Reply], None]) -> None:
        self._sink = sink

    def answer(self, command_id: int, value: object = None) -> None:
        self._sink(Reply(command_id, value))

    def progress(self, command_id: int) -> None:
        self._sink(Reply(command_id, renews=True))

    def refuse(self, command_id: int, error: Refusal) -> None:
        self._sink(Reply(command_id, error=error))


class TrainingDriver:
    """A TrainingScheduler over ``runner`` driven through its commands, as EngineCore and
    the mode switch drive it: one open session holding the device, each command run once
    no batch runs."""

    def __init__(
        self, runner: TrainingRunner | None, boundary: TrainingPauseBoundary | None = None
    ) -> None:
        from cti.config import TrainingPauseBoundary
        from cti.training import TrainingVerb
        from cti.training.scheduler import TrainingScheduler

        assert runner is not None
        self._ids = itertools.count()
        self._replies: dict[int, Reply] = {}
        self.scheduler = TrainingScheduler(
            runner,
            boundary or TrainingPauseBoundary.LOGICAL_BATCH,
            replies=RecordedReplies(self._replied),
        )
        entry = next(self._ids)
        self.scheduler.handle(entry, TrainingVerb.ENTER_TRAINING_SESSION, (), holds=False)
        self.scheduler.switched(True, None)  # the switch hands training the device
        self.answer(entry)

    def _replied(self, reply: Reply) -> None:
        if not reply.renews:
            self._replies[reply.id] = reply

    def send(self, verb: TrainingVerb, *args: object, holds: bool = True) -> int:
        """Schedule the command ``verb`` in the session, training ``holds`` the device; its
        id."""

        command_id = next(self._ids)
        self.scheduler.handle(command_id, verb, args, holds=holds)
        return command_id

    def run(self) -> None:
        """Step until training wants nothing run."""

        while self.scheduler.wants_to_run():
            self.scheduler.step()

    def answered(self, command_id: int) -> bool:
        return command_id in self._replies

    def answer(self, command_id: int) -> object:
        """The command's answer, or its refusal raised."""

        return self._replies.pop(command_id).outcome()

    def call(self, verb: TrainingVerb, *args: object) -> object:
        command_id = self.send(verb, *args)
        self.run()
        return self.answer(command_id)

    def forward_backward(self, batch: TrainingBatch) -> int:
        """Submit ``batch`` and run it to its end; its submission id."""

        from cti.training import TrainingVerb

        return self.call(TrainingVerb.SUBMIT_FORWARD_BACKWARD, batch)

    def observe(self, submission_id: int) -> tuple[ForwardBackwardObservation, ...]:
        from cti.training import TrainingVerb

        return self.call(TrainingVerb.OBSERVE_FORWARD_BACKWARD, submission_id)

    def optim_step(
        self,
    ) -> tuple[OptimizerStepResult, tuple[ForwardBackwardObservation, ...]]:
        from cti.training import TrainingVerb

        submission_id = self.call(TrainingVerb.SUBMIT_OPTIMIZER_STEP)
        return self.call(TrainingVerb.OBSERVE_OPTIMIZER_STEP, submission_id)

    def discard(self) -> None:
        from cti.training import TrainingVerb

        self.call(TrainingVerb.DISCARD_ACCUMULATION)

    def publish(self) -> int:
        """Publish the trained adapter to inference; the new model revision."""

        from cti.training import TrainingVerb

        return self.call(TrainingVerb.PUBLISH_ADAPTER)


def clone_main_gradients(
    runner: TrainingRunner,
    named_live_parameters: tuple[tuple[str, torch.Tensor], ...],
) -> tuple[torch.Tensor, ...]:
    """The runner's FP32 gradients of the open window, one per live parameter, each checked
    finite and nonzero and placed on its parameter's device."""

    import torch

    assert all(parameter.grad is None for _, parameter in named_live_parameters)
    gradients = runner.accumulated_gradients()
    assert len(gradients) == len(named_live_parameters)
    placed = []
    for (name, live), gradient in zip(named_live_parameters, gradients, strict=True):
        assert gradient.layout == torch.strided, name
        assert gradient.dtype == torch.float32, name
        assert gradient.shape == live.shape, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name
        placed.append(gradient.to(live.device))
    return tuple(placed)


def synchronize_optimizer_parameters(
    runner: TrainingRunner,
    named_live_parameters: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    """Resynchronize a test runner after a fixture mutates live parameters: restore a
    checkpoint whose FP32 masters are the live values."""

    from dataclasses import replace

    checkpoint = runner.checkpoint(num_optimizer_steps=0, model_revision=0)
    state = checkpoint.optimizer
    assert state.names == tuple(name for name, _ in named_live_parameters)
    masters = tuple(live.detach().float().cpu() for _, live in named_live_parameters)
    runner.restore(replace(checkpoint, optimizer=state._replace(master=masters)))


def training_logits(
    model: TrainingOutputMixin, input_ids: torch.Tensor, packing: Packing | None = None
) -> torch.Tensor:
    """The LM head's training logits of every input position: ``[batch, length, vocab]``
    of padded rows, ``[tokens, vocab]`` with ``packing`` of packed sequences."""

    logits = model.lm_head.forward_training(model.training_hidden_states(input_ids, packing))
    return logits if packing is not None else logits.view(*input_ids.shape, -1)


def batched_valid_logits(
    model: Qwen3ForCausalLM,
    batch: TrainingBatch,
) -> torch.Tensor:
    import torch

    tokens = batch.tokens
    logits = training_logits(model, tokens.token_ids[:, :-1].to("cuda"))
    parts = [logits[row, : length - 1] for row, length in enumerate(tokens.sequence_lengths())]
    return torch.cat(parts)


def measure_tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> TensorError:
    import torch

    expected_float = expected.float()
    difference = actual.float() - expected_float
    expected_l2 = torch.linalg.vector_norm(expected_float)
    expected_linf = expected_float.abs().max()
    assert expected_l2 > 0
    assert expected_linf > 0
    max_absolute = difference.abs().max()
    return TensorError(
        max_absolute=float(max_absolute),
        global_relative_l2=float(torch.linalg.vector_norm(difference) / expected_l2),
        relative_linf=float(max_absolute / expected_linf),
    )


def measure_gradient_error(
    names: tuple[str, ...],
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> GradientError:
    import torch

    assert len(names) == len(actual) == len(expected)
    difference_squared = torch.zeros((), device="cuda")
    expected_squared = torch.zeros((), device="cuda")
    max_absolute = torch.zeros((), device="cuda")
    relative_errors = []
    relative_l2_by_name = {}
    for name, actual_gradient, expected_gradient in zip(names, actual, expected, strict=True):
        difference = actual_gradient.float() - expected_gradient.float()
        difference_squared += difference.square().sum()
        expected_squared += expected_gradient.float().square().sum()
        max_absolute = torch.maximum(max_absolute, difference.abs().max())
        relative_error = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            expected_gradient.float()
        )
        relative_errors.append(relative_error)
        relative_l2_by_name[name] = float(relative_error)

    relative_errors_tensor = torch.stack(relative_errors)
    worst_index = int(relative_errors_tensor.argmax())
    global_relative_l2 = torch.sqrt(difference_squared / expected_squared)
    return GradientError(
        max_absolute=float(max_absolute),
        global_relative_l2=float(global_relative_l2),
        worst_relative_l2=float(relative_errors_tensor[worst_index]),
        worst_name=names[worst_index],
        above_five_percent=int(torch.count_nonzero(relative_errors_tensor > 5e-2).item()),
        above_ten_percent=int(torch.count_nonzero(relative_errors_tensor > 1e-1).item()),
        relative_l2_by_name=relative_l2_by_name,
    )
