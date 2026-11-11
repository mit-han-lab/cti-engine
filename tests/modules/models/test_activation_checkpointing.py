"""The runner's activation checkpoint policy and the models that run it.

A six-layer stub model (three Gated-DeltaNet-like layers, an attention-like one, then one
of each) trains through ``TrainingRunner`` with and without checkpointing, for three
trainable sets: every layer's adapter, the attention layers' (the first is layer 3) and
the last layer's. The checkpointed step must match the eager one bit for bit, run the
frozen layers before the first trained layer once without autograd, recompute every
later layer once, and leave its zero-element anchor without a gradient. A checkpointed
block that writes its input in place is refused in the forward.

``full_offload`` trains the same way through ``HostBoundaryOffload`` on CPU stand-ins
for its CUDA calls (``tests/support/host_offload.py``): bit for bit, with its copies
ordered by events alone, each offloaded input freed one layer after it was saved, and
only the last layer's input kept between forward and backward. It refuses an input
written after it was saved, a recomputation that fails with a fetch in flight, a
backward that skips the saved inputs, a forward whose inputs backward never reads, a
second backward, a non-contiguous input and one its slots cannot hold, and the runner
refuses padding wider than the slots before any microbatch runs. After each failed
microbatch the runner trains on bit for bit.

The Qwen3 model runs its padded and packed blocks under the policy, and both Qwen3 and
Qwen3.5 declare their adapters by decoder layer and their residual stream's layout.
"""

from __future__ import annotations

import contextlib
import sys
import weakref
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import PretrainedConfig, Qwen3Config

import cti.models.activation_checkpointing as activation_checkpointing
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
from cti.models import create_model, get_model_class
from cti.models.activation_checkpointing import ReentrantActivationCheckpoint
from cti.models.qwen3.model import Qwen3DecoderLayer, Qwen3Model
from cti.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from cti.models.training_contract import Packing
from cti.training import SFT, PaddedTokens, TrainingBatch, TrainingInputError
from cti.training._boundary_offload import HostBoundaryOffload
from cti.training.runner import ModelParameterUniverse, TrainingRunner
from tests.support import reference_ops
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.host_offload import FakeCuda
from tests.support.qwen3_5 import TINY_QWEN3_5_CONFIG, load_hf_config
from tests.support.tp_shards import default_dtype, site
from tests.support.training import (
    AdaptedTrainables,
    TrainingModelStub,
    clone_main_gradients,
    forward_backward,
    host_runner,
)

_KINDS = ("linear",) * 3 + ("attention", "linear", "attention")
_HIDDEN = 8
_RANK = 2
_VOCAB = 16

# ============================== The six-layer stub model ==============================


class _Layer:
    """``x + tanh(x W^T + x A^T B^T + positions / 64)``: a frozen W and a LoRA-shaped A, B.

    Each call logs its grad mode and whether its output joined the autograd graph.
    """

    def __init__(
        self,
        kind: str,
        generator: torch.Generator,
        dtype: torch.dtype,
        trained: Callable[[torch.Tensor], torch.Tensor] = lambda tensor: tensor,
    ) -> None:
        self.kind = kind
        self.trained = trained
        self.weight = (torch.randn(_HIDDEN, _HIDDEN, generator=generator) / 3).to(dtype)
        self.lora_a = (torch.randn(_RANK, _HIDDEN, generator=generator) / 3).to(dtype)
        self.lora_b = (torch.randn(_HIDDEN, _RANK, generator=generator) / 2).to(dtype)
        self.calls: list[tuple[bool, bool]] = []

    def forward_training(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(x, self.trained(self.lora_a)), self.trained(self.lora_b))
        offset = positions.to(x.dtype).unsqueeze(-1) / 64
        output = x + torch.tanh(F.linear(x, self.weight) + update + offset)
        self.calls.append((torch.is_grad_enabled(), output.grad_fn is not None))
        return output


def _segments(
    layers: tuple[_Layer, ...], positions: torch.Tensor
) -> tuple[tuple[Callable[[torch.Tensor], torch.Tensor]], ...]:
    """Each stub layer as its one training segment."""

    return tuple(
        (partial(_Layer.forward_training, layer, positions=positions),) for layer in layers
    )


class _StubModel(TrainingModelStub):
    """An embedding, the six layers and a head. Only the layers' A and B are declared."""

    def __init__(self, dtype: torch.dtype) -> None:
        generator = torch.Generator().manual_seed(20261005)
        self.dtype = dtype
        self.embedding = torch.randn(_VOCAB, _HIDDEN, generator=generator).to(dtype)
        self.layers = tuple(_Layer(kind, generator, dtype, self.trained) for kind in _KINDS)
        self.head = (torch.randn(_VOCAB, _HIDDEN, generator=generator) / 3).to(dtype)
        self._activation_checkpoint: ReentrantActivationCheckpoint | None = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        weights = {
            f"layers.{index}.weight": layer.weight for index, layer in enumerate(self.layers)
        }
        return {"embedding": self.embedding, **weights, "head": self.head}

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return tuple(
            (f"layers.{index}.lora_{factor}", getattr(layer, f"lora_{factor}"))
            for index, layer in enumerate(self.layers)
            for factor in "ab"
        )

    def decoder_layer_trainable_parameters(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        return tuple((layer.lora_a, layer.lora_b) for layer in self.layers)

    def residual_stream_layout(self) -> tuple[int, torch.dtype]:
        return _HIDDEN, self.dtype

    def enable_activation_checkpointing(self, policy: ReentrantActivationCheckpoint) -> None:
        self._activation_checkpoint = policy

    def training_hidden_states(self, input_ids: torch.Tensor, packing: object) -> torch.Tensor:
        x = F.embedding(input_ids, self.embedding)
        positions = torch.arange(input_ids.shape[1])
        policy = self._activation_checkpoint
        if policy is None:
            for layer in self.layers:
                x = _Layer.forward_training(layer, x, positions)
        else:
            x = policy.run_blocks(_segments(self.layers, positions), x)
        return x.flatten(0, 1)

    def training_loss(self, hidden_states: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_states, self.head)
        return F.cross_entropy(logits.float(), target_ids, reduction="sum")


def _batch() -> TrainingBatch:
    generator = torch.Generator().manual_seed(7)
    lengths = (6, 4, 5)
    token_ids = torch.randint(1, _VOCAB, (len(lengths), max(lengths)), generator=generator)
    for row, length in enumerate(lengths):
        token_ids[row, length:] = 0
    tokens = PaddedTokens(token_ids, torch.tensor(lengths), 0)
    return TrainingBatch(tokens, tokens.valid_targets(), SFT())


def _record_anchors(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    """Record the anchor the policy passes to each reentrant checkpoint."""

    anchors: list[torch.Tensor] = []
    checkpoint = activation_checkpointing.torch_checkpoint

    def record(
        function: Callable[..., torch.Tensor],
        x: torch.Tensor,
        anchor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        anchors.append(anchor)
        return checkpoint(function, x, anchor, *args, **kwargs)

    monkeypatch.setattr(activation_checkpointing, "torch_checkpoint", record)
    return anchors


def _runner(
    model: _StubModel,
    mode: ActivationCheckpointMode,
    trains: Callable[[int, str], bool],
) -> TrainingRunner:
    selected = tuple(
        f"layers.{index}.lora_{factor}"
        for index, kind in enumerate(_KINDS)
        if trains(index, kind)
        for factor in "ab"
    )
    return host_runner(
        model,
        TrainingConfig(
            TrainablePlan(selected),
            AdamWConfig(),
            TrainingBatching(
                BatchLayout.PADDED,
                StaticBatching(microbatch_size=8, max_tokens_per_sample=64),
            ),
            activation_checkpointing=mode,
        ),
        trainables=model,
        checks=True,
    )


def _train(
    mode: ActivationCheckpointMode,
    dtype: torch.dtype,
    trains: Callable[[int, str], bool],
) -> tuple[_StubModel, torch.Tensor, dict[str, torch.Tensor]]:
    """One SFT step's model, loss and FP32 gradients of the selected factors."""

    model = _StubModel(dtype)
    runner = _runner(model, mode, trains)
    loss = forward_backward(runner, _batch()).loss_sum
    gradients = _gradients(runner)
    runner.close()
    return model, loss, gradients


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeCuda]:
    fake = FakeCuda().install(monkeypatch)
    yield fake
    fake.require_released()


@pytest.mark.parametrize(
    "mode",
    [ActivationCheckpointMode.FULL, ActivationCheckpointMode.FULL_OFFLOAD],
    ids=lambda mode: mode.value,
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    ("trains", "first"),
    [
        pytest.param(lambda _index, _kind: True, 0, id="all-layers"),
        pytest.param(lambda _index, kind: kind == "attention", 3, id="attention-layers"),
        pytest.param(lambda index, _kind: index == len(_KINDS) - 1, 5, id="last-layer"),
    ],
)
def test_checkpointing_trains_bitwise_like_eager_from_the_first_trained_layer(
    monkeypatch: pytest.MonkeyPatch,
    fake_cuda: FakeCuda,
    mode: ActivationCheckpointMode,
    dtype: torch.dtype,
    trains: Callable[[int, str], bool],
    first: int,
) -> None:
    eager, loss, gradients = _train(ActivationCheckpointMode.NONE, dtype, trains)
    anchors = _record_anchors(monkeypatch)
    checkpointed, checkpointed_loss, checkpointed_gradients = _train(mode, dtype, trains)

    policy = checkpointed._activation_checkpoint
    assert policy is not None and policy.first == first
    assert eager._activation_checkpoint is None
    torch.testing.assert_close(checkpointed_loss, loss, rtol=0, atol=0)
    assert checkpointed_gradients.keys() == gradients.keys()
    for name, gradient in gradients.items():
        torch.testing.assert_close(checkpointed_gradients[name], gradient, rtol=0, atol=0)
    # Eager: every layer once with grad; the frozen prefix's outputs have no grad_fn.
    # Checkpointed: the frozen prefix once without autograd; each later layer inside its
    # checkpoint, without grad, and once more when backward recomputes it.
    for index, (eager_layer, layer) in enumerate(
        zip(eager.layers, checkpointed.layers, strict=True)
    ):
        recomputed = index >= first
        assert eager_layer.calls == [(True, recomputed)]
        assert layer.calls == ([(False, False), (True, True)] if recomputed else [(False, False)])
    # One zero-element anchor per forward, outside the model, and it gets no gradient.
    assert len(anchors) == len(_KINDS) - first
    anchor = anchors[0]
    assert all(other is anchor for other in anchors)
    assert anchor.numel() == 0 and anchor.requires_grad and anchor.grad is None
    assert all(
        tensor is not anchor for _, tensor in ModelParameterUniverse.of(checkpointed).named_tensors
    )
    if mode is ActivationCheckpointMode.FULL:
        assert fake_cuda.offloads == []
        return
    # One checkpoint per layer from first, each as wide as the stream and as long as the
    # batching bound (8 samples of at most 64 tokens); the last input stays on the device.
    segments = len(_KINDS) - first
    ((options, offload),) = fake_cuda.offloads
    # The runner pins the arena from its own host memory, which closing it released.
    assert options.pop("host").nbytes == 0
    assert options == {
        "segments": segments,
        "tokens": 512,
        "width": _HIDDEN,
        "dtype": dtype,
        "device": torch.device("cpu"),
    }
    assert policy._offload is offload
    assert offload.nbytes == (segments - 1) * 512 * _HIDDEN * dtype.itemsize
    # Each offloaded input went out once in the forward and came back once in backward.
    assert fake_cuda.operations("copy").count(("issue", "")) == 2 * (segments - 1)


def test_each_checkpointed_forward_gets_a_fresh_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _StubModel(torch.float32)
    for layer in model.layers:
        layer.lora_a.requires_grad_(True)
    anchors = _record_anchors(monkeypatch)
    policy = ReentrantActivationCheckpoint(0)
    x = F.embedding(torch.tensor([[1, 2, 3]]), model.embedding)
    positions = torch.arange(3)

    for _ in range(2):
        policy.run_blocks(_segments(model.layers, positions), x).sum().backward()

    assert len(anchors) == 2 * len(_KINDS)
    assert anchors[0] is not anchors[len(_KINDS)]
    assert all(anchor.grad is None for anchor in anchors)


class _ExpectingOffload:
    """An offload that keeps nothing: it records the input of each checkpoint."""

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def saving(self) -> contextlib.nullcontext[None]:
        return contextlib.nullcontext()

    def expect(self, tensor: torch.Tensor) -> None:
        self.inputs.append(tensor)


def test_an_offload_checkpoints_each_segment_and_none_each_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pairs of stub layers as two-segment layers: with an offload every segment is a
    checkpoint whose input it holds, without one every layer is; both match eager. The
    first layer's first segment is frozen, as an unadapted mixer is."""

    model = _StubModel(torch.float32)
    for layer in model.layers[1:]:
        layer.lora_a.requires_grad_(True)
    anchors = _record_anchors(monkeypatch)
    x = F.embedding(torch.tensor([[1, 2, 3]]), model.embedding)
    segments = _segments(model.layers, torch.arange(3))
    layers = tuple(
        first + second for first, second in zip(segments[::2], segments[1::2], strict=True)
    )
    lora_a = model.layers[1].lora_a
    eager = activation_checkpointing.run_segments(layers, x)
    eager.sum().backward()
    expected, lora_a.grad = lora_a.grad, None

    for offload, checkpoints in ((None, len(layers)), (_ExpectingOffload(), len(_KINDS))):
        anchors.clear()
        output = ReentrantActivationCheckpoint(0, offload).run_blocks(layers, x)
        output.sum().backward()
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        torch.testing.assert_close(lora_a.grad, expected, rtol=0, atol=0)
        lora_a.grad = None
        assert len(anchors) == checkpoints
        if offload is not None:
            assert len(offload.inputs) == checkpoints and offload.inputs[0] is x


def test_an_offload_trains_a_row_wise_segment_tile_by_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row-wise segment under an offload runs once per tile in the forward and once per
    tile when backward recomputes it, and trains like eager."""

    monkeypatch.setattr(activation_checkpointing, "ROW_TILE", 2)
    model = _StubModel(torch.float32)
    layer = model.layers[0]
    layer.lora_a.requires_grad_(True)
    tiles: list[int] = []

    def row_wise(x: torch.Tensor) -> torch.Tensor:
        tiles.append(x.shape[0])
        return x + torch.tanh(F.linear(F.linear(x, layer.lora_a), layer.lora_b))

    x = F.embedding(torch.tensor([[1, 2, 3, 4, 5]]), model.embedding).requires_grad_(False)
    stream = (x * 2).requires_grad_(True)
    eager = row_wise(stream)
    eager.square().sum().backward()
    expected, expected_input = layer.lora_a.grad, stream.grad
    layer.lora_a.grad = None
    tiles.clear()

    offload = _ExpectingOffload()
    output = ReentrantActivationCheckpoint(0, offload).run_blocks(
        ((activation_checkpointing.RowWise(row_wise),),), x * 2
    )
    torch.testing.assert_close(output, eager.detach())
    assert tiles == [2, 2, 1]
    tiles.clear()
    output.square().sum().backward()
    assert tiles == [2, 2, 1]
    torch.testing.assert_close(layer.lora_a.grad, expected)
    assert len(offload.inputs) == 1 and expected_input is not None


def test_a_grad_disabled_forward_runs_every_layer_plainly(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _StubModel(torch.float32)
    anchors = _record_anchors(monkeypatch)
    policy = ReentrantActivationCheckpoint(3)
    x = F.embedding(torch.tensor([[1, 2, 3]]), model.embedding)
    expected = x
    for layer in model.layers:
        expected = _Layer.forward_training(layer, expected, torch.arange(3))
        layer.calls.clear()

    with torch.no_grad():
        output = policy.run_blocks(_segments(model.layers, torch.arange(3)), x)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert anchors == []
    assert all(layer.calls == [(False, False)] for layer in model.layers)


def test_the_policy_refuses_a_decoder_input_with_grad() -> None:
    model = _StubModel(torch.float32)
    x = F.embedding(torch.tensor([[1, 2]]), model.embedding).requires_grad_(True)

    with pytest.raises(RuntimeError, match="decoder input without grad"):
        ReentrantActivationCheckpoint(0).run_blocks(_segments(model.layers, torch.arange(2)), x)
    assert all(layer.calls == [] for layer in model.layers)


@pytest.mark.parametrize(
    ("first", "writer"),
    [
        pytest.param(0, 0, id="first-layer-0"),
        pytest.param(3, 3, id="first-after-the-frozen-prefix"),
        pytest.param(3, 5, id="later-layer"),
    ],
)
def test_the_policy_refuses_a_checkpointed_block_that_writes_its_input(
    first: int,
    writer: int,
) -> None:
    model = _StubModel(torch.float32)
    x = F.embedding(torch.tensor([[1, 2, 3]]), model.embedding)

    def block(layer: _Layer, hidden: torch.Tensor) -> torch.Tensor:
        if layer is model.layers[writer]:
            hidden.mul_(1.5)
        return _Layer.forward_training(layer, hidden, torch.arange(3))

    layers = tuple((partial(block, layer),) for layer in model.layers)
    with pytest.raises(RuntimeError, match="wrote its input in place"):
        ReentrantActivationCheckpoint(first).run_blocks(layers, x)
    # The forward stops at the writer, before backward could recompute from its input.
    calls = [len(layer.calls) for layer in model.layers]
    assert calls == [1] * (writer + 1) + [0] * (len(_KINDS) - writer - 1)


@pytest.mark.parametrize("first", [-1, True, 1.0])
def test_the_policy_takes_a_non_negative_layer_index(first: object) -> None:
    with pytest.raises(ValueError, match="non-negative decoder layer index"):
        ReentrantActivationCheckpoint(first)  # type: ignore[arg-type]


# ================= full_offload: the checkpointed inputs in host memory =================


def _trains_attention(_index: int, kind: str) -> bool:
    return kind == "attention"


def _offload_runner(model: _StubModel) -> tuple[TrainingRunner, HostBoundaryOffload]:
    """A ``full_offload`` runner training the attention layers (checkpoints 3, 4 and 5), and
    the offload it built."""

    built: list[HostBoundaryOffload] = []
    construct = HostBoundaryOffload.__init__

    def record(self: HostBoundaryOffload, *args: object, **kwargs: object) -> None:
        construct(self, *args, **kwargs)
        built.append(self)

    HostBoundaryOffload.__init__ = record  # type: ignore[method-assign]
    try:
        runner = _runner(model, ActivationCheckpointMode.FULL_OFFLOAD, _trains_attention)
    finally:
        HostBoundaryOffload.__init__ = construct  # type: ignore[method-assign]
    (offload,) = built
    return runner, offload


def _gradients(runner: TrainingRunner) -> dict[str, torch.Tensor]:
    return dict(
        zip(
            runner.trainable_parameters.selected_names,
            clone_main_gradients(runner, runner.trainable_parameters.named_parameters()),
            strict=True,
        )
    )


def _assert_trains_like_eager(runner: TrainingRunner) -> None:
    """A clean step of ``runner`` equals the eager step bit for bit."""

    _, loss, gradients = _train(ActivationCheckpointMode.NONE, torch.float32, _trains_attention)
    torch.testing.assert_close(forward_backward(runner, _batch()).loss_sum, loss, rtol=0, atol=0)
    trained = _gradients(runner)
    assert trained.keys() == gradients.keys()
    for name, gradient in gradients.items():
        torch.testing.assert_close(trained[name], gradient, rtol=0, atol=0)


def _offload_policy(
    model: _StubModel,
    first: int,
    fake_cuda: FakeCuda,
    *,
    tokens: int = 64,
    dtype: torch.dtype = torch.float32,
) -> HostBoundaryOffload:
    """Train ``model``'s layers from ``first`` under a policy with an offload pinned from
    ``fake_cuda``'s host memory; return it."""

    for layer in model.layers[first:]:
        layer.lora_a.requires_grad_(True)
        layer.lora_b.requires_grad_(True)
    offload = HostBoundaryOffload(
        segments=len(_KINDS) - first,
        tokens=tokens,
        width=_HIDDEN,
        dtype=dtype,
        device=torch.device("cuda"),
        host=fake_cuda.host,
    )
    model.enable_activation_checkpointing(ReentrantActivationCheckpoint(first, offload))
    return offload


def _stub_loss(model: _StubModel) -> torch.Tensor:
    rows = _batch().microbatch((0, 1, 2), torch.device("cpu"))
    hidden_states = model.training_hidden_states(rows.input_ids, rows.packing)
    return model.training_loss(hidden_states.index_select(0, rows.target_indices), rows.target_ids)


def _wrap_blocks(
    monkeypatch: pytest.MonkeyPatch,
    before: Callable[[_Layer, torch.Tensor], None],
) -> None:
    """Call ``before(layer, x)`` ahead of every stub block run."""

    forward_training = _Layer.forward_training

    def block(layer: _Layer, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        before(layer, x)
        return forward_training(layer, x, positions)

    monkeypatch.setattr(_Layer, "forward_training", block)


def test_full_offload_orders_copies_and_reuses_events_across_microbatches(
    fake_cuda: FakeCuda,
) -> None:
    runner, _offload = _offload_runner(_StubModel(torch.float32))
    assert len(fake_cuda.events) == 4  # Ready/done for each of the two offloaded layers.
    for _ in range(2):
        fake_cuda.log.clear()
        _assert_trains_like_eager(runner)
        assert len(fake_cuda.events) == 4
        assert fake_cuda.log == [
            ("training", "record", "e0"),
            ("copy", "wait_event", "e0"),
            ("copy", "issue", ""),
            ("copy", "record", "e2"),
            ("training", "record", "e1"),
            ("copy", "wait_event", "e1"),
            ("copy", "issue", ""),
            ("copy", "record", "e3"),
            ("training", "wait_event", "e2"),
            ("training", "wait_event", "e3"),
            ("training", "record", "e1"),
            ("copy", "wait_event", "e1"),
            ("copy", "issue", ""),
            ("copy", "record", "e3"),
            ("training", "wait_event", "e3"),
            ("training", "record", "e0"),
            ("copy", "wait_event", "e0"),
            ("copy", "issue", ""),
            ("copy", "record", "e2"),
            ("training", "wait_event", "e2"),
        ]
        runner.discard()
    runner.close()
    assert fake_cuda.log[-1] == ("copy", "synchronize", "")


def test_full_offload_keeps_only_the_last_input_until_backward(
    monkeypatch: pytest.MonkeyPatch,
    fake_cuda: FakeCuda,
) -> None:
    model = _StubModel(torch.float32)
    offload = _offload_policy(model, 0, fake_cuda)
    inputs: list[tuple[weakref.ref[torch.Tensor], torch.Tensor]] = []
    # The earlier layers' inputs still alive as each layer starts its forward.
    alive: list[list[int]] = []

    def record(_layer: _Layer, x: torch.Tensor) -> None:
        if not torch.is_grad_enabled():
            alive.append([index for index, (ref, _) in enumerate(inputs) if ref() is not None])
            inputs.append((weakref.ref(x), x.clone()))

    _wrap_blocks(monkeypatch, record)
    loss = _stub_loss(model)

    # The forward frees each offloaded input one layer after it was saved: as a layer
    # starts, only the previous layer's input, whose copy may still run, is alive beside
    # its own, and the decoder input (layer 0's), which the model holds until its
    # forward returns.
    assert alive == [[], [0], [0, 1], [0, 2], [0, 3], [0, 4]]
    # Between forward and backward the arena holds every input but the last, byte for
    # byte, and nothing else holds them; the last stays for backward to read first.
    assert [reference() is None for reference, _ in inputs] == [True] * 5 + [False]
    for slot, (_, value) in zip(offload._slots, inputs[:-1], strict=True):
        assert torch.equal(slot[: value.numel()], value.flatten())
    with torch.autograd.set_multithreading_enabled(False):
        loss.backward()
    offload.require_drained()
    offload.close()


def test_full_offload_refuses_an_input_written_after_it_was_saved(
    monkeypatch: pytest.MonkeyPatch,
    fake_cuda: FakeCuda,
) -> None:
    model = _StubModel(torch.float32)
    runner, offload = _offload_runner(model)
    inputs: dict[int, torch.Tensor] = {}

    def write_the_previous_input(layer: _Layer, x: torch.Tensor) -> None:
        index = model.layers.index(layer)
        if not torch.is_grad_enabled():
            inputs[index] = x
            if index == 4:
                # Layer 3's input is being copied to the arena.
                inputs[3].add_(1)

    with monkeypatch.context() as patch:
        _wrap_blocks(patch, write_the_previous_input)
        with pytest.raises(RuntimeError, match="modified in place after it was saved"):
            forward_backward(runner, _batch())

    # The runner dropped the failed microbatch's saved inputs and trains on.
    offload.require_drained()
    _assert_trains_like_eager(runner)
    runner.close()


def test_full_offload_refuses_a_last_input_written_before_backward(
    monkeypatch: pytest.MonkeyPatch,
    fake_cuda: FakeCuda,
) -> None:
    model = _StubModel(torch.float32)
    runner, offload = _offload_runner(model)
    last_inputs: list[torch.Tensor] = []

    def record_the_last_input(layer: _Layer, x: torch.Tensor) -> None:
        if layer is model.layers[-1] and not torch.is_grad_enabled():
            last_inputs.append(x)

    training_loss = model.training_loss

    def write_after_forward(*args: object, **kwargs: object) -> torch.Tensor:
        loss = training_loss(*args, **kwargs)
        last_inputs[0].mul_(2)
        return loss

    with monkeypatch.context() as patch:
        _wrap_blocks(patch, record_the_last_input)
        patch.setattr(model, "training_loss", write_after_forward)
        with pytest.raises(RuntimeError, match="modified in place after it was saved"):
            forward_backward(runner, _batch())

    offload.require_drained()
    _assert_trains_like_eager(runner)
    runner.close()


def test_full_offload_aborts_a_recompute_failure_with_a_fetch_in_flight(
    monkeypatch: pytest.MonkeyPatch,
    fake_cuda: FakeCuda,
) -> None:
    model = _StubModel(torch.float32)
    runner, offload = _offload_runner(model)
    fetched: list[weakref.ref[torch.Tensor]] = []
    copy_in = HostBoundaryOffload._copy_in

    def record_fetch(self: HostBoundaryOffload, saved: object, consumer: object) -> None:
        copy_in(self, saved, consumer)
        fetched.append(weakref.ref(saved.fetched))

    def fail_recompute(layer: _Layer, _x: torch.Tensor) -> None:
        # Reading layer 5's input fetched layer 4's; it is still unread.
        if layer is model.layers[5] and torch.is_grad_enabled():
            raise RuntimeError("injected recompute failure")

    with monkeypatch.context() as patch:
        patch.setattr(HostBoundaryOffload, "_copy_in", record_fetch)
        _wrap_blocks(patch, fail_recompute)
        with pytest.raises(RuntimeError, match="injected recompute failure") as failure:
            forward_backward(runner, _batch())
        # The failure's frames still reach the autograd graph and its saved inputs, yet
        # abort synchronized the copy stream and dropped the fetch.
        assert failure.value is not None
        assert len(fetched) == 1 and fetched[0]() is None
    assert fake_cuda.log[-1] == ("copy", "synchronize", "")
    offload.require_drained()
    _assert_trains_like_eager(runner)
    runner.close()


def test_full_offload_refuses_a_microbatch_whose_backward_skips_its_inputs(
    monkeypatch: pytest.MonkeyPatch,
    fake_cuda: FakeCuda,
) -> None:
    model = _StubModel(torch.float32)
    runner, offload = _offload_runner(model)
    training_loss = model.training_loss

    def loss_beside_the_layers(*args: object, **kwargs: object) -> torch.Tensor:
        training_loss(*args, **kwargs)
        # Backward reaches an adapter but no checkpoint: their saved inputs stay unread.
        return model.layers[3].lora_a.sum()

    with monkeypatch.context() as patch:
        patch.setattr(model, "training_loss", loss_beside_the_layers)
        with pytest.raises(RuntimeError, match="outlived its microbatch"):
            forward_backward(runner, _batch())

    offload.require_drained()
    _assert_trains_like_eager(runner)
    runner.close()


def test_full_offload_refuses_padding_its_slots_cannot_hold_before_any_microbatch(
    fake_cuda: FakeCuda,
) -> None:
    model = _StubModel(torch.float32)
    runner, offload = _offload_runner(model)
    forward_backward(runner, _batch())
    accumulated = _gradients(runner)
    batch = _batch()
    # Three rows padded to 200 tokens feed 3 x 199 tokens to each layer; the slots hold
    # the batching bound, 8 samples of at most 64.
    tokens = batch.tokens
    over_padded = TrainingBatch(
        PaddedTokens(
            F.pad(tokens.token_ids, (0, 200 - tokens.token_ids.shape[1])),
            tokens.lengths,
            tokens.pad_token_id,
        ),
        F.pad(batch.targets, (0, 199 - batch.targets.shape[1])),
        SFT(),
    )

    with pytest.raises(TrainingInputError, match="199-token rows exceed the 512-token"):
        forward_backward(runner, over_padded)

    assert fake_cuda.operations("copy").count(("issue", "")) == 4
    trained = _gradients(runner)
    assert all(torch.equal(trained[name], accumulated[name]) for name in accumulated)
    runner.close()


def test_full_offload_refuses_a_forward_whose_inputs_backward_never_read(
    fake_cuda: FakeCuda,
) -> None:
    model = _StubModel(torch.float32)
    offload = _offload_policy(model, 3, fake_cuda)
    _stub_loss(model)

    with pytest.raises(RuntimeError, match="outlived its microbatch"):
        offload.require_drained()
    with pytest.raises(RuntimeError, match="outlived its microbatch"):
        _stub_loss(model)
    offload.abort()
    # A fresh step trains after the aborted one.
    _stub_loss(model).backward()
    offload.require_drained()
    offload.close()


def test_full_offload_requires_a_contiguous_input(
    monkeypatch: pytest.MonkeyPatch,
    fake_cuda: FakeCuda,
) -> None:
    model = _StubModel(torch.float32)
    offload = _offload_policy(model, 3, fake_cuda)
    forward_training = _Layer.forward_training

    def strided(layer: _Layer, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        output = forward_training(layer, x, positions)
        if layer is model.layers[3]:
            output = output.transpose(0, 1).contiguous().transpose(0, 1)
        return output

    monkeypatch.setattr(_Layer, "forward_training", strided)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _stub_loss(model)
    offload.close()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        pytest.param({"tokens": 14}, "has 15 tokens; the offload holds 14", id="tokens"),
        pytest.param({"dtype": torch.bfloat16}, "the offload holds torch.bfloat16", id="dtype"),
    ],
)
def test_full_offload_refuses_an_input_its_slots_do_not_hold(
    fake_cuda: FakeCuda,
    options: dict[str, object],
    message: str,
) -> None:
    model = _StubModel(torch.float32)
    offload = _offload_policy(model, 3, fake_cuda, **options)

    with pytest.raises(RuntimeError, match=message):
        _stub_loss(model)
    offload.close()


# ============================== Qwen3's training loops ==============================


class _Embedding:
    def forward_training(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32).unsqueeze(-1)


class _Norm:
    def forward_training(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x, None


class _DecoderLayer:
    """Stands in for ``Qwen3DecoderLayer``, whose mixer the model looks up on the class."""

    def __init__(self, *, trainable: bool = True) -> None:
        self.weight = torch.tensor(2.0, requires_grad=trainable)
        self.input_gradient_flags: list[bool] = []
        self.metadata: list[tuple[torch.Tensor, Packing | None]] = []

    def mixer_training(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        packing: Packing | None,
    ) -> torch.Tensor:
        self.input_gradient_flags.append(x.requires_grad)
        self.metadata.append((positions, packing))
        return torch.sin(x).square() * self.weight

    def mlp_training(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.fixture
def _stub_qwen3_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Qwen3DecoderLayer, "mixer_training", _DecoderLayer.mixer_training)


def _qwen3_model(
    layer: _DecoderLayer,
    policy: ReentrantActivationCheckpoint | None,
) -> Qwen3Model:
    model = object.__new__(Qwen3Model)
    model._activation_checkpoint = None
    model.embed_tokens = _Embedding()
    model.layers = SimpleNamespace(op_list=[layer])
    model.norm = _Norm()
    if policy is not None:
        model.enable_activation_checkpointing(policy)
    return model


@pytest.mark.usefixtures("_stub_qwen3_blocks")
@pytest.mark.parametrize("checkpointed, expected_calls", ((False, 1), (True, 2)))
def test_qwen3_padded_training_runs_its_blocks_under_the_policy(
    checkpointed: bool,
    expected_calls: int,
) -> None:
    layer = _DecoderLayer()
    model = _qwen3_model(layer, ReentrantActivationCheckpoint(0) if checkpointed else None)

    output = model.forward_training(torch.tensor([[1, 2, 3]], dtype=torch.int64), None)
    output.sum().backward()

    assert len(layer.metadata) == expected_calls
    for positions, packing in layer.metadata:
        assert positions.tolist() == [0, 1, 2]
        assert packing is None
    # The first layer's input carries no gradient: backward computes no input gradient.
    assert layer.input_gradient_flags == [False] * expected_calls
    assert layer.weight.grad is not None
    expected_grad = torch.sin(torch.tensor([1.0, 2.0, 3.0])).square().sum()
    assert torch.equal(layer.weight.grad, expected_grad)


@pytest.mark.usefixtures("_stub_qwen3_blocks")
@pytest.mark.parametrize("checkpointed, expected_calls", ((False, 1), (True, 2)))
def test_qwen3_packed_training_preserves_metadata_across_block_recomputation(
    checkpointed: bool,
    expected_calls: int,
) -> None:
    layer = _DecoderLayer()
    model = _qwen3_model(layer, ReentrantActivationCheckpoint(0) if checkpointed else None)
    positions = torch.tensor([0, 1, 0], dtype=torch.int64)
    packing = Packing(positions, torch.tensor([0, 2, 3], dtype=torch.int32), 2)

    output = model.forward_training(torch.tensor([1, 2, 3], dtype=torch.int64), packing)
    output.sum().backward()

    assert layer.input_gradient_flags == [False] * expected_calls
    assert layer.weight.grad is not None
    expected_grad = torch.sin(torch.tensor([1.0, 2.0, 3.0])).square().sum()
    assert torch.equal(layer.weight.grad, expected_grad)
    assert len(layer.metadata) == expected_calls
    first_positions, first_packing = layer.metadata[0]
    assert first_positions is positions
    assert first_packing is packing
    # Recomputation reads the same metadata.
    for seen_positions, seen_packing in layer.metadata:
        assert torch.equal(seen_positions, positions)
        assert seen_positions.data_ptr() == positions.data_ptr()
        assert seen_packing is packing


@pytest.mark.usefixtures("_stub_qwen3_blocks")
@pytest.mark.parametrize("checkpointed", (False, True))
def test_qwen3_frozen_decoder_body_preserves_trainable_head_gradient(
    checkpointed: bool,
) -> None:
    layer = _DecoderLayer(trainable=False)
    # No layer trains, so the runner's policy starts past the last one.
    model = _qwen3_model(layer, ReentrantActivationCheckpoint(1) if checkpointed else None)
    head = torch.tensor(3.0, requires_grad=True)

    hidden_states = model.forward_training(torch.tensor([[1, 2, 3]], dtype=torch.int64), None)
    (hidden_states * head).sum().backward()

    assert len(layer.metadata) == 1
    assert head.grad is not None
    expected_grad = 2 * torch.sin(torch.tensor([1.0, 2.0, 3.0])).square().sum()
    assert torch.equal(head.grad, expected_grad)
    assert layer.weight.grad is None


# ========================= Adapters by decoder layer, real models =========================


def _adapter_tensors(adapters: Adapter, prefix: str) -> set[int]:
    """The identities of the LoRA factors ``adapters`` attached under the path ``prefix``."""

    return {
        id(tensor) for name, tensor in adapters.named_parameters() if name.startswith(f"{prefix}.")
    }


def _qwen3_config(_folder: Path) -> PretrainedConfig:
    return Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        hidden_size=128,
        intermediate_size=256,
        vocab_size=256,
        max_position_embeddings=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
        tie_word_embeddings=False,
    )


def _qwen3_5_config(folder: Path) -> PretrainedConfig:
    return load_hf_config(TINY_QWEN3_5_CONFIG, folder)


@pytest.mark.parametrize(
    ("hf_config", "targets", "hidden_size"),
    [
        pytest.param(_qwen3_config, ("attention", "mlp"), 128, id="qwen3"),
        pytest.param(
            _qwen3_5_config, Qwen3_5ForConditionalGeneration.LORA_TARGETS, 32, id="qwen3_5"
        ),
    ],
)
def test_models_declare_each_decoder_layers_adapters_and_residual_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hf_config: Callable[[Path], PretrainedConfig],
    targets: tuple[str, ...],
    hidden_size: int,
) -> None:
    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)
    reference_ops.bind_vision_references(monkeypatch)
    adapters = Adapter(LoRAConfig(targets=targets))  # type: ignore[arg-type]
    with torch.device("meta"), default_dtype(torch.bfloat16):
        config = hf_config(tmp_path)
        model = create_model(
            get_model_class(config.architectures[0]),
            checkpoint_facts(config),
            site=site(),
        )
    adapters.attach(model)
    adapters.materialize(device=torch.device("cpu"), dtype=torch.float32)

    # The decoder layers carry the embedding output: the model's width, in its dtype. The
    # full_offload arena takes this dtype's itemsize.
    assert model.residual_stream_layout() == (hidden_size, torch.bfloat16)

    groups = AdaptedTrainables(model, adapters).decoder_layer_trainable_parameters()

    layers = model.model.layers.op_list
    paths = {id(op): path for path, op in model.named_ops()}
    assert len(groups) == len(layers)
    for group, layer in zip(groups, layers, strict=True):
        assert group
        assert len({id(tensor) for tensor in group}) == len(group)
        assert {id(tensor) for tensor in group} == _adapter_tensors(adapters, paths[id(layer)])
    assert sum(len(group) for group in groups) == len(adapters.named_parameters())
