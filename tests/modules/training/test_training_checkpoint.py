from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import cti.training._checkpoint as checkpoint_module
from cti.config import (
    AdamWConfig,
    BatchLayout,
    LoRAConfig,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.layers import Adapter
from cti.models import CheckpointSource, model_signature
from cti.models.config import RotaryConfig
from cti.models.qwen3.config import Qwen3Config
from cti.training import (
    SFT,
    CheckpointFormatError,
    PaddedTokens,
    TrainingBatch,
)
from cti.training._checkpoint import ADAMW_BACKEND_STEP_SATURATION, TrainingCheckpoint
from cti.training.runner import TrainingRunner
from tests.support.training import (
    TrainingModelStub,
    forward_backward,
    host_runner,
    optim_step,
)

_CPU = torch.device("cpu")
_SELECTED_NAMES = ("selected_fp32", "selected_bf16")


def _padded_batching() -> TrainingBatching:
    return TrainingBatching(
        BatchLayout.PADDED,
        StaticBatching(microbatch_size=8, max_tokens_per_sample=64),
    )


def _signature_config() -> Qwen3Config:
    return Qwen3Config(
        num_layers=1,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=4,
        hidden_size=8,
        vocab_size=16,
        intermediate_size=12,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(
            head_dim=4,
            rotary_dim=4,
            max_position=128,
            base=10_000.0,
            scaling=None,
        ),
        tie_word_embeddings=False,
    )


def test_model_signature_is_canonical_and_bound_to_its_source() -> None:
    source = CheckpointSource("/cache/a", "huggingface", "Qwen/Qwen3-0.6B", "a" * 40)
    tensors = (
        ("model.weight", torch.empty(8, 8, dtype=torch.bfloat16)),
        ("adapter.lora_A", torch.empty(2, 8, dtype=torch.bfloat16)),
    )
    kwargs = {
        "source": source,
        "config": _signature_config(),
        "dtype": torch.bfloat16,
        "trainable": Adapter(
            LoRAConfig(targets=("mlp", "attention"), rank=2, alpha=4.0)
        ).identity(),
    }

    signature = model_signature(named_tensors=tensors, **kwargs)
    reordered = model_signature(
        named_tensors=reversed(tensors),
        **{
            **kwargs,
            "trainable": Adapter(
                LoRAConfig(targets=("attention", "mlp"), rank=2, alpha=4.0)
            ).identity(),
        },
    )

    assert signature == reordered
    # A tensor-parallel rank's shard of the same model resumes the same checkpoints.
    shard = (("model.weight", torch.empty(4, 8, dtype=torch.bfloat16)), tensors[1])
    assert model_signature(named_tensors=shard, **kwargs) == signature
    assert signature.startswith("sha256:")
    assert len(signature) == len("sha256:") + 64
    # The same revision in another cache is the same checkpoint; another revision is not.
    moved = replace(source, folder="/elsewhere/a")
    assert model_signature(named_tensors=tensors, **{**kwargs, "source": moved}) == signature
    for changed in (
        {"source": replace(source, revision="b" * 40)},
        {"source": CheckpointSource("/cache/a")},
        {"dtype": torch.float32},
        {"named_tensors": (*tensors[:-1], ("adapter.lora_B", tensors[-1][1]))},
    ):
        assert model_signature(**{"named_tensors": tensors, **kwargs, **changed}) != signature


class _MixedPrecisionModel(TrainingModelStub):
    """Small deterministic model covering both supported update representations."""

    def __init__(
        self,
        *,
        seed: int = 20260717,
        width: int = 4,
        low_precision_dtype: torch.dtype = torch.bfloat16,
        selected_names: tuple[str, str] = _SELECTED_NAMES,
        reverse_selection_order: bool = False,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.width = width
        self.selected_names = selected_names
        self.reverse_selection_order = reverse_selection_order
        self.selected_fp32 = torch.nn.Parameter(
            torch.randn(7, width, generator=generator, dtype=torch.float32),
            requires_grad=False,
        )
        self.selected_low_precision = torch.nn.Parameter(
            torch.randn(7, width, generator=generator, dtype=low_precision_dtype),
            requires_grad=False,
        )
        self.frozen = torch.nn.Parameter(
            torch.randn(7, width, generator=generator, dtype=torch.float32),
            requires_grad=False,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        selected = [
            (self.selected_names[0], self.selected_fp32),
            (self.selected_names[1], self.selected_low_precision),
        ]
        if self.reverse_selection_order:
            selected.reverse()
        return dict((*selected, ("frozen", self.frozen)))

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        selected = [
            (self.selected_names[0], self.selected_fp32),
            (self.selected_names[1], self.selected_low_precision),
        ]
        if self.reverse_selection_order:
            selected.reverse()
        return tuple(selected)

    def forward_training(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = F.one_hot(input_ids.remainder(self.width), num_classes=self.width)
        return (
            F.linear(features.float(), self.trained(self.selected_fp32))
            + F.linear(
                features.to(self.selected_low_precision.dtype),
                self.trained(self.selected_low_precision),
            ).float()
            + F.linear(features.float(), self.frozen)
        )


def _optimizer_config(*, lr: float = 0.03) -> AdamWConfig:
    return AdamWConfig(
        lr=lr,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.02,
        max_grad_norm=0.25,
    )


def _runner(
    *,
    seed: int = 20260717,
    width: int = 4,
    low_precision_dtype: torch.dtype = torch.bfloat16,
    selected_names: tuple[str, str] = _SELECTED_NAMES,
    reverse_selection_order: bool = False,
    config: AdamWConfig | None = None,
) -> tuple[_MixedPrecisionModel, TrainingRunner]:
    model = _MixedPrecisionModel(
        seed=seed,
        width=width,
        low_precision_dtype=low_precision_dtype,
        selected_names=selected_names,
        reverse_selection_order=reverse_selection_order,
    )
    return model, host_runner(
        model,
        TrainingConfig(
            TrainablePlan(selected_names), config or _optimizer_config(), _padded_batching()
        ),
        trainables=model,
        checks=True,
    )


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


def _commit(runner: TrainingRunner, batch: TrainingBatch) -> float:
    result = forward_backward(runner, batch)
    return optim_step(runner, result.normalization_count)


def _capture(
    runner: TrainingRunner, *, num_optimizer_steps: int, model_revision: int
) -> TrainingCheckpoint:
    return runner.checkpoint(num_optimizer_steps=num_optimizer_steps, model_revision=model_revision)


def _assert_tree_exact(actual: object, expected: object) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert (actual.shape, actual.dtype, actual.device) == (
            expected.shape,
            expected.dtype,
            expected.device,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif is_dataclass(expected) and not isinstance(expected, type):
        for field in fields(expected):
            _assert_tree_exact(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_exact(actual[key], expected[key])
    elif isinstance(expected, Sequence) and not isinstance(expected, str):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_tree_exact(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_rejected_without_mutation(
    runner: TrainingRunner, restore: Callable[[], None], *, match: str
) -> None:
    before = _capture(runner, num_optimizer_steps=1, model_revision=1)
    with pytest.raises(ValueError, match=match):
        restore()
    _assert_tree_exact(_capture(runner, num_optimizer_steps=1, model_revision=1), before)


@pytest.mark.parametrize("steps", [0, 2])
def test_a_saved_checkpoint_resumes_exactly_into_another_runner(tmp_path: Path, steps: int) -> None:
    source_model, source = _runner()
    for index, batch in enumerate(_BATCHES[:steps]):
        _commit(source, batch)
        if index == 0:
            source.publish()
    checkpoint = _capture(source, num_optimizer_steps=steps, model_revision=9)
    path = tmp_path / "training.pt"
    checkpoint.save(path)
    # The BF16 parameter resumes from its FP32 master, not from the BF16 value.
    master = checkpoint.optimizer.master[1]
    assert steps == 0 or not torch.equal(master, source_model.selected_low_precision.float())
    for live, master in zip(checkpoint.optimizer.live, checkpoint.optimizer.master, strict=True):
        assert master.dtype == torch.float32
        if steps:
            assert not torch.equal(live.float(), master)

    # The same frozen weights, trained on another batch first.
    target_model, target = _runner()
    _commit(target, _BATCHES[2])
    live = (target_model.selected_fp32, target_model.selected_low_precision)
    frozen = target_model.frozen.detach().clone()
    loaded = TrainingCheckpoint.load(path)
    target.restore(loaded)

    _assert_tree_exact(loaded, checkpoint)
    _assert_tree_exact(_capture(target, num_optimizer_steps=steps, model_revision=9), checkpoint)
    assert (target_model.selected_fp32, target_model.selected_low_precision) == live
    for actual, expected in zip(live, checkpoint.optimizer.live, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(target_model.frozen, frozen, rtol=0, atol=0)
    assert _commit(target, _BATCHES[0]) == _commit(source, _BATCHES[0])
    _assert_tree_exact(
        _capture(target, num_optimizer_steps=steps + 1, model_revision=10),
        _capture(source, num_optimizer_steps=steps + 1, model_revision=10),
    )


def test_a_checkpoint_past_the_fp32_adamw_step_saturation_round_trips(tmp_path: Path) -> None:
    _model, runner = _runner()
    _commit(runner, _BATCHES[0])
    saturated = _capture(runner, num_optimizer_steps=1, model_revision=1)
    for state in saturated.optimizer.adamw["state"].values():
        state["step"].fill_(ADAMW_BACKEND_STEP_SATURATION)
    runner.restore(saturated)
    path = tmp_path / "training.pt"
    _capture(runner, num_optimizer_steps=ADAMW_BACKEND_STEP_SATURATION + 1, model_revision=1).save(
        path
    )

    assert TrainingCheckpoint.load(path).num_optimizer_steps == ADAMW_BACKEND_STEP_SATURATION + 1


def _corrupt(raw: dict, case: str) -> object:
    state = raw["adamw"]["state"][0]
    if case == "version 4":
        return {**raw, "version": 4}
    if case == "not a dictionary":
        return [raw]
    if case == "unknown key":
        return {**raw, "tp_size": 1}
    if case == "step":
        state["step"] = torch.tensor(3.0)
    elif case == "non-finite moment":
        state["exp_avg"] = torch.full_like(state["exp_avg"], torch.nan)
    elif case == "missing BF16 master":
        raw["master"] = (raw["master"][0], None)
    elif case == "missing FP32 master":
        raw["master"] = (None, raw["master"][1])
    elif case == "duplicate name":
        raw["names"] = (raw["names"][0],) * 2
    return raw


@pytest.mark.parametrize(
    "case",
    [
        "version 4",
        "not a dictionary",
        "unknown key",
        "step",
        "non-finite moment",
        "missing BF16 master",
        "missing FP32 master",
        "duplicate name",
        "truncated",
    ],
)
def test_load_refuses_a_malformed_or_older_file(tmp_path: Path, case: str) -> None:
    _model, runner = _runner()
    _commit(runner, _BATCHES[0])
    path = tmp_path / "training.pt"
    _capture(runner, num_optimizer_steps=1, model_revision=1).save(path)
    if case == "truncated":
        path.write_bytes(path.read_bytes()[:100])
    else:
        torch.save(_corrupt(torch.load(path, weights_only=True), case), path)

    # An older version is refused naming what its successor changed.
    message = (
        "version 4 .* reads version 5, which keeps every parameter's FP32 master"
        if case == "version 4"
        else None
    )
    with pytest.raises(CheckpointFormatError, match=message):
        TrainingCheckpoint.load(path)


@pytest.mark.parametrize(
    ("target_factory", "message"),
    [
        (lambda: _runner(selected_names=("renamed_fp32", "renamed_bf16"))[1], "parameters"),
        (lambda: _runner(reverse_selection_order=True)[1], "parameters"),
        (lambda: _runner(width=5)[1], "shape"),
        (lambda: _runner(low_precision_dtype=torch.float32)[1], "dtype"),
        (lambda: _runner(config=_optimizer_config(lr=0.04))[1], "configuration"),
    ],
    ids=("names", "order", "shape", "dtype", "optimizer-config"),
)
def test_an_incompatible_checkpoint_fails_before_any_change(
    target_factory: Callable[[], TrainingRunner], message: str
) -> None:
    _source_model, source = _runner()
    _commit(source, _BATCHES[0])
    checkpoint = _capture(source, num_optimizer_steps=1, model_revision=1)
    target = target_factory()

    _assert_rejected_without_mutation(target, lambda: target.restore(checkpoint), match=message)


def test_capture_and_restore_refuse_pending_accumulation() -> None:
    _source_model, source = _runner()
    checkpoint = _capture(source, num_optimizer_steps=0, model_revision=0)
    _target_model, target = _runner()
    forward_backward(target, _BATCHES[0])

    with pytest.raises(RuntimeError):
        _capture(target, num_optimizer_steps=0, model_revision=0)
    with pytest.raises(RuntimeError):
        target.restore(checkpoint)
    assert target.holds_gradients
    target.discard()


@pytest.mark.parametrize("failure_stage", ["serialize", "replace"])
def test_atomic_save_preserves_existing_checkpoint_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    _model, runner = _runner()
    destination = tmp_path / "training.pt"
    _capture(runner, num_optimizer_steps=0, model_revision=0).save(destination)
    bytes_before = destination.read_bytes()
    entries_before = set(tmp_path.iterdir())

    _commit(runner, _BATCHES[0])
    replacement = _capture(runner, num_optimizer_steps=1, model_revision=1)
    if failure_stage == "serialize":

        def fail_save(_value: object, stream: object, **_kwargs: object) -> None:
            stream.write(b"partial")  # type: ignore[attr-defined]
            raise OSError("injected serialization failure")

        monkeypatch.setattr(checkpoint_module.torch, "save", fail_save)
    else:

        def fail_replace(source: object, target: object) -> None:
            assert Path(os.fspath(source)).parent == Path(os.fspath(target)).parent
            raise OSError("injected replace failure")

        monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected"):
        replacement.save(destination)

    assert destination.read_bytes() == bytes_before
    assert set(tmp_path.iterdir()) == entries_before
