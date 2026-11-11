"""Qwen3-MoE and Qwen3.5-MoE LoRA inference and training against Transformers FP32 on CPU.

Tiny models adapt every LoRA target, each routed expert's projections included
(``tests.support.moe_lora_models``). Inference compares prefill and decode logits, and a new
adapter must leave them bitwise unchanged; training through the frozen sparse blocks compares
padded and packed SFT logits, losses, adapter gradients and three AdamW steps with
Transformers' per-expert loop, and activation checkpointing must leave the gradients bitwise
unchanged. Kernel operations use CPU references.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoConfig

from cti.config import ActivationCheckpointMode, BatchLayout
from cti.models import create_model
from cti.models.base import BaseLLMModel
from cti.models.config import UnimplementedConfigError
from cti.models.qwen3_moe.model import Qwen3MoeForCausalLM
from cti.training import SFTDatum, TrainingBatch, TrainingInputError
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.forward_rows import decode_entry, extend_entry
from tests.support.moe import ReferenceMoe
from tests.support.moe_lora_models import (
    FAMILIES,
    OPTIMIZER,
    PAD_TOKEN_ID,
    Checkpoint,
    Reference,
    all_rows,
    build_engine,
    sequence_logits,
    sft_batch,
    sft_data,
)
from tests.support.qwen3_5 import load_hf_config
from tests.support.qwen3_5_engine import EagerEngine, bind_reference_kernels
from tests.support.qwen3_5_moe import TINY_QWEN3_5_MOE_CONFIG
from tests.support.qwen3_moe import save_tiny_checkpoint
from tests.support.recompute_audit import assert_recomputed, record_block_runs
from tests.support.tp_shards import build_model, site
from tests.support.training import forward_backward, optim_step, synchronize_optimizer_parameters
from tests.support.training_ops import bind_training_references

_STEPS = 3
_LAYOUTS = pytest.mark.parametrize(
    "layout", [BatchLayout.PADDED, BatchLayout.PACKED], ids=lambda x: x.value
)


@pytest.fixture(autouse=True)
def _reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_reference_kernels(monkeypatch)
    bind_training_references(monkeypatch)


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def checkpoint(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Checkpoint:
    return Checkpoint.save(FAMILIES[request.param], tmp_path_factory.mktemp(request.param))


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    # FP32 on both sides; attention, the delta rule and the expert sums reorder additions.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4, msg=label)


def _inference_logits(model: BaseLLMModel, token_ids: torch.Tensor) -> torch.Tensor:
    """A prefill of all but the last four tokens, then four decode steps."""

    length = token_ids.shape[1]
    engine = EagerEngine(model, token_ids)
    with torch.inference_mode():
        return torch.cat(
            [
                engine.execute(extend_entry(0, 0, length - 4)),
                *(engine.execute(decode_entry(0, p)) for p in range(length - 4, length)),
            ]
        )


def _token_ids(checkpoint: Checkpoint) -> torch.Tensor:
    return torch.tensor([sft_data(checkpoint.family, 0)[0].token_ids], dtype=torch.int32)


def test_lora_inference_matches_transformers_with_the_adapter(checkpoint: Checkpoint) -> None:
    model, _, _ = checkpoint.engine(BatchLayout.PADDED)
    token_ids = _token_ids(checkpoint)

    logits = _inference_logits(model, token_ids)

    with torch.inference_mode():
        expected = Reference(checkpoint).model(input_ids=token_ids.long()).logits[0]
        base = checkpoint.transformers()(input_ids=token_ids.long()).logits[0]
    rows = slice(token_ids.shape[1] - 5, None)
    _assert_close(logits, expected[rows], "prefill and decode logits")
    # The adapter moves the logits far beyond the tolerance.
    assert float((expected[rows] - base[rows]).abs().max()) > 0.1


def test_a_new_adapter_leaves_the_logits_bitwise_unchanged(checkpoint: Checkpoint) -> None:
    adapted, adapters, _ = checkpoint.engine(BatchLayout.PADDED, adapter=False)
    assert any(".mlp.experts." in name for name, _ in adapters.named_parameters())
    base = build_model(
        AutoConfig.from_pretrained(checkpoint.folder), training=True, moe_backend=ReferenceMoe()
    )
    base.load_state_dict({name: tensor.clone() for name, tensor in adapted.state_dict().items()})

    token_ids = _token_ids(checkpoint)
    assert torch.equal(_inference_logits(adapted, token_ids), _inference_logits(base, token_ids))


@_LAYOUTS
def test_sft_trajectory_matches_transformers_fp32_autograd(
    checkpoint: Checkpoint, layout: BatchLayout
) -> None:
    family = checkpoint.family
    model, adapters, runner = checkpoint.engine(layout, microbatch_size=2)
    reference = Reference(checkpoint)

    for step in range(_STEPS):
        batch = sft_batch(family, layout, step)
        with torch.no_grad():
            engine_logits = sequence_logits(model, batch)
        result = forward_backward(runner, batch)
        reference_loss, reference_logits = reference.loss(sft_data(family, step))
        for index, (actual, expected) in enumerate(
            zip(engine_logits, reference_logits, strict=True)
        ):
            _assert_close(actual, expected, f"step {step} sequence {index} logits")
        _assert_close(result.loss_sum, reference_loss, f"step {step} loss")

        # By the engine's names; each tolerance relative to the tensor's own scale.
        expected = checkpoint.to_engine({n: p.grad for n, p in reference.parameters.items()})
        actual = dict(
            zip(
                runner.trainable_parameters.selected_names,
                runner.accumulated_gradients(),
                strict=True,
            )
        )
        assert actual.keys() == expected.keys()
        for name, gradient in expected.items():
            assert torch.count_nonzero(gradient) > 0, name
            scale = 2e-4 * float(gradient.abs().max())
            torch.testing.assert_close(actual[name], gradient, rtol=1e-3, atol=scale, msg=name)

        live_before = {name: value.detach().clone() for name, value in adapters.named_parameters()}
        optim_step(runner, result.normalization_count)
        for name, live in adapters.named_parameters():
            assert live.grad is None
            torch.testing.assert_close(live, live_before[name], rtol=0, atol=0)
        runner.publish()
        reference.step(result.normalization_count)
        expected = checkpoint.to_engine(
            {name: value.detach().clone() for name, value in reference.parameters.items()}
        )
        engine = adapters.state_dict()
        assert engine.keys() == expected.keys()
        for name, value in expected.items():
            # AdamW moves each element by about lr; a gradient near eps amplifies FP32
            # rounding, so each step continues from Transformers' adapter.
            torch.testing.assert_close(
                engine[name], value, rtol=0, atol=0.1 * OPTIMIZER.lr, msg=f"step {step} {name}"
            )
        adapters.load_state_dict(expected)
        synchronize_optimizer_parameters(runner, adapters.named_parameters())

    # No gradient ever reached a frozen base weight, the sparse blocks' included.
    assert all(
        not tensor.requires_grad and tensor.grad is None for tensor in model.state_dict().values()
    )


@pytest.mark.parametrize("last_layer_only", [False, True], ids=["padded-all", "packed-last"])
def test_activation_checkpointing_recomputes_from_the_first_trained_layer_to_the_same_gradients(
    checkpoint: Checkpoint, last_layer_only: bool
) -> None:
    last = checkpoint.family.last_layer
    layout, trains, first = (
        (BatchLayout.PACKED, lambda name: f"layers.{last}." in name, last)
        if last_layer_only
        else (BatchLayout.PADDED, None, 0)
    )
    batch = sft_batch(checkpoint.family, layout, 0)
    results = []
    for mode in (ActivationCheckpointMode.NONE, ActivationCheckpointMode.FULL):
        model, _, runner = checkpoint.engine(layout, activation_checkpointing=mode, trains=trains)
        layers = model.model.layers.op_list
        with record_block_runs(checkpoint.family.layer_class, layers) as runs:
            result = forward_backward(runner, batch)
        parameters = runner.trainable_parameters.named_parameters()
        results.append(
            (
                result.loss_sum,
                dict(zip((n for n, _ in parameters), runner.accumulated_gradients(), strict=True)),
                runs,
            )
        )

    (loss, gradients, runs), (checkpointed_loss, checkpointed_gradients, checkpointed_runs) = (
        results
    )
    # Recomputed sparse blocks route each token as their first run did.
    assert_recomputed(runs, checkpointed_runs, first=first)
    torch.testing.assert_close(checkpointed_loss, loss, rtol=0, atol=0)
    assert checkpointed_gradients.keys() == gradients.keys()
    for name, gradient in gradients.items():
        assert torch.count_nonzero(gradient) > 0, name
        torch.testing.assert_close(checkpointed_gradients[name], gradient, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("family", "freeze_breaker"),
    [
        pytest.param(
            "qwen3_moe",
            lambda model: model.model.layers.op_list[0].mlp.experts.gate_up_proj,
            id="qwen3_moe-experts",
        ),
        pytest.param(
            "qwen3_5_moe",
            lambda model: model.iter_recurrent_layers()[0].A_log,
            id="qwen3_5_moe-gdn-a-log",
        ),
    ],
)
def test_a_base_weight_that_requires_grad_fails_the_training_forward(
    tmp_path: Path, family: str, freeze_breaker: Callable[[BaseLLMModel], torch.Tensor]
) -> None:
    checkpoint = Checkpoint.save(FAMILIES[family], tmp_path)
    model, _, _ = checkpoint.engine(BatchLayout.PADDED)
    freeze_breaker(model).requires_grad_(True)

    with pytest.raises(ValueError, match="frozen"):
        model.training_hidden_states(
            all_rows(sft_batch(checkpoint.family, BatchLayout.PADDED, 0)).input_ids, None
        )


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"attention_dropout": 0.1}, "attention_dropout"),
        ({"output_router_logits": True, "router_aux_loss_coef": 0.001}, "output_router_logits"),
    ],
)
def test_qwen3_moe_training_refuses_a_training_only_field_it_would_ignore(
    tmp_path: Path, changes: dict[str, Any], field: str
) -> None:
    save_tiny_checkpoint(tmp_path, **changes)
    hf_config = AutoConfig.from_pretrained(tmp_path)
    with torch.device("meta"):
        create_model(
            Qwen3MoeForCausalLM, checkpoint_facts(hf_config), site=site(moe_backend=ReferenceMoe())
        )
        with pytest.raises(UnimplementedConfigError, match=f"{field}=") as error:
            create_model(
                Qwen3MoeForCausalLM,
                checkpoint_facts(hf_config),
                site=site(training=True, moe_backend=ReferenceMoe()),
            )
    assert error.value.field == field


@_LAYOUTS
def test_a_qwen3_5_moe_image_text_checkpoint_refuses_vision_placeholders(
    tmp_path: Path, layout: BatchLayout
) -> None:
    family = FAMILIES["qwen3_5_moe"]
    generator = torch.Generator().manual_seed(5)

    def state(model: BaseLLMModel) -> dict[str, torch.Tensor]:
        return {
            name: torch.randn(tensor.shape, generator=generator).to(tensor.dtype) * 0.1
            for name, tensor in model.state_dict().items()
        }

    hf_config = load_hf_config(TINY_QWEN3_5_MOE_CONFIG, tmp_path)
    _, _, runner = build_engine(family, hf_config, state, layout)
    padded = layout is BatchLayout.PADDED
    for placeholder in (
        TINY_QWEN3_5_MOE_CONFIG["image_token_id"],
        TINY_QWEN3_5_MOE_CONFIG["video_token_id"],
    ):
        data = (SFTDatum(token_ids=(3, placeholder, 5, 7), target_mask=(False, False, True, True)),)
        padding = {"pad_token_id": PAD_TOKEN_ID} if padded else {}
        batch = TrainingBatch.sft(data, layout, **padding)
        with pytest.raises(TrainingInputError, match=rf"placeholder tokens \[{placeholder}\]"):
            forward_backward(runner, batch)

    result = forward_backward(runner, sft_batch(family, layout, 0))

    assert torch.isfinite(result.loss_sum)
