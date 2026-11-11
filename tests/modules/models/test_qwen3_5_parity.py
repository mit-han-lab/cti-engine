"""Qwen3.5 against transformers 5.16.1 on CPU: tiny random FP32 text checkpoints, loaded through
the checkpoint mapping and run over the runner's recurrent and paged KV pools with the CPU
reference operations, give Transformers' logits at the same positions."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from cti.config import LoRAConfig
from cti.layers import Adapter
from cti.models.base import BaseLLMModel
from cti.models.qwen3_5.config import Qwen3_5Config
from cti.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from tests.support.forward_rows import decode_entry, extend_entry
from tests.support.qwen3_5 import (
    INTERLEAVED_QWEN3_5_CONFIG,
    TINY_QWEN3_5_CONFIG,
    load_hf_config,
    randomize_hf_weights,
    text_only_config,
)
from tests.support.qwen3_5_engine import EagerEngine, bind_reference_kernels, load_model
from tests.support.qwen3_5_lora import (
    TEXT_ONLY_PREFIX,
    Qwen3_5LoRAMapping,
    merged_checkpoint,
    random_peft_state,
)
from tests.support.training import training_logits
from tests.support.training_ops import bind_training_references

CPU = torch.device("cpu")
# Longer than one 64-token chunk of Transformers' chunked delta rule.
PROMPT_LEN = 70
NUM_REQUESTS = 3
LORA_RANK = 4
LORA_ALPHA = 2.0


@pytest.fixture(autouse=True)
def _reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_reference_kernels(monkeypatch)


@dataclass(frozen=True)
class _Checkpoint:
    folder: Path
    reference: PreTrainedModel

    def reference_outputs(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transformers' final hidden states and logits for one whole sequence."""

        with torch.no_grad():
            hidden = self.reference.model(input_ids=token_ids[None].long()).last_hidden_state
            return hidden[0], self.reference.lm_head(hidden)[0]


def _save_checkpoint(
    config: Mapping[str, Any],
    heads: tuple[int, int],
    tmp_path_factory: pytest.TempPathFactory,
) -> _Checkpoint:
    num_key_heads, num_value_heads = heads
    published = copy.deepcopy(dict(config))
    published["text_config"].update(
        linear_num_key_heads=num_key_heads,
        linear_num_value_heads=num_value_heads,
    )
    folder = tmp_path_factory.mktemp("qwen3_5")
    config = load_hf_config(text_only_config(published), folder)
    reference = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    # The engine declares a BF16 convolution weight.
    randomize_hf_weights(reference, conv_dtype=torch.bfloat16)
    reference.save_pretrained(folder)
    return _Checkpoint(folder, reference)


# Gated DeltaNet (key heads, value heads). With two key heads, value head j reading key
# head j // 2 (transformers' repeat_interleave) differs from reading key head j % 2.
@pytest.fixture(scope="module", params=[(1, 2), (2, 4)], ids=["heads-1x2", "heads-2x4"])
def checkpoint(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> _Checkpoint:
    return _save_checkpoint(TINY_QWEN3_5_CONFIG, request.param, tmp_path_factory)


@pytest.fixture(scope="module")
def interleaved_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> _Checkpoint:
    return _save_checkpoint(INTERLEAVED_QWEN3_5_CONFIG, (2, 4), tmp_path_factory)


def _adapted(
    checkpoint: _Checkpoint, *, training: bool = False
) -> tuple[BaseLLMModel, _Checkpoint]:
    """The engine model with a non-zero PEFT-shaped adapter, and Transformers with it merged."""

    targets = Qwen3_5ForConditionalGeneration.LORA_TARGETS
    reference_state = checkpoint.reference.state_dict()
    peft_state = random_peft_state(
        reference_state,
        prefix=TEXT_ONLY_PREFIX,
        targets=targets,
        rank=LORA_RANK,
        generator=torch.Generator().manual_seed(3),
    )
    merged = copy.deepcopy(checkpoint.reference)
    merged.load_state_dict(
        merged_checkpoint(reference_state, peft_state, scaling=LORA_ALPHA / LORA_RANK)
    )

    adapters = Adapter(LoRAConfig(targets=targets, rank=LORA_RANK, alpha=LORA_ALPHA))
    model = load_model(checkpoint.folder, lora_adapters=adapters, training=training)
    adapters.materialize(device=CPU, dtype=torch.float32)
    text_config = Qwen3_5Config.from_hf(AutoConfig.from_pretrained(checkpoint.folder)).text
    adapters.load_state_dict(
        Qwen3_5LoRAMapping(text_config, prefix=TEXT_ONLY_PREFIX).to_engine(peft_state)
    )
    return model, _Checkpoint(checkpoint.folder, merged)


def _token_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(2)
    vocab_size = TINY_QWEN3_5_CONFIG["text_config"]["vocab_size"]
    return torch.randint(
        vocab_size, (NUM_REQUESTS, PROMPT_LEN), generator=generator, dtype=torch.int32
    )


def _assert_matches(engine: torch.Tensor, reference: torch.Tensor) -> None:
    # Both models run in FP32 end to end. The worst element across seeds, CPU thread counts
    # and MKL code paths (MKL_CBWR=COMPATIBLE) is about 4.2e-5, in the one-shot prefill
    # hidden states.
    torch.testing.assert_close(engine, reference, rtol=1e-4, atol=5e-5)


def test_prefill_matches_transformers(checkpoint: _Checkpoint) -> None:
    token_ids = _token_ids()
    model = load_model(checkpoint.folder)
    engine = EagerEngine(model, token_ids)
    reference_hidden, reference_logits = checkpoint.reference_outputs(token_ids[0])

    batch = engine.prepare(extend_entry(0, 0, PROMPT_LEN))
    hidden = model.forward_hidden(batch)
    logits = model.compute_logits(hidden, batch)

    _assert_matches(hidden, reference_hidden)
    _assert_matches(logits, reference_logits[-1:])


@pytest.mark.parametrize("split", [1, 3, 63, 64, 65])
def test_chunked_prefill_then_decode_matches_transformers(
    checkpoint: _Checkpoint,
    split: int,
) -> None:
    token_ids = _token_ids()
    engine = EagerEngine(load_model(checkpoint.folder), token_ids)
    _, reference_logits = checkpoint.reference_outputs(token_ids[1])
    prompt_len = PROMPT_LEN - 4

    logits = [
        engine.execute(extend_entry(1, 0, split)),
        engine.execute(extend_entry(1, split, prompt_len)),
        *(engine.execute(decode_entry(1, position)) for position in range(prompt_len, PROMPT_LEN)),
    ]

    positions = [split - 1, *range(prompt_len - 1, PROMPT_LEN)]
    _assert_matches(torch.cat(logits), reference_logits[positions])


def test_mixed_batch_and_batched_decode_match_transformers(checkpoint: _Checkpoint) -> None:
    token_ids = _token_ids()
    engine = EagerEngine(load_model(checkpoint.folder), token_ids)
    reference = [checkpoint.reference_outputs(row)[1] for row in token_ids]

    # Requests 0 and 1 prefill together; request 0 then finishes its prompt beside
    # request 2's whole prompt and request 1's first decode; then all three decode.
    first = engine.execute(extend_entry(0, 0, 40), extend_entry(1, 0, 20))
    mixed = engine.execute(extend_entry(0, 40, 66), extend_entry(2, 0, 9), decode_entry(1, 20))
    decoded = [
        engine.execute(
            decode_entry(0, 66 + step), decode_entry(1, 21 + step), decode_entry(2, 9 + step)
        )
        for step in range(3)
    ]

    _assert_matches(first, torch.stack([reference[0][39], reference[1][19]]))
    _assert_matches(mixed, torch.stack([reference[0][65], reference[2][8], reference[1][20]]))
    for step, logits in enumerate(decoded):
        expected = [reference[0][66 + step], reference[1][21 + step], reference[2][9 + step]]
        _assert_matches(logits, torch.stack(expected))


def test_reused_slot_and_interleaved_layers_match_transformers(
    interleaved_checkpoint: _Checkpoint,
) -> None:
    """A first chunk in a reused slot starts from zero state; interleaved layers address
    the KV layers and recurrent rows they are bound to."""

    token_ids = _token_ids()
    model = load_model(interleaved_checkpoint.folder)
    engine = EagerEngine(model, token_ids)
    reference = [interleaved_checkpoint.reference_outputs(row)[1] for row in token_ids]

    served = [
        engine.execute(extend_entry(0, 0, 30), extend_entry(1, 0, 20)),
        engine.execute(decode_entry(0, 30), decode_entry(1, 20)),
    ]
    engine.reuse_slot(0, token_ids[2])
    reused = [
        engine.execute(extend_entry(0, 0, 9), decode_entry(1, 21)),
        engine.execute(extend_entry(0, 9, 40), decode_entry(1, 22)),
        engine.execute(decode_entry(0, 40), decode_entry(1, 23)),
    ]

    expected_served = [(29, 19), (30, 20)]
    for logits, (first, second) in zip(served, expected_served, strict=True):
        _assert_matches(logits, torch.stack([reference[0][first], reference[1][second]]))
    expected_reused = [(8, 21), (39, 22), (40, 23)]
    for logits, (first, second) in zip(reused, expected_reused, strict=True):
        _assert_matches(logits, torch.stack([reference[2][first], reference[1][second]]))


def test_lora_adapter_matches_transformers_with_the_adapter_merged(
    checkpoint: _Checkpoint,
) -> None:
    token_ids = _token_ids()
    model, merged = _adapted(checkpoint)
    engine = EagerEngine(model, token_ids)
    _, merged_logits = merged.reference_outputs(token_ids[1])
    _, base_logits = checkpoint.reference_outputs(token_ids[1])
    prompt_len = PROMPT_LEN - 4

    # Like the runner, execute under inference mode: the Gated DeltaNet operations'
    # autograd guard stays quiet although the adapter's factors require grad.
    with torch.inference_mode():
        logits = [
            engine.execute(extend_entry(1, 0, prompt_len)),
            *(
                engine.execute(decode_entry(1, position))
                for position in range(prompt_len, PROMPT_LEN)
            ),
        ]

    positions = list(range(prompt_len - 1, PROMPT_LEN))
    _assert_matches(torch.cat(logits), merged_logits[positions])
    assert (merged_logits[positions] - base_logits[positions]).abs().max() > 0.1


def test_gated_delta_net_refuses_autograd_through_a_lora_projection(
    checkpoint: _Checkpoint,
) -> None:
    model, _ = _adapted(checkpoint)
    engine = EagerEngine(model, _token_ids())

    # With grad mode on, the adapted in_proj records autograd, which the inference
    # operations refuse.
    with pytest.raises(RuntimeError, match="do not support autograd"):
        engine.execute(extend_entry(0, 0, 8))


def test_training_forward_agrees_with_prefill(
    checkpoint: _Checkpoint,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Report-only (design Q5): no bound is declared until a GPU kernel is accepted.

    The same weights and adapter run a prefill over the inference state and the padded
    training forward through the training references; the largest logit gap over every
    position of two sequences is recorded as ``prefill_vs_training_max_abs_logit``.
    """

    bind_training_references(monkeypatch)
    token_ids = _token_ids()
    model, _ = _adapted(checkpoint, training=True)
    (layer,) = model.iter_recurrent_layers()
    engine = EagerEngine(model, token_ids)

    with torch.inference_mode():
        batch = engine.prepare(extend_entry(0, 0, PROMPT_LEN), extend_entry(1, 0, PROMPT_LEN))
        prefill = model.lm_head.forward_training(model.forward_hidden(batch))
    with torch.no_grad():
        training = training_logits(model, token_ids[:2].long())

    gap = float((training.flatten(0, 1) - prefill).abs().max())
    record_property("prefill_vs_training_max_abs_logit", gap)
    heads = f"{layer.num_key_heads}x{layer.num_value_heads}"
    print(f"prefill_vs_training_max_abs_logit[heads-{heads}]={gap:.3e}")
    assert math.isfinite(gap)
