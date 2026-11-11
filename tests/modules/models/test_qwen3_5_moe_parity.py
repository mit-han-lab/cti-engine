"""Qwen3.5-MoE against transformers 5.16.1 on CPU: a tiny random FP32 text checkpoint saved by
Transformers, loaded through the checkpoint mapping and run over the runner's recurrent and paged
KV pools with the CPU reference operations, gives Transformers' logits at the same positions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from cti.forward import RecurrentExport
from cti.models.base import BaseLLMModel
from tests.support.forward_rows import decode_entry, extend_entry
from tests.support.moe import ReferenceMoe
from tests.support.qwen3_5 import load_hf_config, randomize_hf_weights
from tests.support.qwen3_5_engine import EagerEngine, bind_reference_kernels
from tests.support.qwen3_5_moe import TINY_QWEN3_5_MOE_CONFIG, text_only_config
from tests.support.tp_shards import load_sharded

# Longer than one 64-token chunk of Transformers' chunked delta rule.
PROMPT_LEN = 70
NUM_REQUESTS = 3


@pytest.fixture(autouse=True)
def _reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_reference_kernels(monkeypatch)


@dataclass(frozen=True)
class _Checkpoint:
    folder: Path
    reference: PreTrainedModel

    def logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Transformers' logits for one whole sequence."""

        with torch.no_grad():
            return self.reference(input_ids=token_ids[None].long()).logits[0]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> _Checkpoint:
    folder = tmp_path_factory.mktemp("qwen3_5_moe")
    config = load_hf_config(text_only_config(TINY_QWEN3_5_MOE_CONFIG), folder)
    # The per-expert loop is the reference's own experts implementation.
    reference = AutoModelForCausalLM.from_config(
        config, attn_implementation="eager", experts_implementation="eager"
    ).eval()
    # The engine declares a BF16 convolution weight.
    randomize_hf_weights(reference, conv_dtype=torch.bfloat16)
    reference.save_pretrained(folder)
    return _Checkpoint(folder, reference)


def _load_model(checkpoint: _Checkpoint) -> BaseLLMModel:
    return load_sharded(checkpoint.reference.config, checkpoint.folder, moe_backend=ReferenceMoe())[
        0
    ]


def _token_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(2)
    vocab_size = TINY_QWEN3_5_MOE_CONFIG["text_config"]["vocab_size"]
    return torch.randint(
        vocab_size, (NUM_REQUESTS, PROMPT_LEN), generator=generator, dtype=torch.int32
    )


def _assert_matches(engine: torch.Tensor, reference: torch.Tensor) -> None:
    # Both models run in FP32 end to end; the dense model's measured bound holds.
    torch.testing.assert_close(engine, reference, rtol=1e-4, atol=5e-5)


@pytest.mark.parametrize("split", [1, 3, 64, 65])
def test_chunked_prefill_then_decode_matches_transformers(
    checkpoint: _Checkpoint, split: int
) -> None:
    token_ids = _token_ids()
    engine = EagerEngine(_load_model(checkpoint), token_ids)
    reference = checkpoint.logits(token_ids[1])
    prompt_len = PROMPT_LEN - 4

    logits = [
        engine.execute(extend_entry(1, 0, split)),
        engine.execute(extend_entry(1, split, prompt_len)),
        *(engine.execute(decode_entry(1, position)) for position in range(prompt_len, PROMPT_LEN)),
    ]

    positions = [split - 1, *range(prompt_len - 1, PROMPT_LEN)]
    _assert_matches(torch.cat(logits), reference[positions])


def test_mixed_batches_and_a_reused_slot_match_transformers(checkpoint: _Checkpoint) -> None:
    """Prefills, a mixed batch and batched decodes route each row by its own tokens; a first
    chunk in a reused slot starts from zero state; interleaved layers address the KV layers
    and recurrent rows they are bound to."""

    token_ids = _token_ids()
    model = _load_model(checkpoint)
    engine = EagerEngine(model, token_ids)
    reference = [checkpoint.logits(row) for row in token_ids]

    served = [
        engine.execute(extend_entry(0, 0, 40), extend_entry(1, 0, 20)),
        engine.execute(extend_entry(0, 40, 66), decode_entry(1, 20)),
        engine.execute(decode_entry(0, 66), decode_entry(1, 21)),
    ]
    engine.reuse_slot(0, token_ids[2])
    reused = [
        engine.execute(extend_entry(0, 0, 9), decode_entry(1, 22)),
        engine.execute(extend_entry(0, 9, 40), decode_entry(1, 23)),
        engine.execute(decode_entry(0, 40), decode_entry(1, 24)),
    ]

    for logits, (first, second) in zip(served, [(39, 19), (65, 20), (66, 21)], strict=True):
        _assert_matches(logits, torch.stack([reference[0][first], reference[1][second]]))
    for logits, (first, second) in zip(reused, [(8, 22), (39, 23), (40, 24)], strict=True):
        _assert_matches(logits, torch.stack([reference[2][first], reference[1][second]]))


@pytest.mark.parametrize("budget", [64, 97])
def test_a_prefix_hit_restores_the_exported_state_like_a_cold_prefill(
    checkpoint: _Checkpoint, budget: int
) -> None:
    length = 97
    vocab_size = TINY_QWEN3_5_MOE_CONFIG["text_config"]["vocab_size"]
    generator = torch.Generator().manual_seed(5)
    tokens = torch.randint(vocab_size, (1, length), generator=generator).repeat(2, 1)
    engine = EagerEngine(_load_model(checkpoint), tokens, snapshot_rows=2)
    # The forward whose whole 64-token multiples reach 64 exports the state there, at its end
    # or inside.
    for frontier in range(0, length, budget):
        end = min(frontier + budget, length)
        captures = (RecurrentExport(64, 2),) if frontier < 64 <= end else ()
        engine.execute(extend_entry(0, frontier, end, captures=captures))
    engine._cache_index_table[1, :64] = engine._cache_index_table[0, :64]

    hit = engine.execute(extend_entry(1, 64, length, restore_row=2))

    cold = EagerEngine(_load_model(checkpoint), tokens).execute(extend_entry(1, 0, length))
    _assert_matches(hit, cold)
