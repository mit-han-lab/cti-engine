"""The layer-streamed truth on a tiny random dense Qwen3 on the CPU: one decoder layer resident
at a time, it computes what the whole model computes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import Qwen3ForCausalLM

from tests.support.hf_layer_stream import LayerStream, stream_prompts
from tests.support.hf_truth import load_reference
from tests.support.tiny_configs import tiny_hf_config

PROMPTS = {"short": [3, 17, 42, 8, 61], "long": [(5 * index + 1) % 64 for index in range(50)]}
CONTINUATIONS = {"short": [7, 9, 11], "long": [1, 2, 3, 4]}


@pytest.fixture(scope="module")
def qwen3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("tiny-qwen3")
    torch.manual_seed(0)
    Qwen3ForCausalLM(tiny_hf_config("Qwen3ForCausalLM", folder)).save_pretrained(folder)
    return folder


def test_the_streamed_dense_qwen3_computes_the_whole_models_logits(qwen3: Path) -> None:
    stream = LayerStream(qwen3, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        dumps = stream_prompts(stream, list(PROMPTS.items()), CONTINUATIONS, top_k=2)
    reference = load_reference(qwen3, torch.device("cpu"), torch.float32)
    for name, prompt in PROMPTS.items():
        sequence = prompt + CONTINUATIONS[name]
        with torch.inference_mode():
            logits = reference(input_ids=torch.tensor([sequence]), use_cache=False).logits[0]
        values, ids = logits.topk(2, dim=-1)
        assert torch.equal(dumps[name]["topk_ids"], ids)
        torch.testing.assert_close(dumps[name]["topk_logits"], values, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            dumps[name]["logsumexp"], logits.logsumexp(-1), rtol=1e-6, atol=1e-6
        )
