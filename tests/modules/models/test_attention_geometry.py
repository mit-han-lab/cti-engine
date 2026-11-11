"""Every registered model's MHA requirements, split as the family splits them, are the heads
its attention layers hold on each rank: the backend is built from the requirements alone."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cti.attention.requirement import kv_head_window, qo_head_window
from cti.distributed import Topology
from cti.models import get_model_class
from cti.models.register import MODELS
from tests.support.gated_delta_net import reference_kernel_module
from tests.support.tiny_configs import tiny_hf_config
from tests.support.tp_shards import build_model


@pytest.fixture(autouse=True)
def _gated_delta_net_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())


@pytest.mark.usefixtures("stub_flashinfer")
@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("architecture", sorted(MODELS))
def test_requirements_match_layer_heads(architecture: str, size: int, tmp_path: Path) -> None:
    config = tiny_hf_config(architecture, tmp_path)
    # Heads that two ranks split: 4 query, 2 KV (and Qwen3.5's linear-attention heads).
    text = config.get_text_config()
    text.num_attention_heads, text.num_key_value_heads = 4, 2
    if hasattr(text, "linear_num_key_heads"):
        text.linear_num_key_heads, text.linear_num_value_heads = 2, 4
    requirements = [
        r for r in get_model_class(architecture).attention_requirements(config) if r.family == "mha"
    ]
    assert requirements
    for rank in range(size):
        model = build_model(config, topology=Topology(rank, size))
        layers = {
            (layer.head_dim, layer.num_qo_heads, layer.num_kv_heads)
            for layer in model.iter_attention_layers()
        }
        required = {
            (
                r.size("head_dim"),
                qo_head_window(r.size("qo_heads"), rank, size)[1],
                kv_head_window(r.size("kv_heads"), rank, size)[1],
            )
            for r in requirements
        }
        assert layers == required
