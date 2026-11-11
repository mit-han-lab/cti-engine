"""Qwen3-MoE on CPU: its configuration fields, the published checkpoints' configurations, stop
sets and key tables, and tiny random FP32 checkpoints saved by Transformers 5.16.1, run over the
runner's paged KV pool with the CPU reference operations against Transformers' logits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import safetensors.torch
import torch
from transformers import AutoConfig, PreTrainedModel

from cti.models import ModelCapability
from cti.models.base import BaseLLMModel
from cti.models.config import UnimplementedConfigError
from cti.models.qwen3_moe.config import Qwen3MoeConfig
from cti.models.qwen3_moe.model import Qwen3MoeForCausalLM
from cti.models.weight import load_weight
from cti.resolution import ResolvedModel
from tests.support.forward_rows import decode_entry, extend_entry
from tests.support.moe import ReferenceMoe
from tests.support.qwen3_5_engine import EagerEngine, bind_reference_kernels
from tests.support.qwen3_moe import (
    DUAL_CHUNK_ATTENTION_CONFIG,
    EOS_TOKEN_IDS,
    PUBLISHED,
    QWEN3_30B_A3B_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
    YARN_OVERRIDE,
    published_keys,
    published_shapes,
    save_tiny_checkpoint,
)
from tests.support.tp_shards import build_model, load_sharded

CPU = torch.device("cpu")
PROMPT_LEN = 40
NUM_REQUESTS = 3


def _hf_config(folder: Path, document: Mapping[str, Any]) -> Any:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(document, default=dict), encoding="utf-8")
    return AutoConfig.from_pretrained(folder)


def _meta_model(hf_config: Any, dtype: torch.dtype = torch.bfloat16) -> BaseLLMModel:
    return build_model(hf_config, dtype=dtype, moe_backend=ReferenceMoe())


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"mlp_only_layers": [1]}, "mlp_only_layers"),
        ({"decoder_sparse_step": 2}, "decoder_sparse_step"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"use_sliding_window": True, "sliding_window": 4096}, "sliding_window"),
        (
            {"dual_chunk_attention_config": DUAL_CHUNK_ATTENTION_CONFIG},
            "dual_chunk_attention_config",
        ),
        ({"attn_logit_softcapping": 30.0}, "attn_logit_softcapping"),
        ({"final_logit_softcapping": 30.0}, "final_logit_softcapping"),
    ],
)
def test_config_refuses_each_unimplemented_field_by_name(
    tmp_path: Path, changes: Mapping[str, Any], field: str
) -> None:
    hf_config = _hf_config(tmp_path, {**QWEN3_30B_A3B_CONFIG, **changes})

    with pytest.raises(UnimplementedConfigError, match=f"Qwen3MoeForCausalLM.*{field}=") as error:
        Qwen3MoeForCausalLM.validate_hf_config(hf_config)
    assert error.value.field == field


def test_config_refuses_more_routes_than_experts(tmp_path: Path) -> None:
    hf_config = _hf_config(tmp_path, {**QWEN3_30B_A3B_CONFIG, "num_experts_per_tok": 129})

    with pytest.raises(ValueError, match=r"num_experts_per_tok=129 must be in \[1, num_experts"):
        Qwen3MoeForCausalLM.validate_hf_config(hf_config)


def test_training_only_fields_leave_the_inference_topology_unchanged(tmp_path: Path) -> None:
    """Transformers reads attention_dropout only in training and the router's auxiliary loss
    only into a training loss: the config keeps them and every inference field stays the
    published one."""

    published = Qwen3MoeConfig.from_hf(_hf_config(tmp_path / "a", QWEN3_30B_A3B_CONFIG))
    training = {
        "attention_dropout": 0.1,
        "output_router_logits": True,
        "router_aux_loss_coef": 0.5,
    }
    config = Qwen3MoeConfig.from_hf(
        _hf_config(tmp_path / "b", {**QWEN3_30B_A3B_CONFIG, **training})
    )

    assert (config.attention_dropout, config.output_router_logits) == (0.1, True)
    assert replace(config, attention_dropout=0.0, output_router_logits=False) == published


@pytest.mark.parametrize("variant", sorted(PUBLISHED))
def test_published_checkpoints_read_their_geometry_rope_and_stop_set(
    tmp_path: Path, variant: str
) -> None:
    published = PUBLISHED[variant]
    folder = tmp_path / variant
    _hf_config(folder, published.config)
    eos = published.generation_eos
    (folder / "generation_config.json").write_text(
        json.dumps({"eos_token_id": list(eos) if len(eos) > 1 else eos[0]}), encoding="utf-8"
    )
    token = published.tokenizer_eos
    (folder / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": token,
                "added_tokens_decoder": {str(EOS_TOKEN_IDS[token]): {"content": token}},
            }
        ),
        encoding="utf-8",
    )

    resolved = ResolvedModel.load(str(folder))
    config = Qwen3MoeConfig.from_hf(resolved.facts.hf_config)

    assert resolved.model_class is Qwen3MoeForCausalLM
    document = published.config
    assert (config.num_layers, config.hidden_size) == (
        document["num_hidden_layers"],
        document["hidden_size"],
    )
    assert (config.num_qo_heads, config.num_kv_heads, config.head_dim) == (
        document["num_attention_heads"],
        4,
        128,
    )
    assert (config.num_experts, config.num_experts_per_tok, config.norm_topk_prob) == (128, 8, True)
    assert config.moe_intermediate_size == document["moe_intermediate_size"]
    assert not config.tie_word_embeddings and not config.attention_bias
    rotary = config.rotary_config
    assert (rotary.base, rotary.scaling) == (document["rope_theta"], None)
    assert rotary.max_position == document["max_position_embeddings"]
    # The chat checkpoints stop at <|im_end|> and <|endoftext|>, the base one at the latter.
    assert resolved.facts.stop_token_ids == eos


def test_the_readme_yarn_override_scales_rope_within_the_configured_context(
    tmp_path: Path,
) -> None:
    config = Qwen3MoeConfig.from_hf(_hf_config(tmp_path, {**QWEN3_30B_A3B_CONFIG, **YARN_OVERRIDE}))

    rotary = config.rotary_config
    assert rotary.max_position == QWEN3_30B_A3B_CONFIG["max_position_embeddings"]
    assert rotary.scaling is not None
    assert (rotary.scaling["rope_type"], rotary.scaling["factor"]) == ("yarn", 4.0)
    assert rotary.scaling["original_max_position_embeddings"] == 32768


@pytest.mark.parametrize("num_layers", [48, 94], ids=["30b-a3b", "235b-a22b"])
def test_the_published_key_table_fills_every_parameter(
    stub_flashinfer: None, tmp_path: Path, num_layers: int
) -> None:
    published = PUBLISHED["30b-a3b" if num_layers == 48 else "235b-a22b"]
    model = _meta_model(_hf_config(tmp_path, published.config))
    mapping, declared = model.checkpoint_mapping, model.state_dict()
    shapes = published_shapes(published.config)
    assert len(published_keys(published.config)) == len(shapes)

    parts: dict[str, dict[tuple[int | None, int], tuple[int, ...]]] = {}
    for name, shape in shapes.items():
        (placement,) = mapping.placements(name, declared)
        (part,) = mapping.split(name, torch.empty(shape, device="meta"))
        slots = parts.setdefault(placement.name, {})
        assert slots.setdefault((placement.expert, placement.slot), part.shape) == part.shape

    # Every parameter, and only those, is filled: stacked experts, merged projections, untied head.
    assert parts.keys() == declared.keys()
    for name, slots in parts.items():
        experts = sorted({expert for expert, _ in slots})
        rows = sum(shape[0] for (expert, _), shape in slots.items() if expert == experts[0])
        expected = (rows, *next(iter(slots.values()))[1:])
        if experts != [None]:
            assert experts == list(range(128)), name
            expected = (128, *expected)
        assert expected == tuple(declared[name].shape), name
    layer = "model.layers.0."
    hidden, intermediate = (
        published.config["hidden_size"],
        published.config["moe_intermediate_size"],
    )
    assert declared[layer + "mlp.experts.gate_up_proj"].shape == (128, 2 * intermediate, hidden)
    assert declared[layer + "mlp.experts.down_proj"].shape == (128, hidden, intermediate)
    assert "lm_head.weight" in declared


def test_an_untied_checkpoint_without_its_lm_head_fails_to_load(
    stub_flashinfer: None, tmp_path: Path
) -> None:
    folder = tmp_path / "checkpoint"
    reference = save_tiny_checkpoint(folder)
    (file,) = folder.glob("*.safetensors")
    state = safetensors.torch.load_file(file)
    del state["lm_head.weight"]
    safetensors.torch.save_file(state, file)
    model = _meta_model(reference.config, torch.float32)
    loaded = dict(
        load_weight(str(folder), CPU, mapping=model.checkpoint_mapping, params=model.params())
    )

    with pytest.raises(KeyError, match="lm_head.weight"):
        model.load_state_dict(loaded)


def test_capabilities_and_targets() -> None:
    assert Qwen3MoeForCausalLM.CAPABILITIES == (
        ModelCapability.LORA
        | ModelCapability.TRAINING
        | ModelCapability.PREFIX_REUSE
        | ModelCapability.TENSOR_PARALLEL
        | ModelCapability.SPECULATIVE_TARGET
    )
    assert Qwen3MoeForCausalLM.LORA_TARGETS == ("attention", "experts")
    assert Qwen3MoeForCausalLM.CONFIG is Qwen3MoeConfig


@dataclass(frozen=True)
class _Checkpoint:
    folder: Path
    reference: PreTrainedModel

    def logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Transformers' logits for one whole sequence."""

        with torch.no_grad():
            return self.reference(input_ids=token_ids[None].long()).logits[0]


# Renormalized routes with an unbiased attention, the published form; and raw routing
# probabilities with projection biases.
_CHECKPOINTS = {
    "renormalized": {},
    "raw-probabilities-biased": {"norm_topk_prob": False, "attention_bias": True},
}


@pytest.fixture(scope="module", params=sorted(_CHECKPOINTS))
def checkpoint(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> _Checkpoint:
    folder = tmp_path_factory.mktemp(request.param)
    return _Checkpoint(folder, save_tiny_checkpoint(folder, **_CHECKPOINTS[request.param]))


@pytest.fixture(autouse=True)
def _reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_reference_kernels(monkeypatch)


def _load_model(checkpoint: _Checkpoint) -> BaseLLMModel:
    return load_sharded(checkpoint.reference.config, checkpoint.folder, moe_backend=ReferenceMoe())[
        0
    ]


def _token_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(2)
    vocab_size = TINY_QWEN3_MOE_CONFIG["vocab_size"]
    return torch.randint(
        vocab_size, (NUM_REQUESTS, PROMPT_LEN), generator=generator, dtype=torch.int32
    )


def _assert_matches(engine: torch.Tensor, reference: torch.Tensor) -> None:
    # Both models run in FP32 end to end; routes and their weights are the same.
    torch.testing.assert_close(engine, reference, rtol=1e-4, atol=5e-5)


def test_the_tiny_checkpoints_declare_what_they_test(checkpoint: _Checkpoint) -> None:
    config = Qwen3MoeConfig.from_hf(checkpoint.reference.config)
    model = _load_model(checkpoint)
    biases = {name for name in model.state_dict() if name.endswith("bias")}

    assert (config.num_experts, config.num_experts_per_tok) == (8, 2)
    assert all(
        layer.mlp.experts.renormalize == config.norm_topk_prob
        for layer in model.model.layers.op_list
    )
    assert bool(biases) == config.attention_bias
    if config.attention_bias:
        assert biases == {
            f"model.layers.{layer}.self_attn.{projection}.bias"
            for layer in range(3)
            for projection in ("qkv_proj", "o_proj")
        }


@pytest.mark.parametrize("split", [1, 7, 33])
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
    """Prefills, a mixed batch and batched decodes route each row by its own tokens; a reused
    slot's prompt reads only its own KV rows."""

    token_ids = _token_ids()
    engine = EagerEngine(_load_model(checkpoint), token_ids)
    reference = [checkpoint.logits(row) for row in token_ids]

    served = [
        engine.execute(extend_entry(0, 0, 20), extend_entry(1, 0, 11)),
        engine.execute(extend_entry(0, 20, 30), decode_entry(1, 11)),
        engine.execute(decode_entry(0, 30), decode_entry(1, 12)),
    ]
    engine.reuse_slot(0, token_ids[2])
    reused = [
        engine.execute(extend_entry(0, 0, 9), decode_entry(1, 13)),
        engine.execute(extend_entry(0, 9, 25), decode_entry(1, 14)),
        engine.execute(decode_entry(0, 25), decode_entry(1, 15)),
    ]

    for logits, (first, second) in zip(served, [(19, 10), (29, 11), (30, 12)], strict=True):
        _assert_matches(logits, torch.stack([reference[0][first], reference[1][second]]))
    for logits, (first, second) in zip(reused, [(8, 13), (24, 14), (25, 15)], strict=True):
        _assert_matches(logits, torch.stack([reference[2][first], reference[1][second]]))


def test_a_prefix_hit_reads_the_cached_rows_like_a_cold_prefill(checkpoint: _Checkpoint) -> None:
    """Attention-only layers reuse a prefix through the paged KV pool alone."""

    tokens = _token_ids()[:1].repeat(2, 1)
    engine = EagerEngine(_load_model(checkpoint), tokens)
    engine.execute(extend_entry(0, 0, 24))
    # Request 1's first 24 positions are request 0's cached rows.
    engine._cache_index_table[1, :24] = engine._cache_index_table[0, :24]

    hit = engine.execute(extend_entry(1, 24, PROMPT_LEN))

    _assert_matches(hit, checkpoint.logits(tokens[1])[PROMPT_LEN - 1 : PROMPT_LEN])


def test_load_keeps_the_published_dtype_and_names(checkpoint: _Checkpoint) -> None:
    model = _load_model(checkpoint)
    state = model.state_dict()

    assert {tensor.dtype for tensor in state.values()} == {torch.float32}
    assert state["model.layers.0.mlp.experts.gate_up_proj"].shape == (8, 32, 32)
    hf_experts = checkpoint.reference.model.layers[0].mlp.experts
    assert torch.equal(state["model.layers.0.mlp.experts.gate_up_proj"], hf_experts.gate_up_proj)
    assert torch.equal(state["model.layers.0.mlp.experts.down_proj"], hf_experts.down_proj)
