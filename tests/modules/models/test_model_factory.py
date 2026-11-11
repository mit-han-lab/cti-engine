from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig, Qwen3Config, Qwen3MoeConfig

from cti.forward import InlineBreaks
from cti.kvcache import PagedSlot
from cti.layers import (
    AttentionLayer,
    LinearQKVMerged,
    ParallelLMHead,
    Site,
    VocabParallelEmbedding,
)
from cti.models import create_model, get_model_class
from cti.models.base import BaseLLMModel
from cti.moe import BaseMoeBackend
from cti.multimodal import ImagePreprocessing
from cti.resolution import ResolvedModel
from cti.training.runner import ModelParameterUniverse
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.tp_shards import ReferenceLayerServices, site

CPU = torch.device("cpu")
_SITE = site()


class _TestMoeBackend(BaseMoeBackend):
    def route(self, *args: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("the layer routes inside its backend's forward")

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("construction contract must not execute the MoE backend")


class _IdentityRotary:
    def forward_training(
        self,
        _positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k


class _RecordingRotary:
    def __init__(self) -> None:
        self.positions: list[torch.Tensor] = []

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.positions.append(positions)
        return q, k


class _RecordingLinearCrossEntropyLoss:
    chunk_size = 256

    def __init__(self) -> None:
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.weight = weight
        self.bias = bias
        logits = torch.nn.functional.linear(hidden_states, weight, bias)
        return torch.nn.functional.cross_entropy(logits.float(), target_ids, reduction="sum")


class _RecordingServices(ReferenceLayerServices):
    """Reference services whose training loss is ``loss``."""

    def __init__(self, loss: _RecordingLinearCrossEntropyLoss) -> None:
        super().__init__()
        self._loss = loss

    def training_loss(self) -> _RecordingLinearCrossEntropyLoss:
        return self._loss


def _training_site(loss: _RecordingLinearCrossEntropyLoss):
    return site(training=True, services=_RecordingServices(loss))


class _ParameterUniverseModel:
    def __init__(
        self,
        state: dict[str, torch.Tensor],
        declaration: tuple[tuple[str, torch.Tensor], ...],
    ) -> None:
        self.state = state
        self.declaration = declaration

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.state

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return self.declaration


def _hf_config(architecture: str, *, moe: bool = False) -> PretrainedConfig:
    geometry = {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "hidden_size": 128,
        "vocab_size": 256,
        "intermediate_size": 256,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 16,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "architectures": [architecture],
    }
    if not moe:
        return Qwen3Config(**geometry)
    return Qwen3MoeConfig(
        **geometry,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        norm_topk_prob=True,
    )


def _create_meta_model(
    config: PretrainedConfig,
    *,
    moe_backend: BaseMoeBackend | None = None,
) -> BaseLLMModel:
    with torch.device("meta"):
        return create_model(
            get_model_class(config.architectures[0]),
            checkpoint_facts(config),
            site=site(moe_backend=moe_backend),
        )


def test_model_owns_one_ordered_parameter_universe() -> None:
    base = torch.zeros(2)
    persisted_head = torch.ones(3)
    external_side = torch.full((4,), 2.0)
    model = _ParameterUniverseModel(
        {
            "base.weight": base,
            "head.weight": persisted_head,
        },
        (
            ("head.weight", persisted_head),
            ("side.weight", external_side),
        ),
    )

    universe = ModelParameterUniverse.of(model)

    assert universe.named_tensors == (
        ("base.weight", base),
        ("head.weight", persisted_head),
        ("side.weight", external_side),
    )
    assert universe.named_trainable_parameters == (
        ("head.weight", persisted_head),
        ("side.weight", external_side),
    )


def test_model_parameter_universe_rejects_mismatched_persisted_identity() -> None:
    model = _ParameterUniverseModel(
        {"head.weight": torch.ones(3)},
        (("head.weight", torch.ones(3)),),
    )

    with pytest.raises(ValueError, match="does not match state_dict"):
        ModelParameterUniverse.of(model)


def test_model_parameter_universe_rejects_a_persisted_tensor_under_an_alias() -> None:
    head = torch.ones(3)
    model = _ParameterUniverseModel(
        {"head.weight": head},
        (("renamed.weight", head),),
    )

    with pytest.raises(ValueError, match="is persisted as 'head.weight'"):
        ModelParameterUniverse.of(model)


def test_model_parameter_universe_rejects_duplicate_declarations() -> None:
    model = _ParameterUniverseModel(
        {},
        (
            ("head.weight", torch.ones(3)),
            ("head.weight", torch.zeros(3)),
        ),
    )

    with pytest.raises(ValueError, match="names must be unique"):
        ModelParameterUniverse.of(model)


def test_model_factory_constructs_the_class_from_its_own_config_type() -> None:
    hf_config = object()

    class _RecordingConfig:
        @classmethod
        def from_hf(cls, config: object) -> tuple[str, object]:
            return ("parsed", config)

    class _RecordingModel:
        CONFIG = _RecordingConfig

        def __init__(self, config: object, *, site: Site) -> None:
            self.config = config
            self.site = site

    at = site()
    model = create_model(_RecordingModel, checkpoint_facts(hf_config), site=at)  # type: ignore[arg-type]

    assert isinstance(model, _RecordingModel)
    assert model.config == ("parsed", hf_config)
    assert model.site is at
    # A checkpoint's preprocessing reaches the constructor, which refuses one it does not take.
    preprocessing = ImagePreprocessing(4, 4, 1 / 255, (0.5,) * 3, (0.5,) * 3)
    with pytest.raises(TypeError, match="image_preprocessing"):
        create_model(
            _RecordingModel,  # type: ignore[arg-type]
            checkpoint_facts(hf_config, image_preprocessing=preprocessing),
            site=at,
        )


@pytest.mark.parametrize("architectures", [None, []], ids=["missing", "empty"])
def test_a_checkpoint_must_declare_its_architecture(tmp_path, architectures: object) -> None:
    config = Qwen3Config(num_hidden_layers=2, hidden_size=128, vocab_size=256)
    config.architectures = architectures
    config.save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="must declare architectures"):
        ResolvedModel.load(str(tmp_path))


def _qkv() -> LinearQKVMerged:
    """Two query heads over one KV head, 4 wide."""
    return LinearQKVMerged(
        hidden_size=8, head_dim=4, num_qo_heads=2, num_kv_heads=1, has_bias=False, site=_SITE
    )


def test_attention_layer_training_needs_no_backend() -> None:
    layer = AttentionLayer(qkv=_qkv(), rotary=_IdentityRotary())
    qkv = torch.randn(2, 3, 16, requires_grad=True)

    output = layer.forward_training(qkv, torch.arange(3), None)
    output.square().mean().backward()

    assert output.shape == (2, 3, 8)
    assert qkv.grad is not None


def test_attention_layer_rotates_by_rope_positions_and_hands_k_and_v_to_the_backend() -> None:
    rotary = _RecordingRotary()
    positions = torch.arange(3, dtype=torch.int64)
    batch = SimpleNamespace(
        positions=positions, rope_positions=positions + 100, graph_breaks=InlineBreaks()
    )
    calls = []
    backend = SimpleNamespace(
        SUPPORTS_KV_PROLOGUE=False, forward=lambda *args: calls.append(args) or args[0]
    )
    layer = AttentionLayer(qkv=_qkv(), rotary=rotary)
    layer.bind_state({"kv": PagedSlot(backend, 1)})
    qkv = torch.arange(48.0).view(3, 16)
    layer.forward(qkv, batch)

    assert len(rotary.positions) == 1
    assert rotary.positions[0] is batch.rope_positions
    ((q, k, v, layer_id, attended_batch),) = calls
    assert torch.equal(q, qkv[:, :8].view(3, 2, 4))
    assert torch.equal(k, qkv[:, 8:12]) and torch.equal(v, qkv[:, 12:])
    assert layer_id == 1 and attended_batch is batch


@pytest.mark.parametrize("tied", (False, True))
def test_lm_head_owns_training_projection_resolution(tied: bool) -> None:
    embedding = VocabParallelEmbedding(7, 4, site=_SITE) if tied else None
    operation = _RecordingLinearCrossEntropyLoss()
    head = ParallelLMHead(7, 4, bias=True, tied_embedding=embedding, site=_training_site(operation))

    projection = embedding or head
    projection.weight.normal_()
    assert head.bias is not None
    head.bias.zero_()
    hidden_states = torch.randn(3, 4, requires_grad=True)
    target_ids = torch.tensor((1, 4, 2), dtype=torch.int64)
    head.validate_training_targets(target_ids)
    with pytest.raises(ValueError, match=r"must be in \[0, 7\)"):
        head.validate_training_targets(torch.tensor((1, 7), dtype=torch.int64))
    with pytest.raises(ValueError, match=r"must be in \[0, 7\)"):
        head.validate_training_targets(torch.tensor((-1, 2), dtype=torch.int64))
    loss = head.forward_training_loss(hidden_states, target_ids)

    assert loss.shape == ()
    assert operation.weight is (embedding.weight if embedding is not None else head.weight)
    assert operation.bias is head.bias


def test_tied_lm_head_accumulates_embedding_and_output_gradients() -> None:
    embedding = VocabParallelEmbedding(7, 4, site=_SITE)
    head = ParallelLMHead(
        7, 4, tied_embedding=embedding, site=_training_site(_RecordingLinearCrossEntropyLoss())
    )
    embedding.weight.normal_()
    initial_weight = embedding.weight.detach().clone()
    embedding.weight.requires_grad_()
    input_ids = torch.tensor((0, 3, 5), dtype=torch.int64)
    target_ids = torch.tensor((1, 4, 2), dtype=torch.int64)
    hidden_states = embedding.forward_training(input_ids)
    head.forward_training_loss(hidden_states, target_ids).backward()
    actual_gradient = embedding.weight.grad

    reference_weight = initial_weight.requires_grad_()
    reference_hidden = torch.nn.functional.embedding(input_ids, reference_weight)
    reference_logits = torch.nn.functional.linear(reference_hidden, reference_weight)
    torch.nn.functional.cross_entropy(
        reference_logits.float(),
        target_ids,
        reduction="sum",
    ).backward()

    assert actual_gradient is not None
    assert reference_weight.grad is not None
    torch.testing.assert_close(actual_gradient, reference_weight.grad, rtol=0, atol=0)


def test_model_factory_injects_one_backend_without_serializing_it(stub_flashinfer: None) -> None:
    config = _hf_config("Qwen3MoeForCausalLM", moe=True)
    backend = _TestMoeBackend()
    model = _create_meta_model(config, moe_backend=backend)

    layers = model.model.layers.op_list
    assert len(layers) == config.num_hidden_layers
    assert all(layer.mlp.experts.moe_backend is backend for layer in layers)
    assert all("backend" not in name for name in model.state_dict())


@pytest.mark.parametrize(
    ("architecture", "uses_moe"),
    [
        pytest.param("Qwen3ForCausalLM", False, id="qwen3"),
        pytest.param("Qwen3MoeForCausalLM", True, id="qwen3-moe"),
    ],
)
def test_models_expose_stable_attention_layer_topology(
    stub_flashinfer: None,
    architecture: str,
    uses_moe: bool,
) -> None:
    config = _hf_config(architecture, moe=uses_moe)
    num_layers = config.num_hidden_layers
    moe_backend = _TestMoeBackend() if uses_moe else None
    model = _create_meta_model(config, moe_backend=moe_backend)

    first_traversal = list(model.iter_attention_layers())
    second_traversal = list(model.iter_attention_layers())

    assert len(first_traversal) == num_layers
    assert first_traversal == second_traversal
    assert len({id(layer) for layer in first_traversal}) == num_layers
    assert model.iter_recurrent_layers() == ()
    assert model.iter_state_layers() == tuple(first_traversal)
    assert model.rope_position_axes() == 1
