"""Training kernels come from a model's ``LayerServices``, built only when a layer needs one."""

from __future__ import annotations

from typing import Any

import torch

from cti.layers import ParallelLMHead
from cti.layers.attention import AttentionLayer
from cti.models.training_contract import Packing
from tests.support.tp_shards import ReferenceLayerServices, site


class _IdentityRotary:
    def forward_training(self, _positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor, **_):
        return q, k


class _CountingServices(ReferenceLayerServices):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[str] = []

    def training_loss(self) -> Any:
        self.requests.append("loss")
        return super().training_loss()

    def training_logprobs(self) -> Any:
        self.requests.append("logprobs")
        return super().training_logprobs()

    def packed_training_attention(self) -> Any:
        self.requests.append("varlen")
        return super().packed_training_attention()


def test_only_a_training_head_takes_the_loss_operations() -> None:
    inference = _CountingServices()
    ParallelLMHead(16, 4, site=site(services=inference))
    training = _CountingServices()
    ParallelLMHead(16, 4, site=site(training=True, services=training))

    assert inference.requests == []
    assert training.requests == ["loss", "logprobs"]


def test_packed_attention_is_built_at_the_first_packed_forward() -> None:
    from cti.layers import LinearQKVMerged

    services = _CountingServices()
    layer_site = site(training=True, services=services)
    qkv = LinearQKVMerged(
        hidden_size=8, head_dim=4, num_qo_heads=2, num_kv_heads=1, has_bias=False, site=layer_site
    )
    layer = AttentionLayer(qkv=qkv, rotary=_IdentityRotary())
    assert services.requests == []

    tokens = 5
    positions = torch.arange(tokens, dtype=torch.int64)
    cu_seqlens = torch.tensor((0, 2, tokens), dtype=torch.int32)
    layer.forward_training(torch.randn(tokens, 16), positions, Packing(positions, cu_seqlens, 3))
    assert services.requests == ["varlen"]


class _Ranks:
    """A group of ``size`` ranks that counts the residual reductions it builds."""

    rank = 0

    def __init__(self, size: int) -> None:
        self.size = size
        self.built: list[dict[str, Any]] = []

    def residual_reduction(self, **geometry: Any) -> object:
        self.built.append(geometry)
        return object()


def test_residual_reductions_are_one_per_geometry_and_none_on_one_rank() -> None:
    services, single, pair = ReferenceLayerServices(), _Ranks(1), _Ranks(2)
    geometry = {"hidden_size": 8, "eps": 1e-6, "zero_centered": True}

    assert site(group=single, services=services).residual_reduction(**geometry) is None
    paired = site(group=pair, services=services)
    first = paired.residual_reduction(**geometry)
    assert paired.residual_reduction(**geometry) is first
    assert paired.residual_reduction(**{**geometry, "zero_centered": False}) is not first
    assert single.built == []
    assert len(pair.built) == 2
    # Another model's services build their own.
    other = site(group=pair, services=ReferenceLayerServices())
    assert other.residual_reduction(**geometry) is not first
