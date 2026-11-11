"""The attention family registry: the lazy family table, dispatch by the requirements'
family, and the MHA family's per-rank heads."""

from __future__ import annotations

from types import MappingProxyType

import pytest
import torch

import cti.attention.family as family_module
from cti.attention import (
    AttentionRequest,
    AttentionRequirement,
    resolve_attention_config,
)
from cti.attention import mha as mha_module
from cti.attention.flash_attention import FlashAttentionBackend
from cti.attention.flashinfer import FlashInferBackend
from cti.attention.paged import KVPair
from cti.attention.requirement import kv_head_window
from cti.attention.tensorrt_llm import TensorRTLLMBackend
from cti.distributed import Topology
from cti.forward import RoundShape
from cti.kvcache import GroupRole
from cti.layers import LinearQKVMerged
from cti.utils import DeviceCapability
from tests.support.attention import set_validation
from tests.support.attention_family import ResolvedFake
from tests.support.tp_shards import site

_TARGET = (AttentionRequirement.mha(16, 8, 128),)
_DRAFT = (AttentionRequirement.mha(32, 4, 128),)
_POOLED = AttentionRequirement(
    family="mha", slot="kv", tokens_per_entry=4, geometry=_TARGET[0].geometry
)
_LATENT = AttentionRequirement(family="mla", slot="kv", geometry=(("head_dim", 512),))


def _request(backend: str = "fi", **options: object) -> AttentionRequest:
    """A BF16 single-rank request on SM120 with a page of one token, ``options`` aside."""

    fields = {
        "page_size": 1,
        "capability": DeviceCapability(12, 0),
        "tensor_parallel_size": 1,
        "dtype": torch.bfloat16,
    }
    return AttentionRequest(target=GroupRole.TARGET, backend=backend, **{**fields, **options})  # type: ignore[arg-type]


def test_the_family_table_is_immutable() -> None:
    assert isinstance(family_module.FAMILIES, MappingProxyType)
    with pytest.raises(TypeError):
        family_module.FAMILIES["fake"] = ("tests.support.attention_family", "resolve")  # type: ignore[index]


@pytest.mark.parametrize(
    ("requested", "capability", "page_size", "ranks", "rounds", "draft"),
    [
        ("auto", (10, 0), None, 1, False, False),
        ("auto", (12, 0), 1, 1, False, True),
        ("auto", (9, 0), None, 1, False, False),
        ("auto", (10, 0), None, 1, True, False),
        ("fi", (12, 0), 1, 2, False, True),
        ("fa,fi", (9, 0), None, 1, False, False),
        ("trtllm", (10, 0), 64, 1, False, False),
        ("fi,fi", (12, 0), 1, 1, False, False),
    ],
    ids=["auto-sm100", "auto-sm120", "auto-sm90", "rounds", "tp2", "hybrid", "trtllm", "dup"],
)
def test_the_registry_returns_what_the_mha_family_resolves(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    capability: tuple[int, int],
    page_size: int | None,
    ranks: int,
    rounds: bool,
    draft: bool,
) -> None:
    for backend in (FlashAttentionBackend, FlashInferBackend, TensorRTLLMBackend):
        set_validation(monkeypatch, backend)
    requirements = {GroupRole.TARGET: _TARGET} | ({GroupRole.DRAFT: _DRAFT} if draft else {})
    request = _request(
        requested,
        page_size=page_size,
        capability=DeviceCapability(*capability),
        tensor_parallel_size=ranks,
        rounds=RoundShape(16, 2, 4) if rounds else None,
    )

    resolved = resolve_attention_config(requirements, request)
    direct = mha_module.resolve(requirements, request)

    assert isinstance(resolved, mha_module.ResolvedMha)
    assert (resolved.selected, resolved.pages, resolved.reason, resolved.layouts) == (
        direct.selected,
        direct.pages,
        direct.reason,
        direct.layouts,
    )
    assert resolved == direct


@pytest.mark.parametrize(
    ("requirements", "page_size", "error"),
    [
        ({GroupRole.TARGET: _TARGET, GroupRole.DRAFT: (_LATENT,)}, 1, "one attention family"),
        ({GroupRole.TARGET: (_LATENT,)}, 1, "no attention family 'mla'"),
        ({}, 1, "at least one requirement"),
        ({GroupRole.TARGET: ()}, 1, "at least one requirement"),
        ({GroupRole.TARGET: (_POOLED,)}, 1, "one token per cache entry"),
        ({GroupRole.TARGET: (), GroupRole.DRAFT: _DRAFT}, 1, "target role has no attention"),
        ({GroupRole.TARGET: _TARGET}, 0, "page_size must be positive"),
        ({GroupRole.TARGET: _TARGET}, -16, "page_size must be positive"),
    ],
    ids=[
        "mixed",
        "unknown",
        "empty",
        "empty-role",
        "pooled",
        "no-target",
        "page-zero",
        "page-negative",
    ],
)
def test_the_registry_refuses_what_no_one_family_serves(
    requirements: dict[GroupRole, tuple[AttentionRequirement, ...]], page_size: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        resolve_attention_config(requirements, _request(page_size=page_size))


def test_a_registered_family_receives_the_requirements_and_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        family_module,
        "FAMILIES",
        MappingProxyType(
            {**family_module.FAMILIES, "fake": ("tests.support.attention_family", "resolve")}
        ),
    )
    fake = AttentionRequirement(family="fake", slot="latent", geometry=(("head_dim", 64),))
    requirements = {GroupRole.TARGET: (fake,), GroupRole.DRAFT: (fake,)}
    request = _request("auto", page_size=None, tensor_cores=True)

    resolved = resolve_attention_config(requirements, request)

    assert isinstance(resolved, ResolvedFake)
    assert resolved.requirements is requirements
    assert resolved.request is request


@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize(("qo_heads", "kv_heads"), [(32, 8), (16, 1)])
def test_mha_holds_the_heads_each_ranks_attention_layer_has(
    monkeypatch: pytest.MonkeyPatch, ranks: int, qo_heads: int, kv_heads: int
) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    head_dim = 8
    # A draft of other heads does not change the target's.
    requirements = {
        GroupRole.TARGET: (AttentionRequirement.mha(qo_heads, kv_heads, head_dim),),
        GroupRole.DRAFT: (AttentionRequirement.mha(4, 2, head_dim),),
    }

    resolved = resolve_attention_config(
        requirements, _request(tensor_parallel_size=ranks, tensor_cores=False)
    )

    assert isinstance(resolved, mha_module.ResolvedMha)
    heads = (resolved.num_qo_heads, resolved.num_kv_heads)
    assert heads == (qo_heads // ranks, kv_head_window(kv_heads, 0, ranks)[1])
    for rank in range(ranks):
        # The QKV projection gives the attention layer its rank's heads.
        qkv = LinearQKVMerged(
            qo_heads * head_dim,
            head_dim,
            qo_heads,
            kv_heads,
            has_bias=False,
            site=site(Topology(rank, ranks)),
        )
        assert (qkv.head_dim, qkv.num_qo_heads, qkv.num_kv_heads) == (head_dim, *heads)
    assert resolved.head_dim == head_dim and resolved.tensor_cores is False
    assert resolved.layouts[GroupRole.TARGET, "kv"] == KVPair(
        resolved.num_kv_heads, head_dim, torch.bfloat16
    )


def test_mha_refuses_target_slots_of_two_head_geometries(monkeypatch: pytest.MonkeyPatch) -> None:
    set_validation(monkeypatch, FlashInferBackend)
    other = AttentionRequirement(
        family="mha", slot="other", geometry=(("qo_heads", 32), ("kv_heads", 8), ("head_dim", 128))
    )

    with pytest.raises(ValueError, match="one head geometry"):
        resolve_attention_config({GroupRole.TARGET: (*_TARGET, other)}, _request())
