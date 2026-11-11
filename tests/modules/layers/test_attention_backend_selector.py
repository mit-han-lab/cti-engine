from __future__ import annotations

import pytest
import torch

from cti.attention import AttentionRequest, AttentionRequirement, resolve_attention_config
from cti.attention import flash_attention as flash_attention_module
from cti.attention import flashinfer as flashinfer_module
from cti.attention import tensorrt_llm as trtllm_module
from cti.attention.base import PageConstraint
from cti.attention.flash_attention import FlashAttentionBackend
from cti.attention.flashinfer import FlashInferBackend
from cti.attention.paged import KVPair
from cti.attention.tensorrt_llm import TensorRTLLMBackend
from cti.forward import RoundShape
from cti.kvcache import GroupRole, choose_page_size
from cti.utils import DeviceCapability
from tests.support.attention import CPUAttention, set_validation

_HEAD_DIM_128 = (AttentionRequirement.mha(16, 8, 128),)
_HEAD_DIM_256 = (AttentionRequirement.mha(16, 8, 256),)


def _resolve(requested, *, requirements, tensor_parallel_size=1, **options):
    """One BF16 rank's resolution for the target's ``requirements``."""
    return resolve_attention_config(
        {GroupRole.TARGET: requirements},
        AttentionRequest(
            target=GroupRole.TARGET,
            backend=requested,
            tensor_parallel_size=tensor_parallel_size,
            dtype=torch.bfloat16,
            **options,
        ),
    )


@pytest.mark.parametrize(
    ("tensor_parallel_size", "num_kv_heads"), [(1, 8), (2, 4), (8, 1), (16, 1)]
)
def test_each_roles_slot_gets_this_ranks_kv_pair(monkeypatch, tensor_parallel_size, num_kv_heads):
    set_validation(monkeypatch, FlashInferBackend)
    draft = AttentionRequirement.mha(32, 4, 64)

    resolved = resolve_attention_config(
        {GroupRole.TARGET: _HEAD_DIM_128, GroupRole.DRAFT: (draft,)},
        AttentionRequest(
            target=GroupRole.TARGET,
            backend="fi",
            page_size=1,
            capability=DeviceCapability(12, 0),
            tensor_parallel_size=tensor_parallel_size,
            dtype=torch.float16,
        ),
    )

    # Both models name their slot "kv"; the role keeps them apart. KV heads split, or replicate
    # past the heads.
    assert resolved.layouts == {
        (GroupRole.TARGET, "kv"): KVPair(num_kv_heads, 128, torch.float16),
        (GroupRole.DRAFT, "kv"): KVPair(max(1, 4 // tensor_parallel_size), 64, torch.float16),
    }


def test_one_slot_refuses_two_row_layouts(monkeypatch):
    set_validation(monkeypatch, FlashInferBackend)
    with pytest.raises(ValueError, match="two row layouts"):
        _resolve(
            "fi",
            requirements=_HEAD_DIM_128 + (AttentionRequirement.mha(16, 4, 128),),
            page_size=1,
            capability=DeviceCapability(12, 0),
        )


def _fail_kernel_check(monkeypatch, module, check_name: str, error: Exception) -> None:
    def check_with_failure():
        raise error

    monkeypatch.setattr(module, check_name, check_with_failure)


@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_auto_falls_back_when_flash_attention_extension_cannot_load(
    monkeypatch,
    error_type,
):
    monkeypatch.setattr(flashinfer_module, "check_flashinfer_available", lambda: None)
    _fail_kernel_check(
        monkeypatch,
        flash_attention_module,
        "check_flash_attention_available",
        error_type("cannot load FlashAttention provider"),
    )

    resolved = _resolve(
        "auto",
        page_size=None,
        capability=DeviceCapability(9, 0),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.selected == "fi"
    assert "SGLang FlashAttention extension could not be loaded" in resolved.reason
    assert error_type.__name__ in resolved.reason
    assert "cannot load FlashAttention provider" in resolved.reason


@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_explicit_flash_attention_rejects_extension_load_failure(
    monkeypatch,
    error_type,
):
    _fail_kernel_check(
        monkeypatch,
        flash_attention_module,
        "check_flash_attention_available",
        error_type("cannot load FlashAttention provider"),
    )

    with pytest.raises(
        ValueError, match="SGLang FlashAttention extension could not be loaded"
    ) as caught:
        _resolve(
            "fa",
            page_size=None,
            capability=DeviceCapability(9, 0),
            requirements=_HEAD_DIM_128,
        )

    assert error_type.__name__ in str(caught.value)
    assert "cannot load FlashAttention provider" in str(caught.value)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_auto_selects_trtllm_on_supported_capabilities(monkeypatch, capability):
    set_validation(monkeypatch, TensorRTLLMBackend)

    resolved = _resolve(
        "auto",
        page_size=None,
        capability=DeviceCapability(*capability),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.selected == "trtllm"


@pytest.mark.parametrize(
    "capability",
    [(8, 0), (8, 6), (10, 1), (11, 0), (12, 0), (12, 1), (13, 0)],
)
def test_auto_uses_flashinfer_outside_trtllm_capabilities(monkeypatch, capability):
    set_validation(monkeypatch, FlashInferBackend)

    resolved = _resolve(
        "auto",
        page_size=None,
        capability=DeviceCapability(*capability),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.selected == "fi"


def test_auto_uses_hopper_flash_attention(monkeypatch):
    set_validation(monkeypatch, FlashAttentionBackend)
    set_validation(monkeypatch, FlashInferBackend)

    resolved = _resolve(
        "auto",
        page_size=None,
        capability=DeviceCapability(9, 0),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.selected == "fa,fi"


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
def test_auto_can_fall_back_to_flash_attention_without_flashinfer(
    monkeypatch,
    capability,
):
    set_validation(monkeypatch, TensorRTLLMBackend, "TRTLLM unavailable")
    set_validation(monkeypatch, FlashInferBackend, "FlashInfer unavailable")
    set_validation(monkeypatch, FlashAttentionBackend)

    resolved = _resolve(
        "auto",
        page_size=None,
        capability=DeviceCapability(*capability),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.selected == "fa"
    assert "uses SGLang FlashAttention" in resolved.reason


def test_auto_records_rejection_reason_before_falling_back(monkeypatch):
    set_validation(
        monkeypatch,
        FlashAttentionBackend,
        "SGLang FlashAttention extension could not be loaded",
    )
    set_validation(monkeypatch, FlashInferBackend)

    resolved = _resolve(
        "auto",
        page_size=None,
        capability=DeviceCapability(9, 0),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.selected == "fi"
    assert "rejected fa,fi: SGLang FlashAttention extension could not be loaded" in (
        resolved.reason
    )


def test_auto_records_trtllm_rejection_before_falling_back(monkeypatch):
    set_validation(
        monkeypatch,
        TensorRTLLMBackend,
        "FlashInfer TRTLLM attention functions could not be loaded",
    )
    set_validation(monkeypatch, FlashInferBackend)

    resolved = _resolve(
        "auto",
        page_size=None,
        capability=DeviceCapability(10, 0),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.selected == "fi"
    assert "rejected trtllm" in resolved.reason


def test_explicit_trtllm_rejects_sm120():
    with pytest.raises(ValueError, match=r"TRTLLM.*SM120.*SM100 and SM103"):
        _resolve(
            "trtllm",
            page_size=None,
            capability=DeviceCapability(12, 0),
            requirements=_HEAD_DIM_128,
        )


def test_explicit_backend_rejects_unloadable_extension(monkeypatch):
    set_validation(
        monkeypatch,
        FlashAttentionBackend,
        "SGLang FlashAttention extension could not be loaded",
    )
    set_validation(monkeypatch, FlashInferBackend)

    with pytest.raises(ValueError, match="SGLang FlashAttention extension could not be loaded"):
        _resolve(
            "fa,fi",
            page_size=None,
            capability=DeviceCapability(9, 0),
            requirements=_HEAD_DIM_128,
        )


def test_explicit_supported_backend_is_preserved(monkeypatch):
    set_validation(monkeypatch, FlashAttentionBackend)
    set_validation(monkeypatch, FlashInferBackend)

    resolved = _resolve(
        "fa,fi",
        page_size=None,
        capability=DeviceCapability(9, 0),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.selected == "fa,fi"
    assert resolved.reason == "explicit attention backend"


def test_sm120_does_not_validate_unrelated_backends(monkeypatch):
    set_validation(monkeypatch, FlashInferBackend)
    monkeypatch.setattr(
        FlashAttentionBackend,
        "validate_environment",
        classmethod(lambda cls, capability: pytest.fail("FA should not be checked on SM120")),
    )
    monkeypatch.setattr(
        TensorRTLLMBackend,
        "validate_environment",
        classmethod(lambda cls, capability: pytest.fail("TRTLLM should not be checked on SM120")),
    )

    resolved = _resolve(
        "auto",
        page_size=None,
        capability=DeviceCapability(12, 0),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.selected == "fi"


def test_resolution_fails_when_flashinfer_is_unavailable(monkeypatch):
    set_validation(
        monkeypatch,
        FlashInferBackend,
        "FlashInfer attention functions could not be loaded",
    )

    with pytest.raises(ValueError, match="FlashInfer attention functions could not be loaded"):
        _resolve(
            "auto",
            page_size=None,
            capability=DeviceCapability(12, 0),
            requirements=_HEAD_DIM_128,
        )


@pytest.mark.parametrize(
    ("capability", "rejected"),
    [
        pytest.param((9, 0), "fa,fi", id="sm90"),
        pytest.param((10, 0), "trtllm", id="sm100"),
        pytest.param((10, 3), "trtllm", id="sm103"),
    ],
)
def test_auto_skips_backends_without_an_accepted_operation_test_for_the_head_dim(
    monkeypatch,
    capability,
    rejected,
):
    for backend_class in (FlashAttentionBackend, FlashInferBackend, TensorRTLLMBackend):
        set_validation(monkeypatch, backend_class)

    resolved = _resolve(
        "auto",
        page_size=None,
        capability=DeviceCapability(*capability),
        requirements=_HEAD_DIM_256,
    )

    assert resolved.selected == "fi"
    assert f"rejected {rejected}: " in resolved.reason
    assert "no accepted operation test for head_dim 256" in resolved.reason


@pytest.mark.parametrize(
    ("backend", "unqualified"),
    [("fa", "fa"), ("trtllm", "trtllm"), ("fa,fi", "fa"), ("fi,trtllm", "trtllm")],
)
def test_explicit_backend_without_an_accepted_operation_test_for_the_head_dim_fails(
    monkeypatch,
    backend,
    unqualified,
):
    for backend_class in (FlashAttentionBackend, FlashInferBackend, TensorRTLLMBackend):
        set_validation(monkeypatch, backend_class)

    with pytest.raises(ValueError) as caught:
        _resolve(
            backend,
            page_size=None,
            capability=DeviceCapability(10, 0),
            requirements=_HEAD_DIM_256,
        )

    assert str(caught.value) == (
        f"Attention backend '{backend}' is unavailable: {unqualified} has no accepted "
        "operation test for head_dim 256"
    )


def test_flashinfer_serves_head_dim_256(monkeypatch):
    set_validation(monkeypatch, FlashInferBackend)

    resolved = _resolve(
        "fi",
        page_size=None,
        capability=DeviceCapability(12, 0),
        requirements=_HEAD_DIM_256,
    )

    assert resolved.selected == "fi"


def test_backend_classes_own_architecture_validation():
    assert TensorRTLLMBackend.validate_environment(DeviceCapability(12, 0)) == (
        "FlashInfer TRTLLM attention does not support SM120; "
        "supported capabilities are SM100 and SM103",
    )
    assert FlashAttentionBackend.validate_environment(DeviceCapability(12, 0)) == (
        "SGLang FlashAttention does not support SM120; supported capabilities are SM90 and SM100",
    )
    assert FlashAttentionBackend.validate_environment(DeviceCapability(10, 3)) == (
        "SGLang FlashAttention does not support SM103; supported capabilities are SM90 and SM100",
    )
    assert FlashInferBackend.validate_environment(None) == ("CUDA device is unavailable",)


@pytest.mark.parametrize("backend", ["", "unknown", "fi,", "fa,fi,trtllm"])
def test_resolution_rejects_invalid_backend_names(backend):
    with pytest.raises(ValueError):
        _resolve(backend, page_size=None, capability=None, requirements=_HEAD_DIM_128)


def test_trtllm_declares_its_page_sizes_and_kvcache_chooses_the_smallest(monkeypatch):
    set_validation(monkeypatch, TensorRTLLMBackend)

    resolved = _resolve(
        "trtllm", page_size=None, capability=DeviceCapability(10, 0), requirements=_HEAD_DIM_128
    )

    assert resolved.pages == (PageConstraint(frozenset({16, 32, 64})),)
    assert choose_page_size(None, resolved.pages, (1,)) == 16
    assert choose_page_size(64, resolved.pages, (1,)) == 64
    with pytest.raises(ValueError, match="page size 1 does not suit"):
        choose_page_size(1, resolved.pages, (1,))


def test_explicit_trtllm_refuses_a_page_size_it_does_not_take(monkeypatch):
    set_validation(monkeypatch, TensorRTLLMBackend)

    with pytest.raises(ValueError, match="trtllm does not take page size 1"):
        _resolve(
            "trtllm", page_size=1, capability=DeviceCapability(10, 0), requirements=_HEAD_DIM_128
        )


def test_auto_passes_over_trtllm_for_a_page_size_it_does_not_take(monkeypatch):
    set_validation(monkeypatch, TensorRTLLMBackend)
    set_validation(monkeypatch, FlashInferBackend)

    resolved = _resolve(
        "auto", page_size=1, capability=DeviceCapability(10, 0), requirements=_HEAD_DIM_128
    )

    assert resolved.selected == "fi"
    assert "trtllm: trtllm does not take page size 1" in resolved.reason


def test_flashinfer_takes_any_page_size(monkeypatch):
    set_validation(monkeypatch, FlashInferBackend)

    resolved = _resolve(
        "fi", page_size=None, capability=DeviceCapability(12, 0), requirements=_HEAD_DIM_128
    )

    assert resolved.pages == (PageConstraint(),)
    assert choose_page_size(None, resolved.pages, (1,)) == 1
    assert choose_page_size(16, resolved.pages, (1,)) == 16


def test_duplicate_backend_is_canonical_single(monkeypatch):
    calls = []

    def validate(cls, capability):
        calls.append(capability)
        return ()

    monkeypatch.setattr(FlashInferBackend, "validate_environment", classmethod(validate))
    capability = DeviceCapability(12, 0)

    resolved = _resolve(
        "fi,fi",
        page_size=None,
        capability=capability,
        requirements=_HEAD_DIM_128,
    )

    assert resolved.ragged_backend == "fi"
    assert resolved.decode_backend == "fi"
    assert resolved.selected == "fi"
    assert not resolved.is_hybrid
    assert calls == [capability]


def test_hybrid_plan_preserves_ragged_decode_order(monkeypatch):
    set_validation(monkeypatch, FlashAttentionBackend)
    set_validation(monkeypatch, FlashInferBackend)

    resolved = _resolve(
        "fa,fi",
        page_size=None,
        capability=DeviceCapability(9, 0),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.ragged_backend == "fa"
    assert resolved.decode_backend == "fi"
    assert resolved.selected == "fa,fi"
    assert resolved.is_hybrid


def test_auto_falls_back_when_trtllm_functions_cannot_load(monkeypatch):
    monkeypatch.setattr(flashinfer_module, "check_flashinfer_available", lambda: None)
    _fail_kernel_check(
        monkeypatch,
        trtllm_module,
        "check_trtllm_available",
        ImportError("cannot load TRTLLM provider"),
    )

    resolved = _resolve(
        "auto",
        page_size=None,
        capability=DeviceCapability(10, 0),
        requirements=_HEAD_DIM_128,
    )

    assert resolved.selected == "fi"
    assert "TRTLLM attention functions could not be loaded" in resolved.reason


def test_rounds_run_only_beside_a_fixed_query_backend(monkeypatch):
    for backend_class in (FlashAttentionBackend, FlashInferBackend, TensorRTLLMBackend):
        set_validation(monkeypatch, backend_class)
    rounds = RoundShape(16, 2, 4)

    resolved = _resolve(
        "auto",
        requirements=_HEAD_DIM_128,
        page_size=None,
        capability=DeviceCapability(10, 0),
        rounds=rounds,
    )
    assert resolved.selected == "fi"
    assert "trtllm does not run beside fixed-query rounds" in resolved.reason
    with pytest.raises(ValueError, match="fa does not run beside fixed-query rounds"):
        _resolve(
            "fa,fi",
            requirements=_HEAD_DIM_128,
            page_size=None,
            capability=DeviceCapability(9, 0),
            rounds=rounds,
        )


def test_every_requirement_head_dim_is_qualified(monkeypatch):
    set_validation(monkeypatch, TensorRTLLMBackend)

    with pytest.raises(ValueError, match="no accepted operation test for head_dim 256"):
        resolve_attention_config(
            {GroupRole.TARGET: _HEAD_DIM_128, GroupRole.DRAFT: _HEAD_DIM_256},
            AttentionRequest(
                target=GroupRole.TARGET,
                backend="trtllm",
                page_size=64,
                capability=DeviceCapability(10, 0),
                tensor_parallel_size=1,
                dtype=torch.bfloat16,
            ),
        )


def test_requirements_outside_the_mha_family_are_refused():
    latent = AttentionRequirement(family="mla", slot="kv", geometry=(("head_dim", 512),))
    with pytest.raises(ValueError, match="no attention family 'mla'"):
        _resolve("fi", requirements=(latent,), page_size=None, capability=DeviceCapability(12, 0))


def test_none_capability_means_cuda_is_unavailable():
    with pytest.raises(ValueError, match="CUDA device is unavailable"):
        _resolve("auto", page_size=None, capability=None, requirements=_HEAD_DIM_128)


def test_device_capability_separates_minimum_features_from_kernel_families():
    sm120 = DeviceCapability(12, 0)
    assert sm120.is_at_least(10, 0)
    assert not sm120.is_family(10)
    assert sm120.is_family(12)

    sm103 = DeviceCapability(10, 3)
    assert sm103.is_at_least(10, 0)
    assert sm103.is_family(10)
    assert not sm103.is_at_least(12, 0)


def test_flashinfer_environment_validation_uses_public_kernel_check(monkeypatch):
    calls = []
    monkeypatch.setattr(flashinfer_module, "check_flashinfer_available", lambda: calls.append(()))

    reasons = FlashInferBackend.validate_environment(DeviceCapability(12, 0))

    assert reasons == ()
    assert calls == [()]


@pytest.mark.parametrize(
    ("num_qo_heads", "num_kv_heads", "override", "expected_tensor_cores"),
    [
        (32, 8, None, True),
        (32, 16, None, False),
        (32, 8, False, False),
        (32, 16, True, True),
        # Rank-local heads: 8 query and 2 KV heads at TP 4 keep 2 query heads per KV head.
        (2, 1, None, False),
    ],
)
def test_flashinfer_operation_plan_is_resolved_once_before_execution(
    monkeypatch,
    num_qo_heads,
    num_kv_heads,
    override,
    expected_tensor_cores,
):
    attention = CPUAttention(monkeypatch, prefill="fa3")

    attention.build(heads=(num_qo_heads, num_kv_heads, 8), tensor_cores=override)

    prefill, decode = (operation.options for operation in attention.operations)
    assert (prefill["backend"], decode["backend"]) == ("fa3", "fa2")
    assert decode["use_tensor_cores"] is expected_tensor_cores
    # An explicit tensor-core choice keeps the native routes; otherwise the device tables
    # decide per batch.
    assert (
        prefill["dispatch"] == decode["dispatch"] == ("device" if override is None else "provider")
    )
    assert attention.prefill_choices == [(torch.device("cpu"), torch.bfloat16)]


@pytest.mark.parametrize(
    ("implementation", "dtype", "error_type", "message"),
    [
        ("fa2", torch.float32, ValueError, "requires FP16 or BF16"),
        (
            "unknown",
            torch.bfloat16,
            RuntimeError,
            "unsupported paged-prefill implementation",
        ),
    ],
)
def test_flashinfer_operation_plan_rejects_unsupported_provider_decisions(
    monkeypatch,
    implementation,
    dtype,
    error_type,
    message,
):
    attention = CPUAttention(monkeypatch, prefill=implementation)

    with pytest.raises(error_type, match=message):
        attention.build(dtype=dtype)
    assert attention.operations == []


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_trtllm_environment_validation_accepts_exact_supported_set(monkeypatch, capability):
    monkeypatch.setattr(trtllm_module, "check_trtllm_available", lambda: None)

    assert TensorRTLLMBackend.validate_environment(DeviceCapability(*capability)) == ()


@pytest.mark.parametrize("capability", [(9, 0), (10, 1), (12, 0)])
def test_trtllm_environment_validation_rejects_other_capabilities(capability):
    reasons = TensorRTLLMBackend.validate_environment(DeviceCapability(*capability))

    assert len(reasons) == 1
    assert "supported capabilities are SM100 and SM103" in reasons[0]


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
def test_flash_attention_environment_validation_accepts_exact_supported_set(
    monkeypatch,
    capability,
):
    monkeypatch.setattr(flash_attention_module, "check_flash_attention_available", lambda: None)

    assert FlashAttentionBackend.validate_environment(DeviceCapability(*capability)) == ()


@pytest.mark.parametrize("capability", [(10, 3), (12, 0)])
def test_flash_attention_environment_validation_rejects_other_capabilities(capability):
    reasons = FlashAttentionBackend.validate_environment(DeviceCapability(*capability))

    assert len(reasons) == 1
    assert "supported capabilities are SM90 and SM100" in reasons[0]
