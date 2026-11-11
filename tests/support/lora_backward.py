from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Literal, TypedDict

import torch

SnapshotRole = Literal["reference", "hf", "candidate"]
ParameterSchemaEntry = tuple[str, tuple[int, ...], str]
ProfileValue = str | bool | tuple[int, int]
BackwardProfilePayload = dict[str, ProfileValue]


class BackwardSnapshot(TypedDict):
    role: SnapshotRole
    profile: BackwardProfilePayload
    loss_sum: torch.Tensor
    num_target_tokens: int
    parameter_schema: tuple[ParameterSchemaEntry, ...]
    initial_lora: dict[str, torch.Tensor]
    raw_gradients: dict[str, torch.Tensor]
    trainable_names: tuple[str, ...]
    base_gradients_absent: bool


class BackwardComparisonReceipt(TypedDict):
    verdict: Literal["passed"]
    reference_profile: BackwardProfilePayload
    profile: BackwardProfilePayload
    candidate_profile: BackwardProfilePayload
    num_target_tokens: int
    parameter_schema: dict[str, int]
    loss: dict[str, float]
    raw_gradients: dict[str, object]


class InvalidBackwardSnapshotError(ValueError):
    """A runner payload cannot represent the frozen comparison seam."""


class BackwardMismatchError(AssertionError):
    """Valid snapshots violate an exact or numerical correctness gate."""


def _validated_parameter_schema(
    value: object, dtype: str = "float32"
) -> tuple[ParameterSchemaEntry, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("parameter schema must be a non-empty tuple")
    entries = []
    names = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise ValueError(f"parameter schema entry {index} must contain name, shape, dtype")
        name, shape, entry_dtype = entry
        if not isinstance(name, str) or not name:
            raise ValueError(f"parameter schema entry {index} has an invalid name")
        if name in names:
            raise ValueError(f"parameter schema contains duplicate name: {name}")
        if not isinstance(shape, tuple) or not shape:
            raise ValueError(f"parameter schema entry {name} has an invalid shape")
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
            for dimension in shape
        ):
            raise ValueError(f"parameter schema entry {name} has an invalid shape")
        if entry_dtype != dtype:
            raise ValueError(f"parameter schema entry {name} must use {dtype}")
        names.add(name)
        entries.append((name, shape, entry_dtype))
    return tuple(entries)


@dataclass(frozen=True)
class ProfileFingerprint:
    model: str
    revision: str
    python_version: str
    torch_version: str
    cuda_version: str
    transformers_version: str
    peft_version: str
    device: str
    capability: tuple[int, int]
    dtype: str
    case_id: str
    training_attention: str
    cublas_workspace_config: str
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    float32_matmul_precision: str
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool

    def __post_init__(self) -> None:
        for name in _STRING_PROFILE_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.dtype not in ("float32", "float64", "mixed[bfloat16,float32]"):
            raise ValueError(
                "backward comparison requires float32, float64 or mixed[bfloat16,float32]"
            )
        if (
            not isinstance(self.capability, tuple)
            or len(self.capability) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.capability
            )
        ):
            raise ValueError("capability must contain two non-negative integers")
        for name in _BOOL_PROFILE_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


_BOOL_PROFILE_FIELDS = (
    "deterministic_algorithms",
    "deterministic_warn_only",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
)
_PROFILE_FIELDS = tuple(field.name for field in fields(ProfileFingerprint))
_STRING_PROFILE_FIELDS = tuple(
    name for name in _PROFILE_FIELDS if name not in {*_BOOL_PROFILE_FIELDS, "capability"}
)


@dataclass(frozen=True)
class BackwardComparisonSpec:
    """Gates against the FP64 reference: the candidate's loss within
    ``loss_relative_bound``; its gradient collection's relative L2 error within
    ``collection_error_ratio_bound`` times HF FP32's; each named gradient's error RMS within
    ``tensor_error_ratio_bound`` times HF FP32's plus ``tensor_error_floor``."""

    expected_profile: ProfileFingerprint
    expected_reference_profile: ProfileFingerprint
    expected_parameter_schema: tuple[ParameterSchemaEntry, ...]
    expected_num_target_tokens: int
    loss_relative_bound: float
    collection_error_ratio_bound: float
    tensor_error_ratio_bound: float
    tensor_error_floor: float
    expected_candidate_profile: ProfileFingerprint | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.expected_profile, ProfileFingerprint):
            raise TypeError("expected_profile must be a ProfileFingerprint")
        if self.expected_profile.dtype != "float32":
            raise ValueError("HF backward comparison requires float32")
        if not isinstance(self.expected_reference_profile, ProfileFingerprint):
            raise TypeError("expected_reference_profile must be a ProfileFingerprint")
        if self.expected_reference_profile.dtype != "float64":
            raise ValueError("the backward reference requires float64")
        if self.expected_candidate_profile is not None and not isinstance(
            self.expected_candidate_profile, ProfileFingerprint
        ):
            raise TypeError("expected_candidate_profile must be a ProfileFingerprint")
        _validated_parameter_schema(self.expected_parameter_schema)
        if (
            isinstance(self.expected_num_target_tokens, bool)
            or not isinstance(self.expected_num_target_tokens, int)
            or self.expected_num_target_tokens <= 0
        ):
            raise ValueError("expected_num_target_tokens must be a positive integer")
        for name in (
            "loss_relative_bound",
            "collection_error_ratio_bound",
            "tensor_error_ratio_bound",
            "tensor_error_floor",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class _ValidatedSnapshot:
    profile: ProfileFingerprint
    loss_sum: torch.Tensor
    initial_lora: dict[str, torch.Tensor]
    raw_gradients: dict[str, torch.Tensor]


_SNAPSHOT_KEYS = {
    "role",
    "profile",
    "loss_sum",
    "num_target_tokens",
    "parameter_schema",
    "initial_lora",
    "raw_gradients",
    "trainable_names",
    "base_gradients_absent",
}


def _mapping(value: object, label: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise InvalidBackwardSnapshotError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[object, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise InvalidBackwardSnapshotError(
            f"{label} keys mismatch: missing={sorted(expected - actual, key=repr)}, "
            f"extra={sorted(actual - expected, key=repr)}"
        )


def _parse_profile(value: object, label: str) -> ProfileFingerprint:
    profile = _mapping(value, f"{label}.profile")
    _exact_keys(profile, set(_PROFILE_FIELDS), f"{label}.profile")
    try:
        return ProfileFingerprint(**dict(profile))
    except (TypeError, ValueError) as error:
        raise InvalidBackwardSnapshotError(f"invalid {label}.profile: {error}") from error


def _parse_tensor_map(
    value: object,
    schema: tuple[ParameterSchemaEntry, ...],
    dtype: torch.dtype,
    label: str,
) -> dict[str, torch.Tensor]:
    tensors = _mapping(value, label)
    expected_names = tuple(name for name, _, _ in schema)
    if tuple(tensors) != expected_names:
        raise InvalidBackwardSnapshotError(f"{label} names/order do not match parameter schema")
    result = {}
    for name, shape, _ in schema:
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise InvalidBackwardSnapshotError(f"{label}.{name} must be a tensor")
        if tensor.layout != torch.strided:
            raise InvalidBackwardSnapshotError(f"{label}.{name} must be dense")
        if tensor.device.type != "cpu":
            raise InvalidBackwardSnapshotError(f"{label}.{name} must reside on CPU")
        if tensor.dtype != dtype:
            raise InvalidBackwardSnapshotError(f"{label}.{name} must use {_dtype_name(dtype)}")
        if tuple(tensor.shape) != shape:
            raise InvalidBackwardSnapshotError(
                f"{label}.{name} shape {tuple(tensor.shape)} does not match {shape}"
            )
        if not bool(torch.all(torch.isfinite(tensor))):
            raise InvalidBackwardSnapshotError(f"{label}.{name} must be finite")
        result[name] = tensor
    return result


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _validate_snapshot(
    payload: Mapping[object, object],
    *,
    role: SnapshotRole,
    spec: BackwardComparisonSpec,
) -> _ValidatedSnapshot:
    """Validate one role's snapshot: the reference stores FP64, HF and the candidate FP32."""

    _exact_keys(payload, _SNAPSHOT_KEYS, role)
    actual_role = payload["role"]
    if actual_role not in ("reference", "hf", "candidate"):
        raise InvalidBackwardSnapshotError(f"{role}.role must be 'reference', 'hf' or 'candidate'")
    if actual_role != role:
        raise BackwardMismatchError(f"{role}.role is {actual_role!r}, expected {role!r}")
    profile = _parse_profile(payload["profile"], role)
    if role == "reference":
        expected_profile = spec.expected_reference_profile
    elif role == "candidate":
        expected_profile = spec.expected_candidate_profile or spec.expected_profile
    else:
        expected_profile = spec.expected_profile
    if profile != expected_profile:
        raise BackwardMismatchError(f"{role}.profile does not match the external profile")
    dtype = torch.float64 if role == "reference" else torch.float32
    try:
        schema = _validated_parameter_schema(payload["parameter_schema"], _dtype_name(dtype))
    except ValueError as error:
        raise InvalidBackwardSnapshotError(f"invalid {role}.parameter_schema: {error}") from error
    if tuple((name, shape) for name, shape, _ in schema) != tuple(
        (name, shape) for name, shape, _ in spec.expected_parameter_schema
    ):
        raise BackwardMismatchError(f"{role}.parameter_schema does not match the external schema")

    num_target_tokens = payload["num_target_tokens"]
    if isinstance(num_target_tokens, bool) or not isinstance(num_target_tokens, int):
        raise InvalidBackwardSnapshotError(f"{role}.num_target_tokens must be an integer")
    if num_target_tokens <= 0:
        raise InvalidBackwardSnapshotError(f"{role}.num_target_tokens must be positive")
    if num_target_tokens != spec.expected_num_target_tokens:
        raise BackwardMismatchError(
            f"{role}.num_target_tokens is {num_target_tokens}, "
            f"expected {spec.expected_num_target_tokens}"
        )
    expected_names = tuple(name for name, _, _ in schema)
    trainable_names = payload["trainable_names"]
    if not isinstance(trainable_names, tuple) or not all(
        isinstance(name, str) for name in trainable_names
    ):
        raise InvalidBackwardSnapshotError(f"{role}.trainable_names must be a string tuple")
    if trainable_names != expected_names:
        raise BackwardMismatchError(f"{role}.trainable_names does not match the external schema")
    if not isinstance(payload["base_gradients_absent"], bool):
        raise InvalidBackwardSnapshotError(f"{role}.base_gradients_absent must be bool")
    if not payload["base_gradients_absent"]:
        raise BackwardMismatchError(f"{role} contains a base-parameter gradient")

    loss_sum = payload["loss_sum"]
    if not isinstance(loss_sum, torch.Tensor):
        raise InvalidBackwardSnapshotError(f"{role}.loss_sum must be a tensor")
    if (
        loss_sum.layout != torch.strided
        or loss_sum.device.type != "cpu"
        or loss_sum.dtype != dtype
        or loss_sum.ndim != 0
    ):
        raise InvalidBackwardSnapshotError(
            f"{role}.loss_sum must be a dense CPU {_dtype_name(dtype)} scalar"
        )
    if not bool(torch.isfinite(loss_sum)) or float(loss_sum) < 0:
        raise InvalidBackwardSnapshotError(f"{role}.loss_sum must be finite and non-negative")

    initial_lora = _parse_tensor_map(payload["initial_lora"], schema, dtype, f"{role}.initial_lora")
    raw_gradients = _parse_tensor_map(
        payload["raw_gradients"], schema, dtype, f"{role}.raw_gradients"
    )
    zero_initial_names = [
        name for name, tensor in initial_lora.items() if torch.count_nonzero(tensor).item() == 0
    ]
    if zero_initial_names:
        raise BackwardMismatchError(
            f"{role}.initial_lora tensors must be nonzero: {zero_initial_names}"
        )
    return _ValidatedSnapshot(
        profile=profile,
        loss_sum=loss_sum,
        initial_lora=initial_lora,
        raw_gradients=raw_gradients,
    )


def _profile_payload(profile: ProfileFingerprint) -> BackwardProfilePayload:
    return {name: getattr(profile, name) for name in _PROFILE_FIELDS}


def compare_backward_snapshots(
    *,
    reference: object,
    hf: object,
    candidate: object,
    spec: BackwardComparisonSpec,
) -> BackwardComparisonReceipt:
    """Measure HF FP32 and the candidate against the FP64 reference and gate the candidate's
    error on HF FP32's."""

    reference_snapshot, hf_snapshot, candidate_snapshot = (
        _validate_snapshot(_mapping(payload, role), role=role, spec=spec)
        for role, payload in (("reference", reference), ("hf", hf), ("candidate", candidate))
    )
    for name, _, _ in spec.expected_parameter_schema:
        hf_initial = hf_snapshot.initial_lora[name]
        if not torch.equal(candidate_snapshot.initial_lora[name], hf_initial):
            raise BackwardMismatchError(f"candidate initial LoRA tensor differs: {name}")
        if not torch.equal(reference_snapshot.initial_lora[name], hf_initial.double()):
            raise BackwardMismatchError(f"reference initial LoRA tensor differs: {name}")

    reference_loss = float(reference_snapshot.loss_sum)
    hf_loss = float(hf_snapshot.loss_sum.double())
    candidate_loss = float(candidate_snapshot.loss_sum.double())
    if reference_loss <= 0:
        raise InvalidBackwardSnapshotError("reference loss must be positive")
    loss_relative_error = abs(candidate_loss - reference_loss) / reference_loss

    reference_sums = []
    hf_error_sums = []
    candidate_error_sums = []
    worst_name = ""
    worst_candidate_rms = -1.0
    worst_hf_rms = -1.0
    worst_allowed_rms = -1.0
    worst_utilization = -1.0
    max_absolute_name = ""
    max_absolute_flat_index = -1
    max_absolute_error = -1.0
    failing_names = []
    for name, _, _ in spec.expected_parameter_schema:
        expected = reference_snapshot.raw_gradients[name]
        hf_difference = hf_snapshot.raw_gradients[name].double() - expected
        difference = candidate_snapshot.raw_gradients[name].double() - expected
        reference_sum = float(torch.sum(expected.square(), dtype=torch.float64))
        hf_error_sum = float(torch.sum(hf_difference.square(), dtype=torch.float64))
        error_sum = float(torch.sum(difference.square(), dtype=torch.float64))
        if not all(math.isfinite(value) for value in (reference_sum, hf_error_sum, error_sum)):
            raise InvalidBackwardSnapshotError(f"nonfinite FP64 reduction for {name}")
        hf_rms = math.sqrt(hf_error_sum / expected.numel())
        candidate_rms = math.sqrt(error_sum / expected.numel())
        allowed_rms = spec.tensor_error_ratio_bound * hf_rms + spec.tensor_error_floor
        utilization = candidate_rms / allowed_rms
        if not all(
            math.isfinite(value) for value in (hf_rms, candidate_rms, allowed_rms, utilization)
        ):
            raise InvalidBackwardSnapshotError(f"nonfinite RMS reduction for {name}")
        if candidate_rms > allowed_rms:
            failing_names.append(name)
        if utilization > worst_utilization:
            worst_name = name
            worst_candidate_rms = candidate_rms
            worst_hf_rms = hf_rms
            worst_allowed_rms = allowed_rms
            worst_utilization = utilization
        absolute_error = difference.abs().reshape(-1)
        flat_index = int(torch.argmax(absolute_error))
        tensor_max_absolute_error = float(absolute_error[flat_index])
        if tensor_max_absolute_error > max_absolute_error:
            max_absolute_name = name
            max_absolute_flat_index = flat_index
            max_absolute_error = tensor_max_absolute_error
        reference_sums.append(reference_sum)
        hf_error_sums.append(hf_error_sum)
        candidate_error_sums.append(error_sum)

    collection_reference_sum = math.fsum(reference_sums)
    collection_hf_error_sum = math.fsum(hf_error_sums)
    collection_error_sum = math.fsum(candidate_error_sums)
    if not all(
        math.isfinite(value)
        for value in (collection_reference_sum, collection_hf_error_sum, collection_error_sum)
    ):
        raise InvalidBackwardSnapshotError("nonfinite gradient collection reduction")
    if collection_reference_sum <= 0:
        raise InvalidBackwardSnapshotError("reference gradient collection norm must be nonzero")
    hf_collection_error = math.sqrt(collection_hf_error_sum / collection_reference_sum)
    collection_error = math.sqrt(collection_error_sum / collection_reference_sum)

    metrics: dict[str, object] = {
        "reference_profile": _profile_payload(reference_snapshot.profile),
        "profile": _profile_payload(hf_snapshot.profile),
        "candidate_profile": _profile_payload(candidate_snapshot.profile),
        "num_target_tokens": spec.expected_num_target_tokens,
        "parameter_schema": {
            "count": len(spec.expected_parameter_schema),
            "total_numel": sum(math.prod(shape) for _, shape, _ in spec.expected_parameter_schema),
        },
        "loss": {
            "reference": reference_loss,
            "hf": hf_loss,
            "candidate": candidate_loss,
            "hf_relative_error": abs(hf_loss - reference_loss) / reference_loss,
            "relative_error": loss_relative_error,
            "bound": spec.loss_relative_bound,
        },
        "raw_gradients": {
            "collection_relative_l2": {
                "hf": hf_collection_error,
                "candidate": collection_error,
                "ratio": collection_error / hf_collection_error if hf_collection_error else None,
                "bound": spec.collection_error_ratio_bound,
            },
            "worst_named": {
                "name": worst_name,
                "candidate_rms": worst_candidate_rms,
                "hf_rms": worst_hf_rms,
                "ratio": worst_candidate_rms / worst_hf_rms if worst_hf_rms else None,
                "allowed_rms": worst_allowed_rms,
                "bound": spec.tensor_error_ratio_bound,
                "floor": spec.tensor_error_floor,
            },
            "maximum_absolute": {
                "name": max_absolute_name,
                "flat_index": max_absolute_flat_index,
                "error": max_absolute_error,
            },
        },
    }
    failures = []
    if loss_relative_error > spec.loss_relative_bound:
        failures.append(
            f"loss relative error {loss_relative_error} exceeds {spec.loss_relative_bound}"
        )
    if failing_names:
        failures.append(
            f"named gradient error RMS exceeds {spec.tensor_error_ratio_bound}x HF FP32's "
            f"for {failing_names}"
        )
    collection_allowed = spec.collection_error_ratio_bound * hf_collection_error
    if collection_error > collection_allowed:
        failures.append(
            f"gradient collection relative L2 {collection_error} exceeds {collection_allowed}"
        )
    if failures:
        raise BackwardMismatchError(f"{'; '.join(failures)}; metrics={metrics}")
    return {"verdict": "passed", **metrics}


__all__ = [
    "BackwardComparisonReceipt",
    "BackwardComparisonSpec",
    "BackwardMismatchError",
    "BackwardProfilePayload",
    "BackwardSnapshot",
    "InvalidBackwardSnapshotError",
    "ParameterSchemaEntry",
    "ProfileFingerprint",
    "SnapshotRole",
    "compare_backward_snapshots",
]
