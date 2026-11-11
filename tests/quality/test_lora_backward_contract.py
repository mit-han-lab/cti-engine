from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import transformers

from cti.models.qwen3_5.config import Qwen3_5Config
from tests.support.lora_backward import (
    BackwardComparisonSpec,
    BackwardMismatchError,
    BackwardProfilePayload,
    BackwardSnapshot,
    InvalidBackwardSnapshotError,
    ProfileFingerprint,
    compare_backward_snapshots,
)
from tests.support.qwen3_5 import TINY_QWEN3_5_CONFIG, load_hf_config, text_only_config
from tests.support.qwen3_5_lora import (
    TEXT_ONLY_PREFIX,
    Qwen3_5CanonicalLoRALayout,
    random_peft_state,
)


def _profile_payload() -> BackwardProfilePayload:
    return {
        "model": "test/model",
        "revision": "revision",
        "python_version": "3.12.13",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "transformers_version": "4.57.3",
        "peft_version": "0.19.1",
        "device": "NVIDIA GeForce RTX 5090",
        "capability": (12, 0),
        "dtype": "float32",
        "case_id": "tiny-backward",
        "training_attention": "torch-sdpa",
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }


PROFILE = ProfileFingerprint(**_profile_payload())
REFERENCE_PROFILE = replace(PROFILE, dtype="float64")
UNIT = 2**-10
FLOOR = 2**-20

Snapshots = dict[str, BackwardSnapshot]


def _spec(
    reference: dict[str, torch.Tensor],
    *,
    loss_bound: float = 1.0,
    collection_bound: float = 1.5,
    tensor_bound: float = 4.0,
    floor: float = FLOOR,
) -> BackwardComparisonSpec:
    return BackwardComparisonSpec(
        expected_profile=PROFILE,
        expected_reference_profile=REFERENCE_PROFILE,
        expected_parameter_schema=tuple(
            (name, tuple(tensor.shape), "float32") for name, tensor in reference.items()
        ),
        expected_num_target_tokens=7,
        loss_relative_bound=loss_bound,
        collection_error_ratio_bound=collection_bound,
        tensor_error_ratio_bound=tensor_bound,
        tensor_error_floor=floor,
    )


def _snapshot(
    role: str,
    gradients: dict[str, torch.Tensor],
    spec: BackwardComparisonSpec,
    *,
    loss: float = 1.0,
) -> BackwardSnapshot:
    dtype = torch.float64 if role == "reference" else torch.float32
    dtype_name = str(dtype).removeprefix("torch.")
    initial_lora = {
        name: torch.full(shape, float(index + 1), dtype=dtype)
        for index, (name, shape, _) in enumerate(spec.expected_parameter_schema)
    }
    return {
        "role": role,
        "profile": {**_profile_payload(), "dtype": dtype_name},
        "loss_sum": torch.tensor(loss, dtype=dtype),
        "num_target_tokens": spec.expected_num_target_tokens,
        "parameter_schema": tuple(
            (name, shape, dtype_name) for name, shape, _ in spec.expected_parameter_schema
        ),
        "initial_lora": initial_lora,
        "raw_gradients": {name: tensor.to(dtype, copy=True) for name, tensor in gradients.items()},
        "trainable_names": tuple(name for name, _, _ in spec.expected_parameter_schema),
        "base_gradients_absent": True,
    }


def _trio(
    candidate: dict[str, torch.Tensor],
    hf: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    candidate_loss: float = 1.0,
    hf_loss: float = 1.0,
    reference_loss: float = 1.0,
    **bounds: float,
) -> tuple[Snapshots, BackwardComparisonSpec]:
    spec = _spec(reference, **bounds)
    snapshots = {
        "reference": _snapshot("reference", reference, spec, loss=reference_loss),
        "hf": _snapshot("hf", hf, spec, loss=hf_loss),
        "candidate": _snapshot("candidate", candidate, spec, loss=candidate_loss),
    }
    return snapshots, spec


def _offset(gradients: dict[str, torch.Tensor], offset: float) -> dict[str, torch.Tensor]:
    return {name: tensor + offset for name, tensor in gradients.items()}


def _compare(snapshots: Snapshots, spec: BackwardComparisonSpec) -> dict[str, object]:
    return compare_backward_snapshots(**snapshots, spec=spec)


@pytest.mark.parametrize("numel", [1, 1024])
def test_errors_are_measured_against_the_fp64_reference(numel: int) -> None:
    reference = {"tensor": torch.ones(numel)}
    snapshots, spec = _trio(_offset(reference, UNIT), _offset(reference, UNIT), reference)
    snapshots["reference"]["loss_sum"] = torch.tensor(4.0, dtype=torch.float64)
    snapshots["hf"]["loss_sum"] = torch.tensor(5.0)
    snapshots["candidate"]["loss_sum"] = torch.tensor(3.0)

    receipt = _compare(snapshots, spec)
    assert receipt["loss"] == {
        "reference": 4.0,
        "hf": 5.0,
        "candidate": 3.0,
        "hf_relative_error": 0.25,
        "relative_error": 0.25,
        "bound": 1.0,
    }
    raw = receipt["raw_gradients"]
    assert raw["collection_relative_l2"] == {
        "hf": UNIT,
        "candidate": UNIT,
        "ratio": 1.0,
        "bound": 1.5,
    }
    assert raw["worst_named"] == {
        "name": "tensor",
        "candidate_rms": UNIT,
        "hf_rms": UNIT,
        "ratio": 1.0,
        "allowed_rms": 4 * UNIT + FLOOR,
        "bound": 4.0,
        "floor": FLOOR,
    }


def test_a_candidate_more_accurate_than_hf_passes() -> None:
    reference = {"left": torch.ones(4), "right": torch.full((4,), 2.0)}
    snapshots, spec = _trio(reference, _offset(reference, UNIT), reference)

    raw = _compare(snapshots, spec)["raw_gradients"]
    assert raw["collection_relative_l2"]["candidate"] == 0.0
    assert raw["collection_relative_l2"]["ratio"] == 0.0


@pytest.mark.parametrize(
    ("gate", "message"),
    [
        ("loss", "loss relative error"),
        ("named", "named gradient"),
        ("collection", "collection relative L2"),
    ],
)
def test_numerical_boundaries_are_inclusive(gate: str, message: str) -> None:
    reference = {"subject": torch.ones(1)}
    hf = _offset(reference, UNIT)
    candidate = _offset(reference, UNIT)
    bounds = {}
    candidate_loss = 1.0
    if gate == "loss":
        bounds["loss_bound"] = UNIT
        candidate_loss += UNIT
    elif gate == "named":
        bounds["collection_bound"] = 8.0
        candidate["subject"] = torch.tensor([1.0 + 4 * UNIT + FLOOR])
    else:
        candidate["subject"] = torch.tensor([1.0 + 1.5 * UNIT])

    snapshots, spec = _trio(candidate, hf, reference, candidate_loss=candidate_loss, **bounds)
    _compare(snapshots, spec)
    if gate == "loss":
        snapshots["candidate"]["loss_sum"] += 2**-20
    else:
        snapshots["candidate"]["raw_gradients"]["subject"] += 2**-20
    with pytest.raises(BackwardMismatchError, match=message):
        _compare(snapshots, spec)


@pytest.mark.parametrize("excess", [0.0, 2**-30])
def test_tensor_floor_bounds_a_zero_hf_error(excess: float) -> None:
    reference = {"anchor": torch.ones(1), "subject": torch.zeros(1)}
    candidate = {"anchor": torch.ones(1), "subject": torch.tensor([FLOOR + excess])}
    snapshots, spec = _trio(candidate, reference, reference, collection_bound=1.0)
    # HF's anchor error keeps the collection gate open; on subject HF matches exactly.
    snapshots["hf"]["raw_gradients"]["anchor"] += UNIT

    if excess:
        with pytest.raises(BackwardMismatchError, match="named gradient error RMS"):
            _compare(snapshots, spec)
    else:
        worst = _compare(snapshots, spec)["raw_gradients"]["worst_named"]
        assert worst == {
            "name": "subject",
            "candidate_rms": FLOOR,
            "hf_rms": 0.0,
            "ratio": None,
            "allowed_rms": FLOOR,
            "bound": 4.0,
            "floor": FLOOR,
        }


def test_max_absolute_error_is_diagnostic_only() -> None:
    reference = {"anchor": torch.ones(1), "victim": torch.zeros(16)}
    hf = _offset(reference, UNIT)
    candidate = {name: tensor.clone() for name, tensor in reference.items()}
    candidate["victim"][7] = 4 * UNIT

    receipt = _compare(*_trio(candidate, hf, reference))
    raw = receipt["raw_gradients"]
    assert raw["worst_named"]["ratio"] == 1.0
    assert raw["maximum_absolute"] == {"name": "victim", "flat_index": 7, "error": 4 * UNIT}


def test_collection_gate_catches_broad_drift_within_named_limits() -> None:
    reference = {"left": torch.ones(32), "right": torch.ones(32)}
    snapshots, spec = _trio(_offset(reference, 2 * UNIT), _offset(reference, UNIT), reference)

    with pytest.raises(BackwardMismatchError) as error:
        _compare(snapshots, spec)
    assert "collection relative L2" in str(error.value)
    assert "loss relative error" not in str(error.value)
    assert "named gradient" not in str(error.value)


def test_named_gate_catches_one_drifting_tensor_within_the_collection_limit() -> None:
    reference = {"stable": torch.ones(1024), "drifting": torch.ones(1)}
    hf = _offset(reference, UNIT)
    candidate = {
        "stable": reference["stable"].clone(),
        "drifting": reference["drifting"] + 8 * UNIT,
    }
    snapshots, spec = _trio(candidate, hf, reference)

    with pytest.raises(BackwardMismatchError, match=r"named gradient .*'drifting'") as error:
        _compare(snapshots, spec)
    assert "collection relative L2" not in str(error.value)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("candidate_loss", "loss_sum must be finite"),
        ("hf_loss", "loss_sum must be finite"),
        ("reference_loss", "loss_sum must be finite"),
        ("candidate_gradient", "must be finite"),
        ("hf_gradient", "must be finite"),
        ("reference_gradient", "must be finite"),
        ("zero_reference_loss", "reference loss must be positive"),
        ("zero_reference", "collection norm must be nonzero"),
    ],
)
def test_nonfinite_and_zero_reference_inputs_fail_closed(case: str, message: str) -> None:
    gradients = {"tensor": torch.ones(2)}
    snapshots, spec = _trio(gradients, gradients, gradients)
    role, _, field = case.removeprefix("zero_").partition("_")
    if case == "zero_reference_loss":
        snapshots["reference"]["loss_sum"].zero_()
    elif case == "zero_reference":
        for snapshot in snapshots.values():
            snapshot["raw_gradients"]["tensor"].zero_()
    elif field == "loss":
        snapshots[role]["loss_sum"].fill_(float("nan"))
    else:
        snapshots[role]["raw_gradients"]["tensor"][0] = float("inf")

    with pytest.raises(InvalidBackwardSnapshotError, match=message):
        _compare(snapshots, spec)


@pytest.mark.parametrize(
    ("role", "case", "message"),
    [
        ("candidate", "order", "names/order"),
        ("candidate", "shape", "shape"),
        ("candidate", "dtype", "must use float32"),
        ("hf", "dtype", "must use float32"),
        ("reference", "dtype", "must use float64"),
        ("reference", "loss_dtype", "dense CPU float64 scalar"),
        ("hf", "loss_dtype", "dense CPU float32 scalar"),
        ("reference", "schema_dtype", "must use float64"),
        ("candidate", "empty", "shape"),
        ("candidate", "sparse_loss", "dense"),
        ("candidate", "zero_targets", "must be positive"),
        ("reference", "missing_key", "keys mismatch"),
        ("candidate", "extra_key", "keys mismatch"),
        ("reference", "unknown_role", "role must be"),
    ],
)
def test_malformed_snapshot_schema_fails_closed(role: str, case: str, message: str) -> None:
    gradients = {"left": torch.ones(1), "right": torch.ones(1)}
    snapshots, spec = _trio(gradients, gradients, gradients)
    snapshot = snapshots[role]
    flipped = torch.float32 if role == "reference" else torch.float64
    if case == "order":
        snapshot["raw_gradients"] = dict(reversed(snapshot["raw_gradients"].items()))
    elif case == "shape":
        snapshot["raw_gradients"]["left"] = torch.ones(2)
    elif case == "dtype":
        snapshot["raw_gradients"]["left"] = torch.ones(1, dtype=flipped)
    elif case == "loss_dtype":
        snapshot["loss_sum"] = snapshot["loss_sum"].to(flipped)
    elif case == "schema_dtype":
        snapshot["parameter_schema"] = spec.expected_parameter_schema
    elif case == "empty":
        snapshot["raw_gradients"]["left"] = torch.empty(0)
    elif case == "sparse_loss":
        snapshot["loss_sum"] = torch.ones((), dtype=torch.float32).to_sparse()
    elif case == "zero_targets":
        snapshot["num_target_tokens"] = 0
    elif case == "missing_key":
        del snapshot["base_gradients_absent"]
    elif case == "extra_key":
        snapshot["unexpected"] = True
    else:
        snapshot["role"] = "teacher"

    with pytest.raises(InvalidBackwardSnapshotError, match=message):
        _compare(snapshots, spec)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("profile", "external profile"),
        ("reference_profile", "reference.profile"),
        ("swapped_roles", "reference.role is 'hf'"),
        ("targets", "num_target_tokens"),
        ("trainable", "trainable_names"),
        ("base_gradient", "base-parameter gradient"),
        ("initial", "candidate initial LoRA tensor differs"),
        ("reference_initial", "reference initial LoRA tensor differs"),
        ("zero_initial", "initial_lora tensors must be nonzero"),
    ],
)
def test_exact_snapshot_mismatches_fail(case: str, message: str) -> None:
    gradients = {"tensor": torch.ones(1)}
    snapshots, spec = _trio(gradients, gradients, gradients)
    candidate = snapshots["candidate"]
    if case == "profile":
        candidate["profile"]["case_id"] = "wrong-case"
    elif case == "reference_profile":
        snapshots["reference"]["profile"]["dtype"] = "float32"
    elif case == "swapped_roles":
        snapshots["reference"], snapshots["hf"] = snapshots["hf"], snapshots["reference"]
    elif case == "targets":
        candidate["num_target_tokens"] += 1
    elif case == "trainable":
        candidate["trainable_names"] = ()
    elif case == "base_gradient":
        candidate["base_gradients_absent"] = False
    elif case == "initial":
        candidate["initial_lora"]["tensor"] += 1
    elif case == "reference_initial":
        snapshots["reference"]["initial_lora"]["tensor"] += 2**-40
    else:
        for snapshot in snapshots.values():
            snapshot["initial_lora"]["tensor"].zero_()

    with pytest.raises(BackwardMismatchError, match=message):
        _compare(snapshots, spec)


def test_initial_lora_tensor_may_be_partially_zero() -> None:
    gradients = {"tensor": torch.ones(2)}
    snapshots, spec = _trio(gradients, gradients, gradients)
    for snapshot in snapshots.values():
        initial = snapshot["initial_lora"]["tensor"]
        snapshot["initial_lora"]["tensor"] = torch.tensor([0.0, 1.0], dtype=initial.dtype)

    assert _compare(snapshots, spec)["verdict"] == "passed"


def test_snapshot_payload_round_trips_through_weights_only_load(tmp_path: Path) -> None:
    gradients = {"tensor": torch.ones(1)}
    snapshots, spec = _trio(gradients, gradients, gradients)
    path = tmp_path / "snapshots.pt"
    torch.save(snapshots, path)

    receipt = _compare(torch.load(path, map_location="cpu", weights_only=True), spec)

    assert receipt["verdict"] == "passed"
    decoded = json.loads(json.dumps(receipt))
    assert decoded["profile"] == {**_profile_payload(), "capability": [12, 0]}
    assert decoded["reference_profile"] == {
        **_profile_payload(),
        "dtype": "float64",
        "capability": [12, 0],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss_relative_bound", float("nan")),
        ("collection_error_ratio_bound", 0.0),
        ("tensor_error_ratio_bound", float("inf")),
        ("tensor_error_floor", -1.0),
        ("tensor_error_floor", 0.0),
        ("collection_error_ratio_bound", True),
        ("expected_parameter_schema", ()),
        ("expected_candidate_profile", "mixed[bfloat16,float32]"),
        ("expected_profile", replace(PROFILE, dtype="mixed[bfloat16,float32]")),
        ("expected_profile", REFERENCE_PROFILE),
        ("expected_reference_profile", PROFILE),
        ("expected_reference_profile", None),
    ],
)
def test_invalid_comparison_spec_is_rejected(field: str, value: object) -> None:
    spec = _spec({"tensor": torch.ones(1)})
    with pytest.raises((TypeError, ValueError)):
        replace(spec, **{field: value})


def test_qwen3_5_oracle_maps_canonical_adapters_onto_grouped_a_and_back(tmp_path: Path) -> None:
    hf_config = load_hf_config(text_only_config(TINY_QWEN3_5_CONFIG), tmp_path)
    with torch.device("meta"):
        checkpoint = transformers.Qwen3_5ForCausalLM(hf_config).state_dict()
    peft_state = random_peft_state(
        {name: torch.empty(tensor.shape) for name, tensor in checkpoint.items()},
        prefix=TEXT_ONLY_PREFIX,
        targets=("attention", "linear_attention", "mlp"),
        rank=2,
        generator=torch.Generator().manual_seed(0),
    )
    canonical = {
        key.removeprefix("base_model.model.").removesuffix(".weight"): tensor
        for key, tensor in peft_state.items()
    }
    layout = Qwen3_5CanonicalLoRALayout(Qwen3_5Config.from_hf(hf_config).text)

    engine = layout.to_engine(canonical)

    # Each checkpoint module's A lands on its group; in_proj_qkv's B splits into q, k, v.
    in_proj = "model.layers.0.linear_attn"
    assert torch.equal(
        engine[f"{in_proj}.in_proj.qkv.lora_A"], canonical[f"{in_proj}.in_proj_qkv.lora_A"]
    )
    qkv_b = canonical[f"{in_proj}.in_proj_qkv.lora_B"]
    assert torch.equal(
        torch.cat([engine[f"{in_proj}.in_proj.{name}.lora_B"] for name in "qkv"]), qkv_b
    )
    attention = "model.layers.1.self_attn"
    assert torch.equal(
        engine[f"{attention}.qkv_gate_proj.q_gate.lora_A"],
        canonical[f"{attention}.q_proj.lora_A"],
    )
    # q_proj's rows are each head's query, then its gate.
    q_b = canonical[f"{attention}.q_proj.lora_B"].unflatten(0, (-1, 2, 256))
    assert torch.equal(engine[f"{attention}.qkv_gate_proj.q.lora_B"], q_b[:, 0].flatten(0, 1))
    assert torch.equal(engine[f"{attention}.qkv_gate_proj.gate.lora_B"], q_b[:, 1].flatten(0, 1))

    # Gradients come back the same way: the gate's re-interleaves with the query's.
    names = tuple(reversed(canonical))
    restored = layout.to_canonical(engine, names)
    assert tuple(restored) == names
    for name in names:
        assert torch.equal(restored[name], canonical[name]), name


def test_oracle_names_qwen3_and_qwen3_5_modules_canonically() -> None:
    from tests.gpu.training.lora_backward_hf import runner

    image_text = "base_model.model.model.language_model.layers.3.linear_attn.in_proj_qkv"
    assert runner._canonical_hf_name(f"{image_text}.lora_A.default.weight") == (
        "model.layers.3.linear_attn.in_proj_qkv.lora_A"
    )
    assert runner._canonical_hf_name(
        "base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weight"
    ) == ("model.layers.2.self_attn.q_proj.lora_B")
    assert runner._candidate_targets(("q_proj", "in_proj_qkv", "out_proj", "down_proj")) == (
        "attention",
        "linear_attention",
        "mlp",
    )


def _mixed_candidate(
    gradients: dict[str, torch.Tensor],
) -> tuple[Snapshots, BackwardComparisonSpec]:
    snapshots, spec = _trio(gradients, gradients, gradients)
    candidate = snapshots["candidate"]
    candidate["profile"]["dtype"] = "mixed[bfloat16,float32]"
    return snapshots, replace(
        spec, expected_candidate_profile=ProfileFingerprint(**candidate["profile"])
    )


def test_mixed_candidate_profile_requires_explicit_contract() -> None:
    snapshots, spec = _mixed_candidate({"tensor": torch.ones(2)})
    with pytest.raises(BackwardMismatchError, match="candidate.profile"):
        _compare(snapshots, replace(spec, expected_candidate_profile=None))
    receipt = _compare(snapshots, spec)
    assert receipt["profile"] == snapshots["hf"]["profile"]
    assert receipt["candidate_profile"] == snapshots["candidate"]["profile"]
    assert receipt["reference_profile"] == snapshots["reference"]["profile"]
    assert receipt["verdict"] == "passed"


@pytest.mark.parametrize("role", ["reference", "hf", "candidate"])
@pytest.mark.parametrize("field", ["dtype", "revision", "training_attention"])
def test_role_profiles_reject_drift_in_any_role(role: str, field: str) -> None:
    snapshots, spec = _mixed_candidate({"tensor": torch.ones(2)})
    profile = snapshots[role]["profile"]
    other = "hf" if role == "candidate" else "candidate"
    profile[field] = snapshots[other]["profile"]["dtype"] if field == "dtype" else "unexpected"
    with pytest.raises(BackwardMismatchError, match=f"{role}.profile"):
        _compare(snapshots, spec)


@pytest.mark.parametrize(
    ("case", "message"),
    [("gradient", "named gradient"), ("initial", "initial LoRA"), ("storage", "float32")],
)
def test_mixed_profile_preserves_gradient_and_initial_state_gates(case: str, message: str) -> None:
    snapshots, spec = _mixed_candidate({"tensor": torch.ones(2)})
    candidate = snapshots["candidate"]
    if case == "gradient":
        candidate["raw_gradients"]["tensor"] += 1
    elif case == "initial":
        candidate["initial_lora"]["tensor"] += 1
    else:
        candidate["raw_gradients"]["tensor"] = candidate["raw_gradients"]["tensor"].bfloat16()
    with pytest.raises((BackwardMismatchError, InvalidBackwardSnapshotError), match=message):
        _compare(snapshots, spec)
