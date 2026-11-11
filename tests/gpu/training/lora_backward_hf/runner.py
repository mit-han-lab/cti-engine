from __future__ import annotations

import argparse
import math
import os
import platform
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from tests.support.lora_backward import (
    BackwardProfilePayload,
    BackwardSnapshot,
    ParameterSchemaEntry,
)
from tests.support.training import forward_backward

if TYPE_CHECKING:
    from cti.training import TrainingBatch
    from tests.support.training_runtime import PaddedTrainingOracleRuntime

_CASE_KEYS = {
    "case_id",
    "model",
    "revision",
    "seed",
    "token_ids",
    "lengths",
    "pad_token_id",
    "lora",
    "initial_lora",
}
_LORA_KEYS = {"rank", "alpha", "targets"}

# Qwen3 and Qwen3.5 text layers; an image-text checkpoint nests them in language_model.
_HF_LORA_PATTERN = re.compile(
    r"^base_model\.model\.model\.(?:language_model\.)?layers\.(\d+)\."
    r"(self_attn|linear_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|"
    r"gate_proj|up_proj|down_proj)\.lora_([AB])\.default\.weight$"
)
_EDGE_LORA_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn|mlp)\."
    r"(qkv_proj|o_proj|gate_up_proj|down_proj)\."
    r"(q|k|v|full|gate|up)\.lora_([AB])$"
)
_EDGE_PROJECTION_NAMES = {
    ("qkv_proj", "q"): "q_proj",
    ("qkv_proj", "k"): "k_proj",
    ("qkv_proj", "v"): "v_proj",
    ("o_proj", "full"): "o_proj",
    ("gate_up_proj", "gate"): "gate_proj",
    ("gate_up_proj", "up"): "up_proj",
    ("down_proj", "full"): "down_proj",
}
_EDGE_TARGETS = {
    "q_proj": "attention",
    "k_proj": "attention",
    "v_proj": "attention",
    "o_proj": "attention",
    "in_proj_qkv": "linear_attention",
    "in_proj_z": "linear_attention",
    "in_proj_b": "linear_attention",
    "in_proj_a": "linear_attention",
    "out_proj": "linear_attention",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}


@dataclass(frozen=True)
class _BackwardCase:
    case_id: str
    model: str
    revision: str
    seed: int
    token_ids: torch.Tensor
    lengths: torch.Tensor
    pad_token_id: int
    lora_rank: int
    lora_alpha: float
    lora_targets: tuple[str, ...]
    initial_lora: dict[str, torch.Tensor]


NamedTensors = tuple[tuple[str, torch.Tensor], ...]
Canonicalizer = Callable[[str], str | None]


class _SDPACallObserver(TorchFunctionMode):
    def __init__(self) -> None:
        self.calls = 0

    def __torch_function__(
        self,
        func: Callable[..., object],
        _types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func is F.scaled_dot_product_attention:
            self.calls += 1
        return func(*args, **(kwargs or {}))


def _mapping(
    value: object,
    label: str,
    keys: set[str] | None = None,
) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if keys is not None and set(value) != keys:
        actual = set(value)
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(keys - actual, key=repr)}, "
            f"extra={sorted(actual - keys, key=repr)}"
        )
    return value


def _load_case(path: Path) -> _BackwardCase:
    payload = _mapping(torch.load(path, map_location="cpu", weights_only=True), "case", _CASE_KEYS)

    strings = {}
    for name in ("case_id", "model", "revision"):
        value = payload[name]
        if not isinstance(value, str) or not value:
            raise TypeError(f"case.{name} must be a non-empty string")
        strings[name] = value

    seed = payload["seed"]
    pad_token_id = payload["pad_token_id"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("case.seed must be an integer")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise TypeError("case.pad_token_id must be an integer")

    token_ids = payload["token_ids"]
    lengths = payload["lengths"]
    if not isinstance(token_ids, torch.Tensor) or not isinstance(lengths, torch.Tensor):
        raise TypeError("case token_ids and lengths must be tensors")
    if (
        token_ids.device.type != "cpu"
        or lengths.device.type != "cpu"
        or token_ids.layout != torch.strided
        or lengths.layout != torch.strided
    ):
        raise ValueError("case token tensors must be dense CPU tensors")
    if token_ids.dtype != torch.int64 or lengths.dtype != torch.int64:
        raise TypeError("case token tensors must use int64")
    if token_ids.ndim != 2 or lengths.ndim != 1:
        raise ValueError("case token_ids must be a matrix and lengths must be a vector")
    batch_size, padded_length = token_ids.shape
    if batch_size == 0 or padded_length < 2 or lengths.shape != (batch_size,):
        raise ValueError("case contains an invalid padded batch shape")
    if not bool(torch.all((lengths >= 2) & (lengths <= padded_length))):
        raise ValueError("case lengths must be between 2 and the padded length")
    positions = torch.arange(padded_length).unsqueeze(0)
    if not bool(torch.all(token_ids[positions >= lengths.unsqueeze(1)] == pad_token_id)):
        raise ValueError("case token_ids must be strictly right-padded")

    lora = _mapping(payload["lora"], "case.lora", _LORA_KEYS)
    rank = lora["rank"]
    alpha = lora["alpha"]
    targets = lora["targets"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("case.lora.rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("case.lora.alpha must be a real number")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("case.lora.alpha must be finite and positive")
    if (
        not isinstance(targets, tuple)
        or not targets
        or not all(isinstance(target, str) and target for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("case.lora.targets must be a non-empty tuple of unique strings")

    initial = _mapping(payload["initial_lora"], "case.initial_lora")
    if not initial:
        raise ValueError("case.initial_lora must not be empty")
    initial_lora = {}
    for name, tensor in initial.items():
        if not isinstance(name, str) or not name:
            raise TypeError("case.initial_lora contains an invalid name")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"case.initial_lora.{name} must be a tensor")
        if (
            tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.dtype != torch.float32
            or tensor.numel() == 0
            or not bool(torch.all(torch.isfinite(tensor)))
            or torch.count_nonzero(tensor).item() == 0
        ):
            raise ValueError(f"case.initial_lora.{name} must be finite, nonzero, dense CPU FP32")
        initial_lora[name] = tensor

    return _BackwardCase(
        case_id=strings["case_id"],
        model=strings["model"],
        revision=strings["revision"],
        seed=seed,
        token_ids=token_ids,
        lengths=lengths,
        pad_token_id=pad_token_id,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_targets=targets,
        initial_lora=initial_lora,
    )


def _configure_determinism(seed: int) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before deterministic runner setup")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA startup")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _restore_case_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.initial_seed() != seed or torch.cuda.initial_seed() != seed:
        raise RuntimeError("failed to restore the fixed case seed before forward")


def _canonical_hf_name(name: str) -> str | None:
    match = _HF_LORA_PATTERN.fullmatch(name)
    if match is None:
        return None
    layer, parent, projection, factor = match.groups()
    return f"model.layers.{layer}.{parent}.{projection}.lora_{factor}"


def _canonical_candidate_name(name: str) -> str | None:
    match = _EDGE_LORA_PATTERN.fullmatch(name)
    if match is None:
        return None
    layer, parent, fused_projection, projection_slice, factor = match.groups()
    try:
        projection = _EDGE_PROJECTION_NAMES[(fused_projection, projection_slice)]
    except KeyError as error:
        raise ValueError(f"unsupported candidate LoRA projection: {name}") from error
    return f"model.layers.{layer}.{parent}.{projection}.lora_{factor}"


def _discover_lora(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
    canonicalize: Canonicalizer,
    provider: str,
) -> tuple[NamedTensors, NamedTensors, tuple[str, ...]]:
    adapters = []
    base = []
    trainable_names = []
    canonical_names = set()
    for provider_name, parameter in named_parameters:
        canonical_name = canonicalize(provider_name)
        if canonical_name is None:
            if ".lora_" in provider_name:
                raise ValueError(f"unrecognized {provider} LoRA parameter: {provider_name}")
            base.append((provider_name, parameter))
            if parameter.requires_grad:
                trainable_names.append(provider_name)
            continue
        if canonical_name in canonical_names:
            raise ValueError(f"duplicate canonical {provider} LoRA parameter: {canonical_name}")
        canonical_names.add(canonical_name)
        adapters.append((canonical_name, parameter))
        if parameter.requires_grad:
            trainable_names.append(canonical_name)
    if not adapters:
        raise ValueError(f"{provider} exposed no canonical LoRA parameters")
    return tuple(adapters), tuple(base), tuple(trainable_names)


def _candidate_targets(targets: tuple[str, ...]) -> tuple[str, ...]:
    result = []
    for target in targets:
        try:
            category = _EDGE_TARGETS[target]
        except KeyError as error:
            raise ValueError(
                f"candidate does not recognize canonical LoRA target: {target}"
            ) from error
        if category not in result:
            result.append(category)
    return tuple(result)


def _load_initial_lora(
    named_lora: NamedTensors,
    initial_lora: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Copy the FP32 fixture into ``dtype`` parameters; FP32 to FP64 is exact."""

    live_names = tuple(name for name, _ in named_lora)
    if set(live_names) != set(initial_lora):
        raise ValueError(
            "initial LoRA names differ from live discovery: "
            f"missing={sorted(set(live_names) - set(initial_lora))}, "
            f"extra={sorted(set(initial_lora) - set(live_names))}"
        )
    copies = []
    for name, parameter in named_lora:
        source = initial_lora[name]
        if source.shape != parameter.shape:
            raise ValueError(
                f"initial LoRA shape mismatch for {name}: "
                f"{tuple(source.shape)} != {tuple(parameter.shape)}"
            )
        if parameter.dtype != dtype:
            raise ValueError(
                f"initial LoRA dtype mismatch for {name}: {parameter.dtype} != {dtype}"
            )
        copies.append((parameter, source))
    with torch.no_grad():
        for parameter, source in copies:
            parameter.copy_(source.to(device=parameter.device, dtype=dtype))
    return _clone_named_tensors(named_lora)


def _clone_named_tensors(named_tensors: NamedTensors) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().to(device="cpu", copy=True) for name, tensor in named_tensors}


def _clone_raw_gradients(named_lora: NamedTensors, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    gradients = []
    for name, parameter in named_lora:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"raw LoRA gradient is missing for {name}")
        if (
            gradient.layout != torch.strided
            or gradient.shape != parameter.shape
            or gradient.dtype != dtype
        ):
            raise RuntimeError(f"raw LoRA gradient has invalid layout, shape, or dtype for {name}")
        if not bool(torch.all(torch.isfinite(gradient))):
            raise RuntimeError(f"raw LoRA gradient is nonfinite for {name}")
        gradients.append((name, gradient))
    return _clone_named_tensors(tuple(gradients))


def _parameter_schema(named_lora: NamedTensors) -> tuple[ParameterSchemaEntry, ...]:
    return tuple(
        (name, tuple(parameter.shape), str(parameter.dtype).removeprefix("torch."))
        for name, parameter in named_lora
    )


def _common_dtype(named_lora: NamedTensors, named_base: NamedTensors) -> str:
    dtypes = {tensor.dtype for _, tensor in (*named_lora, *named_base)}
    if len(dtypes) == 1:
        return str(next(iter(dtypes))).removeprefix("torch.")
    return "mixed[" + ",".join(sorted(str(dtype).removeprefix("torch.") for dtype in dtypes)) + "]"


def _observed_profile(
    case: _BackwardCase,
    *,
    dtype: str,
    training_attention: str,
) -> BackwardProfilePayload:
    cuda_version = torch.version.cuda
    return {
        "model": case.model,
        "revision": case.revision,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": cuda_version if cuda_version is not None else "unavailable",
        "transformers_version": version("transformers"),
        "peft_version": version("peft"),
        "device": torch.cuda.get_device_name(),
        "capability": tuple(torch.cuda.get_device_capability()),
        "dtype": dtype,
        "case_id": case.case_id,
        "training_attention": training_attention,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def _snapshot(
    *,
    role: str,
    profile: BackwardProfilePayload,
    loss_sum: torch.Tensor,
    num_target_tokens: int,
    named_lora: NamedTensors,
    initial_lora: dict[str, torch.Tensor],
    raw_gradients: dict[str, torch.Tensor],
    trainable_names: tuple[str, ...],
    named_base: NamedTensors,
) -> BackwardSnapshot:
    if role not in ("reference", "hf", "candidate"):
        raise ValueError(f"unsupported snapshot role: {role}")
    return {
        "role": role,
        "profile": profile,
        "loss_sum": loss_sum.detach().to(device="cpu", copy=True),
        "num_target_tokens": num_target_tokens,
        "parameter_schema": _parameter_schema(named_lora),
        "initial_lora": initial_lora,
        "raw_gradients": raw_gradients,
        "trainable_names": trainable_names,
        "base_gradients_absent": all(tensor.grad is None for _, tensor in named_base),
    }


def _run_hf(
    model_path: Path, case: _BackwardCase, *, role: str, dtype: torch.dtype
) -> BackwardSnapshot:
    """HF PEFT in ``dtype``: FP32 for the ``hf`` role, FP64 for the ``reference`` role."""

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=case.lora_rank,
            lora_alpha=case.lora_alpha,
            target_modules=list(case.lora_targets),
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=False,
            use_rslora=False,
            use_dora=False,
        ),
        autocast_adapter_dtype=True,
    )
    del base
    model.train()
    try:
        named_lora, named_base, trainable_names = _discover_lora(
            model.named_parameters(), _canonical_hf_name, "Hugging Face"
        )
        initial_lora = _load_initial_lora(named_lora, case.initial_lora, dtype)

        input_ids = case.token_ids[:, :-1].to("cuda")
        target_ids = case.token_ids[:, 1:].to("cuda")
        positions = torch.arange(target_ids.shape[1]).unsqueeze(0)
        target_mask = (positions < (case.lengths - 1).unsqueeze(1)).to("cuda")
        implementation = getattr(model.config, "_attn_implementation", None)
        if implementation != "sdpa":
            raise RuntimeError(f"HF training did not select SDPA: {implementation!r}")
        _restore_case_seed(case.seed)
        observer = _SDPACallObserver()
        with observer:
            output = model(input_ids=input_ids, use_cache=False, return_dict=True)
        # Qwen3.5's Gated DeltaNet layers have no self_attn and call no SDPA.
        expected_sdpa_calls = sum(
            hasattr(layer, "self_attn") for layer in model.get_base_model().model.layers
        )
        if observer.calls != expected_sdpa_calls:
            raise RuntimeError(
                f"HF forward made {observer.calls} SDPA calls, expected {expected_sdpa_calls}"
            )
        loss = F.cross_entropy(
            output.logits[target_mask],
            target_ids[target_mask],
            reduction="sum",
        )
        loss.backward()
        raw_gradients = _clone_raw_gradients(named_lora, dtype)
        torch.cuda.synchronize()
        return _snapshot(
            role=role,
            profile=_observed_profile(
                case,
                dtype=_common_dtype(named_lora, named_base),
                training_attention="torch-sdpa",
            ),
            loss_sum=loss,
            num_target_tokens=int(target_mask.sum().item()),
            named_lora=named_lora,
            initial_lora=initial_lora,
            raw_gradients=raw_gradients,
            trainable_names=trainable_names,
            named_base=named_base,
        )
    finally:
        del model


def _candidate_training_attention(model: object, sdpa_calls: int) -> str:
    from cti.layers.attention import AttentionLayer

    attention_layers = tuple(model.iter_attention_layers())
    if not attention_layers:
        raise RuntimeError("candidate model has no attention layers")
    for attention in attention_layers:
        forward_training = type(attention).forward_training
        if (
            not isinstance(attention, AttentionLayer)
            or forward_training is not AttentionLayer.forward_training
        ):
            raise RuntimeError(
                "candidate training attention is not the canonical torch SDPA path: "
                f"{forward_training.__module__}.{forward_training.__qualname__}"
            )
    if sdpa_calls != len(attention_layers):
        raise RuntimeError(
            f"candidate forward made {sdpa_calls} SDPA calls, expected {len(attention_layers)}"
        )
    return "torch-sdpa"


def _candidate_runtime(
    model_path: Path,
    case: _BackwardCase,
    dtype: torch.dtype,
) -> PaddedTrainingOracleRuntime:
    from cti.config import (
        AdamWConfig,
        BatchLayout,
        LoRAConfig,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from tests.support.training_runtime import PaddedTrainingOracleRuntime

    return PaddedTrainingOracleRuntime(
        str(model_path),
        dtype=dtype,
        lora_config=LoRAConfig(
            targets=_candidate_targets(case.lora_targets),
            rank=case.lora_rank,
            alpha=case.lora_alpha,
        ),
        training_config=TrainingConfig(
            TrainablePlan.all_declared(),
            AdamWConfig(),
            TrainingBatching(
                BatchLayout.PADDED,
                StaticBatching(
                    microbatch_size=case.token_ids.shape[0],
                    max_tokens_per_sample=case.token_ids.shape[1],
                ),
            ),
        ),
    )


def _case_batch(case: _BackwardCase) -> TrainingBatch:
    from cti.training import SFT, PaddedTokens, TrainingBatch

    target_positions = torch.arange(case.token_ids.shape[1] - 1).unsqueeze(0)
    return TrainingBatch(
        PaddedTokens(case.token_ids, case.lengths, case.pad_token_id),
        target_positions < (case.lengths - 1).unsqueeze(1),
        SFT(),
    )


def _run_candidate(model_path: Path, case: _BackwardCase) -> BackwardSnapshot:
    from cti.config import TrainablePlan
    from cti.training._parameters import resolve_for_model
    from tests.support.training import (
        AdaptedTrainables,
        clone_main_gradients,
        synchronize_optimizer_parameters,
    )

    runtime = _candidate_runtime(model_path, case, torch.float32)
    trainer = None
    try:
        model = runtime.model
        adapters = runtime.adapters
        raw_named_adapters = adapters.named_parameters()
        named_lora, unexpected, trainable_names = _discover_lora(
            raw_named_adapters, _canonical_candidate_name, "candidate"
        )
        if unexpected:
            raise ValueError(f"candidate adapter set exposed non-LoRA tensors: {unexpected}")
        named_base = tuple(model.state_dict().items())
        trainable_names += tuple(name for name, tensor in named_base if tensor.requires_grad)
        initial_lora = _load_initial_lora(named_lora, case.initial_lora, torch.float32)
        trainable_parameters = resolve_for_model(
            TrainablePlan(tuple(name for name, _ in raw_named_adapters)),
            AdaptedTrainables(model, adapters),
        )
        trainer = runtime.runner
        synchronize_optimizer_parameters(trainer, raw_named_adapters)
        assert trainer.trainable_parameters.selected_names == trainable_parameters.selected_names
        batch = _case_batch(case)
        _restore_case_seed(case.seed)
        observer = _SDPACallObserver()
        with observer:
            result = forward_backward(trainer, batch)
        master_gradients = clone_main_gradients(trainer, raw_named_adapters)
        raw_gradients = dict(
            zip(
                (name for name, _ in named_lora),
                (gradient.cpu() for gradient in master_gradients),
                strict=True,
            )
        )
        torch.cuda.synchronize()
        snapshot = _snapshot(
            role="candidate",
            profile=_observed_profile(
                case,
                dtype=_common_dtype(named_lora, named_base),
                training_attention=_candidate_training_attention(model, observer.calls),
            ),
            loss_sum=result.loss_sum,
            num_target_tokens=result.normalization_count,
            named_lora=named_lora,
            initial_lora=initial_lora,
            raw_gradients=raw_gradients,
            trainable_names=trainable_names,
            named_base=named_base,
        )
    finally:
        try:
            if trainer is not None:
                trainer.discard()
        finally:
            runtime.close()
    return snapshot


def _run_qwen3_5_candidate(model_path: Path, case: _BackwardCase) -> BackwardSnapshot:
    """The engine's Qwen3.5 backward, compared in the canonical adapter layout.

    It runs in BF16, the only dtype of the Gated DeltaNet training operation, so the
    case's initial adapter must hold BF16 values for the initial-state check to pass
    exactly. The snapshot records the FP32 master gradients, accumulated in FP32
    across uses and microbatches.
    """

    from transformers import AutoConfig

    from cti.models.qwen3_5.config import Qwen3_5Config
    from tests.support.qwen3_5_lora import Qwen3_5CanonicalLoRALayout
    from tests.support.training import clone_main_gradients, synchronize_optimizer_parameters

    runtime = _candidate_runtime(model_path, case, torch.bfloat16)
    trainer = None
    try:
        model = runtime.model
        adapters = runtime.adapters
        layout = Qwen3_5CanonicalLoRALayout(
            Qwen3_5Config.from_hf(AutoConfig.from_pretrained(model_path)).text
        )
        names = tuple(case.initial_lora)
        device = next(iter(tuple(t for _, t in adapters.named_parameters()))).device
        adapters.load_state_dict(
            {
                name: tensor.to(device=device, dtype=torch.bfloat16)
                for name, tensor in layout.to_engine(case.initial_lora).items()
            }
        )
        trainer = runtime.runner
        # The runner copied its FP32 masters from the adapter it was built with.
        named_adapters = tuple(adapters.named_parameters())
        synchronize_optimizer_parameters(trainer, named_adapters)
        initial_lora = layout.to_canonical(
            {name: tensor.float().cpu() for name, tensor in adapters.state_dict().items()},
            names,
        )
        named_lora = tuple(initial_lora.items())
        named_base = tuple(model.state_dict().items())
        trainable_names = names + tuple(name for name, tensor in named_base if tensor.requires_grad)
        _restore_case_seed(case.seed)
        observer = _SDPACallObserver()
        with observer:
            result = forward_backward(trainer, _case_batch(case))
        master_gradients = clone_main_gradients(trainer, named_adapters)
        raw_gradients = layout.to_canonical(
            {
                name: gradient.cpu()
                for (name, _), gradient in zip(named_adapters, master_gradients, strict=True)
            },
            names,
        )
        torch.cuda.synchronize()
        snapshot = _snapshot(
            role="candidate",
            profile=_observed_profile(
                case,
                dtype=_common_dtype(named_adapters, named_base),
                training_attention=_candidate_training_attention(model, observer.calls),
            ),
            loss_sum=result.loss_sum,
            num_target_tokens=result.normalization_count,
            named_lora=named_lora,
            initial_lora=initial_lora,
            raw_gradients=_clone_named_tensors(tuple(raw_gradients.items())),
            trainable_names=trainable_names,
            named_base=named_base,
        )
    finally:
        try:
            if trainer is not None:
                trainer.discard()
        finally:
            runtime.close()
    return snapshot


def _is_qwen3_5(model_path: Path) -> bool:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path).model_type in ("qwen3_5", "qwen3_5_text")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("reference", "hf", "candidate"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    case = _load_case(args.case)
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        raise ValueError(f"model path is not a directory: {model_path}")
    if model_path.name != case.revision:
        raise ValueError(
            f"model snapshot {model_path.name!r} does not match case revision {case.revision!r}"
        )
    _configure_determinism(case.seed)
    if args.role == "reference":
        snapshot = _run_hf(model_path, case, role="reference", dtype=torch.float64)
    elif args.role == "hf":
        snapshot = _run_hf(model_path, case, role="hf", dtype=torch.float32)
    elif _is_qwen3_5(model_path):
        snapshot = _run_qwen3_5_candidate(model_path, case)
    else:
        snapshot = _run_candidate(model_path, case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, args.output)


if __name__ == "__main__":
    main()
