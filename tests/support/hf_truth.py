"""The Transformers truth: a checkpoint's own Transformers class on the frozen parity prompts, as
a child process with the optional CUDA kernels blocked (``--help``).

It hides ``fla`` and ``causal_conv1d`` and turns hub kernels off before transformers imports, as
``tests/modules/conftest.py`` does, then proves the torch paths ran. ``--dtype float32`` computes in
IEEE FP32 and refuses ``NVIDIA_TF32_OVERRIDE`` and any non-FP32 parameter, offloaded ones included,
or cached state: the convolution casts its input to its weight's dtype, so one BF16 weight would
compute that step in BF16.

Two modes, each a ``TruthRun``. ``whole`` holds the model on ``--device``, or spread over GPUs
and CPU RAM (``--device-map auto``); it decodes each prompt's greedy continuation and
teacher-forces the prompt and a continuation, its own or ``--truth``'s. ``stream``
(``tests/support/hf_layer_stream.py``) holds one decoder layer at a time: every prompt passes a
layer before the next one loads, so each weight crosses to the device once, and one GPU computes
a truth larger than its memory. It teacher-forces the continuation of a finished run
(``--continuation``) and decodes nothing; with ``--attention-chunk`` attention runs over query
blocks of that many rows, which a long prompt needs. It runs attention-only decoders
(``ReferenceArchitecture.streams``).

Dump keys: ``run_prompt``. ``manifest.json`` is written last: a folder without it holds no
finished run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel

# Marks the last stdout line: the manifest as JSON.
RECEIPT_PREFIX = "CTI_HF_TRUTH="
# The frozen parity prompts: their texts, and their token ids under each pinned checkpoint.
PROMPTS_PATH = Path(__file__).resolve().parents[1] / "gpu" / "models" / "parity_prompts.json"
BLOCKED_PACKAGES = ("fla", "causal_conv1d")
# The functions a Gated DeltaNet modeling module binds to BLOCKED_PACKAGES at its import.
_GATED_DELTA_NET_KERNELS = (
    "causal_conv1d_update",
    "causal_conv1d_fn",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)


@dataclass(frozen=True)
class ReferenceArchitecture:
    """How the child runs one Transformers architecture."""

    # ``transformers.models.<module>.modeling_<module>``.
    module: str
    # Routed experts, which run Transformers' per-expert loop.
    moe: bool = False
    # The functions the modeling module binds to BLOCKED_PACKAGES' kernels at its import.
    kernel_bound: tuple[str, ...] = ()
    # An attention-only decoder, which ``--mode stream`` runs one layer at a time.
    streams: bool = False

    @property
    def modeling(self) -> str:
        return f"transformers.models.{self.module}.modeling_{self.module}"


ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration": ReferenceArchitecture(
        "qwen3_5", kernel_bound=_GATED_DELTA_NET_KERNELS
    ),
    "Qwen3_5ForCausalLM": ReferenceArchitecture("qwen3_5", kernel_bound=_GATED_DELTA_NET_KERNELS),
    "Qwen3_5MoeForConditionalGeneration": ReferenceArchitecture(
        "qwen3_5_moe", moe=True, kernel_bound=_GATED_DELTA_NET_KERNELS
    ),
    "Qwen3_5MoeForCausalLM": ReferenceArchitecture(
        "qwen3_5_moe", moe=True, kernel_bound=_GATED_DELTA_NET_KERNELS
    ),
    "Qwen3MoeForCausalLM": ReferenceArchitecture("qwen3_moe", moe=True, streams=True),
    "Qwen3ForCausalLM": ReferenceArchitecture("qwen3", streams=True),
}
MODES = ("whole", "stream")
DTYPES = ("bfloat16", "float32")
DEVICE_MAPS = ("device", "auto")
# GPU memory --device-map auto leaves free when no --max-gpu-gib is given: the CUDA context,
# the allocator's slack and the activations, the largest of which are the FP32 logits of a
# teacher-forced 1012-token sequence over a 248k vocabulary (1 GiB).
AUTO_GPU_HEADROOM_GIB = 8
# Share of the available CPU RAM that offloaded weights may take. Past it Transformers
# would offload to disk, which this child refuses (it names no offload folder).
_CPU_RAM_SHARE = 0.9
_GIB = 1 << 30
# fp32_precision() of a --dtype float32 run: IEEE FP32 in every matmul and cuDNN operation,
# TF32 nowhere (torch 2.13 otherwise runs cuDNN convolutions in TF32), and no
# NVIDIA_TF32_OVERRIDE for the CUDA libraries to read behind torch.
IEEE_FP32_PRECISION: dict[str, Any] = {
    "float32_matmul_precision": "highest",
    "backends": "ieee",
    "cuda.matmul": "ieee",
    "cuda.matmul.allow_tf32": False,
    "cudnn": "ieee",
    "cudnn.conv": "ieee",
    "cudnn.rnn": "ieee",
    "NVIDIA_TF32_OVERRIDE": None,
}
# Distributions whose versions, with this file's source, make up a run's code version.
_CODE_DISTRIBUTIONS = ("torch", "transformers", "accelerate")
_PROMPT_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
# Rows of the teacher-forced logits converted to FP32 at a time.
_LOGIT_ROWS = 256


def reference_architecture(name: str) -> ReferenceArchitecture:
    """The child's rules for Transformers architecture ``name``."""

    if name not in ARCHITECTURES:
        raise ValueError(f"the truth child runs {sorted(ARCHITECTURES)}, not {name}")
    return ARCHITECTURES[name]


def block_optional_kernels() -> None:
    """Hide ``BLOCKED_PACKAGES`` and turn hub kernels off, before transformers imports."""

    if "transformers" in sys.modules:
        raise RuntimeError("transformers is already imported: its kernel bindings may be bound")
    for package in BLOCKED_PACKAGES:
        loaded = sorted(
            name
            for name, module in sys.modules.items()
            if module is not None and (name == package or name.startswith(f"{package}."))
        )
        if loaded:
            raise RuntimeError(f"{package} is already imported: {loaded}")
        sys.modules[package] = None  # type: ignore[assignment]
    os.environ["USE_HUB_KERNELS"] = "NO"


def loaded_blocked_modules() -> list[str]:
    """Modules of ``BLOCKED_PACKAGES`` that were imported despite the block."""

    return sorted(
        name
        for name, module in sys.modules.items()
        if module is not None
        and any(name == package or name.startswith(f"{package}.") for package in BLOCKED_PACKAGES)
    )


def kernel_bindings(modeling: ModuleType, names: Sequence[str]) -> dict[str, str]:
    """What each function ``names`` lists runs: ``"torch"`` or the bound function.

    ``use_kernel_func_from_hub_with_fallback`` closes over the torch function and the
    implementation it chose; a change to that wrapper fails here rather than passing.
    """

    bindings = {}
    for name in names:
        try:
            closure = inspect.getclosurevars(getattr(modeling, name)).nonlocals
            implementation, torch_function = closure["implementation"], closure["torch_function"]
        except (AttributeError, KeyError, TypeError) as error:
            raise RuntimeError(f"cannot tell what {modeling.__name__}.{name} runs") from error
        bindings[name] = (
            "torch"
            if implementation is torch_function
            else f"{implementation.__module__}.{implementation.__qualname__}"
        )
    return bindings


def tokenize_prompt(
    tokenizer: Any, prompt: Mapping[str, Any], chat_kwargs: Mapping[str, Any]
) -> list[int]:
    """Token ids of one prompt entry of the parity prompts file."""

    if prompt["kind"] == "chat":
        ids = tokenizer.apply_chat_template(
            list(prompt["messages"]), tokenize=True, return_dict=False, **chat_kwargs
        )
    elif prompt["kind"] == "text":
        ids = tokenizer.encode(prompt["text"], add_special_tokens=False)
    else:
        raise ValueError(f"prompt {prompt['name']!r} has unknown kind {prompt['kind']!r}")
    if not isinstance(ids, list) or not all(type(token) is int for token in ids):
        raise TypeError(f"prompt {prompt['name']!r} did not tokenize to a list of ints")
    return ids


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_version() -> dict[str, str]:
    """What a run's numbers depend on besides its inputs: the truth child's source (this file
    and its layer streaming) and the versions of Python and ``_CODE_DISTRIBUTIONS``. Imports
    none of them."""

    sources = (Path(__file__), Path(__file__).with_name("hf_layer_stream.py"))
    return {
        "helper_sha256": hashlib.sha256(
            b"".join(bytes.fromhex(sha256_file(source)) for source in sources)
        ).hexdigest(),
        "python": sys.version.split()[0],
        **{name: importlib.metadata.version(name) for name in _CODE_DISTRIBUTIONS},
    }


def fp32_precision() -> dict[str, Any]:
    """How torch computes FP32 matmuls and cuDNN operations now, in the keys of
    ``IEEE_FP32_PRECISION``."""

    import torch

    backends = torch.backends
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "backends": backends.fp32_precision,
        "cuda.matmul": backends.cuda.matmul.fp32_precision,
        "cuda.matmul.allow_tf32": backends.cuda.matmul.allow_tf32,
        "cudnn": backends.cudnn.fp32_precision,
        "cudnn.conv": backends.cudnn.conv.fp32_precision,
        "cudnn.rnn": backends.cudnn.rnn.fp32_precision,
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }


def use_ieee_fp32() -> None:
    """Compute every FP32 matmul and cuDNN operation in IEEE FP32, with torch's own
    ``fp32_precision`` settings (mixing them with the legacy ``allow_tf32`` flags makes
    torch refuse to report cuDNN's), or raise."""

    import torch

    if "NVIDIA_TF32_OVERRIDE" in os.environ:
        raise RuntimeError(
            "NVIDIA_TF32_OVERRIDE is set: the CUDA libraries read it behind torch's precision "
            "settings, so an FP32 run refuses it; unset it"
        )
    torch.backends.fp32_precision = "ieee"
    for backend in (
        torch.backends.cuda.matmul,
        torch.backends.cudnn,
        torch.backends.cudnn.conv,
        torch.backends.cudnn.rnn,
    ):
        backend.fp32_precision = "ieee"
    check_ieee_fp32()


def check_ieee_fp32() -> None:
    precision = fp32_precision()
    if precision != IEEE_FP32_PRECISION:
        raise RuntimeError(f"FP32 is not computed in IEEE FP32: {precision}")


def load_prompts(path: Path, model_key: str) -> tuple[dict[str, Any], list[tuple[str, list[int]]]]:
    """The parity prompts file and the (name, token ids) of ``model_key``'s prompts."""

    frozen = json.loads(path.read_text(encoding="utf-8"))
    if model_key not in frozen["models"]:
        raise KeyError(
            f"{path} has no token ids for {model_key}; it has {sorted(frozen['models'])}"
        )
    prompts = []
    for prompt in frozen["prompts"]:
        name = prompt["name"]
        if not _PROMPT_NAME.fullmatch(name):
            raise ValueError(f"prompt name {name!r} is not lower-case-kebab")
        ids = prompt["token_ids"][model_key]
        if not ids or not all(type(token) is int and token >= 0 for token in ids):
            raise ValueError(f"prompt {name!r} has invalid token ids for {model_key}")
        prompts.append((name, ids))
    names = [name for name, _ in prompts]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate prompt names in {path}")
    return frozen, prompts


def check_tokenizer(checkpoint: Path, frozen: Mapping[str, Any], model_key: str) -> str:
    """``"matched"`` when the checkpoint's tokenizer re-derives every frozen id list,
    ``"absent"`` when the checkpoint has no tokenizer; raises on any difference."""

    if not any(
        (checkpoint / name).exists() for name in ("tokenizer.json", "tokenizer_config.json")
    ):
        return "absent"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    for prompt in frozen["prompts"]:
        derived = tokenize_prompt(tokenizer, prompt, frozen["chat_template_kwargs"])
        if derived != prompt["token_ids"][model_key]:
            raise ValueError(
                f"prompt {prompt['name']!r}: {checkpoint} tokenizes it differently from the "
                f"frozen ids for {model_key}; the prompts or the tokenizer changed"
            )
    return "matched"


def offload_max_memory(
    device: torch.device,
    max_gpu_gib: float | None,
    *,
    free_gpu_bytes: Sequence[int],
    available_cpu_bytes: int,
) -> dict[int | str, int]:
    """``max_memory`` of ``--device-map auto``: ``device`` and the GPUs after it, one per
    ``free_gpu_bytes`` entry (their free memory), each up to ``max_gpu_gib`` (by default its
    free memory less ``AUTO_GPU_HEADROOM_GIB``), the rest in CPU RAM."""

    cpu = {"cpu": int(available_cpu_bytes * _CPU_RAM_SHARE)}
    if device.type == "cpu":
        if max_gpu_gib is not None or free_gpu_bytes:
            raise ValueError("--max-gpu-gib and --gpus need a CUDA --device")
        return cpu
    if device.type != "cuda":
        raise ValueError(f"--device-map auto runs on a CUDA device or the CPU, not {device}")
    if not free_gpu_bytes:
        raise ValueError(f"no GPU from {device} on")
    first = 0 if device.index is None else device.index
    gpus: dict[int | str, int] = {}
    for index, free in enumerate(free_gpu_bytes, start=first):
        budget = (
            max_gpu_gib * _GIB if max_gpu_gib is not None else free - AUTO_GPU_HEADROOM_GIB * _GIB
        )
        if budget <= 0:
            raise ValueError(
                f"no GPU memory for weights on cuda:{index}: budget {budget / _GIB:.1f} GiB"
            )
        gpus[index] = int(budget)
    return {**gpus, **cpu}


def load_reference(
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_memory: Mapping[int | str, int] | None = None,
    *,
    attention: str = "eager",
    device_map: Mapping[str, Any] | None = None,
) -> PreTrainedModel:
    """The checkpoint's own Transformers class, selected attention, in inference mode: all on
    ``device``, or with ``max_memory`` placed by Transformers' ``"auto"`` device map."""

    import transformers

    if attention not in ("eager", "sdpa") or (device_map is not None and max_memory is not None):
        raise ValueError("expected eager/SDPA attention and one placement policy")
    config = transformers.AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    name = (config.architectures or [None])[0]
    architecture = reference_architecture(name)
    model_class = getattr(transformers, name)
    experts = {"experts_implementation": "eager"} if architecture.moe else {}
    placement: dict[str, Any] = (
        {"device_map": {"": device}}
        if max_memory is None
        else {"device_map": "auto", "max_memory": dict(max_memory)}
    )
    return model_class.from_pretrained(
        checkpoint,
        dtype=dtype,
        attn_implementation=attention,
        local_files_only=True,
        **experts,
        **(placement if device_map is None else {"device_map": dict(device_map)}),
    ).eval()


def parameter_bytes(model: PreTrainedModel) -> dict[str, int]:
    """Parameter bytes by device. ``"offloaded"`` counts the weights accelerate keeps in CPU
    RAM and copies to the GPU for each forward; between forwards they are meta tensors."""

    totals: dict[str, int] = {}
    for parameter in model.parameters():
        where = "offloaded" if parameter.device.type == "meta" else str(parameter.device)
        totals[where] = totals.get(where, 0) + parameter.numel() * parameter.element_size()
    return totals


def parameter_dtypes(model: PreTrainedModel) -> dict[str, list[str]]:
    """The dtypes of the parameters by device, as ``parameter_bytes`` places them. An
    offloaded parameter counts twice: as its meta placeholder, whose dtype accelerate casts
    the streamed copy to, and as the copy itself, held by the offloading hook of the
    parameter's module or an ancestor (Transformers' ``load_offloaded_parameter`` finds it
    the same way)."""

    dtypes: dict[str, set[str]] = {}
    held: dict[str, torch.dtype] = {}
    for prefix, module, hook in _offloading_hooks(model):
        for name, _ in module.named_parameters(recurse=hook.place_submodules):
            full_name = f"{prefix}.{name}" if prefix else name
            held.setdefault(full_name, hook.weights_map[name].dtype)
    for name, parameter in model.named_parameters():
        where = "offloaded" if parameter.device.type == "meta" else str(parameter.device)
        names = dtypes.setdefault(where, set())
        names.add(dtype_name(parameter.dtype))
        if where == "offloaded":
            if name not in held:
                raise RuntimeError(f"{name} is offloaded, but no hook holds its weight")
            names.add(dtype_name(held[name]))
    return {where: sorted(names) for where, names in dtypes.items()}


def pin_offloaded_weights(
    model: PreTrainedModel,
    pin: Callable[[torch.Tensor], torch.Tensor] = lambda tensor: tensor.pin_memory(),
) -> int:
    """Page-lock the CPU RAM copies of the offloaded weights in place, so that each
    forward copies them to the GPU directly instead of through a pageable staging buffer;
    the bytes pinned. ``pin`` stands in for ``Tensor.pin_memory`` where no GPU is."""

    from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset

    stores: dict[int, dict[str, torch.Tensor]] = {}
    for prefix, _, hook in _offloading_hooks(model):
        weights = hook.weights_map
        store = weights.dataset if isinstance(weights, PrefixedDataset) else weights
        if isinstance(store, OffloadedWeightsLoader):
            store = store.state_dict
        if not isinstance(store, dict):
            raise RuntimeError(f"{prefix or 'the model'} streams its weights from {type(store)}")
        stores[id(store)] = store
    pinned = 0
    for store in stores.values():
        for key, tensor in store.items():
            if tensor.device.type == "cpu" and not tensor.is_pinned():
                store[key] = pin(tensor)
                pinned += tensor.numel() * tensor.element_size()
    return pinned


def check_truth(folder: Path, expected: Mapping[str, Any]) -> None:
    """Raise unless ``folder`` holds a finished truth run of ``expected``'s top-level
    manifest values; ``prompt_names`` compares its prompts' names."""

    path = folder / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{folder} holds no finished run: it has no manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual = {key: manifest.get(key) for key in expected if key != "prompt_names"}
    actual["prompt_names"] = [prompt["name"] for prompt in manifest["prompts"]]
    differences = {
        key: (actual[key], value) for key, value in expected.items() if actual[key] != value
    }
    if differences:
        raise ValueError(
            f"the truth in {folder} ran something else (truth, this run): {differences}"
        )


def recurrent_states(
    cache: Any, layer_types: Sequence[str]
) -> tuple[dict[str, torch.Tensor], dict[str, set[str]]]:
    """FP32 CPU copies of each Gated DeltaNet layer's states, and the cache's dtypes by
    state kind, over every layer; none for a model without such layers."""

    import torch

    tensors: dict[str, torch.Tensor] = {}
    dtypes: dict[str, set[str]] = {}
    for index, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention":
            continue
        layer = cache.layers[index]
        for kind, state in (
            ("recurrent_state", layer.recurrent_states[0]),
            ("conv_state", layer.conv_states[0]),
        ):
            if state is None or state.shape[0] != 1:
                raise RuntimeError(f"layer {index} has no single-sequence {kind} after the prompt")
            # A copy: the cached decode that follows updates the cache in place.
            tensors[f"{kind}.{index}"] = state[0].detach().to("cpu", torch.float32, copy=True)
            dtypes.setdefault(kind, set()).add(dtype_name(state.dtype))
    return tensors, dtypes


def run_prompt(
    model: PreTrainedModel,
    input_ids: list[int],
    *,
    device: torch.device,
    greedy_tokens: int,
    top_k: int,
    truth: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, set[str]]]:
    """Greedy continuation, the recurrent states after the prompt (a model with Gated DeltaNet
    layers) and the teacher-forced top-k over the prompt and a continuation: this run's greedy
    one, or ``truth``'s (the truth run's dump of this prompt), at whose top-k ids the logits are
    gathered too. Also the cache's state dtypes (``recurrent_states``).

    Shapes:
        ``input_ids`` [P]; ``greedy_ids``, ``greedy_margins``, ``teacher_forced_ids`` [N];
        ``prefill_logits`` [V]; ``topk_ids``, ``topk_logits``, ``at_truth_ids`` [P + N, k];
        ``logsumexp`` [P + N]; ``recurrent_state.<layer>`` [H, K, V]; ``conv_state.<layer>``
        [C, W]. Ids are int64, the rest FP32 on the CPU; margins are top-1 minus top-2 nats.
    """

    import torch

    layer_types = getattr(model.config.get_text_config(), "layer_types", None) or ()
    prompt = torch.tensor([input_ids], dtype=torch.int64, device=device)
    output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    states, state_dtypes = recurrent_states(cache, layer_types)
    logits = output.logits[0, -1].float()
    prefill_logits = logits.to("cpu", copy=True)
    greedy_ids, margins = [], []
    for step in range(greedy_tokens):
        token = logits.argmax()
        top2 = logits.topk(2).values
        greedy_ids.append(int(token))
        margins.append(float(top2[0] - top2[1]))
        if step + 1 == greedy_tokens:
            break
        output = model(input_ids=token.view(1, 1), past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[0, -1].float()
    del cache, output

    continuation = greedy_ids if truth is None else truth["greedy_ids"].tolist()
    sequence = torch.tensor([input_ids + continuation], dtype=torch.int64, device=device)
    teacher_forced = model(input_ids=sequence, use_cache=False).logits[0]
    reduced = reduce_rows(teacher_forced, top_k=top_k, truth=truth)
    del teacher_forced
    tensors = {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "greedy_ids": torch.tensor(greedy_ids, dtype=torch.int64),
        "greedy_margins": torch.tensor(margins, dtype=torch.float32),
        "prefill_logits": prefill_logits,
        "teacher_forced_ids": torch.tensor(continuation, dtype=torch.int64),
        **reduced,
        **states,
    }
    return tensors, state_dtypes


def reduce_rows(
    logits: torch.Tensor,
    *,
    top_k: int,
    truth: Mapping[str, torch.Tensor] | None = None,
    first_row: int = 0,
) -> dict[str, torch.Tensor]:
    """The top-k ids and logits and the logsumexp of each row of ``logits``, which are rows
    ``first_row`` onward of a teacher-forced sequence, and with ``truth`` the logits at the
    truth's top-k ids of the same rows; in FP32 on the CPU."""

    import torch

    reduced: dict[str, list[torch.Tensor]] = {
        "topk_ids": [],
        "topk_logits": [],
        "logsumexp": [],
        "at_truth_ids": [],
    }
    for start in range(0, len(logits), _LOGIT_ROWS):
        rows = logits[start : start + _LOGIT_ROWS].float()
        values, indices = rows.topk(top_k, dim=-1)
        reduced["topk_ids"].append(indices.cpu())
        reduced["topk_logits"].append(values.cpu())
        reduced["logsumexp"].append(torch.logsumexp(rows, dim=-1).cpu())
        if truth is not None:
            first = first_row + start
            truth_ids = truth["topk_ids"][first : first + len(rows)].to(rows.device)
            reduced["at_truth_ids"].append(rows.gather(1, truth_ids).cpu())
    return {key: torch.cat(pieces) for key, pieces in reduced.items() if pieces}


def content_digest(value: Any) -> str:
    """Canonical content identity, including tensor dtype/shape and original image bytes."""

    import torch

    def encode(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            raw = item.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
            return [str(item.dtype), list(item.shape), hashlib.sha256(raw).hexdigest()]
        if isinstance(item, bytes):
            return hashlib.sha256(item).hexdigest()
        raise TypeError(type(item).__name__)

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=encode).encode()).hexdigest()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _offloading_hooks(model: PreTrainedModel) -> Iterator[tuple[str, Any, Any]]:
    """``(module name, module, hook)`` of each accelerate hook that offloads weights; its
    ``weights_map`` holds them by name, with ``place_submodules`` the submodules' too."""

    from accelerate.hooks import AlignDevicesHook, SequentialHook

    for prefix, module in model.named_modules():
        hook = getattr(module, "_hf_hook", None)
        for align in hook.hooks if isinstance(hook, SequentialHook) else (hook,):
            if isinstance(align, AlignDevicesHook) and align.offload:
                if align.weights_map is None:
                    raise RuntimeError(f"{prefix or 'the model'} is offloaded without its weights")
                yield prefix, module, align


def run_dump(folder: Path, name: str, input_ids: list[int]) -> dict[str, torch.Tensor]:
    """A finished run's dump of prompt ``name``, which must have run ``input_ids``."""

    from safetensors.torch import load_file

    dump = load_file(folder / f"{name}.safetensors")
    if dump["input_ids"].tolist() != input_ids:
        raise ValueError(f"the run in {folder} has other token ids for prompt {name!r}")
    return dump


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="checkpoint folder (an HF snapshot)"
    )
    parser.add_argument(
        "--model-key", required=True, help="REPO@REVISION: the prompts file's id set"
    )
    parser.add_argument(
        "--prompts", type=Path, required=True, help="the frozen parity prompts file"
    )
    parser.add_argument("--output", type=Path, required=True, help="new or empty output folder")
    parser.add_argument("--mode", choices=MODES, default="whole")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument(
        "--device-map",
        choices=DEVICE_MAPS,
        default="device",
        help="device: all on --device; auto: what does not fit on the GPU stays in CPU RAM",
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="--device-map auto: GPUs from --device on"
    )
    parser.add_argument(
        "--max-gpu-gib", type=float, help="--device-map auto: GiB of weights each GPU holds"
    )
    parser.add_argument(
        "--truth",
        type=Path,
        help="whole: a truth run's output folder: teacher-force its greedy continuations",
    )
    parser.add_argument(
        "--continuation",
        type=Path,
        help="stream: a finished run's output folder: teacher-force its greedy continuations",
    )
    parser.add_argument(
        "--attention-chunk", type=int, help="stream: query rows each attention block computes"
    )
    parser.add_argument("--greedy-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args(argv)
    if args.greedy_tokens < 1 or args.top_k < 2:
        parser.error("--greedy-tokens must be at least 1 and --top-k at least 2")
    if args.max_gpu_gib is not None and (
        args.device_map != "auto" or not 0 < args.max_gpu_gib < float("inf")
    ):
        parser.error("--max-gpu-gib takes a positive number, with --device-map auto")
    if args.gpus < 1 or (args.gpus > 1 and args.device_map != "auto"):
        parser.error("--gpus takes a positive count, more than 1 only with --device-map auto")
    if args.mode == "stream":
        if args.continuation is None or args.truth is not None or args.device_map != "device":
            parser.error("--mode stream takes --continuation, on one --device, without --truth")
        if args.attention_chunk is not None and args.attention_chunk < 1:
            parser.error("--attention-chunk takes a positive row count")
    elif args.continuation is not None or args.attention_chunk is not None:
        parser.error("--continuation and --attention-chunk need --mode stream")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    block_optional_kernels()
    started = time.monotonic()

    import torch
    from safetensors.torch import save_file
    from transformers.integrations import hub_kernels

    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    architecture = reference_architecture(config["architectures"][0])
    modeling = importlib.import_module(architecture.modeling)
    fp32 = args.dtype == "float32"
    if fp32:
        use_ieee_fp32()
    bindings = kernel_bindings(modeling, architecture.kernel_bound)
    if hub_kernels._kernels_enabled or any(value != "torch" for value in bindings.values()):
        raise RuntimeError(
            f"not the torch paths: hub kernels {hub_kernels._kernels_enabled}, {bindings}"
        )
    frozen, prompts = load_prompts(args.prompts, args.model_key)
    tokenizer_check = check_tokenizer(args.checkpoint, frozen, args.model_key)
    followed = args.truth if args.mode == "whole" else args.continuation
    if followed is not None:
        check_truth(
            followed,
            {
                "model_key": args.model_key,
                "prompts_sha256": sha256_file(args.prompts),
                "greedy_tokens": args.greedy_tokens,
                "prompt_names": [name for name, _ in prompts],
                **({"top_k": args.top_k} if args.mode == "whole" else {}),
            },
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} is not empty")

    device = torch.device(args.device)
    if args.gpus > 1 and device.type != "cuda":
        raise ValueError(f"--gpus {args.gpus} needs a CUDA --device, not {device}")
    max_memory = None
    if args.device_map == "auto":
        import psutil

        first = 0 if device.index is None else device.index
        max_memory = offload_max_memory(
            device,
            args.max_gpu_gib,
            free_gpu_bytes=(
                [torch.cuda.mem_get_info(index)[0] for index in range(first, first + args.gpus)]
                if device.type == "cuda"
                else []
            ),
            available_cpu_bytes=psutil.virtual_memory().available,
        )
    dtype = getattr(torch, args.dtype)
    run: TruthRun
    if args.mode == "whole":
        run = _WholeRun(args, device, dtype, max_memory)
    else:
        from tests.support.hf_layer_stream import StreamRun

        run = StreamRun(args, device, dtype)
    load_seconds = round(time.monotonic() - started, 3)
    print(f"loaded in {load_seconds:.0f} s: {run.loaded()}", flush=True)
    if max(max(ids) for _, ids in prompts) >= run.vocab_size:
        raise ValueError(f"a prompt token id is outside the vocabulary of {run.vocab_size}")
    summaries: list[dict[str, Any]] = []
    state_dtypes: dict[str, set[str]] = {}
    with torch.inference_mode():
        for name, tensors, seconds, dtypes_of_states in run.prompts(prompts):
            for kind, names in dtypes_of_states.items():
                state_dtypes.setdefault(kind, set()).update(names)
            if fp32 and any(names != {"float32"} for names in state_dtypes.values()):
                raise RuntimeError(f"an FP32 run cached states of other dtypes: {state_dtypes}")
            save_file(
                tensors, args.output / f"{name}.safetensors", metadata={"model_key": args.model_key}
            )
            margins = tensors["greedy_margins"]
            summary = {
                "name": name,
                "file": f"{name}.safetensors",
                "prompt_tokens": len(tensors["input_ids"]),
                "greedy_tokens": len(tensors["greedy_ids"]),
                "min_margin": float(margins.min()),
                "min_margin_step": int(margins.argmin()),
                "seconds": seconds,
            }
            summary.update(run.prompt_report(tensors))
            summaries.append(summary)
            print(f"{name}: {summary['prompt_tokens']} prompt tokens, {seconds:.0f} s", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if fp32:
        check_ieee_fp32()

    loaded = loaded_blocked_modules()
    still_blocked = {package: sys.modules.get(package, ...) is None for package in BLOCKED_PACKAGES}
    if loaded or not all(still_blocked.values()):
        raise RuntimeError(
            f"blocked packages were imported: {loaded}, still blocked {still_blocked}"
        )
    manifest = {
        "model_key": args.model_key,
        "checkpoint": str(args.checkpoint),
        "config_sha256": sha256_file(args.checkpoint / "config.json"),
        "architecture": config["architectures"][0],
        "mode": args.mode,
        "prompts_file": str(args.prompts),
        "prompts_sha256": sha256_file(args.prompts),
        "tokenizer_check": tokenizer_check,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        **run.manifest(),
        "device_map": args.device_map,
        "gpus": args.gpus,
        "max_memory": None if max_memory is None else {str(k): v for k, v in max_memory.items()},
        "teacher_forced": {"whole": "own" if args.truth is None else "truth"}.get(
            args.mode, "continuation"
        ),
        "truth": None if args.truth is None else str(args.truth),
        "continuation": None if args.continuation is None else str(args.continuation),
        "greedy_tokens": args.greedy_tokens,
        "top_k": args.top_k,
        "code_version": code_version(),
        "fp32_precision": fp32_precision(),
        "use_hub_kernels": os.environ["USE_HUB_KERNELS"],
        "hub_kernels_enabled": hub_kernels._kernels_enabled,
        "blocked_packages": still_blocked,
        "loaded_blocked_modules": loaded,
        "kernel_bindings": bindings,
        "state_dtypes": {kind: sorted(names) for kind, names in state_dtypes.items()},
        "prompts": summaries,
        "load_seconds": load_seconds,
        "seconds": round(time.monotonic() - started, 3),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    print(RECEIPT_PREFIX + json.dumps(manifest), flush=True)
    return 0


class TruthRun(Protocol):
    """One mode of the child: its model loaded, then each prompt's dump in turn."""

    vocab_size: int

    def loaded(self) -> str:
        """What loading placed where, for the log."""
        ...

    def prompts(
        self, prompts: Sequence[tuple[str, list[int]]]
    ) -> Iterator[tuple[str, dict[str, torch.Tensor], float, dict[str, set[str]]]]:
        """Each prompt's name, dump, seconds and cached state dtypes (``run_prompt``)."""
        ...

    def prompt_report(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """The mode's own entries in a prompt's manifest summary."""
        ...

    def manifest(self) -> dict[str, Any]:
        """The mode's own manifest entries: attention, experts and parameter placement."""
        ...


class _WholeRun:
    """``--mode whole``: the model resident, each prompt decoded and teacher-forced in turn."""

    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device,
        dtype: torch.dtype,
        max_memory: Mapping[int | str, int] | None,
    ) -> None:
        import torch

        model = load_reference(args.checkpoint, device, dtype, max_memory)
        self._placement, self._dtypes = parameter_bytes(model), parameter_dtypes(model)
        if dtype == torch.float32 and any(names != ["float32"] for names in self._dtypes.values()):
            raise RuntimeError(f"an FP32 run holds parameters of other dtypes: {self._dtypes}")
        self._pinned = pin_offloaded_weights(model) if device.type == "cuda" else 0
        self._model, self._args, self._device = model, args, device
        self.vocab_size = model.config.get_text_config().vocab_size

    def loaded(self) -> str:
        return (
            f"parameter bytes {self._placement}, dtypes {self._dtypes}, "
            f"{self._pinned / _GIB:.1f} GiB of offloaded weights pinned"
        )

    def prompts(
        self, prompts: Sequence[tuple[str, list[int]]]
    ) -> Iterator[tuple[str, dict[str, torch.Tensor], float, dict[str, set[str]]]]:
        args = self._args
        for name, input_ids in prompts:
            started = time.monotonic()
            tensors, state_dtypes = run_prompt(
                self._model,
                input_ids,
                device=self._device,
                greedy_tokens=args.greedy_tokens,
                top_k=args.top_k,
                truth=None if args.truth is None else run_dump(args.truth, name, input_ids),
            )
            yield name, tensors, round(time.monotonic() - started, 3), state_dtypes

    def prompt_report(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        return {}

    def manifest(self) -> dict[str, Any]:
        config = self._model.config
        return {
            "attention": config._attn_implementation,
            "experts": getattr(config.get_text_config(), "_experts_implementation", None),
            "attention_chunk": None,
            "parameter_bytes": self._placement,
            "parameter_dtypes": self._dtypes,
            "pinned_offloaded_bytes": self._pinned,
        }


if __name__ == "__main__":
    sys.exit(main())
