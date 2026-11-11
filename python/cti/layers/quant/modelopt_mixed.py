"""ModelOpt MIXED_PRECISION checkpoints: each listed module is FP8 (static per-tensor
W8A8) or NVFP4 (W4A4 in 1x16 blocks).

The precision of a module is declared three times: by the config groups and the
``quantized_layers`` of ``config.json``'s ``quantization_config``, and by the
``quantized_layers`` of ``hf_quant_config.json``. The three must agree module by module.
Unknown keys, and any KV-cache quantization, are refused by name.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Self

from .fp8_static import Fp8StaticLinear
from .nvfp4 import Nvfp4Linear

if TYPE_CHECKING:
    from ..experts_method import ExpertsMethod
    from ..linear_method import LinearMethod
    from . import CheckpointIndex, CheckpointPart, Documents

_DOCUMENT = "hf_quant_config.json"
_ALGORITHM = "MIXED_PRECISION"
# A config group's weights and input activations (alike, both static) per algorithm.
_GROUPS = MappingProxyType(
    {
        "FP8": {"num_bits": 8, "type": "float", "dynamic": False},
        "NVFP4": {"num_bits": 4, "type": "float", "dynamic": False, "group_size": 16},
    }
)
# A quantized_layers entry per algorithm.
_LAYERS = MappingProxyType(
    {"FP8": {"quant_algo": "FP8"}, "NVFP4": {"quant_algo": "NVFP4", "group_size": 16}}
)
# Header dtypes of quantized weights and block scales, which only listed modules hold.
_QUANTIZED_DTYPES = frozenset({"F8_E4M3", "U8"})
# The scalar scales of a static FP8 module.
_STATIC_SCALES = MappingProxyType({"weight_scale": ("F32", ()), "input_scale": ("F32", ())})


def _fields(
    mapping: Any, where: str, required: frozenset[str], optional: frozenset[str] = frozenset()
) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{where} must be a mapping, not {mapping!r}")
    if unknown := sorted(set(mapping) - required - optional):
        raise ValueError(f"{where} has unknown keys {unknown}")
    if missing := sorted(required - set(mapping)):
        raise ValueError(f"{where} lacks {missing}")
    return mapping


def _names(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{where} must be a list of module names, not {value!r}")
    return value


def _producer(producer: Any, where: str) -> Mapping[str, Any]:
    _fields(producer, where, frozenset({"name", "version"}))
    if producer["name"] != "modelopt":
        raise ValueError(f"{where} name={producer['name']!r}: the producer must be modelopt")
    return producer


def _groups(groups: Any) -> dict[str, str]:
    """Module -> algorithm, from config.json's config_groups."""

    where = "config.json quantization_config config_groups"
    if not isinstance(groups, Mapping):
        raise ValueError(f"{where} must be a mapping")
    modules: dict[str, str] = {}
    for name, group in groups.items():
        _fields(group, f"{where} {name}", frozenset({"weights", "input_activations", "targets"}))
        weights, inputs = group["weights"], group["input_activations"]
        algorithm = next(
            (algo for algo, spec in _GROUPS.items() if weights == spec and inputs == spec), None
        )
        if algorithm is None:
            raise ValueError(
                f"{where} {name}: weights {weights!r} with input_activations {inputs!r} "
                "are neither static FP8 nor NVFP4 in 16-element groups"
            )
        for module in _names(group["targets"], f"{where} {name} targets"):
            if module in modules:
                raise ValueError(f"{where}: {module} is a target of more than one group")
            modules[module] = algorithm
    return modules


def _layers(layers: Any, where: str) -> dict[str, str]:
    """Module -> algorithm, from a quantized_layers mapping."""

    if not isinstance(layers, Mapping):
        raise ValueError(f"{where} must be a mapping")
    modules = {}
    for module, entry in layers.items():
        algorithm = next((algo for algo, spec in _LAYERS.items() if entry == spec), None)
        if algorithm is None:
            raise ValueError(f"{where} {module}: {entry!r} is none of {list(_LAYERS.values())}")
        modules[module] = algorithm
    return modules


@dataclass(frozen=True)
class ModelOptMixedRecipe:
    """ModelOpt ``quant_algo=MIXED_PRECISION`` with per-module FP8 and NVFP4."""

    NAME: ClassVar[str] = "modelopt_mixed"
    # NVFP4 and static FP8 serve, with LoRA adapters, but do not train.
    TRAINS: ClassVar[bool] = False
    # Module -> "FP8" or "NVFP4".
    algorithms: Mapping[str, str]
    # Glob patterns of the modules left unquantized (``ignore``, ``exclude_modules``).
    excluded: tuple[str, ...]

    @classmethod
    def parse(cls, documents: Documents) -> Self | None:
        declared = documents["config.json"].get("quantization_config")
        if not isinstance(declared, Mapping) or declared.get("quant_method") != "modelopt":
            if _DOCUMENT in documents:
                raise ValueError(f"{_DOCUMENT} declares a ModelOpt export that config.json lacks")
            return None
        if declared.get("quant_algo") != _ALGORITHM:
            return None
        where = "config.json quantization_config"
        _fields(
            declared,
            where,
            frozenset(
                {"quant_method", "quant_algo", "producer", "config_groups", "quantized_layers"}
            ),
            frozenset({"ignore"}),
        )
        if _DOCUMENT not in documents:
            raise ValueError(f"a ModelOpt {_ALGORITHM} checkpoint needs {_DOCUMENT}")
        document = _fields(documents[_DOCUMENT], _DOCUMENT, frozenset({"producer", "quantization"}))
        quantization = _fields(
            document["quantization"],
            f"{_DOCUMENT} quantization",
            frozenset({"quant_algo", "quantized_layers"}),
            frozenset({"kv_cache_quant_algo", "exclude_modules"}),
        )
        if quantization["quant_algo"] != _ALGORITHM:
            raise ValueError(
                f"{_DOCUMENT} quant_algo={quantization['quant_algo']!r} differs from "
                f"config.json's {_ALGORITHM}"
            )
        if (kv_cache := quantization.get("kv_cache_quant_algo")) is not None:
            raise ValueError(
                f"{_DOCUMENT} kv_cache_quant_algo={kv_cache!r}: the KV cache stays unquantized"
            )
        if _producer(declared["producer"], f"{where} producer") != _producer(
            document["producer"], f"{_DOCUMENT} producer"
        ):
            raise ValueError(f"{where} producer differs from {_DOCUMENT}'s")
        sources = {
            "config_groups": _groups(declared["config_groups"]),
            "config.json quantized_layers": _layers(
                declared["quantized_layers"], f"{where} quantized_layers"
            ),
            f"{_DOCUMENT} quantized_layers": _layers(
                quantization["quantized_layers"], f"{_DOCUMENT} quantized_layers"
            ),
        }
        algorithms = sources["config_groups"]
        for module in sorted(set().union(*sources.values())):
            claims = {source: modules.get(module) for source, modules in sources.items()}
            if len(set(claims.values())) > 1:
                raise ValueError(f"{module}: the declarations disagree: {claims}")
        ignore = _names(declared.get("ignore", []), f"{where} ignore")
        excluded = _names(quantization.get("exclude_modules", []), f"{_DOCUMENT} exclude_modules")
        if set(ignore) != set(excluded):
            raise ValueError(
                f"{where} ignore {ignore!r} differs from {_DOCUMENT} exclude_modules {excluded!r}"
            )
        return cls(algorithms, tuple(excluded))

    def verify(self, index: CheckpointIndex) -> None:
        """Each listed module has a weight, no excluded pattern names one, and only listed
        modules hold quantized tensors."""

        for module, algorithm in self.algorithms.items():
            if f"{module}.weight" not in index.tensors:
                raise ValueError(f"{module} is listed as {algorithm} but has no weight")
            if pattern := next((p for p in self.excluded if fnmatch.fnmatch(module, p)), None):
                raise ValueError(f"{module} is listed as {algorithm} and excluded by {pattern!r}")
        for name, header in index.tensors.items():
            module = name.rpartition(".")[0]
            if header.dtype in _QUANTIZED_DTYPES and module not in self.algorithms:
                raise ValueError(f"{name} is {header.dtype}, but no declaration lists {module}")

    def method_for(self, part: CheckpointPart) -> type[LinearMethod]:
        """``Fp8StaticLinear`` for a module listed FP8, ``Nvfp4Linear`` for one listed NVFP4,
        each holding exactly the tensors its format stores."""

        algorithm = self.algorithms.get(part.module)
        if algorithm is None:
            raise ValueError(f"{part.module} holds quantized tensors but no declaration lists it")
        found = {leaf: (header.dtype, header.shape) for leaf, header in part.tensors.items()}
        weight = part.tensors.get("weight")
        if weight is None or len(weight.shape) != 2:
            raise ValueError(f"{part.module}: {algorithm} needs a 2-D weight, not {found}")
        rows, cols = weight.shape
        if algorithm == "FP8":
            expected = {"weight": ("F8_E4M3", (rows, cols))} | _STATIC_SCALES
            method: type[LinearMethod] = Fp8StaticLinear
        else:
            expected = {
                "weight": ("U8", (rows, cols)),
                "weight_scale": ("F8_E4M3", (rows, cols * 2 // 16)),
                "weight_scale_2": ("F32", ()),
                "input_scale": ("F32", ()),
            }
            method = Nvfp4Linear
        if found != expected:
            raise ValueError(f"{part.module}: {algorithm} stores {expected}, not {found}")
        return method

    def experts_method_for(self, part: CheckpointPart) -> type[ExpertsMethod]:
        """Routed experts in this convention are not served yet: refused by name."""

        algorithm = self.algorithms.get(part.module, "quantized")
        raise ValueError(
            f"{part.module}: {algorithm} routed experts are not supported yet "
            "(ModelOpt mixed precision serves linear layers only)"
        )
