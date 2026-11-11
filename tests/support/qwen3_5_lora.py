"""PEFT-layout Qwen3.5 LoRA adapters for tests, and their mapping to the engine's layout.

PEFT 0.19.1 (``tuners/lora/layer.py``) gives each targeted ``nn.Linear`` ``lora_A [rank, in]``
and ``lora_B [out, rank]``; the engine keeps one A per checkpoint module and one B per output
slice (``_PLACEMENTS``). The MoE shared expert's linears adapt the same way; the routed
experts' stacked factors are ``tests.support.expert_lora``'s. Written from the PEFT sources
above; no PEFT code is copied."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cti.models.mapping import RowSplit
from cti.models.qwen3_5.config import Qwen3_5TextConfig
from cti.models.qwen3_5_moe.config import Qwen3_5MoeTextConfig

# Where Transformers puts the text decoder layers in each checkpoint layout.
IMAGE_TEXT_PREFIX = "model.language_model."
TEXT_ONLY_PREFIX = "model."


@dataclass(frozen=True)
class _Placement:
    """Where one checkpoint module's A and B live in the engine adapter."""

    projection: str
    a: str
    slices: tuple[str, ...]


# Keyed by Transformers' module name inside a decoder layer.
_PLACEMENTS = {
    "self_attn.q_proj": _Placement("self_attn.qkv_gate_proj", "q_gate", ("q", "gate")),
    "self_attn.k_proj": _Placement("self_attn.qkv_gate_proj", "k", ("k",)),
    "self_attn.v_proj": _Placement("self_attn.qkv_gate_proj", "v", ("v",)),
    "self_attn.o_proj": _Placement("self_attn.o_proj", "full", ("full",)),
    "linear_attn.in_proj_qkv": _Placement("linear_attn.in_proj", "qkv", ("q", "k", "v")),
    "linear_attn.in_proj_z": _Placement("linear_attn.in_proj", "z", ("z",)),
    "linear_attn.in_proj_b": _Placement("linear_attn.in_proj", "b", ("b",)),
    "linear_attn.in_proj_a": _Placement("linear_attn.in_proj", "a", ("a",)),
    "linear_attn.out_proj": _Placement("linear_attn.out_proj", "full", ("full",)),
    "mlp.gate_proj": _Placement("mlp.gate_up_proj", "gate", ("gate",)),
    "mlp.up_proj": _Placement("mlp.gate_up_proj", "up", ("up",)),
    "mlp.down_proj": _Placement("mlp.down_proj", "full", ("full",)),
    "mlp.shared_expert.gate_proj": _Placement("mlp.shared_expert.gate_up_proj", "gate", ("gate",)),
    "mlp.shared_expert.up_proj": _Placement("mlp.shared_expert.gate_up_proj", "up", ("up",)),
    "mlp.shared_expert.down_proj": _Placement("mlp.shared_expert.down_proj", "full", ("full",)),
}
# The checkpoint modules each engine LoRA target adapts.
TARGET_MODULES = {
    "attention": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
    "linear_attention": (
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_a",
        "linear_attn.out_proj",
    ),
    "mlp": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    # And each routed expert, whose stacked factors these PEFT states leave out.
    "experts": (
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
    ),
}

_PEFT_KEY = re.compile(
    r"^base_model\.model\.(?P<prefix>.*?)layers\.(?P<layer>\d+)\."
    r"(?P<module>\w+(?:\.\w+)+)\.lora_(?P<factor>[AB])\.weight$"
)
_ENGINE_KEY = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.(?P<projection>\w+(?:\.\w+)+)\.(?P<name>\w+)"
    r"\.lora_(?P<factor>[AB])$"
)
_BY_A = {(place.projection, place.a): module for module, place in _PLACEMENTS.items()}
_BY_B = {
    (place.projection, name): module
    for module, place in _PLACEMENTS.items()
    for name in place.slices
}


class Qwen3_5LoRAMapping:
    """Convert one Qwen3.5 geometry's adapter state between PEFT's layout and the engine's;
    dense and MoE mixers adapt alike."""

    def __init__(self, config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig, *, prefix: str) -> None:
        linear = config.linear_attention
        key_size = linear.num_key_heads * linear.key_head_dim
        self._qkv_sizes = (key_size, key_size, linear.num_value_heads * linear.value_head_dim)
        self._head_dim = config.attention.head_dim
        self._prefix = prefix

    def to_engine(self, peft_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Name and split PEFT's tensors as the engine's ``Adapter.state_dict()``."""

        engine_state: dict[str, torch.Tensor] = {}
        for key, tensor in peft_state.items():
            match = _PEFT_KEY.match(key)
            if match is None or match["prefix"] != self._prefix:
                raise ValueError(f"{key} adapts no module of the engine's text layers")
            place = _PLACEMENTS.get(match["module"])
            if place is None:
                raise ValueError(f"{key} adapts {match['module']}, which no LoRA target wraps")
            stem = f"model.layers.{match['layer']}.{place.projection}"
            if match["factor"] == "A":
                engine_state[f"{stem}.{place.a}.lora_A"] = tensor
                continue
            for name, part in zip(place.slices, self._split_b(place, tensor), strict=True):
                engine_state[f"{stem}.{name}.lora_B"] = part
        return engine_state

    def to_peft(self, engine_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Name and join the engine's tensors as PEFT saves them."""

        modules: dict[tuple[str, str], None] = {}
        for key in engine_state:
            match = _ENGINE_KEY.match(key)
            lookup = _BY_A if match is not None and match["factor"] == "A" else _BY_B
            module = None if match is None else lookup.get((match["projection"], match["name"]))
            if module is None:
                raise ValueError(f"{key} is no Qwen3.5 adapter tensor")
            modules[(match["layer"], module)] = None

        peft_state: dict[str, torch.Tensor] = {}
        for layer, module in modules:
            place = _PLACEMENTS[module]
            stem = f"model.layers.{layer}.{place.projection}"
            peft_stem = f"base_model.model.{self._prefix}layers.{layer}.{module}"
            peft_state[f"{peft_stem}.lora_A.weight"] = engine_state[f"{stem}.{place.a}.lora_A"]
            peft_state[f"{peft_stem}.lora_B.weight"] = self._join_b(
                place, [engine_state[f"{stem}.{name}.lora_B"] for name in place.slices]
            )
        return peft_state

    def _split_b(self, place: _Placement, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if place.slices == ("q", "gate"):
            return RowSplit((self._head_dim, self._head_dim)).split(b)
        if place.slices == ("q", "k", "v"):
            return b.split(self._qkv_sizes)
        return (b,)

    def _join_b(self, place: _Placement, parts: list[torch.Tensor]) -> torch.Tensor:
        if place.slices == ("q", "gate"):
            query, gate = (part.unflatten(0, (-1, self._head_dim)) for part in parts)
            return torch.stack((query, gate), dim=1).flatten(0, 2)
        return torch.cat(parts)


class Qwen3_5CanonicalLoRALayout:
    """The LoRA backward oracle's canonical layout of a Qwen3.5 text-only adapter.

    Canonical names are PEFT's saved names without ``base_model.model.`` and ``.weight``
    (``model.layers.N.<parent>.<module>.lora_A``): one A and one B per checkpoint module.
    Into the engine, each A maps onto its group's A and each B splits like the weight;
    back, the gate's B (and its gradient) re-interleaves with the query's, and a shared A
    carries the gradient summed over its slices.
    """

    _PEFT_PREFIX = "base_model.model."
    _PEFT_SUFFIX = ".weight"

    def __init__(self, config: Qwen3_5TextConfig) -> None:
        self._mapping = Qwen3_5LoRAMapping(config, prefix=TEXT_ONLY_PREFIX)

    def to_engine(self, canonical: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._mapping.to_engine(
            {
                f"{self._PEFT_PREFIX}{name}{self._PEFT_SUFFIX}": tensor
                for name, tensor in canonical.items()
            }
        )

    def to_canonical(
        self,
        engine: Mapping[str, torch.Tensor],
        names: Iterable[str],
    ) -> dict[str, torch.Tensor]:
        """The canonical tensors of ``engine`` in the order of ``names``."""

        peft = self._mapping.to_peft(engine)
        return {name: peft[f"{self._PEFT_PREFIX}{name}{self._PEFT_SUFFIX}"] for name in names}


def random_peft_state(
    checkpoint: Mapping[str, torch.Tensor],
    *,
    prefix: str,
    targets: Iterable[str],
    rank: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """A non-zero PEFT adapter of ``rank`` for each module of ``targets`` in ``checkpoint``.

    A and B are scaled like the weights (unit-variance rows over the fan-in), so
    ``B @ A`` is about as large as the weight it updates.
    """

    modules = {module for target in targets for module in TARGET_MODULES[target]}
    layer_weight = re.compile(
        rf"^{re.escape(prefix)}layers\.\d+\.(?P<module>\w+(?:\.\w+)+)\.weight$"
    )
    state: dict[str, torch.Tensor] = {}
    for name, weight in checkpoint.items():
        match = layer_weight.match(name)
        if match is None or match["module"] not in modules:
            continue
        out_features, in_features = weight.shape
        stem = "base_model.model." + name.removesuffix(".weight")
        state[f"{stem}.lora_A.weight"] = (
            torch.randn(rank, in_features, generator=generator) * in_features**-0.5
        )
        state[f"{stem}.lora_B.weight"] = (
            torch.randn(out_features, rank, generator=generator) * rank**-0.5
        )
    return state


class _PeftLinear(nn.Module):
    """One checkpoint ``nn.Linear`` with PEFT's adapter: ``base(x) + B(A(x)) * scaling``."""

    def __init__(self, base: nn.Linear, a: torch.Tensor, b: torch.Tensor, scaling: float) -> None:
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.Parameter(a.clone())
        self.lora_B = nn.Parameter(b.clone())
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def attach_peft_lora(
    model: nn.Module,
    peft_state: Mapping[str, torch.Tensor],
    *,
    scaling: float,
) -> dict[str, nn.Parameter]:
    """Adapt ``model``'s modules in place with ``peft_state`` and freeze every base weight.

    Returns the adapter parameters under PEFT's saved names, in ``peft_state``'s order.
    """

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters: dict[str, nn.Parameter] = {}
    for key in peft_state:
        if not key.endswith(".lora_A.weight"):
            continue
        module_name = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        b_key = f"base_model.model.{module_name}.lora_B.weight"
        parent_name, _, leaf = module_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        adapted = _PeftLinear(getattr(parent, leaf), peft_state[key], peft_state[b_key], scaling)
        setattr(parent, leaf, adapted)
        parameters[key] = adapted.lora_A
        parameters[b_key] = adapted.lora_B
    return parameters


def merged_checkpoint(
    checkpoint: Mapping[str, torch.Tensor],
    peft_state: Mapping[str, torch.Tensor],
    *,
    scaling: float,
) -> dict[str, torch.Tensor]:
    """``checkpoint`` with PEFT's merge, ``W + (B @ A) * scaling``, in every adapted module."""

    merged = dict(checkpoint)
    for key, a in peft_state.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        b = peft_state[f"base_model.model.{module}.lora_B.weight"]
        weight = merged[f"{module}.weight"]
        merged[f"{module}.weight"] = weight + (b @ a).to(weight.dtype) * scaling
    return merged
