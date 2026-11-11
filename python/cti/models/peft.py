"""PEFT adapters in the engine's LoRA, as PEFT 0.19.1 saves them for Transformers 5.16.1.

``peft_lora`` reads an adapter's configuration, ``load_peft`` its factors and ``save_peft``
writes a trained adapter as one; ``PeftLayout`` is the one map between PEFT's factor keys and the adapter's factors, both ways:

- a linear module's ``base_model.model.<module>.lora_A.weight`` ``[r, in]`` and ``lora_B``
  ``[out, r]``, through the model's checkpoint mapping: its A is the A of every output slice
  its weight becomes, its B splits as the weight's rows do (and joins back);
- Transformers 4's per-expert modules ``<prefix>.experts.<i>.<projection>`` by their index in
  the expert stacks (``ExpertLoRALayout.per_projection``: their gate and up have their own A);
- Transformers 5's fused expert parameters ``gate_up_proj`` ``[E, 2I, H]`` and ``down_proj``
  ``[E, H, I]``, which PEFT adapts through ``target_parameters``: one A ``[E·r, in]``
  (expert-major rows) and one B ``[out, r·E]`` (rank-major columns) each, the parameters
  wrapped one in another in the experts module's own parameter order whatever order
  ``target_parameters`` lists (and saves, sorted) them in: ``gate_up_proj`` innermost, so
  with both it is ``<prefix>.experts.base_layer`` and ``down_proj`` is ``<prefix>.experts``
  (``ExpertLoRALayout.shared_a``; PEFT's ``utils/transformers_weight_conversion.py``: "the
  gate_up_proj is the inner PEFT ParamWrapper").

Written from PEFT 0.19.1's ``tuners/lora/layer.py`` (``ParamWrapper``) and Transformers
5.16.1's Qwen3.5-MoE experts; no code is copied.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import safetensors
import safetensors.torch
import torch

from cti.config import EXPERT_LORA_RANKS, ExpertLoRALayout
from cti.utils import init_logger

from .mapping import EXPERT_NAME

if TYPE_CHECKING:
    from cti.config import LoRAConfig, LoRATarget
    from cti.layers import Adapter

    from .base import BaseLLMModel

logger = init_logger(__name__)

# The LoRA target of each checkpoint module by name; a sparse model's MLPs are "experts".
_TARGETS = MappingProxyType(
    {
        **dict.fromkeys(("q_proj", "k_proj", "v_proj", "o_proj"), "attention"),
        **dict.fromkeys(
            ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"), "linear_attention"
        ),
        **dict.fromkeys(("gate_proj", "up_proj", "down_proj"), "mlp"),
    }
)
# Settings beyond B @ A * lora_alpha / r, refused when set.
_REFUSED = (
    "use_dora",
    "use_rslora",
    "modules_to_save",
    "rank_pattern",
    "alpha_pattern",
    "layers_to_transform",
    "fan_in_fan_out",
)
# A factor key and its module.
_PEFT_KEY = re.compile(r"^base_model\.model\.(?P<module>.+)\.lora_(?P<factor>[AB])\.weight$")
_BASE_LAYER = ".base_layer"
# Transformers 5's fused expert parameters, and the per-expert checkpoint projections each
# stacks, gate rows first.
EXPERT_PARAMETERS = MappingProxyType(
    {"gate_up_proj": ("gate_proj", "up_proj"), "down_proj": ("down_proj",)}
)
ADAPTER_CONFIG, ADAPTER_WEIGHTS = "adapter_config.json", "adapter_model.safetensors"


def peft_key(module: str, factor: str) -> str:
    """The key of ``module``'s ``factor`` (``"A"`` or ``"B"``)."""

    return f"base_model.model.{module}.lora_{factor}.weight"


def peft_lora(config: LoRAConfig, declared: Sequence[LoRATarget]) -> LoRAConfig:
    """``config`` with the targets (of a model's ``declared`` ones), the rank, the alpha and
    the expert layout of its adapter: ``shared_a`` for Transformers 5's fused expert
    parameters (or no routed experts), ``per_projection`` for Transformers 4's per-expert
    modules. A per-expert rank pads with zeros to a multiple of 8, alpha keeping the
    adapter's scaling."""

    with open(os.path.join(config.adapter, ADAPTER_CONFIG), encoding="utf-8") as file:
        peft = json.load(file)
    refused = [field for field in _REFUSED if peft.get(field)]
    if peft.get("bias", "none") != "none":
        refused.append(f"bias {peft['bias']!r}")
    if not isinstance(modules := peft.get("target_modules"), list):
        refused.append("target_modules other than a list of names")
    if not isinstance(parameters := peft.get("target_parameters") or [], list):
        refused.append("target_parameters other than a list of names")
    if refused:
        raise ValueError(f"PEFT adapter {config.adapter}: unsupported {', '.join(refused)}")
    targets = set()
    for module in modules:
        target = _TARGETS.get(module.rsplit(".", 1)[-1])
        target = "experts" if target == "mlp" and "experts" in declared else target
        if target not in declared:
            raise ValueError(f"PEFT adapter {config.adapter}: no LoRA target adapts {module}")
        targets.add(target)
    for parameter in parameters:
        if parameter.rsplit(".", 1)[-1] not in EXPERT_PARAMETERS or "experts" not in declared:
            raise ValueError(f"PEFT adapter {config.adapter}: no LoRA target adapts {parameter}")
        targets.add("experts")
    layout = ExpertLoRALayout.SHARED_A
    if "experts" in targets and not parameters and _per_expert_modules(config.adapter):
        layout = ExpertLoRALayout.PER_PROJECTION
    rank, step = peft["r"], EXPERT_LORA_RANKS.step
    padded = -(-rank // step) * step if "experts" in targets else rank
    return replace(
        config,
        targets=tuple(target for target in declared if target in targets),
        rank=padded,
        alpha=peft["lora_alpha"] * padded / rank,
        expert_layout=layout,
    )


def _per_expert_modules(path: str) -> bool:
    """Whether the adapter at ``path`` adapts Transformers 4's per-expert modules."""

    with safetensors.safe_open(os.path.join(path, ADAPTER_WEIGHTS), "pt") as file:
        return any(
            (match := _PEFT_KEY.match(key)) and EXPERT_NAME.match(match["module"])
            for key in file.keys()
        )


class _Factors(Protocol):
    """One adapted weight's factor names by slot (``layers.LoRAModule``)."""

    a: tuple[str, ...]
    b: tuple[str, ...]
    experts: int | None
    slot_major_b: bool


@dataclass(frozen=True, slots=True)
class _Slot:
    """One output slot of an adapted weight: its ``LoRAModule`` and its slot index."""

    module: _Factors
    slot: int

    @property
    def a(self) -> str:
        return self.module.a[self.slot]

    @property
    def b(self) -> str:
        return self.module.b[self.slot]


class PeftLayout:
    """Where each PEFT factor of ``model``'s adapter sits among ``adapters``' factors (the
    module docstring's three forms), both ways: ``load`` places PEFT factors into whole
    (logical) factors by name; ``save`` takes whole factors back to PEFT factors.
    ``target_parameters`` names the fused expert parameters the adapter adapts, in any
    order: the adapter's (``load``), or those ``save`` writes."""

    def __init__(
        self,
        model: BaseLLMModel,
        adapters: Adapter,
        target_parameters: Sequence[str] = tuple(EXPERT_PARAMETERS),
    ) -> None:
        self._mapping = model.checkpoint_mapping
        self._modules = adapters.modules()
        self._declared = frozenset(
            stem if module.experts else f"{stem}.weight" for stem, module in self._modules.items()
        )
        named = {name.rsplit(".", 1)[-1] for name in target_parameters}
        # Innermost first: the experts module's parameter order.
        self._parameters = tuple(name for name in EXPERT_PARAMETERS if name in named)

    # --- PEFT -> engine ---------------------------------------------------------------------

    def load(self, tensors: Mapping[str, torch.Tensor], source: str) -> dict[str, torch.Tensor]:
        """Whole factors by name from PEFT factors by key; a factor no key reaches is absent
        (``Adapter.load``: a B zero, an A as initialized)."""

        state: dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            match = _PEFT_KEY.match(key)
            if match is None:
                raise ValueError(f"PEFT adapter {source}: {key} is no LoRA factor")
            module, is_a = match["module"], match["factor"] == "A"
            fused = self._fused(module)
            if fused is not None:
                self._load_fused(state, *fused, tensor, is_a)
                continue
            weight = f"{module}.weight"
            placements = self._mapping.placements(weight, self._declared)
            if not placements:
                raise ValueError(
                    f"PEFT adapter {source}: {key} is no LoRA factor of an adapted module"
                )
            parts = (tensor,) * len(placements) if is_a else self._mapping.split(weight, tensor)
            for placement, part in zip(placements, parts, strict=True):
                slot = _Slot(self._modules[placement.name.removesuffix(".weight")], placement.slot)
                expert = () if placement.expert is None else (placement.expert,)
                self._place(state, slot, part, is_a, expert)
        return state

    def _load_fused(
        self,
        state: dict[str, torch.Tensor],
        prefix: str,
        parameter: str,
        tensor: torch.Tensor,
        is_a: bool,
    ) -> None:
        slots = self._expert_slots(prefix, parameter)
        experts = slots[0].module.experts
        if is_a:
            a = tensor.reshape(experts, -1, tensor.shape[-1])
            for slot in slots:
                self._place(state, slot, a, True, ())
            return
        # [out, r·E], rank-major columns -> [E, out, r], the output rows split per slot.
        b = tensor.reshape(tensor.shape[0], -1, experts).permute(2, 0, 1)
        for slot, part in zip(slots, b.chunk(len(slots), dim=1), strict=True):
            self._place(state, slot, part, False, ())

    def _place(
        self,
        state: dict[str, torch.Tensor],
        slot: _Slot,
        part: torch.Tensor,
        is_a: bool,
        expert: tuple[int, ...],
    ) -> None:
        """Put ``part`` at its slot's (and ``expert``'s) index of its whole factor."""

        module = slot.module
        name = slot.a if is_a else slot.b
        slot_major = not is_a and module.slot_major_b
        index = ((slot.slot,) if slot_major else ()) + expert
        if not index:
            state[name] = part
            return
        stack = state.get(name)
        if stack is None:
            leading = ((len(module.b),) if slot_major else ()) + (
                (module.experts,) if expert else ()
            )
            stack = state[name] = part.new_zeros(*leading, *part.shape)
        stack[index] = part

    # --- engine -> PEFT ---------------------------------------------------------------------

    def save(
        self, state: Mapping[str, torch.Tensor], checkpoint_names: Iterable[str]
    ) -> tuple[dict[str, torch.Tensor], tuple[str, ...], tuple[str, ...]]:
        """PEFT factors by key from whole factors by name, and the ``target_modules`` and
        ``target_parameters`` (sorted, as PEFT saves them) they adapt: each linear checkpoint weight in ``checkpoint_names`` that an adapted module
        takes, and the fused parameters of each experts module whose stacks it names (fused,
        as Transformers 5 stores them, or per expert). Only the shared-A expert layout
        saves: per-projection gate and up A have no PEFT form."""

        tensors: dict[str, torch.Tensor] = {}
        target_modules: set[str] = set()
        experts_modules: set[str] = set()
        for name in checkpoint_names:
            module, _, leaf = name.rpartition(".")
            if leaf in EXPERT_PARAMETERS and module.endswith(".experts"):
                experts_modules.add(module)  # Transformers 5's fused expert stacks
                continue
            if leaf != "weight":
                continue
            try:
                placements = self._mapping.placements(name, self._declared)
            except ValueError:  # a weight no adapted module takes
                continue
            if not placements:
                continue
            if placements[0].expert is not None:  # a per-expert checkpoint weight
                experts_modules.add(EXPERT_NAME.match(module)["prefix"])
                continue
            slots = [
                _Slot(self._modules[p.name.removesuffix(".weight")], p.slot) for p in placements
            ]
            tensors[peft_key(module, "A")] = self._shared_a(slots, state, module)
            parts = [self._b(state, slot, ()) for slot in slots]
            tensors[peft_key(module, "B")] = self._mapping.join(name, parts)
            target_modules.add(module.rsplit(".", 1)[-1])
        for prefix in sorted(experts_modules):
            for depth, parameter in enumerate(reversed(self._parameters)):
                slots = self._expert_slots(prefix, parameter)
                module = prefix + _BASE_LAYER * depth
                a = self._shared_a(slots, state, f"{prefix}.{parameter}")
                tensors[peft_key(module, "A")] = a.flatten(0, 1)
                # [E, out, r] -> [out, r·E], rank-major columns.
                b = torch.cat([self._b(state, slot, ()) for slot in slots], dim=1)
                tensors[peft_key(module, "B")] = b.permute(1, 2, 0).flatten(1, 2)
        # PEFT's relative names: the experts module's parent and itself, then the parameter.
        target_parameters = {
            f"{'.'.join(prefix.split('.')[-2:])}.{parameter}"
            for prefix in experts_modules
            for parameter in self._parameters
        }
        return tensors, tuple(sorted(target_modules)), tuple(sorted(target_parameters))

    def _shared_a(
        self, slots: Sequence[_Slot], state: Mapping[str, torch.Tensor], module: str
    ) -> torch.Tensor:
        names = {slot.a for slot in slots}
        if len(names) != 1:
            raise ValueError(
                f"{module}: its slices have their own A ({', '.join(sorted(names))}), which no "
                "PEFT module holds"
            )
        return state[names.pop()]

    def _b(
        self, state: Mapping[str, torch.Tensor], slot: _Slot, expert: tuple[int, ...]
    ) -> torch.Tensor:
        tensor = state[slot.b]
        index = ((slot.slot,) if slot.module.slot_major_b else ()) + expert
        return tensor[index] if index else tensor

    # --- shared -----------------------------------------------------------------------------

    def _fused(self, module: str) -> tuple[str, str] | None:
        """``(experts module, fused parameter)`` of a module path PEFT reaches through
        ``target_parameters``, or None."""

        if not self._parameters:
            return None
        prefix, depth = module, 0
        while prefix.endswith(_BASE_LAYER):
            prefix, depth = prefix.removesuffix(_BASE_LAYER), depth + 1
        index = len(self._parameters) - 1 - depth
        if not prefix.endswith(".experts") or not 0 <= index < len(self._parameters):
            return None
        return prefix, self._parameters[index]

    def _expert_slots(self, prefix: str, parameter: str) -> list[_Slot]:
        """The slots the fused ``parameter`` of the experts module ``prefix`` stacks, by its
        per-expert checkpoint projections placed through the mapping."""

        slots = []
        for projection in EXPERT_PARAMETERS[parameter]:
            (placement,) = self._mapping.placements(
                f"{prefix}.0.{projection}.weight", self._declared
            )
            slots.append(_Slot(self._modules[placement.name], placement.slot))
        return slots


def load_peft(model: BaseLLMModel, adapters: Adapter, path: str) -> None:
    """Load the PEFT adapter in ``path`` into ``adapters``, attached to ``model``, through
    ``PeftLayout``. A slice the adapter leaves out is zero."""

    with open(os.path.join(path, ADAPTER_CONFIG), encoding="utf-8") as file:
        parameters = json.load(file).get("target_parameters") or ()
    tensors = safetensors.torch.load_file(os.path.join(path, ADAPTER_WEIGHTS))
    state = PeftLayout(model, adapters, parameters).load(tensors, path)
    zero = {
        re.sub(r"\.layers\.\d+\.", ".layers.*.", b)
        for module in adapters.modules().values()
        for b in module.b
        if b not in state
    }
    if zero:
        logger.info_rank0("PEFT adapter %s leaves %s zero", path, ", ".join(sorted(zero)))
    adapters.load(state)


def save_peft(
    path: str,
    model: BaseLLMModel,
    adapters: Adapter,
    state: Mapping[str, torch.Tensor],
    checkpoint_names: Iterable[str],
) -> None:
    """Write ``adapters``' whole factors ``state`` (by name) as a PEFT adapter in the folder
    ``path``, through ``PeftLayout``: ``adapter_config.json`` and ``adapter_model.safetensors``,
    each replaced atomically. ``checkpoint_names`` are the base checkpoint's tensors, whose
    adapted weights the adapter's modules name. A per-projection expert adapter has no PEFT
    form and raises."""

    tensors, target_modules, target_parameters = PeftLayout(model, adapters).save(
        state, checkpoint_names
    )
    config = adapters.config
    peft = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": config.rank,
        "lora_alpha": config.alpha,
        "target_modules": list(target_modules),
        # None, PEFT's default, without fused parameters.
        "target_parameters": list(target_parameters) or None,
        "bias": "none",
        "lora_dropout": 0.0,
    }
    os.makedirs(path, exist_ok=True)
    _replace(
        path,
        ADAPTER_WEIGHTS,
        lambda file: safetensors.torch.save_file(
            {key: tensor.contiguous() for key, tensor in tensors.items()}, file
        ),
    )
    _replace(path, ADAPTER_CONFIG, lambda file: _dump_json(peft, file))


def _dump_json(value: object, file: str) -> None:
    with open(file, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def _replace(folder: str, name: str, write: Callable[[str], None]) -> None:
    """Write ``folder/name`` through a temporary file in ``folder`` and replace it at once."""

    descriptor, temporary = tempfile.mkstemp(dir=folder, prefix=f".{name}.", suffix=".tmp")
    os.close(descriptor)
    try:
        write(temporary)
        os.replace(temporary, os.path.join(folder, name))
    except BaseException:
        os.unlink(temporary)
        raise


__all__ = [
    "ADAPTER_CONFIG",
    "ADAPTER_WEIGHTS",
    "EXPERT_PARAMETERS",
    "PeftLayout",
    "load_peft",
    "peft_key",
    "peft_lora",
    "save_peft",
]
