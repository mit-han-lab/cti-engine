"""Per-expert LoRA for tests, for any sparse block whose routed experts stack gate rows first
(Qwen3-MoE, Qwen3.5-MoE): stacked factors named as ``ExpertLoRAFactors`` names them, their
FP32 merge into the expert weights, the grouped GEMMs' slot layout, and Transformers'
eager expert loop with an unmerged LoRA on each expert's fused gate/up projection and its down
projection, PEFT's ``base(x) + B(A(x)) * scaling``. Written from transformers 5.16.1 ``Qwen3_5MoeExperts`` and
PEFT 0.19.1 ``tuners/lora/layer.py``; no code is copied."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from cti.layers import Placement

# ``ExpertLoRAFactors``' names under its key: the shared gate/up A ``[E, r, H]`` and the
# slice-major B ``[2, E, I, r]`` (gate then up), then the down A ``[E, r, I]`` and B ``[E, H, r]``.
FACTOR_NAMES = (
    "gate_up_proj.gate_up.lora_A",
    "gate_up_proj.gate_up.lora_B",
    "down_proj.full.lora_A",
    "down_proj.full.lora_B",
)
# The per-projection layout's (an imported Transformers 4 adapter, served only).
PER_PROJECTION_NAMES = tuple(
    f"{projection}.lora_{factor}"
    for projection in ("gate_up_proj.gate", "gate_up_proj.up", "down_proj.full")
    for factor in "AB"
)


class NamedExpertLoRA:
    """``moe.ExpertLoRA`` over stacked factors by ``FACTOR_NAMES`` or
    ``PER_PROJECTION_NAMES``."""

    def __init__(self, factors: Mapping[str, torch.Tensor], scaling: float) -> None:
        self.scaling = scaling
        names = FACTOR_NAMES if FACTOR_NAMES[0] in factors else PER_PROJECTION_NAMES
        self.factors = {name: factors[name] for name in names}

    @property
    def gate_up(self) -> tuple[torch.Tensor, ...]:
        if FACTOR_NAMES[0] not in self.factors:
            return tuple(self.factors.values())[:4]
        a, b = (
            self.factors["gate_up_proj.gate_up.lora_A"],
            self.factors["gate_up_proj.gate_up.lora_B"],
        )
        return a, b[0], a, b[1]

    @property
    def gate_up_shared(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if FACTOR_NAMES[0] not in self.factors:
            return None
        return self.factors[FACTOR_NAMES[0]], self.factors[FACTOR_NAMES[1]]

    @property
    def down(self) -> tuple[torch.Tensor, ...]:
        return self.factors["down_proj.full.lora_A"], self.factors["down_proj.full.lora_B"]

    def training(self) -> NamedExpertLoRA:
        return self

    def window(self, placement: Placement) -> NamedExpertLoRA:
        """The factors ``placement`` holds of this whole adapter, each in storage of its own
        as a rank materializes them."""

        windows = factor_windows(placement)
        return NamedExpertLoRA(
            {name: tensor[windows[name]].contiguous() for name, tensor in self.factors.items()},
            self.scaling,
        )


def factor_windows(placement: Placement) -> dict[str, tuple[slice, ...]]:
    """Each factor's index of the whole adapter that ``placement`` holds: its experts, and
    its intermediate window of the gate and up B rows and the down A columns."""

    held = slice(placement.experts.start, placement.experts.stop)
    width = slice(placement.intermediate.start, placement.intermediate.stop)
    return {
        # The shared-A layout's gate/up factors, and the per-projection layout's.
        "gate_up_proj.gate_up.lora_A": (held,),
        "gate_up_proj.gate_up.lora_B": (slice(None), held, width),
        "gate_up_proj.gate.lora_A": (held,),
        "gate_up_proj.up.lora_A": (held,),
        "gate_up_proj.gate.lora_B": (held, width),
        "gate_up_proj.up.lora_B": (held, width),
        "down_proj.full.lora_A": (held, slice(None), width),
        "down_proj.full.lora_B": (held,),
    }


def random_expert_factors(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    rank: int,
    *,
    generator: torch.Generator,
    per_projection: bool = False,
) -> dict[str, torch.Tensor]:
    """Non-zero FP32 factors of ``num_experts`` experts by ``FACTOR_NAMES`` (or
    ``PER_PROJECTION_NAMES``), scaled like the weights (unit-variance rows over the fan-in), so
    ``B @ A`` is about a weight's size."""

    def draw(*shape: int, fan_in: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * fan_in**-0.5

    if per_projection:
        gate_up = {
            f"gate_up_proj.{half}.lora_{factor}": (
                draw(num_experts, rank, hidden_size, fan_in=hidden_size)
                if factor == "A"
                else draw(num_experts, intermediate_size, rank, fan_in=rank)
            )
            for half in ("gate", "up")
            for factor in "AB"
        }
    else:
        gate_up = {
            "gate_up_proj.gate_up.lora_A": draw(num_experts, rank, hidden_size, fan_in=hidden_size),
            "gate_up_proj.gate_up.lora_B": draw(
                2, num_experts, intermediate_size, rank, fan_in=rank
            ),
        }
    return {
        **gate_up,
        "down_proj.full.lora_A": draw(
            num_experts, rank, intermediate_size, fan_in=intermediate_size
        ),
        "down_proj.full.lora_B": draw(num_experts, hidden_size, rank, fan_in=rank),
    }


def random_layer_adapter(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    shared_size: int,
    rank: int,
    *,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """A non-zero FP32 adapter of a whole sparse block by ``Adapter`` name: the routed
    experts' factors and the shared expert's, scaled as ``random_expert_factors``'."""

    routed = random_expert_factors(
        num_experts, hidden_size, intermediate_size, rank, generator=generator
    )
    adapter = {f"experts.{name}": tensor for name, tensor in routed.items()}
    shapes = {
        "gate_up_proj.gate": (shared_size, hidden_size),
        "gate_up_proj.up": (shared_size, hidden_size),
        "down_proj.full": (hidden_size, shared_size),
    }
    for name, (output_size, input_size) in shapes.items():
        a = torch.randn(rank, input_size, generator=generator) * input_size**-0.5
        adapter[f"shared_expert.{name}.lora_A"] = a
        adapter[f"shared_expert.{name}.lora_B"] = (
            torch.randn(output_size, rank, generator=generator) * rank**-0.5
        )
    return adapter


def layer_adapter_window(
    adapter: Mapping[str, torch.Tensor], placement: Placement, rank: int, size: int
) -> dict[str, torch.Tensor]:
    """Rank ``rank`` of ``size``'s part of a ``random_layer_adapter``: its routed factors'
    windows; the shared expert's B rows and A columns of its window of the shared width."""

    routed = factor_windows(placement)
    shared = adapter["shared_expert.down_proj.full.lora_A"].shape[-1]
    width = slice(rank * shared // size, (rank + 1) * shared // size)
    local = {}
    for name, tensor in adapter.items():
        stem = name.removeprefix("experts.")
        if stem in routed:
            local[name] = tensor[routed[stem]]
        elif name.startswith("shared_expert.gate_up_proj.") and name.endswith("lora_B"):
            local[name] = tensor[width]
        elif name == "shared_expert.down_proj.full.lora_A":
            local[name] = tensor[:, width]
        else:
            local[name] = tensor
    return local


def merged_experts(
    gate_up: torch.Tensor,
    down: torch.Tensor,
    factors: Mapping[str, torch.Tensor],
    *,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 expert stacks with each expert's ``B @ A * scaling`` merged in."""

    if FACTOR_NAMES[0] in factors:
        a, b = (factors[f"gate_up_proj.gate_up.lora_{factor}"].double() for factor in "AB")
        gate_up_update = torch.cat((b[0] @ a, b[1] @ a), 1) * scaling
    else:
        halves = [
            factors[f"gate_up_proj.{half}.lora_B"].double()
            @ factors[f"gate_up_proj.{half}.lora_A"].double()
            for half in ("gate", "up")
        ]
        gate_up_update = torch.cat(halves, 1) * scaling
    down_a, down_b = (factors[f"down_proj.full.lora_{factor}"].double() for factor in "AB")
    merged_gate_up = gate_up.double() + gate_up_update
    merged_down = down.double() + (down_b @ down_a) * scaling
    return merged_gate_up.float(), merged_down.float()


class LoRAExperts(nn.Module):
    """Transformers' eager loop over stacked SiLU experts (``gate_up [E, 2I, H]`` gate rows
    first, ``down [E, H, I]``) with each expert's fused gate/up projection and its down
    projection adapted the PEFT way. ``parameters`` holds the trainable factors by
    ``FACTOR_NAMES``."""

    def __init__(
        self,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        factors: Mapping[str, torch.Tensor],
        *,
        scaling: float,
    ) -> None:
        super().__init__()
        self.register_buffer("gate_up", gate_up.detach().clone())
        self.register_buffer("down", down.detach().clone())
        self.factors = nn.ParameterDict(
            {name.replace(".", "_"): nn.Parameter(factors[name].clone()) for name in FACTOR_NAMES}
        )
        self.scaling = scaling

    @property
    def named_factors(self) -> dict[str, nn.Parameter]:
        return {name: self.factors[name.replace(".", "_")] for name in FACTOR_NAMES}

    def _project(
        self, x: torch.Tensor, weight: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        return F.linear(x, weight) + F.linear(F.linear(x, a), b) * self.scaling

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        factors = self.named_factors
        gate_up_a, gate_up_b = (factors[f"gate_up_proj.gate_up.lora_{f}"] for f in "AB")
        down_a, down_b = (factors[f"down_proj.full.lora_{f}"] for f in "AB")
        for expert in top_k_index.unique().tolist():
            tokens, slots = torch.where(top_k_index == expert)
            x = hidden_states[tokens]
            # PEFT's B over the fused projection's rows: the gate's, then the up's.
            b = torch.cat((gate_up_b[0, expert], gate_up_b[1, expert]))
            gate, up = self._project(x, self.gate_up[expert], gate_up_a[expert], b).chunk(2, -1)
            routed = self._project(
                F.silu(gate) * up, self.down[expert], down_a[expert], down_b[expert]
            )
            output.index_add_(0, tokens, routed * top_k_weights[tokens, slots, None])
        return output


def expert_lora_deltas(lora: object, x: torch.Tensor, expert: int, projection: str) -> torch.Tensor:
    """``x``'s delta under local ``expert``'s factors of a ``moe.ExpertLoRA`` ``lora``:
    ``projection`` is ``"gate_up"`` (gate then up columns) or ``"down"``."""

    if projection == "gate_up":
        gate_a, gate_b, up_a, up_b = lora.gate_up
        pairs = ((gate_a, gate_b), (up_a, up_b))
    else:
        pairs = (lora.down,)
    deltas = [F.linear(F.linear(x, a[expert]), b[expert]) for a, b in pairs]
    return torch.cat(deltas, -1) * lora.scaling


__all__ = [
    "FACTOR_NAMES",
    "LoRAExperts",
    "NamedExpertLoRA",
    "PER_PROJECTION_NAMES",
    "expert_lora_deltas",
    "factor_windows",
    "layer_adapter_window",
    "merged_experts",
    "random_expert_factors",
    "random_layer_adapter",
]
