"""LoRA factors that a layer holds once an adapter attaches to it, and the adapter: the
factors it attaches to a model's projections."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol

import torch
import torch.nn.functional as F

from ctikernels.linear import lora_delta_factors_
from ctikernels.linear.lora import (
    LoRAInferencePlan,
    lora_projection,
    try_prepare_lora_inference_plan,
)
from ctikernels.linear.training import lora_projection_training

from .placement import WeightShard

if TYPE_CHECKING:
    from cti.config import ExpertLoRALayout, LoRAConfig

    from .base import BaseOP
    from .linear import ProjectionSlice
    from .linear_method import FrozenBase
    from .placement import Placement


@dataclass(frozen=True, slots=True)
class LoRAModule:
    """The factor names of one adapted weight's output slices, in output order: slice ``i``
    is ``b[i]`` times ``a[i]``. ``experts`` stacks each factor by that many experts; a name
    repeated in ``a`` is one A the slices share, and with ``slot_major_b`` the slices' B are
    one tensor stacking them on its first axis, slot by slot."""

    a: tuple[str, ...]
    b: tuple[str, ...]
    experts: int | None = None
    slot_major_b: bool = False


@dataclass(frozen=True, slots=True)
class FactorPlacement:
    """Where one rank's factor sits in its logical tensor: the ``window`` it holds (None:
    the whole tensor); ``reduce``, whether the ranks' local gradients are partial sums of the
    logical gradient, which the host's group sums; ``owner``, whether this rank counts the
    factor once, in norms and gathers. Each distinct window has exactly one owner."""

    window: WeightShard | None
    reduce: bool
    owner: bool


def _placement(window: WeightShard | None, *, partial: bool, owns_whole: bool) -> FactorPlacement:
    """A factor's placement: a window's first replica owns it; a whole factor, the rank that
    ``owns_whole`` (its host's first)."""

    owner = window.replica == 0 if window is not None else owns_whole
    return FactorPlacement(window, partial, owner)


class LoRAFactors(Protocol):
    """The trainable factors one host holds: allocated by ``materialize``, drawing from
    ``generator`` as one rank would draw the whole logical factors, then named;
    ``placements`` places each in its logical tensor, in ``parameters`` order."""

    def materialize(
        self, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator
    ) -> None: ...

    def parameters(self) -> tuple[torch.Tensor, ...]: ...

    def named_parameters(self, prefix: str) -> tuple[tuple[str, torch.Tensor], ...]: ...

    def placements(self) -> tuple[FactorPlacement, ...]: ...

    @property
    def trains_together(self) -> bool:
        """Whether its factors train all together or not at all: kernels that take one
        factor dtype read them, so a bound FP32 master cannot sit beside a frozen factor."""
        ...

    def modules(self, prefix: str) -> dict[str, LoRAModule]:
        """Each adapted weight's factors, by the weight's path without ``.weight``."""
        ...

    def bind_training(self, masters: Mapping[int, torch.Tensor]) -> None:
        """Train on the FP32 masters of the factors ``masters`` keys by ``id``
        (``Adapter.bind_training``); a factor it does not key trains frozen."""
        ...

    def release_training(self) -> None: ...


def _training_factor(factor: torch.Tensor, masters: Mapping[int, torch.Tensor]) -> torch.Tensor:
    """``factor`` as training reads it: its bound FP32 master, which the training kernels
    cast at each use (saving the master, not a cast, for backward), or the factor itself.
    Only training paths read a master: inference reads the factor, and a residency that
    offloads the masters releases them while serving holds the device."""

    return masters.get(id(factor), factor)


class LoRAProjection:
    """LoRA factors of one linear owner: one B per output slice, one A per input group. Its
    ``product`` and ``training_product`` replace the owner's.

    ``shared_a`` maps an A name to the slices of one checkpoint module that share it (Qwen3.5's
    ``q_proj`` holds the query and the gate); every other slice owns an A named like it.

    ``windows`` are the host's: one row window per output slice of a column-parallel host,
    whose B take them and whose A every rank holds whole with a partial gradient; or the
    input-column window of a row-parallel host, which its A takes and whose B every rank
    holds whole with a partial gradient. A B on a window several ranks hold (a replicated
    KV head) has a partial gradient too. ``logical_input_size`` is the host's whole input;
    ``owns_whole`` marks the host's rank that counts a whole factor once.
    """

    trains_together = False

    def __init__(
        self,
        *,
        input_size: int,
        output_slices: tuple[ProjectionSlice, ...],
        rank: int,
        scaling: float,
        shared_a: Mapping[str, tuple[str, ...]] | None = None,
        windows: tuple[WeightShard, ...] = (),
        logical_input_size: int | None = None,
        owns_whole: bool,
    ) -> None:
        if input_size <= 0:
            raise ValueError("LoRA input size must be positive")
        if not output_slices or any(output_slice.size <= 0 for output_slice in output_slices):
            raise ValueError("LoRA output slices must be non-empty and positive")

        names = [output_slice.name for output_slice in output_slices]
        if len(set(names)) != len(names):
            raise ValueError("LoRA output slice names must be unique")
        expected_offset = 0
        for output_slice in output_slices:
            if output_slice.offset != expected_offset:
                raise ValueError("LoRA output slices must be contiguous and ordered")
            expected_offset += output_slice.size

        self.input_size = input_size
        self.logical_input_size = input_size if logical_input_size is None else logical_input_size
        self.output_slices = output_slices
        self._windows = windows
        self._owns_whole = owns_whole
        self.rank = rank
        self.scaling = scaling
        self._input_groups = self._group_slices(output_slices, shared_a or {})
        self._factors: tuple[torch.Tensor, ...] | None = None
        self._named_factors: tuple[tuple[str, torch.Tensor], ...] = ()
        self._inference_plan: LoRAInferencePlan | None = None
        # The delta add_delta_ runs, chosen by the factors' device when they materialize.
        self._delta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        # The FP32 masters training reads, by factor id; inference reads the factors.
        self._masters: dict[int, torch.Tensor] = {}

    @property
    def is_materialized(self) -> bool:
        return self._factors is not None

    @property
    def factors(self) -> tuple[torch.Tensor, ...]:
        """``(A0, B0, A1, B1, ...)``: each output slice's pair, in output order.

        A shared A appears once for each of its slices; the linear owner computes its
        low-rank input once per slice.
        """

        return self._require_materialized()

    @property
    def training_factors(self) -> tuple[torch.Tensor, ...]:
        """``factors`` as training reads them: each bound one's FP32 master
        (``bind_training``), whose uses' gradients, a shared A's two included, the kernels
        return in FP32 to sum on it; an unbound factor as it is."""

        masters = self._masters
        return tuple(_training_factor(factor, masters) for factor in self.factors)

    def bind_training(self, masters: Mapping[int, torch.Tensor]) -> None:
        self._masters = {
            id(factor): masters[id(factor)] for factor in self.parameters() if id(factor) in masters
        }

    def release_training(self) -> None:
        self._masters = {}

    def materialize(
        self, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator
    ) -> None:
        if self.is_materialized:
            raise RuntimeError("LoRA projection is already materialized")

        input_window = self._input_window
        pairs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        named_factors = []
        for a_name, slice_indices in self._input_groups:
            logical = torch.empty(self.rank, self.logical_input_size, dtype=dtype)
            torch.nn.init.kaiming_uniform_(logical, a=math.sqrt(5), generator=generator)
            if input_window is not None:
                logical = logical.narrow(1, input_window.start, self.input_size)
            a = logical.to(device=device, copy=True).requires_grad_(True)
            named_factors.append((f"{a_name}.lora_A", a))
            for index in slice_indices:
                output_slice = self.output_slices[index]
                b = torch.zeros(
                    output_slice.size,
                    self.rank,
                    dtype=dtype,
                    device=device,
                )
                b.requires_grad_(True)
                named_factors.append((f"{output_slice.name}.lora_B", b))
                pairs[index] = (a, b)

        self._factors = tuple(
            factor for index in range(len(self.output_slices)) for factor in pairs[index]
        )
        self._named_factors = tuple(named_factors)
        self._delta = self._kernel_delta if device.type == "cuda" else self._torch_delta
        self._inference_plan = try_prepare_lora_inference_plan(
            self._factors,
            input_size=self.input_size,
            output_sizes=tuple(output_slice.size for output_slice in self.output_slices),
            scaling=self.scaling,
        )

    def parameters(self) -> tuple[torch.Tensor, ...]:
        """Each A, then the B of each slice it serves, in order of the A's first slice."""

        self._require_materialized()
        return tuple(tensor for _, tensor in self._named_factors)

    def named_parameters(self, prefix: str) -> tuple[tuple[str, torch.Tensor], ...]:
        self._require_materialized()
        return tuple((f"{prefix}.{name}", tensor) for name, tensor in self._named_factors)

    def placements(self) -> tuple[FactorPlacement, ...]:
        """A column-parallel host's B take its output windows and its A are whole partial
        copies; a row-parallel host's A take its input window and its B are whole partial
        copies. A B on a window several ranks hold is partial too."""

        column = bool(self._windows) and self._input_window is None
        placements = []
        for name, _ in self.named_parameters(""):
            window = None
            if name.endswith(".lora_A"):
                partial, window = column, self._input_window
            elif column:
                window = self._windows[self._slice_index(name)]
                partial = window.replicas > 1
            else:
                partial = bool(self._windows)
            placements.append(_placement(window, partial=partial, owns_whole=self._owns_whole))
        return tuple(placements)

    def modules(self, prefix: str) -> dict[str, LoRAModule]:
        a_names = {index: a for a, indices in self._input_groups for index in indices}
        a = tuple(f"{prefix}.{a_names[index]}.lora_A" for index in range(len(self.output_slices)))
        b = tuple(f"{prefix}.{output_slice.name}.lora_B" for output_slice in self.output_slices)
        return {prefix: LoRAModule(a, b)}

    @property
    def _input_window(self) -> WeightShard | None:
        windows = self._windows
        return windows[0] if windows and windows[0].axis == 1 else None

    def _slice_index(self, name: str) -> int:
        slice_name = name.rsplit(".", 2)[-2]
        return next(i for i, s in enumerate(self.output_slices) if s.name == slice_name)

    def product(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """``x`` through the frozen weight and the factors, for inference."""

        return lora_projection(
            x,
            weight,
            bias,
            self.factors,
            scaling=self.scaling,
            inference_plan=self._inference_plan,
        )

    def warm_up(self, max_rows: int, weight: torch.Tensor) -> list[dict]:
        """Compile the inference plan's routes for 1..``max_rows`` rows, the delta's and the
        base product's over ``weight``, so that no forward or capture compiles one; the
        routes, or none without a specialized plan (none before ``materialize``)."""

        plan = self._inference_plan
        return [] if plan is None else plan.warm_up(max_rows, weight)

    def training_product(
        self, x: torch.Tensor, base: FrozenBase, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """``base(x)`` plus the training factors' delta, differentiable in ``x`` and them."""

        y = lora_projection_training(x, base, self.training_factors, scaling=self.scaling)
        return y if bias is None else y + bias

    def add_delta_(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``y`` plus each slice's ``scaling * x A^T B^T`` in place, for inference, after a base
        product that does not take the adapter; returns ``y``. The factors' device chose the
        delta when they materialized: cti-kernels' on CUDA, torch on the CPU."""

        if self._delta is None:
            raise RuntimeError("LoRA projection is not materialized")
        return self._delta(y, x)

    def _kernel_delta(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """cti-kernels' ``lora_delta_factors_``: one shrink of every slice, its scaled low rank
        rounded to ``x``'s dtype, and one expand into ``y``. Each row count a graph captures
        is warmed first (``warm_up_delta``)."""

        lora_delta_factors_(
            y.view(-1, y.shape[-1]),
            x.reshape(-1, x.shape[-1]).contiguous(),
            self.frozen_factors,
            scaling=self.scaling,
            output_sizes=self.output_sizes,
        )
        return y

    def _torch_delta(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``x A^T`` once per A however many slices share it, then one ``addmm_`` per slice."""

        x_2d, y_2d = x.reshape(-1, x.shape[-1]), y.view(-1, y.shape[-1])
        factors = self.factors
        for _, indices in self._input_groups:
            low_rank = F.linear(x_2d, factors[2 * indices[0]])
            for index in indices:
                output_slice = self.output_slices[index]
                y_2d.narrow(1, output_slice.offset, output_slice.size).addmm_(
                    low_rank, factors[2 * index + 1].t(), alpha=self.scaling
                )
        return y

    @property
    def frozen_factors(self) -> tuple[torch.Tensor, ...]:
        """``factors`` as the inference kernels read them: views that require no gradient."""

        return tuple(factor.detach() for factor in self.factors)

    @property
    def output_sizes(self) -> tuple[int, ...]:
        return tuple(output_slice.size for output_slice in self.output_slices)

    def warm_up_delta(self, device: torch.device) -> None:
        """Run ``add_delta_`` once eagerly, on one row of zeros, so that its kernels compile
        before any forward or capture."""

        dtype = self.factors[0].dtype
        x = torch.zeros(1, self.input_size, dtype=dtype, device=device)
        y = torch.zeros(1, sum(s.size for s in self.output_slices), dtype=dtype, device=device)
        self.add_delta_(y, x)

    def _require_materialized(
        self,
    ) -> tuple[torch.Tensor, ...]:
        if self._factors is None:
            raise RuntimeError("LoRA projection has not been materialized")
        return self._factors

    @staticmethod
    def _group_slices(
        output_slices: tuple[ProjectionSlice, ...],
        shared_a: Mapping[str, tuple[str, ...]],
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Name each A and list its slice indices, in order of the A's first slice."""

        indices = {output_slice.name: index for index, output_slice in enumerate(output_slices)}
        shared_by_index: dict[int, str] = {}
        for a_name, slice_names in shared_a.items():
            if not a_name or a_name in indices:
                raise ValueError(
                    f"Shared LoRA A name {a_name!r} must be non-empty and differ from "
                    "every output slice name"
                )
            if not isinstance(slice_names, tuple) or len(slice_names) < 2:
                raise ValueError(
                    f"Shared LoRA A {a_name!r} must span a tuple of at least two output slices"
                )
            unknown = [name for name in slice_names if name not in indices]
            if unknown:
                raise ValueError(f"Shared LoRA A {a_name!r} names unknown output slices: {unknown}")
            for name in slice_names:
                if indices[name] in shared_by_index:
                    raise ValueError(f"Output slice {name!r} is shared more than once")
                shared_by_index[indices[name]] = a_name

        groups: dict[str, list[int]] = {}
        for index, output_slice in enumerate(output_slices):
            groups.setdefault(shared_by_index.get(index, output_slice.name), []).append(index)
        return tuple((a_name, tuple(slice_indices)) for a_name, slice_indices in groups.items())


@dataclass(frozen=True, slots=True)
class _ExpertFactor:
    """One stacked factor of ``ExpertLoRAFactors``: its name under ``experts``, its local
    shape, the axis that stacks experts, the axis an intermediate window cuts (None: the
    factor is whole on every rank of a split, a partial copy) and the projection slots it
    serves (``gate_up_proj``'s gate 0 and up 1, or ``down_proj``'s 0)."""

    name: str
    shape: tuple[int, ...]
    expert_axis: int
    window_axis: int | None
    slots: tuple[int, ...]


class ExpertLoRAFactors:
    """The per-expert LoRA of the routed experts ``placement`` gives one rank, as
    ``moe.ExpertLoRA``, in ``layout`` (``ExpertLoRALayout``), stacked by held expert as the
    expert weights are and named like a linear's factors under ``experts``:

    - ``shared_a`` (PEFT's adapter on a fused ``gate_up_proj``; what training uses):
      ``gate_up_proj.gate_up.lora_A`` ``[E, r, H]``, one A the gate and up halves share, as a
      linear's shared A is named, and ``gate_up_proj.gate_up.lora_B`` ``[2, E, I, r]``, the
      gate's B then the up's, slice-major (cti-kernels' layout);
    - ``per_projection`` (an imported per-expert adapter, served only): an A ``[E, r, H]``
      and a B ``[E, I, r]`` on each half, ``gate_up_proj.gate.*`` and ``gate_up_proj.up.*``;
    - and in both, the down projection's ``down_proj.full.lora_A`` ``[E, r, I]`` and
      ``lora_B`` ``[E, H, r]``.

    A rank holding a window of every expert's intermediate width holds that window of the
    gate and up B and of the down A; the gate/up A and the down B are whole there, with
    partial gradients. Each expert draws its factors from its own seed, which the block's
    generator gives every expert alike on every rank: a rank draws only the experts it holds.
    ``owns_whole`` marks the block's rank that counts a whole factor once.
    """

    trains_together = True

    def __init__(
        self,
        placement: Placement,
        hidden_size: int,
        *,
        rank: int,
        scaling: float,
        owns_whole: bool,
        layout: ExpertLoRALayout,
    ):
        from cti.config import ExpertLoRALayout

        self.placement = placement
        self._owns_whole = owns_whole
        self.rank = rank
        self.scaling = scaling
        self.layout = ExpertLoRALayout(layout)
        experts, width = len(placement.experts), len(placement.intermediate)
        if self.layout is ExpertLoRALayout.SHARED_A:
            gate_up = (
                _ExpertFactor(
                    "gate_up_proj.gate_up.lora_A", (experts, rank, hidden_size), 0, None, (0, 1)
                ),
                _ExpertFactor(
                    "gate_up_proj.gate_up.lora_B", (2, experts, width, rank), 1, 2, (0, 1)
                ),
            )
        else:
            gate_up = tuple(
                factor
                for half, slot in (("gate", 0), ("up", 1))
                for factor in (
                    _ExpertFactor(
                        f"gate_up_proj.{half}.lora_A",
                        (experts, rank, hidden_size),
                        0,
                        None,
                        (slot,),
                    ),
                    _ExpertFactor(
                        f"gate_up_proj.{half}.lora_B", (experts, width, rank), 0, 1, (slot,)
                    ),
                )
            )
        self._factors = (
            *gate_up,
            _ExpertFactor("down_proj.full.lora_A", (experts, rank, width), 0, 2, (0,)),
            _ExpertFactor("down_proj.full.lora_B", (experts, hidden_size, rank), 0, None, (0,)),
        )
        self._named_factors: tuple[tuple[str, torch.Tensor], ...] = ()
        # The FP32 masters training reads, by factor id; inference reads the factors.
        self._masters: dict[int, torch.Tensor] = {}

    def training(self) -> ExpertLoRAFactors | _TrainingExpertLoRA:
        """These factors as training reads them: with masters bound (``bind_training``),
        their FP32 masters, which the kernels cast at each use; else the factors."""

        return self if not self._masters else _TrainingExpertLoRA(self, self._masters)

    def bind_training(self, masters: Mapping[int, torch.Tensor]) -> None:
        self._masters = {
            id(factor): masters[id(factor)] for factor in self.parameters() if id(factor) in masters
        }

    def release_training(self) -> None:
        self._masters = {}

    @property
    def gate_up(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(gate A, gate B, up A, up B)``; in ``shared_a`` the one A twice and the
        slice-major B's halves, views of the trainable factors."""

        factors = self.parameters()
        if len(factors) == 4:
            a, b = factors[0], factors[1]
            return a, b[0], a, b[1]
        return factors[:4]

    @property
    def gate_up_shared(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(A, B)`` of the shared-A layout's gate and up, or None."""

        factors = self.parameters()
        return (factors[0], factors[1]) if len(factors) == 4 else None

    @property
    def down(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.parameters()[-2:]

    def materialize(
        self, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator
    ) -> None:
        if self._named_factors:
            raise RuntimeError("per-expert LoRA is already materialized")
        placement = self.placement
        seeds = torch.randint(2**62, (placement.num_experts,), generator=generator).tolist()
        window = placement.intermediate
        named = []
        for factor in self._factors:
            if factor.name.endswith(".lora_B"):
                tensor = torch.zeros(factor.shape, dtype=dtype, device=device)
            else:
                # Each held expert's A from its own seed, drawn at the whole input width and
                # cut to a window when the window cuts it.
                whole = list(factor.shape)
                if factor.window_axis is not None:
                    whole[factor.window_axis] = placement.intermediate_size
                tensor = torch.empty(whole, dtype=dtype, device=device)
                for local, expert in enumerate(placement.experts):
                    expert_generator = torch.Generator(device).manual_seed(seeds[expert])
                    torch.nn.init.kaiming_uniform_(
                        tensor[local], a=math.sqrt(5), generator=expert_generator
                    )
                if factor.window_axis is not None:
                    tensor = tensor.narrow(factor.window_axis, window.start, len(window))
            named.append((f"experts.{factor.name}", tensor.contiguous().requires_grad_(True)))
        self._named_factors = tuple(named)

    def parameters(self) -> tuple[torch.Tensor, ...]:
        """The trainable factors: the gate/up ones, then the down A and B."""

        return tuple(tensor for _, tensor in self._require_materialized())

    def named_parameters(self, prefix: str) -> tuple[tuple[str, torch.Tensor], ...]:
        head = f"{prefix}." if prefix else ""
        return tuple((head + name, tensor) for name, tensor in self._require_materialized())

    def placements(self) -> tuple[FactorPlacement, ...]:
        """A rank holding some experts holds their window of the expert axis, each complete.
        A rank holding an intermediate window of every expert holds that window of the
        factors it cuts; a whole one fed by (or feeding) the cut side is a partial copy."""

        placement = self.placement
        experts, window = placement.experts, placement.intermediate
        split = len(window) != placement.intermediate_size
        if split and not placement.holds_every_expert:
            raise ValueError("per-expert LoRA places an expert or an intermediate window, not both")
        self._require_materialized()
        placements = []
        for factor in self._factors:
            cut = factor.window_axis is not None
            held = None
            if not placement.holds_every_expert:
                held = WeightShard(factor.expert_axis, experts.start, experts.stop)
            elif split and cut:
                held = WeightShard(factor.window_axis, window.start, window.stop)
            placements.append(
                _placement(held, partial=split and not cut, owns_whole=self._owns_whole)
            )
        return tuple(placements)

    def modules(self, prefix: str) -> dict[str, LoRAModule]:
        """Each projection's factor names by slot; the fused B stacks its slots on axis 0."""

        head = f"{prefix}.experts" if prefix else "experts"
        modules: dict[str, tuple[dict[int, str], dict[int, str]]] = {}
        packed: set[str] = set()
        for factor in self._factors:
            stem = f"{head}.{factor.name.split('.')[0]}"
            a, b = modules.setdefault(stem, ({}, {}))
            side = a if factor.name.endswith(".lora_A") else b
            for slot in factor.slots:
                side[slot] = f"{head}.{factor.name}"
            if factor.name.endswith(".lora_B") and len(factor.slots) > 1:
                packed.add(stem)
        experts = self.placement.num_experts
        return {
            stem: LoRAModule(
                tuple(a[slot] for slot in sorted(a)),
                tuple(b[slot] for slot in sorted(b)),
                experts,
                slot_major_b=stem in packed,
            )
            for stem, (a, b) in modules.items()
        }

    def _require_materialized(self) -> tuple[tuple[str, torch.Tensor], ...]:
        if not self._named_factors:
            raise RuntimeError("per-expert LoRA has not been materialized")
        return self._named_factors


class _TrainingExpertLoRA:
    """``ExpertLoRAFactors`` as training reads it (``moe.ExpertLoRA``): its factors' FP32
    masters, in the same views."""

    def __init__(self, factors: ExpertLoRAFactors, masters: Mapping[int, torch.Tensor]) -> None:
        self._scaling = factors.scaling
        # Every factor is bound (Adapter.check_training_selection): one dtype, as the kernels
        # take it.
        self._factors = tuple(_training_factor(factor, masters) for factor in factors.parameters())

    @property
    def scaling(self) -> float:
        return self._scaling

    @property
    def gate_up(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        factors = self._factors
        if len(factors) == 4:
            a, b = factors[0], factors[1]
            return a, b[0], a, b[1]
        gate_a, gate_b, up_a, up_b = factors[:4]
        return gate_a, gate_b, up_a, up_b

    @property
    def gate_up_shared(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        factors = self._factors
        return (factors[0], factors[1]) if len(factors) == 4 else None

    @property
    def down(self) -> tuple[torch.Tensor, torch.Tensor]:
        down_a, down_b = self._factors[-2:]
        return down_a, down_b

    def training(self) -> _TrainingExpertLoRA:
        return self


class Adapter:
    """One adapter: the LoRA factors it attaches to a host's projections, by state-dict path;
    a trainable part (``models.training_contract.TrainablePart``)."""

    def __init__(self, config: LoRAConfig) -> None:
        self.config = config
        self._projections: dict[str, LoRAFactors] = {}
        self._hosts: dict[int, str] = {}
        self._attached = False
        self._materialized = False

    def attach(self, host: BaseOP) -> None:
        """Attach factors once to every op of ``host`` whose ``lora_target`` is configured."""

        if self._attached:
            raise RuntimeError("LoRA adapter is already attached")
        config = self.config
        for path, op in host.named_ops():
            if op.lora_target in config.targets:
                self._projections[path] = op.attach_lora(config)
                self._hosts[id(op)] = path
        self._attached = True

    def materialize(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self._materialized:
            raise RuntimeError("LoRA adapter is already materialized")
        if not self._projections:
            raise RuntimeError("LoRA adapter has no projections")

        for path, projection in self._projections.items():
            projection.materialize(device=device, dtype=dtype, generator=self._generator(path))
        self._materialized = True

    def check_training_selection(self, names: tuple[str, ...]) -> None:
        """Refuse a selection that trains some factors of a projection whose factors train
        together (``LoRAFactors.trains_together``) and not the rest."""

        chosen = frozenset(names)
        for path, projection in self._projections.items():
            if not projection.trains_together:
                continue
            factors = {name for name, _ in projection.named_parameters(path)}
            if chosen & factors and not factors <= chosen:
                raise ValueError(
                    f"{path}: these LoRA factors train all together or not at all; "
                    f"selected {sorted(chosen & factors)} of {sorted(factors)}"
                )

    def bind_training(self, masters: Mapping[str, torch.Tensor]) -> None:
        """Train on FP32 ``masters`` of the factors they name (``named_parameters`` names):
        training reads each through a fresh cast at every use, so its gradient accumulates
        on the master in FP32, while inference keeps reading the factors, which change only
        when published. A factor ``masters`` does not name trains frozen."""

        named = dict(self.named_parameters())
        unknown = sorted(set(masters) - set(named))
        if unknown:
            raise ValueError(f"masters name no LoRA factor: {unknown}")
        for name, master in masters.items():
            factor = named[name]
            if (master.dtype, master.shape, master.device) != (
                torch.float32,
                factor.shape,
                factor.device,
            ):
                raise ValueError(
                    f"master of {name} must be an FP32 {tuple(factor.shape)} tensor on "
                    f"{factor.device}"
                )
        by_factor = {id(named[name]): master for name, master in masters.items()}
        for projection in self._projections.values():
            projection.bind_training(by_factor)

    def release_training(self) -> None:
        """Train on the factors themselves again (``bind_training`` undone)."""

        for projection in self._projections.values():
            projection.release_training()

    def _generator(self, path: str) -> torch.Generator:
        """The CPU generator of the factors at ``path``, from the adapter seed and the path."""

        digest = hashlib.sha256(f"{self.config.seed}:{path}".encode()).digest()
        return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") >> 1)

    def parameters_in(self, op: BaseOP) -> tuple[torch.Tensor, ...]:
        """The factors attached to ``op`` or to an op under it, in adapter order."""

        self._require_materialized()
        paths = {self._hosts[id(inner)] for _, inner in op.named_ops() if id(inner) in self._hosts}
        return tuple(
            tensor
            for path, projection in self._projections.items()
            if path in paths
            for tensor in projection.parameters()
        )

    def identity(self) -> dict[str, object]:
        """What a checkpoint of these factors depends on beyond their names and dtypes."""

        config = self.config
        return {
            "targets": sorted(config.targets),
            "rank": config.rank,
            "alpha": float(config.alpha),
        }

    def modules(self) -> dict[str, LoRAModule]:
        """Each adapted weight's factors, by the weight's path without ``.weight``."""

        return {
            stem: module
            for path, projection in self._projections.items()
            for stem, module in projection.modules(path).items()
        }

    def placements(self) -> tuple[FactorPlacement, ...]:
        """Each factor's placement, in ``named_parameters`` order."""

        self._require_materialized()
        return tuple(
            placement
            for projection in self._projections.values()
            for placement in projection.placements()
        )

    def named_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        self._require_materialized()
        return tuple(
            item
            for key, projection in self._projections.items()
            for item in projection.named_parameters(key)
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach() for name, tensor in self.named_parameters()}

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            missing = sorted(set(expected) - set(state_dict))
            unexpected = sorted(set(state_dict) - set(expected))
            raise ValueError(f"Invalid LoRA state dict: missing={missing}, unexpected={unexpected}")

        validated = []
        for name, destination in expected.items():
            source = state_dict[name]
            if source.shape != destination.shape:
                raise ValueError(
                    f"LoRA tensor {name} has shape {tuple(source.shape)}, "
                    f"expected {tuple(destination.shape)}"
                )
            if source.dtype != destination.dtype:
                raise ValueError(
                    f"LoRA tensor {name} has dtype {source.dtype}, expected {destination.dtype}"
                )
            validated.append((destination, source))

        with torch.no_grad():
            for destination, source in validated:
                destination.copy_(source)

    def load(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load whole factors: each takes the window its placement holds, its rank
        zero-padded to the adapter's. A B ``state`` lacks is zero and an A it lacks keeps
        its initial values, so the slice adds nothing yet still trains."""

        local = self.state_dict()
        for (path, held), placement in zip(local.items(), self.placements(), strict=True):
            tensor = state.get(path)
            if tensor is None:
                if path.endswith("B"):
                    local[path] = torch.zeros_like(held)
                continue
            b = path.endswith("B")
            padding = self.config.rank - tensor.shape[-1 if b else -2]
            tensor = F.pad(tensor, (0, padding) if b else (0, 0, 0, padding))
            if placement.window is not None:
                tensor = placement.window.select(tensor)
            local[path] = tensor.to(held.dtype)
        self.load_state_dict(local)

    def _require_materialized(self) -> None:
        if not self._materialized:
            raise RuntimeError("LoRA adapter has not been materialized")


__all__ = [
    "Adapter",
    "ExpertLoRAFactors",
    "FactorPlacement",
    "LoRAFactors",
    "LoRAModule",
    "LoRAProjection",
]
