# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/models/weight.py`); see THIRD_PARTY_NOTICES.

"""How a model's checkpoint tensors become its parameters: the declarative vocabulary every
model's ``checkpoint_mapping`` is written in, without IO."""

from __future__ import annotations

import re
from collections.abc import Callable, Container, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

# A stacked expert's checkpoint name: ``<prefix>.experts.<idx>.<name>``.
EXPERT_NAME = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


class CheckpointTransform(Protocol):
    """Cut one checkpoint tensor into the runtime parts of a rule's targets."""

    def cuts(self, rows: int) -> bool:
        """Whether a tensor each row of which covers ``rows`` weight rows can be cut."""
        ...

    def split(self, tensor: torch.Tensor, rows: int = 1) -> tuple[torch.Tensor, ...]: ...

    def join(self, parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """The one tensor ``split`` cuts into ``parts`` (whole weight rows)."""
        ...


@dataclass(frozen=True, slots=True)
class RowSplit:
    """Cut rows into one part per size, the sizes repeating every ``sum(sizes)`` rows.

    ``RowSplit((q, k, v))`` takes all query, then all key, then all value rows;
    ``RowSplit((d, d))`` takes per-head ``[query, gate]`` rows to all queries, then all gates.
    """

    sizes: tuple[int, ...]

    def cuts(self, rows: int) -> bool:
        return not any(size % rows for size in self.sizes)

    def split(self, tensor: torch.Tensor, rows: int = 1) -> tuple[torch.Tensor, ...]:
        sizes = tuple(size // rows for size in self.sizes)
        groups = tensor.unflatten(0, (-1, sum(sizes)))
        return tuple(part.flatten(0, 1) for part in groups.split(sizes, dim=1))

    def join(self, parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        import torch

        groups = [
            part.unflatten(0, (-1, size)) for part, size in zip(parts, self.sizes, strict=True)
        ]
        return torch.cat(groups, dim=1).flatten(0, 1)


@dataclass(frozen=True, slots=True)
class Reshape:
    """Cut a tensor that is never quantized with ``function``."""

    function: Callable[[torch.Tensor], tuple[torch.Tensor, ...]]

    def cuts(self, rows: int) -> bool:
        return rows == 1

    def split(self, tensor: torch.Tensor, rows: int = 1) -> tuple[torch.Tensor, ...]:
        return tuple(self.function(tensor))

    def join(self, parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise ValueError("a reshaped checkpoint tensor has no inverse")


@dataclass(frozen=True, slots=True)
class CheckpointRule:
    """Turn one checkpoint tensor into one or more runtime parts.

    ``suffix`` matches whole trailing dotted components of a renamed name; each target replaces it.
    ``transform`` (the declaring model's) cuts the CPU tensor into one tensor per target, ideally
    views; without one the rule only renames. A ``RowSplit`` cuts a weight's companions too (a
    scalar companion goes whole to each target); a ``Reshape`` only a tensor that is never
    quantized. A target listed ``n`` times takes
    those parts, in order, as slots ``0..n-1`` of one parameter, each with its own window; no
    merge may match it.
    """

    suffix: str
    targets: tuple[str, ...]
    transform: CheckpointTransform | None = None

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError(f"checkpoint rule {self.suffix!r} has no targets")
        if self.transform is None and len(self.targets) != 1:
            raise ValueError(
                f"checkpoint rule {self.suffix!r} needs a transform to produce "
                f"{len(self.targets)} tensors"
            )

    def matches(self, name: str) -> bool:
        return name == self.suffix or name.endswith(f".{self.suffix}")

    def target_names(self, name: str, companion: str = "") -> tuple[str, ...]:
        """The targets of weight ``name``, or of its ``companion`` leaf beside it."""

        stem = name.removesuffix(self.suffix)
        targets = (stem + target for target in self.targets)
        if not companion:
            return tuple(targets)
        return tuple(f"{target.rpartition('.')[0]}.{companion}" for target in targets)

    def split(self, name: str, tensor: torch.Tensor, rows: int = 1) -> tuple[torch.Tensor, ...]:
        """Cut ``tensor``: a weight, or a companion each row of which covers ``rows`` weight rows."""
        transform = self.transform
        if transform is None:
            return (tensor,)
        if tensor.ndim == 0:
            return (tensor,) * len(self.targets)
        if not transform.cuts(rows):
            raise ValueError(f"{name}: rule {self.suffix!r} cannot cut {rows}-row blocks")
        parts = transform.split(tensor, rows)
        if len(parts) != len(self.targets):
            raise ValueError(
                f"{name}: checkpoint rule {self.suffix!r} returned {len(parts)} tensors "
                f"for {len(self.targets)} targets"
            )
        return parts

    def join(self, name: str, parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """The checkpoint tensor ``name`` whose targets' parts ``split`` gives, from them."""
        if self.transform is None:
            (part,) = parts
            return part
        return self.transform.join(tuple(parts))


@dataclass(frozen=True, slots=True)
class MergeRule:
    """Concatenate sibling checkpoint modules along dim 0 into one runtime module.

    A parameter of a module named like ``parts[i]`` becomes slot ``i`` of the same
    parameter of the sibling module ``target``: with parts ``("q_proj", "k_proj",
    "v_proj")``, ``self_attn.k_proj.weight`` is slot 1 of ``self_attn.qkv_proj.weight``.
    """

    target: str
    parts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CheckpointPlacement:
    """Where one runtime part of a checkpoint tensor lands in the model."""

    # The model parameter that receives the part.
    name: str
    slot: int = 0
    num_slots: int = 1
    expert: int | None = None


@dataclass(frozen=True, slots=True)
class CheckpointMapping:
    """How a model's checkpoint tensors become its parameters.

    Each model declares one as ``checkpoint_mapping``; the shared loader knows no
    model-specific transform. For each checkpoint tensor the mapping applies, in
    order: the first matching prefix rename, the skips, the first matching rule, the
    merges and expert stacking. Every resulting name must be a model parameter. A
    renamed name in ``optional_names``, such as the LM head of a model that ties it
    to the embedding, is dropped when the model declares no parameter of that name.
    Every other leaf of a module whose weight a rule matches, but its bias, is that
    weight's companion (a quantized weight's scales) and follows it; a stacked expert's
    companion stacks as ``<projection>_<leaf>``.
    """

    prefix_renames: tuple[tuple[str, str], ...] = ()
    skip_prefixes: tuple[str, ...] = ()
    rules: tuple[CheckpointRule, ...] = ()
    merges: tuple[MergeRule, ...] = ()
    stack_experts: bool = False
    optional_names: tuple[str, ...] = ()

    def placements(
        self,
        checkpoint_name: str,
        declared: Container[str],
    ) -> tuple[CheckpointPlacement, ...]:
        """Locate each runtime part of one checkpoint tensor among ``declared`` names.

        A skipped tensor and an undeclared optional one have no placements. A part
        that lands on an undeclared name raises, naming the checkpoint tensor.
        """

        name = self._rename(checkpoint_name)
        weight, companion = self._companion(name)
        if name.startswith(self.skip_prefixes) or (
            (optional := f"{name.rpartition('.')[0]}.weight") in self.optional_names
            and optional not in declared
        ):
            return ()
        rule = self._rule_for(weight)
        parts = (name,) if rule is None else rule.target_names(weight, companion)
        placements = tuple(
            self._place(part, parts[:index].count(part), parts.count(part))
            for index, part in enumerate(parts)
        )
        for placement in placements:
            if placement.name not in declared:
                raise ValueError(
                    f"checkpoint tensor {checkpoint_name!r} maps to {placement.name!r}, "
                    "which the model does not declare"
                )
        return placements

    def split(
        self, checkpoint_name: str, tensor: torch.Tensor, weight_rows: int | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Return the runtime parts of one checkpoint tensor, aligned with its placements;
        a companion's ``weight_rows`` are its weight's, each of its rows covering a block."""

        name = self._rename(checkpoint_name)
        weight, companion = self._companion(name)
        rule = self._rule_for(weight)
        if rule is None:
            return (tensor,)
        rows = 1
        if companion and tensor.ndim and weight_rows is not None:
            rows = -(-weight_rows // tensor.shape[0])
        return rule.split(name, tensor, rows)

    def join(self, checkpoint_name: str, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        """The weight ``checkpoint_name`` from its runtime parts, aligned with its placements:
        ``split``'s inverse."""

        rule = self._rule_for(self._rename(checkpoint_name))
        if rule is None:
            (part,) = parts
            return part
        return rule.join(checkpoint_name, tuple(parts))

    def weight_of(self, checkpoint_name: str) -> str | None:
        """The checkpoint weight a companion follows; ``None`` for any other tensor."""

        name = self._rename(checkpoint_name)
        weight, companion = self._companion(name)
        if not companion:
            return None
        return checkpoint_name.removesuffix(name.rpartition(".")[2]) + "weight"

    def _companion(self, name: str) -> tuple[str, str]:
        """``name``'s weight and companion leaf; a weight's or another tensor's are itself, "".
        A companion is a leaf beside a weight a rule matches, which no rule matches itself."""

        module, _, leaf = name.rpartition(".")
        weight = f"{module}.weight"
        if leaf in ("weight", "bias") or self._rule_for(name) or not self._rule_for(weight):
            return name, ""
        return weight, leaf

    def _rename(self, checkpoint_name: str) -> str:
        for prefix, replacement in self.prefix_renames:
            if checkpoint_name.startswith(prefix):
                return replacement + checkpoint_name.removeprefix(prefix)
        return checkpoint_name

    def _rule_for(self, name: str) -> CheckpointRule | None:
        return next((rule for rule in self.rules if rule.matches(name)), None)

    def _place(self, part: str, slot: int, num_slots: int) -> CheckpointPlacement:
        """Place one part; a rule target repeated ``num_slots`` times fills slot ``slot``."""

        name = part
        module, _, parameter = part.rpartition(".")
        parent, _, leaf = module.rpartition(".")
        for merge in self.merges:
            if leaf in merge.parts:
                if num_slots > 1:
                    raise ValueError(
                        f"checkpoint rule target {part!r} is repeated and also merged into "
                        f"{merge.target!r}, so its slots would be ambiguous"
                    )
                merged_module = f"{parent}.{merge.target}" if parent else merge.target
                name = f"{merged_module}.{parameter}"
                slot, num_slots = merge.parts.index(leaf), len(merge.parts)
                break
        expert = None
        if self.stack_experts and (match := EXPERT_NAME.match(name)) is not None:
            packed_name = match.group("name").removesuffix(".weight").replace(".", "_")
            name = f"{match.group('prefix')}.{packed_name}"
            expert = int(match.group("idx"))
        return CheckpointPlacement(name, slot, num_slots, expert)
