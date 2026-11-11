from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from cti.config import TrainablePlan

NamedTensor = tuple[str, torch.Tensor]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ModelParameterUniverse:
    """Validated names and live tensor identities for one fixed model topology."""

    named_tensors: tuple[NamedTensor, ...]
    named_trainable_parameters: tuple[NamedTensor, ...]

    def __init__(
        self,
        model_state: Mapping[str, torch.Tensor],
        named_trainable_parameters: Iterable[NamedTensor],
    ) -> None:
        if not isinstance(model_state, Mapping):
            raise TypeError("Model state_dict() must return a mapping")
        persisted = self._validated_named_tensors(
            model_state.items(),
            owner="Model state",
        )
        declared = self._validated_named_tensors(
            named_trainable_parameters,
            owner="Model-declared trainable parameters",
        )

        tensors_by_name = dict(persisted)
        names_by_tensor_id = {id(tensor): name for name, tensor in persisted}
        for name, tensor in declared:
            persisted_tensor = tensors_by_name.get(name)
            if persisted_tensor is not None:
                if persisted_tensor is not tensor:
                    raise ValueError(
                        f"Model-declared trainable parameter {name!r} does not match state_dict()"
                    )
                continue
            persisted_name = names_by_tensor_id.get(id(tensor))
            if persisted_name is not None:
                raise ValueError(
                    f"Model-declared trainable tensor {name!r} is persisted as {persisted_name!r}"
                )

        declared_names = frozenset(name for name, _ in declared)
        frozen_persisted = tuple(pair for pair in persisted if pair[0] not in declared_names)
        object.__setattr__(self, "named_tensors", (*frozen_persisted, *declared))
        object.__setattr__(self, "named_trainable_parameters", declared)

    @classmethod
    def of(cls, owner: SupportsParameterUniverse) -> ModelParameterUniverse:
        """``owner``'s tensors with its trainable declaration, validated."""

        return cls(owner.state_dict(), owner.named_trainable_parameters())

    @staticmethod
    def _validated_named_tensors(
        named_tensors: Iterable[NamedTensor],
        *,
        owner: str,
    ) -> tuple[NamedTensor, ...]:
        try:
            result = tuple(named_tensors)
        except TypeError as error:
            raise TypeError(f"{owner} must be iterable") from error

        names: set[str] = set()
        tensor_ids: set[int] = set()
        for index, pair in enumerate(result):
            try:
                name, tensor = pair
            except (TypeError, ValueError) as error:
                raise TypeError(f"{owner} item {index} must be a name/tensor pair") from error
            if not isinstance(name, str):
                raise TypeError(f"{owner} name must be a string")
            if not name:
                raise ValueError(f"{owner} name must be non-empty")
            if name != name.strip():
                raise ValueError(f"{owner} name must not contain surrounding whitespace")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{owner} tensor {name!r} must be a torch.Tensor")
            if name in names:
                raise ValueError(f"{owner} names must be unique")
            if id(tensor) in tensor_ids:
                raise ValueError(f"{owner} tensors must be unique")
            names.add(name)
            tensor_ids.add(id(tensor))
        return result


class SupportsParameterUniverse(Protocol):
    """The owner of a model's tensors and of their trainable declaration."""

    def state_dict(self) -> Mapping[str, torch.Tensor]: ...

    def named_trainable_parameters(self) -> Iterable[NamedTensor]: ...

    def decoder_layer_trainable_parameters(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Each decoder layer's declared trainable tensors, in execution order."""

        ...

    def check_training_selection(self, names: tuple[str, ...]) -> None:
        """Refuse, with a ``ValueError``, a selection of declared tensors it cannot train."""

        ...

    def bind_training_weights(self, masters: Mapping[str, torch.Tensor]) -> None:
        """Train on the FP32 ``masters`` of the declared tensors they name, which training
        then reads instead; the tensors themselves stay what inference reads."""

        ...

    def release_training_weights(self) -> None:
        """Train on the declared tensors themselves again."""

        ...


@dataclass(frozen=True, slots=True)
class TrainableParameterMetadata:
    """Tensor-free description of one selected model parameter."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


def _require_disjoint_storages(named_tensors: tuple[NamedTensor, ...]) -> None:
    """Refuse two tensors whose storages overlap on one device, through whatever storage
    objects wrap the memory."""

    regions = sorted(
        (str(tensor.device), storage.data_ptr(), storage.data_ptr() + storage.nbytes(), name)
        for name, tensor in named_tensors
        if (storage := tensor.untyped_storage()).nbytes()
    )
    for (device, _, stop, owner), (other, start, _, name) in pairwise(regions):
        if device == other and start < stop:
            raise ValueError(f"universe tensors {owner!r} and {name!r} share storage")


class TrainableParameterSet:
    """One resolved selection over a model's tensors: the selected ones and the rest,
    each in universe order."""

    __slots__ = ("_named_complement", "_named_parameters", "metadata")

    def __init__(
        self, named_parameters: tuple[NamedTensor, ...], named_complement: tuple[NamedTensor, ...]
    ) -> None:
        self._named_parameters = named_parameters
        self._named_complement = named_complement
        self.metadata = tuple(
            TrainableParameterMetadata(name, tuple(parameter.shape), parameter.dtype)
            for name, parameter in named_parameters
        )

    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._named_parameters)

    @property
    def complement_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._named_complement)

    def named_parameters(self) -> tuple[NamedTensor, ...]:
        return self._named_parameters

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return tuple(parameter for _, parameter in self._named_parameters)

    def named_complement(self) -> tuple[NamedTensor, ...]:
        return self._named_complement

    def complement(self) -> tuple[torch.Tensor, ...]:
        return tuple(tensor for _, tensor in self._named_complement)


def describe_for_model(
    plan: TrainablePlan, owner: SupportsParameterUniverse
) -> tuple[TrainableParameterMetadata, ...]:
    """Describe, in order, the parameters ``resolve_for_model`` selects, without
    resolving them."""

    declared = ModelParameterUniverse.of(owner).named_trainable_parameters
    names = frozenset(name for name, _ in declared)
    if plan.selected_names is not None:
        undeclared = tuple(name for name in plan.selected_names if name not in names)
        if undeclared:
            raise ValueError(
                "selected_names must belong to the model-declared non-base parameters: "
                f"{undeclared!r}"
            )
        names = frozenset(plan.selected_names)
    owner.check_training_selection(tuple(name for name, _ in declared if name in names))
    return tuple(
        TrainableParameterMetadata(name, tuple(tensor.shape), tensor.dtype)
        for name, tensor in declared
        if name in names
    )


def resolve_for_model(
    plan: TrainablePlan, owner: SupportsParameterUniverse
) -> TrainableParameterSet:
    """Select over ``owner``'s tensors and set exactly the selected ones trainable; a
    refused selection or universe changes no flag."""

    universe = ModelParameterUniverse.of(owner)
    declared = tuple(name for name, _ in universe.named_trainable_parameters)
    selected = declared if plan.selected_names is None else plan.selected_names
    if not selected:
        raise ValueError("model declares no non-base trainable parameters")
    names = frozenset(name for name, _ in universe.named_tensors)
    unknown_names = tuple(name for name in selected if name not in names)
    if unknown_names:
        raise ValueError(f"selected_names contain unknown names: {unknown_names!r}")
    undeclared_names = tuple(name for name in selected if name not in declared)
    if undeclared_names:
        raise ValueError(
            "selected_names must belong to the model-declared non-base parameters: "
            f"{undeclared_names!r}"
        )

    chosen = frozenset(selected)
    owner.check_training_selection(tuple(name for name in declared if name in chosen))
    for name, tensor in universe.named_tensors:
        if tensor.layout != torch.strided:
            raise ValueError(f"universe tensor {name!r} must have strided layout")
        # A frozen tensor may hold any dtype, a packed quantized weight's included.
        if name in chosen and not tensor.is_floating_point():
            raise ValueError(
                f"selected tensor {name!r} must have a floating dtype, not {tensor.dtype}"
            )
        if not tensor.is_leaf:
            raise ValueError(f"universe tensor {name!r} must be a leaf")
        if tensor.grad is not None:
            raise ValueError(f"universe tensor {name!r} must not have a pre-existing gradient")
        if torch.is_inference(tensor):
            raise ValueError(f"universe tensor {name!r} must not be an inference tensor")
    _require_disjoint_storages(universe.named_tensors)

    for name, tensor in universe.named_tensors:
        tensor.requires_grad_(name in chosen)
    return TrainableParameterSet(
        tuple(pair for pair in universe.named_tensors if pair[0] in chosen),
        tuple(pair for pair in universe.named_tensors if pair[0] not in chosen),
    )


def first_trainable_layer(owner: SupportsParameterUniverse, selected: Iterable[str]) -> int:
    """The first of ``owner``'s decoder layers holding a ``selected`` parameter, by name, or
    the layer count."""

    named = dict(ModelParameterUniverse.of(owner).named_trainable_parameters)
    selected_ids = {id(named[name]) for name in selected}
    layers = owner.decoder_layer_trainable_parameters()
    return next(
        (
            index
            for index, tensors in enumerate(layers)
            if any(id(tensor) in selected_ids for tensor in tensors)
        ),
        len(layers),
    )


__all__ = [
    "ModelParameterUniverse",
    "SupportsParameterUniverse",
    "TrainableParameterMetadata",
    "TrainableParameterSet",
    "describe_for_model",
    "first_trainable_layer",
    "resolve_for_model",
]
