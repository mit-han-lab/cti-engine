"""Quantization recipes and formats.

A recipe reads one checkpoint convention from the checkpoint's JSON documents and its tensor
headers and says, per module, which format's method computes it. Format names live only in
this package. ``RECIPES`` loads each recipe module on first use; the documents a recipe reads
beside ``config.json`` are static data of its entry, so listing them imports no recipe.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

if TYPE_CHECKING:
    import torch

    from ..base import BaseOP
    from ..experts_method import ExpertsMethod
    from ..linear_method import LinearMethod

Documents = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class TensorHeader:
    """One checkpoint tensor as its safetensors header records it."""

    dtype: str  # the safetensors dtype tag, such as "BF16"
    shape: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointIndex:
    """Every checkpoint tensor's header, by name; no tensor data."""

    tensors: Mapping[str, TensorHeader]

    @classmethod
    def from_headers(cls, headers: Mapping[str, Mapping[str, Any]]) -> CheckpointIndex:
        """Index safetensors headers (``dtype`` tag and ``shape`` by tensor name)."""

        return cls(
            {
                name: TensorHeader(str(header["dtype"]), tuple(int(d) for d in header["shape"]))
                for name, header in headers.items()
            }
        )


@dataclass(frozen=True)
class CheckpointPart:
    """One checkpoint module: its tensors by leaf name."""

    module: str
    tensors: Mapping[str, TensorHeader]


class QuantRecipe(Protocol):
    """One checkpoint convention: it claims a checkpoint from its documents."""

    NAME: ClassVar[str]
    # Whether the convention's checkpoints train; every one serves and serves LoRA adapters.
    TRAINS: ClassVar[bool]

    @classmethod
    def parse(cls, documents: Documents) -> Self | None:
        """The recipe a checkpoint's documents declare, or ``None`` when they declare
        another convention; a claimed but malformed declaration raises, naming the key."""
        ...

    def verify(self, index: CheckpointIndex) -> None:
        """Check the declaration against the checkpoint's tensor headers."""
        ...

    def method_for(self, part: CheckpointPart) -> type[LinearMethod]:
        """The method of a linear layer that holds a quantized checkpoint module; raises for
        one the recipe does not claim."""
        ...

    def experts_method_for(self, part: CheckpointPart) -> type[ExpertsMethod]:
        """The method of the routed experts that stack a quantized checkpoint module (one
        expert's projection); raises for one the recipe does not claim."""
        ...


@dataclass(frozen=True, slots=True)
class RecipeEntry:
    load: Callable[[], type[QuantRecipe]]
    # Documents beside config.json that the recipe reads when the checkpoint has them.
    metadata_files: tuple[str, ...] = ()


def _fp8_block() -> type[QuantRecipe]:
    from .fp8_block import Fp8BlockRecipe

    return Fp8BlockRecipe


def _modelopt_mixed() -> type[QuantRecipe]:
    from .modelopt_mixed import ModelOptMixedRecipe

    return ModelOptMixedRecipe


RECIPES: Mapping[str, RecipeEntry] = MappingProxyType(
    {
        "fp8_block": RecipeEntry(_fp8_block),
        "modelopt_mixed": RecipeEntry(_modelopt_mixed, metadata_files=("hf_quant_config.json",)),
    }
)


def _documents() -> tuple[str, ...]:
    """The checkpoint documents recipes read: ``config.json`` and every declared file."""

    declared = {name for entry in RECIPES.values() for name in entry.metadata_files}
    return ("config.json", *sorted(declared))


def parse(documents: Documents) -> QuantRecipe | None:
    """The one recipe claiming a checkpoint, from its documents by file name; ``None`` for
    a checkpoint that declares no quantization."""

    claims = {
        name: recipe
        for name, entry in RECIPES.items()
        if (recipe := entry.load().parse(documents)) is not None
    }
    if len(claims) > 1:
        raise ValueError(f"the checkpoint is claimed by several recipes: {sorted(claims)}")
    if claims:
        return next(iter(claims.values()))
    declared = documents["config.json"].get("quantization_config")
    if isinstance(declared, Mapping):
        declared = {key: declared[key] for key in ("quant_method", "quant_algo") if key in declared}
    if declared is not None:
        raise ValueError(f"no recipe claims the checkpoint's quantization_config {declared!r}")
    if metadata := sorted(set(documents) - {"config.json"}):
        raise ValueError(f"no recipe claims the checkpoint's {metadata}")
    return None


# Header dtypes a weight keeps when it is not quantized.
_FLOATS = frozenset({"F32", "BF16", "F16"})


class CheckpointFormats:
    """The methods one checkpoint stores its modules' weights for, read before the model is
    built, and given to the model's ops once it is (``adopt``).

    ``document(name)`` reads one of the checkpoint's JSON documents (``None``: it has none of
    that name); ``headers`` are its tensors' safetensors headers, by name; ``experts`` names
    the modules that are one expert's projection. Construction refuses, on every rank alike
    and before any model is built, what the checkpoint declares but this engine cannot
    serve: the recipe and its keys against the headers, each quantized module's claim, and
    the device's provider row.
    """

    def __init__(
        self,
        document: Callable[[str], Mapping[str, Any] | None],
        headers: Mapping[str, Mapping[str, Any]],
        experts: Collection[str],
        device: torch.device,
    ) -> None:
        documents = {name: found for name in _documents() if (found := document(name)) is not None}
        recipe = parse(documents)
        # The recipe of a checkpoint that does not train, by name.
        self.untrainable = None if recipe is None or recipe.TRAINS else recipe.NAME
        index = CheckpointIndex.from_headers(headers)
        if recipe is not None:
            recipe.verify(index)
        leaves: dict[str, dict[str, TensorHeader]] = {}
        for name, header in index.tensors.items():
            module, _, leaf = name.rpartition(".")
            leaves.setdefault(module, {})[leaf] = header
        self._methods: dict[str, type[Any]] = {}
        self._rows: dict[type[Any], str] = {}
        for module, tensors in leaves.items():
            part = CheckpointPart(module, tensors)
            if "weight" not in tensors or _plain(part):
                continue
            if recipe is None:
                found = {leaf: header.dtype for leaf, header in tensors.items()}
                raise ValueError(f"{module} holds {found}, but no recipe declares it")
            choose = recipe.experts_method_for if module in experts else recipe.method_for
            method = self._methods[module] = choose(part)
            if method not in self._rows:
                self._rows[method] = method.computed_by(device)

    def adopt(
        self, model: BaseOP, modules: Mapping[str, Mapping[int, Sequence[str]]]
    ) -> dict[str, int]:
        """Give each op of ``model`` (on the meta device, no adapter attached) the method its
        checkpoint modules (``modules``: by op path, then by the op's merge slot) store their
        weights for, refusing a window that cuts a method's blocks; return how many ops each
        method and provider row computes, an opaque summary for logs and receipts. Ops of
        floating-point modules keep their defaults; a projection whose slots take different
        methods computes each run of them apart."""

        from ..linear_method import Unquantized

        ops = dict(model.named_ops())
        summary: Counter[str] = Counter()
        for path, slots in modules.items():
            methods = []
            for slot in sorted(slots):
                chosen = {self._methods.get(module, Unquantized) for module in slots[slot]}
                if len(chosen) > 1:
                    raise ValueError(f"{path}: the modules of its slot {slot} take {chosen}")
                methods.extend(chosen)
            if set(methods) == {Unquantized}:
                continue
            op = ops[path]
            for method in set(methods) - {Unquantized}:
                summary[f"{method.__name__} on {self._rows[method]}"] += 1
            if len(set(methods)) == 1:
                op.use_method(methods[0])
            else:
                op.split(methods)
            op.params(prefix=path)  # Refuses a window that cuts the method's blocks, by path.
        return dict(summary)


def _plain(part: CheckpointPart) -> bool:
    """Whether a module holds a floating-point weight (and bias) alone."""

    plain = set(part.tensors) <= {"weight", "bias"}
    return plain and {header.dtype for header in part.tensors.values()} <= _FLOATS


__all__ = [
    "CheckpointFormats",
]
