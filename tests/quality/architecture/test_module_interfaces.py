"""Packages meet only through their specified interfaces (GI).

The import edges between units are held by ImportEdges; the rest of the boundary is exports
(``__all__`` through the package facade), private names, and a few named seams. Rules
count positions like the other architecture guards and match code written the normal way.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import pytest

from tests.support.architecture import (
    PLAN_PREFIX_KIND,
    Hit,
    Imports,
    Module,
    Names,
    Package,
    Rule,
    Types,
    bound_names,
    exports,
    isinstance_targets,
    lexical,
    private,
    protocol,
    under,
)
from tests.support.imports import MODULE_SCOPE
from tests.support.paths import REPO_ROOT

# I1: the import edges between units.

# The two halves of inference: units of their own beside the facade and ``_step``.
INFERENCE_HALVES = ("inference._control", "inference._execution")
# The control plane's rank unit: one rank's loop and its control plane. The rest of
# ``runtime`` is the parent's side.
RUNTIME_RANK = ("runtime._core", "runtime._control")


def _structure_names(package: Package, module: Module) -> Iterator[Hit]:
    yield from Names(r"^(Thread|offline_mode|resolve_prefix_reuse)$")(package, module)
    yield from PLAN_PREFIX_KIND(package, module)


@dataclass(frozen=True)
class UnitEdges:
    """A unit's I1 row: what it imports at run time and in annotations only. A target
    is a unit or a module under it, reached directly or through package facades."""

    unit: str
    runtime: tuple[str, ...] = ()
    annotation: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportEdges:
    """I1-edges: every runtime and annotation-only import of another unit is in EDGES. A unit
    imports itself freely. Edges that arrive with a commit are additions; narrowings to named
    types are left to S4 and review."""

    EDGES: ClassVar[tuple[UnitEdges, ...]] = (
        UnitEdges("utils", (), ()),
        UnitEdges("env", (), ()),
        UnitEdges("sampling", (), ()),
        UnitEdges("memory", ("utils",), ()),
        UnitEdges("multimodal", ("utils",), ()),
        # The serving wire protocol: every hop's messages, their codec and the sampling results.
        UnitEdges("protocol", ("multimodal", "utils"), ()),
        UnitEdges("kvcache", ("utils",), ()),
        UnitEdges("moe", ("utils",), ()),
        UnitEdges("distributed", ("utils",), ()),
        UnitEdges("attention", ("utils",), ("kvcache", "forward")),
        UnitEdges("recurrent", ("utils",), ("kvcache", "forward")),
        UnitEdges(
            "layers",
            # The MoE op from LayerServices, Topology for the expert split; the MHA KV-head
            # rule from attention's value leaf.
            ("kvcache", "moe", "distributed", "attention.requirement", "utils"),
            ("attention", "recurrent", "forward"),
        ),
        UnitEdges(
            "models",
            # The model class gives its AttentionRequirement at resolution.
            ("layers", "multimodal", "forward", "attention.requirement", "utils"),
            ("recurrent",),
        ),
        # The forward contract: the batch a runner hands its model.
        UnitEdges("forward", (), ("attention", "recurrent")),
        # The resolved plan's value types.
        UnitEdges(
            "plan",
            ("utils",),
            ("models", "attention", "kvcache", "layers", "forward", "multimodal"),
        ),
        # The one derivation point; the composition root hands it the speculative method.
        UnitEdges(
            "resolution",
            ("plan", "models", "attention", "kvcache", "layers", "multimodal", "utils"),
            (),
        ),
        # The facade's lazy exports and ``_step``, the values the two halves exchange.
        UnitEdges(
            "inference",
            (*INFERENCE_HALVES, "forward", "sampling", "kvcache", "utils"),
            ("plan", "memory"),
        ),
        # Admission, scheduling and the logical KV leases.
        UnitEdges(
            "inference._control",
            (
                "inference._step",
                "forward",
                "protocol",
                "kvcache",
                "multimodal",
                "sampling",
                "utils",
            ),
            ("memory",),
        ),
        # The executor: warm-up, graphs, sampler and the speculative rounds.
        UnitEdges(
            "inference._execution",
            (
                "inference._step",
                "forward",
                "attention",
                "kvcache",
                "recurrent",
                "models",
                "memory",
                "multimodal",
                "sampling",
                "utils",
            ),
            ("plan", "distributed", "layers"),
        ),
        UnitEdges(
            "training",
            (
                "models.training_contract",
                "models.base",
                "models.activation_checkpointing",
                "protocol",
                "utils",
            ),
            ("memory",),
        ),
        # The device's resources: the composition root hands the worker its plan and
        # checkpoint, and builds what runs on them.
        UnitEdges(
            "worker",
            ("plan", "models", "layers", "kvcache", "distributed", "memory", "utils"),
            (),
        ),
        # The parent's side: process supervision, command RPC and the closed command and
        # reply values. Training only through its facade, which I1-training holds.
        UnitEdges(
            "runtime", ("protocol", "training", "tokenizer", "multimodal", "memory", "utils"), ()
        ),
        # A rank: the composition root, its loop and its control plane.
        *(
            UnitEdges(
                unit,
                (
                    "runtime._protocol",
                    *RUNTIME_RANK,
                    "worker",
                    "inference",
                    "training",
                    "resolution",
                    "plan",
                    "tokenizer",
                    "protocol",
                    "memory",
                    "env",
                    "utils",
                ),
                ("distributed",),
            )
            for unit in RUNTIME_RANK
        ),
        UnitEdges("tokenizer", ("protocol", "multimodal", "utils"), ()),
        UnitEdges("config", ("utils",), ()),
        UnitEdges("llm", ("runtime", "protocol", "tokenizer", "utils"), ()),
        UnitEdges("server", ("runtime", "protocol", "utils"), ("plan",)),
        UnitEdges("__main__", ("server",), ()),
    )
    # Leaves any unit imports: config is the public options as pure data; sampling is a
    # request's sampling law and its replies' logprobs. env, the engine's switches, is a
    # leaf too, but only a rank's entry
    # imports it. A leaf imports the standard library, utils and other leaves alone
    # (I1-leaves).
    SHARED: ClassVar[tuple[str, ...]] = ("config", "sampling")
    LEAVES: ClassVar[tuple[str, ...]] = ("env", *SHARED)

    @staticmethod
    def unit(path: str) -> str:
        """The unit of a package path: a half of inference or a runtime rank unit, else its
        first directory or a root module."""

        parts = path.removesuffix(".py").removesuffix("/__init__").split("/")
        nested = ".".join(parts[:2])
        return nested if nested in (*INFERENCE_HALVES, *RUNTIME_RANK) else parts[0]

    @classmethod
    def unit_of(cls, target: str) -> str:
        """The unit of a dotted ``cti`` name."""

        return cls.unit("/".join(target.split(".")[1:]))

    def row(self, unit: str) -> UnitEdges:
        return next((row for row in self.EDGES if row.unit == unit), UnitEdges(unit))

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        row = self.row(self.unit(module.path))
        for qualname, name, annotation_only in package.imported(module):
            target = package.defining(name)
            allowed = self.SHARED + row.runtime + (row.annotation if annotation_only else ())
            qualified = tuple(f"cti.{edge}" for edge in allowed)
            if self.unit_of(target) == row.unit or under(target, qualified):
                continue
            into_training = any(under(edge, ("cti.training",)) for edge in qualified)
            if not (into_training and self._training_facade(package, name)):
                yield qualname, target

    @staticmethod
    def _training_facade(package: Package, name: str) -> bool:
        """Whether ``name`` comes through the training package itself, its public surface: a
        unit with an edge into training may import it there."""

        if name == "cti.training":
            return True
        child = name.removeprefix("cti.training.").partition(".")[0]
        return name.startswith("cti.training.") and not package.is_module(f"training/{child}")


@dataclass(frozen=True)
class InferenceInternals:
    """I1-inference: inference modules no other package imports; the runner serves the
    composition root alone, prefix_state the cache alone, and the inference engine reaches
    neither the cache nor kvcache."""

    PRIVATE: ClassVar[tuple[str, ...]] = (
        "cti.inference._control._cache",
        "cti.inference._control._prefix_state",
        "cti.inference._control._scheduling",
    )
    RUNNER: ClassVar[tuple[str, ...]] = ("cti.inference._execution.runner",)
    ROOT: ClassVar[str] = "runtime/_core.py"
    PREFIX_STATE: ClassVar[tuple[str, ...]] = ("cti.inference._control._prefix_state",)
    PREFIX_STATE_READER: ClassVar[str] = "inference/_control/_cache.py"
    ENGINE: ClassVar[str] = "inference/_control/_engine.py"
    BELOW_THE_ENGINE: ClassVar[tuple[str, ...]] = (
        "cti.inference._control._cache",
        "cti.kvcache",
    )

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        inside = module.path.startswith("inference/")
        reader = module.path == self.PREFIX_STATE_READER
        for qualname, name, _ in package.imported(module):
            target = package.defining(name)
            if (
                (not inside and under(target, self.PRIVATE))
                or (not inside and module.path != self.ROOT and under(target, self.RUNNER))
                or (inside and not reader and under(target, self.PREFIX_STATE))
                or (module.path == self.ENGINE and under(target, self.BELOW_THE_ENGINE))
            ):
                yield qualname, target


@dataclass(frozen=True)
class InferenceSplit:
    """I1-inference-split: neither half of inference imports the other, at run time or in
    annotations; they meet in ``_step``."""

    OTHER: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "inference/_control/": "cti.inference._execution",
            "inference/_execution/": "cti.inference._control",
        }
    )

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        other = next(
            (half for path, half in self.OTHER.items() if module.path.startswith(path)), None
        )
        if other is None:
            return
        for qualname, name, _ in package.imported(module):
            target = package.defining(name)
            if under(target, (other,)):
                yield qualname, target


@dataclass(frozen=True)
class TrainingInternals:
    """I1-training: outside training only the package and these modules are imported; the
    runner, by the composition root alone."""

    PUBLIC: ClassVar[frozenset[str]] = frozenset({"scheduler"})
    RUNNER: ClassVar[str] = "runner"
    ROOT: ClassVar[str] = "runtime/_core.py"

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        if module.path.startswith("training/"):
            return
        public = self.PUBLIC | ({self.RUNNER} if module.path == self.ROOT else set())
        for qualname, name, _ in package.imported(module):
            child = name.removeprefix("cti.training.").partition(".")[0]
            if (
                name.startswith("cti.training.")
                and child not in public
                and package.is_module(f"training/{child}")
            ):
                yield qualname, name


class RunnerReach:
    """I2-reach: composition code that reads two attributes deep off an InferenceRunner (or
    what ``build_executor`` returns), directly or through a name assigned the first read."""

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        runner = Types(module, "InferenceRunner", factories=("build_executor",))
        reads = {  # names assigned one read off a runner
            (lexical(scope), target): node.value.attr
            for scope, node in module.nodes
            if isinstance(node, ast.Assign | ast.AnnAssign)
            and isinstance(node.value, ast.Attribute)
            and runner.holds(scope, node.value.value)
            for target in bound_names(node)
        }
        for scope, node in module.nodes:
            if not isinstance(node, ast.Attribute):
                continue
            inner = node.value
            if isinstance(inner, ast.Attribute) and runner.holds(scope, inner.value):
                yield scope, f"{inner.attr}.{node.attr}"
            elif isinstance(inner, ast.Name) and (first := reads.get((lexical(scope), inner.id))):
                yield scope, f"{first}.{node.attr}"


# I3.


@dataclass(frozen=True)
class PrivateAccess:
    """I3-private: private names one module takes from another: imports of ``_name``;
    ``x._name`` unless ``x`` is ``self``, ``cls``, ``super()``, a library module, or one of
    the module's (non-Protocol) classes or an instance of one (``Types``); ``_name=``
    keywords unless the callee is a function or class the module defines or a method of such
    a receiver; and Protocols declaring ``_`` members."""

    OWN: ClassVar[frozenset[str]] = frozenset({"self", "cls"})

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        classes = [node for _, node in module.nodes if isinstance(node, ast.ClassDef)]
        owned = [Types(module, node.name) for node in classes if not protocol(module, node)]
        defined = {
            node.name
            for _, node in module.nodes
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }
        for token in module.tokens:
            source = token.chain.rpartition(".")[0]
            ours = token.kind == "import" and under(token.chain, ("cti",))
            if ours and private(token.ident) and source != module.name:
                yield token.qualname, token.chain
        for scope, node in module.nodes:
            if isinstance(node, ast.ClassDef) and protocol(module, node):
                names = set().union(*map(bound_names, node.body))
                yield from ((scope, name) for name in sorted(names) if private(name))
            elif isinstance(node, ast.Attribute) and private(node.attr):
                if not self._own(module, scope, node.value, owned):
                    yield scope, node.attr
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    own = func.id in defined
                else:
                    receiver = func.value if isinstance(func, ast.Attribute) else func
                    own = self._own(module, scope, receiver, owned)
                if not own:
                    yield from (
                        (scope, f"{item.arg}=") for item in node.keywords if private(item.arg or "")
                    )

    def _own(self, module: Module, scope: str, receiver: ast.expr, owned: list[Types]) -> bool:
        if isinstance(receiver, ast.Call) and module.ident(receiver.func) == "super":
            return True
        if isinstance(receiver, ast.Name):
            chain = module.chain(receiver)
            if receiver.id in self.OWN or not (chain.startswith("?") or under(chain, ("cti",))):
                return True
        return any(types.holds(scope, receiver) for types in owned)


@dataclass(frozen=True)
class Facade:
    """One row of the facade table: a public module, the names it exports (``__all__`` exactly,
    space-separated; None: the module's own ``__all__``, unchecked) and the units that may
    import them, as module names under ``cti`` (a unit covers its submodules)."""

    module: str
    exported: str | None
    users: tuple[str, ...]

    @property
    def names(self) -> frozenset[str] | None:
        return None if self.exported is None else frozenset(self.exported.split())


ALL_UNITS = ("",)


# The facade table, one row per public module.
FACADES: tuple[Facade, ...] = (
    Facade(
        "runtime",
        "EngineClient TrainingSession TrainingFuture EngineProcessError FailurePhase",
        ("llm", "server"),
    ),
    Facade("worker", "GPUWorker", RUNTIME_RANK),
    Facade(
        "inference",
        "build_executor InferenceEngine RequestSeeds load_speculative ServingLimits Executor TextDecoder"
        " HostedModels DeviceBudget ResidentBeside Forward PageMap RequestTable Round"
        " RoundCompletion Sampled GraphStats",
        RUNTIME_RANK,
    ),
    Facade(
        "training",
        "TrainingBatch PaddedTokens PackedTokens SFTDatum SFT GRPO ForwardBackwardCompletion"
        " ForwardBackwardObservation OptimizerStepResult TrainingCheckpoint"
        " TrainingVerb Refusal CommandStateError TrainingInputError InvalidGradientError"
        " SwitchRejectedError CheckpointFormatError",
        ("runtime", "llm", "server"),
    ),
    Facade("training.scheduler", "TrainingScheduler TrainingReplies", RUNTIME_RANK),
    Facade(
        "training.runner",
        "TrainingRunner TrainingAllocation SupportsParameterUniverse ModelParameterUniverse"
        " TrainingMemoryPlan prepare_model_training device_stack_limit",
        RUNTIME_RANK,
    ),
    Facade(
        "protocol",
        "AbortBackendMsg AbortMsg AdmissionPriority BaseBackendMsg BaseFrontendMsg"
        " BaseTokenizerMsg ChatImagePart ChatPrompt ChatTextPart DetokenizeMsg GenerationOutput"
        " InferenceReply Prompt PromptReply RejectionKind RequestAbortedMsg RequestRejectedError"
        " RequestEnd RequestRejectedMsg SampledSequence SamplingParams SamplingPrompt SamplingResult"
        " SamplingSpec ServingEndpoints TextPrompt TokenPrompt TokenUsage TokenizeMsg UserMsg"
        " UserReply context_length_exceeded",
        ("runtime", "inference", "tokenizer", "llm", "server", "training"),
    ),
    Facade(
        "protocol.link",
        "ZmqAsyncPullQueue ZmqAsyncPushQueue ZmqPullQueue ZmqPushQueue ZmqRequestTransport",
        ("runtime", "tokenizer", "server"),
    ),
    Facade("config", None, ALL_UNITS),
    Facade("plan", None, ("resolution", "worker", *RUNTIME_RANK, "inference")),
    Facade("resolution", None, RUNTIME_RANK),
    Facade(
        "models",
        "get_model_class create_model read_checkpoint CheckpointFacts CheckpointSource peft_lora"
        " open_weights"
        " load_peft save_peft model_signature BaseLLMModel ModelCapability ModelConfig"
        " FeatureRequest"
        " TokenHead",
        ("resolution", "worker", "inference", "plan"),
    ),
    Facade("models.training_contract", None, ("training",)),
    Facade("models.activation_checkpointing", None, ("training",)),
    Facade("models.dflash", "DFlashConfig DFlashDraftModel DFlash2DraftModel", ("inference",)),
    Facade(
        "layers",
        "Adapter AttentionLayer BaseOP BidirectionalAttention BlockConv ExpertParallel"
        " ExpertPlacement ExpertTensorParallel FeatureTaps LayerNorm LayerServices"
        " LinearColParallelMerged LinearQKVMerged LinearReplicated LinearRowParallel"
        " MoELayer OPList ParallelLMHead Param Placement PrologueFactory RMSNorm RMSNormFused"
        " ResidualStreamRMSNorm RotaryEmbedding SiluAndMul Site VisionRotaryEmbedding"
        " VocabParallelEmbedding prepare_inference",
        ("models", "worker", "resolution", "inference", "plan"),
    ),
    Facade("layers.quant", "CheckpointFormats", ("models",)),
    Facade("layers.gated_delta_net", "GatedDeltaNet prefill_exports_states", ("models",)),
    Facade(
        "attention",
        "AttentionRequest AttentionRequirement BaseAttnBackend BaseAttnMetadata"
        " FixedQueryGeometry FixedQueryShape MixedQuery ResolvedFamily SupportsFixedQuery"
        " create_fixed_query_backends fixed_query_bytes resolve_attention_config",
        # models states its AttentionRequirement.
        ("resolution", "inference", "layers", "plan", "forward", "models"),
    ),
    # The value leaf: the MHA KV-head rule the QKV projection shards by.
    Facade(
        "attention.requirement", "AttentionRequirement kv_head_window qo_head_window", ("layers",)
    ),
    Facade(
        "kvcache",
        "BaseCacheHandle InferenceState PageGroup RecurrentRows RecurrentStatePool"
        " RestorePlan SlotStateSpec SnapshotPolicy Spec StateCapacity StateData StateRowIndex"
        " StateSlot TableGeometry create_prefix_cache next_snapshot_position StateBudget"
        " plan_capacity CapacityError PageTable choose_page_size"
        # Row layouts, KV groups and layer binding.
        " RowLayout GroupRole SlotKey ResolvedSlot GroupLayout KVLayout StateLayer BoundSlot"
        " PagedSlot group_layers",
        ("inference", "worker", "resolution", "layers", "attention", "recurrent", "plan"),
    ),
    Facade(
        "recurrent",
        "RecurrentBackend RecurrentLayer RecurrentMetadata RecurrentPhase RecurrentVerify"
        " RecurrentGeometry LayerReplay RoundTransaction RecurrentRounds RoundStatus",
        ("inference", "layers", "models", "forward"),
    ),
    Facade(
        "moe", "BaseMoeBackend SparseBlock TrainingBlock ExpertLoRA create_moe_backend", ("layers",)
    ),
    Facade("forward", None, ("models", "layers", "attention", "recurrent", "inference", "plan")),
    Facade(
        "memory",
        "MemoryManager MemoryProvider GraphPool Mode ResidencyRestoreError native_backing"
        " configure_memory_saver_subprocess free_bytes unified_memory",
        ("runtime", "worker", "inference", "recurrent", "training"),
    ),
    Facade(
        "distributed",
        "Topology CollectiveGroup ParallelGroup",
        ("worker", "layers", "inference", "training"),
    ),
    Facade(
        "multimodal",
        "ImageLimits ImageSpan ImageInputs ImagePreprocessing ProcessedPrompt RopePositionLayout"
        " InputProcessor create_input_processor rope_layout read_image_preprocessing",
        ("tokenizer", "protocol", "resolution", "models", "inference", "runtime", "plan"),
    ),
    Facade(
        "tokenizer",
        "PromptEncoder IncrementalDetokenizer load_text_decoder tokenize_worker",
        ("llm", "runtime"),
    ),
    Facade("server", "launch_server OnlineServer", ("__main__",)),
    Facade("env", None, RUNTIME_RANK),
    Facade("sampling", None, ALL_UNITS),
    Facade("utils", None, ALL_UNITS),
)


@dataclass(frozen=True)
class FacadeImports:
    """I1-facades: from outside its unit, a cti name comes through a package facade
    that exports it, into a unit its row names; each facade exports exactly its row's names."""

    table: tuple[Facade, ...]

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        rows = {row.module: row for row in self.table}
        row = rows.get(module.name.removeprefix(f"{package.root}.") or "")
        if module.path.endswith("__init__.py") and row is not None and row.names is not None:
            found = exports(module)
            yield from (("__all__", f"+{name}") for name in sorted(found - row.names))
            yield from (("__all__", f"-{name}") for name in sorted(row.names - found))
        importer = module.name.removeprefix(f"{package.root}.")
        for token in module.tokens:
            if token.kind == "name" or not under(token.chain, (package.root,)):
                continue
            target = token.chain.removeprefix(f"{package.root}.")
            if target.split(".")[0] == importer.split(".")[0]:
                continue  # inside the unit
            if token.kind == "attribute":
                # A name read off an imported facade module: the row's next name.
                found = max((row for row in rows if under(target, (row,))), key=len, default=None)
                if found is None or found == target:
                    continue
                target = f"{found}.{target.removeprefix(f'{found}.').split('.')[0]}"
            if not self._allowed(rows, target, importer):
                yield token.qualname, f"{package.root}.{target}"

    @staticmethod
    def _allowed(rows: Mapping[str, Facade], target: str, importer: str) -> bool:
        source, _, name = target.rpartition(".")
        for module, exported in ((target, None), (source, name)):
            row = rows.get(module)
            if row is None or not under(importer, row.users) and row.users != ALL_UNITS:
                continue
            if exported is None or row.names is None or exported in row.names:
                return True
        return False


@dataclass(frozen=True)
class PrivateModules:
    """I1-private-modules: a module whose name starts with ``_`` serves its own package; from
    outside the package, its names come through the package entry (``__init__``)."""

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        for token in module.tokens:
            parts = token.chain.split(".")
            if token.kind != "import" or parts[0] != package.root:
                continue
            index = next(
                (
                    i
                    for i, part in enumerate(parts[1:], 1)
                    if private(part) and package.is_module("/".join(parts[1 : i + 1]))
                ),
                None,
            )
            if index is not None and not under(module.name, (".".join(parts[:index]),)):
                yield token.qualname, ".".join(parts[: index + 1])


@dataclass(frozen=True)
class UtilsLevel:
    """I2-utils-level: a helper lives at the nearest common owner of its users. A name
    ``utils/`` defines is imported by at least two top-level units; a name a package's
    ``_utils`` defines, by at least two of that package's modules; ``utils/`` imports no
    other cti unit."""

    SHARED: ClassVar[str] = "utils/"
    LOCAL: ClassVar[str] = "_utils"

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        shared = module.path.startswith(self.SHARED)
        folder = self.SHARED if shared else self._package_of(module.path)
        if folder is None:
            return
        if shared:
            for qualname, name in Imports(package.root)(package, module):
                if not name.startswith(f"{package.root}.utils"):
                    yield qualname, f"imports:{name}"
        users: dict[str, set[str]] = {}
        for other in package.modules.values():
            # Shared helpers count users by top-level unit outside utils; a package's own
            # helpers by module inside the package.
            inside = other.path.startswith(folder)
            if inside == shared or other is module:
                continue
            for token in other.tokens:
                defined = package.defining(token.chain)
                if token.kind == "import" and defined.startswith(f"{module.name}."):
                    user = ImportEdges.unit(other.path) if shared else other.path
                    users.setdefault(defined.rpartition(".")[2], set()).add(user)
        for name in sorted(self._defined(module)):
            if len(users.get(name, ())) < 2:
                yield MODULE_SCOPE, f"sink:{name}"

    def _package_of(self, path: str) -> str | None:
        """The package folder of a ``_utils`` module or package, else None."""

        parts = path.removesuffix(".py").split("/")
        if self.LOCAL not in parts:
            return None
        return "/".join(parts[: parts.index(self.LOCAL)]) + "/"

    @staticmethod
    def _defined(module: Module) -> set[str]:
        """Public names the module defines; imports and type variables are not helpers."""

        return {
            name
            for node in module.tree.body
            if not isinstance(node, ast.Import | ast.ImportFrom)
            and not (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and module.ident(node.value.func) == "TypeVar"
            )
            for name in bound_names(node)
            if not name.startswith("_") and name != "__all__"
        }


@dataclass(frozen=True)
class PrivateInTests:
    """I1-private-tests: a test reaches the engine's underscore modules and names (imported,
    or read off an imported module) only for the package it tests, ``tests/**/<package>/``
    or a folder ``TESTED`` maps to its packages; other tests, ``tests/support`` included,
    go through public interfaces."""

    # Test folders not named after the packages they test.
    TESTED: ClassVar[Mapping[str, frozenset[str]]] = MappingProxyType(
        {
            "modules/config/": frozenset({"config", "resolution"}),
            "modules/execution/": frozenset({"inference", "plan", "resolution", "worker"}),
            # The train/infer switch: the worker's device memory across a KV suspension.
            "gpu/switching/": frozenset({"worker"}),
        }
    )

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        tested = next(
            (owners for folder, owners in self.TESTED.items() if module.path.startswith(folder)),
            frozenset(),
        )
        for token in module.tokens:
            parts = token.chain.split(".")
            if token.kind not in {"import", "attribute"} or parts[0] != "cti":
                continue
            found = [i for i, part in enumerate(parts[1:], 1) if private(part)]
            last = len(parts) - 1
            reached = found and (token.kind == "import" or found[0] == last)
            if reached and parts[1] not in tested and f"/{parts[1]}/" not in f"/{module.path}":
                yield token.qualname, ".".join(parts[: found[0] + 1])


@dataclass(frozen=True)
class RuntimeImports:
    """Runtime imports (annotation-only ones aside, function-level ones included) of
    cti names under ``modules``."""

    modules: tuple[str, ...]

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        for qualname, name, annotation_only in package.imported(module):
            target = package.defining(name)
            if under(target, self.modules) and not annotation_only:
                yield qualname, target


@dataclass(frozen=True)
class LeafImports:
    """I1-leaves: a leaf imports, at run time or in annotations, the standard library,
    ``utils`` (the shared base, which I2 keeps at the bottom) and other leaves alone."""

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        allowed = tuple(f"{package.root}.{unit}" for unit in ("utils", *ImportEdges.LEAVES))
        for scope, node in module.nodes:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.partition(".")[0] in (*sys.stdlib_module_names, "__future__"):
                    continue
                if not under(name, allowed):
                    yield scope, name


@dataclass(frozen=True)
class TextCodecs:
    """I1-text-codec: the execution core imports no tokenizer or transformers frontend;
    ``PretrainedConfig`` is a config."""

    MODULES: ClassVar[tuple[str, ...]] = ("transformers", "cti.tokenizer")
    CONFIG: ClassVar[str] = "transformers.PretrainedConfig"

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        for qualname, chain in Imports(*self.MODULES)(package, module):
            if chain != self.CONFIG:
                yield qualname, chain


def _initialize_raises(package: Package, module: Module) -> Iterator[Hit]:
    """Raise statements in an ``_initialize`` method."""

    for scope, node in module.nodes:
        if isinstance(node, ast.Raise) and scope.split(".")[1:2] == ["_initialize"]:
            yield scope, "raise"


# I5-I6.


class WideSignatures:
    """I5-wide: at most five parameters (self and cls aside, *args and **kwargs one each) in
    the state and attention packages, the attention and GDN layers, create_model and
    plan_capacity; registered contracts excepted. forward_into(q, k, v, layer, batch, out) is
    the MHA backend contract: the method of MHA_BACKEND and its subclasses in attention; the
    kernel operation Protocols' __call__ is the kernel ABI.
    Opposite actions take no flag."""

    SCOPE: ClassVar[tuple[str, ...]] = (
        "kvcache/",
        "attention/",
        "recurrent",
        "layers/attention.py",
        "layers/gated_delta_net/",
        "models/__init__.py",
    )
    LIMIT: ClassVar[int] = 5
    # Files where only the named functions are in scope.
    ONLY: ClassVar[MappingProxyType[str, str]] = MappingProxyType(
        {"models/__init__.py": "create_model"}
    )
    MHA_BACKEND: ClassVar[str] = "BaseAttnBackend"
    EXCEPTIONS: ClassVar[MappingProxyType[str, int]] = MappingProxyType({"forward_into": 6})
    KERNEL_PROTOCOLS: ClassVar[tuple[str, ...]] = ("layers/gated_delta_net/operations.py",)
    OPPOSITES: ClassVar[tuple[str, ...]] = ("un", "de", "dis")

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        only = self.ONLY.get(module.path)
        classes = {scope: node for scope, node in module.nodes if isinstance(node, ast.ClassDef)}
        for scope, node in module.nodes:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if only is not None and scope != only:
                continue
            owner = classes.get(scope.rpartition(".")[0])
            if node.name == "__call__" and module.path in self.KERNEL_PROTOCOLS:
                if owner is not None and protocol(module, owner):
                    continue
            arguments = node.args
            positional = [*arguments.posonlyargs, *arguments.args]
            decorators = {module.ident(item) for item in node.decorator_list}
            receiver = owner is not None and "staticmethod" not in decorators
            if receiver:
                positional = positional[1:]
            keywords = arguments.kwonlyargs
            count = len(positional) + len(keywords) + bool(arguments.vararg) + bool(arguments.kwarg)
            contract = (
                owner is not None
                and module.path.startswith("attention/")
                and owner.name in package.subclasses(self.MHA_BACKEND)
            )
            limit = self.EXCEPTIONS.get(node.name, self.LIMIT) if contract else self.LIMIT
            yield from [(scope, "wide")] * max(count - limit, 0)  # one per extra parameter
            verb = node.name.lstrip("_").partition("_")[0]
            opposites = {f"{prefix}{verb}" for prefix in self.OPPOSITES}  # lock: unlock
            opposites |= {verb.removeprefix(p) for p in self.OPPOSITES if verb.startswith(p)}
            for argument in [*positional, *arguments.kwonlyargs]:
                if argument.arg in opposites - {verb, ""} and verb:
                    yield scope, f"flag:{argument.arg}"


@dataclass(frozen=True)
class TypeDispatch:
    """I6-dispatch: isinstance or issubclass on training batch types outside training, and
    the worker's on model types (tuples and ``|`` unions unfolded)."""

    def __call__(self, package: Package, module: Module) -> Iterator[Hit]:
        outside = not module.path.startswith("training/")
        worker = module.path.startswith("worker")
        for scope, kind in isinstance_targets(module):
            target = package.defining(kind)
            batch = under(target, ("cti.training",)) and target.endswith("Batch")
            model = under(target, ("cti.models",)) or target.endswith("Model")
            if (outside and batch) or (worker and model):
                yield scope, target


class InterfaceGuards:
    """GI's rules, with their probes and ledgers."""

    I1_EDGES: ClassVar[Rule] = Rule(
        "I1-edges",
        ImportEdges(),
        probes=(
            (
                "inference/_control/_cache.py",
                ("import cti.worker", "from cti import worker"),
            ),
            # Only a rank's entry reads env; the parent's side reaches no rank module.
            ("worker/__init__.py", ("from cti.env import ENV",)),
            (
                "runtime/_process.py",
                (
                    "from cti.worker import GPUWorker",
                    "def _f():\n    from cti.runtime._core import EngineCore",
                ),
            ),
            ("multimodal/inputs.py", ("from cti.training import TrainingCheckpoint",)),
            ("kvcache/budget.py", ("from cti.multimodal import ImageInputs",)),
            (
                "forward.py",
                (
                    "from cti.multimodal import ImageInputs",
                    "if TYPE_CHECKING:\n    from cti.multimodal import ImageInputs",
                ),
            ),
            ("attention/__init__.py", ('_FAMILIES = {"mha": ("cti.worker", "Mha")}',)),
            # The server speaks to the engine only through its own row.
            ("server/args.py", ("from cti.engine import batch",)),
            (
                "server/launch.py",
                (
                    "from cti.inference import SamplingSpec",
                    "def _f():\n    from cti.tokenizer.server import tokenize_worker",
                ),
            ),
        ),
        ledger="""
runtime/_client.py
    cti.runtime._control.ControlChannel
    cti.runtime._control.Follower
    cti.runtime._control.Leader
    cti.runtime._core.run_engine_core
""",
    )
    I1_INFERENCE: ClassVar[Rule] = Rule(
        "I1-inference",
        InferenceInternals(),
        probes=(
            (
                "forward.py",
                ("from cti.inference._control._scheduling.request import Request",),
            ),
            ("inference/_control/_engine.py", ("from cti.kvcache import state",)),
            (
                "llm/llm.py",
                ("from cti.inference._execution.runner import InferenceRunner",),
            ),
        ),
        ledger="""
inference/_control/_engine.py
    cti.inference._control._cache.CacheManager
""",
    )
    I1_INFERENCE_SPLIT: ClassVar[Rule] = Rule(
        "I1-inference-split",
        InferenceSplit(),
        probes=(
            (
                "inference/_control/_engine.py",
                (
                    "from cti.inference._execution.runner import InferenceRunner",
                    "if TYPE_CHECKING:\n    from cti.inference import HostedModels",
                ),
            ),
            (
                "inference/_execution/runner.py",
                (
                    "from cti.inference._control._scheduler import InferenceScheduler",
                    "def _f():\n    from cti.inference import InferenceEngine",
                ),
            ),
        ),
        scope=("inference/_control/", "inference/_execution/"),
    )
    I1_TRAINING: ClassVar[Rule] = Rule(
        "I1-training",
        TrainingInternals(),
        probes=(
            ("worker/__init__.py", ("from cti.training._optimizer import AdamW",)),
            ("llm/llm.py", ("from cti.training.runner import TrainingRunner",)),
        ),
    )
    # The optimizer-state contract and the checkpoint do not reach live training owners.
    I1_TRAINING_CONTRACTS: ClassVar[Rule] = Rule(
        "I1-training-contracts",
        Imports(
            *(
                f"cti.training.{owner}"
                for owner in ("_optimizer", "_parameters", "runner", "scheduler")
            )
        ),
        probes=(("training/_checkpoint.py", ("from cti.training._optimizer import X",)),),
        scope=("training/_checkpoint.py",),
    )
    # A debt: GDN replay's GraphPool.
    I1_FACADES: ClassVar[Rule] = Rule(
        "I1-facades",
        FacadeImports(FACADES),
        probes=(
            ("llm/llm.py", ("from cti.training._checkpoint import TrainingCheckpoint",)),
            # A facade name, into a unit its row does not name.
            ("tokenizer/prompt.py", ("from cti.worker import GPUWorker",)),
            # A facade exporting a name its row does not list.
            ("worker/__init__.py", ('__all__ = ["GPUWorker", "DeviceMemory"]',)),
        ),
        ledger="""
layers/gated_delta_net/replay.py
    cti.memory.GraphPool
""",
    )
    I1_PRIVATE_MODULES: ClassVar[Rule] = Rule(
        "I1-private-modules",
        PrivateModules(),
        probes=(("llm/llm.py", ("from cti.protocol._codec import Codec",)),),
    )
    # A leaf imports the standard library, utils and other leaves alone.
    I1_LEAVES: ClassVar[Rule] = Rule(
        "I1-leaves",
        LeafImports(),
        probes=(
            ("sampling.py", ("import torch",)),
            ("sampling.py", ("from cti.models import LoRAConfig",)),
        ),
        scope=tuple(f"{leaf}.py" for leaf in ImportEdges.LEAVES),
    )
    # The public configuration names no model or training internals.
    I1_CONFIG: ClassVar[Rule] = Rule(
        "I1-config",
        RuntimeImports(("cti.models", "cti.training")),
        probes=(("config.py", ("def _f():\n    from cti.models import config",)),),
        scope=("config.py",),
    )
    I1_TEXT_CODEC: ClassVar[Rule] = Rule(
        "I1-text-codec",
        TextCodecs(),
        probes=(("inference/_execution/runner.py", ("from transformers import AutoProcessor",)),),
        scope=("inference/", "plan.py", "resolution.py", "forward.py", "models/", "worker/"),
        ledger="""
models/checkpoint.py
    transformers.AutoConfig
""",
    )
    I2_REACH: ClassVar[Rule] = Rule(
        "I2-reach",
        RunnerReach(),
        probes=(
            (
                "runtime/_core.py",
                (
                    "def _probe(runner: InferenceRunner):\n    return runner.model.config",
                    "def _probe(self):\n    runner = build_executor()\n"
                    "    model = runner.model\n    return model.config",
                ),
            ),
        ),
        scope=("runtime/_core.py",),
    )
    I2_UTILS_LEVEL: ClassVar[Rule] = Rule(
        "I2-utils-level",
        UtilsLevel(),
        probes=(("utils/misc.py", ("def lonely_helper(): ...", "from cti import forward")),),
        ledger="""
utils/arch.py
    sink:get_device_capability
""",
    )
    I3_PRIVATE: ClassVar[Rule] = Rule(
        "I3-private",
        PrivateAccess(),
        probes=(
            (
                "protocol/_request.py",
                (
                    "def _f(other):\n    return other._secret",
                    "from cti.inference._control._scheduling.request import _private",
                    "def _f(other):\n    other.method(_flag=True)",
                    "class _Port(Protocol):\n    def _hook(self): ...",
                ),
            ),
        ),
        ledger="""
layers/base.py
    _tensors 3
models/activation_checkpointing.py
    _version 2
runtime/_training.py
    _set_error
    _set_result
training/_boundary_offload.py
    _version 2
utils/logger.py
    _which= 3
""",
    )
    I4_ADDRESSING: ClassVar[Rule] = Rule(
        "I4-addressing",
        Names(r"^(cache_write_indices|cache_index_table)$"),
        probes=(("models/base.py", ("_x = _batch.cache_write_indices",)),),
        scope=(
            "inference/_execution/",
            "inference/_step.py",
            "plan.py",
            "resolution.py",
            "forward.py",
            "models/",
        ),
        owners=("inference/_execution/_speculative/",),
    )
    I4_ATTENTION_METADATA: ClassVar[Rule] = Rule(
        "I4-attn-metadata",
        Names(r"^attn_metadata$"),
        probes=(("models/base.py", ("_x = _batch.attn_metadata",)),),
        owners=("attention/",),
        ledger="""
forward.py
    attn_metadata
inference/_execution/_speculative/dflash.py
    attn_metadata 4
""",
    )
    I4_RECURRENT_METADATA: ClassVar[Rule] = Rule(
        "I4-recurrent-metadata",
        Names(r"^(RecurrentMetadata|recurrent_metadata)$"),
        probes=(("inference/_execution/runner.py", ("_batch.recurrent_metadata = None",)),),
        owners=("recurrent",),
        ledger="""
forward.py
    RecurrentMetadata 2
    recurrent_metadata
inference/_execution/_speculative/dflash.py
    recurrent_metadata 3
layers/gated_delta_net/layer.py
    RecurrentMetadata 2
    recurrent_metadata
""",
    )
    I4_STATE_FIELDS: ClassVar[Rule] = Rule(
        "I4-state-fields",
        Names(r"^(restore_row|captures|exports|capture_row|StateCapture)$"),
        probes=(("inference/_control/_scheduling/request.py", ("_x = _row.restore_row",)),),
        scope=("inference/_control/_scheduling/", "inference/_control/_engine.py"),
        ledger="""
inference/_control/_scheduling/prefill.py
    exports 5
inference/_control/_scheduling/rounds.py
    captures 3
""",
    )
    I4_LOADER: ClassVar[Rule] = Rule(
        "I4-loader",
        Names(r"^load_weight$"),
        probes=(("models/base.py", ("_layer.load_weight(None)",)),),
        owners=("models/weight.py",),
    )
    I4_PROVIDER: ClassVar[Rule] = Rule(
        "I4-provider",
        Names(r"^provider$"),
        probes=(("layers/linear.py", ('_x = _config.provider == "a"',)),),
        scope=("layers/", "recurrent"),
        # Each weight format's per-device provider table.
        owners=("layers/quant/",),
        ledger="""
layers/gated_delta_net/operations.py
    provider 2
layers/services.py
    provider
""",
    )
    I4_INVOKE_ATTENTION: ClassVar[Rule] = Rule(
        "I4-invoke-attention",
        Names(r"^invoke_attention$"),
        probes=(("models/base.py", ("_layer.invoke_attention()",)),),
        scope=("models/",),
        ledger="",
    )
    I4_CHAT_TEMPLATE: ClassVar[Rule] = Rule(
        "I4-chat-template",
        Names(r"^apply_chat_template$"),
        probes=(("tokenizer/__init__.py", ("_tokenizer.apply_chat_template([])",)),),
        owners=("multimodal/",),
        ledger="""
tokenizer/prompt.py
    apply_chat_template
""",
    )
    I4_GLOBALS: ClassVar[Rule] = Rule(
        "I4-globals",
        Names(r"^globals$"),
        probes=(("protocol/_usage.py", ("_names = globals()",)),),
        scope=("protocol/",),
        ledger="",
    )
    # Only the worker (building groups) and the layers (running collectives) import
    # distributed.
    I4_DISTRIBUTED_EDGES: ClassVar[Rule] = Rule(
        "I4-distributed-edges",
        Imports("cti.distributed"),
        probes=(("inference/_control/_engine.py", ("from cti.distributed import ParallelGroup",)),),
        owners=("worker/", "layers/", "distributed/"),
        ledger="""
inference/_execution/runner.py
    cti.distributed.ParallelGroup
""",
    )
    I4_INITIALIZE_RAISES: ClassVar[Rule] = Rule(
        "I4-initialize-raises",
        _initialize_raises,
        probes=(
            (
                "worker/__init__.py",
                ("class _W:\n    def _initialize(self):\n        raise ValueError",),
            ),
        ),
        scope=("inference/_execution/runner.py", "worker/"),
    )
    I5_WIDE: ClassVar[Rule] = Rule(
        "I5-wide",
        WideSignatures(),
        probes=(
            (
                "kvcache/state.py",
                ("def _wide(a, b, c, d, e, f): ...", "def _lock(x, *, unlock): ..."),
            ),
        ),
        scope=WideSignatures.SCOPE,
        ledger="""
layers/gated_delta_net/layer.py
    wide
""",
    )
    I6_DISPATCH: ClassVar[Rule] = Rule(
        "I6-dispatch",
        TypeDispatch(),
        probes=(
            (
                "inference/_control/_engine.py",
                (
                    "from cti.training import PackedSFTBatch\n"
                    "def _f(b):\n    return isinstance(b, PackedSFTBatch)",
                ),
            ),
        ),
    )
    # One owner per concern: threads and offline mode named where they belong; the prefix
    # reuse resolvers gone.
    I7_STRUCTURE: ClassVar[Rule] = Rule(
        "I7-structure",
        _structure_names,
        probes=(
            ("worker/__init__.py", ("import threading\n_t = threading.Thread(target=print)",)),
        ),
        ledger="""
inference/_execution/runner.py
    prefix_cache
models/_download.py
    Thread 2
plan.py
    prefix_cache
resolution.py
    prefix_cache 3
runtime/_client.py
    Thread 2
runtime/_process.py
    Thread 2
server/online_server.py
    Thread
""",
    )
    I7_INFERENCE_PROCESSES: ClassVar[Rule] = Rule(
        "I7-inference-processes",
        Imports("zmq", "multiprocessing", "threading", "cti.protocol.link"),
        probes=(
            (
                "inference/_control/_engine.py",
                ("import zmq", "from cti.protocol.link import ZmqRequestTransport"),
            ),
        ),
        scope=("inference/",),
    )
    # The offline facade samples through the runtime facade: no request link of its own.
    I7_LLM_LINK: ClassVar[Rule] = Rule(
        "I7-llm-link",
        Imports("zmq", "cti.protocol.link"),
        probes=(("llm/llm.py", ("import zmq", "from cti.protocol.link import ZmqPushQueue")),),
        scope=("llm/",),
    )

    @classmethod
    def rules(cls) -> tuple[Rule, ...]:
        return tuple(value for value in vars(cls).values() if isinstance(value, Rule))


@pytest.mark.parametrize("rule", InterfaceGuards.rules(), ids=lambda rule: rule.name)
def test_positions_match_the_ledger(rule: Rule, package: Package) -> None:
    rule.check(package)


@pytest.mark.parametrize("rule", InterfaceGuards.rules(), ids=lambda rule: rule.name)
def test_rule_catches_its_probes(rule: Rule, package: Package) -> None:
    rule.check_probes(package)


@pytest.mark.parametrize(
    "source",
    [
        "class _A:\n    def f(self):\n        return self._x",
        "class _A:\n    _m = 1\ndef f(a: _A):\n    return a._m",
        "class _A:\n    def f(self):\n        return super()._x",
        "def _g(*, _flag):\n    pass\ndef f():\n    _g(_flag=1)",
        "import torch\ntorch._C",
    ],
)
def test_private_access_to_what_the_module_owns_is_free(source: str, package: Package) -> None:
    assert not list(PrivateAccess()(package, package.parse("inference/_control/_cache.py", source)))


@pytest.mark.parametrize("path", ["plan.py", "inference/_execution/runner.py"])
def test_execution_core_annotates_with_multimodal_freely(path: str, package: Package) -> None:
    source = (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
        "    from cti.multimodal import ImageInputs"
    )
    assert not list(ImportEdges()(package, package.parse(path, source)))


def test_every_unit_has_a_row(package: Package) -> None:
    units = {ImportEdges.unit(path) for path in package.modules} - {"__init__"}
    rows = {row.unit for row in ImportEdges.EDGES}
    assert units <= rows


class SuiteGuards:
    """Rules over the test tree itself (paths relative to ``tests/``)."""

    I1_PRIVATE_TESTS: ClassVar[Rule] = Rule(
        "I1-private-tests",
        PrivateInTests(),
        probes=(("support/paths.py", ("from cti.protocol._codec import X",)),),
        # Runtime test support: support/gpu.py sets a worker-building test
        # process up as a rank (runtime._core.configure_rank_process);
        # modules/runtime/test_training_sessions.py runs the mode switch over the scheduler
        # with training's own metrics. integration/runtime/test_tp_gloo.py white-boxes the
        # execution half's InferenceRunner rank by rank.
        ledger="""
integration/runtime/test_tp_gloo.py
    cti.inference._execution
modules/models/test_activation_checkpointing.py
    cti.training._boundary_offload
modules/models/test_qwen3_5_training.py
    cti.training._batch
    cti.training._objective
modules/runtime/test_training_sessions.py
    cti.training._objective 2
support/gpu.py
    cti.runtime._core
support/host_offload.py
    cti.training._boundary_offload
    cti.training._host_memory
support/training.py
    cti.training._checkpoint
    cti.training._objective
""",
    )

    @classmethod
    def rules(cls) -> tuple[Rule, ...]:
        return (cls.I1_PRIVATE_TESTS,)


@pytest.fixture(scope="module")
def suite() -> Package:
    return Package.scan(REPO_ROOT / "tests", "tests")


@pytest.mark.parametrize("rule", SuiteGuards.rules(), ids=lambda rule: rule.name)
def test_suite_positions_match_the_ledger(rule: Rule, suite: Package) -> None:
    rule.check(suite)
    rule.check_probes(suite)
