"""The public engine and server options, and their launch-file and command-line spelling.

``EngineOptions`` holds one section per owner and ``ServerOptions`` the online server's;
each public field is defined once, in its section. A launch file is a YAML document whose
root keys are those sections, and the command line has one flag per field path
(``cti.utils.dataclass_tree``), which also type-checks every value; the sections
check only ranges and how their fields relate. Nothing is renamed or aliased.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, TypeAlias, cast, get_args

from cti.utils import DataclassTree


@dataclass(frozen=True, kw_only=True)
class Checkpoint:
    """Load the model's weights from its checkpoint."""


@dataclass(frozen=True, kw_only=True)
class RandomWeights:
    """Draw the model's weights from a standard normal distribution seeded by ``seed``,
    for tests and benchmarks that need no trained model."""

    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")


# The hub a repository id downloads from.
ModelHub = Literal["huggingface", "modelscope"]


@dataclass(frozen=True, kw_only=True)
class ModelOptions:
    """The checkpoint, how its weights are made and the precision it runs in."""

    # A local directory or a repository id on ``source``.
    path: str
    source: ModelHub = "huggingface"
    # The engine executes in bfloat16; ``auto`` also checks that the checkpoint stores BF16,
    # FP16 or FP32 weights, which load converted.
    dtype: Literal["auto", "bfloat16"] = "bfloat16"
    weights: Checkpoint | RandomWeights = Checkpoint()
    # Caps the sequence length below the model's maximum position.
    max_seq_len: int | None = None
    # Seeds the initialization of trainable parts such as LoRA factors.
    initialization_seed: int = 42
    # Load an image-text checkpoint as text only: no vision tower is built or loaded, and
    # requests with images are refused. No effect on a checkpoint without one.
    text_only: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path must not be empty")
        if self.dtype not in ("auto", "bfloat16"):
            raise ValueError(f'dtype must be "auto" or "bfloat16", not {self.dtype!r}')
        if self.max_seq_len is not None and self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive or None")
        if not 0 <= self.initialization_seed < 2**63:
            raise ValueError("initialization_seed must be in [0, 2**63)")


@dataclass(frozen=True, kw_only=True)
class ParallelOptions:
    """How many GPUs one engine spans; tensor and expert parallelism share them. Its ranks
    may span ``num_nodes`` hosts, an equal share on each: every host runs the same command
    but for ``node_rank``, node 0 the one serving, the others joining it at
    ``dist_init_addr`` (``host:port``), as SGLang's ``--nnodes``, ``--node-rank`` and
    ``--dist-init-addr`` do."""

    num_ranks: int = 1
    # A MoE model's routed experts: whole experts split over the ranks, or every expert's
    # intermediate width split over them.
    enable_expert_parallel: bool = True
    num_nodes: int = 1
    node_rank: int = 0
    dist_init_addr: str | None = None

    def __post_init__(self) -> None:
        if self.num_ranks <= 0:
            raise ValueError("num_ranks must be positive")
        if self.num_nodes <= 0 or self.num_ranks % self.num_nodes:
            raise ValueError("num_nodes must be positive and divide num_ranks")
        if not 0 <= self.node_rank < self.num_nodes:
            raise ValueError("node_rank must be in [0, num_nodes)")
        if self.num_nodes > 1:
            host, _, port = (self.dist_init_addr or "").rpartition(":")
            if not host or not port.isdigit() or not 0 < int(port) < 65536:
                raise ValueError("dist_init_addr must be host:port when num_nodes > 1")

    @property
    def ranks_per_node(self) -> int:
        return self.num_ranks // self.num_nodes

    @property
    def node_ranks(self) -> range:
        """The global ranks this node runs."""

        start = self.node_rank * self.ranks_per_node
        return range(start, start + self.ranks_per_node)

    def local_device(self, rank: int) -> int:
        """The CUDA device on its node that global ``rank`` runs on."""

        return rank % self.ranks_per_node


@dataclass(frozen=True, kw_only=True)
class MemoryOptions:
    """Fractions of the GPU memory free at boot."""

    # What weights and the KV cache may use; None is the device's default, resolved at boot:
    # 0.9 on a discrete GPU, 0.6 on unified memory, which the host shares.
    inference_ratio: float | None = None
    # The peak-planning budget; None is the inference ratio.
    peak_ratio: float | None = None
    # Recurrent-state bytes per full-attention KV byte (SGLang's mamba_full_memory_ratio,
    # same default): of the inference budget left after weights, the recurrent rows of
    # running requests and prefix snapshots take ratio / (1 + ratio), the KV cache the rest.
    recurrent_to_kv_ratio: float = 0.9

    def __post_init__(self) -> None:
        if self.inference_ratio is not None and not 0 < self.inference_ratio <= 1:
            raise ValueError("inference_ratio must be in (0, 1]")
        ratio = self.recurrent_to_kv_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not 0 < ratio < math.inf
        ):
            raise ValueError("recurrent_to_kv_ratio must be a positive finite number")
        peak, inference = self.peak_ratio, self.inference_ratio
        if peak is not None and not (0 < peak <= 1 and (inference is None or inference <= peak)):
            # An unset inference_ratio is checked once the device resolves it.
            raise ValueError("peak_ratio must be in [inference_ratio, 1]")


@dataclass(frozen=True, kw_only=True)
class SchedulingOptions:
    """Bounds on what one inference step runs."""

    # None: as many as the inference memory budget holds, at most 256.
    max_running_requests: int | None = None
    # Query tokens one step may carry across its requests.
    max_batch_tokens: int = 8192
    # Prompt tokens one step may carry; rounds and decodes come first. None: an online
    # server's per-device default, else max_batch_tokens.
    max_prompt_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_running_requests is not None and self.max_running_requests <= 0:
            raise ValueError("max_running_requests must be positive or None")
        if self.max_batch_tokens < (self.max_running_requests or 1):
            raise ValueError("max_batch_tokens must be at least max_running_requests")
        if self.max_prompt_tokens is not None and not (
            0 < self.max_prompt_tokens <= self.max_batch_tokens
        ):
            raise ValueError("max_prompt_tokens must be positive and at most max_batch_tokens")


@dataclass(frozen=True, kw_only=True)
class CacheOptions:
    """Which prefix cache serves requests; ``auto`` lets the engine choose for the model."""

    prefix: Literal["auto", "naive", "radix"] = "auto"


# The most speculative tokens a round takes, the bonus token aside.
_MAX_SPECULATIVE_TOKENS = 16


@dataclass(frozen=True, kw_only=True)
class SpeculativeOptions:
    """A speculative checkpoint: requests decode in verified multi-token rounds of
    ``num_speculative_tokens`` proposed tokens each (the bonus token aside, as vLLM counts
    them), 1 to 16, whatever block the checkpoint was trained on. None: the checkpoint's
    default, its trained block's."""

    model_path: str
    num_speculative_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must not be empty")
        count = self.num_speculative_tokens
        if count is not None and (
            type(count) is not int or not 1 <= count <= _MAX_SPECULATIVE_TOKENS
        ):
            raise ValueError(
                f"num_speculative_tokens must be an integer from 1 to "
                f"{_MAX_SPECULATIVE_TOKENS}, not {count!r}"
            )


# "experts" adapts each expert of a sparse block, routed and shared, and never its router or
# shared-expert gate.
LoRATarget = Literal["attention", "linear_attention", "mlp", "experts"]
LORA_TARGETS = frozenset(get_args(LoRATarget))
# The ranks every LoRA takes, and those the "experts" target takes.
LORA_MAX_RANK = 128
EXPERT_LORA_RANKS = range(8, LORA_MAX_RANK + 1, 8)


class ExpertLoRALayout(StrEnum):
    """How the "experts" target's adapter shapes each routed expert's SiLU-gated gate and up
    projections: ``shared_a``, PEFT's adapter on the checkpoint's fused ``gate_up_proj`` (one
    A, the B over both halves' rows), which training uses; ``per_projection``, an A and a B
    on each half, as an adapter on Transformers 4's per-expert ``gate_proj`` and ``up_proj``
    loads, served only. The down projection has one adapter either way."""

    SHARED_A = "shared_a"
    PER_PROJECTION = "per_projection"


@dataclass(frozen=True)
class LoRAConfig:
    """``seed`` draws each factor's initial values from its own state-dict path, so they
    do not depend on construction order, the rank count or the global RNG. ``adapter``, a
    PEFT adapter directory, loads at launch; its targets, rank, alpha and expert layout
    replace these. ``expert_layout`` (``ExpertLoRALayout``) shapes the "experts" target's
    adapter: an imported adapter's format sets it, and training takes ``shared_a`` only."""

    targets: tuple[LoRATarget, ...] = ()
    rank: int = 8
    alpha: float = 16.0
    seed: int = 0
    adapter: str | None = None
    expert_layout: ExpertLoRALayout = ExpertLoRALayout.SHARED_A

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        object.__setattr__(self, "targets", targets)
        if not self.targets and self.adapter is None:
            raise ValueError("LoRA targets must not be empty")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("LoRA targets must be unique")
        invalid = set(self.targets) - LORA_TARGETS
        if invalid:
            raise ValueError(f"Unsupported LoRA targets: {sorted(invalid)}")
        if not 1 <= self.rank <= LORA_MAX_RANK:
            raise ValueError(f"LoRA rank must be in [1, {LORA_MAX_RANK}], not {self.rank}")
        if "experts" in self.targets and self.rank not in EXPERT_LORA_RANKS:
            raise ValueError(
                f"per-expert LoRA takes ranks {EXPERT_LORA_RANKS.start} to "
                f"{EXPERT_LORA_RANKS.stop - 1} in steps of {EXPERT_LORA_RANKS.step}, "
                f"not {self.rank}"
            )
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if self.seed < 0:
            raise ValueError("LoRA seed must be non-negative")
        object.__setattr__(self, "expert_layout", ExpertLoRALayout(self.expert_layout))


@dataclass(frozen=True)
class TrainablePlan:
    """Selection over a model's fixed declaration of non-base parameters; None selects the
    whole declaration."""

    selected_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        names = self.selected_names
        if names is None:
            return
        if not names:
            raise ValueError("selected_names must be non-empty")
        if any(not name or name != name.strip() for name in names):
            raise ValueError("selected names must be non-empty, without surrounding whitespace")
        if len(set(names)) != len(names):
            raise ValueError("selected_names must be unique")

    @classmethod
    def all_declared(cls) -> TrainablePlan:
        """Select every parameter in the model's explicit non-base declaration."""

        return cls()


@dataclass(frozen=True)
class AdamWConfig:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0

    def __post_init__(self) -> None:
        # Stored as floats, the form a training checkpoint records.
        for name in ("lr", "eps", "weight_decay", "max_grad_norm"):
            if (value := getattr(self, name)) is not None:
                if not math.isfinite(value := float(value)):
                    raise ValueError(f"{name} must be finite")
                object.__setattr__(self, name, value)
        object.__setattr__(self, "betas", tuple(float(beta) for beta in self.betas))
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if len(self.betas) != 2:
            raise ValueError("betas must contain exactly two values")
        if not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("AdamW beta values must be in [0, 1)")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


class BatchLayout(StrEnum):
    """Sequence representation supplied to a training objective."""

    PADDED = "padded"
    PACKED = "packed"


@dataclass(frozen=True, slots=True)
class StaticBatching:
    """Split input-order samples into fixed-count microbatches."""

    microbatch_size: int
    max_tokens_per_sample: int

    def __post_init__(self) -> None:
        if self.microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if self.max_tokens_per_sample <= 0:
            raise ValueError("max_tokens_per_sample must be positive")

    @property
    def max_tokens_per_microbatch(self) -> int:
        """The most tokens one microbatch holds: every sample at its limit."""

        return self.microbatch_size * self.max_tokens_per_sample


@dataclass(frozen=True, slots=True)
class DynamicBatching:
    """Split input-order samples under per-microbatch token and sequence budgets.

    The sequence budget bounds per-sequence training memory, such as recurrent chunk
    states, that the startup memory plan reserves.
    """

    max_tokens_per_microbatch: int
    max_sequences_per_microbatch: int = 256

    def __post_init__(self) -> None:
        if self.max_tokens_per_microbatch <= 0:
            raise ValueError("max_tokens_per_microbatch must be positive")
        if self.max_sequences_per_microbatch <= 0:
            raise ValueError("max_sequences_per_microbatch must be positive")


BatchingPolicy: TypeAlias = StaticBatching | DynamicBatching


@dataclass(frozen=True, slots=True)
class TrainingBatching:
    """One explicit valid combination of layout and batching policy."""

    layout: BatchLayout
    policy: BatchingPolicy

    def __post_init__(self) -> None:
        if self.layout is BatchLayout.PADDED and isinstance(self.policy, DynamicBatching):
            raise ValueError("dynamic batching requires packed layout")


@dataclass(frozen=True, slots=True)
class SFTShapeProfile:
    """One exact ordered SFT layout fixed at runner construction."""

    sequence_lengths: tuple[int, ...]
    padded_width: int | None = None
    num_target_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.sequence_lengths:
            raise ValueError("SFT profile sequence lengths must not be empty")
        if any(length < 2 for length in self.sequence_lengths):
            raise ValueError("SFT profile sequence lengths must be at least two")
        padded_width = self.padded_width
        if padded_width is None:
            object.__setattr__(self, "padded_width", max(self.sequence_lengths))
        elif padded_width < max(self.sequence_lengths):
            raise ValueError("SFT profile padded width cannot truncate a sequence")

        max_target_tokens = self.total_tokens - self.batch_size
        if self.num_target_tokens is None:
            object.__setattr__(self, "num_target_tokens", max_target_tokens)
        elif not self.batch_size <= self.num_target_tokens <= max_target_tokens:
            raise ValueError("SFT profile target-token count is outside its layout")

    @property
    def batch_size(self) -> int:
        return len(self.sequence_lengths)

    @property
    def total_tokens(self) -> int:
        return sum(self.sequence_lengths)

    @property
    def has_dense_targets(self) -> bool:
        return self.num_target_tokens == self.total_tokens - self.batch_size

    @property
    def padded_input_tokens(self) -> int:
        """Input tokens of the padded layout, padding included."""

        return self.batch_size * (cast(int, self.padded_width) - 1)


class ActivationCheckpointMode(StrEnum):
    """Supported activation-retention policies for eager training."""

    NONE = "none"
    # Recompute every decoder layer from the first that trains; keep each one's input.
    FULL = "full"
    # As FULL, with each layer's input but the last held in pinned host memory.
    FULL_OFFLOAD = "full_offload"


class TrainingPauseBoundary(StrEnum):
    """Safe training boundary used for an automatic serving pause."""

    LOGICAL_BATCH = "logical_batch"
    MICROBATCH = "microbatch"


class TrainingStateResidency(StrEnum):
    """What training's state leaves the device with while serving holds it."""

    # Nothing: the FP32 masters, AdamW moments and gradients stay.
    RESIDENT = "resident"
    # The AdamW moments, kept in a pinned host mirror refreshed after every optimizer step.
    OFFLOAD_OPTIMIZER = "offload_optimizer"
    # The moments and the FP32 masters, mirrored likewise, and an open accumulation
    # window's gradients, copied to pinned host memory at the switch.
    OFFLOAD_ALL = "offload_all"

    @property
    def offloads_optimizer_state(self) -> bool:
        return self is not TrainingStateResidency.RESIDENT

    @property
    def offloads_masters_and_gradients(self) -> bool:
        return self is TrainingStateResidency.OFFLOAD_ALL


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Construction policy for one worker-owned training runner."""

    trainable_plan: TrainablePlan
    optimizer: AdamWConfig
    batching: TrainingBatching
    activation_checkpointing: ActivationCheckpointMode = ActivationCheckpointMode.NONE
    training_pause_boundary: TrainingPauseBoundary = TrainingPauseBoundary.MICROBATCH
    # None: OFFLOAD_OPTIMIZER on a discrete GPU, RESIDENT where the device shares the host's
    # memory (GB10) or is not a GPU.
    training_state_residency: TrainingStateResidency | None = None
    sft_profile: SFTShapeProfile | None = None


@dataclass(frozen=True, kw_only=True)
class EngineTuning:
    """Internal tuning, set only from Python (tests, benchmarks); not a public option.

    ``attention_backend`` names an attention backend (or a ragged and a decode one,
    comma-separated), ``auto`` letting the engine choose for the device and model. Graph
    sizes default to what the scheduling bounds and free memory allow. Prefix reuse keeps
    ``snapshot_rows`` recurrent snapshots, or as many as ``snapshot_memory_ratio`` of the
    KV budget holds; ``ResolvedModel.check`` validates them with kvcache's budget rule,
    which the parent process does not import.
    """

    attention_backend: str = "auto"
    # None: the smallest page size the selected attention backends take.
    page_size: int | None = None
    cuda_graph_bs: tuple[int, ...] | None = None
    cuda_graph_max_bs: int | None = None
    # Row counts breakable graphs are captured at: None for RowBuckets' default ladder,
    # () for none.
    graph_row_buckets: tuple[int, ...] | None = None
    # Captures the vision tower's graphs at token budgets from the smallest image's tokens
    # up to max_batch_tokens; opt-in until measured.
    vision_graphs: bool = False
    snapshot_rows: int | None = None
    snapshot_memory_ratio: float | None = None
    use_pynccl: bool = True
    distributed_timeout: float = 60.0
    # Forces FlashInfer's tensor-core decode on or off; None chooses by head grouping.
    flashinfer_tensor_cores: bool | None = None

    def __post_init__(self) -> None:
        if not self.attention_backend:
            raise ValueError("attention_backend must not be empty")
        if self.page_size is not None and self.page_size <= 0:
            raise ValueError("page_size must be positive")
        for name, sizes in (
            ("cuda_graph_bs", self.cuda_graph_bs or ()),
            ("graph_row_buckets", self.graph_row_buckets or ()),
        ):
            if any(size <= 0 for size in sizes):
                raise ValueError(f"{name} must hold positive sizes")
        # Zero captures no decode graph.
        if self.cuda_graph_max_bs is not None and self.cuda_graph_max_bs < 0:
            raise ValueError("cuda_graph_max_bs must be non-negative or None")
        if not self.distributed_timeout > 0:
            raise ValueError("distributed_timeout must be positive")


@dataclass(frozen=True, kw_only=True)
class EngineOptions:
    """One engine's public options, one section per owner."""

    model: ModelOptions
    parallel: ParallelOptions = ParallelOptions()
    memory: MemoryOptions = MemoryOptions()
    scheduling: SchedulingOptions = SchedulingOptions()
    cache: CacheOptions = CacheOptions()
    speculative: SpeculativeOptions | None = None
    lora: LoRAConfig | None = None
    training: TrainingConfig | None = None


@dataclass(frozen=True, kw_only=True)
class ServerOptions:
    """Where an online server listens, its tokenizer processes, its image fetches and
    what its logprobs read."""

    host: str = "127.0.0.1"
    port: int = 1919  # 0: one the OS picks, logged when ready
    # Tokenizer processes; a call's prompts go to one of them, preprocessed concurrently.
    num_tokenizers: int = 1
    # Hosts http(s) image URLs may reach, redirects included; empty admits any.
    allowed_media_domains: tuple[str, ...] = ()
    # The directory file:// image URLs may read under; empty refuses them.
    allowed_local_media_path: str = ""
    # Attempts per image URL, counted as vLLM's VLLM_MEDIA_FETCH_MAX_RETRIES (0 is 1).
    media_fetch_max_retries: int = 3
    # The parser of the model's tool calls; auto reads it from its chat template.
    tool_call_parser: Literal["auto", "hermes", "qwen3_coder"] = "auto"

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        if self.num_tokenizers < 1:
            raise ValueError("num_tokenizers must be at least 1")
        if self.allowed_local_media_path and not os.path.isdir(self.allowed_local_media_path):
            raise ValueError("allowed_local_media_path must be an existing directory")
        if self.media_fetch_max_retries < 0:
            raise ValueError("media_fetch_max_retries must be non-negative")


@dataclass(frozen=True)
class LaunchFile:
    """One launch document: ``EngineOptions``' sections and a ``server`` section, each
    absent key keeping its default."""

    engine: EngineOptions = field(metadata={"inline": True})
    server: ServerOptions = ServerOptions()

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> LaunchFile:
        return cls.parse(DataclassTree.read_yaml(path))

    @classmethod
    def parse(
        cls, document: object, overrides: Mapping[tuple[str, ...], object] | None = None
    ) -> LaunchFile:
        """The options ``document`` spells, with each override set at its field path."""

        return DataclassTree(cls).decode(document, overrides)


__all__ = [
    "EXPERT_LORA_RANKS",
    "ExpertLoRALayout",
    "LORA_TARGETS",
    "ActivationCheckpointMode",
    "AdamWConfig",
    "BatchLayout",
    "BatchingPolicy",
    "CacheOptions",
    "Checkpoint",
    "DynamicBatching",
    "EngineOptions",
    "EngineTuning",
    "LaunchFile",
    "LoRAConfig",
    "LoRATarget",
    "MemoryOptions",
    "ModelHub",
    "ModelOptions",
    "ParallelOptions",
    "RandomWeights",
    "SFTShapeProfile",
    "SchedulingOptions",
    "ServerOptions",
    "SpeculativeOptions",
    "StaticBatching",
    "TrainablePlan",
    "TrainingBatching",
    "TrainingConfig",
    "TrainingPauseBoundary",
    "TrainingStateResidency",
]
