"""Build and load a model's rank-local parameters and rebuild single-rank ones from them."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from cti.distributed import Topology
from cti.layers import ExpertParallel, LayerServices, Site
from cti.layers.gated_delta_net.operations import GatedDeltaNetOps
from cti.layers.placement import WeightShard
from cti.models import create_model, get_model_class
from cti.models.weight import load_weight
from tests.support import reference_ops, training_ops
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.gated_delta_net import (
    ReferenceGatedDeltaNetDecode,
    ReferenceGatedDeltaNetPrefill,
    ReferenceGatedDeltaNetTraining,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from cti.distributed import CollectiveGroup
    from cti.layers import Adapter, BaseOP
    from cti.layers.services import PrologueGeometry
    from cti.models.base import BaseLLMModel
    from cti.moe import BaseMoeBackend
    from cti.multimodal import ImagePreprocessing

CPU = torch.device("cpu")


@dataclass(frozen=True, slots=True)
class DryGroup:
    """A group with a rank and a size and no communication: its collectives raise."""

    rank: int
    size: int

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("a dry group does not communicate")

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("a dry group does not communicate")

    def residual_reduction(self, **geometry: Any) -> Any:
        return _DryReduction()


class _DryReduction:
    """A dry group's residual reduction: it never runs."""

    def __call__(self, *args: Any) -> Any:
        raise RuntimeError("a dry group does not communicate")

    def prepare(self, token_counts: Any) -> None:
        pass


class ReferenceLayerServices(LayerServices):
    """``LayerServices`` whose kernels are the CPU references, in any dtype, and whose MoE
    backend is ``moe`` on every device (none: a build that only reads shapes)."""

    def __init__(self, moe: BaseMoeBackend | None = None) -> None:
        super().__init__()
        self._moe = moe

    def moe(self, device: torch.device) -> BaseMoeBackend | None:
        return self._moe

    def attention_prologue(
        self,
        device: torch.device,
        geometry: PrologueGeometry,
        *,
        required: bool,
        norm_dtype: torch.dtype,
    ) -> Any:
        # The reference composes: a layer takes it only where it requires the prologue.
        return reference_ops.Prologue(device=device, **asdict(geometry)) if required else None

    def gated_delta_net(
        self, device: torch.device, *, num_key_heads: int, num_value_heads: int, training: bool
    ) -> GatedDeltaNetOps:
        heads = {"num_k_heads": num_key_heads, "num_v_heads": num_value_heads, "device": device}
        return GatedDeltaNetOps(
            ReferenceGatedDeltaNetDecode(**heads),
            ReferenceGatedDeltaNetPrefill(**heads),
            ReferenceGatedDeltaNetTraining(**heads) if training else None,
        )

    def training_loss(self) -> Any:
        return training_ops.LinearCrossEntropyLoss()

    def training_logprobs(self) -> Any:
        return training_ops.LinearSelectedTokenLogprobs()

    def packed_training_attention(self) -> Any:
        return training_ops.CausalVarlenAttention()


def site(
    topology: Topology | None = None,
    *,
    group: CollectiveGroup | None = None,
    moe_backend: BaseMoeBackend | None = None,
    **options: Any,
) -> Site:
    """The site, its buffers on the CPU and its kernels the references, of the rank of a live
    ``group``, or else of rank ``topology`` (a single rank without one) over a group that
    refuses to communicate. ``moe_backend`` is its services' MoE backend; ``options`` override
    ``Site`` fields."""

    if group is None:
        topology = topology or Topology(0, 1)
        group = DryGroup(topology.rank, topology.size)
    else:
        assert topology is None, "a live group has its own rank"
    root = Site(group, ExpertParallel(), CPU, services=ReferenceLayerServices(moe_backend))
    return replace(root, **options)


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Make ``dtype`` torch's default floating dtype inside the block, as the runner does
    while it builds its model."""

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def build_model(
    hf_config: PretrainedConfig,
    *,
    topology: Topology | None = None,
    group: CollectiveGroup | None = None,
    dtype: torch.dtype = torch.float32,
    lora_adapters: Adapter | None = None,
    image_preprocessing: ImagePreprocessing | None = None,
    **options: Any,
) -> BaseLLMModel:
    """Build the engine model at ``site(topology, group=group, **options)`` on the meta
    device, as the runner does; ``model.test_services`` keeps that site's services, which
    a test assigns the attention backend and the pool as the runner does."""

    model_site = site(topology, group=group, **options)
    with torch.device("meta"), default_dtype(dtype):
        model_class = get_model_class(hf_config.architectures[0])
        model = create_model(
            model_class,
            checkpoint_facts(hf_config, image_preprocessing=image_preprocessing),
            site=model_site,
        )
    model.test_services = model_site.services
    if lora_adapters is not None:
        lora_adapters.attach(model)
    # Settled for inference once its adapter is attached, as prepare_inference settles it.
    for _, op in model.named_ops():
        op.settle_inference()
    return model


def windows(model: BaseOP) -> dict[str, tuple[WeightShard, ...]]:
    """The checkpoint windows of every sharded parameter of ``model``, by path; a stacked
    expert tensor's experts read as its window on axis 0."""

    result = {}
    for param in model.params():
        if param.experts is not None:
            result[param.path] = (WeightShard(0, param.experts.start, param.experts.stop),)
        elif param.windows:
            result[param.path] = param.windows
    return result


def load_sharded(
    hf_config: PretrainedConfig,
    folder: Path,
    *,
    dtype: torch.dtype = torch.float32,
    **options: Any,
) -> tuple[BaseLLMModel, dict[str, torch.Tensor], dict[str, tuple[WeightShard, ...]]]:
    """Load ``folder`` through the mapping and windows of ``build_model(hf_config,
    **options)``; return the model, its loaded tensors and its windows."""

    model = build_model(hf_config, dtype=dtype, **options)
    loaded = dict(
        load_weight(str(folder), CPU, mapping=model.checkpoint_mapping, params=model.params())
    )
    # The model's own loader checks every name, shape and dtype against its parameters.
    model.load_state_dict(dict(loaded))
    return model, loaded, windows(model)


def qwen3_checkpoint_shapes(config: PretrainedConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor a dense Qwen3 checkpoint of ``config`` stores, by Hugging Face name."""

    hidden, head_dim = config.hidden_size, config.head_dim
    shapes = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
    }
    if not config.tie_word_embeddings:
        shapes["lm_head.weight"] = (config.vocab_size, hidden)
    query_size = config.num_attention_heads * head_dim
    kv_size = config.num_key_value_heads * head_dim
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        shapes |= {
            f"{prefix}.input_layernorm.weight": (hidden,),
            f"{prefix}.post_attention_layernorm.weight": (hidden,),
            f"{prefix}.self_attn.q_proj.weight": (query_size, hidden),
            f"{prefix}.self_attn.k_proj.weight": (kv_size, hidden),
            f"{prefix}.self_attn.v_proj.weight": (kv_size, hidden),
            f"{prefix}.self_attn.o_proj.weight": (hidden, query_size),
            f"{prefix}.self_attn.q_norm.weight": (head_dim,),
            f"{prefix}.self_attn.k_norm.weight": (head_dim,),
            f"{prefix}.mlp.gate_proj.weight": (config.intermediate_size, hidden),
            f"{prefix}.mlp.up_proj.weight": (config.intermediate_size, hidden),
            f"{prefix}.mlp.down_proj.weight": (hidden, config.intermediate_size),
        }
    return shapes


def assert_tiles(intervals: set[tuple[int, int]], full_size: int) -> None:
    """Distinct rank windows cover [0, full_size) without gaps or overlaps."""

    position = 0
    for start, stop in sorted(intervals):
        assert start == position, f"windows leave a gap or overlap at {position}: {intervals}"
        position = stop
    assert position == full_size


def rebuild_parameter(
    name: str,
    reference: torch.Tensor,
    ranks: Sequence[tuple[Mapping[str, torch.Tensor], Mapping[str, tuple[WeightShard, ...]]]],
) -> torch.Tensor:
    """Reassemble one TP1 parameter from every rank's local tensor and windows alone.

    Each window of a merged parameter tiles its own part; a part the ranks replicate
    must hold the same values on every rank, and padding must be zero.
    """

    windows = [shards.get(name) for _, shards in ranks]
    if all(window is None for window in windows):
        for local, _ in ranks:
            assert torch.equal(local[name], reference), f"{name} is not replicated exactly"
        return reference
    assert all(window is not None for window in windows), f"{name} is sharded on some ranks"
    axis = windows[0][0].axis
    slots = []
    for slot in range(len(windows[0])):
        slot_windows = [window[slot] for window in windows]
        assert all(window.axis == axis for window in slot_windows)
        full_size = max(window.stop for window in slot_windows)
        assert_tiles({(window.start, window.stop) for window in slot_windows}, full_size)
        local = ranks[0][0][name]
        shape = list(local.shape)
        shape[axis] = full_size
        slots.append(
            (
                torch.full(shape, torch.nan, dtype=local.dtype),
                torch.zeros(full_size, dtype=torch.bool),
            )
        )
    for (local, _), rank_windows in zip(ranks, windows, strict=True):
        pieces = local[name].split([window.local_size for window in rank_windows], dim=axis)
        for (full, written), window, piece in zip(slots, rank_windows, pieces, strict=True):
            length = window.stop - window.start
            assert not piece.narrow(axis, length, piece.shape[axis] - length).count_nonzero(), (
                f"{name} padding is not zero"
            )
            target = full.narrow(axis, window.start, length)
            values = piece.narrow(axis, 0, length)
            if written[window.start : window.stop].any():
                assert torch.equal(target, values), f"{name} replicas differ"
            target.copy_(values)
            written[window.start : window.stop] = True
    return torch.cat([full for full, _ in slots], dim=axis)


__all__ = [
    "assert_tiles",
    "DryGroup",
    "build_model",
    "load_sharded",
    "qwen3_checkpoint_shapes",
    "rebuild_parameter",
    "site",
    "windows",
]
