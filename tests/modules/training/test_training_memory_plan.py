"""The startup training-memory plan: its components, their residency, and its accuracy.

The accuracy table holds what one RTX PRO 6000 measured (2026-10-22, a cti-kernels
prerelease, torch 2.13, RESIDENT) for padded SFT with every LoRA target adapted,
microbatches of two equal rows: the FP32 master weights and AdamW moments as the bytes they
hold (large segments only they occupy, their own blocks elsewhere), the first window's FP32
gradients likewise, and the requested-bytes peak over one microbatch at
its token count. The plan must
bound each from above, the state components by at most 1.5% or one small segment, and the
activations by at
most 5% (10% where ``_HEADROOM`` says why). Below about 1,000 tokens the accumulation step, whose FP32 gradients the plan
counts separately, peaks above the activations, so the table starts at 1,024.
"""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from cti.config import (
    ActivationCheckpointMode,
    AdamWConfig,
    BatchLayout,
    DynamicBatching,
    LoRAConfig,
    SFTShapeProfile,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
    TrainingStateResidency,
)
from cti.layers import Adapter
from cti.memory import native_backing
from cti.models import create_model, get_model_class
from cti.models.base import BaseLLMModel
from cti.models.training_contract import TrainingActivations
from cti.models.training_output import (
    TrainingLossChunks,
    attention_backward_tile_bytes,
    loss_working_bytes,
)
from cti.training._batching import plan_microbatches
from cti.training._memory_plan import (
    TrainingMemoryPlan,
    plan_model_training_memory,
    plan_training_memory,
    pool_slack_bytes,
    prepare_model_training,
)
from cti.training._parameters import TrainableParameterMetadata, describe_for_model
from cti.training._staging import SFTStaging, staging_device_bytes
from cti.training.runner import ModelParameterUniverse
from tests.support import reference_ops
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.gated_delta_net import reference_kernel_module
from tests.support.qwen3_5 import QWEN3_5_9B_CONFIG, load_hf_config, text_only_config
from tests.support.tp_shards import default_dtype, site
from tests.support.training import AdaptedTrainables
from tests.support.training_ops import bind_training_references

_MIB = 1024**2
# The loss operations' default logits chunks in cti-kernels.
_LOSS_CHUNKS = TrainingLossChunks(cross_entropy=256, logprobs=64)
_NONE = ActivationCheckpointMode.NONE
_FULL = ActivationCheckpointMode.FULL
_OFFLOAD = ActivationCheckpointMode.FULL_OFFLOAD

# Qwen/Qwen3-8B config.json at b968826d9c46dd6066d109eabc6255188de91218.
_QWEN3_8B_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}
_CONFIGS = {"Qwen3.5-9B": text_only_config(QWEN3_5_9B_CONFIG), "Qwen3-8B": _QWEN3_8B_CONFIG}

# (model, LoRA rank, mode, tokens): measured (master, moments, gradients, activation peak).
_MEASURED = {
    ("Qwen3.5-9B", 16, _NONE, 1024): (173113344, 346226688, 173113344, 5765211856),
    ("Qwen3.5-9B", 16, _NONE, 8192): (173113344, 346226688, 173113344, 44235201232),
    ("Qwen3.5-9B", 16, _OFFLOAD, 1024): (173113344, 346226688, 173113344, 391304484),
    ("Qwen3.5-9B", 16, _OFFLOAD, 8192): (173113344, 346226688, 173113344, 1958000804),
    ("Qwen3.5-9B", 4, _NONE, 1024): (43278336, 86556672, 43278336, 5757755344),
    ("Qwen3-8B", 16, _NONE, 1024): (174587904, 349175808, 174587904, 4335735024),
    ("Qwen3-8B", 16, _NONE, 8192): (174587904, 349175808, 174587904, 33654059248),
    ("Qwen3-8B", 16, _OFFLOAD, 1024): (174587904, 349175808, 174587904, 298344184),
    ("Qwen3-8B", 16, _OFFLOAD, 8192): (174587904, 349175808, 174587904, 1196010768),
}
# The plan holds every LoRA gradient live through Qwen3's attention segment, which sets the
# offloaded peak: a safe overcount.
_HEADROOM = {("Qwen3-8B", 16, _OFFLOAD, 1024): 1.10, ("Qwen3-8B", 16, _OFFLOAD, 8192): 1.10}


@pytest.fixture(autouse=True)
def _reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)
    monkeypatch.setitem(sys.modules, "ctikernels.gdn", reference_kernel_module())
    reference_ops.bind_norm_references(monkeypatch)
    bind_training_references(monkeypatch)


def _meta_model(name: str, rank: int, folder: Path) -> tuple[BaseLLMModel, AdaptedTrainables]:
    """``name``'s model on the meta device with an adapter of ``rank``, and the two as the
    host's trainables."""

    hf_config = load_hf_config(_CONFIGS[name], folder)
    model_class = get_model_class(hf_config.architectures[0])
    adapters = Adapter(LoRAConfig(targets=model_class.LORA_TARGETS, rank=rank, alpha=2.0 * rank))
    meta = torch.device("meta")
    with meta, default_dtype(torch.bfloat16):
        model = create_model(
            model_class, checkpoint_facts(hf_config), site=site(device=meta, training=True)
        )
    adapters.attach(model)
    adapters.materialize(device=meta, dtype=torch.bfloat16)
    return model, AdaptedTrainables(model, adapters)


def _padded_config(mode: ActivationCheckpointMode, tokens: int) -> TrainingConfig:
    return TrainingConfig(
        trainable_plan=TrainablePlan.all_declared(),
        optimizer=AdamWConfig(),
        batching=TrainingBatching(
            BatchLayout.PADDED, StaticBatching(microbatch_size=2, max_tokens_per_sample=tokens // 2)
        ),
        activation_checkpointing=mode,
    )


@pytest.mark.parametrize(("model", "rank", "mode", "tokens"), list(_MEASURED))
def test_plan_matches_measured_allocations(
    tmp_path: Path,
    model: str,
    rank: int,
    mode: ActivationCheckpointMode,
    tokens: int,
) -> None:
    built, trainables = _meta_model(model, rank, tmp_path)
    plan = plan_model_training_memory(
        _padded_config(mode, tokens),
        built,
        trainables=trainables,
        rank=rank,
        allocator=native_backing,
    )

    master, moments, gradients, activations = _MEASURED[model, rank, mode, tokens]
    for predicted, measured in (
        (plan.master_weights, master),
        (plan.optimizer_state, moments),
        (plan.gradients, gradients),
    ):
        # The plan models a fresh pool; reality differs by at most one small segment a
        # component partly fills (back-filling earlier allocations' room).
        assert (
            measured
            <= predicted
            <= max(1.015 * measured, measured + native_backing.SMALL_SEGMENT_BYTES)
        )
    headroom = _HEADROOM.get((model, rank, mode, tokens), 1.05)
    assert activations <= plan.activations <= headroom * activations
    assert plan.input_staging == 0
    # RESIDENT, the default: every state component stays beside inference.
    assert plan.resident_bytes == plan.master_weights + plan.optimizer_state + plan.gradients


def test_preparation_warms_the_training_ops_up_before_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, trainables = _meta_model("Qwen3-8B", 4, tmp_path)
    config = _padded_config(_NONE, 256)
    calls: list[tuple[str, int]] = []
    activations = model.training_activations
    layer = SimpleNamespace(warm_up_training=lambda tokens: calls.append(("warm", tokens)))
    monkeypatch.setattr(model, "iter_recurrent_layers", lambda: (layer, layer))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(("sync", 0)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    stream = object()
    monkeypatch.setattr(
        torch.cuda,
        "stream",
        lambda bound: calls.append(("stream", bound is stream)) or nullcontext(),
    )
    monkeypatch.setattr(torch.cuda, "current_blas_handle", lambda: calls.append(("blas", 0)))
    monkeypatch.setattr(
        model,
        "training_activations",
        lambda tokens, *rest, rank, row_tile: (
            calls.append(("plan", tokens))
            or activations(tokens, *rest, rank=rank, row_tile=row_tile)
        ),
    )

    plan = prepare_model_training(
        config, model, trainables=trainables, rank=4, stream=stream, allocator=native_backing
    )

    tokens = config.batching.policy.max_tokens_per_microbatch
    # Each layer holding a training op compiles at the planned microbatch first, and the
    # training stream binds its cuBLAS workspace, so the plan and the KV budget see both.
    assert calls == [
        ("warm", tokens),
        ("warm", tokens),
        ("sync", 0),
        ("stream", True),
        ("blas", 0),
        ("plan", tokens),
    ]
    assert plan == plan_model_training_memory(
        config, model, trainables=trainables, rank=4, allocator=native_backing
    )


def test_state_components_follow_the_selected_parameters() -> None:
    parameters = (
        TrainableParameterMetadata("a", (4, 64), torch.bfloat16),
        TrainableParameterMetadata("b", (64, 4), torch.float32),
    )
    activations = TrainingActivations(layers=(((10, 3),),), head_saved=1, head_working=2)

    plan = plan_training_memory(
        parameters, activations, _NONE, first=0, boundary_bytes=0, allocator=native_backing
    )

    # One small segment each: both parameters' FP32 masters (the FP32 one's too: training
    # never writes what inference reads), both moments of both, both FP32 gradients.
    # Backward also holds the masters' 1,024 + 1,024 bytes of new FP32 gradients.
    assert plan == TrainingMemoryPlan(
        master_weights=2 << 20,
        optimizer_state=2 << 20,
        gradients=2 << 20,
        input_staging=0,
        # Both FP32 parameters step in one group, after the commit's gradient-norm scalar.
        optimizer_step=2 << 20,
        activations=10 + 1 + 3 + 1024 + 1024,
        # A small and a large segment beside the 14 bytes of activations, a quarter of them for
        # fragmentation, and the gradients' own small segment.
        activation_segments=(22 << 20) + 4 + (2 << 20),
    )
    # One scope in a fresh pool maps the activations' segments.
    assert plan.scope_bytes == 2062 + (24 << 20) + 4


def test_gradients_map_small_segments_beside_the_activations_slack() -> None:
    parameters = tuple(
        TrainableParameterMetadata(f"p{index}", (16, 4096), torch.bfloat16) for index in range(3)
    )
    activations = TrainingActivations(layers=(((40 << 20, 0),),), head_saved=0, head_working=0)

    plan = plan_training_memory(
        parameters, activations, _NONE, first=0, boundary_bytes=0, allocator=native_backing
    )

    # Three 256 KiB FP32 gradients: their payload in the activations, their own 2 MiB small
    # segment beside the rest's slack, which takes no fragmentation share for them.
    gradients = 3 * (256 << 10)
    assert plan.activations == (40 << 20) + gradients
    assert plan.activation_segments == (
        (2 << 20) + (20 << 20) + math.ceil(0.25 * (40 << 20)) + (2 << 20)
    )


def test_each_residency_keeps_its_share_beside_inference() -> None:
    parameters = (TrainableParameterMetadata("a", (1024, 3072), torch.bfloat16),)
    activations = TrainingActivations(layers=(((10, 3),),), head_saved=1, head_working=2)
    plans = {
        residency: plan_training_memory(
            parameters,
            activations,
            _NONE,
            first=0,
            boundary_bytes=0,
            staging_bytes=(4096,),
            allocator=native_backing,
            residency=residency,
        )
        for residency in TrainingStateResidency
    }
    resident, offload_optimizer, offload_all = (
        plans[residency] for residency in TrainingStateResidency
    )
    elements = 1024 * 3072

    # The moments (8 bytes per element) leave with serving's turn, then the FP32 master and
    # gradient (8 more); the staging stays, and the peak is the same.
    assert resident.resident_bytes - offload_optimizer.resident_bytes == 8 * elements
    assert offload_optimizer.resident_bytes - offload_all.resident_bytes == 8 * elements
    assert offload_all.resident_bytes == offload_all.input_staging
    assert len({plan.training_bytes for plan in plans.values()}) == 1


def test_an_oversized_parameter_steps_in_a_group_of_its_own() -> None:
    parameters = (
        TrainableParameterMetadata("small", (16, 4096), torch.bfloat16),
        TrainableParameterMetadata("large", (512, 12288), torch.bfloat16),
    )
    activations = TrainingActivations(layers=(((0, 0),),), head_saved=0, head_working=0)

    plan = plan_training_memory(
        parameters, activations, _NONE, first=0, boundary_bytes=0, allocator=native_backing
    )

    # 24 MiB of FP32 square roots for the large parameter alone, and a small segment for the
    # gradient-norm scalar.
    assert plan.optimizer_step == 26 << 20


@pytest.mark.parametrize(
    ("mode", "first", "expected"),
    [
        # All saves and the head, then the larger of the loss and the busiest backward:
        # layer 1's MLP segment, once its mixer's backward waits for it. Backward holds the
        # master's 200 bytes of FP32 gradients.
        (_NONE, 0, 10 + 20 + 1 + max(50, 30 + 200)),
        (_NONE, 1, 20 + 1 + max(50, 30 + 200)),
        # Every layer input, then the larger of the head and the widest layer's backward.
        (_FULL, 0, 2 * 7 + max(51, 20 + 30 + 200)),
        # Four inputs around the forward's head, or three around the widest segment's
        # backward, which no other segment's saves join.
        (_OFFLOAD, 0, max(4 * 7 + 51, 3 * 7 + 12 + 30 + 200)),
    ],
)
def test_activation_peak_follows_the_checkpointing_mode(
    mode: ActivationCheckpointMode, first: int, expected: int
) -> None:
    activations = TrainingActivations(
        layers=(((10, 5),), ((8, 20), (12, 30))), head_saved=1, head_working=50
    )
    parameters = (TrainableParameterMetadata("a", (50,), torch.bfloat16),)

    plan = plan_training_memory(
        parameters, activations, mode, first=first, boundary_bytes=7, allocator=native_backing
    )

    assert plan.activations == expected


def test_a_frozen_layer_forward_can_set_the_peak() -> None:
    # Layer 0 runs without autograd, beside its input and output.
    activations = TrainingActivations(
        layers=(((300, 400),), ((20, 6),)), head_saved=1, head_working=50
    )
    parameters = (TrainableParameterMetadata("a", (50,), torch.bfloat16),)

    plan = plan_training_memory(
        parameters, activations, _OFFLOAD, first=1, boundary_bytes=7, allocator=native_backing
    )

    assert plan.activations == 2 * 7 + 300 + 400


def test_selected_parameters_are_described_in_optimizer_order(tmp_path: Path) -> None:
    _, trainables = _meta_model("Qwen3-8B", 4, tmp_path)
    declared = [
        name for name, _ in ModelParameterUniverse.of(trainables).named_trainable_parameters
    ]
    plan = TrainablePlan((declared[3], declared[1]))

    described = describe_for_model(plan, trainables)

    assert [parameter.name for parameter in described] == [declared[1], declared[3]]
    with pytest.raises(ValueError, match="model-declared"):
        describe_for_model(TrainablePlan(("model.embed_tokens.weight",)), trainables)


def test_dynamic_batching_bounds_the_sequences_of_a_microbatch() -> None:
    policy = DynamicBatching(max_tokens_per_microbatch=100, max_sequences_per_microbatch=2)

    assert plan_microbatches(policy, (10, 10, 10, 90, 5)) == ((0, 1), (2, 3), (4,))


@pytest.mark.parametrize("name", ["Qwen3-8B", "Qwen3.5-9B"])
def test_attention_backward_holds_a_flash_tile_per_sequence(tmp_path: Path, name: str) -> None:
    config = _CONFIGS[name].get("text_config", _CONFIGS[name])
    heads, head_dim = config["num_attention_heads"], config["head_dim"]
    # 128 FP32 rows per sequence of the query-gradient accumulator, at the head dimension
    # FlashAttention rounds to, and of the softmax row sums.
    rounded = -(-head_dim // 32) * 32 if head_dim <= 192 else 256
    tile = attention_backward_tile_bytes(128, heads, head_dim)
    assert tile == 4 * 128 * 128 * heads * (rounded + 1)

    # Packed short sequences: 128 more sequences add their tiles to every attention
    # layer's mixer backward.
    model, _ = _meta_model(name, 16, tmp_path)
    fewer, more = (model.training_activations(768, sequences, rank=16) for sequences in (128, 256))
    layer_types = config.get("layer_types", ["full_attention"] * config["num_hidden_layers"])
    for layer, kind in enumerate(layer_types):
        if kind == "full_attention":
            (_, mixer_more), _ = more.layers[layer]
            (_, mixer_fewer), _ = fewer.layers[layer]
            assert mixer_more - mixer_fewer == tile, layer


def test_short_microbatches_size_the_head_for_grpo_log_probabilities() -> None:
    # Four FP32 logits chunks of the 32 rows against the fused cross entropy's BF16 chunk.
    assert loss_working_bytes(32, 4, 1000, _LOSS_CHUNKS) == 32 * 2 * 4 + 16 * 32 * 1000
    assert loss_working_bytes(1024, 4, 1000, _LOSS_CHUNKS) == (1024 * (2 * 4 + 4) + 2 * 1000 * 512)


def test_the_loss_head_plans_with_its_services_operations_chunks() -> None:
    import inspect

    from cti.layers import ParallelLMHead
    from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss
    from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs

    def default_chunk(operation: type) -> int:
        return inspect.signature(operation).parameters["chunk_size"].default

    # A training head takes its services' operations, whose chunks the references share.
    head = ParallelLMHead(64, 4, site=site(training=True))
    assert head.training_chunk_rows == (
        default_chunk(FusedLinearCrossEntropyLoss),
        default_chunk(ChunkedLinearSelectedTokenLogprobs),
    )
    assert _LOSS_CHUNKS == TrainingLossChunks(*head.training_chunk_rows)
    halved = TrainingLossChunks(cross_entropy=128, logprobs=32)
    assert loss_working_bytes(1024, 4, 1000, halved) == 1024 * (2 * 4 + 4) + 2 * 1000 * 256


@pytest.mark.parametrize("packed", [False, True], ids=("padded", "packed"))
@pytest.mark.parametrize(
    "profile",
    [
        SFTShapeProfile((2, 4)),
        SFTShapeProfile((3, 3)),
        SFTShapeProfile((2, 4), num_target_tokens=2),
    ],
    ids=("dense", "full", "sparse"),
)
def test_staging_bytes_are_the_buffers_the_runner_stages_in(
    profile: SFTShapeProfile, packed: bool
) -> None:
    layout = BatchLayout.PACKED if packed else BatchLayout.PADDED
    staging = SFTStaging(profile, layout, device=torch.device("meta"))
    packing = staging._packing
    values = (
        staging._fixed,
        *staging._buffers,
        *(() if packing is None else (packing.positions, packing.cu_seqlens)),
    )
    buffers = {id(value): value.nbytes for value in values if value is not None}

    assert sorted(staging_device_bytes(profile, packed=packed)) == sorted(buffers.values())


@pytest.mark.parametrize(
    ("payloads", "expected"),
    [
        # 32 blocks of 64 KiB fill a 2 MiB small segment; the 33rd opens another.
        ((64 * 1024,) * 33, 4 * _MIB),
        # Six 3 MiB blocks fill a 20 MiB segment; the 2 MiB rest cannot hold a seventh.
        ((3 * _MIB,) * 7, 40 * _MIB),
        # The 3 MiB block takes the 4 MiB rest; its 1 MiB rest stays unsplit.
        ((8 * _MIB, 8 * _MIB, 3 * _MIB, _MIB + 512), 40 * _MIB),
        # From 10 MiB a block gets its own segment rounded to 2 MiB.
        ((0, 16 * _MIB, 11 * _MIB), 28 * _MIB),
    ],
)
def test_pooled_allocation_packs_blocks_into_segments(payloads, expected):
    assert native_backing.pooled(payloads) == expected


def test_the_step_holds_the_gradient_norm_beside_its_group() -> None:
    # Four FP32 temporaries of 512 KiB fill a 2 MiB small segment; the commit's normalized
    # gradient norm, live through the step, opens a second.
    parameters = tuple(
        TrainableParameterMetadata(name, (128, 1024), torch.float32) for name in "abcd"
    )
    activations = TrainingActivations(layers=(((0, 0),),), head_saved=0, head_working=0)

    plan = plan_training_memory(
        parameters, activations, _NONE, first=0, boundary_bytes=0, allocator=native_backing
    )

    assert plan.optimizer_step == 4 << 20


@pytest.mark.parametrize(("largest_block", "separate"), [(64 << 20, False), (16 << 20, True)])
def test_logits_chunks_wider_than_every_decoder_block_map_their_own_segments(
    largest_block: int, separate: bool
) -> None:
    parameters = (TrainableParameterMetadata("a", (4, 64), torch.float32),)
    activations = TrainingActivations(
        layers=(((30 << 20, 30 << 20),),),
        head_saved=0,
        head_working=40 << 20,
        head_blocks=(20 << 20, 20 << 20),
        largest_block=largest_block,
    )

    plan = plan_training_memory(
        parameters, activations, _NONE, first=0, boundary_bytes=0, allocator=native_backing
    )

    shared = pool_slack_bytes(plan.activations - 1024, native_backing)
    # Beside either peak, the FP32 gradient's 1,024 bytes take a small segment of their own.
    gradients = 2 << 20
    if not separate:
        assert plan.activation_segments == shared + gradients
    else:
        # The decoder phase's peak and a partly used small and large segment, then the two
        # chunks' own 20 MiB segments.
        decoder = (60 << 20) + 1024
        separated = decoder + (22 << 20) + (40 << 20)
        assert plan.activation_segments == max(shared, separated - plan.activations) + gradients


@pytest.mark.parametrize(
    ("live", "expected"),
    [(0, 0), (1, (22 << 20) + 1), (4 << 30, (22 << 20) + (1 << 30))],
)
def test_a_fresh_pool_maps_two_segment_classes_and_a_quarter_beyond_live_blocks(live, expected):
    assert pool_slack_bytes(live, native_backing) == expected


@pytest.mark.parametrize(
    "size",
    [
        *(512, _MIB - 512, _MIB, _MIB + 1, _MIB + 512, 10 * _MIB - 512),
        *(10 * _MIB - 511, 10 * _MIB - 1, 10 * _MIB, 10 * _MIB + 1, 1 << 30),
    ],
)
def test_the_plan_and_the_kv_budget_round_segments_alike(size: int) -> None:
    # The training plan's pooled placement and the KV budget's segment rule agree at every
    # class boundary: both round the request up to a block first, as the allocator does.
    assert native_backing.pooled([size]) == native_backing.segment(size)
