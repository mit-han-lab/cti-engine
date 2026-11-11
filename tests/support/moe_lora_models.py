"""Tiny Qwen3-MoE and Qwen3.5-MoE checkpoints with a non-zero adapter of every LoRA target, the
engine model and training runner over them, and their Transformers FP32 reference: PEFT's
adapters on the linears, ``LoRAExperts`` on the routed experts, and the runner's AdamW."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from cti.config import (
    ActivationCheckpointMode,
    AdamWConfig,
    BatchLayout,
    LoRAConfig,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.layers import Adapter
from cti.models.base import BaseLLMModel
from cti.models.qwen3_5_moe.config import Qwen3_5MoeConfig
from cti.models.qwen3_5_moe.model import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForConditionalGeneration,
)
from cti.models.qwen3_moe.model import Qwen3MoeDecoderLayer, Qwen3MoeForCausalLM
from cti.models.weight import load_weight
from cti.training import SFTDatum, TrainingBatch
from cti.training.runner import TrainingRunner
from tests.support.expert_lora import LoRAExperts, random_expert_factors
from tests.support.moe import ReferenceMoe
from tests.support.qwen3_5 import load_hf_config, randomize_hf_weights
from tests.support.qwen3_5_lora import (
    TEXT_ONLY_PREFIX,
    Qwen3_5LoRAMapping,
    attach_peft_lora,
    random_peft_state,
)
from tests.support.qwen3_5_moe import TINY_QWEN3_5_MOE_CONFIG, text_only_config
from tests.support.qwen3_moe import TINY_QWEN3_MOE_CONFIG, save_tiny_checkpoint
from tests.support.tp_shards import CPU, build_model
from tests.support.training import AdaptedTrainables, host_runner, training_logits

# The smallest rank per-expert LoRA takes.
RANK, ALPHA = 8, 2.0
PAD_TOKEN_ID = 0
# The first sequence spans two of Transformers' 64-token delta-rule chunks.
SFT_LENGTHS = (70, 5, 9)
OPTIMIZER = AdamWConfig(lr=1e-2, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.01, max_grad_norm=1.0)
_ROUTED = ".mlp.experts."


@dataclass(frozen=True)
class Family:
    """A family's tiny text checkpoint: ``text_tokens`` bounds its text token ids,
    ``last_layer`` is its last decoder layer, and ``peft_targets`` adapt its PEFT linears.
    Subclasses save the checkpoint and name PEFT's tensors as the engine's (``to_engine``)."""

    model_class: type[BaseLLMModel]
    layer_class: type[torch.nn.Module]
    text_tokens: int
    last_layer: int
    changes: Mapping[str, Any] = field(default_factory=dict)
    peft_targets: ClassVar[tuple[str, ...]]


# Each attention module PEFT adapts, by the Qwen3-MoE engine projection slice that holds it.
_QWEN3_SLICES = MappingProxyType(
    {
        "q_proj": "qkv_proj.q",
        "k_proj": "qkv_proj.k",
        "v_proj": "qkv_proj.v",
        "o_proj": "o_proj.full",
    }
)


class Qwen3MoeFamily(Family):
    peft_targets = ("attention",)

    def save(self, folder: Path) -> PreTrainedModel:
        return save_tiny_checkpoint(folder, **self.changes)

    def to_engine(self, folder: Path, peft: Mapping[str, torch.Tensor]) -> dict:
        engine = {}
        for key, tensor in peft.items():
            stem = key.removeprefix("base_model.model.").removesuffix(".weight")
            parent, module, factor = stem.rsplit(".", 2)
            engine[f"{parent}.{_QWEN3_SLICES[module]}.{factor}"] = tensor
        return engine


class Qwen3_5MoeFamily(Family):
    # The shared expert's linears with the "experts" target.
    peft_targets = Qwen3_5MoeForConditionalGeneration.LORA_TARGETS

    def save(self, folder: Path) -> PreTrainedModel:
        config = load_hf_config(text_only_config(TINY_QWEN3_5_MOE_CONFIG), folder)
        reference = AutoModelForCausalLM.from_config(config)
        randomize_hf_weights(reference, conv_dtype=torch.bfloat16)
        reference.save_pretrained(folder)
        return reference

    def to_engine(self, folder: Path, peft: Mapping[str, torch.Tensor]) -> dict:
        text = Qwen3_5MoeConfig.from_hf(AutoConfig.from_pretrained(folder)).text
        return Qwen3_5LoRAMapping(text, prefix=TEXT_ONLY_PREFIX).to_engine(peft)


# Qwen3-MoE in its published form (renormalized routes, unbiased attention) and with raw
# routing probabilities and attention biases; Qwen3.5-MoE with both mixers and a shared expert.
FAMILIES = MappingProxyType(
    {
        "qwen3_moe": Qwen3MoeFamily(
            Qwen3MoeForCausalLM, Qwen3MoeDecoderLayer, TINY_QWEN3_MOE_CONFIG["vocab_size"], 2
        ),
        "qwen3_moe-raw-biased": Qwen3MoeFamily(
            Qwen3MoeForCausalLM,
            Qwen3MoeDecoderLayer,
            TINY_QWEN3_MOE_CONFIG["vocab_size"],
            2,
            {"norm_topk_prob": False, "attention_bias": True},
        ),
        # Layers 1 and 2 are Gated DeltaNet layers.
        "qwen3_5_moe": Qwen3_5MoeFamily(
            Qwen3_5MoeForConditionalGeneration,
            Qwen3_5MoeDecoderLayer,
            TINY_QWEN3_5_MOE_CONFIG["image_token_id"],
            3,
        ),
    }
)


def build_engine(
    family: Family,
    hf_config: PretrainedConfig,
    state: Callable[[BaseLLMModel], Mapping[str, torch.Tensor]],
    layout: BatchLayout,
    *,
    adapter: Mapping[str, torch.Tensor] | None = None,
    activation_checkpointing: ActivationCheckpointMode = ActivationCheckpointMode.NONE,
    trains: Callable[[str], bool] | None = None,
    microbatch_size: int = 4,
) -> tuple[BaseLLMModel, Adapter, TrainingRunner]:
    """The engine model as the runner builds it for training, its base weights ``state``,
    with ``adapter`` by engine name or new (zero) factors; ``trains`` selects the adapters
    that train by name, every one by default."""

    adapters = Adapter(LoRAConfig(targets=family.model_class.LORA_TARGETS, rank=RANK, alpha=ALPHA))
    model = build_model(
        hf_config, lora_adapters=adapters, training=True, moe_backend=ReferenceMoe()
    )
    model.load_state_dict(dict(state(model)))
    adapters.materialize(device=CPU, dtype=torch.float32)
    if adapter is not None:
        adapters.load_state_dict(adapter)
    plan = (
        TrainablePlan.all_declared()
        if trains is None
        else TrainablePlan(tuple(name for name, _ in adapters.named_parameters() if trains(name)))
    )
    runner = host_runner(
        model,
        TrainingConfig(
            plan,
            OPTIMIZER,
            TrainingBatching(
                layout, StaticBatching(microbatch_size=microbatch_size, max_tokens_per_sample=128)
            ),
            activation_checkpointing=activation_checkpointing,
        ),
        trainables=AdaptedTrainables(model, adapters),
        checks=True,
    )
    return model, adapters, runner


@dataclass(frozen=True)
class Checkpoint:
    """A family's tiny text checkpoint with a non-zero adapter of every target: PEFT's for the
    linears, stacked factors of each layer's routed experts by ``tests.support.expert_lora``
    name, as the engine names them."""

    family: Family
    folder: Path
    reference_adapter: Mapping[str, torch.Tensor]

    @classmethod
    def save(cls, family: Family, folder: Path) -> Checkpoint:
        reference = family.save(folder)
        generator = torch.Generator().manual_seed(3)
        adapter = random_peft_state(
            reference.state_dict(),
            prefix=TEXT_ONLY_PREFIX,
            targets=family.peft_targets,
            rank=RANK,
            generator=generator,
        )
        text = reference.config.get_text_config()
        for layer in range(text.num_hidden_layers):
            factors = random_expert_factors(
                text.num_experts,
                text.hidden_size,
                text.moe_intermediate_size,
                RANK,
                generator=generator,
            )
            adapter |= {f"model.layers.{layer}{_ROUTED}{n}": f for n, f in factors.items()}
        return cls(family, folder, adapter)

    def to_engine(self, reference: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Reference tensors by the engine's names: PEFT's renamed, the routed factors' own."""

        routed = {n: t for n, t in reference.items() if _ROUTED in n}
        linears = {n: t for n, t in reference.items() if _ROUTED not in n}
        return self.family.to_engine(self.folder, linears) | routed

    def engine(
        self, layout: BatchLayout, *, adapter: bool = True, **options: Any
    ) -> tuple[BaseLLMModel, Adapter, TrainingRunner]:
        """The engine over this checkpoint, with its adapter or new (zero) factors."""

        def state(model: BaseLLMModel) -> dict[str, torch.Tensor]:
            mapping, params = model.checkpoint_mapping, model.params()
            return dict(load_weight(str(self.folder), CPU, mapping=mapping, params=params))

        hf_config = AutoConfig.from_pretrained(self.folder)
        engine_adapter = self.to_engine(self.reference_adapter) if adapter else None
        return build_engine(
            self.family, hf_config, state, layout, adapter=engine_adapter, **options
        )

    def transformers(self) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(
            self.folder,
            dtype=torch.float32,
            attn_implementation="sdpa",
            experts_implementation="eager",
        )


class Reference:
    """Transformers' per-expert loop with the checkpoint's adapter, PEFT's way, and the
    runner's AdamW."""

    def __init__(self, checkpoint: Checkpoint) -> None:
        model = checkpoint.transformers()
        scaling = ALPHA / RANK
        adapter = checkpoint.reference_adapter
        peft = {n: t for n, t in adapter.items() if _ROUTED not in n}
        self.parameters = attach_peft_lora(model, peft, scaling=scaling)
        for index, layer in enumerate(model.model.layers):
            prefix = f"model.layers.{index}{_ROUTED}"
            factors = {
                n.removeprefix(prefix): t for n, t in adapter.items() if n.startswith(prefix)
            }
            experts = layer.mlp.experts
            layer.mlp.experts = LoRAExperts(
                experts.gate_up_proj, experts.down_proj, factors, scaling=scaling
            )
            named = layer.mlp.experts.named_factors.items()
            self.parameters |= {f"{prefix}{name}": factor for name, factor in named}
        self.model: PreTrainedModel = model.train()
        self._optimizer = torch.optim.AdamW(
            self.parameters.values(),
            lr=OPTIMIZER.lr,
            betas=OPTIMIZER.betas,
            eps=OPTIMIZER.eps,
            weight_decay=OPTIMIZER.weight_decay,
            foreach=False,
        )

    def loss(self, data: tuple[SFTDatum, ...]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """The summed SFT loss of ``data``, backpropagated, and each sequence's logits."""

        logits = [self.model(input_ids=torch.tensor([d.token_ids[:-1]])).logits[0] for d in data]
        losses = []
        for sequence_logits, datum in zip(logits, data, strict=True):
            targets = torch.tensor(datum.token_ids[1:])
            mask = torch.tensor(datum.target_mask[1:])
            losses.append(F.cross_entropy(sequence_logits[mask], targets[mask], reduction="sum"))
        loss = torch.stack(losses).sum()
        loss.backward()
        return loss.detach(), tuple(sequence.detach() for sequence in logits)

    def step(self, normalization_count: int) -> None:
        """The runner's commit: divide by the count, clip the global norm, then AdamW."""

        parameters = tuple(self.parameters.values())
        with torch.no_grad():
            gradients = [parameter.grad.div_(normalization_count) for parameter in parameters]
            norm = torch.nn.utils.get_total_norm(gradients)
            torch.nn.utils.clip_grads_with_norm_(parameters, OPTIMIZER.max_grad_norm, norm)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)


def sft_data(family: Family, step: int) -> tuple[SFTDatum, ...]:
    """Random text whose first two tokens are an unsupervised prompt."""

    generator = torch.Generator().manual_seed(100 + step)
    data = []
    for length in SFT_LENGTHS:
        tokens = torch.randint(1, family.text_tokens, (length,), generator=generator)
        mask = (False, False, *([True] * (length - 2)))
        data.append(SFTDatum(token_ids=tuple(tokens.tolist()), target_mask=mask))
    return tuple(data)


def sft_batch(family: Family, layout: BatchLayout, step: int) -> TrainingBatch:
    padding = {"pad_token_id": PAD_TOKEN_ID} if layout is BatchLayout.PADDED else {}
    return TrainingBatch.sft(sft_data(family, step), layout, **padding)


def all_rows(batch: TrainingBatch):
    """Every row of ``batch`` as one host microbatch."""

    return batch.microbatch(tuple(range(batch.batch_size)), CPU)


def sequence_logits(model: BaseLLMModel, batch: TrainingBatch) -> tuple[torch.Tensor, ...]:
    """Each sequence's logits of the engine's training forward."""

    rows = all_rows(batch)
    logits = training_logits(model, rows.input_ids, rows.packing)
    if rows.packing is None:
        return tuple(row[: length - 1] for row, length in zip(logits, SFT_LENGTHS, strict=True))
    return logits.split([length - 1 for length in SFT_LENGTHS])


__all__ = [
    "FAMILIES",
    "OPTIMIZER",
    "PAD_TOKEN_ID",
    "Checkpoint",
    "Family",
    "Reference",
    "all_rows",
    "build_engine",
    "sequence_logits",
    "sft_batch",
    "sft_data",
]
