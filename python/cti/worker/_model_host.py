# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/engine/engine.py`, `python/minisgl/utils/torch_utils.py`);
# see THIRD_PARTY_NOTICES.

"""The rank's models, built, placed, loaded and signed before the inference state is sized."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

import torch

from cti.layers import Adapter, LayerServices, Site, prepare_inference
from cti.models import create_model, load_peft, model_signature, open_weights, save_peft
from cti.utils import init_logger

if TYPE_CHECKING:
    from cti.distributed import CollectiveGroup
    from cti.models import BaseLLMModel, CheckpointFacts, ModelConfig
    from cti.plan import Plan

logger = init_logger(__name__)


class HostedCheckpoint(Protocol):
    """The checkpoint the composition root resolved: its model class, the facts read from
    it, and the engine configuration derived from them."""

    model_class: type[BaseLLMModel]
    facts: CheckpointFacts
    model_config: ModelConfig


class ModelHost:
    """The target at its ``Site`` with its LoRA adapter, and the speculative draft at the
    same site with services of its own, both with their weights. Built before capacity
    planning, so the budget counts them; training trains the same target, selecting from
    the parameter universe this host owns."""

    def __init__(
        self,
        plan: Plan,
        checkpoint: HostedCheckpoint,
        *,
        device: torch.device,
        ranks: CollectiveGroup,
        training: bool,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """``training`` places the target's layers for training too; ``progress`` hears each
        long phase as it begins."""

        progress = progress or _quiet
        self.checkpoint = checkpoint
        self.device = device
        self.adapter = Adapter(plan.lora) if plan.lora is not None else None
        # The target's services; the attention backend and the pool join them once built.
        self.services = LayerServices()
        site = Site(
            ranks,
            plan.experts,
            device,
            training=training,
            services=self.services,
        )
        method = plan.speculative
        self.draft: BaseLLMModel | None = None
        # Downloaded now, after resolution validated the checkpoint.
        weights = open_weights(checkpoint.facts.source, plan.weights, progress)
        # The checkpoint's tensors, which name the weights a saved adapter adapts.
        self._checkpoint_names = weights.names()
        # Read before the model is built, so a checkpoint the engine cannot serve fails first.
        weight_methods = weights.methods(device)
        if training and (recipe := weight_methods.untrainable) is not None:
            raise ValueError(
                f"{recipe} checkpoints serve and serve LoRA adapters; they do not train"
            )
        # The checkpoint's methods declare their tensors as the model's, before any loads.
        with torch.device("meta"), _default_dtype(plan.dtype):
            self.model = create_model(checkpoint.model_class, checkpoint.facts, site=site)
            methods = weight_methods.adopt(self.model)
            if method is not None:
                # The draft serves inference alone, its kernels through services of its own.
                self.draft = method.model_class(
                    method.config,
                    round_width=method.round_width,
                    site=replace(site, training=False, services=LayerServices()),
                )
        if methods:
            logger.info_rank0("Weight methods: %s", methods)
        if self.adapter is not None:
            # Once, before the weights load.
            self.adapter.attach(self.model)
        self.plan = plan
        progress("loading weights")
        self.model.load_state_dict(
            weights.state_dict(self.model, device, show_progress=ranks.rank == 0, progress=progress)
        )
        if self.draft is not None:

            def loading(phase: str) -> None:
                progress(f"draft {phase}")

            self.draft.load_state_dict(
                open_weights(method.source, plan.weights, loading).state_dict(
                    self.draft, device, show_progress=False, progress=loading
                )
            )
        if self.adapter is not None:
            self.adapter.materialize(device=device, dtype=plan.dtype)
            if plan.lora.adapter is not None:
                load_peft(self.model, self.adapter, plan.lora.adapter)
        self._signature: str | None = None

    @property
    def models(self) -> tuple[BaseLLMModel, ...]:
        """The target, then the draft."""

        return (self.model,) if self.draft is None else (self.model, self.draft)

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        """The target's tensors."""

        return self.model.state_dict()

    def named_trainable_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        """The target's trainable declaration: the adapter's factors."""

        adapter = self.adapter
        return () if adapter is None else adapter.named_parameters()

    def decoder_layer_trainable_parameters(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        """The adapter's factors under each of the target's decoder layers, in order."""

        adapter, model = self.adapter, self.model
        if adapter is None:
            return tuple(() for _ in model.decoder_layer_paths())
        ops = dict(model.named_ops())
        return tuple(adapter.parameters_in(ops[path]) for path in model.decoder_layer_paths())

    def check_training_selection(self, names: tuple[str, ...]) -> None:
        if self.adapter is not None:
            self.adapter.check_training_selection(names)

    def bind_training_weights(self, masters: Mapping[str, torch.Tensor]) -> None:
        """Train the adapter on ``masters`` (``Adapter.bind_training``)."""

        if self.adapter is None:
            raise ValueError("the engine has no adapter to train")
        self.adapter.bind_training(masters)

    def release_training_weights(self) -> None:
        if self.adapter is not None:
            self.adapter.release_training()

    def save_adapter(self, path: str, state: Mapping[str, torch.Tensor]) -> None:
        """Write the adapter's whole factors ``state`` (by name) as a PEFT adapter in the
        folder ``path`` (``models.save_peft``); a ``ValueError`` is an adapter with no PEFT
        form."""

        if self.adapter is None:
            raise ValueError("the engine has no adapter to save")
        if self._checkpoint_names is None:
            raise ValueError("random weights name no checkpoint weights for a PEFT adapter")
        save_peft(path, self.model, self.adapter, state, self._checkpoint_names)

    def prepare_inference(
        self,
        max_rows: int,
        *,
        max_sequences: int,
        progress: Callable[[str], None],
    ) -> int:
        return prepare_inference(
            self.models,
            max_rows=max_rows,
            max_sequences=max_sequences,
            dtype=self.plan.dtype,
            device=self.device,
            progress=progress,
        )

    def signature(self) -> str:
        """The immutable construction identity delta checkpoints carry."""

        if self._signature is None:
            plan, checkpoint = self.plan, self.checkpoint
            self._signature = model_signature(
                source=checkpoint.facts.source,
                config=checkpoint.model_class.CONFIG.from_hf(checkpoint.facts.hf_config),
                dtype=plan.dtype,
                trainable=None if self.adapter is None else self.adapter.identity(),
                named_tensors=tuple(
                    {**self.state_dict(), **dict(self.named_trainable_parameters())}.items()
                ),
            )
        return self._signature


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def _quiet(phase: str) -> None:
    pass


__all__ = ["ModelHost"]
