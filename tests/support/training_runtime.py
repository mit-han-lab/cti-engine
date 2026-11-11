"""Training-only runtime ownership for deterministic GPU correctness oracles."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig

from cti.config import BatchLayout
from cti.distributed import Topology
from cti.distributed.runtime import ParallelGroup
from cti.layers import Adapter, ExpertParallel, LayerServices, Site
from cti.models import create_model, get_model_class
from cti.models.weight import load_weight
from cti.training.runner import TrainingRunner
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.tp_shards import default_dtype
from tests.support.training import AdaptedTrainables, host_runner

if TYPE_CHECKING:
    from cti.config import LoRAConfig, TrainingConfig
    from cti.models.base import BaseLLMModel


class PaddedTrainingOracleRuntime:
    """Own the production model/trainer stack without inference-only resources.

    Deterministic FP32 tests use this boundary because FlashInfer paged attention
    intentionally supports only FP16/BF16. BF16 integration tests continue to use
    ``Rank`` and therefore cover the complete inference/training runtime. Gated
    DeltaNet layers get their training operation here, as ``InferenceRunner`` binds it, and a
    MoE model the engine's MoE backend. ``model_path`` is a local checkpoint directory;
    ``initialize_adapters`` sets the adapters before the optimizer copies them into its masters.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        initialization_seed: int = 42,
        initialize_adapters: Callable[[Adapter], None] | None = None,
    ) -> None:
        if training_config.batching.layout is not BatchLayout.PADDED:
            raise ValueError("PaddedTrainingOracleRuntime requires padded batching")

        self._parallel: ParallelGroup | None = None
        self._model: BaseLLMModel | None = None
        self._adapters: Adapter | None = None
        self._runner: TrainingRunner | None = None
        try:
            self._initialize(
                model_path,
                dtype=dtype,
                lora_config=lora_config,
                training_config=training_config,
                initialization_seed=initialization_seed,
                initialize_adapters=initialize_adapters,
            )
        except BaseException:
            self.close()
            raise

    def _initialize(
        self,
        model_path: str,
        *,
        dtype: torch.dtype,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        initialization_seed: int,
        initialize_adapters: Callable[[Adapter], None] | None,
    ) -> None:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        hf_config = AutoConfig.from_pretrained(model_path)
        parallel = ParallelGroup(
            Topology(0, 1),
            device=device,
            timeout_seconds=60.0,
            init_method=None,
            use_pynccl=False,
        )
        self._parallel = parallel

        torch.manual_seed(initialization_seed)
        adapters = Adapter(lora_config)
        with torch.device("meta"), default_dtype(dtype):
            model = create_model(
                get_model_class(hf_config.architectures[0]),
                checkpoint_facts(hf_config),
                site=Site(
                    parallel,
                    ExpertParallel(),
                    device,
                    training=True,
                    services=LayerServices(),
                ),
            )
        adapters.attach(model)
        model.load_state_dict(
            dict(
                load_weight(
                    model_path,
                    device,
                    mapping=model.checkpoint_mapping,
                    params=model.params(),
                )
            )
        )
        adapters.materialize(device=device, dtype=dtype)
        if initialize_adapters is not None:
            initialize_adapters(adapters)
        self._model = model
        self._adapters = adapters
        self._runner = host_runner(
            model, training_config, trainables=AdaptedTrainables(model, adapters)
        )

    @property
    def model(self) -> BaseLLMModel:
        if self._model is None:
            raise RuntimeError("training oracle runtime is closed")
        return self._model

    @property
    def adapters(self) -> Adapter:
        if self._adapters is None:
            raise RuntimeError("training oracle runtime is closed")
        return self._adapters

    @property
    def runner(self) -> TrainingRunner:
        if self._runner is None:
            raise RuntimeError("training oracle runtime is closed")
        return self._runner

    def close(self) -> None:
        runner = self._runner
        self._runner = None
        if runner is not None:
            runner.discard()
            runner.close()
        self._model = None
        self._adapters = None
        parallel = self._parallel
        self._parallel = None
        if parallel is not None:
            parallel.close()
