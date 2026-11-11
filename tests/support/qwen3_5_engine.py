"""Engine models on CPU over the runner's inference state, with reference kernels bound."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig

import cti.layers.activation as activation_module
from cti.forward import ForwardInputs
from cti.kvcache import RecurrentStatePool
from cti.recurrent import RecurrentBackend
from tests.support import reference_ops
from tests.support.forward_rows import decode_entry, extend_entry, forward_rows
from tests.support.gated_delta_net import fp32_history
from tests.support.paged_attention import (
    ReferencePagedAttentionBackend,
    bind_model_state,
    kv_group,
)
from tests.support.tp_shards import CPU, load_sharded

if TYPE_CHECKING:
    import pytest

    from cti.distributed.runtime import ParallelGroup
    from cti.forward import ImageBatch
    from cti.layers import Adapter
    from cti.models.base import BaseLLMModel
    from cti.multimodal import ImagePreprocessing
    from tests.support.forward_rows import Entry

# FP32 row-parallel reductions reorder additions; nothing else may differ. Measured worst
# (tiny Qwen3.5 TP2): 3e-6 in logits up to 4, 1e-6 in state; a wrong window is O(1).
TP_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}
# A chunked prefill beside a whole prompt, a mixed batch, then two decode steps.
STEPS = (
    (extend_entry(0, 0, 12), extend_entry(1, 0, 9)),
    (extend_entry(0, 12, 20), decode_entry(1, 9)),
    (decode_entry(0, 20), decode_entry(1, 10)),
    (decode_entry(0, 21), decode_entry(1, 11)),
)


def bind_reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the CPU references at every kernel seam the model reaches beyond its services
    (``tp_shards.ReferenceLayerServices``)."""

    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)
    reference_ops.bind_norm_references(monkeypatch)
    monkeypatch.setattr(
        activation_module.inference_activation, "silu_and_mul", reference_ops.silu_and_mul
    )
    monkeypatch.setattr("ctikernels.indexing", reference_ops.indexing)
    monkeypatch.setattr("ctikernels.store_cache", reference_ops.store_cache)
    reference_ops.bind_vision_references(monkeypatch)


def load_model(
    folder: Path,
    *,
    lora_adapters: Adapter | None = None,
    group: ParallelGroup | None = None,
    training: bool = False,
    image_preprocessing: ImagePreprocessing | None = None,
) -> BaseLLMModel:
    """Build and load the engine model as the runner does, in FP32, on the rank of
    ``group`` (a single rank without one)."""

    hf_config = AutoConfig.from_pretrained(folder)
    return load_sharded(
        hf_config,
        folder,
        group=group,
        lora_adapters=lora_adapters,
        training=training,
        image_preprocessing=image_preprocessing,
    )[0]


def run_steps(
    folder: Path, *, group: ParallelGroup | None = None
) -> tuple[EagerEngine, list[torch.Tensor]]:
    """Teacher-force ``STEPS`` over two seeded token rows; return the engine and step logits."""

    model = load_model(folder, group=group)
    generator = torch.Generator().manual_seed(2)
    vocab_size = model.lm_head.num_embeddings
    token_ids = torch.randint(vocab_size, (2, 24), generator=generator, dtype=torch.int32)
    engine = EagerEngine(model, token_ids)
    with torch.inference_mode():
        return engine, [engine.execute(*entries) for entries in STEPS]


def assert_ranks_match_single_rank(ranks: list[list], reference: list[torch.Tensor]) -> None:
    """Every rank's step logits equal rank 0's and match the single-rank model's."""

    for rank, logits in enumerate(ranks):
        assert logits == ranks[0], f"rank {rank} logits differ from rank 0"
    for step, (values, expected) in enumerate(zip(ranks[0], reference, strict=True)):
        logits = torch.tensor(values, dtype=torch.float32)
        assert logits.shape == expected.shape
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(logits, expected, **TP_TOLERANCE, msg=f"step {step}")
        assert torch.equal(logits.argmax(-1), expected.argmax(-1)), f"step {step} greedy token"


class EagerEngine:
    """The model over the runner's inference state, one eager batch at a time. KV rows are
    scattered, and both pools start as NaN, so reading an unwritten value poisons the logits."""

    def __init__(
        self,
        model: BaseLLMModel,
        token_ids: torch.Tensor,
        *,
        snapshot_rows: int = 0,
        image_rows: int = 0,
    ) -> None:
        num_requests, max_seq_len = token_ids.shape
        paged_layers = tuple(model.iter_attention_layers())
        group = kv_group(
            len(paged_layers),
            paged_layers[0].num_kv_heads,
            paged_layers[0].head_dim,
            num_requests * max_seq_len,
        )
        for layer in group.layers:
            layer.fill_(torch.nan)
        generator = torch.Generator().manual_seed(1)
        self._cache_index_table = torch.randperm(
            num_requests * max_seq_len, generator=generator, dtype=torch.int32
        ).view(num_requests, max_seq_len)
        self._attention_backend = ReferencePagedAttentionBackend(group, self._cache_index_table)
        services = model.test_services
        self.recurrent_state: RecurrentStatePool | None = None
        self._recurrent_backend: RecurrentBackend | None = None
        if recurrent_layers := model.iter_recurrent_layers():
            # ctikernels.gdn keeps the convolution history in BF16. FP32 history leaves the
            # chunk and decode boundaries unrounded, so the whole engine model runs in FP32
            # like the reference; the layer tests cover the BF16 history.
            self.recurrent_state = RecurrentStatePool(
                [fp32_history(layer.state_spec) for layer in recurrent_layers],
                num_slots=num_requests,
                num_snapshot_rows=snapshot_rows,
                device=CPU,
            )
            for index in range(len(recurrent_layers)):
                for view in self.recurrent_state.layer_views(index):
                    view.fill_(torch.nan)
            self._recurrent_backend = RecurrentBackend(self.recurrent_state, device=CPU)
        bind_model_state(model, self._attention_backend, self.recurrent_state)
        if image_rows:
            # The encoder-output rows start as NaN too.
            hidden = model.model.embed_tokens.weight.shape[1]
            services.image_rows = torch.full((image_rows, hidden), torch.nan)
        self._model = model
        self._token_ids = token_ids.clone()

    def reuse_slot(self, request_index: int, token_ids: torch.Tensor) -> None:
        """Hand a finished request's slot, token row and KV rows to a new request."""

        self._token_ids[request_index] = token_ids

    def prepare(
        self,
        *entries: Entry,
        rope_positions: torch.Tensor | None = None,
        images: ImageBatch | None = None,
    ) -> ForwardInputs:
        """Prepare an unpadded batch as ``InferenceRunner.submit`` does.

        Without ``rope_positions`` every axis reads the sequence positions, as for text.
        """

        rows = forward_rows(*entries)
        requests = torch.cat([torch.full((e.length,), e.slot) for e in entries])
        positions = torch.cat([torch.arange(e.start, e.end) for e in entries])
        recurrent = self._recurrent_backend
        return ForwardInputs(
            self._token_ids[requests, positions],
            positions if rope_positions is None else rope_positions,
            self._attention_backend.prepare_metadata(rows),
            rows.size,
            rows.uses_ragged_attention,
            None if recurrent is None else recurrent.prepare_metadata(rows),
            images,
            logit_rows=torch.tensor(rows.logit_rows),
        )

    def execute(
        self,
        *entries: Entry,
        rope_positions: torch.Tensor | None = None,
        images: ImageBatch | None = None,
    ) -> torch.Tensor:
        """The logits ``[entries, vocab]`` of each entry's last token."""

        batch = self.prepare(*entries, rope_positions=rope_positions, images=images)
        if self._recurrent_backend is not None:
            self._recurrent_backend.restore(batch)
        logits = self._model.forward(batch)
        if self._recurrent_backend is not None:
            self._recurrent_backend.export(batch)
        return logits


__all__ = [
    "STEPS",
    "TP_TOLERANCE",
    "EagerEngine",
    "assert_ranks_match_single_rank",
    "bind_reference_kernels",
    "load_model",
    "run_steps",
]
