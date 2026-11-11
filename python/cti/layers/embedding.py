# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/layers/embedding.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from cti.utils import div_ceil, nvtx_annotate

from ._vocab_loss import ShardedCrossEntropy, ShardedLogprobs
from .base import BaseOP
from .linear import ParallelEntry, ProjectionSlice
from .linear_method import (
    GemmCallSite,
    InferenceGemm,
    LinearMethod,
    Unquantized,
    declare_method,
    require_meta,
)
from .placement import Param, WeightShard

if TYPE_CHECKING:
    from cti.distributed import CollectiveGroup
    from ctikernels.cross_entropy import LinearCrossEntropyLoss
    from ctikernels.logprob import LinearSelectedTokenLogprobs

    from .site import Site


# A packed logit word holds a logit's ordered bits above ``_ID_MASK - ID``, so a larger word
# has the larger logit and then the smaller ID.
_ID_MASK = 0xFFFFFFFF


def _pack(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The int64 words of ``logits`` at vocabulary ``ids``, which broadcast against them."""

    # Adding zero turns -0.0 into 0.0, so equal values pack equally.
    bits = (logits.float() + 0.0).view(torch.int32)
    # Flipping a negative float's magnitude bits makes integer order follow float order.
    key = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).long() << 32
    return key | (_ID_MASK - ids)


def _columns(logits: torch.Tensor, start: int) -> torch.Tensor:
    """The vocabulary IDs of ``logits``' columns, the first being ``start``."""

    return torch.arange(start, start + logits.shape[-1], device=logits.device)


def _unpack_ids(words: torch.Tensor) -> torch.Tensor:
    return _ID_MASK - (words & _ID_MASK)


def _unpack_by_id(words: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The FP32 logits and IDs of each row's ``words``, by ascending ID."""

    ids = _unpack_ids(words)
    order = ids.argsort(-1)
    key = (words.gather(-1, order) >> 32).int()
    return (key ^ ((key >> 31) & 0x7FFFFFFF)).view(torch.float32), ids.gather(-1, order)


def _topk_by_id(logits: torch.Tensor, k: int, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` largest of each ``[R, V]`` row as FP32 logits and their vocabulary IDs (the
    first column being ``start``), by ascending ID; ties go to the smaller ID and -0 equals
    +0. cti-kernels' ``topk_by_id`` on CUDA, the same words composed in torch elsewhere."""

    if logits.is_cuda:
        from ctikernels.sampling import topk_by_id

        values, ids = topk_by_id(logits.contiguous(), k)
        return values, ids + start if start else ids
    return _unpack_by_id(_pack(logits, _columns(logits, start)).topk(k, sorted=False).values)


class LocalVocab:
    """One rank holds the whole vocabulary: its lookups and logits are the output."""

    @staticmethod
    def sum(y: torch.Tensor) -> torch.Tensor:
        return y

    @staticmethod
    def argmax(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(-1)

    @staticmethod
    def topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _topk_by_id(logits, k)

    @staticmethod
    def project(logits: torch.Tensor) -> torch.Tensor:
        return logits

    @staticmethod
    def lookup_training(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, weight)

    @staticmethod
    def logits_training(
        x: torch.Tensor, product: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        return product(x)

    @staticmethod
    def training_loss(operation: LinearCrossEntropyLoss) -> LinearCrossEntropyLoss:
        return operation

    @staticmethod
    def training_logprobs(
        operation: LinearSelectedTokenLogprobs,
    ) -> LinearSelectedTokenLogprobs:
        return operation


class ShardedVocab:
    """Each rank holds one shard of a vocabulary of ``num_embeddings`` IDs, its own at
    ``vocab_range``: lookups sum over the ranks and logits gather from them."""

    def __init__(
        self, group: CollectiveGroup, vocab_range: tuple[int, int], num_embeddings: int
    ) -> None:
        self._group = group
        self._vocab_range = vocab_range
        self._num_embeddings = num_embeddings
        self._entry = ParallelEntry(group)

    def sum(self, y: torch.Tensor) -> torch.Tensor:
        return self._group.all_reduce(y)

    def argmax(self, logits: torch.Tensor) -> torch.Tensor:
        # Each rank sends one packed (value, ID) word per row instead of its logits.
        start, count = self._vocab_range
        logits[:, count:] = float("-inf")
        index = logits.argmax(-1, keepdim=True)
        packed = _pack(logits.gather(-1, index), start + index).squeeze(-1)
        return _unpack_ids(self._gather(packed).amax(0))

    def topk(self, logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Each rank sends its own top k packed words per row; their top k is the global one.
        start, count = self._vocab_range
        logits[:, count:] = float("-inf")
        values, ids = _topk_by_id(logits, min(k, logits.shape[-1]), start)
        ranks = self._gather(_pack(values, ids)).transpose(0, 1).flatten(1)
        return _unpack_by_id(ranks.topk(k, sorted=False).values)

    def project(self, logits: torch.Tensor) -> torch.Tensor:
        # Rank r holds vocabulary columns [r * n, (r + 1) * n); the padding trails the last.
        logits = self._gather(logits).transpose(0, 1).flatten(1)
        # A vocabulary the ranks pad leaves strided rows; round sampling reads dense ones.
        return logits[:, : self._num_embeddings].contiguous()

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        """Every rank's ``x``, stacked on a new leading rank axis."""

        return self._group.all_gather(x).view(-1, *x.shape)

    def lookup_training(self, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # The frozen rows and the IDs take no gradient: the sum needs no backward.
        start, count = self._vocab_range
        local = x - start
        owned = (local >= 0) & (local < count)
        y = F.embedding(local.clamp(0, weight.shape[0] - 1), weight)
        return self.sum(y * owned.unsqueeze(-1).to(y.dtype))

    def enter(self, x: torch.Tensor) -> torch.Tensor:
        return self._entry.enter(x)

    def logits_training(
        self, x: torch.Tensor, product: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        logits = product(self.enter(x))
        return _GatherVocab.apply(logits, self)[..., : self._num_embeddings]

    # cti-kernels' training operations take the whole vocabulary; until they take a shard,
    # a sharded head trains through ShardedLogprobs.
    def training_loss(self, operation: LinearCrossEntropyLoss) -> ShardedCrossEntropy:
        return ShardedCrossEntropy(self._entry, self._vocab_range, self._gather)

    def training_logprobs(self, operation: LinearSelectedTokenLogprobs) -> ShardedLogprobs:
        return ShardedLogprobs(self._entry, self._vocab_range, self._gather)


class _GatherVocab(torch.autograd.Function):
    """The ranks' logits shards side by side; each rank's gradient is its own columns."""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, vocab: ShardedVocab) -> torch.Tensor:
        ctx.rank, ctx.width = vocab._group.rank, logits.shape[-1]
        return torch.cat(vocab._gather(logits.contiguous()).unbind(0), dim=-1)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient.narrow(-1, ctx.rank * ctx.width, ctx.width).contiguous(), None


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        site: Site,
    ):
        tp_rank = site.group.rank
        self.tp_size = site.group.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        # Trailing ranks of a small vocabulary may own no rows at all.
        start_idx = min(self.num_embeddings_tp * tp_rank, num_embeddings)
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        # One rank looks up every ID. Several ranks read zeros outside their own rows and
        # sum their lookups.
        self._vocab_window: WeightShard | None = None
        self._lookup_range: tuple[int, int] | None = None
        self._vocab: LocalVocab | ShardedVocab = LocalVocab()
        if self.tp_size > 1:
            self._vocab_window = WeightShard(0, start_idx, finish_idx, self.num_embeddings_tp)
            self._lookup_range = self.vocab_range
            self._vocab = ShardedVocab(site.group, self.vocab_range, num_embeddings)

    def _param(self, path: str, name: str, tensor: torch.Tensor) -> Param:
        windows = () if self._vocab_window is None else (self._vocab_window,)
        return Param(path, tuple(tensor.shape), tensor.dtype, windows)

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from ctikernels import indexing

        return self._vocab.sum(
            indexing(weights=self.weight, indices=x, vocab_range=self._lookup_range)
        )

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        return self._vocab.lookup_training(self.weight, x)


class ParallelLMHead(VocabParallelEmbedding):
    """The vocabulary projection. An untied head holds its weight as its ``method`` declares
    it (``Unquantized`` unless the checkpoint quantizes it) and computes its logits by that
    method; a tied head reads the embedding's rows."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        *,
        tied_embedding: VocabParallelEmbedding | None = None,
        site: Site,
    ):
        """A ``tied_embedding`` lends its rows; the head then stores no tensor of its own. A
        training head takes its loss and log-probabilities from ``site.services``."""
        super().__init__(num_embeddings, embedding_dim, site=site)
        self.local_output_size, self.local_input_size = self.num_embeddings_tp, embedding_dim
        self.gate_stride = 0
        self.output_row_stride = self.local_output_size
        self._tied_embedding = tied_embedding
        self.services = site.services
        self._declared: tuple[str, ...] = ("weight",)
        self.bias: torch.Tensor | None = None
        self.method: LinearMethod | None = None
        self._tied_gemm: InferenceGemm | None = None
        if tied_embedding is None:
            self.method, self._declared = declare_method(self, Unquantized, self._declared)
        elif not bias:
            self._tied_gemm = InferenceGemm(lambda: tied_embedding.weight, site.services)
        if bias:
            self.bias = torch.empty(self.num_embeddings_tp)
        self._trains = site.training
        if site.training:
            loss, logprobs = site.services.training_loss(), site.services.training_logprobs()
            # Token rows per logits chunk of each, which size the head's working set.
            self.training_chunk_rows = (loss.chunk_size, logprobs.chunk_size)
            self._training_loss = self._vocab.training_loss(loss)
            self._training_logprobs = self._vocab.training_logprobs(logprobs)

    def use_method(self, method: type[LinearMethod]) -> None:
        """Store and multiply an untied head's weight as ``method`` does, before weights load,
        on the device and in the dtype in effect."""

        if self._tied_embedding is not None:
            raise ValueError("a tied LM head reads the embedding's rows and holds no method")
        if self._trains and not method.dense:
            # A training head multiplies the dense matrix.
            raise ValueError(f"{method.__name__} has no training path for an LM head")
        require_meta(tensor for _, tensor in self._tensors())
        self.method, self._declared = declare_method(self, method, self._declared)

    @property
    def windows(self) -> tuple[WeightShard, ...]:
        return () if self._vocab_window is None else (self._vocab_window,)

    @property
    def output_slices(self) -> tuple[ProjectionSlice, ...]:
        return (ProjectionSlice(name="full", offset=0, size=self.num_embeddings_tp),)

    @property
    def head_weight(self) -> torch.Tensor:
        """The vocabulary rows the head projects onto: its own or the tied embedding's."""
        return (self._tied_embedding or self).weight

    def _tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        # A tied head reads the embedding's rows; the checkpoint mapping skips a redundant
        # head copy without reading it.
        return iter(()) if self._tied_embedding is not None else super()._tensors()

    def _param(self, path: str, name: str, tensor: torch.Tensor) -> Param:
        windows = self.windows
        if windows and name != "bias" and self.method is not None:
            try:
                windows = self.method.windows(name, windows)
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from None
        return Param(path, tuple(tensor.shape), tensor.dtype, windows)

    def inference_gemms(self) -> Iterator[GemmCallSite]:
        if self.method is not None:
            yield from self.method.call_sites()
        elif self._tied_gemm is not None:
            yield self._tied_gemm
        yield from super().inference_gemms()

    def validate_training_targets(self, target_ids: torch.Tensor) -> None:
        """Validate CPU targets before they cross into the fused CUDA loss kernel."""

        if target_ids.device.type != "cpu":
            raise ValueError("Training targets must be validated on CPU")
        if target_ids.ndim != 1 or target_ids.numel() == 0:
            raise ValueError("Training targets must be a non-empty vector")
        if target_ids.dtype != torch.int64:
            raise TypeError("Training targets must use int64")
        minimum_tensor, maximum_tensor = torch.aminmax(target_ids)
        minimum = int(minimum_tensor)
        maximum = int(maximum_tensor)
        if minimum < 0 or maximum >= self.num_embeddings:
            raise ValueError(
                f"Training targets must be in [0, {self.num_embeddings}); "
                f"observed range [{minimum}, {maximum}]"
            )

    def argmax(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy token ID of every row of ``x``; ties go to the smallest ID."""

        return self.gather_argmax(self.local_logits(x))

    @nvtx_annotate("LMHead")
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits of every row of ``x``, contiguous."""

        return self.gather_logits(self.local_logits(x))

    def local_logits(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's vocabulary-shard logits of every row of ``x``: one weight read for
        rows that ``gather_argmax`` and ``gather_logits`` then split between them."""

        return self._logits(x)

    def gather_argmax(self, local: torch.Tensor) -> torch.Tensor:
        """Greedy token ID of every row of ``local_logits``; ties go to the smallest ID.
        Several ranks may overwrite ``local``'s padding columns."""

        return self._vocab.argmax(local)

    def gather_logits(self, local: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits of every row of ``local_logits``, contiguous."""

        return self._vocab.project(local)

    @staticmethod
    def topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``k`` largest of full-vocabulary ``logits`` (FP32) and their token IDs
        (int64), ordered by ascending ID; ties go to the smaller ID."""

        return LocalVocab.topk(logits, k)

    def gather_topk(self, local: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``k`` largest logits of every row of ``local_logits`` (FP32) and their token
        IDs (int64), ordered by ascending ID; ties go to the smaller ID. Several ranks may
        overwrite ``local``'s padding columns."""

        return self._vocab.topk(local, k)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        def product(x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, self.head_weight, self.bias)

        return self._vocab.logits_training(x, product)

    def forward_training_loss(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self._training_loss.forward(
            hidden_states,
            self.head_weight,
            target_ids,
            bias=self.bias,
        )

    def forward_training_logprobs(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        return self._training_logprobs.forward(
            hidden_states,
            self.head_weight,
            target_ids,
            bias=self.bias,
            temperature=temperature,
        )

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's vocabulary-shard logits of every row of ``x``, for inference."""

        if self.method is not None:
            return self.method.apply(x, self.bias)
        if self.bias is None:
            return self._tied_gemm(x)
        return F.linear(x, self.head_weight, self.bias)
