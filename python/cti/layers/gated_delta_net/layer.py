"""The Gated DeltaNet layer: frozen weights between a merged input projection and an
output projection, run by its model's shared operations over its bound pool row.

The layer computes ``y`` from the merged projection rows ``projected``; its columns
are ``q | k | v | z | b | a`` (``ctikernels.gdn``'s fusion boundary). The short causal
convolution, the gates, the delta-rule recurrence and the gated RMSNorm all run inside
the operations. The model owns the projections and their order.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import torch

from cti.kvcache import SlotStateSpec, StateSlot

from ..base import BaseOP
from ..placement import Param, WeightShard
from .replay import GatedDeltaNetK2

if TYPE_CHECKING:
    from cti.forward import ForwardInputs, RoundShape
    from cti.kvcache import BoundSlot
    from cti.recurrent import (
        LayerReplay,
        RecurrentGeometry,
        RecurrentMetadata,
        RecurrentVerify,
    )

    from ..site import Site


class GatedDeltaNetWeights(NamedTuple):
    """The frozen weights in the operations' argument order (inference and training)."""

    conv_weight: torch.Tensor  # BF16 [conv_channels, conv_kernel_size]
    A_log: torch.Tensor  # FP32 [num_value_heads]
    dt_bias: torch.Tensor  # FP32 [num_value_heads]
    norm_weight: torch.Tensor  # FP32 [value_head_dim]


class GatedDeltaNet(BaseOP):
    """A Gated DeltaNet mixer.

    It holds no inference state: it reads the pool row views ``bind_state`` gives it. Under TP rank ``r`` owns key heads ``[r*Hk/tp, (r+1)*Hk/tp)`` and the value heads that
    read them, so the recurrence needs no collective; head counts, weights and state are this
    rank's. The operations are chosen at construction, the training one with
    ``site.training``.
    """

    # The fusion contract of ctikernels.gdn: 128-wide heads and a 4-tap convolution.
    HEAD_DIM = 128
    CONV_KERNEL_SIZE = 4
    # Speculative rounds verify and commit these layers' blocks in one K2 transaction.
    transactional = True
    ROUND_TRANSACTION = GatedDeltaNetK2
    # A prefill call captures the state after its most whole multiples of this many tokens:
    # ctikernels.gdn's GatedDeltaNetPrefill.CAPTURE_GRID, restated so that the state spec
    # imports no kernel; building the prefill checks they agree (``operations._prefill``).
    EXPORT_GRANULARITY = 64
    # ctikernels.gdn keeps the convolution history in BF16.
    CONV_DTYPE = torch.bfloat16

    def __init__(
        self,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        site: Site,
    ) -> None:
        if key_head_dim != self.HEAD_DIM or value_head_dim != self.HEAD_DIM:
            raise ValueError(
                f"Gated DeltaNet head dimensions must be {self.HEAD_DIM}, got key "
                f"{key_head_dim} and value {value_head_dim}"
            )
        if conv_kernel_size != self.CONV_KERNEL_SIZE:
            raise ValueError(
                f"Gated DeltaNet convolution kernel size must be {self.CONV_KERNEL_SIZE}, "
                f"got {conv_kernel_size}"
            )
        if num_key_heads < 1 or num_value_heads % num_key_heads:
            raise ValueError(
                f"num_value_heads ({num_value_heads}) must be a multiple of num_key_heads "
                f"({num_key_heads})"
            )
        rank, size = site.group.rank, site.group.size
        num_key_heads, num_value_heads = self.rank_heads(num_key_heads, num_value_heads, size)
        key_size = num_key_heads * key_head_dim
        value_size = num_value_heads * value_head_dim
        conv_channels = 2 * key_size + value_size
        self.device = site.device
        # This rank's geometry: the operations' heads and the pool row's state.
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.state_spec = gated_delta_net_state(
            num_value_heads, key_head_dim, value_head_dim, conv_channels, conv_kernel_size - 1
        )
        # Checkpoint shapes and declared dtypes: the loader casts to these.
        self.conv_weight = torch.empty(conv_channels, 1, conv_kernel_size, dtype=self.CONV_DTYPE)
        self.A_log = torch.empty(num_value_heads, dtype=torch.float32)
        self.dt_bias = torch.empty(num_value_heads, dtype=torch.float32)
        self.norm_weight = torch.empty(value_head_dim, dtype=torch.float32)
        # The pool row's state and convolution history, once bound.
        self._row: tuple[torch.Tensor, torch.Tensor] | None = None
        # The model's K2 is shared through its services.
        self._services = site.services
        self._replay: LayerReplay | None = None
        self._decode, self._prefill, self._training_op = site.services.gated_delta_net(
            site.device,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            training=site.training,
        )
        # The checkpoint merges the convolution from its q, k and v channel groups; each
        # rank reads its heads' channels of each group, and its value heads' gate weights.
        self._conv_windows: tuple[WeightShard, ...] = ()
        self._value_head_window: WeightShard | None = None
        if size > 1:
            key = WeightShard(0, rank * key_size, (rank + 1) * key_size)
            value = WeightShard(0, rank * value_size, (rank + 1) * value_size)
            self._conv_windows = (key, key, value)
            self._value_head_window = WeightShard(
                0, rank * num_value_heads, (rank + 1) * num_value_heads
            )

    @staticmethod
    def rank_heads(num_key_heads: int, num_value_heads: int, size: int) -> tuple[int, int]:
        """The key and value heads one of ``size`` tensor-parallel ranks owns."""

        if num_key_heads % size:
            raise ValueError(
                f"Gated DeltaNet key heads ({num_key_heads}) must be divisible by the "
                f"tensor-parallel size ({size}), so that each rank owns whole key "
                "heads and the value heads that read them"
            )
        return num_key_heads // size, num_value_heads // size

    def _param(self, path: str, name: str, tensor: torch.Tensor) -> Param:
        windows: tuple[WeightShard, ...] = ()
        if name == "conv_weight":
            windows = self._conv_windows
        elif name in ("A_log", "dt_bias") and self._value_head_window is not None:
            windows = (self._value_head_window,)
        # The gated norm's weight spans one head's width, so every rank holds all of it.
        return Param(path, tuple(tensor.shape), tensor.dtype, windows)

    def warm_up(self, max_tokens: int, *, max_sequences: int) -> None:
        """Compile the prefill (shared by the layers of one head geometry, so a second
        layer's call compiles nothing) for every extend forward within these bounds, captures
        or not; a non-CuTe provider compiles per shape on first use instead."""

        self._prefill.warm_up(max_tokens, max_sequences=max_sequences)

    def warm_up_training(self, max_tokens: int) -> None:
        """Run the training op forward and backward once over one sequence of
        ``max_tokens`` tokens, so that its kernels compile and autotune now: autotuning
        benchmarks every candidate config, and one needing more local memory than the device
        allows grows it for the life of the process."""

        keys = self.num_key_heads * self.HEAD_DIM
        values = self.num_value_heads * self.HEAD_DIM
        width = 2 * (keys + values + self.num_value_heads)
        device = self.A_log.device
        with torch.enable_grad():
            projected = torch.zeros(
                max_tokens, width, dtype=self.conv_weight.dtype, device=device, requires_grad=True
            )
            cu_seqlens = torch.tensor([0, max_tokens], dtype=torch.int32, device=device)
            self.forward_training(projected, cu_seqlens, max_tokens).sum().backward()

    @property
    def exports_prefill_states(self) -> bool:
        """Whether an extend forward exports the batch's captures."""

        return self._prefill.supports_state_capture

    def state_slots(self) -> tuple[StateSlot, ...]:
        """One slot-row state: ``state_spec``."""

        return (StateSlot("recurrent", state=self.state_spec),)

    def bind_state(self, bound: Mapping[str, BoundSlot]) -> None:
        """Read and write the pool row views of the ``recurrent`` slot from now on."""

        views = bound["recurrent"]
        assert isinstance(views, Mapping)
        self._row = (views["state"], views["conv"])

    def _bound_row(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._row is None:
            raise RuntimeError("the Gated DeltaNet layer is not bound to its pool row")
        return self._row

    def reset_scratch(self) -> None:
        """Forget the decode's scratch for the bound pool row, whose memory came back
        without its contents: eagerly, before the next decode or graph replay."""

        self._decode.reset_scratch(self._bound_row()[0])

    def create_replay(
        self, *, max_batch: int, rounds: RoundShape, geometry: RecurrentGeometry
    ) -> LayerReplay:
        """The layer's replay in its model's K2 over a pool of ``geometry`` for rounds of
        ``rounds``' shape, made once on its bound row; the layer keeps it for its verify
        forwards."""

        if self._replay is not None:
            raise RuntimeError("the Gated DeltaNet layer's replay is already made")
        if max_batch != rounds.max_rows:
            raise ValueError(f"a replay of {rounds.max_rows} rows cannot serve {max_batch}")
        transaction = self._services.gated_delta_net_k2(rounds, geometry)
        self._replay = replay = transaction.join(self, self._bound_row())
        return replay

    @property
    def weights(self) -> GatedDeltaNetWeights:
        """The weights as the operations take them (the convolution as ``[channels, taps]``)."""

        conv_weight = self.conv_weight
        return GatedDeltaNetWeights(
            conv_weight.view(conv_weight.shape[0], conv_weight.shape[-1]),
            self.A_log,
            self.dt_bias,
            self.norm_weight,
        )

    def forward(self, projected: torch.Tensor, batch: ForwardInputs) -> torch.Tensor:
        """``y [tokens, num_value_heads * value_head_dim]``, a break of ``batch``'s graph:
        extend rows go to the prefill operation, decode rows to the decode operation, and
        verified blocks to the layer's replay."""

        width = self.num_value_heads * self.HEAD_DIM
        return batch.graph_breaks.eager(self._core_into, projected, batch, width=width)

    def _core_into(
        self,
        projected: torch.Tensor,
        batch: ForwardInputs,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        metadata = batch.recurrent_metadata
        num_extend = metadata.num_extend_requests
        verify = metadata.verify
        if verify is not None and num_extend == 0:
            return self._into(out, self._verified(projected, verify))
        state, conv_state = self._bound_row()
        if num_extend == 0:
            # The decode writes into a graph's buffer when it is given one.
            return self._decode(
                projected, *self.weights, metadata.slot_ids, state, conv_state, out=out
            )
        if num_extend == len(metadata.slot_ids) and verify is None:
            return self._extend(projected, metadata, out=out)

        # Mixed: the prefill writes the extend rows of y in place; decode rows or verified
        # blocks follow.
        split = metadata.num_extend_tokens
        if out is None:
            out = projected.new_empty(projected.shape[0], self.num_value_heads * self.HEAD_DIM)
        self._extend(projected, metadata, out=out[:split])
        if verify is not None:
            # The verified blocks' rows: verify leaves live state unchanged.
            out[split:] = self._verified(projected[split:], verify)
            return out
        self._decode(
            projected[split:],
            *self.weights,
            metadata.slot_ids[num_extend:],
            state,
            conv_state,
            out=out[split:],
        )
        return out

    def _verified(self, projected: torch.Tensor, verify: RecurrentVerify) -> torch.Tensor:
        if self._replay is None:
            raise RuntimeError("a verify forward reached a Gated DeltaNet layer without a replay")
        return self._replay.verify(projected, verify)

    @staticmethod
    def _into(out: torch.Tensor | None, y: torch.Tensor) -> torch.Tensor:
        if out is None:
            return y
        out.copy_(y)
        return out

    def forward_training(
        self,
        projected: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_sequence_length: int,
    ) -> torch.Tensor:
        """``y`` of the packed sequences ``cu_seqlens`` bounds in ``projected``'s rows, none
        longer than ``max_sequence_length``."""

        return self._training_op(
            projected, *self.weights, cu_seqlens, max_sequence_length=max_sequence_length
        )

    def training_bytes(self, tokens: int, sequences: int) -> tuple[int, int]:
        """What the training op keeps from forward to backward over ``tokens`` tokens in up
        to ``sequences`` sequences, and what its backward allocates beside them, as the op
        reports them for the provider it resolved."""

        return self._training_op.training_bytes(tokens, sequences)

    def _extend(
        self,
        projected: torch.Tensor,
        metadata: RecurrentMetadata,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``y`` of the batch's leading extend rows of ``projected``, written into ``out``
        when given."""

        state, conv_state = self._bound_row()
        return self._prefill(
            projected[: metadata.num_extend_tokens],
            *self.weights,
            metadata.cu_seqlens,
            metadata.slot_ids[: metadata.num_extend_requests],
            metadata.has_initial_state,
            state,
            conv_state,
            out=out,
            capture_rows=metadata.capture_rows,
        )


def gated_delta_net_state(
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    conv_channels: int,
    conv_history: int = 3,
) -> SlotStateSpec:
    """One request's Gated DeltaNet state: FP32 ``state`` ``[num_value_heads,
    value_head_dim, key_head_dim]`` (value-major, the kernels' pool layout), then ``conv``,
    the last ``conv_history`` pre-convolution rows ``[conv_history, conv_channels]``."""
    return SlotStateSpec(
        (
            ("state", (num_value_heads, value_head_dim, key_head_dim), torch.float32),
            ("conv", (conv_history, conv_channels), GatedDeltaNet.CONV_DTYPE),
        ),
        export_granularity=GatedDeltaNet.EXPORT_GRANULARITY,
    )


__all__ = [
    "GatedDeltaNet",
    "GatedDeltaNetWeights",
    "gated_delta_net_state",
]
