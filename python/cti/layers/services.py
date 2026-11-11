"""What one model's layers share beyond their weights.

The composition root creates one ``LayerServices`` per model instance and passes it in
``Site.services``. A layer takes its kernels from it at construction: one per geometry and
device, so the layers of one geometry share an operation and its compilations. Resources that
exist only once the model is built (feature taps, GEMM workspace, encoder-output rows) are
assigned here once by the root; layers read them at forward. Inference state is bound to each
layer instead (``kvcache.StateLayer.bind_state``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

from cti.utils import init_logger

if TYPE_CHECKING:
    from cti.forward import RoundShape
    from cti.moe import BaseMoeBackend
    from cti.recurrent import RecurrentGeometry
    from ctikernels.attention.inference import Prologue
    from ctikernels.attention.varlen import CausalVarlenAttention
    from ctikernels.cross_entropy import LinearCrossEntropyLoss
    from ctikernels.logprob import LinearSelectedTokenLogprobs

    from .gated_delta_net.operations import GatedDeltaNetOps
    from .gated_delta_net.replay import GatedDeltaNetK2
    from .norm import ResidualReduction

logger = init_logger(__name__)


class FeatureTaps:
    """Writes the residual stream after selected decoder layers into a borrowed buffer.

    Column block ``k`` of ``buffer[t]`` receives token ``t``'s stream after layer
    ``layer_ids[k]``, before the final norm: the residual the next norm stores, the FP32 sum
    of the layer's branch output and residual input rounded once to the stream dtype. Rows
    past a forward's tokens keep stale values.
    """

    def __init__(self, layer_ids: tuple[int, ...], buffer: torch.Tensor) -> None:
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("tap layer ids must be unique and non-empty")
        if buffer.ndim != 2 or buffer.shape[1] % len(layer_ids):
            raise ValueError("the tap buffer must be [rows, len(layer_ids) * hidden]")
        self.layer_ids = layer_ids
        self.buffer = buffer
        self.hidden_size = buffer.shape[1] // len(layer_ids)
        self._columns = {
            layer: slice(slot * self.hidden_size, (slot + 1) * self.hidden_size)
            for slot, layer in enumerate(layer_ids)
        }

    def store(self, layer_id: int, stream: torch.Tensor) -> None:
        columns = self._columns.get(layer_id)
        if columns is not None:
            self.buffer[: stream.shape[0], columns].copy_(stream)


@dataclass(frozen=True, slots=True)
class PrologueGeometry:
    """What an attention prologue depends on beside its device."""

    num_q_heads: int
    num_kv_heads: int
    eps: float
    head_dim: int
    rotary_dim: int
    zero_centered: bool
    # Each frequency's position axis: the prologue then takes ``[3, tokens]`` positions too.
    position_axes: tuple[int, ...] | None = None


class LayerServices:
    """One model's shared kernels, training's among them, and its taps, GEMM workspace and
    encoder-output rows once built.

    Each of ``taps``, ``gemm_workspace`` and ``image_rows`` is assigned at most once; a model
    without taps, GEMMs without split-K scratch, or a model without an image encoder read
    ``None``. ``image_rows`` ``[rows, hidden]`` are the rows the scheduler's EncoderCache
    accounts for.
    """

    taps: FeatureTaps | None = None
    gemm_workspace: torch.Tensor | None = None
    image_rows: torch.Tensor | None = None

    def __init__(self) -> None:
        # Each operation once per key, so the layers of one geometry share it.
        self._prologues: dict[tuple[torch.device, PrologueGeometry], tuple[Prologue, bool]] = {}
        self._gated_delta_nets: dict[tuple[torch.device, int, int, bool], GatedDeltaNetOps] = {}
        self._gated_delta_net_k2: dict[tuple[RoundShape, RecurrentGeometry], GatedDeltaNetK2] = {}
        self._residual_reductions: dict[tuple[int, float, bool], ResidualReduction] = {}
        self._moe: dict[torch.device, BaseMoeBackend | None] = {}
        self._training_loss: LinearCrossEntropyLoss | None = None
        self._training_logprobs: LinearSelectedTokenLogprobs | None = None
        self._packed_training_attention: CausalVarlenAttention | None = None
        self._gated_input: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    def __setattr__(self, name: str, value: object) -> None:
        if not name.startswith("_") and name in self.__dict__:
            raise RuntimeError(f"the layer services' {name} is already assigned")
        super().__setattr__(name, value)

    def gated_input(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """``x * sigmoid(gate)`` as cti-kernels' gated ``Linear`` applies it (in FP32, rounded
        once to ``x``'s dtype), for the gated products that do not fuse the gate."""

        if self._gated_input is None:
            from ctikernels.linear import gated_input

            self._gated_input = gated_input
        return self._gated_input

    def attention_prologue(
        self,
        device: torch.device,
        geometry: PrologueGeometry,
        *,
        required: bool,
        norm_dtype: torch.dtype,
    ) -> Prologue | None:
        """The attention prologue (``ctikernels.attention.inference.Prologue``) of
        ``geometry`` on ``device``. ``required``: the operation, fused where cti-kernels runs
        it as one kernel and composed of its separate operations elsewhere; a geometry it has
        no variant for raises. Otherwise only the fused kernel, of BF16 norm weights
        (``norm_dtype``); None elsewhere."""

        # The fused kernel takes BF16 projections, norm weights and caches only.
        if not required and norm_dtype != torch.bfloat16:
            return None
        key = (device, geometry)
        if key not in self._prologues:
            from ctikernels.attention.inference import Prologue

            try:
                operation = Prologue(device=device, **asdict(geometry))
            except ValueError:  # cti-kernels has no variant of this geometry
                if required:
                    raise
                return None
            kind = operation.provider
            logger.info_rank0("Attention prologue %s: %s", asdict(geometry), kind)
            self._prologues[key] = operation, kind == "cute"
        operation, fused = self._prologues[key]
        return operation if required or fused else None

    def gated_delta_net(
        self, device: torch.device, *, num_key_heads: int, num_value_heads: int, training: bool
    ) -> GatedDeltaNetOps:
        """The Gated DeltaNet operations of this head geometry on ``device``, the training
        operation too when ``training``."""

        key = (device, num_key_heads, num_value_heads, training)
        if key not in self._gated_delta_nets:
            from .gated_delta_net.operations import resolve_gated_delta_net_ops

            self._gated_delta_nets[key] = resolve_gated_delta_net_ops(
                device,
                num_key_heads=num_key_heads,
                num_value_heads=num_value_heads,
                training=training,
            )
        return self._gated_delta_nets[key]

    def gated_delta_net_k2(
        self, rounds: RoundShape, geometry: RecurrentGeometry
    ) -> GatedDeltaNetK2:
        """The model's Gated DeltaNet K2 for rounds of ``rounds``' shape over a pool of
        ``geometry``: its layers' replays share it."""

        key = (rounds, geometry)
        if key not in self._gated_delta_net_k2:
            from .gated_delta_net.replay import GatedDeltaNetK2

            self._gated_delta_net_k2[key] = GatedDeltaNetK2(geometry, shape=rounds)
        return self._gated_delta_net_k2[key]

    def residual_reduction(
        self,
        build: Callable[[], ResidualReduction],
        *,
        hidden_size: int,
        eps: float,
        zero_centered: bool,
    ) -> ResidualReduction:
        """The residual norm reduction of this geometry, which ``build`` makes collectively
        once for this model's layers."""

        key = (hidden_size, eps, zero_centered)
        if key not in self._residual_reductions:
            self._residual_reductions[key] = build()
        return self._residual_reductions[key]

    def moe(self, device: torch.device) -> BaseMoeBackend | None:
        """The routed experts' backend on ``device``: cti-kernels' fused experts and whole-block
        op on a GPU; none elsewhere, where a build only reads shapes."""

        if device not in self._moe:
            from cti.moe import create_moe_backend

            self._moe[device] = create_moe_backend() if device.type == "cuda" else None
        return self._moe[device]

    def training_loss(self) -> LinearCrossEntropyLoss:
        """SFT's fused linear cross entropy."""

        if self._training_loss is None:
            from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss

            self._training_loss = FusedLinearCrossEntropyLoss()
        return self._training_loss

    def training_logprobs(self) -> LinearSelectedTokenLogprobs:
        """GRPO's chunked selected-token log-probabilities."""

        if self._training_logprobs is None:
            from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs

            self._training_logprobs = ChunkedLinearSelectedTokenLogprobs()
        return self._training_logprobs

    def packed_training_attention(self) -> CausalVarlenAttention:
        """Packed SFT's causal varlen attention, built at its first packed forward."""

        if self._packed_training_attention is None:
            from ctikernels.attention.varlen import CausalVarlenAttention

            self._packed_training_attention = CausalVarlenAttention()
        return self._packed_training_attention


__all__ = ["FeatureTaps", "LayerServices", "PrologueGeometry"]
