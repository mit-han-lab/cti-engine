"""The Gated DeltaNet operations of one head geometry, which a model's ``LayerServices``
resolves once for the layers that share it.

The operations are ``ctikernels.gdn``'s in serving and torch references in CPU
tests; nothing here selects a provider or branches on the device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch

from cti.utils import init_logger

from .layer import GatedDeltaNet

if TYPE_CHECKING:
    from ctikernels.gdn import GatedDeltaNetDecode, GatedDeltaNetPrefill

logger = init_logger(__name__)


class GatedDeltaNetOps(NamedTuple):
    """The operations of one head geometry on one device, shared by its layers."""

    decode: GatedDeltaNetDecode
    prefill: GatedDeltaNetPrefill
    training: GatedDeltaNetTrainingOperation | None


def resolve_gated_delta_net_ops(
    device: torch.device,
    *,
    num_key_heads: int,
    num_value_heads: int,
    training: bool,
) -> GatedDeltaNetOps:
    """The BF16 operations of one head geometry on ``device`` (training too when
    ``training``)."""

    from ctikernels.gdn import GatedDeltaNetDecode

    if torch.get_default_dtype() != torch.bfloat16:
        raise ValueError("Gated DeltaNet layers require BF16 model execution")
    decode = GatedDeltaNetDecode(
        num_k_heads=num_key_heads, num_v_heads=num_value_heads, device=device
    )
    prefill = _prefill(device, num_key_heads=num_key_heads, num_value_heads=num_value_heads)
    logger.info_rank0(
        "Gated DeltaNet providers on %s: decode %s, prefill %s",
        device,
        decode.provider,
        prefill.provider,
    )
    return GatedDeltaNetOps(
        decode,
        prefill,
        (
            _resolve_training_operation(
                device, num_key_heads=num_key_heads, num_value_heads=num_value_heads
            )
            if training
            else None
        ),
    )


def prefill_exports_states(
    device: torch.device, *, num_key_heads: int, num_value_heads: int
) -> bool:
    """Whether the prefill ``resolve_gated_delta_net_ops`` builds for this head geometry on
    ``device`` exports the captured states prefix reuse restores."""

    return _prefill(
        device, num_key_heads=num_key_heads, num_value_heads=num_value_heads
    ).supports_state_capture


def _prefill(
    device: torch.device, *, num_key_heads: int, num_value_heads: int
) -> GatedDeltaNetPrefill:
    """The one construction of a head geometry's prefill on ``device``, which captures
    states on the grid the layers' state spec restates."""

    from ctikernels.gdn import GatedDeltaNetPrefill

    if (grid := GatedDeltaNetPrefill.CAPTURE_GRID) != GatedDeltaNet.EXPORT_GRANULARITY:
        raise RuntimeError(
            f"ctikernels.gdn's prefill captures on a {grid}-token grid, the layers export on "
            f"{GatedDeltaNet.EXPORT_GRANULARITY}"
        )
    return GatedDeltaNetPrefill(
        num_k_heads=num_key_heads, num_v_heads=num_value_heads, device=device
    )


class GatedDeltaNetTrainingOperation(Protocol):
    """``ctikernels.gdn.GatedDeltaNetTraining``: packed sequences from zero state,
    differentiable in ``projected`` only; ``max_sequence_length`` bounds every sequence."""

    def __call__(
        self,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        max_sequence_length: int,
    ) -> torch.Tensor: ...


def _resolve_training_operation(
    device: torch.device,
    *,
    num_key_heads: int,
    num_value_heads: int,
) -> GatedDeltaNetTrainingOperation:
    """The training operation of one head geometry on ``device``."""

    from ctikernels.gdn import GatedDeltaNetTraining

    return GatedDeltaNetTraining(
        num_k_heads=num_key_heads,
        num_v_heads=num_value_heads,
        device=device,
    )


__all__ = [
    "GatedDeltaNetOps",
    "GatedDeltaNetTrainingOperation",
    "prefill_exports_states",
    "resolve_gated_delta_net_ops",
]
