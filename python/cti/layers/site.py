"""Where a layer is built: its rank's group, the expert split, the device of its buffers and
its model's services.

A model passes its ``Site`` down unchanged; each layer reads it to size, window, initialize
and choose its kernels at construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cti.distributed import CollectiveGroup

    from .norm import ResidualReduction
    from .placement import ExpertPlacement
    from .services import LayerServices


@dataclass(frozen=True, slots=True)
class Site:
    """Everything a layer's construction depends on.

    ``group`` is the group an exit reduces over (construction communicates only as the
    services build the group's residual reductions, alike on every rank);
    ``experts`` splits routed experts; ``device`` holds computed buffers such as RoPE tables,
    while parameters follow the ambient device (meta until weights load); ``services`` are
    the model's own (``LayerServices``).
    """

    group: CollectiveGroup
    experts: ExpertPlacement
    device: torch.device
    training: bool = False
    services: LayerServices = field(kw_only=True)

    def residual_reduction(
        self, *, hidden_size: int, eps: float, zero_centered: bool
    ) -> ResidualReduction | None:
        """The one decision on where a residual stream's branch outputs are summed: none on
        one rank; otherwise one reduction per norm geometry, built by the group once for the
        model's services, that its residual norms run and that the row-parallel exits before
        them leave their shares to (``deferred``). A model's services are its own, so the
        target and the draft never share one."""

        if self.group.size == 1:
            return None
        geometry = {"hidden_size": hidden_size, "eps": eps, "zero_centered": zero_centered}
        group = self.group
        return self.services.residual_reduction(
            lambda: group.residual_reduction(**geometry), **geometry
        )


__all__ = ["Site"]
