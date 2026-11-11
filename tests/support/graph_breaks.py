"""The padded reference a breakable graph's replay equals bit for bit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cti.forward import BreakCall, GraphBreaks

if TYPE_CHECKING:
    from cti.forward import ForwardInputs


class PaddedBreaks(GraphBreaks):
    """Run each break on the first ``rows`` rows of a padded forward and zero the rest, as a
    breakable graph's replay does: the reference its replays equal bit for bit."""

    def __init__(self, rows: int) -> None:
        if rows < 1:
            raise ValueError("a padded forward needs at least one real row")
        self.rows = rows

    def eager(
        self,
        call: BreakCall,
        x: torch.Tensor,
        batch: ForwardInputs,
        *,
        width: int,
    ) -> torch.Tensor:
        out = x.new_zeros(x.shape[0], width)
        call(x[: self.rows], batch, out[: self.rows])
        return out
