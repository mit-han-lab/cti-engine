"""cti-kernels' ``Linear`` replaced by a recording operation, for CPU tests of the GEMM wiring."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import cti.layers.linear_method


@dataclass(frozen=True)
class GemmCall:
    rows: int
    shape: tuple[int, int]
    gate_stride: int  # 0: no gate
    workspace: torch.Tensor | None


# M buckets of the fake's dispatch: every row count of a bucket shares one plan.
BUCKETS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16))


class GemmRecorder:
    """Every ``Linear`` the layers construct from now on records its calls.

    ``decision`` buckets rows as ``BUCKETS`` and reports a CuTe kernel up to ``cute_rows``
    rows (gated: up to ``fused_rows``), cuBLAS above. ``workspace_bytes`` stands in for
    split-K scratch: one FP32 value per output column, and one more per input column at a
    gated call site. ``gate_fused`` admits up to ``fused_rows`` rows.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, fused_rows: int = 0, cute_rows: int = 16
    ) -> None:
        self.fused_rows = fused_rows
        self.cute_rows = cute_rows
        self.operations: list[FakeLinear] = []
        monkeypatch.setattr(cti.layers.linear_method, "Linear", self._construct)

    @property
    def calls(self) -> list[GemmCall]:
        return [call for operation in self.operations for call in operation.calls]

    def _construct(self) -> FakeLinear:
        self.operations.append(FakeLinear(self))
        return self.operations[-1]


class FakeLinear:
    def __init__(self, recorder: GemmRecorder) -> None:
        self._recorder = recorder
        self.calls: list[GemmCall] = []

    @property
    def provider(self) -> str:
        """The last call's route: ``fake-fused`` for a gated call, else ``fake``."""

        return "fake-fused" if self.calls and self.calls[-1].gate_stride else "fake"

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        gate: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        stride = 0 if gate is None else gate.stride(0)
        call = GemmCall(x.shape[0], tuple(weight.shape), stride, workspace)
        self.calls.append(call)
        if gate is not None:
            x = (x.float() * gate.float().sigmoid()).to(x.dtype)
        y = F.linear(x, weight)
        return y if out is None else out.copy_(y)

    def workspace_bytes(
        self, n: int, k: int, max_m: int = 64, device: object = None, gate_stride: int = 0
    ) -> int:
        return 4 * (n + (k if gate_stride else 0))

    def gate_fused(self, n: int, k: int, gate_stride: int, m: int, device: object = None) -> bool:
        return m <= self._recorder.fused_rows

    def decision(
        self, n: int, k: int, m: int, *, gate_stride: int = 0, device: object = None
    ) -> dict[str, object]:
        bucket = next((b for b in BUCKETS if b[0] <= m <= b[1]), None)
        limit = self._recorder.fused_rows if gate_stride else self._recorder.cute_rows
        cute = bucket is not None and bucket[1] <= limit
        return {"provider": "cute" if cute else "cublas", "bucket": bucket}
