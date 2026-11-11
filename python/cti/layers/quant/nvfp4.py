"""NVFP4, ModelOpt's 4-bit format: packed E2M1 weights with one E4M3 scale per 16 columns and
an FP32 global scale per checkpoint module; activations quantize at run time per token and 16
columns under a calibrated FP32 input scale (``numerics.py``)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch
import torch.nn.functional as F

from ctikernels.linear.nvfp4 import NVFP4Linear, NVFP4Weight

from ..linear_method import GemmRoutes
from ..placement import WeightShard
from . import numerics
from ._utils import block_windows
from .fp8_static import QuantizedMethod, merged_input_scale

# Packed E2M1 values per stored byte.
_PACKED = 2
# Weight columns per stored column of each leaf windowed by columns.
_COLUMN_BLOCKS = MappingProxyType({"weight": _PACKED, "weight_scale": numerics.FP4_BLOCK})


class Nvfp4Kernel:
    """cti-kernels' ``NVFP4Linear``, one per merge slot (each takes one global scale), their
    outputs concatenated; it picks its implementation per device, weight shape and row
    count."""

    def __init__(
        self, weights: Sequence[NVFP4Weight], input_scale: torch.Tensor, *, dtype: torch.dtype
    ) -> None:
        self._operations = [NVFP4Linear(weight, input_scale, out_dtype=dtype) for weight in weights]

    def warm_up(self, max_rows: int, *, out_row_stride: int) -> GemmRoutes:
        """Run each slot eagerly at every row count up to ``max_rows`` its ``prewarm``
        distinguishes, so that no forward or graph capture compiles."""

        for operation in self._operations:
            operation.prewarm(max_rows)
        return Counter({((1, max_rows), f"nvfp4 {op.provider}"): 1 for op in self._operations})

    def workspace_bytes(self) -> int:
        return 0

    def __call__(
        self,
        x: torch.Tensor,
        out: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``out``, a strided view, takes a copy: ``NVFP4Linear`` writes contiguous rows."""

        x_2d = x.reshape(-1, x.shape[-1])
        outputs = [operation(x_2d) for operation in self._operations]
        y = outputs[0] if len(outputs) == 1 else torch.cat(outputs, -1)
        return out.copy_(y) if out is not None else y.view(*x.shape[:-1], -1)


class Nvfp4Reference:
    """The contract computed by torch, for the CPU and as the tests' oracle:
    the activation quantized and dequantized by the input scale, each slot's weight
    dequantized by its block and global scales, both in the activation dtype, multiplied with
    FP32 accumulation."""

    def __init__(
        self, weights: Sequence[NVFP4Weight], input_scale: torch.Tensor, *, dtype: torch.dtype
    ) -> None:
        self._weights, self._input_scale = weights, input_scale

    def warm_up(self, max_rows: int, *, out_row_stride: int) -> GemmRoutes:
        return Counter()

    def dequantized_activation(self, x: torch.Tensor) -> torch.Tensor:
        codes, scale = numerics.quantize_nvfp4_activation(x, self._input_scale)
        return numerics.dequantize_nvfp4_activation(codes, scale, self._input_scale)

    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantized_activation(x).to(x.dtype)

    def dequantized(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.cat(
            [numerics.dequantize_nvfp4_weight(*weight) for weight in self._weights]
        ).to(dtype)

    def workspace_bytes(self) -> int:
        return 0

    def __call__(
        self,
        x: torch.Tensor,
        out: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = F.linear(self.fake_quantize(x), self.dequantized(x.dtype))
        return y if out is None else out.copy_(y)


# Each device type's provider of the NVFP4 product.
PROVIDERS: Mapping[str, type[Nvfp4Kernel | Nvfp4Reference]] = MappingProxyType(
    {
        "cpu": Nvfp4Reference,
        "cuda": Nvfp4Kernel,
    }
)


class Nvfp4Linear(QuantizedMethod):
    """A ModelOpt NVFP4 weight: the packed rows of every merged module with their block
    scales, and each module's global and input scale by merge slot. Every slot's activation
    quantizes by the largest input scale; each slot's output takes its own global scale."""

    providers = PROVIDERS

    def declare(self) -> dict[str, torch.Tensor]:
        layer = self._layer
        rows, cols = layer.local_output_size, layer.local_input_size
        if cols % numerics.FP4_BLOCK:
            raise ValueError(
                f"an NVFP4 weight's {cols} local columns cut its {numerics.FP4_BLOCK}-column blocks"
            )
        slots = len(layer.output_slices)
        return {
            "weight": torch.empty(rows, cols // _PACKED, dtype=torch.uint8),
            "weight_scale": torch.empty(rows, cols // numerics.FP4_BLOCK, dtype=numerics.E4M3),
            "weight_scale_2": torch.empty(slots, dtype=torch.float32),
            "input_scale": torch.empty(slots, dtype=torch.float32),
        }

    def windows(self, leaf: str, windows: tuple[WeightShard, ...]) -> tuple[WeightShard, ...]:
        """Column windows of the packed weight and the block scales, in their stored columns;
        the per-slot scalars load whole."""

        block = _COLUMN_BLOCKS.get(leaf)
        return () if block is None else block_windows(windows, {1: block})

    def bind(self) -> None:
        """Merge the input scales into one ``[1]``."""

        self._layer.input_scale = merged_input_scale(self._layer.input_scale)
        super().bind()

    def operands(self) -> tuple[list[NVFP4Weight], torch.Tensor]:
        """Each merge slot's weight as the kernels take it, and the input scale."""

        layer, weights, row = self._layer, [], 0
        for slot, piece in enumerate(layer.output_slices):
            rows = slice(row, row + piece.size)
            global_scale = layer.weight_scale_2[slot : slot + 1]
            weights.append(NVFP4Weight(layer.weight[rows], layer.weight_scale[rows], global_scale))
            row += piece.size
        return weights, layer.input_scale
