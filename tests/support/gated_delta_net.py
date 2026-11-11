# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`); see THIRD_PARTY_NOTICES.

"""Torch reference Gated DeltaNet operations for CPU tests, and small layers bound to them.

They are constructed like ``ctikernels.gdn``'s operations (``num_k_heads``,
``num_v_heads``, ``device``) and take the same decode and prefill arguments (merged
``projected`` rows ``q | k | v | z | b | a``, frozen weights, slot-indexed FP32 state
and BF16 convolution-history pools updated in place, slot -1 for padding or a
stateless sequence), so engine tests bind them through the same seam as the kernels.
Each token runs through one FP32 step shared by decode and prefill, so any chunking
of a sequence gives bit-identical results; ``y`` is rounded to BF16 once. The training
reference (``GatedDeltaNetTraining``'s contract) runs the same step from zero state with
autograd, so a gradient flows to ``projected`` and never to a weight.

The math follows the Qwen3.5 Gated DeltaNet of Transformers 5.16.1
(``models/qwen3_5/modeling_qwen3_5.py``: causal depthwise convolution with SiLU,
``l2norm``, ``torch_recurrent_gated_delta_rule``, ``Qwen3_5RMSNormGated``) and the
pool semantics of the gdn kernel line's contract reference
(``ctikernels/gdn/contract_reference.py``); it is rewritten here, not copied.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn.functional as F

from cti.forward import ForwardInputs, ForwardRows
from cti.kvcache import RecurrentStatePool, SlotStateSpec
from cti.layers import LayerServices
from cti.layers.gated_delta_net import GatedDeltaNet
from cti.layers.gated_delta_net.operations import GatedDeltaNetOps
from cti.recurrent import RecurrentBackend
from tests.support.forward_rows import Entry, forward_rows

HEAD_DIM = 128
EPS = 1e-6
F32 = torch.float32


def _require_frozen(*weights: torch.Tensor) -> None:
    if any(weight.requires_grad for weight in weights):
        raise ValueError("Gated DeltaNet base weights are frozen: no weight may require grad")


def _require_no_autograd(projected: torch.Tensor) -> None:
    """Refuse autograd as ``ctikernels.gdn`` does; under ``torch.inference_mode()`` grad mode
    is off, so LoRA factors that require grad pass."""

    if torch.is_grad_enabled() and projected.requires_grad:
        raise RuntimeError("Gated DeltaNet inference operations do not support autograd")


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + EPS)


def _sequence(
    num_key_heads: int,
    rows: torch.Tensor,
    history: torch.Tensor,
    state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``rows [L, P]`` after ``history [taps - 1, Dc]`` from ``state [Hv, K, V]`` (FP32).

    Returns FP32 ``y [L, Hv * V]``, the final state and the last ``taps - 1``
    pre-convolution rows.
    """

    num_value_heads = A_log.shape[0]
    conv_channels, taps = conv_weight.shape
    weight = conv_weight.to(F32).t()  # [taps, Dc]
    decay_scale = -torch.exp(A_log.to(F32))
    group = num_value_heads // num_key_heads
    key_width = num_key_heads * HEAD_DIM
    value_width = num_value_heads * HEAD_DIM
    window = torch.cat((history.to(F32), rows[:, :conv_channels].to(F32)))
    outputs = []
    for token in range(rows.shape[0]):
        mixed = F.silu((window[token : token + taps] * weight).sum(0))
        q, k, v = mixed.split((key_width, key_width, value_width))
        q = _l2norm(q.view(num_key_heads, HEAD_DIM)).repeat_interleave(group, 0)
        q = q * HEAD_DIM**-0.5
        k = _l2norm(k.view(num_key_heads, HEAD_DIM)).repeat_interleave(group, 0)
        v = v.view(num_value_heads, HEAD_DIM)
        z = rows[token, conv_channels : conv_channels + value_width].to(F32)
        b = rows[token, conv_channels + value_width :][:num_value_heads].to(F32)
        a = rows[token, conv_channels + value_width :][num_value_heads:].to(F32)
        decay = torch.exp(decay_scale * F.softplus(a + dt_bias.to(F32)))
        beta = torch.sigmoid(b)

        state = state * decay[:, None, None]
        delta = (v - (state * k[:, :, None]).sum(1)) * beta[:, None]
        state = state + k[:, :, None] * delta[:, None, :]
        o = (state * q[:, :, None]).sum(1)  # [Hv, V]
        normed = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
        gated = norm_weight.to(F32) * normed * F.silu(z.view(num_value_heads, HEAD_DIM))
        outputs.append(gated.reshape(-1))
    y = torch.stack(outputs) if outputs else rows.new_zeros(0, value_width, dtype=F32)
    return y, state, window[window.shape[0] - (taps - 1) :]


def _read_state(pool_row: torch.Tensor) -> torch.Tensor:
    """A pool row ``[Hv, V, K]`` (value-major) as a contiguous ``[Hv, K, V]``, so every
    reduction runs in one order whether the state came from the pool or from zero."""

    return pool_row.transpose(-1, -2).to(F32).contiguous()


def _write_slot(
    state: torch.Tensor,
    conv_state: torch.Tensor,
    slot: int,
    final_state: torch.Tensor,
    history: torch.Tensor,
) -> None:
    state[slot] = final_state.transpose(-1, -2)
    conv_state[slot] = history.to(conv_state.dtype)


class _ReferenceOperation:
    """The construction of a ``ctikernels.gdn`` operation: one head geometry, one device."""

    provider = "reference"

    def __init__(self, *, num_k_heads: int, num_v_heads: int, device: torch.device | str) -> None:
        if num_k_heads < 1 or num_v_heads % num_k_heads:
            raise ValueError("num_v_heads must be a positive multiple of num_k_heads")
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.device = torch.device(device)

    def _check(self, conv_weight: torch.Tensor, A_log: torch.Tensor) -> None:
        conv_channels = (2 * self.num_k_heads + self.num_v_heads) * HEAD_DIM
        if conv_weight.shape[0] != conv_channels or A_log.shape != (self.num_v_heads,):
            raise ValueError(
                f"weights do not match the operation's ({self.num_k_heads}, "
                f"{self.num_v_heads}) heads"
            )


class ReferenceGatedDeltaNetDecode(_ReferenceOperation):
    """``GatedDeltaNetDecode``'s contract: one token per row; slot -1 rows give zero; ``y``
    goes into ``out`` when given. The reference keeps no scratch, so ``reset_scratch`` has
    nothing to reset."""

    def __init__(
        self,
        *,
        num_k_heads: int,
        num_v_heads: int,
        device: torch.device | str,
        batch_invariant: bool = False,
    ) -> None:
        super().__init__(num_k_heads=num_k_heads, num_v_heads=num_v_heads, device=device)
        self.batch_invariant = batch_invariant

    def reset_scratch(self, pool: torch.Tensor | None = None) -> None:
        pass

    def __call__(
        self,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        slot_indices: torch.Tensor,
        state: torch.Tensor,
        conv_state: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require_frozen(conv_weight, A_log, dt_bias, norm_weight)
        _require_no_autograd(projected)
        self._check(conv_weight, A_log)
        y = projected.new_zeros(projected.shape[0], A_log.shape[0] * HEAD_DIM)
        for row, slot in enumerate(slot_indices.tolist()):
            if slot < 0:
                continue
            row_y, final_state, history = _sequence(
                self.num_k_heads,
                projected[row : row + 1],
                conv_state[slot],
                _read_state(state[slot]),
                conv_weight,
                A_log,
                dt_bias,
                norm_weight,
            )
            y[row] = row_y[0].to(y.dtype)
            _write_slot(state, conv_state, slot, final_state, history)
        if out is None:
            return y
        out.copy_(y)
        return out


class ReferenceGatedDeltaNetPrefill(_ReferenceOperation):
    """``GatedDeltaNetPrefill``'s contract on its CuTe provider: varlen sequences from zero
    or their slot; ``y`` goes into ``out`` when given, and sequence n's state after its most
    whole multiples of ``CAPTURE_GRID`` tokens, when that is at least one, into pool row
    ``capture_rows[n]`` unless it is -1."""

    provider = "cute"  # the contract it follows, which exports captured states
    supports_state_capture = True
    CAPTURE_GRID = 64

    def __init__(self, *, num_k_heads: int, num_v_heads: int, device: torch.device | str) -> None:
        super().__init__(num_k_heads=num_k_heads, num_v_heads=num_v_heads, device=device)
        # Each warm_up call's (max_tokens, max_sequences).
        self.warm_ups: list[tuple[int, int | None]] = []

    def warm_up(self, max_tokens: int, *, max_sequences: int | None = None) -> int:
        self.warm_ups.append((max_tokens, max_sequences))
        return 0

    def __call__(
        self,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        cu_seqlens: torch.Tensor,
        slot_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        state: torch.Tensor,
        conv_state: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        capture_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require_frozen(conv_weight, A_log, dt_bias, norm_weight)
        _require_no_autograd(projected)
        self._check(conv_weight, A_log)
        if capture_rows is not None and (
            capture_rows.dtype != torch.int32 or capture_rows.shape != slot_indices.shape
        ):
            raise ValueError("capture_rows must be int32 [N], one row per sequence")
        num_value_heads = A_log.shape[0]
        taps = conv_weight.shape[1]
        shape = (projected.shape[0], num_value_heads * HEAD_DIM)
        if out is not None and (
            out.shape != shape or out.dtype != projected.dtype or not out.is_contiguous()
        ):
            raise ValueError("out must be a contiguous BF16 [tokens, Hv * V] tensor")
        y = projected.new_zeros(shape) if out is None else out
        bounds = cu_seqlens.tolist()
        targets = [-1] * len(slot_indices) if capture_rows is None else capture_rows.tolist()
        for sequence, (slot, continued, target) in enumerate(
            zip(slot_indices.tolist(), has_initial_state.tolist(), targets, strict=True)
        ):
            start, end = bounds[sequence], bounds[sequence + 1]
            if slot >= 0 and continued:
                current_state = _read_state(state[slot])
                history = conv_state[slot]
            else:
                current_state = torch.zeros(
                    num_value_heads, HEAD_DIM, HEAD_DIM, dtype=F32, device=projected.device
                )
                history = torch.zeros(
                    taps - 1, conv_weight.shape[0], dtype=F32, device=projected.device
                )
            # Run up to the capture, then the rest; every chunking gives the same bits.
            point = (end - start) // self.CAPTURE_GRID * self.CAPTURE_GRID
            cuts = [(start + point, target)] if target >= 0 and point else []
            cursor = start
            for stop, row in [*cuts, (end, slot)]:
                rows, current_state, history = _sequence(
                    self.num_k_heads,
                    projected[cursor:stop],
                    history,
                    current_state,
                    conv_weight,
                    A_log,
                    dt_bias,
                    norm_weight,
                )
                y[cursor:stop] = rows.to(y.dtype)
                if row >= 0:
                    _write_slot(state, conv_state, row, current_state, history)
                cursor = stop
        return y


class ReferenceGatedDeltaNetTraining(_ReferenceOperation):
    """``GatedDeltaNetTraining``'s contract: packed sequences, each from zero state and
    history and keeping no final state, differentiable in ``projected`` only.

    A weight that requires grad raises before anything else is checked.
    ``max_sequence_length`` is a host integer from 1 to the token count, below 2^31 rows of
    ``projected``'s stride; unlike the kernel, which never reads ``cu_seqlens`` on the
    host, the reference also refuses a sequence longer than it.
    """

    def training_bytes(self, tokens: int, sequences: int) -> tuple[int, int]:
        """The ``(saved, backward)`` cti-kernels' Triton provider reports, the one the
        training-memory accuracy table's RTX PRO 6000 resolves (``gdn/triton/training.py``;
        64-token chunks, ``dqkw`` key blocks of 32)."""

        hk, hv = self.num_k_heads, self.num_v_heads
        keys, values = hk * HEAD_DIM, hv * HEAD_DIM
        projected = 2 * keys + 2 * values + 2 * hv
        chunks = (tokens + 63 * sequences) // 64
        state = 2 * hv * HEAD_DIM * HEAD_DIM
        row = 2 * projected + 4 * keys + 8 * values + 8 * hk + (12 + 2 * 64) * hv + 4
        saved = tokens * row + chunks * (state + 8) + 8 * (sequences + 1)
        row = 2 * projected + 6 * values + 8 * keys + 4 * hv * (HEAD_DIM // 32)
        return saved, tokens * row + max(chunks * state, 4 * values * tokens)

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
    ) -> torch.Tensor:
        _require_frozen(conv_weight, A_log, dt_bias, norm_weight)
        self._check(conv_weight, A_log)
        bounds = cu_seqlens.tolist()
        if isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int):
            raise TypeError("max_sequence_length must be an integer")
        if not 1 <= max_sequence_length <= projected.shape[0]:
            raise ValueError("max_sequence_length must be within the packed token extent")
        if max_sequence_length * projected.stride(0) >= 2**31:
            raise ValueError("a sequence reaches 2^31 elements of projected")
        if max(end - start for start, end in pairwise(bounds)) > max_sequence_length:
            raise ValueError("a sequence exceeds max_sequence_length")
        num_value_heads = A_log.shape[0]
        taps = conv_weight.shape[1]
        state = torch.zeros(num_value_heads, HEAD_DIM, HEAD_DIM, dtype=F32, device=projected.device)
        history = torch.zeros(taps - 1, conv_weight.shape[0], dtype=F32, device=projected.device)
        outputs = [
            _sequence(
                self.num_k_heads,
                projected[start:end],
                history,
                state,
                conv_weight,
                A_log,
                dt_bias,
                norm_weight,
            )[0]
            for start, end in pairwise(bounds)
        ]
        return torch.cat(outputs).to(projected.dtype)


def reference_kernel_module(*, training: bool = True) -> types.ModuleType:
    """A stand-in ``ctikernels.gdn`` that exports the reference operations under the
    kernels' names, so tests bind them through the runner's resolution functions.

    ``training=False`` stands in for a release without the training operation.
    """

    module = types.ModuleType("ctikernels.gdn")
    module.GatedDeltaNetDecode = ReferenceGatedDeltaNetDecode  # type: ignore[attr-defined]
    module.GatedDeltaNetPrefill = ReferenceGatedDeltaNetPrefill  # type: ignore[attr-defined]
    if training:
        module.GatedDeltaNetTraining = ReferenceGatedDeltaNetTraining  # type: ignore[attr-defined]
    return module


# ============================ Layers over a pool ============================
# A small layer, bound to the reference operations over a pool of NUM_SLOTS slots, for the
# layer tests and the decode-graph tests of its backend.

NUM_KEY_HEADS = 1
NUM_VALUE_HEADS = 2
CONV_CHANNELS = 2 * NUM_KEY_HEADS * HEAD_DIM + NUM_VALUE_HEADS * HEAD_DIM
PROJECTED_WIDTH = CONV_CHANNELS + NUM_VALUE_HEADS * HEAD_DIM + 2 * NUM_VALUE_HEADS
OUTPUT_WIDTH = NUM_VALUE_HEADS * HEAD_DIM
NUM_SLOTS = 4


def gdn_layer(
    *,
    seed: int = 0,
    heads: tuple[int, int] = (NUM_KEY_HEADS, NUM_VALUE_HEADS),
    training: bool = False,
) -> GatedDeltaNet:
    """A small layer with seeded checkpoint weights over recording reference operations;
    ``layer.test_services`` keeps its services (``RecordingServices``)."""

    from tests.support.tp_shards import site

    num_key_heads, num_value_heads = heads
    services = RecordingServices()
    layer = GatedDeltaNet(
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        conv_kernel_size=4,
        site=site(services=services, training=training),
    )
    layer.test_services = services
    conv_channels = layer.conv_weight.shape[0]
    generator = torch.Generator().manual_seed(seed)
    layer.load_state_dict(
        {
            "conv_weight": (0.5 * torch.randn(conv_channels, 1, 4, generator=generator)).to(
                torch.bfloat16
            ),
            "A_log": torch.log(torch.rand(num_value_heads, generator=generator) * 15 + 1),
            "dt_bias": torch.randn(num_value_heads, generator=generator),
            "norm_weight": 1 + 0.1 * torch.randn(HEAD_DIM, generator=generator),
        }
    )
    return layer


def projected_rows(num_tokens: int, *, seed: int = 1, width: int = PROJECTED_WIDTH) -> torch.Tensor:
    """Seeded BF16 ``projected`` rows of a layer's input."""

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, width, generator=generator).to(torch.bfloat16)


class RecordingOperation:
    """Wrap an operation and keep the arguments and result of every call."""

    def __init__(self, operation) -> None:
        self._operation = operation
        self.calls: list[tuple[torch.Tensor, ...]] = []
        self.keywords: list[dict[str, object]] = []
        self.results: list[torch.Tensor] = []
        self.resets: list[torch.Tensor | None] = []

    def __call__(self, *arguments: torch.Tensor, **keywords: object) -> torch.Tensor:
        self.calls.append(arguments)
        self.keywords.append(keywords)
        self.results.append(self._operation(*arguments, **keywords))
        return self.results[-1]

    def reset_scratch(self, pool: torch.Tensor | None = None) -> None:
        self.resets.append(pool)
        self._operation.reset_scratch(pool)


class RecordingServices(LayerServices):
    """Services whose Gated DeltaNet operations are the references, each wrapped in a
    ``RecordingOperation``, kept as ``ops``."""

    def gated_delta_net(
        self, device: torch.device, *, num_key_heads: int, num_value_heads: int, training: bool
    ) -> GatedDeltaNetOps:
        heads = {"num_k_heads": num_key_heads, "num_v_heads": num_value_heads, "device": device}
        self.ops = GatedDeltaNetOps(
            RecordingOperation(ReferenceGatedDeltaNetDecode(**heads)),
            RecordingOperation(ReferenceGatedDeltaNetPrefill(**heads)),
            RecordingOperation(ReferenceGatedDeltaNetTraining(**heads)) if training else None,
        )
        return self.ops


@dataclass(frozen=True)
class BoundLayer:
    backend: RecurrentBackend
    pool: RecurrentStatePool
    decode: RecordingOperation
    prefill: RecordingOperation


def bind_reference(layer: GatedDeltaNet, num_snapshot_rows: int = 0) -> BoundLayer:
    """Bind ``layer`` to row 0 of a pool of deterministic garbage rows."""

    pool = RecurrentStatePool(
        (layer.state_spec,),
        num_slots=NUM_SLOTS,
        num_snapshot_rows=num_snapshot_rows,
        device=torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(123)
    state, conv_state = pool.layer_views(0)
    state.normal_(generator=generator)
    conv_state.copy_(torch.randn(conv_state.shape, generator=generator))
    layer.bind_state({"recurrent": pool.layer_state(0)})
    services = layer.test_services
    return BoundLayer(
        RecurrentBackend(pool, device=pool.device), pool, services.ops.decode, services.ops.prefill
    )


def fp32_history(spec: SlotStateSpec) -> SlotStateSpec:
    """``spec`` with an FP32 convolution history, as EagerEngine keeps it."""

    return SlotStateSpec(
        tuple(
            (name, shape, torch.float32 if name == "conv" else dtype)
            for name, shape, dtype in spec.tensors
        ),
        export_granularity=spec.export_granularity,
    )


def layer_rows(*entries: Entry, padding: int = 0) -> ForwardRows:
    """The rows of ``entries`` padded with ``padding`` dummy rows of slot ``NUM_SLOTS``."""

    return forward_rows(*entries, padding=padding, padding_slot=NUM_SLOTS)


def layer_inputs(rows: ForwardRows, metadata: object) -> ForwardInputs:
    """What a Gated DeltaNet layer reads of ``rows``: their recurrent ``metadata``."""

    return ForwardInputs(
        None, None, None, rows.size, rows.uses_ragged_attention, recurrent_metadata=metadata
    )


def run_layer(
    layer: GatedDeltaNet,
    backend: RecurrentBackend,
    projected: torch.Tensor,
    *entries: Entry,
    padding: int = 0,
) -> torch.Tensor:
    """Run ``layer`` over ``entries`` after the backend prepares their metadata."""

    rows = layer_rows(*entries, padding=padding)
    return layer.forward(projected, layer_inputs(rows, backend.prepare_metadata(rows)))


def pool_snapshot(pool: RecurrentStatePool) -> tuple[torch.Tensor, torch.Tensor]:
    state, conv_state = pool.layer_views(0)
    return state.clone(), conv_state.clone()


__all__ = [
    "CONV_CHANNELS",
    "NUM_KEY_HEADS",
    "NUM_SLOTS",
    "NUM_VALUE_HEADS",
    "OUTPUT_WIDTH",
    "PROJECTED_WIDTH",
    "BoundLayer",
    "RecordingOperation",
    "RecordingServices",
    "ReferenceGatedDeltaNetDecode",
    "ReferenceGatedDeltaNetPrefill",
    "ReferenceGatedDeltaNetTraining",
    "bind_reference",
    "gdn_layer",
    "layer_inputs",
    "layer_rows",
    "pool_snapshot",
    "projected_rows",
    "reference_kernel_module",
    "run_layer",
]
