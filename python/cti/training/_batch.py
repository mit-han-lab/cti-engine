"""A training batch: its tokens in one layout, the target positions its loss reads, and its
objective. The layout and the objective are independent axes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, TypeAlias

import torch

from cti.config import BatchLayout
from cti.models.training_contract import Packing

if TYPE_CHECKING:
    from cti.protocol import SamplingResult

    from ._objective import GRPO, SFT


@dataclass(frozen=True, slots=True)
class SFTDatum:
    """One immutable supervised sequence in causal-target coordinates."""

    token_ids: tuple[int, ...]
    target_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if type(self.token_ids) is not tuple or type(self.target_mask) is not tuple:
            raise TypeError("token_ids and target_mask must be tuples")
        if len(self.token_ids) < 2:
            raise ValueError("token_ids must contain at least two tokens")
        if len(self.target_mask) != len(self.token_ids):
            raise ValueError("token_ids and target_mask must have the same length")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.token_ids):
            raise ValueError("token_ids must be non-negative integers")
        if any(type(selected) is not bool for selected in self.target_mask):
            raise TypeError("target_mask must contain booleans")
        if self.target_mask[0]:
            raise ValueError("the first token cannot be a causal target")


@dataclass(frozen=True, slots=True)
class DeviceRows:
    """One microbatch on the training device: model inputs, the flat input positions whose
    hidden states the loss reads (None: every position) and their target tokens."""

    input_ids: torch.Tensor
    packing: Packing | None
    target_indices: torch.Tensor | None
    target_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class PaddedTokens:
    """Right-padded CPU rows. Target coordinates are ``[rows, width - 1]``: position ``p``
    of a row predicts its token ``p + 1``."""

    LAYOUT: ClassVar[BatchLayout] = BatchLayout.PADDED

    token_ids: torch.Tensor
    lengths: torch.Tensor
    pad_token_id: int

    @classmethod
    def of(
        cls, sequences: Sequence[Sequence[int]], *, pad_token_id: int, padded_length: int | None
    ) -> PaddedTokens:
        width = max(map(len, sequences))
        if padded_length is not None:
            if type(padded_length) is not int or padded_length < width:
                raise ValueError("padded_length must be an integer that truncates no sequence")
            width = padded_length
        token_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.int64)
        for row, sequence in enumerate(sequences):
            token_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.int64)
        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.int64)
        return cls(token_ids, lengths, pad_token_id)

    def __post_init__(self) -> None:
        _require_cpu(token_ids=self.token_ids, lengths=self.lengths)
        if self.token_ids.ndim != 2 or self.token_ids.dtype != torch.int64:
            raise TypeError("token_ids must be an int64 matrix")
        rows, width = self.token_ids.shape
        if rows == 0 or width < 2:
            raise ValueError("token_ids must hold at least one row of two columns")
        if self.lengths.shape != (rows,) or self.lengths.dtype != torch.int64:
            raise TypeError("lengths must be an int64 vector with one length per row")
        if type(self.pad_token_id) is not int or self.pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer")
        if bool(torch.any(self.lengths < 2)) or bool(torch.any(self.lengths > width)):
            raise ValueError("every row must hold between two and width tokens")
        padding = torch.arange(width).unsqueeze(0) >= self.lengths.unsqueeze(1)
        if bool(torch.any(self.token_ids[padding] != self.pad_token_id)):
            raise ValueError("token_ids must be strictly right-padded with pad_token_id")

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @property
    def width(self) -> int:
        """Input columns of every row."""

        return self.token_ids.shape[1] - 1

    @property
    def target_shape(self) -> tuple[int, ...]:
        return (self.batch_size, self.width)

    def sequence_lengths(self) -> list[int]:
        return self.lengths.tolist()

    def valid_targets(self) -> torch.Tensor:
        return torch.arange(self.width).unsqueeze(0) < (self.lengths - 1).unsqueeze(1)

    def targets_of(self, rows: Sequence[Sequence[bool]]) -> torch.Tensor:
        targets = torch.zeros(self.target_shape, dtype=torch.bool)
        for row, selected in enumerate(rows):
            targets[row, : len(selected)] = torch.tensor(selected, dtype=torch.bool)
        return targets

    def row_target_counts(self, targets: torch.Tensor) -> torch.Tensor:
        return targets.sum(dim=1)

    def row_first_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Each row's first target position."""

        return targets.int().argmax(dim=1)

    def row_tokens(self, row: int) -> torch.Tensor:
        return self.token_ids[row, : int(self.lengths[row])]

    def check_width(self, groups: tuple[tuple[int, ...], ...], max_tokens: int) -> None:
        # Padding past the longest sample widens every row.
        if max(len(group) for group in groups) * self.width > max_tokens:
            raise ValueError(
                f"padded microbatches of {self.width}-token rows exceed the "
                f"{max_tokens}-token microbatch bound"
            )

    def target_token_ids(self) -> torch.Tensor:
        # Padding columns hold the pad token.
        token_ids = self.token_ids.reshape(-1)
        return torch.cat((token_ids, token_ids.new_tensor([self.pad_token_id])))

    def select(
        self, indices: tuple[int, ...], targets: torch.Tensor
    ) -> tuple[torch.Tensor, Packing | None, torch.Tensor, torch.Tensor]:
        """The rows at ``indices``: their input ids, packing, target tokens at every
        position and targets."""

        token_ids = self.token_ids
        if indices != tuple(range(self.batch_size)):
            rows = torch.tensor(indices, dtype=torch.int64)
            token_ids, targets = token_ids.index_select(0, rows), targets.index_select(0, rows)
        return token_ids[:, :-1], None, token_ids[:, 1:].reshape(-1), targets


@dataclass(frozen=True, slots=True)
class PackedTokens:
    """Sequences back to back, bounded by ``cu_seqlens``. Target coordinates are each
    sequence's input positions, its tokens but the last, back to back."""

    LAYOUT: ClassVar[BatchLayout] = BatchLayout.PACKED

    token_ids: torch.Tensor
    cu_seqlens: torch.Tensor

    @classmethod
    def of(cls, sequences: Sequence[Sequence[int]]) -> PackedTokens:
        boundaries = [0]
        for sequence in sequences:
            boundaries.append(boundaries[-1] + len(sequence))
        if boundaries[-1] > torch.iinfo(torch.int32).max:
            raise ValueError("packed token count exceeds int32 boundary capacity")
        return cls(
            torch.tensor(
                [token for sequence in sequences for token in sequence], dtype=torch.int64
            ),
            torch.tensor(boundaries, dtype=torch.int32),
        )

    def __post_init__(self) -> None:
        _require_cpu(token_ids=self.token_ids, cu_seqlens=self.cu_seqlens)
        if self.token_ids.ndim != 1 or self.token_ids.dtype != torch.int64:
            raise TypeError("token_ids must be an int64 vector")
        if self.cu_seqlens.ndim != 1 or self.cu_seqlens.dtype != torch.int32:
            raise TypeError("cu_seqlens must be an int32 vector")
        if self.cu_seqlens.numel() < 2 or self.cu_seqlens[0].item() != 0:
            raise ValueError("cu_seqlens must start at zero and describe a sequence")
        if self.cu_seqlens[-1].item() != self.token_ids.numel():
            raise ValueError("cu_seqlens must end at the token count")
        if bool(torch.any(torch.diff(self.cu_seqlens) < 2)):
            raise ValueError("every sequence must contain at least two tokens")

    @property
    def batch_size(self) -> int:
        return self.cu_seqlens.numel() - 1

    @property
    def target_shape(self) -> tuple[int, ...]:
        return (self.token_ids.numel() - self.batch_size,)

    @property
    def input_cu_seqlens(self) -> torch.Tensor:
        return self.cu_seqlens - torch.arange(self.batch_size + 1, dtype=torch.int32)

    def sequence_lengths(self) -> list[int]:
        return torch.diff(self.cu_seqlens).tolist()

    def valid_targets(self) -> torch.Tensor:
        return torch.ones(self.target_shape, dtype=torch.bool)

    def targets_of(self, rows: Sequence[Sequence[bool]]) -> torch.Tensor:
        return torch.tensor([selected for row in rows for selected in row], dtype=torch.bool)

    def row_target_counts(self, targets: torch.Tensor) -> torch.Tensor:
        rows = torch.repeat_interleave(
            torch.arange(self.batch_size), torch.diff(self.input_cu_seqlens).to(torch.int64)
        )
        return torch.zeros(self.batch_size, dtype=torch.int64).index_add_(0, rows, targets.long())

    def row_first_targets(self, targets: torch.Tensor) -> torch.Tensor:
        boundaries = self.input_cu_seqlens.tolist()
        return torch.tensor(
            [
                int(targets[start:stop].int().argmax())
                for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
            ],
            dtype=torch.int64,
        )

    def row_tokens(self, row: int) -> torch.Tensor:
        return self.token_ids[int(self.cu_seqlens[row]) : int(self.cu_seqlens[row + 1])]

    def check_width(self, groups: tuple[tuple[int, ...], ...], max_tokens: int) -> None:
        """Packed microbatches stay within the batching policy's bound."""

    def target_token_ids(self) -> torch.Tensor:
        return self.token_ids

    def select(
        self, indices: tuple[int, ...], targets: torch.Tensor
    ) -> tuple[torch.Tensor, Packing | None, torch.Tensor, torch.Tensor]:
        """The contiguous sequences at ``indices``: their input ids, packing, target tokens
        at every input position and targets."""

        first, stop = indices[0], indices[-1] + 1
        if indices != tuple(range(first, stop)):
            raise ValueError("packed microbatch indices must describe one contiguous range")
        token_start, token_stop = int(self.cu_seqlens[first]), int(self.cu_seqlens[stop])
        token_ids = self.token_ids[token_start:token_stop]
        cu_seqlens = self.cu_seqlens[first : stop + 1] - token_start
        lengths = torch.diff(cu_seqlens)
        input_cu_seqlens = cu_seqlens - torch.arange(stop - first + 1, dtype=torch.int32)
        input_start = int(self.input_cu_seqlens[first])
        keep_input = torch.ones(token_ids.numel(), dtype=torch.bool)
        keep_input[cu_seqlens[1:].to(torch.int64) - 1] = False
        keep_target = torch.ones(token_ids.numel(), dtype=torch.bool)
        keep_target[cu_seqlens[:-1].to(torch.int64)] = False
        packing = Packing(
            positions=torch.cat([torch.arange(length - 1) for length in lengths.tolist()]),
            cu_seqlens=input_cu_seqlens,
            max_sequence_length=int(lengths.max()) - 1,
        )
        return (
            token_ids[keep_input],
            packing,
            token_ids[keep_target],
            targets[input_start : input_start + int(input_cu_seqlens[-1])],
        )


TokenLayout: TypeAlias = PaddedTokens | PackedTokens


@dataclass(frozen=True)
class TrainingBatch:
    """Borrowed CPU tokens in one layout, the ``targets`` its loss reads, a bool tensor in
    the layout's target coordinates, and the objective that trains on them."""

    tokens: TokenLayout
    targets: torch.Tensor
    objective: SFT | GRPO
    _row_target_counts: torch.Tensor = field(init=False, repr=False, compare=False)

    @classmethod
    def sft(
        cls,
        data: Sequence[SFTDatum],
        layout: BatchLayout,
        *,
        pad_token_id: int | None = None,
        padded_length: int | None = None,
    ) -> TrainingBatch:
        """Ordered supervised ``data`` in ``layout``; padded rows take ``pad_token_id``."""

        from ._objective import SFT

        data = tuple(data)
        if not data or any(not isinstance(datum, SFTDatum) for datum in data):
            raise TypeError("data must be a non-empty sequence of SFTDatum values")
        tokens = _layout_of(
            [datum.token_ids for datum in data], layout, pad_token_id, padded_length
        )
        return cls(tokens, tokens.targets_of([datum.target_mask[1:] for datum in data]), SFT())

    @classmethod
    def grpo(
        cls,
        sampling: SamplingResult,
        rewards: Sequence[float],
        layout: BatchLayout,
        *,
        group_size: int,
        pad_token_id: int | None = None,
        padded_length: int | None = None,
    ) -> TrainingBatch:
        """One grouped rollout in ``layout``: each sample's response tokens are its targets."""

        from ._objective import GRPO

        objective, prompts, responses = GRPO.of(sampling, rewards, group_size=group_size)
        tokens = _layout_of(
            [prompt + response for prompt, response in zip(prompts, responses, strict=True)],
            layout,
            pad_token_id,
            padded_length,
        )
        targets = tokens.targets_of(
            [
                (False,) * (len(prompt) - 1) + (True,) * len(response)
                for prompt, response in zip(prompts, responses, strict=True)
            ]
        )
        return cls(tokens, targets, objective)

    def __post_init__(self) -> None:
        from ._objective import GRPO, SFT

        if not isinstance(self.tokens, PaddedTokens | PackedTokens):
            raise TypeError("tokens must be PaddedTokens or PackedTokens")
        if not isinstance(self.objective, SFT | GRPO):
            raise TypeError("objective must be SFT or GRPO")
        _require_cpu(targets=self.targets)
        if self.targets.dtype != torch.bool or tuple(self.targets.shape) != (
            self.tokens.target_shape
        ):
            raise TypeError("targets must be a bool tensor in the layout's target coordinates")
        if bool(torch.any(self.targets & ~self.tokens.valid_targets())):
            raise ValueError("targets cannot select padding positions")
        counts = self.tokens.row_target_counts(self.targets)
        if bool(torch.any(counts == 0)):
            raise ValueError("every sequence must select at least one target")
        object.__setattr__(self, "_row_target_counts", counts)
        self.objective.validate(self)

    @property
    def batch_size(self) -> int:
        return self.tokens.batch_size

    @property
    def num_targets(self) -> int:
        return int(self._row_target_counts.sum())

    @property
    def row_target_counts(self) -> torch.Tensor:
        return self._row_target_counts

    def microbatch(self, indices: tuple[int, ...], device: torch.device) -> DeviceRows:
        """The rows at ``indices`` on ``device``."""

        input_ids, packing, target_ids, targets = self.tokens.select(indices, self.targets)
        targets = targets.reshape(-1)
        target_indices = None if bool(targets.all()) else targets.nonzero().flatten()
        if target_indices is not None:
            target_ids = target_ids.index_select(0, target_indices)
        return DeviceRows(
            input_ids.to(device),
            None if packing is None else packing.to(device),
            None if target_indices is None else target_indices.to(device),
            target_ids.to(device),
        )


def _layout_of(
    sequences: list[Sequence[int]],
    layout: BatchLayout,
    pad_token_id: int | None,
    padded_length: int | None,
) -> TokenLayout:
    if layout is BatchLayout.PACKED:
        if pad_token_id is not None or padded_length is not None:
            raise ValueError("packed tokens take no padding")
        return PackedTokens.of(sequences)
    if pad_token_id is None:
        raise ValueError("padded tokens require pad_token_id")
    return PaddedTokens.of(sequences, pad_token_id=pad_token_id, padded_length=padded_length)


def _require_cpu(**tensors: object) -> None:
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must reside on CPU")


__all__ = [
    "DeviceRows",
    "PackedTokens",
    "PaddedTokens",
    "SFTDatum",
    "TokenLayout",
    "TrainingBatch",
]
