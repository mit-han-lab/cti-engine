# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/layers/base.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

from .placement import Param

if TYPE_CHECKING:
    from .linear_method import GemmCallSite

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


class BaseOP:
    """A node of the model tree. Public tensor attributes are its parameters and public op
    attributes its children; one walk over them serves ``state_dict``, ``load_state_dict``,
    ``params`` and ``named_ops``. ``lora_target`` names the adapter target that may attach
    to the op (``attach_lora``); ``None``: none may.
    """

    lora_target: str | None = None

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    @staticmethod
    def concat_prefix(prefix: str, name: str) -> str:
        """``name`` under the state-dict ``prefix``."""

        return f"{prefix}.{name}" if prefix else name

    def _tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name, value in self.__dict__.items():
            if not name.startswith("_") and isinstance(value, torch.Tensor):
                yield name, value

    def _children(self) -> Iterator[tuple[str, BaseOP]]:
        for name, value in self.__dict__.items():
            if not name.startswith("_") and isinstance(value, BaseOP):
                yield name, value

    def _param(self, path: str, name: str, tensor: torch.Tensor) -> Param:
        """The ``Param`` of tensor ``name``; a layer that shards it adds its windows."""

        return Param(path, tuple(tensor.shape), tensor.dtype)

    def named_ops(self, *, prefix: str = "") -> Iterator[tuple[str, BaseOP]]:
        """This op and every op under it, by state-dict path."""

        yield prefix, self
        for name, child in self._children():
            yield from child.named_ops(prefix=self.concat_prefix(prefix, name))

    def inference_gemms(self) -> Iterator[GemmCallSite]:
        """The inference GEMM call sites of the tree ``state_dict`` walks (an op that holds
        one yields it too); a call site reachable twice is yielded twice."""
        for _, child in self._children():
            yield from child.inference_gemms()

    def settle_inference(self) -> None:
        """Fix what this op runs for inference, once its methods and adapter are final
        (``prepare_inference`` calls it on every op of the tree); nothing by default."""

    def warm_up_rows(
        self, max_rows: int, *, dtype: torch.dtype, device: torch.device, warmed: set
    ) -> None:
        """Compile what this op and those under it pick by row count, for every forward of
        at most ``max_rows`` rows of ``dtype`` activations; an op adds its key to ``warmed``
        and skips a key already there."""
        for _, child in self._children():
            child.warm_up_rows(max_rows, dtype=dtype, device=device, warmed=warmed)

    def state_dict(self, *, prefix: str = "") -> _STATE_DICT:
        return {
            self.concat_prefix(path, name): tensor
            for path, op in self.named_ops(prefix=prefix)
            for name, tensor in op._tensors()
        }

    def params(self, *, prefix: str = "") -> tuple[Param, ...]:
        """Every tensor this rank stores, with the checkpoint windows it loads."""

        return tuple(
            op._param(self.concat_prefix(path, name), name, tensor)
            for path, op in self.named_ops(prefix=prefix)
            for name, tensor in op._tensors()
        )

    def load_state_dict(self, state_dict: _STATE_DICT, *, prefix: str = "") -> None:
        """Replace every parameter with ``state_dict``'s tensor of its path, shape and dtype;
        a key that no parameter takes raises."""

        state_dict = dict(state_dict)
        for path, op in list(self.named_ops(prefix=prefix)):
            for name, param in list(op._tensors()):
                item = state_dict.pop(self.concat_prefix(path, name))
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(op, name, item)
        if state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    """An op without parameters or children."""

    def _tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(())

    def _children(self) -> Iterator[tuple[str, BaseOP]]:
        return iter(())


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    """Ops at the paths ``0``, ``1``, ... under the list's own."""

    def __init__(self, ops: List[T]):
        self.op_list = ops

    def _tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(())

    def _children(self) -> Iterator[tuple[str, BaseOP]]:
        return ((str(index), op) for index, op in enumerate(self.op_list))
