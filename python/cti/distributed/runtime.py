# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/distributed/impl.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from cti.utils import init_logger

from .groups import ResidualReduction, Topology

if TYPE_CHECKING:
    from ctikernels import PyNCCLCommunicator

logger = init_logger(__name__)


class _RankRecords(logging.Filter):
    """Adds this process's rank to every engine log record, at the package logger's handler
    where they all pass, and drops the ``*_rank0`` ones off rank 0."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self._rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self._rank
        return self._rank == 0 or not getattr(record, "rank0", False)

    @classmethod
    @contextmanager
    def installed(cls, rank: int) -> Iterator[None]:
        records = cls(rank)
        handlers = logging.getLogger("cti").handlers
        for handler in handlers:
            handler.addFilter(records)
        try:
            yield
        finally:
            for handler in handlers:
                handler.removeFilter(records)


class _PyNCCL:
    """The ranks' PyNCCL communicator, used for the model's device collectives."""

    def __init__(self, topology: Topology, cpu_group: dist.ProcessGroup) -> None:
        from ctikernels import init_pynccl

        self._size = topology.size
        self._comm: PyNCCLCommunicator = init_pynccl(
            tp_rank=topology.rank,
            tp_size=topology.size,
            tp_cpu_group=cpu_group,
        )

    @staticmethod
    def reduces(dtype: torch.dtype) -> bool:
        """PyNCCL sums 16-bit floats only; ParallelGroup reduces every other dtype over its
        torch.distributed NCCL group until cti-kernels' PyNCCL reduces FP32 and integer tensors."""

        return dtype in (torch.bfloat16, torch.float16)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self._comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        # PyNCCL gathers 16-bit floats only, and a gather moves bytes: any tensor travels as
        # bfloat16 words, copied to aligned storage plus a pad byte only when it needs them.
        data = x.reshape(-1).view(torch.uint8)
        size = data.numel()
        if size % 2 or data.storage_offset() % 2:
            data = torch.cat((data, data.new_zeros(size % 2)))
        words = data.view(torch.bfloat16)
        result = words.new_empty(self._size * words.numel())
        self._comm.all_gather(result, words)
        ranks = result.view(torch.uint8).view(self._size, data.numel())[:, :size]
        return ranks.reshape(-1).view(x.dtype).view(self._size * x.shape[0], *x.shape[1:])


class _Summing:
    """The group's in-place sum, as cti-kernels' collectives call a communicator."""

    def __init__(self, group: ParallelGroup) -> None:
        self._group = group

    def all_reduce(self, x: torch.Tensor, op: str) -> None:
        if op != "sum":
            raise ValueError(f"the group only sums: {op!r}")
        self._group.all_reduce(x)


class ParallelGroup:
    """One rank's group of the ranks: it owns the rendezvous Store (a private one on a
    single rank), the default, CPU and device process groups and the PyNCCL communicator,
    and runs the ranks' lockstep and collective operations over them."""

    def __init__(
        self,
        topology: Topology,
        *,
        device: torch.device,
        timeout_seconds: float,
        init_method: str | None,
        use_pynccl: bool,
    ) -> None:
        """``init_method`` is the multi-rank rendezvous."""

        if topology.size > 1 and init_method is None:
            raise ValueError("a multi-rank group requires its rendezvous init_method")
        if dist.is_initialized():
            raise RuntimeError("ParallelGroup cannot own an initialized default process group")
        self.topology = topology
        self._device = device
        self._cleanup = ExitStack()
        self._pynccl: _PyNCCL | None = None
        try:
            self._initialize(
                device=device,
                timeout=timedelta(seconds=timeout_seconds),
                init_method=init_method,
                use_pynccl=use_pynccl,
            )
        except BaseException:
            try:
                self.close()
            except BaseException:
                logger.exception("ParallelGroup cleanup also failed during construction")
            raise

    @property
    def rank(self) -> int:
        return self.topology.rank

    @property
    def size(self) -> int:
        return self.topology.size

    def is_primary(self) -> bool:
        return self.topology.rank == 0

    @property
    def backends(self) -> tuple[str, str]:
        return str(dist.get_backend(self._cpu_group)), str(dist.get_backend(self._device_group))

    def barrier(self) -> None:
        self._cpu_group.barrier().wait()

    def broadcast_cpu(self, tensor: torch.Tensor, *, root: int = 0) -> None:
        self._cpu_group.broadcast(tensor, root=root).wait()

    def broadcast_device(self, tensor: torch.Tensor, *, root: int = 0) -> None:
        """Overwrite every other rank's contiguous device ``tensor`` with ``root``'s, in stream
        order. The ranks gather every rank's bytes and keep ``root``'s, over the communicator
        of the model's own collectives, so a CUDA graph may capture it."""

        if self.size == 1:
            return
        data = tensor.view(-1).view(torch.uint8)
        gathered = self.all_gather(data).view(self.size, -1)
        if self.rank != root:  # the root holds its bytes already
            data.copy_(gathered[root])

    def differs_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """A device bool: whether any rank's contiguous device ``tensor`` differs from this
        rank's, in stream order; a CUDA graph may capture it, as ``broadcast_device``."""

        if self.size == 1:
            return torch.zeros((), dtype=torch.bool, device=tensor.device)
        gathered = self.all_gather(tensor.view(-1).view(torch.uint8)).view(self.size, -1)
        return (gathered != gathered[self.rank]).any()

    def min_max_int(self, value: int) -> tuple[int, int]:
        if self.size == 1:
            return value, value
        values = torch.tensor([value, -value], dtype=torch.int64)
        dist.all_reduce(values, op=dist.ReduceOp.MIN, group=self._cpu_group)
        return int(values[0]), -int(values[1])

    def require_same(self, value: int, what: str) -> None:
        """Raise on every rank unless every rank passed the same int64 ``value``."""

        low, high = self.min_max_int(value)
        if low != high:
            raise RuntimeError(f"tensor-parallel ranks diverged at {what}")

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.size == 1:
            return tensor
        if self._pynccl is not None and self._pynccl.reduces(tensor.dtype):
            return self._pynccl.all_reduce(tensor)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self._device_group)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.size == 1:
            return tensor
        if self._pynccl is not None:
            return self._pynccl.all_gather(tensor)
        shape = list(tensor.shape)
        shape[0] *= self.size
        result = tensor.new_empty(shape)
        dist.all_gather_into_tensor(result, tensor, group=self._device_group)
        return result

    def residual_reduction(
        self, *, hidden_size: int, eps: float, zero_centered: bool
    ) -> ResidualReduction:
        """cti-kernels' all-reduce + residual add + RMSNorm over the ranks, in the dtype the
        model is built in (torch's default), built collectively now. Where its fused exchange
        does not run, it sums through this group's own all-reduce, then adds and normalizes
        as the unfused norm does. Closed with the group."""

        from ctikernels.collective.inference import AllReduceAddRMSNorm

        # Even while a model is built on the meta device: the kernel places its device
        # buffers itself, and its calibration sums host tensors over the CPU group.
        with torch.device("cpu"):
            reduction = AllReduceAddRMSNorm(
                rank=self.rank,
                world_size=self.size,
                cpu_group=self._cpu_group,
                device=self._device,
                hidden=hidden_size,
                # Mailboxes for the kernel's most rows; longer calls exchange in slices.
                max_tokens=0,
                eps=eps,
                zero_centered=zero_centered,
                dtype=torch.get_default_dtype(),
                communicator=_Summing(self),
            )
        self._cleanup.callback(reduction.close)
        logger.info_rank0(
            "Residual reduction (hidden %d, zero_centered=%s): %s",
            hidden_size,
            zero_centered,
            reduction.provider,
        )
        return reduction

    def close(self) -> None:
        """Destroy what the group owns, in reverse order; later calls do nothing."""

        self._cleanup.close()

    def _initialize(
        self,
        *,
        device: torch.device,
        timeout: timedelta,
        init_method: str | None,
        use_pynccl: bool,
    ) -> None:
        topology = self.topology
        self._cleanup.enter_context(_RankRecords.installed(topology.rank))
        # A single rank owns a private Store, kept alive as long as its default group.
        store = self._store = None
        if topology.size == 1:
            store = self._store = dist.TCPStore(
                host_name="127.0.0.1",
                port=0,
                world_size=1,
                is_master=True,
                timeout=timeout,
                wait_for_workers=True,
            )
            self._cleanup.callback(setattr, self, "_store", None)
        try:
            dist.init_process_group(
                backend="gloo",
                rank=topology.rank,
                world_size=topology.size,
                timeout=timeout,
                store=store,
                init_method=None if store is not None else init_method,
            )
        finally:
            # A partial initialization is torn down as a complete one.
            if dist.is_initialized():
                self._cleanup.callback(dist.destroy_process_group)
        ranks = list(range(topology.size))
        self._cpu_group = self._new_group(ranks, "gloo", timeout)
        # A multi-rank NCCL communicator connects here, before memory sizing, so its buffers
        # are counted and no rank-local first use leaves peers waiting. Omitting device_id
        # keeps a size-one communicator lazy; size-one collectives never initialize it.
        eager = {"device_id": device} if topology.size > 1 else {}
        self._device_group = self._new_group(ranks, "nccl", timeout, **eager)
        if topology.size > 1 and use_pynccl:
            self._pynccl = _PyNCCL(topology, self._cpu_group)

    def _new_group(
        self, ranks: list[int], backend: str, timeout: timedelta, **options: object
    ) -> dist.ProcessGroup:
        group = dist.new_group(ranks=ranks, backend=backend, timeout=timeout, **options)
        if group is None:
            raise RuntimeError(f"{backend} group initialization returned no process group")
        self._cleanup.callback(dist.destroy_process_group, group)
        return group


__all__ = ["ParallelGroup"]
