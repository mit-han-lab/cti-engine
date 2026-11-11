"""Attention-backend test helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cti.attention import (
    AttentionRequest,
    AttentionRequirement,
    BaseAttnBackend,
    ResolvedFamily,
    resolve_attention_config,
)
from cti.attention.mha import AttentionBackend
from cti.kvcache import GroupRole, PageGroup
from cti.utils import DeviceCapability
from tests.support import reference_ops
from tests.support.paged_attention import kv_group


def set_validation(
    monkeypatch: pytest.MonkeyPatch, backend_class: type[BaseAttnBackend], *reasons: str
) -> None:
    """Make ``backend_class.validate_environment`` report ``reasons`` on any device."""

    monkeypatch.setattr(
        backend_class,
        "validate_environment",
        classmethod(lambda cls, capability: reasons),
    )


class PagedAttentionOperation:
    """``ctikernels``' ``FlashInferPagedAttention`` on the CPU: it keeps what it was built
    with and logs each ``plan`` and ``run``; ``run`` writes 3 into ``out`` and returns it."""

    def __init__(
        self, log: list[str], float_workspace: torch.Tensor, int_workspace: torch.Tensor, **options
    ) -> None:
        self.log = log
        self.float_workspace = float_workspace
        self.int_workspace = int_workspace
        self.options = options
        self.zeroed = bool((float_workspace == 0).all())  # FlashInfer needs zeroed scratch
        self.plans: list[dict] = []
        self.runs: list[dict] = []

    def plan(self, **kwargs) -> None:
        self.log.append("plan")
        self.plans.append(kwargs)

    def run(self, **kwargs) -> torch.Tensor:
        self.log.append("run")
        self.runs.append(kwargs)
        return kwargs["out"].fill_(3.0)


class CPUAttention:
    """Attention backends resolved and built through the attention family on the CPU, every
    device accepted.

    The ``ctikernels`` symbols they call are fakes: FlashInfer's paged-attention operation
    (``operations``, in construction order) and its prefill choice (``prefill``, asked as
    ``prefill_choices``), the KV store and the paged-KV metadata build (the references). The CUDA event, device
    query (``num_sms``) and pinned host memory are CPU stand-ins. ``log`` holds every plan,
    run, store and ``kv_metadata`` build, and each ``synchronize`` and ``record`` of a CUDA
    event, in order."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, num_sms: int = 4, prefill: str = "fa2"
    ) -> None:
        self.log: list[str] = []
        self.operations: list[PagedAttentionOperation] = []
        self.prefill_choices: list[tuple[torch.device, torch.dtype]] = []
        self.group: PageGroup | None = None
        log = self.log

        def operation(*args, **options) -> PagedAttentionOperation:
            self.operations.append(PagedAttentionOperation(log, *args, **options))
            return self.operations[-1]

        def choose_prefill(device: torch.device, dtype: torch.dtype) -> str:
            self.prefill_choices.append((device, dtype))
            return prefill

        def store(**kwargs) -> None:
            log.append("store")
            reference_ops.store_cache(**kwargs)

        def kv_metadata(*args, **kwargs) -> None:
            log.append("kv_metadata")
            reference_ops.paged_kv_metadata(*args, **kwargs)

        flashinfer = "cti.attention.flashinfer"
        monkeypatch.setattr(f"{flashinfer}.FlashInferPagedAttention", operation)
        monkeypatch.setattr(f"{flashinfer}.paged_kv_metadata", kv_metadata)
        monkeypatch.setattr(
            f"{flashinfer}.select_flashinfer_prefill_implementation", choose_prefill
        )
        # CPU tensors stand in for the device ones the metadata checks for.
        monkeypatch.setattr(f"{flashinfer}.FIMetadata.__post_init__", lambda metadata: None)
        monkeypatch.setattr("ctikernels.store_cache", store)
        event = SimpleNamespace(
            record=lambda: log.append("record"), synchronize=lambda: log.append("synchronize")
        )
        monkeypatch.setattr(torch.cuda, "Event", lambda: event)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda device: SimpleNamespace(multi_processor_count=num_sms),
        )
        for name in ("tensor", "arange", "ones"):
            allocate = getattr(torch, name)

            def unpinned(*args, _allocate=allocate, **kwargs):
                kwargs.pop("pin_memory", None)
                return _allocate(*args, **kwargs)

            monkeypatch.setattr(torch, name, unpinned)
        for backend in AttentionBackend:
            set_validation(monkeypatch, backend.backend_class)

    @staticmethod
    def resolve(
        name: str,
        *,
        heads: tuple[int, int, int] = (4, 2, 8),
        page_size: int = 1,
        tensor_cores: bool | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> ResolvedFamily:
        """One rank's plan of backend ``name`` for ``(qo heads, KV heads, head_dim)``."""

        return resolve_attention_config(
            {GroupRole.TARGET: (AttentionRequirement.mha(*heads),)},
            AttentionRequest(
                target=GroupRole.TARGET,
                backend=name,
                page_size=page_size,
                capability=DeviceCapability(10, 0),
                tensor_parallel_size=1,
                dtype=dtype,
                tensor_cores=tensor_cores,
            ),
        )

    def workspace_bytes(
        self,
        name: str = "fi",
        *,
        heads: tuple[int, int, int] = (4, 2, 8),
        page_size: int = 1,
        max_running_req: int,
    ) -> int:
        """The device scratch building backend ``name`` reserves."""

        return self.resolve(name, heads=heads, page_size=page_size).workspace_bytes(
            device=torch.device("cpu"), max_running_req=max_running_req
        )

    def build(
        self,
        name: str = "fi",
        *,
        heads: tuple[int, int, int] = (4, 2, 8),
        page_size: int = 1,
        num_pages: int = 16,
        max_running_req: int = 4,
        dtype: torch.dtype = torch.bfloat16,
        tensor_cores: bool | None = None,
    ) -> BaseAttnBackend:
        """Backend ``name`` over ``self.group``, two layers of ``num_pages`` pages of
        ``dtype``; the page table's two slots map the pages in order."""

        _, num_kv_heads, head_dim = heads
        self.group = kv_group(
            2, num_kv_heads, head_dim, num_pages, dtype=dtype, page_size=page_size
        )
        pages = torch.arange(num_pages, dtype=torch.int32).view(2, -1)
        resolved = self.resolve(
            name, heads=heads, page_size=page_size, tensor_cores=tensor_cores, dtype=dtype
        )
        return resolved.build(
            (self.group,),
            SimpleNamespace(page_size=page_size, pages=pages),  # type: ignore[arg-type]
            max_running_req=max_running_req,
            capability=DeviceCapability(10, 0),
        )


__all__ = ["CPUAttention", "PagedAttentionOperation", "set_validation"]
