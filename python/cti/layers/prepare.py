"""Inference warm-up of the models' layers, after their weights load and before any forward or
graph capture: every route their inference GEMMs can take compiles, one split-K workspace
serves them all, and every op that compiles per row count, shared attention prologue and
recurrent prefill compiles for the forwards the plan admits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

import torch

from cti.utils import init_logger
from ctikernels.linear import release_cublas_workspaces, release_scratch

from .linear_method import GemmCallSite, GemmRoutes

if TYPE_CHECKING:
    from .base import BaseOP

logger = init_logger(__name__)


def prepare_inference(
    ops: Iterable[BaseOP],
    *,
    max_rows: int,
    max_sequences: int,
    dtype: torch.dtype,
    device: torch.device,
    progress: Callable[[str], None] = lambda detail: None,
) -> int:
    """Compile everything the inference forwards of the op trees ``ops`` (the models) run, for
    every forward of at most ``max_rows`` rows and ``max_sequences`` sequences, so that no
    forward or graph capture compiles; return the bytes of the split-K workspace it assigns.

    Before anything compiles, every op of the trees settles what its inference runs
    (``BaseOP.settle_inference``), its methods and adapter being final.

    First every route their GEMMs can take, then one workspace, sized for the largest call
    any route makes, for the services of every model they belong to: cti-kernels' ``Linear``
    dispatches from the per-device table it ships (cuBLAS outside it), and the GEMMs run in
    turn on one stream, captured graphs included. Then what ops pick by row count
    (``BaseOP.warm_up_rows``, each key once), the attention layers' fused prologues (each
    shared one once) and what the recurrent layers' extend forwards run (CuTe providers;
    another recurrent prefill provider compiles per shape on first use); their row bound
    doubles up to ``max_rows``, each step compiling only the shapes new to it.

    ``progress(detail)`` hears each GEMM call site and row bucket, each op key as it is
    claimed, and each prologue and recurrent layer of each step as it begins.
    """

    ops = tuple(ops)
    nodes = [node for op in ops for _, node in op.named_ops()]
    for node in nodes:
        node.settle_inference()
    gemms: list[GemmCallSite] = list(
        {id(gemm): gemm for op in ops for gemm in op.inference_gemms()}.values()
    )
    size = _warm_up(gemms, max_rows, device=device, progress=lambda d: progress(f"GEMM {d}"))
    if size:
        workspace = torch.empty(-(-size // 4), dtype=torch.float32, device=device)
        for services in {id(gemm.services): gemm.services for gemm in gemms}.values():
            services.gemm_workspace = workspace
    _warm_up_ops(
        ops,
        nodes,
        max_tokens=max_rows,
        max_sequences=max_sequences,
        dtype=dtype,
        device=device,
        progress=progress,
    )
    return size


class _ReportedKeys(set):
    """The keys the ops' row warm-ups claimed (``BaseOP.warm_up_rows``), each new one
    reported to ``progress`` as its op claims it, before it compiles."""

    def __init__(self, progress: Callable[[str], None]) -> None:
        super().__init__()
        self._progress = progress

    def add(self, key: object) -> None:
        if key not in self:
            parts = key if isinstance(key, tuple) else (key,)
            self._progress(
                " ".join(part.__name__ if isinstance(part, type) else str(part) for part in parts)
            )
        super().add(key)


def _warm_up_ops(
    ops: Sequence[BaseOP],
    nodes: Sequence[BaseOP],
    *,
    max_tokens: int,
    max_sequences: int,
    dtype: torch.dtype,
    device: torch.device,
    progress: Callable[[str], None],
) -> None:
    from .attention import AttentionLayer
    from .gated_delta_net.layer import GatedDeltaNet

    warmed = _ReportedKeys(lambda key: progress(f"rows: {key}"))
    for op in ops:
        op.warm_up_rows(max_tokens, dtype=dtype, device=device, warmed=warmed)
    prologues = tuple(
        {
            id(prologue): prologue
            for layer in nodes
            if isinstance(layer, AttentionLayer) and (prologue := layer.prologue) is not None
        }.values()
    )
    recurrent = tuple(layer for layer in nodes if isinstance(layer, GatedDeltaNet))
    bounds = [1 << power for power in range((max_tokens - 1).bit_length())] + [max_tokens]
    for step, tokens in enumerate(bounds, 1):
        where = f"{step}/{len(bounds)}, {tokens} tokens"
        for index, prologue in enumerate(prologues, 1):
            progress(f"{where}: prologue {index}/{len(prologues)}")
            prologue.warm_up(tokens)
        for index, layer in enumerate(recurrent, 1):
            progress(f"{where}: recurrent layer {index}/{len(recurrent)}")
            layer.warm_up(tokens, max_sequences=min(tokens, max_sequences))


def _warm_up(
    gemms: Sequence[GemmCallSite],
    max_rows: int,
    *,
    device: torch.device,
    progress: Callable[[str], None],
) -> int:
    """Compile the routes of ``gemms`` and release the warm-up's scratch; return the split-K
    workspace bytes the largest of their calls needs."""

    routes: GemmRoutes = Counter()
    with torch.inference_mode():
        for step, gemm in enumerate(gemms, 1):
            site = f"{step}/{len(gemms)}"
            progress(site)
            routes.update(
                gemm.warm_up(
                    max_rows,
                    device,
                    renew=lambda bucket, site=site: progress(f"{site}: {bucket}"),
                )
            )
    by_rows: dict[tuple[int, int], list[str]] = {}
    for (rows, route), count in sorted(routes.items()):
        by_rows.setdefault(rows, []).append(f"{route} {count}")
    logger.info_rank0(
        "Inference GEMM routes by rows: %s",
        "; ".join(
            f"{low if low == high else f'{low}-{high}'}: {', '.join(names)}"
            for (low, high), names in by_rows.items()
        ),
    )
    if device.type == "cuda":
        # The warm-up's eager split-K scratch and the cuBLAS workspaces of the streams that
        # ran it are dropped, as a CUDA graph capture does, so no long-lived buffer allocated
        # next splits (and pins) one of their segments before the runner measures what the
        # KV cache may take.
        torch.cuda.synchronize()
        release_scratch(device)
        release_cublas_workspaces()
        torch.cuda.empty_cache()
    return max((gemm.workspace_bytes(device) for gemm in gemms), default=0)


__all__ = ["prepare_inference"]
