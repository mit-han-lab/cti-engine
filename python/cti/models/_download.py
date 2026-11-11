# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/utils/hf.py`); see THIRD_PARTY_NOTICES.

"""Fetch a checkpoint from its hub in two steps at one revision: the files that describe it
when it is read, its weights once it is validated."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

import httpx
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import filter_repo_objects
from tqdm.asyncio import tqdm

if TYPE_CHECKING:
    from cti.config import ModelHub

# What reads a checkpoint (configs, tokenizer and processor files, chat templates), by
# pattern so a new model's files need no new entry; .bin/.pt duplicates and docs stay behind.
_METADATA_PATTERNS = ("*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken")
_WEIGHT_PATTERNS = ("*.safetensors",)
# A file still downloading reports its bytes at most this often, well inside the parent's
# startup stall window, so one large shard on a slow link keeps renewing it.
BYTE_PROGRESS_SECONDS = 30.0


@dataclass(frozen=True)
class CheckpointSource:
    """Where a checkpoint's files are: a local ``folder``, which for a repository id is its
    snapshot on ``hub`` at ``revision`` (None where the hub pins none), the weights joining
    it once validated."""

    folder: str
    hub: ModelHub | None = None
    repo_id: str | None = None
    revision: str | None = None

    def identity(self) -> dict[str, str | None]:
        """What names the checkpoint across machines: its repository at its revision, else
        its folder."""

        if self.repo_id is None:
            return {"folder": self.folder}
        return {"hub": self.hub, "repo_id": self.repo_id, "revision": self.revision}


def fetch_metadata(path: str, hub: ModelHub) -> CheckpointSource:
    """The source of the checkpoint at ``path``, a local directory or a repository id on
    ``hub`` whose descriptive files (not its weights) download at one revision."""

    if Path(path).is_dir():
        return CheckpointSource(path)
    try:
        if hub == "modelscope":
            return CheckpointSource(_modelscope(path, _METADATA_PATTERNS), hub, path)
        folder = snapshot_download(
            path, allow_patterns=list(_METADATA_PATTERNS), tqdm_class=_DisabledTqdm
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{path}' is neither a local directory nor a valid model ID: {e}"
        ) from e
    # The hub's cache keeps a snapshot under its commit (``snapshots/<sha>``).
    return CheckpointSource(folder, hub, path, Path(folder).name)


def fetch_weights(source: CheckpointSource, progress: Callable[[str], None]) -> None:
    """Download ``source``'s weights into its folder, at its revision; ``progress`` hears each
    file's bytes while it downloads and the file once it completes."""

    if source.repo_id is None:
        return
    if source.hub == "modelscope":
        _modelscope(source.repo_id, _WEIGHT_PATTERNS, progress)
        return
    _download_each_file(source.repo_id, source.revision, progress)


def _download_each_file(
    repo_id: str,
    revision: str,
    progress: Callable[[str], None],
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Download the revision's weight files into the cache on 8 workers, reporting each as it
    completes and its bytes while it downloads."""

    try:
        info = HfApi().model_info(repo_id, revision=revision)
    except (OSError, httpx.HTTPError):
        # Offline or refused: the cached snapshot serves, or the download names why.
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=list(_WEIGHT_PATTERNS),
            tqdm_class=_DisabledTqdm,
        )
        return
    names = [sibling.rfilename for sibling in info.siblings or ()]
    files = list(filter_repo_objects(names, allow_patterns=list(_WEIGHT_PATTERNS)))
    # Feature-detected: a hub version without the parameter reports only completed files.
    counts_bytes = "tqdm_class" in inspect.signature(hf_hub_download).parameters
    lock = Lock()  # the workers' reports reach one channel

    def report(message: str) -> None:
        with lock:
            progress(message)

    def download(name: str) -> None:
        if counts_bytes:
            bars = _FileBytes(f"downloading weights: {name}", report, clock).bar_class()
            hf_hub_download(repo_id, name, revision=revision, tqdm_class=bars)
        else:
            hf_hub_download(repo_id, name, revision=revision)

    with ThreadPoolExecutor(max_workers=8) as pool:
        downloads = {pool.submit(download, name): name for name in files}
        for index, done in enumerate(as_completed(downloads)):
            done.result()
            report(f"downloading weights: file {index + 1}/{len(files)} ({downloads[done]})")


def _modelscope(
    repo_id: str, patterns: tuple[str, ...], progress: Callable[[str], None] | None = None
) -> str:
    """``repo_id``'s files matching ``patterns`` from ModelScope (the ``modelscope`` extra);
    ``progress`` hears every ``BYTE_PROGRESS_SECONDS`` while they download (the library's own
    callbacks are not called on every download route)."""

    from modelscope import snapshot_download as modelscope_snapshot_download

    if progress is None:
        return modelscope_snapshot_download(repo_id, allow_patterns=list(patterns))
    done = Event()

    def beat() -> None:
        start = time.monotonic()
        while not done.wait(BYTE_PROGRESS_SECONDS):
            progress(f"downloading weights: {time.monotonic() - start:.0f} s")

    ticker = Thread(target=beat, daemon=True)
    ticker.start()
    try:
        return modelscope_snapshot_download(repo_id, allow_patterns=list(patterns))
    finally:
        done.set()
        ticker.join()


class _DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


class _FileBytes:
    """One file's byte reports, at most one per ``BYTE_PROGRESS_SECONDS`` across the bars
    the hub opens for it."""

    def __init__(self, label: str, report: Callable[[str], None], clock: Callable[[], float]):
        self._label = label
        self._report = report
        self._clock = clock
        self._last = clock()
        self._lock = Lock()

    def bar_class(self) -> type[tqdm]:
        """A silent bar class that counts its bytes and reports them here."""

        file = self

        class _Bar(_DisabledTqdm):
            def update(self, n: float | None = 1) -> bool | None:
                self.n += n or 0  # a disabled bar counts nothing itself
                file.received(self.n, self.total)
                return None

        return _Bar

    def received(self, done: float, total: float | None) -> None:
        now = self._clock()
        with self._lock:
            if now - self._last < BYTE_PROGRESS_SECONDS:
                return
            self._last = now
        size = f"{done / 1e9:.1f}" if total is None else f"{done / 1e9:.1f}/{total / 1e9:.1f}"
        self._report(f"{self._label} {size} GB")
