"""The IPC addresses that link one online server's processes."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ServingEndpoints:
    """Where each process of one online server listens: every link's receiver binds it.

    Requests flow API server -> tokenizer -> engine -> detokenizer -> API server. Each of the
    ``num_tokenizers`` tokenizer processes binds its own address under ``tokenizer`` (its
    ``index``), so the API server sends a call's messages, its aborts too, to one tokenizer
    in order. An offline engine has none: its sampler sends to ``engine`` itself and reads
    ``detokenizer``.
    """

    engine: str
    frontend: str
    detokenizer: str
    tokenizer: str
    num_tokenizers: int = 0

    @classmethod
    def create(cls, num_tokenizers: int = 0, *, directory: str | None = None) -> ServingEndpoints:
        """Fresh addresses for ``num_tokenizers`` separate tokenizer processes; inside
        ``directory`` when its owner removes them, else under /tmp."""

        prefix = (
            f"ipc:///tmp/cti-{os.getpid()}-{uuid.uuid4().hex[:8]}-"
            if directory is None
            else f"ipc://{directory}/"
        )
        return cls(
            prefix + "engine",
            prefix + "frontend",
            prefix + "detokenizer",
            prefix + "tokenizer",
            num_tokenizers,
        )

    def link(
        self,
        target: Literal["engine", "frontend", "detokenizer", "tokenizer"],
        *,
        receiving: bool,
        index: int = 0,
    ) -> tuple[str, bool]:
        """``target``'s address (tokenizer process ``index``'s for ``tokenizer``) and whether
        this side of its link binds it: the receiver does."""

        if target == "tokenizer":
            if not 0 <= index < self.num_tokenizers:
                raise ValueError(f"no tokenizer {index} of {self.num_tokenizers}")
            return f"{self.tokenizer}-{index}", receiving
        address = {
            "engine": self.engine,
            "frontend": self.frontend,
            "detokenizer": self.detokenizer,
        }[target]
        return address, receiving


__all__ = ["ServingEndpoints"]
